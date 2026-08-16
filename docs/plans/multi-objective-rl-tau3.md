# Plan: Multi-Objective RL + Observation Improvements (tau3 retail)

## 1. Motivation

HarnessX RL hiện tại là single-objective:

- **Observation = execution trace**, nhưng `TrajectoryBridge.to_record` (bridge.py:47) chỉ giữ
  `completion = final_output` (string) + `reward` scalar. Toàn bộ trace (messages, tool calls,
  tool results per-step) **không bao giờ tới tay trainer**.
- **Verifier cố định per task**: `evaluate()` (loop.py:76) gọi `verifier(task, final_output) -> bool`
  rồi gán `traj.reward = 1.0 if ok else 0.0`. Binary, chỉ ở cuối episode (outcome reward),
  không có tín hiệu trung gian / partial credit.
- **GRPO chỉ nhận scalar reward**: `RLRecord.reward: float`, `group_relative_advantage` nhận
  `list[float]`, `clipped_objective` là scalar.

Mục tiêu: nâng cấp RL thành **multi-objective** (vector reward + Pareto/weighted-sum scalarization)
và **cải thiện observations** (giữ đủ trace, per-step structured observation, process reward,
observation processors evolvable qua AEGIS loop).

## 2. Objective space (tau3 retail)

Mỗi task tau3 → vector reward 3 chiều, tính từ `DialogueResult`
(`db_state` + `transcript` + `turns`):

| Objective   | Định nghĩa                                                                 | Range   | Direction |
|-------------|----------------------------------------------------------------------------|---------|-----------|
| `correctness` | Fraction của các `expected` field-check thoả mãn trong `verify_retail`      | [0, 1]  | max       |
| `efficiency`  | `1 - turns / max_turns`                                                     | [0, 1]  | max       |
| `tool_safety` | `1 - min(1, tool_errors / max(1, tool_calls))`                              | [0, 1]  | max       |

- `correctness` là **soft** (partial credit) thay vì binary; option `--binary` giữ binary.
- `tool_errors` = số message tool có `{"error": ...}` trong transcript.
- Đặt tại `benchmarks/tau3/scorer.py` dưới dạng `objective_scorer(task, traj) -> dict[str, float]`.

## 3. Thay đổi data model — `rl/bridge.py`

`RLRecord` thêm các field (giữ `reward: float` scalar để backward-compatible):

- `rewards: dict[str, float]` — objective vector.
- `observations: list[dict[str, Any]]` — per-step observation slices
  (assistant message, tool calls, tool results, process reward).
- `metrics: dict[str, float]` — steps, tool_calls, tool_errors, turns.

`TrajectoryBridge.to_record(...)` nhận thêm:
- `objective_scorer: Callable[[Any, Trajectory], dict[str, float]] | None`
- `objective_weights: dict[str, float] | None`

Logic:
1. Build `observations` từ `Trajectory.steps`; nếu `steps` rỗng, fallback parse
   `final_output.transcript` (tau3 `DialogueResult`).
2. Tính `metrics` từ trace.
3. Nếu có scorer → `rewards = scorer(task, traj)`; scalarize bằng weighted sum
   (hoặc đúng primary objective `correctness`) vào `reward`.
4. Không có scorer/weights → hành vi cũ y nguyên.

## 4. Multi-objective math — `rl/objectives.py` (mới) + `rl/grpo.py`

`rl/objectives.py`:
- `@dataclass ObjectiveSpec(name, weight, minimize=False)`.
- `scalarize(rewards: dict[str, float], specs) -> float` — weighted sum,
  objective `minimize` đổi dấu; chuẩn hoá theo tổng weight.
- `pareto_dominance(a, b, specs) -> bool` — a dominates b nếu tốt hơn ≥1 objective
  và không kém mọi objective khác.
- `non_dominated_sort(records: list[RLRecord], specs) -> list[list[int]]` — các Pareto fronts
  (front 0 = non-dominated).

`rl/grpo.py`:
- `GRPOConfig` thêm `objective_weights: dict[str, float]` + `scalarization: str = "weighted_sum"`.
- `multi_group_relative_advantage(rewards: list[dict[str, float]], group_ids, specs)`
  → `dict[int, dict[str, float]]` per-objective advantage (cùng công thức
  `(r - mu) / (sigma + eps)` per group), cộng thêm `scalarized` cho mỗi sample.

## 5. Trainer — `rl/coevolution.py`

`GRPOTrainer.update()`:
- Nếu records có `rewards` dict (multi-objective):
  - Tính per-objective advantages + scalarized advantage (weighted sum).
  - `objective = clipped_objective(scalarized_adv, ratios=[1.0], config)`.
  - `TrainResult.detail` thêm: `per_objective_advantages`, `pareto_fronts` (số front,
    non-dominated count), `objective_weights`.
- `CollectOnlyTrainer`: log mean per-objective khi có `rewards`.

## 6. Cải thiện observations

### 6.1 Trace level (tau3)

Hiện `DialogueHarness.run()` (adapter.py:43) trả `Trajectory(final_output=DialogueResult)` với
`steps = []` → observation chỉ là 1 blob JSON.

- `DialogueRunner` ghi vào `Trajectory.steps` các `StepRecord` (messages, tool_calls,
  tool_results) — observation có cấu trúc per-turn.
- Annotation process-reward vào step: tool error → `reward=-1`, các bước khác `0`.

### 6.2 Harness level (benchmarks chạy qua `RunLoop` — gaia/webshop/swebench)

Processor mới đăng ký vào `processors/registry.py` + `evolver.KIND_SPECS`/`_GROUP_BY_KIND`
để AEGIS loop tự đề xuất:

- `observation_capture` (STEP_END): ghi compact per-step observation vào trajectory metadata.
- `tool_summarizer` (AFTER_TOOL): cô đọng tool result lớn → trace nhỏ, tín hiệu đậm đặc.
- `reflection` (STEP_END): prompt model tóm tắt progress → observation giàu hơn (optional).

`TrajectoryBridge` có thể nén observations qua `Digester` khi quá dài (giới hạn size).

## 7. Experiment script — `scripts/run_tau3_experiment.py`

CLI:
- `--data` (default `examples/data/tau3_sample.jsonl`)
- `--rounds` (default 4)
- `--weights correctness=1.0,efficiency=0.3,tool_safety=0.5`
- `--n-rollouts`, `--concurrency`
- `--api-only` → dùng `CollectOnlyTrainer`, ngược lại `GRPOTrainer`
- `--binary` → correctness binary thay vì soft

Wiring: `DialogueHarness` + tau3 `objective_scorer` + `MixedPolicyBuffer` + trainer +
`Journal("tau3_mo_rl")`. Output: curves + audit log per-objective + Pareto front stats.

## 8. Tests

- `test_objectives.py`: scalarize, pareto_dominance, non_dominated_sort, multi advantage.
- `test_multiobjective_rl.py`: RLRecord rewards/observations, bridge packing từ steps
  và từ transcript, GRPOTrainer weighted + detail.
- `test_tau3_scorer.py`: 3 objectives trên retail tasks (cancel-order, change-address).
- `test_tau3_trajectory.py`: DialogueRunner fill steps + process reward.
- Giữ nguyên + chạy lại `test_coevolution.py`, `test_benchmarks_rl.py`.

## 9. Thứ tự implement

1. `rl/objectives.py` + mở rộng `grpo.py`/`bridge.py`/`coevolution.py` (backward-compatible).
2. tau3 scorer + DialogueRunner/Trajectory steps + process reward.
3. Processors mới + đăng ký Evolver kinds.
4. `scripts/run_tau3_experiment.py`.
5. Tests + `ruff`/`pytest` (`uv run`).