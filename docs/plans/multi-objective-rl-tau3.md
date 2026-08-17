# Plan: Multi-Objective RL + Observation Improvements (tau3 retail)

## 1. Nguyên tắc an toàn: KHÔNG sửa `src/harnessx/`

Thí nghiệm được thiết kế để **không làm hỏng code đang có**:

1. **Git branch riêng**: `git checkout -b experiment/multi-objective-rl`.
   `main` không bao giờ bị đụng; muốn quay lại chỉ cần `git checkout main`.
2. **Toàn bộ code mới nằm ngoài `src/harnessx/`** trong package `experiments/`,
   chỉ *import* harnessx qua các seam sẵn có — không sửa file nào trong `src/`.
3. **Nếu thiếu seam**: chỉ thêm **additive** (field/func mới có default) trên branch,
   không đổi hành vi cũ, giữ nguyên test cũ pass. `main` luôn sạch.

### Các seam sẵn có (đã verify trong code)

| Seam | Vị trí | Cách dùng |
|---|---|---|
| `ModelTrainer` protocol | rl/coevolution.py:40 | Trainer experiment implement protocol, nhét vào `EvolutionLoop(trainer=...)` |
| `buffer` param | evolve/loop.py:101 | `MixedPolicyBuffer` có sẵn; reward vector + observations chứa trong `RLRecord.extra` (field có sẵn) — **không sửa RLRecord** |
| `_bridge` override | evolve/loop.py:113 | Subclass `EvolutionLoop`, gắn `MultiObjectiveBridge` trong `__post_init__` |
| `register(kind)` | processors/registry.py:18 | Processor mới đăng ký từ `experiments/`, AEGIS loop dùng được |
| `DialogueHarness` | benchmarks/tau3/adapter.py | Giữ nguyên; observation sinh từ `DialogueResult.transcript` có sẵn |

### Verify không hỏng
- `uv run pytest` — bộ test hiện tại phải xanh y hệt trước & sau.
- `uv run ruff check`.
- Chạy thử với `EchoProvider`/fake (tests/unit/harnessx/fakes.py) trước khi dùng API thật.

## 2. Cấu trúc package `experiments/`

```
experiments/multiobj/objectives.py    # ObjectiveSpec, scalarize, pareto_dominance,
                                      # non_dominated_sort, multi_group_relative_advantage
experiments/multiobj/bridge.py        # MultiObjectiveBridge (subclass TrajectoryBridge)
                                      #  → ghi rewards/observations/metrics vào RLRecord.extra
experiments/multiobj/trainer.py       # MultiObjectiveGRPOTrainer (implement ModelTrainer)
                                      #  → per-objective advantage + weighted-sum + Pareto front
experiments/multiobj/scorers/tau3.py  # objective_scorer(task, traj) -> dict[str, float]
experiments/multiobj/processors.py    # observation_capture / tool_summarizer / reflection
                                      #  → register() + subclass Evolver để thêm kinds
experiments/run_tau3.py               # entrypoint wiring EvolutionLoop
tests/experiments/                    # unit tests cho experiments
```

## 3. Objective space (tau3 retail)

Mỗi task tau3 → vector reward 3 chiều từ `DialogueResult` (`db_state` + `transcript` + `turns`):

| Objective   | Định nghĩa                                                          | Range  | Direction |
|-------------|---------------------------------------------------------------------|--------|-----------|
| `correctness` | Fraction các `expected` field-check thoả mãn (soft; `--binary` giữ binary) | [0,1] | max       |
| `efficiency`  | `1 - turns / max_turns`                                              | [0,1]  | max       |
| `tool_safety` | `1 - min(1, tool_errors / max(1, tool_calls))`                       | [0,1]  | max       |

## 4. Multi-objective math — `experiments/multiobj/objectives.py`

- `@dataclass ObjectiveSpec(name, weight=1.0, minimize=False)`.
- `scalarize(rewards, specs) -> float` — weighted sum, objective `minimize` đổi dấu,
  chuẩn hoá theo tổng weight.
- `pareto_dominance(a, b, specs) -> bool` — a dominates b nếu ≥ mọi objective và > ít nhất 1.
- `non_dominated_sort(records, specs) -> list[list[int]]` — các Pareto fronts (NSGA-II).
- `multi_group_relative_advantage(rewards: list[dict], group_ids, specs)` →
  per-objective advantage (`(r-mu)/(sigma+eps)` per group) + `scalarized`.

## 5. Bridge — `experiments/multiobj/bridge.py`

`MultiObjectiveBridge(TrajectoryBridge)` override `to_record(...)`:
1. Build per-step observations từ `Trajectory.steps`; nếu rỗng → parse `final_output.transcript`.
2. Tính `metrics` (steps, tool_calls, tool_errors, turns).
3. `rewards = scorer(task, traj)`; scalarize → `reward` (weighted-sum / correctness khi không có weights).
4. Ghi `rewards`/`observations`/`metrics` vào `RLRecord.extra` — **không đổi RLRecord**.
5. Không có scorer → hành vi cũ y nguyên (delegate lên `TrajectoryBridge.to_record`).

## 6. Trainer — `experiments/multiobj/trainer.py`

`MultiObjectiveGRPOTrainer` implement `ModelTrainer.update(records, config)`:
- Đọc `extra["rewards"]`; nếu không có → fallback scalar path (như `GRPOTrainer`).
- Tính per-objective advantage + scalarized advantage, `clipped_objective`.
- `TrainResult.detail` thêm: `per_objective_advantages`, `pareto_fronts`, `non_dominated`,
  `mean_objectives`, `objective_weights`.
- `CollectOnly`-style logging mean per-objective khi `api_only`.

## 7. Cải thiện observations

### 7.1 Trace level (tau3)
`DialogueRunner` (giữ nguyên trong src) đã trả `transcript` per-turn đầy đủ. Bridge chuyển
transcript thành per-step observations; không cần sửa runner. Option: annotation process-reward
(tool error → -1) tính trong bridge từ transcript.

### 7.2 Harness level (benchmark chạy qua `RunLoop`)
Processor mới trong `experiments/multiobj/processors.py`, đăng ký qua `register()` +
subclass `Evolver` thêm vào `KIND_SPECS` để AEGIS loop tự đề xuất:
- `observation_capture` (STEP_END): ghi compact per-step observation vào trajectory metadata.
- `tool_summarizer` (AFTER_TOOL): cô đọng tool result lớn.
- `reflection` (STEP_END): prompt model tóm tắt progress (optional).

## 8. Experiment script — `experiments/run_tau3.py`

CLI: `--data`, `--rounds`, `--weights correctness=1.0,efficiency=0.3,tool_safety=0.5`,
`--n-rollouts`, `--concurrency`, `--api-only`, `--binary`.

Wiring: `EvolutionLoop` subclass với `_bridge = MultiObjectiveBridge`, `trainer` theo
`--api-only`, `buffer = MixedPolicyBuffer`, `Journal("tau3_mo_rl")`. Output: curves +
audit log per-objective + Pareto front stats.

## 9. Tests (`tests/experiments/`)

- `test_objectives.py`: scalarize, pareto_dominance, non_dominated_sort, multi advantage.
- `test_multiobj_bridge.py`: extra chứa rewards/observations từ steps và từ transcript.
- `test_multiobj_trainer.py`: trainer weighted + detail; fallback scalar path.
- `test_tau3_scorer.py`: 3 objectives trên retail tasks.
- `test_processors.py`: kinds mới đăng ký đúng hook, Evolver subclass sinh candidate hợp lệ.
- Bộ test cũ (`test_coevolution.py`, `test_benchmarks_rl.py`, ...) phải giữ nguyên xanh.

## 10. Thứ tự implement

1. Branch `experiment/multi-objective-rl` + `experiments/` skeleton.
2. `objectives.py` → `bridge.py` → `trainer.py`.
3. `scorers/tau3.py` + tests.
4. `processors.py` + Evolver subclass + tests.
5. `run_tau3.py`.
6. Chạy `uv run pytest` + `uv run ruff check` toàn repo (main xanh y hệt).