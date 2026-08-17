# Multi-Objective RL — Cải tiến tiếp theo (TODO list)

Trạng thái: chưa implement. Mỗi mục độc lập, nên làm theo thứ tự. Không sửa
`src/harnessx/` (theo nguyên tắc trong `docs/plans/multi-objective-rl-tau3.md`).

## 1. Baseline comparison (single vs multi)
- `experiments/run_tau3.py` thêm chế độ `--baseline`: chạy loop với scorer chỉ giữ
  `correctness` (weights `{correctness: 1.0}`) để so sánh pass@N và mean-objectives
  với run multi-objective.
- In bảng delta như `scripts/verify_gaia.py`: `baseline=... final=... delta=...`.

## 2. Soft pass@N
- `evaluate()` trong `harnessx.evolve.loop` vẫn gán `traj.success` từ binary verifier;
  thêm metric `soft@N` = mean của `correctness` fraction trên N rollouts, journal cùng
  curves để thấy partial credit thay vì chỉ pass/fail.

## 3. Min-max normalize objectives trước scalarize
- `experiments/multiobj/objectives.py`: thêm `normalize_rewards(records, specs)`
  min-max qua batch/buffer, dùng khi scalarize RAW rewards (cho `RLRecord.reward`) —
  tránh objective có scale lớn (price, cost, token) áp đảo weighted sum.

## 4. Stratified buffer sampling
- `experiments/loop.py` `_train`: sample theo `group_id` (task) thay vì `random.sample`
  toàn bộ — giữ cấu trúc group cho group-relative advantage của GRPO.

## 5. Pareto shaping / Tchebycheff scalarization
- `experiments/multiobj/trainer.py`: thưởng nhẹ cho record non-dominated (front 0)
  hoặc thêm option `scalarization="tchebycheff"` trong `MultiObjectiveGRPOTrainer`.
- Giữ weighted-sum làm default; Pareto dùng để shaping + report.

## 6. Integration test cho MultiObjectiveEvolutionLoop
- `tests/experiments/test_loop.py`: chạy loop 1 round với fake task/provider, assert
  buffer records có `extra["rewards"]` + `process_rewards`, trainer detail có
  `pareto_fronts` / `non_dominated`.

## 7. (Xét sau) Observation compression
- Nếu transcript tau3 quá lớn: compress qua `harnessx.evolve.digester.Digester` trước
  khi ghi vào `extra["observations"]` (giới hạn token/size) — giữ tín hiệu, giảm size.