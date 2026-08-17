import asyncio

import pytest

from experiments.multiobj.trainer import (
    MultiObjectiveCollectOnlyTrainer,
    MultiObjectiveGRPOTrainer,
)
from harnessx.rl import RLRecord, TrainResult
from harnessx.rl.grpo import GRPOConfig


def _record(task_id: str, rewards: dict[str, float]) -> RLRecord:
    return RLRecord(
        task_id=task_id,
        completion="x",
        reward=0.0,
        group_id=task_id,
        extra={"rewards": rewards},
    )


def test_trainer_reports_per_objective_detail():
    records = [
        _record("g1", {"correctness": 1.0, "efficiency": 0.8}),
        _record("g1", {"correctness": 0.0, "efficiency": 0.6}),
        _record("g2", {"correctness": 1.0, "efficiency": 0.9}),
        _record("g2", {"correctness": 0.0, "efficiency": 0.5}),
    ]
    trainer = MultiObjectiveGRPOTrainer(batch_size=4, objective_weights={"correctness": 1.0, "efficiency": 0.5})

    async def main() -> TrainResult:
        return await trainer.update(records, GRPOConfig())

    result = asyncio.run(main())
    assert result.records == 4
    assert result.objective is not None
    assert result.detail["objectives"] == ["correctness", "efficiency"]
    assert result.detail["objective_weights"] == {"correctness": 1.0, "efficiency": 0.5}
    assert "per_objective_advantages" in result.detail
    assert result.detail["pareto_fronts"] >= 1
    assert result.detail["non_dominated"] >= 1
    assert result.detail["mean_objectives"]["correctness"] == pytest.approx(0.5)


def test_trainer_falls_back_to_scalar():
    records = [RLRecord(task_id="t", completion="x", reward=1.0), RLRecord(task_id="t", completion="y", reward=0.0)]
    trainer = MultiObjectiveGRPOTrainer(batch_size=4)

    async def main() -> TrainResult:
        return await trainer.update(records, GRPOConfig())

    result = asyncio.run(main())
    assert result.records == 2
    assert result.detail == {"batch_size": 4}


def test_trainer_uniform_weights_when_unspecified():
    records = [_record("g1", {"correctness": 1.0, "efficiency": 0.8}), _record("g1", {"correctness": 0.0, "efficiency": 0.6})]
    trainer = MultiObjectiveGRPOTrainer(batch_size=2)

    async def main() -> TrainResult:
        return await trainer.update(records, GRPOConfig())

    result = asyncio.run(main())
    assert result.detail["objective_weights"] == {"correctness": 1.0, "efficiency": 1.0}


def test_collect_only_logs_objectives():
    records = [_record("g1", {"correctness": 1.0}), _record("g1", {"correctness": 0.0})]
    trainer = MultiObjectiveCollectOnlyTrainer(log=False)

    async def main() -> TrainResult:
        return await trainer.update(records, GRPOConfig())

    result = asyncio.run(main())
    assert result.records == 2
    assert result.updated is False