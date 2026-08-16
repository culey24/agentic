import asyncio

from harnessx import HarnessConfig, ModelConfig
from harnessx.evolve import EvolutionLoop
from harnessx.rl import (
    CollectOnlyTrainer,
    GRPOTrainer,
    MixedPolicyBuffer,
    group_relative_advantage,
)
from harnessx.tracing.journal import Journal
from tests.unit.harnessx.fakes import EchoProvider, Task


def test_coevolution_collects_records() -> None:
    async def verify(task: Task, final_output: str) -> bool:
        return final_output == "answer: 4"

    async def main() -> None:
        model = ModelConfig(main=EchoProvider("fake"))
        harness = model.agentic(HarnessConfig(max_steps=2))
        tasks = [Task(id=f"t{i}", description="x") for i in range(3)]
        buffer = MixedPolicyBuffer(capacity=1000)
        journal = Journal("coevolve_test")
        loop = EvolutionLoop(
            meta_provider=None,
            harness=harness,
            tasks=tasks,
            verifier=verify,
            journal=journal,
            n_rollouts=1,
            max_rounds=2,
            buffer=buffer,
            trainer=CollectOnlyTrainer(),
        )
        result = await loop.run()
        assert result["buffer_size"] == len(tasks) * 2
        assert result["train_steps"] > 0

    asyncio.run(main())


def test_grpo_trainer_reports_objective() -> None:
    async def main() -> None:
        from harnessx.core.trajectory import Trajectory
        from harnessx.rl import TrajectoryBridge

        bridge = TrajectoryBridge()
        buffer = MixedPolicyBuffer(capacity=100)
        for tid, reward in [("a", 1.0), ("a", 0.0), ("b", 1.0), ("b", 0.0)]:
            traj = Trajectory(task_id=tid, final_output="x", reward=reward)
            buffer.insert(bridge.to_record(traj))
        trainer = GRPOTrainer(batch_size=4)
        result = await trainer.update(list(buffer))
        assert result.records == 4
        assert result.objective is not None

    asyncio.run(main())


def test_group_relative_advantage_zero_std() -> None:
    adv = group_relative_advantage([1.0, 1.0], ["a", "a"])
    assert adv[0] == 0.0 and adv[1] == 0.0
