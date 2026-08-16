import asyncio

from harnessx import HarnessConfig, ModelConfig
from harnessx.core.hooks import Hook
from tests.unit.harnessx.fakes import EchoProvider, Task


def test_basic_run() -> None:
    async def main() -> None:
        model = ModelConfig(main=EchoProvider("fake"))
        harness = model.agentic(HarnessConfig(max_steps=3))
        traj = await harness.run(Task(id="t1", description="2+2"), task_id="t1")
        assert traj.final_output == "answer: 4"
        assert len(traj.steps) == 1

    asyncio.run(main())


def test_processor_pipeline() -> None:
    from harnessx.core.processor import Order, Processor

    seen: list[str] = []

    class Recorder(Processor):
        _singleton_group = "recorder"
        _order = Order.PRE

        async def process(self, event):
            seen.append(type(event).__name__)
            yield event

    async def main() -> None:
        model = ModelConfig(main=EchoProvider("fake"))
        config = HarnessConfig(max_steps=2).add(Hook.TASK_START, Recorder())
        traj = await model.agentic(config).run(
            Task(id="t2", description="x"), task_id="t2"
        )
        assert "TaskStartEvent" in seen
        assert traj.final_output == "answer: 4"

    asyncio.run(main())
