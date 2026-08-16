from __future__ import annotations

import asyncio
from typing import Any

from harnessx.benchmarks.tau3.adapter import Tau3Task, verify_tau3
from harnessx.benchmarks.tau3.retail import RetailDomain, ScriptedUserSimulator
from harnessx.benchmarks.tau3.runner import DialogueRunner
from harnessx.events import Message
from harnessx.providers.base import Provider, ProviderResponse


class ScriptedProvider(Provider):
    def __init__(self, steps: list[Any]) -> None:
        super().__init__("fake")
        self.steps = iter(steps)

    async def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        step = next(self.steps, "done")
        if isinstance(step, dict):
            return ProviderResponse(content="", tool_calls=step["tool_calls"])
        return ProviderResponse(content=step, stop_reason="end_turn")


def _make_task() -> Tau3Task:
    return Tau3Task(
        task_id="retail-001",
        domain="retail",
        opening="Hi, I want to cancel my order #123.",
        script=[{"reply": "Thanks!", "stop": True}],
        expected={
            "orders": [{"key_field": "order_id", "key": "123", "fields": {"status": "cancelled"}}]
        },
    )


def test_tau3_dialogue_cancels_order() -> None:
    async def main() -> None:
        domain = RetailDomain()
        db = domain.build_db()
        task = _make_task()
        provider = ScriptedProvider(
            [
                {"tool_calls": [{"id": "1", "name": "cancel_order", "arguments": {"order_id": "123"}}]},
                "Your order #123 has been cancelled.",
            ]
        )
        runner = DialogueRunner(
            provider=provider,
            tools=domain.tools(),
            user_simulator=ScriptedUserSimulator(task),
            db=db,
            opening=task.opening,
        )
        result = await runner.run()
        assert result.stopped
        assert await verify_tau3(task, result)

    asyncio.run(main())


def test_tau3_dialogue_wrong_action_fails() -> None:
    async def main() -> None:
        domain = RetailDomain()
        db = domain.build_db()
        task = _make_task()
        provider = ScriptedProvider(
            [
                {"tool_calls": [{"id": "1", "name": "cancel_order", "arguments": {"order_id": "124"}}]},
                "Cancelled order #124.",
            ]
        )
        runner = DialogueRunner(
            provider=provider,
            tools=domain.tools(),
            user_simulator=ScriptedUserSimulator(task),
            db=db,
            opening=task.opening,
        )
        result = await runner.run()
        assert not await verify_tau3(task, result)

    asyncio.run(main())
