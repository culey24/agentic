"""Dialogue runner for τ³-Bench.

τ³-Bench is a multi-turn dialogue between the agent, a user simulator, and a
tool environment. The runner alternates agent model calls, tool execution, and
user replies until the user ends the dialogue; the verifier then checks the
final database state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from harnessx.benchmarks.tau3.db import Database
from harnessx.benchmarks.tau3.domain import ToolSpec, UserSimulator
from harnessx.benchmarks.tau3.retail import stringify_result
from harnessx.events import Message, MessageRole
from harnessx.providers.base import Provider

logger = logging.getLogger("harnessx.tau3")


@dataclass
class DialogueResult:
    db_state: dict[str, Any] = field(default_factory=dict)
    transcript: list[dict[str, Any]] = field(default_factory=list)
    turns: int = 0
    stopped: bool = False


class DialogueRunner:
    def __init__(
        self,
        provider: Provider,
        tools: list[ToolSpec],
        user_simulator: UserSimulator,
        db: Database,
        opening: str,
        max_turns: int = 200,
    ) -> None:
        self.provider = provider
        self.tools = {t.name: t for t in tools}
        self.user_simulator = user_simulator
        self.db = db
        self.opening = opening
        self.max_turns = max_turns

    def tool_schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self.tools.values()]

    async def run(self) -> DialogueResult:
        messages: list[Message] = [Message(role=MessageRole.USER, content=self.opening)]
        transcript: list[dict[str, Any]] = []
        turns = 0
        stopped = False

        for _ in range(self.max_turns):
            turns += 1
            response = await self.provider.generate(messages, tools=self.tool_schemas())
            messages.append(
                Message(
                    role=MessageRole.ASSISTANT,
                    content=response.content,
                    tool_calls=response.tool_calls,
                )
            )
            transcript.append({"role": "assistant", "content": response.content, "tool_calls": response.tool_calls})

            if response.tool_calls:
                for tc in response.tool_calls:
                    result = await self._execute(tc["name"], tc.get("arguments", {}))
                    messages.append(
                        Message(
                            role=MessageRole.TOOL,
                            content=stringify_result(result),
                            tool_call_id=tc.get("id"),
                            name=tc["name"],
                        )
                    )
                    transcript.append({"role": "tool", "name": tc["name"], "result": result})
                continue

            reply = await self.user_simulator.respond(response.content)
            transcript.append({"role": "user", "content": reply.content})
            if reply.stop:
                stopped = True
                break
            messages.append(Message(role=MessageRole.USER, content=reply.content))

        return DialogueResult(
            db_state=self.db.snapshot(),
            transcript=transcript,
            turns=turns,
            stopped=stopped,
        )

    async def _execute(self, name: str, args: dict[str, Any]) -> Any:
        tool = self.tools.get(name)
        if tool is None:
            return {"error": f"unknown tool {name!r}"}
        try:
            return await tool.handler(self.db, args)
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}
