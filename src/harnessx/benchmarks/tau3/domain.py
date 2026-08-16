"""Domain abstraction for τ³-Bench.

A domain owns a database, a set of tools (schema + handler), and the user
simulator policy. The verifier checks the final database state against the
task's expected outcome.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from harnessx.benchmarks.tau3.db import Database


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[Database, dict[str, Any]], Awaitable[Any]]

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class UserMessage:
    content: str
    stop: bool = False


class UserSimulator:
    """Rule-based user persona. Responds to agent messages and eventually stops."""

    async def respond(self, agent_message: str) -> UserMessage:
        raise NotImplementedError


class Domain:
    name: str = "domain"

    def build_db(self) -> Database:
        raise NotImplementedError

    def tools(self) -> list[ToolSpec]:
        raise NotImplementedError

    def user_simulator(self, task: Any, provider: Any = None) -> UserSimulator:
        raise NotImplementedError
