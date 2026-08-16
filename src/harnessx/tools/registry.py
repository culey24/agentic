"""Tool registry and builtin tool base."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Tool:
    name: str
    fn: Callable[..., Any]
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    is_async: bool = False

    async def run(self, **kwargs: Any) -> Any:
        result = self.fn(**kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(
        self,
        name: str,
        fn: Callable[..., Any],
        description: str = "",
        parameters: dict[str, Any] | None = None,
    ) -> ToolRegistry:
        self._tools[name] = Tool(
            name=name,
            fn=fn,
            description=description,
            parameters=parameters or {"type": "object", "properties": {}},
        )
        return self

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def __contains__(self, name: str) -> bool:
        return name in self._tools
