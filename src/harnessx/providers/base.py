"""Provider base: async LLM backends (Anthropic, OpenAI-compatible, etc.)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from harnessx.events import Message


@dataclass
class ProviderResponse:
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str | None = None
    raw: Any = None


class Provider:
    """Base class for model backends.

    Subclasses implement :meth:`generate`, returning a ``ProviderResponse``.
    """

    def __init__(self, model: str, **kwargs: Any) -> None:
        self.model = model
        self.kwargs = kwargs

    async def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r})"
