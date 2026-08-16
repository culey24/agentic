from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from harnessx.events import Message
from harnessx.providers.base import Provider, ProviderResponse


class EchoProvider(Provider):
    async def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        return ProviderResponse(content="answer: 4", stop_reason="end_turn")


@dataclass
class Task:
    id: str
    description: str
