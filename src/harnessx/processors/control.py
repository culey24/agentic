"""Control & safety processors (D7)."""

from __future__ import annotations

from collections.abc import AsyncIterator

from harnessx.core.processor import Processor
from harnessx.events import Event, ToolCallEvent
from harnessx.processors.registry import register


@register("tool_approval")
class ToolApprovalProcessor(Processor):
    _singleton_group = "tool_approval"

    def __init__(self, allowlist: list[str] | None = None) -> None:
        self.allowlist = allowlist

    async def process(self, event: Event) -> AsyncIterator[Event]:
        if (
            isinstance(event, ToolCallEvent)
            and self.allowlist is not None
            and event.name not in self.allowlist
        ):
            event.approved = False
        yield event
