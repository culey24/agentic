"""Context-assembly processors (D2)."""

from __future__ import annotations

from collections.abc import AsyncIterator

from harnessx.core.processor import Order, Processor
from harnessx.events import BeforeModelEvent, Event, TaskStartEvent
from harnessx.processors.registry import register


@register("system_prompt")
class SystemPromptProcessor(Processor):
    _singleton_group = "system_prompt"
    _order = Order.PRE

    def __init__(self, prompt: str = "") -> None:
        self.prompt = prompt

    async def process(self, event: Event) -> AsyncIterator[Event]:
        if isinstance(event, TaskStartEvent):
            event.system_prompt = self.prompt
        yield event


@register("history_trim")
class HistoryTrimProcessor(Processor):
    _singleton_group = "history_trim"
    _order = Order.NORMAL

    def __init__(self, max_messages: int = 40) -> None:
        self.max_messages = max_messages

    async def process(self, event: Event) -> AsyncIterator[Event]:
        if isinstance(event, BeforeModelEvent) and len(event.messages) > self.max_messages:
            event.messages = event.messages[-self.max_messages :]
        yield event
