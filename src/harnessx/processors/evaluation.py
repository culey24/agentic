"""Evaluation & reward processors (D6)."""

from __future__ import annotations

from collections.abc import AsyncIterator

from harnessx.core.processor import Processor
from harnessx.events import Event, TaskEndEvent
from harnessx.processors.registry import register


@register("reward_annotate")
class RewardAnnotateProcessor(Processor):
    _singleton_group = "reward_annotate"

    def __init__(self, reward: float | None = None) -> None:
        self.reward = reward

    async def process(self, event: Event) -> AsyncIterator[Event]:
        if isinstance(event, TaskEndEvent) and event.reward is None:
            event.reward = self.reward
        yield event
