"""Processor abstraction: the typed atomic unit of harness behavior.

A processor consumes one event and yields zero or more events, producing
exactly one of five outcomes: pass-through, transform, split, intercept, or
interrupt. Processors at a given hook consume and yield the same event type,
so they compose by sequential application.
"""

from __future__ import annotations

import enum
from collections.abc import AsyncIterator
from typing import Any, ClassVar

from harnessx.events import Event


class Order(enum.IntEnum):
    PRE = 0
    NORMAL = 1
    POST = 2


class Outcome(enum.Enum):
    PASS = "pass"
    TRANSFORM = "transform"
    SPLIT = "split"
    INTERCEPT = "intercept"
    INTERRUPT = "interrupt"


class Processor:
    """Base class for all processors.

    Class-level metadata governing composition:

    - ``_singleton_group``: names a mutual-exclusion class; at most one
      processor per group may occupy a hook.
    - ``_order``: ordering hint within a hook (PRE / NORMAL / POST).
    - ``_after``: soft dependencies on other singleton groups.
    """

    _singleton_group: ClassVar[str | None] = None
    _order: ClassVar[Order] = Order.NORMAL
    _after: ClassVar[list[str]] = []

    name: ClassVar[str] = "processor"

    async def process(self, event: Event) -> AsyncIterator[Event]:
        yield event

    def configure(self, **kwargs: Any) -> Processor:
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise AttributeError(f"{type(self).__name__} has no attribute {key!r}")
            setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        return f"{type(self).__name__}(group={self._singleton_group!r})"
