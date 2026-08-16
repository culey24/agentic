"""Mutable per-run state shared across processors within a single task run.

Processor instances may hold instance-private state, but the run loop and
observability layer share this object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from harnessx.events import Event, Message


@dataclass
class RunState:
    task_id: str
    task: Any = None
    messages: list[Message] = field(default_factory=list)
    step: int = 0
    tool_results: dict[str, Any] = field(default_factory=dict)
    final_output: Any = None
    success: bool | None = None
    reward: float | None = None
    events: list[Event] = field(default_factory=list)
    terminated: bool = False
