"""Lifecycle event types emitted by the run loop.

Each hook point (see :class:`harnessx.core.hooks.Hook`) has a dedicated event
type. Processors attach to hooks and consume/yield the matching event type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    role: MessageRole
    content: str
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Event:
    """Base event. Processors may mutate event fields; hook contracts are
    validated by the run loop after each processor invocation."""


@dataclass
class TaskStartEvent(Event):
    task_id: str
    task: Any
    system_prompt: str | None = None


@dataclass
class StepStartEvent(Event):
    step: int
    messages: list[Message]


@dataclass
class BeforeModelEvent(Event):
    messages: list[Message]
    system_prompt: str | None = None


@dataclass
class ModelResponseEvent(Event):
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    raw: Any = None
    stop_reason: str | None = None


@dataclass
class ToolCallEvent(Event):
    name: str
    arguments: dict[str, Any]
    tool_call_id: str | None = None
    approved: bool = True


@dataclass
class ToolResultEvent(Event):
    name: str
    result: Any
    tool_call_id: str | None = None
    is_error: bool = False


@dataclass
class StepEndEvent(Event):
    step: int
    messages: list[Message]


@dataclass
class TaskEndEvent(Event):
    task_id: str
    final_output: Any = None
    success: bool | None = None
    reward: float | None = None
