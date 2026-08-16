"""Lifecycle hook points at which processors attach.

Matches Table 1 of the HarnessX paper: each hook has a permitted set of
modifications, enforced by the run loop after every processor invocation.
"""

from __future__ import annotations

import enum

from harnessx.events import (
    BeforeModelEvent,
    Event,
    ModelResponseEvent,
    StepEndEvent,
    StepStartEvent,
    TaskEndEvent,
    TaskStartEvent,
    ToolCallEvent,
    ToolResultEvent,
)


class Hook(enum.Enum):
    TASK_START = "task_start"
    STEP_START = "step_start"
    BEFORE_MODEL = "before_model"
    AFTER_MODEL = "after_model"
    BEFORE_TOOL = "before_tool"
    AFTER_TOOL = "after_tool"
    STEP_END = "step_end"
    TASK_END = "task_end"

    @property
    def event_type(self) -> type[Event]:
        return _HOOK_EVENT_TYPE[self]

    @property
    def readonly(self) -> bool:
        return self in (Hook.STEP_END, Hook.TASK_END)


_HOOK_EVENT_TYPE: dict[Hook, type[Event]] = {
    Hook.TASK_START: TaskStartEvent,
    Hook.STEP_START: StepStartEvent,
    Hook.BEFORE_MODEL: BeforeModelEvent,
    Hook.AFTER_MODEL: ModelResponseEvent,
    Hook.BEFORE_TOOL: ToolCallEvent,
    Hook.AFTER_TOOL: ToolResultEvent,
    Hook.STEP_END: StepEndEvent,
    Hook.TASK_END: TaskEndEvent,
}
