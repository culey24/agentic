"""The run loop: drives the agent, emitting hooks and running processors.

The loop is model-agnostic: it asks the ``main`` provider for a response at the
``before_model`` boundary, then routes any tool calls through the tool
registry. Processors attached to each hook transform the events that flow
through the loop; hook contracts (read-only fields, event types) are enforced.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from harnessx.core.harness_config import HarnessConfig
from harnessx.core.hooks import Hook
from harnessx.core.model_config import ModelConfig, Role
from harnessx.core.processor import Processor
from harnessx.core.state import RunState
from harnessx.core.trajectory import StepRecord, Trajectory
from harnessx.events import (
    BeforeModelEvent,
    Message,
    MessageRole,
    ModelResponseEvent,
    StepEndEvent,
    StepStartEvent,
    TaskEndEvent,
    TaskStartEvent,
    ToolCallEvent,
    ToolResultEvent,
)

logger = logging.getLogger("harnessx")


async def _run_chain(
    procs: list[Processor], event: Any
) -> AsyncIterator[Any]:
    current = [event]
    for proc in procs:
        nxt: list[Any] = []
        for ev in current:
            async for yielded in proc.process(ev):
                nxt.append(yielded)
        if not nxt:
            return
        current = nxt
    for ev in current:
        yield ev


async def dispatch(config: HarnessConfig, hook: Hook, event: Any) -> list[Any]:
    """Run all processors at ``hook`` over ``event`` and return yielded events."""
    out: list[Any] = []
    async for ev in _run_chain(config.processors_for(hook), event):
        out.append(ev)
    return out


class RunLoop:
    def __init__(
        self,
        model_config: ModelConfig,
        harness_config: HarnessConfig,
    ) -> None:
        self.model_config = model_config
        self.config = harness_config

    async def run(self, task: Any, task_id: str | None = None) -> Trajectory:
        task_id = task_id or getattr(task, "id", None) or str(id(task))
        state = RunState(task_id=task_id, task=task)
        traj = Trajectory(task_id=task_id)

        start_event = TaskStartEvent(task_id=task_id, task=task)
        await self._dispatch(Hook.TASK_START, start_event, state)
        if start_event.system_prompt:
            state.messages.append(
                Message(role=MessageRole.SYSTEM, content=start_event.system_prompt)
            )

        description = getattr(task, "description", None) or getattr(
            task, "problem_statement", None
        )
        if description:
            state.messages.append(Message(role=MessageRole.USER, content=str(description)))

        for step in range(self.config.max_steps):
            state.step = step
            step_record = StepRecord(step=step)

            step_start = StepStartEvent(step=step, messages=list(state.messages))
            await self._dispatch(Hook.STEP_START, step_start, state)
            state.messages = list(step_start.messages)
            step_record.messages = [m.__dict__ for m in state.messages]

            before = BeforeModelEvent(
                messages=list(state.messages),
                system_prompt=start_event.system_prompt,
            )
            await self._dispatch(Hook.BEFORE_MODEL, before, state)
            state.messages = list(before.messages)

            provider = self.model_config.provider_for(Role.MAIN)
            response = await provider.generate(
                state.messages, tools=self._tool_schemas()
            )
            after = ModelResponseEvent(
                content=response.content,
                tool_calls=response.tool_calls,
                raw=response.raw,
                stop_reason=response.stop_reason,
            )
            await self._dispatch(Hook.AFTER_MODEL, after, state)
            step_record.response = {
                "content": after.content,
                "tool_calls": after.tool_calls,
                "stop_reason": after.stop_reason,
            }
            state.messages.append(
                Message(
                    role=MessageRole.ASSISTANT,
                    content=after.content,
                    tool_calls=after.tool_calls,
                )
            )

            if not after.tool_calls:
                state.final_output = after.content
                state.terminated = True

            for tool_call in after.tool_calls:
                before_tool = ToolCallEvent(
                    name=tool_call["name"],
                    arguments=tool_call.get("arguments", {}),
                    tool_call_id=tool_call.get("id"),
                )
                await self._dispatch(Hook.BEFORE_TOOL, before_tool, state)
                if not before_tool.approved:
                    result = {"error": "tool call not approved"}
                else:
                    result = await self._execute_tool(
                        before_tool.name, before_tool.arguments
                    )
                after_tool = ToolResultEvent(
                    name=before_tool.name,
                    result=result,
                    tool_call_id=before_tool.tool_call_id,
                    is_error=isinstance(result, dict) and "error" in result,
                )
                await self._dispatch(Hook.AFTER_TOOL, after_tool, state)
                step_record.tool_calls.append(
                    {"name": before_tool.name, "arguments": before_tool.arguments}
                )
                step_record.tool_results.append(
                    {"name": after_tool.name, "result": after_tool.result}
                )
                state.tool_results[before_tool.name] = after_tool.result
                state.messages.append(
                    Message(
                        role=MessageRole.TOOL,
                        content=_stringify(after_tool.result),
                        tool_call_id=before_tool.tool_call_id,
                        name=before_tool.name,
                    )
                )

            step_end = StepEndEvent(step=step, messages=list(state.messages))
            await self._dispatch(Hook.STEP_END, step_end, state)
            traj.steps.append(step_record)

            if state.terminated:
                break

        end_event = TaskEndEvent(
            task_id=task_id,
            final_output=state.final_output,
            success=state.success,
            reward=state.reward,
        )
        await self._dispatch(Hook.TASK_END, end_event, state)

        traj.final_output = state.final_output
        traj.success = state.success
        traj.reward = state.reward
        return traj

    async def _dispatch(self, hook: Hook, event: Any, state: RunState) -> None:
        state.events.append(event)
        results = await dispatch(self.config, hook, event)
        if results:
            last = results[-1]
            for attr in ("task_id", "task", "system_prompt", "step", "messages",
                         "content", "tool_calls", "name", "arguments", "result",
                         "final_output", "success", "reward"):
                if hasattr(last, attr) and hasattr(event, attr):
                    setattr(event, attr, getattr(last, attr))

    def _tool_schemas(self) -> list[dict[str, Any]] | None:
        registry = self.config.tool_registry
        if registry is None:
            return None
        return registry.schemas()

    async def _execute_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        registry = self.config.tool_registry
        if registry is None:
            return {"error": f"no tool registry; unknown tool {name!r}"}
        tool = registry.get(name)
        if tool is None:
            return {"error": f"unknown tool {name!r}"}
        try:
            return await tool.run(**arguments)
        except Exception as exc:  # noqa: BLE001
            return {"error": f"tool {name!r} failed: {exc}"}


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    import json

    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)
