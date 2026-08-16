"""Shared text-action run loop for action-space benchmarks (ALFWorld, WebShop).

These benchmarks expose a text environment: the agent receives an observation,
emits a raw text action (not a structured tool call), and the environment
returns the next observation. This module provides the environment protocol and
a harness that drives the loop.
"""

from __future__ import annotations

from typing import Any, Protocol

from harnessx.core.harness import Harness
from harnessx.core.trajectory import Trajectory
from harnessx.events import Message, MessageRole


class TextEnv(Protocol):
    async def reset(self, task: Any) -> str: ...

    async def step(self, action: str) -> tuple[str, bool, Any]: ...


class TextGameHarness(Harness):
    """Harness variant for text-action environments.

    Reads the environment from ``task.env``, drives the observe→act→observe
    loop, and returns a trajectory whose ``final_output`` is a result dict with
    ``done``, ``success``, and ``reward`` populated by the environment.
    """

    async def run(self, task: Any, task_id: str | None = None) -> Trajectory:
        task_id = task_id or getattr(task, "id", None) or str(id(task))
        env: TextEnv = task.env
        provider = self.model_config.provider_for("main")

        system_prompt = self._system_prompt(task)
        obs = await env.reset(task)
        messages: list[Message] = []
        if system_prompt:
            messages.append(Message(role=MessageRole.SYSTEM, content=system_prompt))
        messages.append(Message(role=MessageRole.USER, content=obs))

        done = False
        reward: Any = None
        success: bool = False
        for _ in range(self.harness_config.max_steps):
            response = await provider.generate(messages)
            action = response.content.strip()
            messages.append(Message(role=MessageRole.ASSISTANT, content=action))
            obs, done, reward = await env.step(action)
            if done:
                success = bool(getattr(env, "success", False))
                break
            messages.append(Message(role=MessageRole.USER, content=obs))

        result = {
            "done": done,
            "success": success,
            "reward": reward,
            "final_action": messages[-1].content if messages else None,
        }
        return Trajectory(task_id=task_id, final_output=result)

    def _system_prompt(self, task: Any) -> str | None:
        prompt = getattr(task, "system_prompt", None)
        return prompt
