"""Offline MDP rollout tests using a scripted fake provider (no GPU)."""

from __future__ import annotations

import asyncio
import json

import pytest

from experiments.multiobj.scorers.tau3 import make_tau3_scorer
from experiments.rl.colab.local_provider import parse_assistant
from experiments.rl.colab.rollout import (
    Tau3Rollout,
    single_objective_reward,
    weighted_objective_reward,
)
from harnessx.benchmarks.tau3 import Tau3Task
from harnessx.benchmarks.tau3.retail import RetailDomain
from harnessx.providers.base import Provider, ProviderResponse


class ScriptedProvider(Provider):
    """Emits a fixed plan of tool calls / messages, mimicking the local provider."""

    def __init__(self, plan: list[dict]) -> None:
        super().__init__("scripted")
        self.plan = list(plan)
        self.index = 0

    def tokenize(self, messages: list, tools: list | None = None) -> list[int]:
        return list(range(10 + self.index))

    def compute_logprobs(self, prompt_tokens: list[int], completion_tokens: list[int]) -> dict:
        n = len(completion_tokens) or 1
        return {"logprobs": [-0.5] * n, "ref_logprobs": [-0.5] * n}

    async def generate(self, messages: list, tools: list | None = None, **kwargs) -> ProviderResponse:
        step = self.plan[self.index]
        self.index += 1
        if "tool_call" in step:
            call = step["tool_call"]
            text = json.dumps({"tool_call": {"name": call["name"], "arguments": call.get("arguments", {})}}, ensure_ascii=False)
            content, tool_calls = parse_assistant(text, tools)
            return ProviderResponse(content=content, tool_calls=tool_calls, raw={"text": text, "token_ids": [1, 2, 3]})
        text = step["content"]
        return ProviderResponse(content=text, tool_calls=[], raw={"text": text, "token_ids": [4, 5]})


def _task() -> Tau3Task:
    return Tau3Task(
        task_id="colab-cancel-001",
        domain="retail",
        opening="Please cancel order #123.",
        script=[
            {"reply": "Yes, please cancel it.", "stop": False},
            {"reply": "Great, thanks!", "stop": True},
        ],
        expected={"orders": [{"key_field": "order_id", "key": "123", "fields": {"status": "cancelled"}}]},
        instruction="You are Alice. Cancel order #123.",
    )


def _rollout(plan: list[dict]) -> any:
    provider = ScriptedProvider(plan)
    domain = RetailDomain()
    scorer = make_tau3_scorer(max_turns=200)
    runner = Tau3Rollout(provider=provider, domain=domain, max_turns=200)

    async def main():
        return await runner.run(_task(), scorer, single_objective_reward)

    return asyncio.run(main())


def test_successful_cancel_turn_structure() -> None:
    record = _rollout(
        [
            {"tool_call": {"name": "cancel_order", "arguments": {"order_id": "123"}}},
            {"content": "I've cancelled order #123."},
            {"content": "Glad to help!"},
        ]
    )
    assert record.rewards["correctness"] == 1.0
    assert record.reward == pytest.approx(1.0)
    assert record.process_reward == 0.0
    assert len(record.turns) == 3
    assert all(t.completion_tokens for t in record.turns)
    assert all(len(t.log_probs) == len(t.completion_tokens) for t in record.turns)


def test_process_reward_penalizes_tool_error() -> None:
    record = _rollout(
        [
            {"tool_call": {"name": "cancel_order", "arguments": {"order_id": "999"}}},
            {"tool_call": {"name": "cancel_order", "arguments": {"order_id": "123"}}},
            {"content": "Done."},
            {"content": "Glad to help!"},
        ]
    )
    assert record.rewards["correctness"] == 1.0
    assert record.process_reward == -1.0
    assert record.reward == pytest.approx(0.0)
    assert record.turns[0].process_reward == -1.0


def test_single_vs_weighted_reward() -> None:
    record = _rollout(
        [
            {"tool_call": {"name": "cancel_order", "arguments": {"order_id": "123"}}},
            {"content": "Done."},
            {"content": "Glad to help!"},
        ]
    )
    single = single_objective_reward(record.rewards)
    multi = weighted_objective_reward({"correctness": 1.0, "efficiency": 0.3, "tool_safety": 0.5})(record.rewards)
    assert single == pytest.approx(1.0)
    assert multi < single  # efficiency < 1 pulls the weighted sum down
    assert 0.0 <= multi <= 1.0


def test_max_turns_budget_enforced() -> None:
    task = Tau3Task(
        task_id="colab-max-001",
        domain="retail",
        opening="Tell me everything.",
        script=[{"reply": f"more {i}", "stop": False} for i in range(50)],
        expected={},
        instruction=None,
    )
    plan = [{"content": f"msg {i}"} for i in range(50)]
    provider = ScriptedProvider(plan)
    domain = RetailDomain()
    scorer = make_tau3_scorer(max_turns=200)
    runner = Tau3Rollout(provider=provider, domain=domain, max_turns=4)

    async def main():
        return await runner.run(task, scorer, single_objective_reward)

    record = asyncio.run(main())
    assert record.turns_count == 4
    assert record.stopped is False