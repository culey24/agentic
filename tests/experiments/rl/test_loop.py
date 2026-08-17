"""Offline tests for the pure RL loop using a deterministic good provider."""

from __future__ import annotations

import asyncio

from experiments.multiobj.scorers.tau3 import make_tau3_scorer
from experiments.rl.colab.grpo_trainer import GRPOStats
from experiments.rl.colab.local_provider import parse_assistant
from experiments.rl.colab.loop import PureRLLoop
from experiments.rl.colab.rollout import (
    RolloutRecord,
    deserialize_rollout,
    serialize_rollout,
)
from harnessx.benchmarks.tau3 import Tau3Task
from harnessx.benchmarks.tau3.retail import RetailDomain
from harnessx.providers.base import Provider, ProviderResponse


class GoodCancelProvider(Provider):
    """Deterministically solves cancel-#123 dialogues (tool call then messages)."""

    def __init__(self) -> None:
        super().__init__("good-cancel")
        self.generate_calls = 0

    def tokenize(self, messages: list, tools: list | None = None) -> list[int]:
        return list(range(4))

    def compute_logprobs(self, prompt: list[int], completion: list[int]) -> dict:
        n = len(completion) or 1
        return {"logprobs": [-0.5] * n, "ref_logprobs": [-0.5] * n}

    async def generate(self, messages: list, tools: list | None = None, **kwargs) -> ProviderResponse:
        self.generate_calls += 1
        used_tool = any(getattr(m, "tool_calls", None) for m in messages)
        if not used_tool:
            text = '{"tool_call": {"name": "cancel_order", "arguments": {"order_id": "123"}}}'
            content, calls = parse_assistant(text, tools)
            return ProviderResponse(content=content, tool_calls=calls, raw={"text": text, "token_ids": [1, 2, 3]})
        text = "I've cancelled order #123."
        return ProviderResponse(content=text, tool_calls=[], raw={"text": text, "token_ids": [4, 5]})


class FakeTrainer:
    def __init__(self) -> None:
        self.updates: list[list[RolloutRecord]] = []

    def update(self, rollouts: list[RolloutRecord]) -> GRPOStats:
        self.updates.append(rollouts)
        n_tokens = sum(r.total_tokens for r in rollouts)
        return GRPOStats(loss=0.5, mean_reward=1.0, mean_advantage=0.5, std_advantage=0.1, mean_kl=0.1, n_tokens=n_tokens, n_trajectories=len(rollouts))


def _cancel_task() -> Tau3Task:
    return Tau3Task(
        task_id="colab-cancel-001",
        domain="retail",
        opening="Please cancel order #123.",
        script=[
            {"reply": "Yes, please cancel it.", "stop": False},
            {"reply": "Great, thanks!", "stop": True},
        ],
        expected={"orders": [{"key_field": "order_id", "key": "123", "fields": {"status": "cancelled"}}]},
        instruction=None,
    )


def test_loop_runs_and_checkpoints(tmp_path) -> None:
    provider = GoodCancelProvider()
    trainer = FakeTrainer()
    scorer = make_tau3_scorer(max_turns=200)
    loop = PureRLLoop(
        provider=provider,
        tasks=[_cancel_task()],
        domain=RetailDomain(),
        scorer=scorer,
        reward_fn=lambda r: r["correctness"],
        trainer=trainer,
        max_turns=200,
        rollouts_per_task=2,
        rounds=2,
        concurrency=1,
        checkpoint_dir=tmp_path,
    )

    async def main():
        return await loop.run()

    summary = asyncio.run(main())

    assert summary.history == [1.0, 1.0]
    assert summary.final_pass_rate == 1.0
    assert summary.n_records == 4
    assert summary.objective_means["correctness"] == 1.0
    assert summary.pareto_fronts >= 1
    assert len(trainer.updates) == 2

    rounds_path = tmp_path / "rounds.jsonl"
    recs_path = tmp_path / "rollouts.jsonl"
    assert rounds_path.exists()
    assert recs_path.exists()
    lines = recs_path.read_text().strip().splitlines()
    assert len(lines) == 4


def test_serialize_roundtrip(tmp_path) -> None:
    rec = RolloutRecord(
        task_id="t1",
        reward=0.7,
        rewards={"correctness": 1.0, "efficiency": 0.5},
        process_reward=-1.0,
        turns_count=2,
        stopped=True,
        db_state={"orders": []},
        transcript=[{"role": "user", "content": "hi"}],
    )
    from experiments.rl.colab.rollout import TurnRecord

    rec.turns = [TurnRecord(prompt_tokens=[1, 2], completion_tokens=[3, 4], log_probs=[-0.1, -0.2], ref_log_probs=[-0.1, -0.2], content="x")]
    data = serialize_rollout(rec)
    roundtrip = deserialize_rollout("t1", data)
    assert roundtrip.reward == 0.7
    assert roundtrip.rewards == {"correctness": 1.0, "efficiency": 0.5}
    assert roundtrip.process_reward == -1.0
    assert roundtrip.turns[0].completion_tokens == [3, 4]
    assert roundtrip.turns[0].log_probs == [-0.1, -0.2]
    assert roundtrip.turns_count == 2