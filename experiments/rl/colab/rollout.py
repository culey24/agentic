"""Multi-turn tau3 rollout as an MDP for agentic GRPO.

Each assistant turn is one action: the policy emits either a JSON tool call or a
plain message. The environment (retail tools + user simulator) returns
observations, and per-turn process rewards (tool errors -> -1) plus a terminal
objective vector (correctness / efficiency / tool_safety) close the loop.

For training, every assistant turn records ``prompt_tokens`` (conversation so
far), ``completion_tokens`` + behavior/ref log-probs, so the GRPO trainer can
apply a token-level policy gradient over the whole trajectory without re-rollout.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from harnessx.benchmarks.tau3.db import Database
from harnessx.benchmarks.tau3.domain import Domain
from harnessx.benchmarks.tau3.retail import stringify_result
from harnessx.benchmarks.tau3.runner import DialogueResult
from harnessx.core.trajectory import Trajectory
from harnessx.events import Message, MessageRole
from harnessx.providers.base import Provider

RewardFn = Callable[[dict[str, float]], float]


@dataclass
class TurnRecord:
    prompt_tokens: list[int]
    completion_tokens: list[int]
    log_probs: list[float] = field(default_factory=list)
    ref_log_probs: list[float] = field(default_factory=list)
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    process_reward: float = 0.0

    @property
    def n_tokens(self) -> int:
        return len(self.completion_tokens)


@dataclass
class RolloutRecord:
    task_id: str
    turns: list[TurnRecord] = field(default_factory=list)
    rewards: dict[str, float] = field(default_factory=dict)
    reward: float = 0.0
    process_reward: float = 0.0
    turns_count: int = 0
    stopped: bool = False
    db_state: dict[str, Any] = field(default_factory=dict)
    transcript: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return sum(t.n_tokens for t in self.turns)

    @property
    def generated_text(self) -> str:
        return "\n".join(t.content for t in self.turns if t.content)


def _is_error(result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    if "error" in result:
        return True
    inner = result.get("result")
    return isinstance(inner, dict) and "error" in inner


def _build_trajectory(record: RolloutRecord) -> Trajectory:
    final = DialogueResult(
        db_state=record.db_state,
        transcript=record.transcript,
        turns=record.turns_count,
        stopped=record.stopped,
    )
    return Trajectory(task_id=record.task_id, final_output=final)


class Tau3Rollout:
    """Runs tau3 dialogues step-by-step, capturing per-turn training data."""

    def __init__(
        self,
        provider: Provider,
        domain: Domain,
        max_turns: int = 200,
        capture_logprobs: bool = True,
        user_provider: Provider | None = None,
    ) -> None:
        self.provider = provider
        self.domain = domain
        self.max_turns = max_turns
        self.capture_logprobs = capture_logprobs
        self.user_provider = user_provider

    async def run(
        self,
        task: Any,
        scorer: Callable[[Any, Trajectory], dict[str, float]],
        reward_fn: RewardFn,
    ) -> RolloutRecord:
        db = self.domain.build_db()
        tools = self.domain.tools()
        tool_schemas = [t.schema() for t in tools]
        user_sim = self.domain.user_simulator(task, provider=self.user_provider)

        messages: list[Message] = [Message(role=MessageRole.USER, content=task.opening)]
        transcript: list[dict[str, Any]] = []
        turns: list[TurnRecord] = []
        stopped = False

        for _ in range(self.max_turns):
            prompt_tokens = self.provider.tokenize(messages, tool_schemas)
            response = await self.provider.generate(messages, tool_schemas)
            completion_tokens = list(response.raw.get("token_ids") or [])

            log_probs: list[float] = []
            ref_log_probs: list[float] = []
            if self.capture_logprobs and completion_tokens:
                lp = self.provider.compute_logprobs(prompt_tokens, completion_tokens)
                log_probs = lp.get("logprobs", [])
                ref_log_probs = lp.get("ref_logprobs", [])

            messages.append(
                Message(
                    role=MessageRole.ASSISTANT,
                    content=response.content,
                    tool_calls=response.tool_calls,
                )
            )
            transcript.append(
                {"role": "assistant", "content": response.content, "tool_calls": response.tool_calls}
            )

            process_reward = 0.0
            if response.tool_calls:
                for tc in response.tool_calls:
                    result = await self._execute(tc.get("name"), tc.get("arguments", {}), tools, db)
                    if _is_error(result):
                        process_reward -= 1.0
                    messages.append(
                        Message(
                            role=MessageRole.TOOL,
                            content=stringify_result(result),
                            tool_call_id=tc.get("id"),
                            name=tc.get("name"),
                        )
                    )
                    transcript.append({"role": "tool", "name": tc.get("name"), "result": result})
                turns.append(
                    TurnRecord(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        log_probs=log_probs,
                        ref_log_probs=ref_log_probs,
                        content=response.content,
                        tool_calls=response.tool_calls,
                        process_reward=process_reward,
                    )
                )
                continue

            reply = await user_sim.respond(response.content)
            transcript.append({"role": "user", "content": reply.content})
            turns.append(
                TurnRecord(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    log_probs=log_probs,
                    ref_log_probs=ref_log_probs,
                    content=response.content,
                    process_reward=process_reward,
                )
            )
            if reply.stop:
                stopped = True
                break
            messages.append(Message(role=MessageRole.USER, content=reply.content))

        record = RolloutRecord(
            task_id=getattr(task, "id", None) or getattr(task, "task_id", "unknown"),
            turns=turns,
            turns_count=len(turns),
            stopped=stopped,
            db_state=db.snapshot(),
            transcript=transcript,
        )
        record.process_reward = sum(t.process_reward for t in turns)

        traj = _build_trajectory(record)
        record.rewards = scorer(task, traj)
        record.reward = float(reward_fn(record.rewards)) + record.process_reward
        return record

    async def _execute(
        self, name: str, args: dict[str, Any], tools: list[Any], db: Database
    ) -> Any:
        tool = next((t for t in tools if t.name == name), None)
        if tool is None:
            return {"error": f"unknown tool {name!r}"}
        try:
            return await tool.handler(db, args)
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}


def single_objective_reward(records: dict[str, float]) -> float:
    """Single-objective arm: correctness only."""
    return float(records.get("correctness", 0.0))


def weighted_objective_reward(weights: dict[str, float]) -> RewardFn:
    """Multi-objective arm: weighted-sum scalarization."""
    from experiments.multiobj.objectives import make_specs, scalarize

    specs = make_specs(weights)

    def _fn(records: dict[str, float]) -> float:
        return scalarize(records, specs)

    return _fn


def serialize_rollout(record: RolloutRecord) -> dict[str, Any]:
    """JSON-serializable form (for journal / checkpointing to Drive)."""
    return {
        "task_id": record.task_id,
        "reward": record.reward,
        "rewards": record.rewards,
        "process_reward": record.process_reward,
        "turns": [
            {
                "prompt_tokens": t.prompt_tokens,
                "completion_tokens": t.completion_tokens,
                "log_probs": t.log_probs,
                "ref_log_probs": t.ref_log_probs,
                "content": t.content,
                "tool_calls": t.tool_calls,
                "process_reward": t.process_reward,
            }
            for t in record.turns
        ],
        "db_state": record.db_state,
        "transcript": record.transcript,
        "turns_count": record.turns_count,
        "stopped": record.stopped,
        "total_tokens": record.total_tokens,
    }


def deserialize_rollout(task_id: str, data: dict[str, Any]) -> RolloutRecord:
    turns = [
        TurnRecord(
            prompt_tokens=t["prompt_tokens"],
            completion_tokens=t["completion_tokens"],
            log_probs=t.get("log_probs", []),
            ref_log_probs=t.get("ref_log_probs", []),
            content=t.get("content", ""),
            tool_calls=t.get("tool_calls", []),
            process_reward=t.get("process_reward", 0.0),
        )
        for t in data.get("turns", [])
    ]
    return RolloutRecord(
        task_id=task_id,
        turns=turns,
        rewards=data.get("rewards", {}),
        reward=data.get("reward", 0.0),
        process_reward=data.get("process_reward", 0.0),
        turns_count=data.get("turns_count", 0),
        stopped=data.get("stopped", False),
        db_state=data.get("db_state", {}),
        transcript=data.get("transcript", []),
    )


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)