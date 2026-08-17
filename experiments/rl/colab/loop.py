"""Pure RL loop for the tau3 multi-vs-single objective experiment.

Deliberately *not* the AEGIS co-evolution loop: the harness (domain + tools +
scripted user simulator) is fixed across rounds, so any change in the pass rate
/ objective metrics is attributable to the GRPO reward shaping alone.

Each round:
  1. roll out ``n_rollouts`` trajectories per task (temperature > 0 sampling),
  2. compute per-objective rewards + scalar reward per trajectory,
  3. run one GRPO step (token-level, all assistant turns) on the batch,
  4. checkpoint records + metrics to ``checkpoint_dir``.

End of run: pass@N history, per-objective means, and Pareto front stats over the
accumulated reward vectors.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from experiments.multiobj.objectives import make_specs, non_dominated_sort
from experiments.rl.colab.grpo_trainer import GRPOTrainer
from experiments.rl.colab.rollout import (
    RolloutRecord,
    Tau3Rollout,
    deserialize_rollout,
    serialize_rollout,
)
from harnessx.benchmarks.tau3.domain import Domain
from harnessx.benchmarks.tau3.retail import verify_retail
from harnessx.core.trajectory import Trajectory

logger = logging.getLogger("experiments.rl.loop")


@dataclass
class RoundResult:
    round: int
    pass_rate: float
    solved: int
    total: int
    mean_reward: float
    mean_rewards: dict[str, float] = field(default_factory=dict)
    n_records: int = 0
    train_loss: float | None = None
    train_kl: float | None = None
    n_tokens: int = 0


@dataclass
class LoopSummary:
    history: list[float] = field(default_factory=list)
    rounds: list[RoundResult] = field(default_factory=list)
    final_pass_rate: float = 0.0
    best_pass_rate: float = 0.0
    objective_means: dict[str, float] = field(default_factory=dict)
    pareto_fronts: int = 0
    non_dominated: int = 0
    n_records: int = 0
    n_tokens: int = 0


class PureRLLoop:
    def __init__(
        self,
        provider: Any,
        tasks: list[Any],
        domain: Domain,
        scorer: Callable[[Any, Trajectory], dict[str, float]],
        reward_fn: Callable[[dict[str, float]], float],
        trainer: GRPOTrainer | None,
        max_turns: int = 200,
        rollouts_per_task: int = 8,
        rounds: int = 6,
        concurrency: int = 4,
        checkpoint_dir: str | Path | None = None,
        user_provider: Any | None = None,
    ) -> None:
        self.provider = provider
        self.tasks = tasks
        self.domain = domain
        self.scorer = scorer
        self.reward_fn = reward_fn
        self.trainer = trainer
        self.max_turns = max_turns
        self.rollouts_per_task = rollouts_per_task
        self.rounds = rounds
        self.concurrency = concurrency
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._rollout = Tau3Rollout(
            provider=provider,
            domain=domain,
            max_turns=max_turns,
            user_provider=user_provider,
        )
        self._sem = asyncio.Semaphore(concurrency)
        self._all_records: list[RolloutRecord] = []

    async def run(self) -> LoopSummary:
        summary = LoopSummary()
        for r in range(self.rounds):
            batch = await self._rollout_batch()
            self._all_records.extend(batch)
            result = self._round_summary(r, batch)
            summary.history.append(result.pass_rate)
            summary.rounds.append(result)

            if self.trainer is not None and batch:
                stats = self.trainer.update(batch)
                result.train_loss = stats.loss
                result.train_kl = stats.mean_kl
                result.n_tokens = stats.n_tokens
                summary.n_tokens += stats.n_tokens
            else:
                summary.n_tokens += sum(rec.total_tokens for rec in batch)

            summary.n_records += len(batch)
            self._checkpoint(r, batch, result)
            logger.info(
                "round %d pass@%d=%.3f mean_reward=%.3f %s",
                r, self.rollouts_per_task, result.pass_rate, result.mean_reward,
                f"loss={result.train_loss:.4f}" if result.train_loss is not None else "(no train)",
            )

        summary.final_pass_rate = summary.history[-1] if summary.history else 0.0
        summary.best_pass_rate = max(summary.history) if summary.history else 0.0
        self._finalize(summary)
        return summary

    async def _rollout_batch(self) -> list[RolloutRecord]:
        async def _one(task: Any) -> RolloutRecord:
            async with self._sem:
                return await self._rollout.run(task, self.scorer, self.reward_fn)

        jobs = [t for t in self.tasks for _ in range(self.rollouts_per_task)]
        return list(await asyncio.gather(*(_one(t) for t in jobs)))

    def _round_summary(self, r: int, batch: list[RolloutRecord]) -> RoundResult:
        by_task: dict[str, list[RolloutRecord]] = {}
        for rec in batch:
            by_task.setdefault(rec.task_id, []).append(rec)

        total = len(self.tasks)
        solved = 0
        for t in self.tasks:
            tid = getattr(t, "id", None) or getattr(t, "task_id", None)
            recs = by_task.get(tid, [])
            if any(verify_retail(t, rec.db_state) for rec in recs):
                solved += 1

        return RoundResult(
            round=r,
            pass_rate=solved / total if total else 0.0,
            solved=solved,
            total=total,
            mean_reward=_mean([rec.reward for rec in batch]),
            mean_rewards=_objective_means(batch),
            n_records=len(batch),
        )

    def _checkpoint(self, r: int, batch: list[RolloutRecord], result: RoundResult) -> None:
        if self.checkpoint_dir is None:
            return
        rec_path = self.checkpoint_dir / "rollouts.jsonl"
        with open(rec_path, "a") as f:
            f.writelines(json.dumps(serialize_rollout(rec), ensure_ascii=False) + "\n" for rec in batch)
        summary_path = self.checkpoint_dir / "rounds.jsonl"
        with open(summary_path, "a") as f:
            f.write(json.dumps(asdict(result), ensure_ascii=False, default=str) + "\n")

    def _finalize(self, summary: LoopSummary) -> None:
        records = list(self._all_records)
        if self.checkpoint_dir is not None:
            checkpointed = self._read_checkpointed()
            if checkpointed:
                records = checkpointed
        if not records:
            return
        summary.n_records = len(records)
        summary.objective_means = _objective_means(records)
        vectors = [r.rewards for r in records if r.rewards]
        if vectors:
            names = sorted({n for v in vectors for n in v})
            specs = make_specs({n: 1.0 for n in names})
            fronts = non_dominated_sort(vectors, specs)
            summary.pareto_fronts = len(fronts)
            summary.non_dominated = len(fronts[0]) if fronts else 0

    def _read_checkpointed(self) -> list[RolloutRecord]:
        if self.checkpoint_dir is None:
            return []
        path = self.checkpoint_dir / "rollouts.jsonl"
        if not path.exists():
            return []
        records: list[RolloutRecord] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                records.append(deserialize_rollout(data.get("task_id", "unknown"), data))
        return records


def _objective_means(batch: list[RolloutRecord]) -> dict[str, float]:
    names: set[str] = set()
    for rec in batch:
        names.update(rec.rewards.keys())
    means: dict[str, float] = {}
    for name in sorted(names):
        vals = [rec.rewards.get(name, 0.0) for rec in batch]
        means[name] = _mean(vals)
    return means


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0