"""Evaluation helpers and the adaptation loop (Section 4.4).

One round of adaptation: execute the current harness on the adaptation batch,
selectively invoke Digester/Planner/Evolver, then run Critic + deterministic
gate. A round commits a new harness only if a candidate clears all checks;
otherwise it is a no-op.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from math import comb
from typing import Any, Protocol

from harnessx.core.harness import Harness
from harnessx.core.harness_config import HarnessConfig
from harnessx.core.trajectory import Trajectory
from harnessx.evolve.builder import Builder
from harnessx.evolve.critic import Critic
from harnessx.evolve.digester import Digester
from harnessx.evolve.evolver import Evolver
from harnessx.evolve.gate import Gate
from harnessx.evolve.planner import Planner
from harnessx.evolve.types import Digest, VerdictAction
from harnessx.tracing.journal import Journal

logger = logging.getLogger("harnessx.evolve")


class Verifier(Protocol):
    async def __call__(self, task: Any, final_output: Any) -> bool: ...


@dataclass
class EvaluationResult:
    task_id: str
    outcomes: list[bool] = field(default_factory=list)
    trajectories: list[Trajectory] = field(default_factory=list)

    @property
    def solved(self) -> bool:
        return any(self.outcomes)


def pass_at_k(outcomes: list[bool], k: int) -> float:
    n = len(outcomes)
    c = sum(outcomes)
    k = min(k, n)
    if n == 0:
        return 0.0
    if c == 0:
        return 0.0
    return 1.0 - comb(n - c, k) / comb(n, k)


async def evaluate(
    harness: Harness,
    tasks: list[Any],
    verifier: Verifier,
    n_rollouts: int = 2,
    concurrency: int = 10,
) -> dict[str, EvaluationResult]:
    sem = asyncio.Semaphore(concurrency)

    async def run_task(task: Any) -> EvaluationResult:
        task_id = getattr(task, "id", None) or str(id(task))
        async with sem:
            trajectories: list[Trajectory] = []
            outcomes: list[bool] = []
            for _ in range(n_rollouts):
                traj = await harness.run(task, task_id=task_id)
                ok = await verifier(task, traj.final_output)
                traj.success = ok
                traj.reward = 1.0 if ok else 0.0
                trajectories.append(traj)
                outcomes.append(ok)
            return EvaluationResult(task_id=task_id, outcomes=outcomes, trajectories=trajectories)

    results = await asyncio.gather(*(run_task(t) for t in tasks))
    return {r.task_id: r for r in results}


@dataclass
class EvolutionLoop:
    meta_provider: Any
    harness: Harness
    tasks: list[Any]
    verifier: Verifier
    journal: Journal
    n_rollouts: int = 2
    concurrency: int = 10
    candidates_per_round: int = 4
    max_rounds: int = 15
    patience: int = 3
    noise_threshold: float = 0.05
    base_config: HarnessConfig | None = None

    def __post_init__(self) -> None:
        self.base_config = self.base_config or self.harness.harness_config
        self.digester = Digester(self.meta_provider)
        self.planner = Planner(self.meta_provider)
        self.evolver = Evolver(self.meta_provider, self.candidates_per_round)
        self.critic = Critic(self.meta_provider)
        self.gate = Gate(Builder())

    async def run(self) -> dict[str, Any]:
        prior_edits: list[str] = []
        best_rate = -1.0
        stale_rounds = 0
        history: list[float] = []

        for round_ in range(self.max_rounds):
            results = await evaluate(
                self.harness, self.tasks, self.verifier,
                n_rollouts=self.n_rollouts, concurrency=self.concurrency,
            )
            outcomes = {tid: r.solved for tid, r in results.items()}
            rate = sum(outcomes.values()) / len(self.tasks) if self.tasks else 0.0
            history.append(rate)
            self.journal.record_curve(round_, rate)
            self.journal.log(
                round_,
                f"pass@{self.n_rollouts}={rate:.3f}",
                pass_rate=rate,
                solved=sum(outcomes.values()),
            )
            logger.info("round %d pass_rate=%.3f", round_, rate)

            trajectories = {
                tid: r.trajectories[-1] for tid, r in results.items()
            }
            digests = await self.digester.digest(trajectories, outcomes)
            landscape = await self.planner.plan(round_, digests, prior_edits)

            shipped = await self._try_ship(round_, landscape, digests, results, outcomes)

            if shipped is None:
                self.journal.log(round_, "no candidate shipped (no-op round)")
                stale_rounds += 1
            else:
                shipped_config, components = shipped
                self.harness.harness_config = shipped_config
                self.base_config = shipped_config
                prior_edits.extend(components)
                stale_rounds = 0

            if rate > best_rate:
                best_rate = rate
                stale_rounds = 0
            if stale_rounds >= self.patience:
                logger.info("early stop after %d stale rounds", stale_rounds)
                break

        return {"history": history, "best": best_rate, "final": history[-1] if history else None}

    async def _try_ship(
        self,
        round_: int,
        landscape: Any,
        digests: list[Digest],
        results: dict[str, EvaluationResult],
        outcomes: dict[str, bool],
    ) -> tuple[HarnessConfig, list[str]] | None:
        candidates = await self.evolver.evolve(landscape, self.base_config)
        for candidate in candidates:
            verdict = await self.critic.critique(candidate, digests)
            self.journal.audit(
                "critic", verdict.action.value, candidate=candidate.name,
                reason=verdict.reason,
            )
            if verdict.action == VerdictAction.NO_OP:
                continue

            seesaw = self._make_seesaw(results, outcomes)
            gate_result = await self.gate.check(
                candidate.manifest, self.base_config, seesaw=seesaw
            )
            self.journal.audit(
                "gate", "pass" if gate_result.passed else "fail",
                candidate=candidate.name, gate_stage=gate_result.stage,
                reason=gate_result.reason,
            )
            if gate_result.passed:
                self.journal.log(
                    round_,
                    f"shipped {candidate.name}: {candidate.manifest.intended_effect}",
                    candidate=candidate.name,
                )
                return gate_result.config, candidate.manifest.edited_components
        return None

    def _make_seesaw(
        self,
        results: dict[str, EvaluationResult],
        outcomes: dict[str, bool],
    ) -> Callable[[HarnessConfig], tuple[bool, str]]:
        previously_passing = [tid for tid, ok in outcomes.items() if ok]

        async def seesaw(config: HarnessConfig) -> tuple[bool, str]:
            if not previously_passing:
                return True, "no previously passing tasks"
            candidate_harness = Harness(
                model_config=self.harness.model_config, harness_config=config
            )
            subset = previously_passing[:10]
            new_results = await evaluate(
                candidate_harness,
                [t for t in self.tasks if getattr(t, "id", str(id(t))) in subset],
                self.verifier,
                n_rollouts=1,
                concurrency=self.concurrency,
            )
            new_ok = sum(r.solved for r in new_results.values())
            if new_ok < len(subset) * (1 - self.noise_threshold):
                return False, f"seesaw regression: {new_ok}/{len(subset)}"
            return True, f"seesaw ok: {new_ok}/{len(subset)}"

        return seesaw
