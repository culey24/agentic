"""EvolutionLoop wired for multi-objective experiments.

Subclasses ``harnessx.evolve.loop.EvolutionLoop`` and swaps in
:class:`MultiObjectiveBridge` (which carries reward vectors + per-step
observations in ``RLRecord.extra``) and a ``MultiObjectiveGRPOTrainer``.
Nothing under ``src/harnessx`` is modified.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from experiments.multiobj.bridge import MultiObjectiveBridge
from experiments.multiobj.scorers.tau3 import make_tau3_scorer
from experiments.multiobj.trainer import MultiObjectiveGRPOTrainer
from harnessx.evolve.loop import EvolutionLoop
from harnessx.rl.buffer import MixedPolicyBuffer
from harnessx.rl.grpo import GRPOConfig

logger = logging.getLogger("experiments.multiobj")


@dataclass
class MultiObjectiveEvolutionLoop(EvolutionLoop):
    """Co-evolution loop collecting multi-objective RL records.

    ``objective_weights`` scalarizes the reward vector into the record's scalar
    ``reward`` (weighted sum) and drives per-objective advantages. The task
    objects needed by the scorer are resolved from ``self.tasks`` by task id.
    """

    objective_weights: dict[str, float] | None = None
    binary_correctness: bool = False
    max_turns: int = 200
    buffer: Any = field(default=None)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.buffer is None:
            self.buffer = MixedPolicyBuffer(capacity=10000)
        scorer = make_tau3_scorer(
            max_turns=self.max_turns, binary=self.binary_correctness
        )
        self._bridge = MultiObjectiveBridge(
            objective_scorer=scorer, objective_weights=self.objective_weights
        )
        self._task_by_id = {
            getattr(t, "id", None) or str(id(t)): t for t in (self.tasks or [])
        }
        if self.trainer is None:
            self.trainer = MultiObjectiveGRPOTrainer(
                batch_size=self.train_batch_size,
                objective_weights=self.objective_weights,
            )

    async def _collect_records(self, results: dict[str, Any], version: int) -> None:
        if self.buffer is None:
            return
        for result in results.values():
            task = self._task_by_id.get(result.task_id)
            for traj in result.trajectories:
                record = self._bridge.to_record(
                    traj, harness_version=f"R{version}", task=task
                )
                self.buffer.insert(record)

    async def _train(self) -> None:
        if self.buffer is None or self.trainer is None:
            return
        batch = self.buffer.sample(min(self.train_batch_size, len(self.buffer)))
        if not batch:
            return
        result = await self.trainer.update(batch, GRPOConfig())
        self._train_steps += 1
        self.journal.audit(
            "train", "update",
            step=self._train_steps,
            records=result.records,
            objective=result.objective,
            updated=result.updated,
            detail=result.detail,
        )
        logger.info("train step %d on %d records", self._train_steps, result.records)