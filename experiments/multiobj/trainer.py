"""Multi-objective GRPO trainer implementing the ``ModelTrainer`` protocol.

Reads reward vectors from ``RLRecord.extra["rewards"]`` (written by
:class:`MultiObjectiveBridge`), computes per-objective group-relative advantages
with weighted-sum scalarization, and reports Pareto front structure. Falls back
to the scalar ``GRPOTrainer`` path when records carry no reward vector.
"""

from __future__ import annotations

import logging

from experiments.multiobj.objectives import (
    make_specs,
    multi_group_relative_advantage,
    non_dominated_sort,
)
from harnessx.rl import RLRecord, TrainResult
from harnessx.rl.coevolution import _DEFAULT_GRPO, GRPOTrainer
from harnessx.rl.grpo import GRPOConfig, clipped_objective

logger = logging.getLogger("experiments.multiobj")


class MultiObjectiveCollectOnlyTrainer:
    """API-only trainer: accumulates records, logs scalar + per-objective means."""

    def __init__(self, log: bool = True) -> None:
        self.log = log

    async def update(self, records: list[RLRecord], config: GRPOConfig | None = None) -> TrainResult:
        if self.log and records:
            scalar = [r.reward for r in records]
            logger.info(
                "collect-only: %d records, mean scalar reward=%.3f",
                len(records), _mean(scalar),
            )
            multi = [r.extra.get("rewards") for r in records]
            if any(isinstance(rw, dict) and rw for rw in multi):
                names = sorted({name for rw in multi if isinstance(rw, dict) for name in rw})
                means = {
                    name: _mean([(rw or {}).get(name, 0.0) for rw in multi])
                    for name in names
                }
                logger.info("collect-only objectives: %s", means)
        return TrainResult(records=len(records), updated=False)


class MultiObjectiveGRPOTrainer(GRPOTrainer):
    """GRPO over reward vectors (reporting, not updating).

    Without cached token log-probabilities, importance ratios are unavailable;
    this mirrors ``harnessx.rl.coevolution.GRPOTrainer`` (which reports
    statistics only). Swap in a VERL-backed trainer for actual gradient steps.
    """

    def __init__(self, batch_size: int = 256, objective_weights: dict[str, float] | None = None) -> None:
        super().__init__(batch_size=batch_size)
        self.objective_weights = objective_weights
        self._scalar = GRPOTrainer(batch_size=batch_size)

    async def update(self, records: list[RLRecord], config: GRPOConfig | None = None) -> TrainResult:
        config = config or _DEFAULT_GRPO
        batch = records[: self.batch_size]
        if not batch:
            return TrainResult(records=0)
        rewards = [r.extra.get("rewards") for r in batch]
        if not any(isinstance(rw, dict) and rw for rw in rewards):
            return await self._scalar.update(batch, config)
        return self._update_multi(batch, rewards, config)

    def _update_multi(
        self,
        batch: list[RLRecord],
        rewards: list[dict[str, float]],
        config: GRPOConfig,
    ) -> TrainResult:
        names = sorted({name for rw in rewards for name in rw})
        weights = self.objective_weights or {name: 1.0 for name in names}
        specs = make_specs(weights)
        groups = [r.group_id or r.task_id for r in batch]
        vec = multi_group_relative_advantage(rewards, groups, specs, config.epsilon)

        scalarized = [vec[i]["scalarized"] for i in range(len(batch))]
        ratios = [1.0] * len(batch)
        objective = clipped_objective(scalarized, ratios, config)

        fronts = non_dominated_sort(rewards, specs)
        detail = {
            "batch_size": self.batch_size,
            "objectives": names,
            "objective_weights": weights,
            "per_objective_advantages": {
                name: [vec[i][name] for i in range(len(batch))] for name in names
            },
            "pareto_fronts": len(fronts),
            "non_dominated": len(fronts[0]) if fronts else 0,
            "mean_objectives": {
                name: _mean([rw.get(name, 0.0) for rw in rewards]) for name in names
            },
        }
        logger.info(
            "multi-objective GRPO: objectives=%s fronts=%d non_dominated=%d objective=%.4f",
            detail["mean_objectives"],
            detail["pareto_fronts"],
            detail["non_dominated"],
            objective,
        )
        return TrainResult(
            records=len(batch),
            advantages=scalarized,
            objective=objective,
            updated=False,
            detail=detail,
        )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0