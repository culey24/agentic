"""Model trainer interface and harness-model co-evolution glue (Section 5).

Co-evolution interleaves harness evolution (non-parametric) with parametric
model updates over a shared mixed-policy replay buffer. Real GRPO training
needs open-weight checkpoints and GPUs (e.g. VERL); this module provides the
pluggable ``ModelTrainer`` seam plus two implementations:

- :class:`CollectOnlyTrainer` — no model update; only accumulates records. Used
  for API-only runs (the model cannot be fine-tuned through the API).
- :class:`GRPOTrainer` — computes group-relative advantages and the clipped
  GRPO objective (from :mod:`harnessx.rl.grpo`) over a sampled batch. Without
  token log-probabilities it reports statistics only; swap in a VERL-backed
  trainer to actually apply gradients.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from harnessx.rl.bridge import RLRecord
from harnessx.rl.grpo import GRPOConfig, clipped_objective, group_relative_advantage

logger = logging.getLogger("harnessx.rl")


@dataclass
class TrainResult:
    records: int
    advantages: list[float] = field(default_factory=list)
    objective: float | None = None
    updated: bool = False
    detail: dict[str, Any] = field(default_factory=dict)


_DEFAULT_GRPO = GRPOConfig()


class ModelTrainer(Protocol):
    async def update(self, records: list[RLRecord], config: GRPOConfig | None = None) -> TrainResult: ...


class CollectOnlyTrainer:
    """Accumulates records without performing any parametric update."""

    def __init__(self, log: bool = True) -> None:
        self.log = log

    async def update(self, records: list[RLRecord], config: GRPOConfig | None = None) -> TrainResult:
        if self.log and records:
            rewards = [r.reward for r in records]
            logger.info("collect-only: %d records, mean reward=%.3f", len(records), sum(rewards) / len(rewards))
        return TrainResult(records=len(records), updated=False)


class GRPOTrainer:
    """Computes GRPO training signal over a batch (reporting, not updating).

    Without cached token log-probabilities from the generating checkpoint, the
    importance ratios are unavailable, so this trainer reports advantage /
    objective statistics. A VERL-backed trainer can replace it.
    """

    def __init__(self, batch_size: int = 256) -> None:
        self.batch_size = batch_size

    async def update(self, records: list[RLRecord], config: GRPOConfig | None = None) -> TrainResult:
        config = config or _DEFAULT_GRPO
        batch = records[: self.batch_size]
        if not batch:
            return TrainResult(records=0)
        rewards = [r.reward for r in batch]
        groups = [r.group_id or r.task_id for r in batch]
        advantages = group_relative_advantage(rewards, groups, config.epsilon)
        ratios = [1.0] * len(batch)
        objective = clipped_objective([advantages[i] for i in range(len(batch))], ratios, config)
        return TrainResult(
            records=len(batch),
            advantages=[advantages[i] for i in range(len(batch))],
            objective=objective,
            updated=False,
            detail={"batch_size": self.batch_size},
        )
