"""Training bridge (D9): convert execution trajectories into RL records.

Co-evolution groups trajectories by task identity across harness versions and
computes group-relative advantages. This module emits records that a GRPO
trainer (e.g. VERL) can consume, without requiring token log-probabilities
(which are unavailable from closed API providers and left optional).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from harnessx.core.trajectory import Trajectory


@dataclass
class RLRecord:
    task_id: str
    completion: str
    reward: float
    group_id: str | None = None
    harness_version: str | None = None
    log_probs: list[float] | None = None
    tokens: list[int] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TrajectoryBridge:
    """Converts a trajectory into a reward-annotated RL record.

    ``group_id`` is the task identity used by cross-harness GRPO grouping
    (Section 5.3): trajectories from the same task across successive harness
    versions share a group.
    """

    def to_record(
        self,
        traj: Trajectory,
        harness_version: str | None = None,
        log_probs: list[float] | None = None,
        tokens: list[int] | None = None,
    ) -> RLRecord:
        completion = (
            traj.final_output
            if isinstance(traj.final_output, str)
            else _stringify(traj.final_output)
        )
        return RLRecord(
            task_id=traj.task_id,
            completion=completion,
            reward=float(traj.reward or 0.0),
            group_id=traj.task_id,
            harness_version=harness_version,
            log_probs=log_probs,
            tokens=tokens,
        )


def _stringify(value: Any) -> str:
    import json

    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)
