"""Serializable trajectory records used by observability, AEGIS, and RL.

Each step captures the messages, model response, tool calls/results, and any
per-step reward. Trajectories are the raw material for both the Digester
(trace compression) and the training bridge (D9).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class StepRecord:
    step: int
    messages: list[dict[str, Any]] = field(default_factory=list)
    response: dict[str, Any] | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    reward: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    task_id: str
    steps: list[StepRecord] = field(default_factory=list)
    final_output: Any = None
    success: bool | None = None
    reward: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
