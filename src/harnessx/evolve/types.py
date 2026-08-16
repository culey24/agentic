"""Shared data types for the AEGIS pipeline."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Digest:
    task_id: str
    success: bool
    failure_category: str | None = None
    implicated_components: list[str] = field(default_factory=list)
    evidence: str = ""
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "success": self.success,
            "failure_category": self.failure_category,
            "implicated_components": self.implicated_components,
            "evidence": self.evidence,
            "summary": self.summary,
        }


@dataclass
class Landscape:
    round: int
    failing_tasks: list[str] = field(default_factory=list)
    implicated_components: list[str] = field(default_factory=list)
    prior_edits: list[str] = field(default_factory=list)
    untried_edit_types: list[str] = field(default_factory=list)
    directions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round,
            "failing_tasks": self.failing_tasks,
            "implicated_components": self.implicated_components,
            "prior_edits": self.prior_edits,
            "untried_edit_types": self.untried_edit_types,
            "directions": self.directions,
        }


class VerdictAction(str, enum.Enum):
    SHIP = "ship"
    NO_OP = "no_op"
    REVISE = "revise"


@dataclass
class Verdict:
    action: VerdictAction
    ranking: int = 0
    reason: str = ""
    revision_request: str = ""
