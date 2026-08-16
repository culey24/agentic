"""Change manifests and typed harness edits.

The Evolver produces builder-edit candidates, each carrying a change manifest
that records the edited components, the intended effect, and the tasks expected
to improve or regress. The Critic compares the manifest against trace evidence;
the deterministic gate decides what ships.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class EditOp(str, enum.Enum):
    INSERT = "insert"
    REPLACE = "replace"
    REMOVE = "remove"
    SET = "set"


@dataclass
class Edit:
    op: EditOp
    hook: str | None = None
    group: str | None = None
    kind: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    path: str | None = None
    value: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op.value,
            "hook": self.hook,
            "group": self.group,
            "kind": self.kind,
            "params": self.params,
            "path": self.path,
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Edit:
        return cls(
            op=EditOp(data["op"]),
            hook=data.get("hook"),
            group=data.get("group"),
            kind=data.get("kind"),
            params=data.get("params") or {},
            path=data.get("path"),
            value=data.get("value"),
        )


@dataclass
class ChangeManifest:
    id: str
    edits: list[Edit] = field(default_factory=list)
    edited_components: list[str] = field(default_factory=list)
    intended_effect: str = ""
    expected_improve: list[str] = field(default_factory=list)
    expected_regress: list[str] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "edits": [e.to_dict() for e in self.edits],
            "edited_components": self.edited_components,
            "intended_effect": self.intended_effect,
            "expected_improve": self.expected_improve,
            "expected_regress": self.expected_regress,
            "rationale": self.rationale,
        }


@dataclass
class Candidate:
    manifest: ChangeManifest
    round: int
    number: int
    source: str = "evolver"

    @property
    def name(self) -> str:
        return f"C-R{self.round}-{self.number:02d}"
