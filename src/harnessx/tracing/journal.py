"""Run journal: first-person memo, curves, and audit events (Section 13).

Each evolution run writes a self-describing directory ``output/runs/<name>/``
with a journal, per-round pass-rate curves, and an audit log of every
stage/gate/commit decision.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Journal:
    name: str
    root: Path | None = None
    entries: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.root is None:
            self.root = Path("output") / "runs" / self.name
        self.root.mkdir(parents=True, exist_ok=True)

    def log(self, round_: int, memo: str, **extra: Any) -> None:
        entry = {"round": round_, "memo": memo, **extra}
        self.entries.append(entry)
        with open(self.root / "journal.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")

    def record_curve(self, round_: int, pass_rate: float, **extra: Any) -> None:
        curves = self._load_json("curves.json", [])
        curves.append({"round": round_, "pass_rate": pass_rate, **extra})
        self._dump_json("curves.json", curves)

    def audit(self, stage: str, event: str, **extra: Any) -> None:
        with open(self.root / "audit.jsonl", "a") as f:
            f.write(json.dumps({"stage": stage, "event": event, **extra}) + "\n")

    def _load_json(self, filename: str, default: Any) -> Any:
        path = self.root / filename
        if not path.exists():
            return default
        return json.loads(path.read_text())

    def _dump_json(self, filename: str, data: Any) -> None:
        (self.root / filename).write_text(json.dumps(data, indent=2))
