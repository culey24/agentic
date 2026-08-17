"""Multi-objective reward scorer for τ³-Bench trajectories.

Reads the final database state from ``Trajectory.final_output`` (a
``DialogueResult`` produced by ``DialogueHarness``) and returns a reward vector:

- ``correctness`` — soft fraction of expected DB checks satisfied
  (binary 0/1 when ``binary=True``).
- ``efficiency`` — ``1 - turns / max_turns``.
- ``tool_safety`` — ``1 - min(1, tool_errors / max(1, tool_calls))``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from harnessx.benchmarks.tau3.retail import _key_field_for
from harnessx.core.trajectory import Trajectory

ObjectiveScorer = Callable[[Any, Trajectory], dict[str, float]]


def make_tau3_scorer(max_turns: int = 200, binary: bool = False) -> ObjectiveScorer:
    def score(task: Any, traj: Trajectory) -> dict[str, float]:
        final = traj.final_output
        db_state = _db_state(final)
        correctness = _correctness(task, db_state, binary=binary)
        turns = _turns(final) or 1
        efficiency = 1.0 - min(1.0, turns / max(1, max_turns))
        tool_calls, tool_errors = _tool_metrics(final, traj)
        tool_safety = 1.0 - min(1.0, tool_errors / max(1, tool_calls))
        return {
            "correctness": correctness,
            "efficiency": efficiency,
            "tool_safety": tool_safety,
        }

    return score


def _db_state(final: Any) -> dict[str, Any]:
    db = getattr(final, "db_state", None)
    if db is not None:
        return db
    if isinstance(final, dict) and isinstance(final.get("db_state"), dict):
        return final["db_state"]
    return {}


def _turns(final: Any) -> int:
    turns = getattr(final, "turns", None)
    if isinstance(turns, (int, float)):
        return int(turns)
    if isinstance(final, dict) and isinstance(final.get("turns"), (int, float)):
        return int(final["turns"])
    return 0


def _tool_metrics(final: Any, traj: Trajectory) -> tuple[int, int]:
    transcript = getattr(final, "transcript", None)
    if not isinstance(transcript, list) or not transcript:
        transcript = traj.steps
    tool_calls = 0
    tool_errors = 0
    for entry in transcript:
        if isinstance(entry, dict):
            if entry.get("role") == "tool":
                tool_calls += 1
                if _is_error_dict(entry.get("result")):
                    tool_errors += 1
            for r in entry.get("tool_results") or []:
                tool_calls += 1
                if isinstance(r, dict) and _is_error_dict(r.get("result")):
                    tool_errors += 1
        else:
            for r in getattr(entry, "tool_results", None) or []:
                tool_calls += 1
                if isinstance(r, dict) and _is_error_dict(r.get("result")):
                    tool_errors += 1
    return tool_calls, tool_errors


def _is_error_dict(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if "error" in value:
        return True
    result = value.get("result")
    return isinstance(result, dict) and "error" in result


def _correctness(task: Any, db_state: dict[str, Any], binary: bool = False) -> float:
    expected = getattr(task, "expected", None) or {}
    if not expected:
        return 0.0
    checks: list[tuple[bool, str]] = []
    for table, table_checks in expected.items():
        rows = db_state.get(table, [])
        for check in table_checks:
            key_field = check.get("key_field", _key_field_for(table))
            row = next((r for r in rows if r.get(key_field) == check["key"]), None)
            if row is None:
                checks.append((False, "row_missing"))
                continue
            for field, want in check.get("fields", {}).items():
                checks.append((row.get(field) == want, f"{field}:{want}"))
    if not checks:
        return 0.0
    fraction = sum(1 for ok, _ in checks if ok) / len(checks)
    return 1.0 if binary and fraction >= 1.0 else (0.0 if binary else fraction)