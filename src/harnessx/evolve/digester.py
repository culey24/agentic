"""Digester: compresses raw trajectories into structured per-task digests.

The paper's Digester compresses ~10M raw tokens per GAIA iteration into ~10K of
structured summaries (binary outcome, failure category, implicated component
IDs, evidence excerpts). The default implementation here is deterministic; an
LLM-backed digester can be supplied for richer summarization.
"""

from __future__ import annotations

import logging
from typing import Any

from harnessx.core.trajectory import Trajectory
from harnessx.evolve.llm import call_json
from harnessx.evolve.types import Digest

logger = logging.getLogger("harnessx.evolve")

_CATEGORY_COMPONENTS: dict[str, list[str]] = {
    "empty_output": ["D2.context", "D6.evaluation"],
    "tool_error": ["D4.tools"],
    "step_limit": ["D7.control", "D2.context"],
    "incorrect_answer": ["D2.context", "D4.tools"],
}


class Digester:
    def __init__(self, provider: Any = None) -> None:
        self.provider = provider

    async def digest(
        self,
        trajectories: dict[str, Trajectory],
        outcomes: dict[str, bool],
    ) -> list[Digest]:
        if self.provider is None:
            return [self._heuristic(tid, traj, outcomes[tid]) for tid, traj in trajectories.items()]
        try:
            return await self._llm_digest(trajectories, outcomes)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM digester failed (%s); falling back to heuristic", exc)
            return [self._heuristic(tid, traj, outcomes[tid]) for tid, traj in trajectories.items()]

    def _heuristic(
        self, task_id: str, traj: Trajectory, success: bool
    ) -> Digest:
        if success:
            return Digest(task_id=task_id, success=True, summary="solved")
        category = "incorrect_answer"
        if not traj.final_output:
            category = "empty_output"
        elif any(
            isinstance(r, dict) and "error" in r
            for s in traj.steps
            for r in s.tool_results
        ):
            category = "tool_error"
        if traj.metadata.get("max_steps_hit"):
            category = "step_limit"
        components = _CATEGORY_COMPONENTS.get(category, ["D2.context"])
        evidence = traj.final_output if isinstance(traj.final_output, str) else ""
        return Digest(
            task_id=task_id,
            success=False,
            failure_category=category,
            implicated_components=components,
            evidence=evidence[:500],
            summary=f"failed ({category})",
        )

    async def _llm_digest(
        self,
        trajectories: dict[str, Trajectory],
        outcomes: dict[str, bool],
    ) -> list[Digest]:
        payload = [
            {
                "task_id": tid,
                "success": outcomes[tid],
                "final_output": traj.final_output,
                "steps": len(traj.steps),
            }
            for tid, traj in trajectories.items()
        ]
        system = (
            "You are the Digester stage of an agent-harness evolution engine. "
            "Compress the provided trajectories into per-task structured summaries. "
            "Respond with JSON: {\"digests\": [{\"task_id\": str, \"success\": bool, "
            "\"failure_category\": str|null, \"implicated_components\": [str], "
            "\"evidence\": str, \"summary\": str}]}. "
            "failure_category must be one of: empty_output, tool_error, step_limit, "
            "incorrect_answer."
        )
        data = await call_json(
            self.provider, system, "Trajectories:\n" + _json(payload)
        )
        return [Digest(**_normalize_digest(d)) for d in data["digests"]]


def _normalize_digest(d: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": d["task_id"],
        "success": bool(d.get("success", False)),
        "failure_category": d.get("failure_category"),
        "implicated_components": d.get("implicated_components") or [],
        "evidence": d.get("evidence") or "",
        "summary": d.get("summary") or "",
    }


def _json(obj: Any) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False, default=str)
