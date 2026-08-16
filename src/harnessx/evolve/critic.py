"""Critic: compares change manifests against trace evidence (Section 4.3).

Defends against reward hacking: a candidate must be justified by observed
failures. The Critic returns a verdict (ship / no_op / revise with a single
revision request). Its judgment is advisory; only the deterministic gate governs
acceptance.
"""

from __future__ import annotations

import logging
from typing import Any

from harnessx.evolve.llm import call_json
from harnessx.evolve.manifest import Candidate
from harnessx.evolve.types import Digest, Verdict, VerdictAction

logger = logging.getLogger("harnessx.evolve")


class Critic:
    def __init__(self, provider: Any = None) -> None:
        self.provider = provider

    async def critique(
        self, candidate: Candidate, digests: list[Digest]
    ) -> Verdict:
        if self.provider is None:
            return self._heuristic(candidate, digests)
        try:
            return await self._llm_critique(candidate, digests)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM critic failed (%s); falling back to heuristic", exc)
            return self._heuristic(candidate, digests)

    def _heuristic(self, candidate: Candidate, digests: list[Digest]) -> Verdict:
        manifest = candidate.manifest
        failing = {d.task_id for d in digests if not d.success}
        if not manifest.edits:
            return Verdict(action=VerdictAction.NO_OP, reason="empty manifest")
        if manifest.expected_improve and not (set(manifest.expected_improve) & failing):
            return Verdict(
                action=VerdictAction.NO_OP,
                reason="expected_improve tasks are not among observed failures",
            )
        return Verdict(action=VerdictAction.SHIP, reason="manifest aligns with evidence")

    async def _llm_critique(
        self, candidate: Candidate, digests: list[Digest]
    ) -> Verdict:
        system = (
            "You are the Critic stage of an agent-harness evolution engine. "
            "Compare the candidate change manifest against the trace evidence and "
            "respond with JSON: {\"action\": \"ship|no_op|revise\", \"ranking\": int, "
            "\"reason\": str, \"revision_request\": str}."
        )
        user = (
            "Candidate:\n"
            + _json(candidate.manifest.to_dict())
            + "\nDigests:\n"
            + _json([d.to_dict() for d in digests])
        )
        data = await call_json(self.provider, system, user)
        return Verdict(
            action=VerdictAction(data.get("action", "no_op")),
            ranking=int(data.get("ranking", 0)),
            reason=data.get("reason", ""),
            revision_request=data.get("revision_request", ""),
        )


def _json(obj: Any) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False, default=str)
