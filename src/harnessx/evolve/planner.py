"""Planner: builds the adaptation landscape (Section 4.3).

The Planner aggregates failing tasks, prior edits, implicated components, and
untried edit types into a landscape that guides candidate generation. It is the
primary defense against under-exploration.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from harnessx.evolve.llm import call_json
from harnessx.evolve.types import Digest, Landscape


class Planner:
    def __init__(self, provider: Any = None) -> None:
        self.provider = provider

    async def plan(
        self,
        round_: int,
        digests: list[Digest],
        prior_edits: list[str] | None = None,
    ) -> Landscape:
        prior_edits = prior_edits or []
        if self.provider is None:
            return self._heuristic(round_, digests, prior_edits)
        return await self._llm_plan(round_, digests, prior_edits)

    def _heuristic(
        self, round_: int, digests: list[Digest], prior_edits: list[str]
    ) -> Landscape:
        failing = [d.task_id for d in digests if not d.success]
        components = Counter(
            c for d in digests if not d.success for c in d.implicated_components
        )
        untried = ["D2.context", "D4.tools", "D7.control", "D6.evaluation"]
        directions = [
            f"Address {comp} for {count} failing tasks"
            for comp, count in components.most_common()
        ]
        return Landscape(
            round=round_,
            failing_tasks=failing,
            implicated_components=[c for c, _ in components.most_common()],
            prior_edits=prior_edits,
            untried_edit_types=untried,
            directions=directions,
        )

    async def _llm_plan(
        self, round_: int, digests: list[Digest], prior_edits: list[str]
    ) -> Landscape:
        system = (
            "You are the Planner stage of an agent-harness evolution engine. "
            "Given failure digests and prior edits, produce an adaptation landscape. "
            "Respond with JSON matching the Landscape schema: "
            "{\"failing_tasks\": [str], \"implicated_components\": [str], "
            "\"untried_edit_types\": [str], \"directions\": [str]}."
        )
        user = _digests_json(digests) + "\nPrior edits:\n" + _json(prior_edits)
        data = await call_json(self.provider, system, user)
        return Landscape(
            round=round_,
            failing_tasks=data.get("failing_tasks", []),
            implicated_components=data.get("implicated_components", []),
            prior_edits=prior_edits,
            untried_edit_types=data.get("untried_edit_types", []),
            directions=data.get("directions", []),
        )


def _digests_json(digests: list[Digest]) -> str:
    return _json([d.to_dict() for d in digests])


def _json(obj: Any) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False, default=str)
