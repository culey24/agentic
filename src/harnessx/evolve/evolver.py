"""Evolver: produces typed builder-edit candidates with change manifests.

Candidates reference only known processor kinds and hooks. The Evolver is
type-safe but not behavior-safe; the Critic and deterministic gate decide what
ships. Default heuristic emits simple structural edits; an LLM-backed evolver
proposes richer combinations.
"""

from __future__ import annotations

from typing import Any

from harnessx.core.harness_config import HarnessConfig
from harnessx.core.hooks import Hook
from harnessx.evolve.llm import call_json
from harnessx.evolve.manifest import Candidate, ChangeManifest, Edit, EditOp
from harnessx.evolve.types import Landscape
from harnessx.processors.registry import kinds as all_kinds

HOOKS = [h.value for h in Hook]

KIND_SPECS: dict[str, dict[str, Any]] = {
    "system_prompt": {"params": {"prompt": "str"}},
    "history_trim": {"params": {"max_messages": "int"}},
    "tool_approval": {"params": {"allowlist": "list[str]"}},
    "reward_annotate": {"params": {"reward": "float"}},
}

_GROUP_BY_KIND = {
    "system_prompt": "system_prompt",
    "history_trim": "history_trim",
    "tool_approval": "tool_approval",
    "reward_annotate": "reward_annotate",
}


class Evolver:
    def __init__(self, provider: Any = None, candidates_per_round: int = 4) -> None:
        self.provider = provider
        self.K = candidates_per_round

    async def evolve(
        self,
        landscape: Landscape,
        base_config: HarnessConfig,
    ) -> list[Candidate]:
        if self.provider is None:
            return self._heuristic(landscape)
        return await self._llm_evolve(landscape, base_config)

    def _heuristic(self, landscape: Landscape) -> list[Candidate]:
        edits_by_comp = {
            "D2.context": Edit(
                op=EditOp.INSERT,
                hook="task_start",
                group="system_prompt",
                kind="system_prompt",
                params={"prompt": "Think step by step and answer concisely."},
            ),
            "D4.tools": Edit(
                op=EditOp.INSERT,
                hook="before_tool",
                group="tool_approval",
                kind="tool_approval",
                params={"allowlist": []},
            ),
            "D7.control": Edit(
                op=EditOp.INSERT,
                hook="before_model",
                group="history_trim",
                kind="history_trim",
                params={"max_messages": 20},
            ),
        }
        candidates: list[Candidate] = []
        components = landscape.implicated_components or list(edits_by_comp)
        for i in range(self.K):
            comp = components[i % len(components)]
            edit = edits_by_comp.get(comp)
            if edit is None:
                continue
            manifest = ChangeManifest(
                id=f"C-R{landscape.round}-{i:02d}",
                edits=[edit],
                edited_components=[comp],
                intended_effect=f"improve {comp}",
                expected_improve=landscape.failing_tasks[:5],
            )
            candidates.append(
                Candidate(manifest=manifest, round=landscape.round, number=i)
            )
        return candidates

    async def _llm_evolve(
        self, landscape: Landscape, base_config: HarnessConfig
    ) -> list[Candidate]:
        surface = {
            "hooks": HOOKS,
            "kinds": {
                k: v for k, v in KIND_SPECS.items() if k in all_kinds()
            },
        }
        system = (
            "You are the Evolver stage of an agent-harness evolution engine. "
            "Produce K typed harness edits as change manifests. Respond with JSON: "
            "{\"candidates\": [{\"id\": str, \"edited_components\": [str], "
            "\"intended_effect\": str, \"expected_improve\": [str], "
            "\"expected_regress\": [str], \"edits\": [{\"op\": \"insert|replace|"
            "remove|set\", \"hook\": str|null, \"group\": str|null, \"kind\": "
            "str|null, \"params\": {}, \"path\": str|null, \"value\": null}]}]}. "
            "Only use hooks and kinds from the provided surface."
        )
        user = (
            "Landscape:\n"
            + _json(landscape.to_dict())
            + "\nEdit surface:\n"
            + _json(surface)
        )
        data = await call_json(self.provider, system, user)
        candidates = []
        for i, c in enumerate(data["candidates"]):
            edits = [_edit_from_dict(e) for e in c.get("edits", [])]
            manifest = ChangeManifest(
                id=c.get("id") or f"C-R{landscape.round}-{i:02d}",
                edits=edits,
                edited_components=c.get("edited_components", []),
                intended_effect=c.get("intended_effect", ""),
                expected_improve=c.get("expected_improve", []),
                expected_regress=c.get("expected_regress", []),
            )
            candidates.append(
                Candidate(manifest=manifest, round=landscape.round, number=i)
            )
        return candidates


def _edit_from_dict(d: dict[str, Any]) -> Edit:
    return Edit(
        op=EditOp(d["op"]),
        hook=d.get("hook"),
        group=d.get("group"),
        kind=d.get("kind"),
        params=d.get("params") or {},
        path=d.get("path"),
        value=d.get("value"),
    )


def _json(obj: Any) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False, default=str)
