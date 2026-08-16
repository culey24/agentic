"""Evolver: produces typed builder-edit candidates with change manifests.

Candidates reference only known processor kinds and hooks. The Evolver is
type-safe but not behavior-safe; the Critic and deterministic gate decide what
ships. Default heuristic emits simple structural edits; an LLM-backed evolver
proposes richer combinations.
"""

from __future__ import annotations

import logging
from typing import Any

from harnessx.core.harness_config import HarnessConfig
from harnessx.core.hooks import Hook
from harnessx.evolve.llm import call_json
from harnessx.evolve.manifest import Candidate, ChangeManifest, Edit, EditOp
from harnessx.evolve.types import Landscape
from harnessx.processors.registry import kinds as all_kinds

logger = logging.getLogger("harnessx.evolve")

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
        try:
            return await self._llm_evolve(landscape, base_config)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM evolver failed (%s); falling back to heuristic", exc)
            return self._heuristic(landscape)

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
            "settable_paths": ["max_steps"],
        }
        system = (
            "You are the Evolver stage of an agent-harness evolution engine. "
            "Produce K typed harness edits as change manifests. Respond with JSON: "
            "{\"candidates\": [{\"id\": str, \"edited_components\": [str], "
            "\"intended_effect\": str, \"expected_improve\": [str], "
            "\"expected_regress\": [str], \"edits\": [{\"op\": \"insert|replace|"
            "remove|set\", \"hook\": str|null, \"group\": str|null, \"kind\": "
            "str|null, \"params\": {}, \"path\": str|null, \"value\": null}]}]}.\n"
            "Schema rules (violations make a candidate unusable):\n"
            "- insert: requires hook AND kind; group optional.\n"
            "- replace: requires hook AND kind; use group to target an existing "
            "processor.\n"
            "- remove: requires hook AND group (e.g. system_prompt, history_trim, "
            "tool_approval, reward_annotate).\n"
            "- set: requires path from settable_paths, otherwise omit.\n"
            "- Do NOT invent kinds or hooks; only use the provided surface.\n"
            "- Prefer small, targeted edits that directly address the failing "
            "tasks."
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
            raw_edits = c.get("edits", [])
            edits = [_edit_from_dict(e) for e in raw_edits if _valid_edit(e)]
            if not edits:
                continue
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


def _valid_edit(d: dict[str, Any]) -> bool:
    try:
        op = EditOp(d.get("op"))
    except ValueError:
        return False
    if op == EditOp.SET:
        return bool(d.get("path")) and d.get("path") in ("max_steps",)
    if op == EditOp.REMOVE:
        return bool(d.get("hook")) and bool(d.get("group"))
    if op in (EditOp.INSERT, EditOp.REPLACE):
        return bool(d.get("hook")) and bool(d.get("kind"))
    return False


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
