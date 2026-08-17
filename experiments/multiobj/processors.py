"""Observation-improving processors, registered via ``harnessx.processors.registry``.

These attach to the standard ``RunLoop`` pipeline (used by gaia / webshop /
swebench-style benchmarks) and enrich what ends up in the trajectory trace —
the raw material of ``MultiObjectiveBridge`` observations.

- ``tool_summarizer`` (AFTER_TOOL): truncates oversized tool results so the
  trace stays compact and signal-dense.
- ``observation_capture`` (BEFORE_MODEL): appends a compact observation note
  built from recent tool results, so the model (and the trace) see a distilled
  summary of what happened since the last step.

Both kinds are registered through the existing ``register`` seam and exposed to
the AEGIS Evolver via :class:`MultiObjectiveEvolver`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, ClassVar

from harnessx.core.hooks import Hook
from harnessx.core.processor import Order, Processor
from harnessx.events import (
    BeforeModelEvent,
    Event,
    Message,
    MessageRole,
    ToolResultEvent,
)
from harnessx.evolve.evolver import (
    KIND_SPECS,
    Evolver,
    _edit_from_dict,
    _json,
    _valid_edit,
)
from harnessx.evolve.llm import call_json
from harnessx.evolve.manifest import Candidate, ChangeManifest
from harnessx.processors.registry import register

_HOOKS = list(Hook)


@register("tool_summarizer")
class ToolSummarizerProcessor(Processor):
    _singleton_group = "tool_summarizer"
    _order = Order.NORMAL

    def __init__(self, max_chars: int = 400) -> None:
        self.max_chars = max_chars

    async def process(self, event: Event) -> AsyncIterator[Event]:
        if isinstance(event, ToolResultEvent):
            event.result = _condense(event.result, self.max_chars)
        yield event


@register("observation_capture")
class ObservationCaptureProcessor(Processor):
    _singleton_group = "observation_capture"
    _order = Order.POST

    def __init__(self, max_notes: int = 3) -> None:
        self.max_notes = max_notes

    async def process(self, event: Event) -> AsyncIterator[Event]:
        if isinstance(event, BeforeModelEvent):
            notes = _recent_tool_notes(event.messages, self.max_notes)
            if notes:
                event.messages.append(
                    Message(
                        role=MessageRole.SYSTEM,
                        content="Observation notes:\n" + "\n".join(notes),
                    )
                )
        yield event


def _condense(value: Any, max_chars: int) -> Any:
    if isinstance(value, str):
        return value[:max_chars] + ("…" if len(value) > max_chars else "")
    if isinstance(value, dict):
        return {k: _condense(v, max_chars) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_condense(v, max_chars) for v in value]
    return value


def _recent_tool_notes(messages: list[Message], max_notes: int) -> list[str]:
    notes: list[str] = []
    for msg in messages:
        if msg.role == MessageRole.TOOL and msg.content:
            text = msg.content if isinstance(msg.content, str) else str(msg.content)
            text = text.strip()
            if not text:
                continue
            if len(text) > 120:
                text = text[:120] + "…"
            notes.append(f"- {msg.name or 'tool'}: {text}")
    return notes[-max_notes:]


class MultiObjectiveEvolver(Evolver):
    """Evolver that also proposes the observation-improving processor kinds."""

    EXTRA_KINDS: ClassVar[dict[str, dict[str, Any]]] = {
        "tool_summarizer": {"params": {"max_chars": "int"}},
        "observation_capture": {"params": {"max_notes": "int"}},
    }

    async def _llm_evolve(self, landscape: Any, base_config: Any) -> list[Candidate]:
        kinds = dict(KIND_SPECS)
        kinds.update(self.EXTRA_KINDS)
        surface = {
            "hooks": [h.value for h in _HOOKS],
            "kinds": kinds,
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
            "- remove: requires hook AND group.\n"
            "- set: requires path from settable_paths, otherwise omit.\n"
            "- Do NOT invent kinds or hooks; only use the provided surface.\n"
            "- Prefer small, targeted edits that directly address the failing tasks."
        )
        user = (
            "Landscape:\n"
            + _json(landscape.to_dict())
            + "\nEdit surface:\n"
            + _json(surface)
        )
        data = await call_json(self.provider, system, user)
        return _build_candidates(self, data, landscape)


def _build_candidates(evolver: Evolver, data: dict[str, Any], landscape: Any) -> list[Candidate]:
    candidates: list[Candidate] = []
    for i, c in enumerate(data.get("candidates", [])):
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