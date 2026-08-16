"""Deterministic acceptance gate (Section 4.3).

The gate defends against catastrophic forgetting and malformed edits. It checks,
in order: manifest completeness, config normalization, build/smoke tests, and a
seesaw (regression) constraint on previously-passing tasks. The first failure
halts the round. LLM judgment is decoupled from acceptance: only these
deterministic checks govern what ships.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from harnessx.core.harness_config import HarnessConfig
from harnessx.evolve.builder import Builder
from harnessx.evolve.manifest import ChangeManifest


@dataclass
class GateResult:
    passed: bool
    stage: str
    reason: str = ""
    config: HarnessConfig | None = None
    detail: dict[str, Any] = field(default_factory=dict)


SeesawFn = Callable[[HarnessConfig], Any]


class Gate:
    def __init__(self, builder: Builder | None = None) -> None:
        self.builder = builder or Builder()

    async def check(
        self,
        manifest: ChangeManifest,
        base_config: HarnessConfig,
        seesaw: SeesawFn | None = None,
    ) -> GateResult:
        if not manifest.edits:
            return GateResult(False, "manifest", "no edits in manifest")
        if not manifest.intended_effect:
            return GateResult(False, "manifest", "missing intended_effect")

        try:
            config = self.builder.apply_many(base_config, manifest.edits)
        except Exception as exc:  # noqa: BLE001
            return GateResult(False, "build", f"edit application failed: {exc}")

        norm_err = self._validate_normalized(config)
        if norm_err:
            return GateResult(False, "normalize", norm_err)

        if seesaw is not None:
            result = seesaw(config)
            if inspect.isawaitable(result):
                result = await result
            ok, msg = result
            if not ok:
                return GateResult(False, "seesaw", msg)

        return GateResult(True, "gate", "accepted", config=config)

    @staticmethod
    def _validate_normalized(config: HarnessConfig) -> str | None:
        for hook, procs in config.processors.items():
            groups = [p._singleton_group for p in procs if p._singleton_group]
            if len(groups) != len(set(groups)):
                return f"duplicate singleton groups at {hook.value}: {groups}"
        return None
