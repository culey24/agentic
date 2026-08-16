"""Builder: applies typed edits to a ``HarnessConfig`` via a substitution algebra.

Supports insert / replace / remove of processors (keyed by singleton group) and
``set`` of config fields. Every operation is type-safe: a substitution cannot
violate the hook's event-type contract or the singleton-group mutual-exclusion
invariant.
"""

from __future__ import annotations

import copy

from harnessx.core.harness_config import HarnessConfig
from harnessx.core.hooks import Hook
from harnessx.evolve.manifest import Edit, EditOp
from harnessx.processors.registry import create


class Builder:
    def apply(self, config: HarnessConfig, edit: Edit) -> HarnessConfig:
        if edit.op == EditOp.INSERT:
            return self._insert(config, edit)
        if edit.op == EditOp.REPLACE:
            return self._replace(config, edit)
        if edit.op == EditOp.REMOVE:
            return self._remove(config, edit)
        if edit.op == EditOp.SET:
            return self._set(config, edit)
        raise ValueError(f"unsupported edit op {edit.op}")

    def apply_many(self, config: HarnessConfig, edits: list[Edit]) -> HarnessConfig:
        result = copy.deepcopy(config)
        for edit in edits:
            result = self.apply(result, edit)
        return result

    def _insert(self, config: HarnessConfig, edit: Edit) -> HarnessConfig:
        hook = Hook(edit.hook)
        processor = create(edit.kind or "", edit.params)
        config.add(hook, processor)
        return config

    def _replace(self, config: HarnessConfig, edit: Edit) -> HarnessConfig:
        hook = Hook(edit.hook)
        processor = create(edit.kind or "", edit.params)
        if edit.group:
            config.remove_group(hook, edit.group)
            config.add(hook, processor)
        else:
            config.replace_group(hook, processor)
        return config

    def _remove(self, config: HarnessConfig, edit: Edit) -> HarnessConfig:
        hook = Hook(edit.hook)
        config.remove_group(hook, edit.group or "")
        return config

    def _set(self, config: HarnessConfig, edit: Edit) -> HarnessConfig:
        if edit.path is None:
            raise ValueError("SET edit requires a path")
        obj: object = config
        segments = edit.path.split(".")
        for seg in segments[:-1]:
            obj = getattr(obj, seg)
        setattr(obj, segments[-1], edit.value)
        return config
