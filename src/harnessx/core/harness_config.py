"""Harness configuration: the full behavior pipeline, independent of model.

``HarnessConfig = (P, S)`` where ``P`` maps each hook to a list of processors
and ``S`` holds orthogonal slot resources (tool registry, tracer, workspace,
sandbox, plugins). Slots are singletons shared across processors; processor
state is instance-private.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from harnessx.core.hooks import Hook
from harnessx.core.processor import Order, Processor


@dataclass
class HarnessConfig:
    processors: dict[Hook, list[Processor]] = field(default_factory=dict)
    tool_registry: Any = None
    tracer: Any = None
    workspace: Any = None
    sandbox: Any = None
    plugins: list[Any] = field(default_factory=list)
    max_steps: int = 20

    def add(self, hook: Hook, processor: Processor) -> HarnessConfig:
        group = processor._singleton_group
        if group is not None:
            for existing in self.processors.setdefault(hook, []):
                if existing._singleton_group == group:
                    raise ValueError(
                        f"singleton group {group!r} already present at {hook.value}"
                    )
        self.processors.setdefault(hook, []).append(processor)
        return self

    def processors_for(self, hook: Hook) -> list[Processor]:
        procs = list(self.processors.get(hook, []))
        procs.sort(key=lambda p: (p._order, p._order is Order.PRE))
        return procs

    def remove_group(self, hook: Hook, group: str) -> bool:
        procs = self.processors.get(hook, [])
        kept = [p for p in procs if p._singleton_group != group]
        removed = len(kept) != len(procs)
        if kept:
            self.processors[hook] = kept
        else:
            self.processors.pop(hook, None)
        return removed

    def replace_group(self, hook: Hook, processor: Processor) -> HarnessConfig:
        self.remove_group(hook, processor._singleton_group)
        self.add(hook, processor)
        return self
