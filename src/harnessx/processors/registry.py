"""Processor registry: maps processor *kinds* (strings) to factories.

The AEGIS Evolver produces typed builder edits that reference a ``kind``; the
Builder resolves the kind through this registry to instantiate a concrete
processor. New processor kinds can be registered by third-party extensions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from harnessx.core.processor import Processor

_factories: dict[str, Callable[..., Processor]] = {}


def register(kind: str) -> Callable[[Callable[..., Processor]], Callable[..., Processor]]:
    def decorator(factory: Callable[..., Processor]) -> Callable[..., Processor]:
        _factories[kind] = factory
        return factory

    return decorator


def create(kind: str, params: dict[str, Any] | None = None) -> Processor:
    params = params or {}
    if kind not in _factories:
        raise KeyError(f"unknown processor kind {kind!r}")
    return _factories[kind](**params)


def kinds() -> list[str]:
    return sorted(_factories)
