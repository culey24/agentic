"""Benchmark adapter protocol and shared task/verifier types."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol

from harnessx.tools.registry import ToolRegistry


@dataclass
class Task:
    id: str
    description: str
    ground_truth: Any = None
    metadata: dict[str, Any] | None = None


class Verifier(Protocol):
    async def __call__(self, task: Any, final_output: Any) -> bool: ...


class BenchmarkAdapter(Protocol):
    name: str

    def load_tasks(self) -> list[Any]: ...

    def verifier(self) -> Verifier: ...

    def default_tools(self) -> ToolRegistry | None: ...


def normalize_answer(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


async def exact_match(task: Any, final_output: Any) -> bool:
    if final_output is None:
        return False
    target = task.ground_truth
    if target is None:
        return False
    return normalize_answer(str(target)) in normalize_answer(str(final_output))
