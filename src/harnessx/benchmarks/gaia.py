"""GAIA benchmark adapter.

GAIA (Level 1-3) is a multi-step retrieval benchmark with exact-match
verification. The paper samples 103 tasks across three levels (39/52/12).
Tasks are loaded from a JSONL file; each record has a question, a level, and a
ground-truth final answer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from harnessx.benchmarks.base import Task, exact_match
from harnessx.tools import web
from harnessx.tools.registry import ToolRegistry


class GAIATask(Task):
    def __init__(
        self,
        task_id: str,
        question: str,
        level: int,
        final_answer: str,
        file_name: str | None = None,
    ) -> None:
        super().__init__(
            id=task_id,
            description=question,
            ground_truth=final_answer,
            metadata={"level": level, "file_name": file_name},
        )
        self.question = question
        self.level = level
        self.final_answer = final_answer
        self.file_name = file_name


class GAIAAdapter:
    name = "gaia"

    def __init__(self, data_path: str | Path | None = None) -> None:
        self.data_path = Path(data_path) if data_path else None

    def load_tasks(self) -> list[GAIATask]:
        if self.data_path is None:
            raise ValueError("GAIA requires a data_path to a JSONL file")
        tasks: list[GAIATask] = []
        with open(self.data_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                tasks.append(
                    GAIATask(
                        task_id=rec.get("task_id") or rec.get("id", ""),
                        question=rec["Question"],
                        level=int(rec.get("Level", 1)),
                        final_answer=rec.get("Final answer")
                        or rec.get("final_answer", ""),
                        file_name=rec.get("file_name"),
                    )
                )
        return tasks

    def verifier(self) -> Any:
        return exact_match

    def default_tools(self) -> ToolRegistry | None:
        registry = ToolRegistry()
        registry.register(
            "browse",
            web.browse,
            description="Fetch a URL and return its text content.",
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"},
                },
                "required": ["url"],
            },
        )
        registry.register(
            "search",
            web.search,
            description="Search the web for a query.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        )
        return registry

    def default_harness_config(self) -> Any:
        from harnessx.core.harness_config import HarnessConfig
        from harnessx.core.hooks import Hook
        from harnessx.processors.context import SystemPromptProcessor

        config = HarnessConfig(tool_registry=self.default_tools(), max_steps=20)
        config.add(
            Hook.TASK_START,
            SystemPromptProcessor(
                prompt=(
                    "You are a helpful research agent. Solve the task step by "
                    "step. Use the provided tools to gather information when "
                    "needed. When finished, output only the final answer."
                )
            ),
        )
        return config
