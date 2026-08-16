"""τ³-Bench adapter: task loader, dialogue harness, and rule-compliance verifier."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harnessx.benchmarks.tau3.domain import Domain
from harnessx.benchmarks.tau3.retail import get_domain, verify_retail
from harnessx.benchmarks.tau3.runner import DialogueResult, DialogueRunner
from harnessx.core.harness import Harness
from harnessx.core.harness_config import HarnessConfig
from harnessx.core.hooks import Hook
from harnessx.core.trajectory import Trajectory
from harnessx.processors.context import SystemPromptProcessor


@dataclass
class Tau3Task:
    task_id: str
    domain: str
    opening: str
    script: list[dict[str, Any]] = field(default_factory=list)
    expected: dict[str, Any] = field(default_factory=dict)
    instruction: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.task_id


class DialogueHarness(Harness):
    """Harness variant that drives the τ³-Bench multi-turn dialogue loop.

    Overrides :meth:`run` to alternate model calls, tool execution, and user
    simulator replies, returning a trajectory whose ``final_output`` is the
    resulting database state (consumed by the verifier).
    """

    async def run(self, task: Any, task_id: str | None = None) -> Trajectory:
        task_id = task_id or getattr(task, "task_id", None) or str(id(task))
        domain: Domain = get_domain(task.domain)
        db = domain.build_db()
        provider = self.model_config.provider_for("main")
        user_provider = self.model_config.provider_for("user")
        user_simulator = domain.user_simulator(task, provider=user_provider)
        runner = DialogueRunner(
            provider=provider,
            tools=domain.tools(),
            user_simulator=user_simulator,
            db=db,
            opening=task.opening,
            max_turns=self.harness_config.max_steps,
        )
        result: DialogueResult = await runner.run()
        return Trajectory(task_id=task_id, final_output=result)


async def verify_tau3(task: Any, result: Any) -> bool:
    if not isinstance(result, DialogueResult):
        return False
    return verify_retail(task, result.db_state)


class Tau3Adapter:
    name = "tau3"

    def __init__(self, data_path: str | Path | None = None) -> None:
        self.data_path = Path(data_path) if data_path else None

    def load_tasks(self) -> list[Tau3Task]:
        if self.data_path is None:
            raise ValueError("tau3 requires a data_path to a JSONL file")
        tasks: list[Tau3Task] = []
        with open(self.data_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                tasks.append(
                    Tau3Task(
                        task_id=rec["task_id"],
                        domain=rec.get("domain", "retail"),
                        opening=rec["opening"],
                        script=rec.get("script", []),
                        expected=rec.get("expected", {}),
                        instruction=rec.get("instruction"),
                        metadata={k: v for k, v in rec.items() if k not in
                                  ("task_id", "domain", "opening", "script", "expected", "instruction")},
                    )
                )
        return tasks

    def verifier(self) -> Any:
        return verify_tau3

    def default_harness_config(self) -> HarnessConfig:
        config = HarnessConfig(max_steps=200)
        config.add(
            Hook.TASK_START,
            SystemPromptProcessor(
                prompt=(
                    "You are a customer-service agent. Help the user by calling "
                    "tools to look up and modify their account, orders, and "
                    "payments. Confirm changes with the user before finalizing."
                )
            ),
        )
        return config
