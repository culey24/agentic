"""ALFWorld adapter: embodied text planning with goal-completion verification.

The full ALFWorld benchmark uses a TextWorld-based environment. This module
defines the adapter interface and a minimal self-contained ``MiniAlfWorldEnv``
(pick-and-place state machine) for offline runs and tests; a real TextWorld env
implementing the same :class:`TextEnv` protocol can be dropped in.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ALFWorldTask:
    task_id: str
    task_type: str
    object: str
    source: str
    target: str
    system_prompt: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    env: Any = None

    @property
    def id(self) -> str:
        return self.task_id


class MiniAlfWorldEnv:
    """A small pick-and-place environment used for offline runs and tests."""

    def __init__(self) -> None:
        self.object = ""
        self.source = ""
        self.target = ""
        self.held = False
        self.at: str | None = None
        self.success = False

    async def reset(self, task: ALFWorldTask) -> str:
        self.object = task.object
        self.source = task.source
        self.target = task.target
        self.held = False
        self.at = None
        self.success = False
        return (
            f"You are in a room. There is {self.object} on {self.source}. "
            f"There is {self.target} nearby. Goal: put {self.object} in {self.target}."
        )

    async def step(self, action: str) -> tuple[str, bool, Any]:
        action = action.lower().strip().rstrip(".")
        if action.startswith("go to "):
            self.at = action[len("go to "):].strip()
            return f"You arrive at {self.at}.", False, 0.0

        take = re.match(r"take (.+?) from (.+)", action)
        if take:
            obj, loc = take.group(1).strip(), take.group(2).strip()
            if self.at != loc:
                return f"You are not at {loc}.", False, 0.0
            if self.held or obj != self.object:
                return "You cannot take that.", False, 0.0
            self.held = True
            self.source = ""
            return f"You take {obj}.", False, 0.0

        put = re.match(r"put (.+?) (?:in|on) (.+)", action)
        if put:
            obj, loc = put.group(1).strip(), put.group(2).strip()
            if not self.held or obj != self.object:
                return "You are not holding that.", False, 0.0
            if self.at != loc:
                return f"You are not at {loc}.", False, 0.0
            self.held = False
            self.success = loc == self.target
            if self.success:
                return "Done. The object is in place.", True, 1.0
            return f"You put {obj} in {loc}, but that is not the goal.", False, 0.0

        return "Invalid action.", False, 0.0


async def verify_alfworld(task: Any, result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    return bool(result.get("done")) and bool(result.get("success"))


class ALFWorldAdapter:
    name = "alfworld"

    def __init__(
        self,
        data_path: str | Path | None = None,
        env_factory: Any = None,
    ) -> None:
        self.data_path = Path(data_path) if data_path else None
        self.env_factory = env_factory or MiniAlfWorldEnv

    def load_tasks(self) -> list[ALFWorldTask]:
        if self.data_path is None:
            raise ValueError("alfworld requires a data_path to a JSONL file")
        tasks: list[ALFWorldTask] = []
        with open(self.data_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                task = ALFWorldTask(
                    task_id=rec["task_id"],
                    task_type=rec.get("task_type", "pick_and_place"),
                    object=rec["object"],
                    source=rec["source"],
                    target=rec["target"],
                    system_prompt=rec.get("system_prompt"),
                    metadata={k: v for k, v in rec.items() if k not in
                              ("task_id", "task_type", "object", "source", "target", "system_prompt")},
                )
                task.env = self.env_factory()
                tasks.append(task)
        return tasks

    def verifier(self) -> Any:
        return verify_alfworld

    def default_harness_config(self) -> Any:
        from harnessx.core.harness_config import HarnessConfig

        return HarnessConfig(max_steps=15)
