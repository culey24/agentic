"""SWE-bench Verified adapter: code-editing agent with patch-resolution checks.

The agent works in a local checkout (``repo_path``) using view/edit/run tools to
fix the issue described in ``problem_statement``. The verifier runs the
``FAIL_TO_PASS`` and ``PASS_TO_PASS`` tests and requires them all to pass.

The official SWE-bench evaluation uses Docker images per instance; this
from-scratch adapter assumes a local checkout at the base commit with the test
patch already applied, and runs the tests directly.
"""

from __future__ import annotations

import asyncio
import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harnessx.core.harness import Harness
from harnessx.core.run_loop import RunLoop
from harnessx.core.trajectory import Trajectory
from harnessx.tools.registry import ToolRegistry


@dataclass
class SWEBenchTask:
    instance_id: str
    repo: str
    problem_statement: str
    repo_path: str
    fail_to_pass: list[str] = field(default_factory=list)
    pass_to_pass: list[str] = field(default_factory=list)
    test_command: list[str] | None = None
    system_prompt: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.instance_id

    @property
    def description(self) -> str:
        return self.problem_statement


def make_swe_tools(repo_path: str) -> ToolRegistry:
    root = Path(repo_path).resolve()

    def view_file(path: str) -> str:
        p = (root / path).resolve()
        if not str(p).startswith(str(root)):
            return {"error": "path escapes repo"}
        if not p.exists():
            return {"error": f"file not found: {path}"}
        return p.read_text()

    def edit_file(path: str, old: str, new: str) -> dict[str, Any]:
        p = (root / path).resolve()
        if not str(p).startswith(str(root)):
            return {"error": "path escapes repo"}
        if not p.exists():
            return {"error": f"file not found: {path}"}
        content = p.read_text()
        if old not in content:
            return {"error": "old string not found"}
        p.write_text(content.replace(old, new, 1))
        return {"status": "ok"}

    async def run_command(command: str) -> dict[str, Any]:
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=str(root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        return {"status": "ok" if proc.returncode == 0 else "error",
                "returncode": proc.returncode,
                "output": stdout.decode(errors="replace")[-4000:]}

    registry = ToolRegistry()
    registry.register(
        "view_file",
        view_file,
        description="Read the contents of a file in the repository.",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    )
    registry.register(
        "edit_file",
        edit_file,
        description="Replace the first occurrence of `old` with `new` in a file.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old": {"type": "string"},
                "new": {"type": "string"},
            },
            "required": ["path", "old", "new"],
        },
    )
    registry.register(
        "run_command",
        run_command,
        description="Run a shell command in the repository root.",
        parameters={
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    )
    return registry


class SWEHarness(Harness):
    """Harness variant that binds view/edit/run tools to the task's repo."""

    async def run(self, task: Any, task_id: str | None = None) -> Trajectory:
        task_id = task_id or getattr(task, "id", None) or str(id(task))
        config = copy.deepcopy(self.harness_config)
        config.tool_registry = make_swe_tools(task.repo_path)
        if task.system_prompt:
            from harnessx.core.hooks import Hook
            from harnessx.processors.context import SystemPromptProcessor

            existing = config.processors_for(Hook.TASK_START)
            if not any(p._singleton_group == "system_prompt" for p in existing):
                config.add(Hook.TASK_START, SystemPromptProcessor(prompt=task.system_prompt))
        loop = RunLoop(self.model_config, config)
        return await loop.run(task, task_id=task_id)


async def verify_swebench(task: Any, result: Any) -> bool:
    tests = list(task.fail_to_pass) + list(task.pass_to_pass)
    if not tests:
        return False
    command = list(task.test_command) if task.test_command else ["pytest", "-q", *tests]
    proc = await asyncio.create_subprocess_exec(
        *command,
        cwd=str(task.repo_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    stdout, _ = await proc.communicate()
    _ = stdout
    return proc.returncode == 0


class SWEBenchAdapter:
    name = "swebench"

    def __init__(self, data_path: str | Path | None = None) -> None:
        self.data_path = Path(data_path) if data_path else None

    def load_tasks(self) -> list[SWEBenchTask]:
        if self.data_path is None:
            raise ValueError("swebench requires a data_path to a JSONL file")
        tasks: list[SWEBenchTask] = []
        with open(self.data_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                tasks.append(
                    SWEBenchTask(
                        instance_id=rec["instance_id"],
                        repo=rec.get("repo", ""),
                        problem_statement=rec["problem_statement"],
                        repo_path=rec["repo_path"],
                        fail_to_pass=rec.get("FAIL_TO_PASS", []),
                        pass_to_pass=rec.get("PASS_TO_PASS", []),
                        test_command=rec.get("test_command"),
                        system_prompt=rec.get("system_prompt"),
                        metadata={k: v for k, v in rec.items() if k not in
                                  ("instance_id", "repo", "problem_statement", "repo_path",
                                   "FAIL_TO_PASS", "PASS_TO_PASS", "test_command", "system_prompt")},
                    )
                )
        return tasks

    def verifier(self) -> Any:
        return verify_swebench

    def default_harness_config(self) -> Any:
        from harnessx.core.harness_config import HarnessConfig
        from harnessx.core.hooks import Hook
        from harnessx.processors.context import SystemPromptProcessor

        config = HarnessConfig(max_steps=200)
        config.add(
            Hook.TASK_START,
            SystemPromptProcessor(
                prompt=(
                    "You are an expert software engineer. Fix the issue described "
                    "by editing the repository using the provided tools. When done, "
                    "state the change you made."
                )
            ),
        )
        return config
