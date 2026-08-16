from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

from harnessx import HarnessConfig, ModelConfig
from harnessx.benchmarks.alfworld import ALFWorldTask, MiniAlfWorldEnv, verify_alfworld
from harnessx.benchmarks.swebench import SWEBenchTask, make_swe_tools, verify_swebench
from harnessx.benchmarks.text_env import TextGameHarness
from harnessx.benchmarks.webshop import MiniWebShopEnv, WebShopTask, verify_webshop
from harnessx.events import Message
from harnessx.providers.base import Provider, ProviderResponse


class ScriptedTextProvider(Provider):
    def __init__(self, actions: list[str]) -> None:
        super().__init__("fake")
        self.actions = iter(actions)

    async def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        action = next(self.actions, "done")
        return ProviderResponse(content=action, stop_reason="end_turn")


def test_alfworld_harness_solves_task() -> None:
    async def main() -> None:
        task = ALFWorldTask(
            task_id="alfworld-001",
            task_type="pick_and_place",
            object="mug 1",
            source="shelf 1",
            target="cabinet 1",
        )
        task.env = MiniAlfWorldEnv()
        provider = ScriptedTextProvider(
            ["go to shelf 1", "take mug 1 from shelf 1", "go to cabinet 1", "put mug 1 in cabinet 1"]
        )
        harness = TextGameHarness(ModelConfig(main=provider), HarnessConfig(max_steps=10))
        traj = await harness.run(task)
        assert await verify_alfworld(task, traj.final_output)

    asyncio.run(main())


def test_webshop_harness_buys_match() -> None:
    async def main() -> None:
        task = WebShopTask(
            task_id="webshop-001",
            goal={"color": "red"},
            catalog=[
                {"title": "Red Shirt", "price": 10, "attributes": {"color": "red", "size": "M"}},
                {"title": "Blue Shirt", "price": 12, "attributes": {"color": "blue", "size": "M"}},
            ],
        )
        task.env = MiniWebShopEnv()
        provider = ScriptedTextProvider(["search[shirt]", "click[item 1]", "click[Buy Now]"])
        harness = TextGameHarness(ModelConfig(main=provider), HarnessConfig(max_steps=10))
        traj = await harness.run(task)
        assert await verify_webshop(task, traj.final_output)

    asyncio.run(main())


def test_swe_tools(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n")

    async def main() -> None:
        registry = make_swe_tools(str(tmp_path))
        view = registry.get("view_file")
        assert await view.run(path="a.py") == "x = 1\n"

        edit = registry.get("edit_file")
        result = await edit.run(path="a.py", old="x = 1", new="x = 2")
        assert result["status"] == "ok"

        run = registry.get("run_command")
        result = await run.run(command="python -c 'print(1)'")
        assert result["status"] == "ok"

    asyncio.run(main())


def test_swebench_verify(tmp_path: Path) -> None:
    (tmp_path / "test_ok.py").write_text("def test_passes():\n    assert True\n")

    async def main() -> None:
        task = SWEBenchTask(
            instance_id="repo-1",
            repo="repo",
            problem_statement="fix",
            repo_path=str(tmp_path),
            fail_to_pass=["test_ok.py"],
            pass_to_pass=[],
            test_command=[sys.executable, "-m", "pytest", "-q", "test_ok.py"],
        )
        assert await verify_swebench(task, None)

    asyncio.run(main())
