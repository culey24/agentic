"""WebShop adapter: web interaction with attribute-match verification.

WebShop agents search a catalog, inspect products, and buy one; the reward is
the fraction of goal attributes matched by the purchased product. This module
defines the adapter and a minimal self-contained catalog for offline runs; the
real WebShop server can be wired in via the same :class:`TextEnv` protocol.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class WebShopTask:
    task_id: str
    goal: dict[str, Any]
    catalog: list[dict[str, Any]] = field(default_factory=list)
    system_prompt: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    env: Any = None

    @property
    def id(self) -> str:
        return self.task_id


class MiniWebShopEnv:
    """A minimal catalog-backed WebShop for offline runs and tests."""

    def __init__(self) -> None:
        self.goal: dict[str, Any] = {}
        self.catalog: list[dict[str, Any]] = []
        self.results: list[dict[str, Any]] = []
        self.current_product: dict[str, Any] | None = None
        self.success = False

    async def reset(self, task: WebShopTask) -> str:
        self.goal = dict(task.goal)
        self.catalog = list(task.catalog)
        self.results = []
        self.current_product = None
        self.success = False
        return "WebShop. Use search[...] to find products, then buy one."

    async def step(self, action: str) -> tuple[str, bool, Any]:
        action = action.strip()
        if action.startswith("search["):
            query = action[len("search["):-1].lower()
            self.results = [
                p for p in self.catalog if query in p["title"].lower()
            ]
            if not self.results:
                return "No results found.", False, 0.0
            lines = [f"{i + 1}. {p['title']} - ${p['price']}" for i, p in enumerate(self.results)]
            return "Results:\n" + "\n".join(lines), False, 0.0

        if action.startswith("click[item "):
            idx = int(action[len("click[item "):-1]) - 1
            if not (0 <= idx < len(self.results)):
                return "Invalid item.", False, 0.0
            self.current_product = self.results[idx]
            attrs = ", ".join(f"{k}: {v}" for k, v in self.current_product["attributes"].items())
            return (
                f"{self.current_product['title']} - ${self.current_product['price']}\n"
                f"Attributes: {attrs}\nUse click[Buy Now] to purchase."
            ), False, 0.0

        if action == "click[Buy Now]":
            if self.current_product is None:
                return "No product selected.", False, 0.0
            reward = self._attribute_match(self.goal, self.current_product["attributes"])
            self.success = reward >= 1.0
            return (
                f"Purchased {self.current_product['title']}.",
                True,
                reward,
            )

        return "Invalid action.", False, 0.0

    @staticmethod
    def _attribute_match(goal: dict[str, Any], attrs: dict[str, Any]) -> float:
        if not goal:
            return 0.0
        matched = sum(1 for k, v in goal.items() if attrs.get(k) == v)
        return matched / len(goal)


async def verify_webshop(task: Any, result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    return bool(result.get("done")) and float(result.get("reward") or 0.0) >= 1.0


class WebShopAdapter:
    name = "webshop"

    def __init__(
        self,
        data_path: str | Path | None = None,
        env_factory: Any = None,
    ) -> None:
        self.data_path = Path(data_path) if data_path else None
        self.env_factory = env_factory or MiniWebShopEnv

    def load_tasks(self) -> list[WebShopTask]:
        if self.data_path is None:
            raise ValueError("webshop requires a data_path to a JSONL file")
        tasks: list[WebShopTask] = []
        with open(self.data_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                task = WebShopTask(
                    task_id=rec["task_id"],
                    goal=rec.get("goal", {}),
                    catalog=rec.get("catalog", []),
                    system_prompt=rec.get("system_prompt"),
                    metadata={k: v for k, v in rec.items() if k not in
                              ("task_id", "goal", "catalog", "system_prompt")},
                )
                task.env = self.env_factory()
                tasks.append(task)
        return tasks

    def verifier(self) -> Any:
        return verify_webshop

    def default_harness_config(self) -> Any:
        from harnessx.core.harness_config import HarnessConfig

        return HarnessConfig(max_steps=20)
