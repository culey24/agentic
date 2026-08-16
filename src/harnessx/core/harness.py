"""Harness: the executable pairing of a model config and a harness config.

An agent in HarnessX is a processor pipeline bound to a model, both
independently substitutable: ``agent = model_config.agentic(harness_config)``.
"""

from __future__ import annotations

from typing import Any

from harnessx.core.harness_config import HarnessConfig
from harnessx.core.model_config import ModelConfig
from harnessx.core.run_loop import RunLoop
from harnessx.core.trajectory import Trajectory


class Harness:
    def __init__(
        self, model_config: ModelConfig, harness_config: HarnessConfig
    ) -> None:
        self.model_config = model_config
        self.harness_config = harness_config

    async def run(self, task: Any, task_id: str | None = None) -> Trajectory:
        loop = RunLoop(self.model_config, self.harness_config)
        return await loop.run(task, task_id=task_id)
