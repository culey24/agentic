"""Mixed-policy replay buffer for off-policy GRPO (Section 5.4).

A FIFO buffer of capacity C holding RL records generated under successive
behavior policies. The off-policy bias is bounded by the model-version lag
floor(C/s) where s is the rollout batch size. Log-probabilities are cached to
disk at insertion so GRPO replays without new rollout cost.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

from harnessx.rl.bridge import RLRecord


class MixedPolicyBuffer:
    def __init__(self, capacity: int = 10000, cache_dir: str | Path | None = None) -> None:
        self.capacity = capacity
        self._buffer: deque[RLRecord] = deque(maxlen=capacity)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def insert(self, record: RLRecord) -> None:
        self._buffer.append(record)
        if self.cache_dir and record.log_probs is not None:
            path = self.cache_dir / f"{record.task_id}.jsonl"
            with open(path, "a") as f:
                f.write(json.dumps(record.to_dict()) + "\n")

    def sample(self, n: int) -> list[RLRecord]:
        import random

        return random.sample(list(self._buffer), min(n, len(self._buffer)))

    def __len__(self) -> int:
        return len(self._buffer)

    def __iter__(self) -> Any:
        return iter(self._buffer)
