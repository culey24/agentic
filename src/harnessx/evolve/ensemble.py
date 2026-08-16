"""Variant isolation via ensemble routing (Section 4.5).

Maintains up to K harness variants and routes each task to the variant with the
highest estimated success rate on that task's cluster across prior rounds. When
a candidate improves a subset of tasks but regresses others, it forks a new
variant instead of overwriting a shared harness, preventing cross-task
interference on heterogeneous benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from harnessx.core.harness_config import HarnessConfig


@dataclass
class Variant:
    name: str
    config: HarnessConfig
    successes: dict[str, int] = field(default_factory=dict)
    attempts: dict[str, int] = field(default_factory=dict)

    def record(self, task_id: str, success: bool) -> None:
        self.successes[task_id] = self.successes.get(task_id, 0) + int(success)
        self.attempts[task_id] = self.attempts.get(task_id, 0) + 1

    def success_rate(self, task_id: str) -> float | None:
        attempts = self.attempts.get(task_id)
        if not attempts:
            return None
        return self.successes.get(task_id, 0) / attempts

    def avg_success_rate(self) -> float:
        if not self.attempts:
            return 0.0
        return sum(self.successes.values()) / sum(self.attempts.values())


class Ensemble:
    def __init__(self, base_config: HarnessConfig, max_variants: int = 5) -> None:
        self.max_variants = max_variants
        self.variants: list[Variant] = [Variant("v0", base_config)]

    def route(self, task_id: str) -> Variant:
        best = self.variants[0]
        best_rate = best.success_rate(task_id)
        for variant in self.variants[1:]:
            rate = variant.success_rate(task_id)
            if rate is None:
                continue
            if best_rate is None or rate > best_rate:
                best, best_rate = variant, rate
        return best

    def fork(self, config: HarnessConfig) -> Variant:
        if len(self.variants) >= self.max_variants:
            weakest = min(self.variants, key=lambda v: v.avg_success_rate())
            self.variants.remove(weakest)
        variant = Variant(f"v{len(self.variants)}", config)
        self.variants.append(variant)
        return variant

    def replace(self, variant: Variant, config: HarnessConfig) -> None:
        variant.config = config
