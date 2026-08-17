"""Multi-objective reward math: scalarization, Pareto, and per-objective advantages.

Imported from ``harnessx`` only (reuses the existing scalar GRPO machinery in
``harnessx.rl.grpo``); nothing under ``src/harnessx`` is modified.
"""

from __future__ import annotations

from dataclasses import dataclass

from harnessx.rl.grpo import group_relative_advantage

EPSILON = 1e-8


@dataclass(frozen=True)
class ObjectiveSpec:
    name: str
    weight: float = 1.0
    minimize: bool = False

    def normalized(self, reward: float) -> float:
        return -reward if self.minimize else reward


def make_specs(weights: dict[str, float] | None) -> list[ObjectiveSpec]:
    """Build specs from a ``{name: weight}`` mapping (all objectives maximize)."""
    if not weights:
        return []
    return [ObjectiveSpec(name=n, weight=w) for n, w in weights.items()]


def scalarize(rewards: dict[str, float], specs: list[ObjectiveSpec]) -> float:
    """Weighted-sum scalarization of an objective vector.

    ``minimize`` objectives contribute negatively; the result is normalized by
    the total applied weight. Returns ``0.0`` when no specs or rewards exist.
    """
    if not specs or not rewards:
        return 0.0
    total = 0.0
    wsum = 0.0
    for spec in specs:
        if spec.name not in rewards:
            continue
        total += spec.weight * spec.normalized(rewards[spec.name])
        wsum += spec.weight
    return total / wsum if wsum else 0.0


def pareto_dominance(
    a: dict[str, float], b: dict[str, float], specs: list[ObjectiveSpec]
) -> bool:
    """True if ``a`` dominates ``b``: >= every objective and > at least one."""
    strictly_better = False
    for spec in specs:
        if spec.name not in a or spec.name not in b:
            continue
        va, vb = spec.normalized(a[spec.name]), spec.normalized(b[spec.name])
        if va > vb:
            strictly_better = True
        elif va < vb:
            return False
    return strictly_better


def non_dominated_sort(
    records: list[dict[str, float]],
    specs: list[ObjectiveSpec],
) -> list[list[int]]:
    """Return Pareto front indices; ``fronts[0]`` holds the non-dominated records."""
    n = len(records)
    dominated_by: list[int] = [0] * n
    dominates: list[list[int]] = [[] for _ in range(n)]
    fronts: list[list[int]] = [[]]

    for i in range(n):
        for j in range(i + 1, n):
            a_better = pareto_dominance(records[i], records[j], specs)
            b_better = pareto_dominance(records[j], records[i], specs)
            if a_better and not b_better:
                dominates[i].append(j)
                dominated_by[j] += 1
            elif b_better and not a_better:
                dominates[j].append(i)
                dominated_by[i] += 1

    for i in range(n):
        if dominated_by[i] == 0:
            fronts[0].append(i)

    front_idx = 0
    while fronts[front_idx]:
        nxt: list[int] = []
        for i in fronts[front_idx]:
            for j in dominates[i]:
                dominated_by[j] -= 1
                if dominated_by[j] == 0:
                    nxt.append(j)
        front_idx += 1
        fronts.append(nxt)
    fronts.pop()
    return fronts


def multi_group_relative_advantage(
    rewards: list[dict[str, float]],
    group_ids: list[str],
    specs: list[ObjectiveSpec],
    epsilon: float = EPSILON,
) -> dict[int, dict[str, float]]:
    """Per-objective group-relative advantages plus a scalarized value.

    Each objective is normalized within its task group exactly like
    ``harnessx.rl.grpo.group_relative_advantage``; the resulting vector is then
    scalarized with the weighted sum into ``"scalarized"``.
    """
    per_objective: dict[str, dict[int, float]] = {}
    for spec in specs:
        col = [r.get(spec.name, 0.0) for r in rewards]
        per_objective[spec.name] = group_relative_advantage(col, group_ids, epsilon)

    out: dict[int, dict[str, float]] = {}
    for i in range(len(rewards)):
        vec = {name: adv[i] for name, adv in per_objective.items()}
        vec["scalarized"] = scalarize(
            {spec.name: vec[spec.name] for spec in specs}, specs
        )
        out[i] = vec
    return out