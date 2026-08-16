"""Cross-harness GRPO objective and group-relative advantage (Section 5.3).

Groups trajectories by task identity across harness versions and computes the
group-relative advantage (Eq. 3). The clipped surrogate objective (Eq. 4) and
KL anchor are provided as pure functions; the actual gradient step requires a
parametric trainer (VERL) and GPUs, which are out of scope for API-only runs.
"""

from __future__ import annotations

from dataclasses import dataclass


def group_relative_advantage(
    rewards: list[float],
    group_ids: list[str],
    epsilon: float = 1e-8,
) -> dict[int, float]:
    """Compute per-sample advantage ``(r - mu(group)) / (sigma(group) + eps)``.

    Returns a mapping from index -> advantage.
    """
    groups: dict[str, list[int]] = {}
    for i, gid in enumerate(group_ids):
        groups.setdefault(gid, []).append(i)

    advantages: dict[int, float] = {}
    for indices in groups.values():
        group_rewards = [rewards[i] for i in indices]
        mu = sum(group_rewards) / len(group_rewards)
        var = sum((r - mu) ** 2 for r in group_rewards) / len(group_rewards)
        sigma = var ** 0.5
        for i in indices:
            advantages[i] = (rewards[i] - mu) / (sigma + epsilon)
    return advantages


@dataclass
class GRPOConfig:
    clip_ratio: float = 0.2
    beta: float = 0.04
    epsilon: float = 1e-8


def clipped_objective(
    advantages: list[float],
    ratios: list[float],
    config: GRPOConfig,
) -> float:
    """Surrogate objective ``mean(min(r*A, clip(r)*A))`` (Eq. 4)."""

    values = []
    for adv, ratio in zip(advantages, ratios):
        clipped = min(max(ratio, 1 - config.clip_ratio), 1 + config.clip_ratio)
        values.append(min(ratio * adv, clipped * adv))
    return sum(values) / len(values) if values else 0.0


def kl_penalty(kl_divergence: float, beta: float) -> float:
    """KL anchor term ``beta * D_KL(pi_theta || pi_ref)``."""
    return beta * kl_divergence
