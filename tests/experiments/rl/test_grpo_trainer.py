"""Unit tests for the custom GRPO trainer math (no torch required)."""

from __future__ import annotations

import pytest

from experiments.rl.colab.grpo_trainer import (
    flattened_turns,
    grpo_token_loss,
    trajectory_rewards,
)
from experiments.rl.colab.rollout import RolloutRecord, TurnRecord


def test_loss_lower_when_improving_under_positive_advantage() -> None:
    adv = 1.0
    baseline = grpo_token_loss(adv, logp_cur=-0.5, logp_old=-0.5, logp_ref=-0.5)
    improved = grpo_token_loss(adv, logp_cur=-0.1, logp_old=-0.5, logp_ref=-0.5)
    assert improved < baseline


def test_loss_higher_when_worse_under_positive_advantage() -> None:
    adv = 1.0
    baseline = grpo_token_loss(adv, logp_cur=-0.5, logp_old=-0.5, logp_ref=-0.5)
    worse = grpo_token_loss(adv, logp_cur=-0.9, logp_old=-0.5, logp_ref=-0.5)
    assert worse > baseline


def test_negative_advantage_encourages_decreasing_likelihood() -> None:
    adv = -1.0
    baseline = grpo_token_loss(adv, logp_cur=-0.5, logp_old=-0.5, logp_ref=-0.5)
    worse = grpo_token_loss(adv, logp_cur=-0.9, logp_old=-0.5, logp_ref=-0.5)
    assert worse < baseline


def test_kl_penalty_pushes_policy_toward_reference() -> None:
    # deviation from the reference (ref >> cur) raises the loss
    near_ref = grpo_token_loss(1.0, logp_cur=-0.5, logp_old=-0.5, logp_ref=-0.5, beta=0.04)
    far_from_ref = grpo_token_loss(1.0, logp_cur=-2.0, logp_old=-0.5, logp_ref=-0.5, beta=0.04)
    assert far_from_ref > near_ref


def test_clipping_caps_large_ratios() -> None:
    # a huge likelihood jump under a clipped ratio still moves the loss, but
    # the ratio is capped by the clip bound.
    adv = 1.0
    loss = grpo_token_loss(adv, logp_cur=5.0, logp_old=-5.0, logp_ref=0.0, clip_ratio=0.2)
    # clip ratio = 1.2 => adv_term = 1.2, kl = 0.0 - 5.0 = -5.0
    assert loss == pytest.approx(-1.2 + 0.04 * -5.0)


def test_trajectory_rewards() -> None:
    rollouts = [
        RolloutRecord(task_id="a", reward=1.0),
        RolloutRecord(task_id="a", reward=0.0),
        RolloutRecord(task_id="b", reward=1.0),
    ]
    rewards, groups = trajectory_rewards(rollouts)
    assert rewards == [1.0, 0.0, 1.0]
    assert groups == ["a", "a", "b"]


def test_flattened_turns() -> None:
    r = RolloutRecord(
        task_id="a",
        turns=[TurnRecord(prompt_tokens=[1], completion_tokens=[2, 3]), TurnRecord(prompt_tokens=[1], completion_tokens=[4])],
    )
    flat = flattened_turns([r])
    assert len(flat) == 2
    assert [idx for idx, _, _ in flat] == [0, 0]
    assert sum(t.n_tokens for _, _, t in flat) == 3