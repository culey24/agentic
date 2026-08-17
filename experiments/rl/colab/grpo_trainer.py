"""Agentic GRPO trainer: token-level policy gradient over multi-turn rollouts.

GRPO (DeepSeek-R1 style) normally trains single-turn responses; here the policy
gradient is applied over *all assistant tokens across all turns* of a tau3
trajectory, with a trajectory-level group-relative advantage and a per-token KL
anchor against the frozen reference policy.

Unlike the reporting-only :class:`harnessx.rl.coevolution.GRPOTrainer`, this one
performs an actual gradient step on a LoRA model exposed by the local provider.

Core math is a pure function (:func:`grpo_token_loss`) so it can be unit-tested
without torch; the trainer orchestrates padded forwards on the GPU.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from experiments.rl.colab.rollout import RolloutRecord
from harnessx.rl.grpo import GRPOConfig, group_relative_advantage

logger = logging.getLogger("experiments.rl.grpo")


def grpo_token_loss(
    advantage: float,
    logp_cur: float,
    logp_old: float,
    logp_ref: float,
    clip_ratio: float = 0.2,
    beta: float = 0.04,
) -> float:
    """Per-token GRPO loss for one generated token.

    ``-min(ratio*A, clip(ratio)*A) + beta*(logp_ref - logp_cur)`` where
    ``ratio = exp(logp_cur - logp_old)``. High advantage with improved
    likelihood lowers the loss; deviation from the reference is penalized.
    """
    ratio = math.exp(logp_cur - logp_old)
    clipped = min(max(ratio, 1.0 - clip_ratio), 1.0 + clip_ratio)
    adv_term = min(ratio * advantage, clipped * advantage)
    kl = logp_ref - logp_cur
    return -adv_term + beta * kl


def trajectory_rewards(rollouts: list[RolloutRecord]) -> tuple[list[float], list[str]]:
    return [r.reward for r in rollouts], [r.task_id for r in rollouts]


def flattened_turns(
    rollouts: list[RolloutRecord],
) -> list[tuple[int, RolloutRecord, Any]]:
    """Flatten all assistant turns into (trajectory_index, record, turn)."""
    out: list[tuple[int, RolloutRecord, Any]] = []
    for i, rec in enumerate(rollouts):
        for turn in rec.turns:
            out.append((i, rec, turn))
    return out


@dataclass
class GRPOStats:
    loss: float
    mean_reward: float
    mean_advantage: float
    std_advantage: float
    mean_kl: float = 0.0
    n_tokens: int = 0
    n_trajectories: int = 0


class GRPOTrainer:
    """Performs a real GRPO gradient step on a LoRA model via the provider."""

    def __init__(
        self,
        provider: Any,
        config: GRPOConfig | None = None,
        lr: float = 5e-5,
        save_path: str | None = None,
        logprob_fn: Callable | None = None,
    ) -> None:
        self.provider = provider
        self.config = config or GRPOConfig()
        self.lr = lr
        self.save_path = save_path
        # ``logprob_fn`` is an injection seam for unit tests; real runs use the
        # provider's batched forward with gradients.
        self.logprob_fn = logprob_fn or getattr(provider, "training_logprobs", None)
        self._optimizer = None

    def _make_optimizer(self) -> Any:
        import torch

        params = [p for p in self.provider._model.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=self.lr)

    def update(self, rollouts: list[RolloutRecord], config: GRPOConfig | None = None) -> GRPOStats:
        config = config or self.config
        rewards, groups = trajectory_rewards(rollouts)
        advantages = group_relative_advantage(rewards, groups, config.epsilon)

        examples = [
            (turn.prompt_tokens, turn.completion_tokens)
            for _, _, turn in flattened_turns(rollouts)
        ]
        turn_objs = [turn for _, _, turn in flattened_turns(rollouts)]
        turn_indices = [idx for idx, _, _ in flattened_turns(rollouts)]

        if not examples:
            return GRPOStats(loss=0.0, mean_reward=_mean(rewards), mean_advantage=0.0, std_advantage=0.0)

        import torch

        self.provider._model.train()
        if self._optimizer is None:
            self._optimizer = self._make_optimizer()
        self._optimizer.zero_grad()

        logp_cur, _starts, lengths = self.logprob_fn(examples)
        loss_terms: list[Any] = []
        kl_terms: list[Any] = []
        n_tokens = 0

        for k, turn in enumerate(turn_objs):
            adv = advantages[turn_indices[k]]
            cur = logp_cur[k]
            old = torch.tensor(
                _padded(turn.log_probs, lengths[k]),
                dtype=torch.float32,
                device=cur.device,
            )
            ref = torch.tensor(
                _padded(turn.ref_log_probs, lengths[k]),
                dtype=torch.float32,
                device=cur.device,
            )
            ratio = torch.exp(cur - old)
            clipped = torch.clamp(ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio)
            adv_term = torch.minimum(ratio * adv, clipped * adv)
            kl = ref - cur
            loss_terms.append(-adv_term + config.beta * kl)
            kl_terms.append(kl)
            n_tokens += lengths[k]

        loss = torch.cat(loss_terms).mean()
        loss.backward()
        self._optimizer.step()
        self.provider._model.eval()

        if self.save_path:
            self.provider.save_lora(self.save_path)

        adv_list = list(advantages.values())
        mean_kl = torch.cat(kl_terms).mean().item() if kl_terms else 0.0
        stats = GRPOStats(
            loss=loss.item(),
            mean_reward=_mean(rewards),
            mean_advantage=_mean(adv_list),
            std_advantage=_std(adv_list),
            mean_kl=mean_kl,
            n_tokens=n_tokens,
            n_trajectories=len(rollouts),
        )
        logger.info(
            "GRPO step: loss=%.4f mean_reward=%.3f adv=%.3f+-%.3f kl=%.4f tokens=%d",
            stats.loss, stats.mean_reward, stats.mean_advantage, stats.std_advantage,
            stats.mean_kl, stats.n_tokens,
        )
        return stats


def _padded(values: list[float], length: int) -> list[float]:
    return values[:length] if len(values) >= length else values + [0.0] * (length - len(values))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mu = _mean(values)
    return (sum((v - mu) ** 2 for v in values) / len(values)) ** 0.5