"""Entrypoint for the tau3 multi-vs-single objective GRPO experiment on Colab.

Usage:
  python experiments/rl/colab/run.py --arm single --rounds 6 --rollouts 8
  python experiments/rl/colab/run.py --arm multi --weights correctness=1.0,efficiency=0.3,tool_safety=0.5
  python experiments/rl/colab/run.py --echo --rounds 1 --rollouts 1 --no-train   # offline smoke
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.multiobj.scorers.tau3 import make_tau3_scorer
from experiments.rl.colab.grpo_trainer import GRPOTrainer
from experiments.rl.colab.loop import PureRLLoop
from experiments.rl.colab.rollout import (
    single_objective_reward,
    weighted_objective_reward,
)
from harnessx.benchmarks.tau3 import Tau3Adapter
from harnessx.benchmarks.tau3.retail import RetailDomain
from harnessx.providers.base import Provider, ProviderResponse


def _load_dotenv(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


class EchoRolloutProvider(Provider):
    """Offline smoke provider: never calls tools, returns a plain message."""

    def tokenize(self, messages: list, tools: list | None = None) -> list[int]:
        return [0, 1, 2]

    def compute_logprobs(self, prompt: list[int], completion: list[int]) -> dict:
        n = len(completion) or 1
        return {"logprobs": [-0.5] * n, "ref_logprobs": [-0.5] * n}

    async def generate(self, messages: list, tools: list | None = None, **kwargs) -> ProviderResponse:
        return ProviderResponse(content="No tool call needed.", tool_calls=[], raw={"text": "No tool call needed.", "token_ids": [4, 5]})


def _parse_weights(raw: str | None) -> dict[str, float] | None:
    if not raw:
        return None
    weights: dict[str, float] = {}
    for token in raw.split(","):
        if "=" not in token:
            continue
        name, _, value = token.partition("=")
        weights[name.strip()] = float(value.strip())
    return weights or None


def _build_reward_fn(arm: str, weights: dict[str, float] | None):
    if arm == "multi":
        return weighted_objective_reward(weights or {"correctness": 1.0, "efficiency": 0.3, "tool_safety": 0.5})
    return single_objective_reward


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="examples/data/tau3_colab.jsonl")
    parser.add_argument("--arm", choices=["single", "multi"], default="single")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--rollouts", type=int, default=8, help="rollouts per task per round")
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--out", default=None, help="checkpoint dir")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--kl-beta", type=float, default=0.04)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--no-train", action="store_true", help="collect-only, skip GRPO step")
    parser.add_argument("--save-lora", default=None, help="path to persist the LoRA adapter after each update")
    parser.add_argument("--echo", action="store_true", help="offline smoke test without a model")
    args = parser.parse_args()

    weights = _parse_weights(args.weights)
    if args.echo:
        provider = EchoRolloutProvider("echo")
        trainer = None
    else:
        from experiments.rl.colab.local_provider import LocalQwenProvider
        from harnessx.rl.grpo import GRPOConfig

        provider = LocalQwenProvider(model_name=args.model)
        if args.no_train:
            trainer = None
        else:
            trainer = GRPOTrainer(
                provider,
                lr=args.lr,
                save_path=args.save_lora,
                config=GRPOConfig(clip_ratio=args.clip_ratio, beta=args.kl_beta),
            )

    adapter = Tau3Adapter(args.data)
    tasks = adapter.load_tasks()
    if not tasks:
        sys.exit(f"no tasks loaded from {args.data}")

    scorer = make_tau3_scorer(max_turns=args.max_turns)
    reward_fn = _build_reward_fn(args.arm, weights)
    out = args.out or f"output/runs/colab_{args.arm}"
    loop = PureRLLoop(
        provider=provider,
        tasks=tasks,
        domain=RetailDomain(),
        scorer=scorer,
        reward_fn=reward_fn,
        trainer=trainer,
        max_turns=args.max_turns,
        rollouts_per_task=args.rollouts,
        rounds=args.rounds,
        concurrency=args.concurrency,
        checkpoint_dir=out,
    )

    print(
        f"arm={args.arm} tasks={len(tasks)} rounds={args.rounds} rollouts={args.rollouts} "
        f"train={trainer is not None} weights={weights}\n"
    )
    summary = asyncio.run(loop.run())

    print("\n=== RESULTS ===")
    print("pass@N per round: " + ", ".join(f"R{i}={r:.3f}" for i, r in enumerate(summary.history)))
    print(f"final pass_rate = {summary.final_pass_rate:.3f}  best = {summary.best_pass_rate:.3f}")
    print(f"records = {summary.n_records}  tokens = {summary.n_tokens}")
    print("objective means:", {k: round(v, 3) for k, v in summary.objective_means.items()})
    print(f"pareto_fronts = {summary.pareto_fronts}  non_dominated = {summary.non_dominated}")
    print("round details:")
    for r in summary.rounds:
        extra = f" loss={r.train_loss:.4f} kl={r.train_kl:.4f}" if r.train_loss is not None else ""
        print(f"  R{r.round} pass={r.pass_rate:.3f} reward={r.mean_reward:.3f} {r.mean_rewards}{extra}")
    print(f"\ncheckpoint under {out}/")


if __name__ == "__main__":
    main()