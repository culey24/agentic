"""Multi-objective RL experiment on τ³-Bench (retail domain).

Runs the AEGIS co-evolution loop collecting reward vectors
(correctness / efficiency / tool_safety) via :class:`MultiObjectiveBridge`,
then reports per-objective means, scalarized reward, and Pareto front stats
from the replay buffer.

Run:
  uv run python experiments/run_tau3.py
  uv run python experiments/run_tau3.py --echo --rounds 1   # offline smoke test
  uv run python experiments/run_tau3.py --weights correctness=1.0,efficiency=0.3,tool_safety=0.5 --api-only

Environment: OPENROUTER_API_KEY (or ANTHROPIC_API_KEY + --provider anthropic).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.loop import MultiObjectiveEvolutionLoop
from experiments.multiobj import (
    MultiObjectiveCollectOnlyTrainer,
    MultiObjectiveGRPOTrainer,
)
from harnessx import ModelConfig
from harnessx.benchmarks.tau3 import (
    DialogueHarness,
    Tau3Adapter,
    verify_tau3,
)
from harnessx.events import Message
from harnessx.providers.base import Provider, ProviderResponse
from harnessx.rl import MixedPolicyBuffer
from harnessx.tracing.journal import Journal


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


class EchoProvider(Provider):
    async def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, object]] | None = None,
        **kwargs: object,
    ) -> ProviderResponse:
        return ProviderResponse(content="No tool call needed.", stop_reason="end_turn")


def _build_provider(args: argparse.Namespace, role: str) -> Provider:
    if args.echo:
        return EchoProvider("echo")
    if args.provider == "anthropic":
        from harnessx.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(args.model)
    from harnessx.providers.openrouter_provider import OpenRouterProvider

    return OpenRouterProvider(args.model)


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


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="examples/data/tau3_sample.jsonl")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--n-rollouts", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--weights", default=None,
                        help="objective weights, e.g. correctness=1.0,efficiency=0.3,tool_safety=0.5")
    parser.add_argument("--binary", action="store_true",
                        help="use binary correctness instead of soft fraction")
    parser.add_argument("--api-only", action="store_true",
                        help="collect records without trainer update (CollectOnly)")
    parser.add_argument("--echo", action="store_true", help="offline smoke test with an echo provider")
    parser.add_argument("--provider", choices=["openrouter", "anthropic"], default="openrouter")
    parser.add_argument("--model", default="qwen/qwen-2.5-72b-instruct")
    parser.add_argument("--max-steps", type=int, default=200, help="harness max_steps")
    parser.add_argument("--name", default="tau3_mo_rl", help="journal run name")
    args = parser.parse_args()

    _load_dotenv()
    weights = _parse_weights(args.weights)

    adapter = Tau3Adapter(data_path=args.data)
    tasks = adapter.load_tasks()
    if not tasks:
        sys.exit(f"no tasks loaded from {args.data}")
    print(
        f"tasks={len(tasks)} rounds={args.rounds} rollouts={args.n_rollouts} "
        f"concurrency={args.concurrency} api_only={args.api_only} weights={weights}\n"
    )

    provider = _build_provider(args, "main")
    model_cfg = ModelConfig(main=provider, user=provider, meta=provider if not args.echo else None)

    harness_config = adapter.default_harness_config()
    harness_config.max_steps = args.max_turns
    harness = DialogueHarness(model_cfg, harness_config)

    buffer = MixedPolicyBuffer(capacity=10000)
    trainer = (
        MultiObjectiveCollectOnlyTrainer()
        if args.api_only
        else MultiObjectiveGRPOTrainer(batch_size=256)
    )
    loop = MultiObjectiveEvolutionLoop(
        meta_provider=None if args.echo else provider,
        harness=harness,
        tasks=tasks,
        verifier=verify_tau3,
        journal=Journal(args.name),
        n_rollouts=args.n_rollouts,
        max_rounds=args.rounds,
        concurrency=args.concurrency,
        buffer=buffer,
        trainer=trainer,
        objective_weights=weights,
        binary_correctness=args.binary,
        max_turns=args.max_turns,
    )
    result = await loop.run()

    print("\n=== RESULTS ===")
    print(f"rounds run: {len(result['history'])}")
    print("pass@N per round: " + ", ".join(f"R{i}={r:.3f}" for i, r in enumerate(result["history"])))
    print(f"final pass_rate = {result['final']:.3f}")
    print(f"buffer size = {result['buffer_size']}  train steps = {result['train_steps']}")
    if buffer and len(buffer):
        records = list(buffer)
        rewards = [r.extra.get("rewards") for r in records]
        if any(isinstance(rw, dict) and rw for rw in rewards):
            names = sorted({n for rw in rewards if isinstance(rw, dict) for n in rw})
            print("objective means over buffer:")
            for name in names:
                vals = [(rw or {}).get(name, 0.0) for rw in rewards]
                print(f"  {name:<12} mean={sum(vals) / len(vals):.3f}")
    print(f"\njournal + audit under output/runs/{args.name}/")


if __name__ == "__main__":
    asyncio.run(main())