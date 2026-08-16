"""Verify HarnessX reimplementation end-to-end against the paper's claims.

Small-scale GAIA experiment:
  1. Baseline: static harness pass@2 on the task subset.
  2. Evolution: run the AEGIS loop for a few rounds, tracking pass@2 per round.
  3. Print a comparison table (like paper Table 3) and a verdict.

Run:
  OPENROUTER_API_KEY=... uv run python scripts/verify_gaia.py
  Optionally: HX_MODEL=model-id HX_META=model-id HX_DATA=path HX_TASKS=N HX_ROUNDS=N
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harnessx.benchmarks.gaia import GAIAAdapter
from harnessx.evolve.loop import EvolutionLoop, evaluate
from harnessx.providers.openrouter_provider import OpenRouterProvider
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


async def main() -> None:
    _load_dotenv()
    model_id = os.environ.get("HX_MODEL", "qwen/qwen-2.5-72b-instruct")
    meta_id = os.environ.get("HX_META", model_id)
    data = os.environ.get("HX_DATA", "examples/data/gaia_verify.jsonl")
    n_tasks = int(os.environ.get("HX_TASKS", "0"))
    rounds = int(os.environ.get("HX_ROUNDS", "3"))
    rollouts = int(os.environ.get("HX_ROLLOUTS", "2"))
    concurrency = int(os.environ.get("HX_CONCURRENCY", "10"))

    adapter = GAIAAdapter(data_path=data)
    tasks = adapter.load_tasks()
    if n_tasks:
        tasks = tasks[:n_tasks]
    print(f"model={model_id} meta={meta_id} tasks={len(tasks)} rounds={rounds} rollouts={rollouts}\n")

    provider = OpenRouterProvider(model_id)
    meta = OpenRouterProvider(meta_id)

    model_cfg = __import__("harnessx").ModelConfig(main=provider, meta=meta)
    harness = model_cfg.agentic(adapter.default_harness_config())

    baseline = await evaluate(harness, tasks, adapter.verifier(), n_rollouts=rollouts, concurrency=concurrency)
    base_rate = sum(r.solved for r in baseline.values()) / len(tasks)
    print(f"baseline pass@{rollouts} = {base_rate:.3f} ({sum(r.solved for r in baseline.values())}/{len(tasks)})\n")

    loop = EvolutionLoop(
        meta_provider=meta,
        harness=harness,
        tasks=tasks,
        verifier=adapter.verifier(),
        journal=Journal("verify_gaia"),
        n_rollouts=rollouts,
        max_rounds=rounds,
        concurrency=concurrency,
    )
    result = await loop.run()

    print("\n=== RESULTS ===")
    print(f"rounds run: {len(result['history'])}")
    print("pass@2 per round: " + ", ".join(f"R{i}={r:.3f}" for i, r in enumerate(result["history"])))
    final_rate = result["final"] or base_rate
    print(f"baseline={base_rate:.3f}  final={final_rate:.3f}  delta={final_rate - base_rate:+.3f}")
    print("\njournal + audit written under output/runs/verify_gaia/")


if __name__ == "__main__":
    asyncio.run(main())
