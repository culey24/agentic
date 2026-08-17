"""Dry-run: load the local model and roll out one tau3 task (no training).

De-risk step before GRPO: verify the model actually emits tool calls and that
rewards compute. Runs entirely inside the project venv via ``uv run`` so the
notebook kernel does not need torch/unsloth on its own path.

Usage:
  uv run python experiments/rl/colab/dry_run.py [--task 0] [--max-turns 20]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.multiobj.scorers.tau3 import make_tau3_scorer
from experiments.rl.colab.local_provider import LocalQwenProvider
from experiments.rl.colab.rollout import Tau3Rollout, single_objective_reward
from harnessx.benchmarks.tau3 import Tau3Adapter
from harnessx.benchmarks.tau3.retail import RetailDomain


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="examples/data/tau3_colab.jsonl")
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    args = parser.parse_args()

    tasks = Tau3Adapter(args.data).load_tasks()
    task = tasks[args.task]
    print(f"task[{args.task}] = {task.task_id}: {task.opening[:80]}")

    provider = LocalQwenProvider(model_name=args.model, max_tokens=256)
    provider.load()
    print(f"model loaded: {provider.model_name}")

    scorer = make_tau3_scorer(max_turns=args.max_turns)
    record = await Tau3Rollout(
        provider, RetailDomain(), max_turns=args.max_turns
    ).run(task, scorer, single_objective_reward)

    print("\n=== ROLLOUT ===")
    for i, turn in enumerate(record.turns):
        calls = [f"{c['name']}({c.get('arguments')})" for c in turn.tool_calls]
        print(f"turn {i}: content={turn.content[:80]!r} tool_calls={calls} process_reward={turn.process_reward}")
    print("\n=== REWARDS ===")
    print(f"objectives = {record.rewards}")
    print(f"scalar     = {record.reward}")
    print(f"turns      = {record.turns_count}  stopped = {record.stopped}")
    print(f"db orders  = {record.db_state.get('orders', [])}")


if __name__ == "__main__":
    asyncio.run(main())