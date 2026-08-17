"""Multi-objective bridge: packs reward vectors + per-step observations.

Subclasses ``harnessx.rl.bridge.TrajectoryBridge`` and stores the multi-objective
payload in the existing ``RLRecord.extra`` dict, so no code under
``src/harnessx`` needs to change. Without an ``objective_scorer`` it falls back
to the parent single-objective behavior.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from experiments.multiobj.objectives import make_specs, scalarize
from harnessx.core.trajectory import Trajectory
from harnessx.rl import RLRecord
from harnessx.rl.bridge import TrajectoryBridge

logger = logging.getLogger("experiments.multiobj")

ObjectiveScorer = Callable[[Any, Trajectory], dict[str, float]]


class MultiObjectiveBridge(TrajectoryBridge):
    def __init__(
        self,
        objective_scorer: ObjectiveScorer | None = None,
        objective_weights: dict[str, float] | None = None,
    ) -> None:
        self.objective_scorer = objective_scorer
        self.objective_weights = objective_weights

    def to_record(
        self,
        traj: Trajectory,
        harness_version: str | None = None,
        log_probs: list[float] | None = None,
        tokens: list[int] | None = None,
        task: Any = None,
    ) -> RLRecord:
        if self.objective_scorer is None:
            return super().to_record(
                traj,
                harness_version=harness_version,
                log_probs=log_probs,
                tokens=tokens,
            )

        observations = _extract_observations(traj)
        process_rewards = [o.get("reward") or 0.0 for o in observations]
        metrics = _extract_metrics(traj, observations)
        rewards = _safe_score(self.objective_scorer, task or traj, traj)
        reward = _scalar_reward(rewards, self.objective_weights)

        record = super().to_record(
            traj,
            harness_version=harness_version,
            log_probs=log_probs,
            tokens=tokens,
        )
        record.reward = reward
        record.extra = {
            "rewards": rewards,
            "observations": observations,
            "process_rewards": process_rewards,
            "metrics": metrics,
        }
        return record


def _safe_score(scorer: ObjectiveScorer, task: Any, traj: Trajectory) -> dict[str, float]:
    try:
        out = scorer(task, traj)
        return {k: float(v) for k, v in out.items()}
    except Exception as exc:  # noqa: BLE001
        logger.warning("objective scorer failed (%s); using empty rewards", exc)
        return {}


def _scalar_reward(rewards: dict[str, float], weights: dict[str, float] | None) -> float:
    if not rewards:
        return 0.0
    if weights:
        return scalarize(rewards, make_specs(weights))
    if "correctness" in rewards:
        return rewards["correctness"]
    values = list(rewards.values())
    return sum(values) / len(values)


def _extract_observations(traj: Trajectory) -> list[dict[str, Any]]:
    if traj.steps:
        obs_list = [
            {
                "messages": getattr(step, "messages", []),
                "response": getattr(step, "response", None),
                "tool_calls": getattr(step, "tool_calls", []),
                "tool_results": getattr(step, "tool_results", []),
            }
            for step in traj.steps
        ]
    else:
        transcript = _transcript_from_final(traj.final_output)
        obs_list = [dict(entry) for entry in transcript] if transcript else []
    for obs in obs_list:
        obs["reward"] = _process_reward(obs)
    return obs_list


def _process_reward(obs: dict[str, Any]) -> float:
    for r in obs.get("tool_results") or []:
        if isinstance(r, dict) and _is_error(r):
            return -1.0
    if obs.get("role") == "tool" and _is_error(obs.get("result")):
        return -1.0
    return 0.0


def _transcript_from_final(final: Any) -> list[dict[str, Any]] | None:
    transcript = getattr(final, "transcript", None)
    if isinstance(transcript, list):
        return transcript
    if isinstance(final, dict) and isinstance(final.get("transcript"), list):
        return final["transcript"]
    return None


def _is_error(r: Any) -> bool:
    if not isinstance(r, dict):
        return False
    if "error" in r:
        return True
    result = r.get("result")
    return isinstance(result, dict) and "error" in result


def _extract_metrics(
    traj: Trajectory, observations: list[dict[str, Any]]
) -> dict[str, float]:
    tool_calls = 0
    tool_errors = 0
    for o in observations:
        calls = o.get("tool_calls") or []
        tool_calls += len(calls)
        for r in o.get("tool_results") or []:
            tool_calls += 1
            if isinstance(r, dict) and _is_error(r):
                tool_errors += 1
        if o.get("role") == "tool":
            tool_calls += 1
            result = o.get("result")
            if isinstance(result, dict) and "error" in result:
                tool_errors += 1
    metrics: dict[str, float] = {
        "steps": float(len(traj.steps) or len(observations)),
        "tool_calls": float(tool_calls),
        "tool_errors": float(tool_errors),
    }
    final = traj.final_output
    turns = getattr(final, "turns", None)
    if isinstance(turns, (int, float)):
        metrics["turns"] = float(turns)
    return metrics