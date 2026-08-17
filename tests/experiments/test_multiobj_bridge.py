import pytest

from experiments.multiobj.bridge import MultiObjectiveBridge
from experiments.multiobj.scorers.tau3 import make_tau3_scorer
from harnessx.benchmarks.tau3 import DialogueResult, Tau3Task
from harnessx.core.trajectory import StepRecord, Trajectory
from harnessx.rl import RLRecord, TrajectoryBridge


def _task():
    return Tau3Task(
        task_id="retail-001",
        domain="retail",
        opening="hi",
        script=[],
        expected={
            "orders": [
                {
                    "key_field": "order_id",
                    "key": "123",
                    "fields": {"status": "cancelled"},
                }
            ]
        },
    )


def _dialogue_result():
    return DialogueResult(
        db_state={
            "orders": [
                {
                    "order_id": "123",
                    "status": "cancelled",
                    "address": "1 Main St",
                    "items": [],
                }
            ]
        },
        transcript=[
            {"role": "assistant", "content": "Let me check.", "tool_calls": None},
            {
                "role": "tool",
                "name": "cancel_order",
                "result": {"status": "success", "order_status": "cancelled"},
            },
            {"role": "user", "content": "Thank you!"},
        ],
        turns=2,
        stopped=True,
    )


def test_bridge_extra_from_transcript():
    task = _task()
    traj = Trajectory(task_id="retail-001", final_output=_dialogue_result())
    scorer = make_tau3_scorer(max_turns=200)
    bridge = MultiObjectiveBridge(objective_scorer=scorer)
    record = bridge.to_record(traj, task=task)

    assert record.extra["rewards"]["correctness"] == 1.0
    assert record.extra["rewards"]["efficiency"] == 1.0 - 2 / 200
    assert record.extra["rewards"]["tool_safety"] == 1.0
    assert record.extra["metrics"]["tool_calls"] == 1
    assert record.extra["metrics"]["turns"] == 2
    assert len(record.extra["observations"]) == 3
    assert record.extra["process_rewards"] == [0.0, 0.0, 0.0]
    assert record.reward == pytest.approx(1.0)


def test_bridge_extra_from_steps():
    task = _task()
    steps = [
        StepRecord(
            step=0,
            tool_calls=[{"name": "cancel_order", "arguments": {}}],
            tool_results=[{"name": "cancel_order", "result": {"error": "boom"}}],
            reward=-1.0,
        )
    ]
    traj = Trajectory(task_id="retail-001", steps=steps, final_output=_dialogue_result())
    scorer = make_tau3_scorer(max_turns=200)
    bridge = MultiObjectiveBridge(objective_scorer=scorer)
    record = bridge.to_record(traj, task=task)

    # metrics are derived from the per-step trace, not the transcript blob
    assert record.extra["metrics"]["tool_errors"] == 1
    assert record.extra["metrics"]["tool_calls"] == 2
    assert len(record.extra["observations"]) == 1
    assert record.extra["process_rewards"] == [-1.0]
    # rewards still come from the DialogueResult verifier inputs
    assert record.extra["rewards"]["correctness"] == 1.0


def test_bridge_falls_back_to_scalar_without_scorer():
    task = _task()
    traj = Trajectory(task_id="retail-001", final_output="42", reward=1.0)
    bridge = MultiObjectiveBridge()
    record = bridge.to_record(traj, task=task)
    assert isinstance(record, RLRecord)
    assert record.reward == 1.0
    assert record.extra == {}


def test_bridge_scalar_matches_base_without_scorer():
    task = _task()
    traj = Trajectory(task_id="retail-001", final_output="42", reward=0.5)
    base = TrajectoryBridge().to_record(traj)
    mo = MultiObjectiveBridge().to_record(traj, task=task)
    assert mo.reward == base.reward
    assert mo.completion == base.completion