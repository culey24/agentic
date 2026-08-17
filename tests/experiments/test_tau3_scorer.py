from experiments.multiobj.scorers.tau3 import make_tau3_scorer
from harnessx.benchmarks.tau3 import DialogueResult, Tau3Task
from harnessx.core.trajectory import StepRecord, Trajectory


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
                    "fields": {"status": "cancelled", "address": "2 Oak St"},
                }
            ]
        },
    )


def _result(status: str, address: str, turns: int, errors: int = 0):
    transcript = [
        {"role": "assistant", "content": "ok", "tool_calls": []},
        {"role": "tool", "name": "t", "result": {"error": "x"} if errors else {"ok": 1}},
    ]
    return DialogueResult(
        db_state={
            "orders": [
                {"order_id": "123", "status": status, "address": address, "items": []}
            ]
        },
        transcript=transcript,
        turns=turns,
        stopped=True,
    )


def test_correctness_soft_fraction():
    scorer = make_tau3_scorer(max_turns=200)
    # 1 of 2 checks satisfied -> 0.5
    scores = scorer(_task(), type("T", (), {"final_output": _result("cancelled", "WRONG", 3)})())
    assert scores["correctness"] == 0.5


def test_correctness_binary():
    scorer = make_tau3_scorer(max_turns=200, binary=True)
    full = type("T", (), {"final_output": _result("cancelled", "2 Oak St", 3)})()
    partial = type("T", (), {"final_output": _result("cancelled", "WRONG", 3)})()
    assert scorer(_task(), full)["correctness"] == 1.0
    assert scorer(_task(), partial)["correctness"] == 0.0


def test_efficiency():
    scorer = make_tau3_scorer(max_turns=100)
    traj = type("T", (), {"final_output": _result("cancelled", "2 Oak St", 10)})()
    assert scorer(_task(), traj)["efficiency"] == 0.9


def test_tool_safety_penalizes_errors():
    scorer = make_tau3_scorer(max_turns=100)
    clean = type("T", (), {"final_output": _result("cancelled", "2 Oak St", 3, errors=0)})()
    errored = type("T", (), {"final_output": _result("cancelled", "2 Oak St", 3, errors=1)})()
    assert scorer(_task(), clean)["tool_safety"] == 1.0
    assert scorer(_task(), errored)["tool_safety"] == 0.0


def test_scorer_counts_errors_from_step_records_without_transcript():
    """Regression: standard-RunLoop traces carry StepRecord objects, not dicts."""
    task = _task()
    traj = Trajectory(
        task_id="retail-001",
        steps=[
            StepRecord(
                step=0,
                tool_calls=[{"name": "cancel_order", "arguments": {}}],
                tool_results=[
                    {"name": "cancel_order", "result": {"error": "boom"}},
                    {"name": "get_user_details", "result": {"user_id": "u1"}},
                ],
            )
        ],
        final_output=DialogueResult(db_state={}, transcript=[], turns=4, stopped=True),
    )
    scorer = make_tau3_scorer(max_turns=100)
    scores = scorer(task, traj)
    assert scores["tool_safety"] == 0.5
    assert scores["correctness"] == 0.0