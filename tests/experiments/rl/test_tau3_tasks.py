"""Sanity checks for the synthesized tau3 task suite.

Every task's ``expected`` state must be realizable from the seed database, i.e.
when the final DB state satisfies ``expected``, ``verify_retail`` must pass.
"""

from __future__ import annotations

import copy

import pytest

from experiments.rl.colab.tau3_tasks import make_task_suite
from harnessx.benchmarks.tau3 import Tau3Adapter, verify_retail
from harnessx.benchmarks.tau3.retail import _seed_db


def _task_dicts() -> list[dict]:
    return make_task_suite()


def _apply_expected(db_state: dict, expected: dict) -> dict:
    state = copy.deepcopy(db_state)
    for table, checks in expected.items():
        for check in checks:
            key_field = check.get("key_field", "id")
            row = next((r for r in state[table] if r.get(key_field) == check["key"]), None)
            assert row is not None, f"missing row {table}.{key_field}={check['key']}"
            row.update(check["fields"])
    return state


def test_suite_loads_through_adapter(tmp_path) -> None:
    import json

    p = tmp_path / "tasks.jsonl"
    with open(p, "a") as f:
        f.writelines(json.dumps(task) + "\n" for task in _task_dicts())
    tasks = Tau3Adapter(p).load_tasks()
    assert len(tasks) == len(_task_dicts())
    assert all(t.domain == "retail" for t in tasks)


@pytest.mark.parametrize("task", _task_dicts(), ids=lambda t: t["task_id"])
def test_expected_state_is_reachable_from_seed(task: dict) -> None:
    seed = _seed_db()
    state = _apply_expected(seed.snapshot(), task["expected"])
    from harnessx.benchmarks.tau3 import Tau3Task

    t = Tau3Task(
        task_id=task["task_id"],
        domain="retail",
        opening=task["opening"],
        script=task["script"],
        expected=task["expected"],
        instruction=task.get("instruction"),
    )
    assert verify_retail(t, state) is True


@pytest.mark.parametrize("task", _task_dicts(), ids=lambda t: t["task_id"])
def test_expected_state_does_not_exist_in_seed(task: dict) -> None:
    """Unless the task explicitly expects no change, the raw seed must fail."""
    if task["task_id"] == "colab-trap-pay-001":
        pytest.skip("trap-pay expects an unchanged payment (refusal task)")
    seed = _seed_db().snapshot()
    from harnessx.benchmarks.tau3 import Tau3Task

    t = Tau3Task(
        task_id=task["task_id"],
        domain="retail",
        opening=task["opening"],
        script=task["script"],
        expected=task["expected"],
        instruction=task.get("instruction"),
    )
    assert verify_retail(t, seed) is False