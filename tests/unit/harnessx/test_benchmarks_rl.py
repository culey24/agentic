from harnessx.benchmarks.base import normalize_answer
from harnessx.benchmarks.gaia import GAIATask
from harnessx.core.trajectory import Trajectory
from harnessx.rl import TrajectoryBridge, group_relative_advantage


def test_normalize_answer() -> None:
    assert normalize_answer("Paris, France") == normalize_answer("paris france")
    assert normalize_answer("The Answer") in normalize_answer("the answer is X")


def test_gaia_task() -> None:
    task = GAIATask("g1", "Q?", 1, "42")
    assert task.ground_truth == "42"
    assert task.metadata["level"] == 1


def test_group_relative_advantage() -> None:
    rewards = [1.0, 0.0, 1.0, 0.0]
    groups = ["a", "a", "b", "b"]
    adv = group_relative_advantage(rewards, groups)
    assert adv[0] > adv[1]
    assert adv[2] > adv[3]


def test_bridge() -> None:
    traj = Trajectory(task_id="t1", final_output="42", reward=1.0)
    record = TrajectoryBridge().to_record(traj, harness_version="v1")
    assert record.group_id == "t1"
    assert record.reward == 1.0
