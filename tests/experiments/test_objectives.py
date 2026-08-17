from experiments.multiobj.objectives import (
    ObjectiveSpec,
    make_specs,
    multi_group_relative_advantage,
    non_dominated_sort,
    pareto_dominance,
    scalarize,
)


def test_scalarize_weighted_sum():
    specs = [ObjectiveSpec("a", weight=1.0), ObjectiveSpec("b", weight=1.0)]
    assert scalarize({"a": 0.8, "b": 0.2}, specs) == 0.5


def test_scalarize_minimize_flips_sign():
    specs = [ObjectiveSpec("cost", weight=1.0, minimize=True)]
    assert scalarize({"cost": 0.3}, specs) == -0.3


def test_scalarize_empty():
    assert scalarize({}, [ObjectiveSpec("a")]) == 0.0
    assert scalarize({"a": 0.5}, []) == 0.0


def test_make_specs():
    specs = make_specs({"a": 2.0, "b": 1.0})
    assert [s.name for s in specs] == ["a", "b"]
    assert specs[0].weight == 2.0
    assert make_specs(None) == []


def test_pareto_dominance():
    specs = [ObjectiveSpec("a"), ObjectiveSpec("b")]
    assert pareto_dominance({"a": 1.0, "b": 0.5}, {"a": 0.9, "b": 0.5}, specs)
    assert not pareto_dominance({"a": 1.0, "b": 0.5}, {"a": 1.0, "b": 0.5}, specs)
    assert not pareto_dominance({"a": 1.0, "b": 0.0}, {"a": 0.5, "b": 0.5}, specs)


def test_non_dominated_sort():
    specs = [ObjectiveSpec("a"), ObjectiveSpec("b")]
    records = [
        {"a": 1.0, "b": 1.0},
        {"a": 0.5, "b": 0.5},
        {"a": 0.0, "b": 0.0},
    ]
    fronts = non_dominated_sort(records, specs)
    assert fronts[0] == [0]
    assert fronts[1] == [1]
    assert fronts[2] == [2]


def test_multi_group_relative_advantage():
    specs = [ObjectiveSpec("correctness"), ObjectiveSpec("efficiency")]
    rewards = [
        {"correctness": 1.0, "efficiency": 0.8},
        {"correctness": 0.0, "efficiency": 0.6},
        {"correctness": 1.0, "efficiency": 0.9},
        {"correctness": 0.0, "efficiency": 0.5},
    ]
    groups = ["g1", "g1", "g2", "g2"]
    out = multi_group_relative_advantage(rewards, groups, specs)
    assert out[0]["correctness"] > out[1]["correctness"]
    assert out[2]["correctness"] > out[3]["correctness"]
    assert "scalarized" in out[0]