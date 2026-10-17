from copy import deepcopy
import pytest

from analysis.mo_alns_study import snapshot, platform_test, mechanism_test, clustered_improvement


def job(success=True):
    return {"summary": {"instance_id": "one", "pressure_type": "easy", "preference": [1, 0, 0], "seed": 11,
                         "initial_quality": .4, "quality": .3},
            "candidates": [{"completed_seconds": 3, "feasible": success, "quality": .4,
                            "action_trace_sha256": "a", "objectives": [1, 2, 3]},
                           {"completed_seconds": 12, "feasible": True, "quality": .3,
                            "action_trace_sha256": "b", "objectives": [0, 2, 3]}],
            "events": [{"kind": "neighbour", "completed_seconds": 15, "completed_neighbourhoods": 20,
                        "weight_updates": 4}]}


def test_snapshot_cannot_use_late_candidate_or_late_iteration():
    data = job(False)
    before = snapshot(data, 10)
    after = snapshot(data, 20)
    assert not before["success"] and before["quality"] == 1
    assert before["completed_neighbourhoods"] == 0
    assert after["quality"] == .3 and after["completed_neighbourhoods"] == 20


def test_new_success_disqualifies_budget_plateau():
    data = job(False)
    assert not platform_test([snapshot(data, 10)], [snapshot(data, 20)])["passed"]


def test_mechanism_requires_each_stratum():
    a = snapshot(job(), 20)
    b = deepcopy(a)
    b.update(instance_id="two", pressure_type="fatigue_bottleneck", weight_updates=0)
    assert not mechanism_test([a, b])["passed"]


def test_bootstrap_uses_instances_instead_of_preferences():
    one = job()
    duplicated = [deepcopy(one) for _ in range(13)]
    stats = clustered_improvement(duplicated, "initial_quality")
    assert stats["instance_count"] == 1
    assert stats["mean_gain"] == pytest.approx(.1)
    assert stats["confidence_interval_95"] == pytest.approx([.1, .1])
