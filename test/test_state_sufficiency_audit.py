"""Historical observation aliases replayed against the repaired graph contract."""
import numpy as np
import pytest

from scripts.audit_state_sufficiency import (
    MODES, PROCESSING, SERVING, effective, worker_alias_instance, worker_histories,
    pair_comparison, current_relations, virtual_observation, reconstructed_quantities,
    event_order_checks, precision_checks, reconstructed_events, actual_events,
)
from scripts.audit_observation_reward import build_alias_instance, reachable_histories


@pytest.mark.parametrize("mode", MODES)
def test_reachable_processing_alias_is_reward_relevant(config, fixed_instance, mode):
    instance = build_alias_instance(fixed_instance)
    states, paths = reachable_histories(effective(config, mode), instance)
    result = pair_comparison(states)
    # Schema 10 also exposes actual starts/ends and sequential fatigue via derived edges.
    assert not result["original_equal"]
    assert not result["current_observation_equal"]
    wait = next(item for item in result["outcomes"] if item["is_wait"])
    assert wait["reward_difference"] == pytest.approx(1/144, abs=1e-14)
    assert not wait["next_observation_equal"]
    assert not result["virtual_relation_equality"]["relations"]
    assert all(len(path) == 8 for path in paths)
    for env in states:
        relations = current_relations(env)
        assert relations[PROCESSING].edge_features.shape[1] == 0
        assert relations[PROCESSING].num_edges == 7
        assert relations[SERVING].num_edges == 0


@pytest.mark.parametrize("mode", MODES)
def test_reachable_worker_alias_hides_individual_commitments(config, fixed_instance, mode):
    instance = worker_alias_instance(fixed_instance)
    # Retain the benchmark's physical machine parameters and worker qualifications.
    assert instance.machines == fixed_instance.machines
    assert [w.qualified_modules for w in instance.workers] == [w.qualified_modules for w in fixed_instance.workers]
    states, paths = worker_histories(effective(config, mode), instance)
    result = pair_comparison(states)
    assert not result["original_equal"]
    assert not result["current_observation_equal"]
    assert result["current_committed_loads"][0] != result["current_committed_loads"][1]
    wait = next(item for item in result["outcomes"] if item["is_wait"])
    assert wait["reward_difference"] == 0
    assert not wait["next_observation_equal"]
    assert not result["virtual_relation_equality"]["relations"]
    for env in states:
        obs = virtual_observation(env)
        recovered = reconstructed_quantities(env, obs)
        assert reconstructed_events(env, obs, recovered) == actual_events(env)
        np.testing.assert_allclose(recovered["committed"], env._committed_worker_loads, atol=1e-5)
        assert recovered["stage_lengths"] == {0: 20, 5: 40}
        serving = current_relations(env)[SERVING]
        assert serving.num_edges == 2 and serving.edge_features.shape == (2, 0)


def test_event_serial_order_is_irrelevant_to_physical_outcomes(config, fixed_instance):
    for result in event_order_checks(config, fixed_instance):
        assert result["original_equal"]
        assert all(item["reward_difference"] == 0 and item["next_observation_equal"] for item in result["outcomes"])


def test_float32_precision_is_distinct_from_structural_alias(config, fixed_instance):
    result = precision_checks(config, fixed_instance)
    assert result["original_equal"] and result["virtual_relation_equal"]
    assert result["cost_rewards"][0] != result["cost_rewards"][1]
    assert 0 < abs(result["scalar_difference"]) < 1e-10
