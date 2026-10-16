import numpy as np
import pytest

from analysis.gae_lambda_diagnostic import estimates, sign_statistics, aggregate
from agent.ppo.buffer import RolloutBuffer, Transition


@pytest.mark.parametrize("gamma", [1.0, 0.9])
def test_matches_production_buffer_and_monte_carlo(gamma):
    rewards, values = [0.1, 0.2, -2.0], [0.6, 0.5, 0.4]
    adv, mc = estimates(rewards, values, gamma=gamma)
    np.testing.assert_allclose(mc, [0.1 + gamma*0.2 - gamma**2*2, 0.2 - gamma*2, -2])
    for lam, expected in adv.items():
        buffer = RolloutBuffer()
        buffer.transitions = [Transition(None, np.array([False]), 0, 0, v, r, i == 2)
                              for i, (r, v) in enumerate(zip(rewards, values))]
        buffer.compute_gae(last_value=1000, gamma=gamma, gae_lambda=lam)
        np.testing.assert_allclose([t.advantage for t in buffer.transitions], expected)
    np.testing.assert_allclose(adv[1], mc-values)


def test_failure_terminal_reward_propagates_and_near_zero_is_explicit():
    a, _ = estimates([0.0]*199 + [-2], [0.5]*200)
    assert a[1][0] == pytest.approx(-2.5)
    assert abs(a[0.95][0]) < abs(a[1][0])
    stats = sign_statistics([0, 1, -1], [1, -1, -1])
    assert stats["nonzero_sign_agreement"] == 0.5
    assert stats["near_zero_fraction"] == pytest.approx(1/3)


def test_choice_only_retains_forced_rewards_and_keeps_episodes_separate():
    episodes = []
    for i, rewards in enumerate(([0.0, 0.1, -2.0], [0.0, 0.1, 0.8])):
        episodes.append({"role": "last", "region": "cost", "gamma": 1,
            "success": i == 1, "rewards": np.array(rewards), "values": np.array([0.6, 0.5, 0.4]),
            "legal_counts": np.array([1, 4, 1]), "metadata": {"instance_id": str(i)}})
    rows, pairs, _ = aggregate(episodes)
    selected = next(r for r in rows if r["region"] == "cost" and r["outcome"] == "failure"
                    and r["scope"] == "choice_steps" and r["lambda"] == 1)
    assert selected["steps"] == 1
    assert selected["rmse"] < 1e-10
    assert selected["advantage_mean"] == pytest.approx(-2.4)


def test_rejects_invalid_or_nonfinite_episode():
    with pytest.raises(ValueError):
        estimates([], [])
    with pytest.raises(ValueError):
        estimates([1], [float("nan")])
