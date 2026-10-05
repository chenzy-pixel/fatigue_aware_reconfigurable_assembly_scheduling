"""Nine-global observation, external sampling boundaries and bootstrap contracts."""
from copy import deepcopy
from dataclasses import replace
import pickle

import numpy as np
import pytest
import torch

from agent.baselines import HeuristicPolicy
from agent.ppo import PPOAgent, RolloutBuffer
from agent.ppo.network import build_actor_critic, infer_checkpoint_network_spec
from agent.ppo.parallel import _worker_roll_forward
from agent.ppo.parallel import ParallelEpisodeRunner
from environment import AssemblySchedulingEnv, DecisionType
from environment.observation_schema import GLOBAL_FEATURE_NAMES, OBSERVATION_SCHEMA_VERSION
from result.metrics import aggregate_evaluation_rows, evaluation_selection_key
from training.protocol import LexicographicCheckpointSelector


def test_nine_features_keep_order_information_and_objectives(config, fixed_instance):
    env = AssemblySchedulingEnv(config)
    obs = env.reset(fixed_instance)
    assert OBSERVATION_SCHEMA_VERSION == 10
    assert obs.global_feature_names == GLOBAL_FEATURE_NAMES
    assert len(obs.global_features) == 9
    assert obs.global_features[4] == 0 and obs.global_features[5] == 1
    policy = HeuristicPolicy()
    for _ in range(100):
        env.step(policy.select_action(env), build_observation=False)
        obs = env.observe()
        names = obs.node_feature_names['order']
        active = np.mean(obs.node_features['order'][:, names.index('released')]
                         - obs.node_features['order'][:, names.index('completed')])
        assert active == pytest.approx(obs.action_set_features[
            obs.action_set_feature_names.index('active_order_ratio')])
        assert obs.global_features[3] == float(env.decision_type == DecisionType.PRODUCTION)
        scales = config['objective_scalarizer']['scales']
        np.testing.assert_allclose(obs.global_features[6:9], np.asarray(env._objective_vector()) /
                                   [scales[k] for k in ('flow', 'cost', 'variance')], rtol=1e-6)
        for copied in (obs.copy(), pickle.loads(pickle.dumps(obs))):
            np.testing.assert_array_equal(copied.global_features, obs.global_features)
        if env.task_done:
            break
    net = build_actor_critic(obs, {**config['network'], 'hidden_dim': 16})
    assert net.global_encoder[0].in_features == 9


@pytest.mark.parametrize('guard,reason', [('max_decisions','decision_limit'),
                                        ('max_zero_time_actions','zero_time_action_limit')])
def test_external_guard_preserves_physical_bootstrap_state(config, fixed_instance, guard, reason):
    bounded_cfg = deepcopy(config)
    bounded_cfg['environment'][guard] = 1
    bounded = AssemblySchedulingEnv(bounded_cfg)
    reference = AssemblySchedulingEnv(config)
    before = bounded.reset(fixed_instance)
    reference.reset(fixed_instance)
    action = HeuristicPolicy().select_action(reference)
    expected, expected_reward, _, _, _ = reference.step(action)
    actual, reward, terminated, truncated, _ = bounded.step(action)
    assert not terminated and truncated
    assert bounded.terminal_reason == reason
    assert bounded.task_done and bounded.sampling_truncated
    assert not bounded.task_failed and not bounded.task_succeeded
    assert bounded.decision_type == reference.decision_type
    assert bounded.decision_type != DecisionType.TERMINAL
    assert bounded._flow_penalty == 0 and reward.failure == 0
    assert reward == expected_reward
    assert bounded._objective_vector() == reference._objective_vector()
    assert bounded.schedule_log == reference.schedule_log
    assert bounded.reconfiguration_log == reference.reconfiguration_log
    np.testing.assert_array_equal(actual.global_features, expected.global_features)
    np.testing.assert_array_equal(actual.action_set_features, expected.action_set_features)
    np.testing.assert_array_equal(bounded.get_action_mask(), reference.get_action_mask())
    assert bounded.metrics()['objective_complete'] is False
    with pytest.raises(RuntimeError):
        bounded.step(action)
    buffer = RolloutBuffer(preserve_graph=True)
    buffer.add(before, reference.get_action_mask(), action, 0, 3, reward.scalarize(config['reward']), done=terminated)
    buffer.compute_gae(last_value=7, gamma=0.9, gae_lambda=0.95)
    assert buffer.transitions[-1].return_value == pytest.approx(reward.scalarize(config['reward']) + 0.9*7)


def test_worker_external_guard_returns_observation_and_mask(config, fixed_instance):
    settings = deepcopy(config)
    settings['environment']['max_decisions'] = 1
    env = AssemblySchedulingEnv(settings)
    obs = env.reset(fixed_instance)
    response = _worker_roll_forward(0, env, obs, preserve_graph=True,
        requested_action=HeuristicPolicy().select_action(env), drain_physical_forced_actions=True,
        max_environment_steps=None)
    assert response.truncated and not response.terminated
    assert response.observation is not None and response.action_mask is not None
    assert response.metrics['sampling_truncated'] and not response.metrics['task_failed']


def test_guard_precedence_completion_and_failure(config, fixed_instance):
    settings = deepcopy(config)
    settings['environment'].update(max_decisions=1, max_zero_time_actions=1)
    env = AssemblySchedulingEnv(settings)
    env.reset(fixed_instance)
    env.step(HeuristicPolicy().select_action(env))
    assert env.terminal_reason == 'decision_limit'
    metrics = env.metrics()
    assert metrics['decision_count'] == metrics['zero_time_action_count'] == 1
    # A real horizon boundary takes precedence over the external guard.
    reference = AssemblySchedulingEnv(config)
    reference.reset(fixed_instance)
    policy = HeuristicPolicy()
    while reference.current_tick == 0:
        reference.step(policy.select_action(reference), build_observation=False)
    cfg = deepcopy(config)
    cfg['environment']['max_decisions'] = reference._decision_count
    short = replace(fixed_instance, horizon=reference.current_time)
    env = AssemblySchedulingEnv(cfg)
    env.reset(short)
    while not env.task_done:
        _, reward, _, _, _ = env.step(policy.select_action(env), build_observation=False)
    assert env.terminated and not env.truncated and env.task_failed
    assert env.terminal_reason == 'horizon'
    assert reward.failure == -config['reward']['terminal_failure_penalty']


@pytest.mark.parametrize('schema', [5,6,7,8,9])
@pytest.mark.parametrize('allow', [False,True])
def test_old_checkpoints_rejected_even_with_migration_flag(config, fixed_instance, tmp_path, schema, allow):
    obs = AssemblySchedulingEnv(config).reset(fixed_instance)
    agent = PPOAgent(build_actor_critic(obs, {**config['network'], 'hidden_dim':16}), config['ppo'])
    path = tmp_path/'old.pt'
    agent.save(path)
    payload = torch.load(path, weights_only=False)
    payload['network_spec']['observation_schema_version'] = schema
    torch.save(payload,path)
    with pytest.raises(ValueError, match='requires retraining'):
        agent.load(path, allow_observation_migration=allow, load_optimizer=True)
    with pytest.raises(ValueError, match='requires retraining'):
        infer_checkpoint_network_spec(payload)


def test_incomplete_evaluation_cannot_select_or_report_formal_quality(config):
    row = dict(terminated=False, truncated=True, sampling_truncated=True,
               task_succeeded=False, task_failed=False, schedule_violation_count=0,
               decisions=1, inference_time_seconds=0, solve_time_seconds=0)
    result = aggregate_evaluation_rows([row], dataset='test', policy='ppo', manifest='test')
    assert not result['evaluation_complete'] and result['completion_coverage'] == 0
    assert result['completion_rate'] is None
    assert result['completed_metrics'] == {} and result['all_instance_metrics'] == {}
    assert result['preference_balanced_quality_score'] is None
    with pytest.raises(ValueError, match='incomplete evaluation'):
        evaluation_selection_key(result)
    selector = LexicographicCheckpointSelector.from_config(config)
    assert selector.observe(result, completed_episodes=1, physical_safety_pass=True) == 'ineligible'
    assert not selector.has_best


def test_guard_preserves_committed_reconfiguration(config, fixed_instance):
    settings = deepcopy(config)
    reference = AssemblySchedulingEnv(config)
    bounded = AssemblySchedulingEnv(settings)
    reference.reset(fixed_instance)
    bounded.reset(fixed_instance)
    rule = HeuristicPolicy()
    for _ in range(100):
        action = rule.select_action(reference)
        starting_worker = reference.decision_type == DecisionType.WORKER and action != reference.wait_action
        if starting_worker:
            bounded.config['environment']['max_decisions'] = bounded._decision_count + 1
        expected, expected_reward, _, _, _ = reference.step(action)
        actual, reward, _, _, _ = bounded.step(action)
        if starting_worker:
            assert bounded.sampling_truncated and not bounded.task_failed
            assert bounded._active_committed_worker_tasks == reference._active_committed_worker_tasks
            np.testing.assert_array_equal(bounded._committed_worker_loads, reference._committed_worker_loads)
            assert bounded.reconfiguration_log == reference.reconfiguration_log
            assert bounded._flow_penalty == 0 and reward == expected_reward
            np.testing.assert_array_equal(actual.global_features, expected.global_features)
            np.testing.assert_array_equal(actual.action_set_features, expected.action_set_features)
            return
    pytest.fail('no worker commitment reached')


@pytest.mark.parametrize('compression', [False, True])
def test_parallel_engineering_cutoff_bootstraps_without_cross_episode_gae(config, fixed_instance, monkeypatch, compression):
    settings = deepcopy(config)
    settings['device'] = 'cpu'
    settings['environment']['max_decisions'] = 3
    settings['training'].update(forced_action_compression=compression, worker_timeout_seconds=120)
    obs = AssemblySchedulingEnv(settings).reset(fixed_instance)
    agent = PPOAgent(build_actor_critic(obs, {**settings['network'], 'hidden_dim':16}), settings['ppo'])
    monkeypatch.setattr(agent, 'value_batch', lambda observations,masks: [7.0]*len(observations))
    with ParallelEpisodeRunner(config=settings,template=fixed_instance,episode_count=2,worker_count=2) as runner:
        result = runner.collect_training_batch(agent,[0,1],gamma=1,gae_lambda=0.95)
    assert len(result.episodes) == 2
    saw_policy = False
    for episode in result.episodes:
        assert episode.metrics['sampling_truncated'] and not episode.metrics['task_failed']
        assert episode.step_count == 3
        assert episode.reward_components['failure'] == 0
        assert episode.reward_sum == pytest.approx(episode.expected_reward)
        if episode.buffer.transitions:
            saw_policy = True
            last = episode.buffer.transitions[-1]
            assert not last.done
            assert last.return_value == pytest.approx(last.reward + 7.0)
    assert saw_policy


def test_generation_precheck_does_not_use_training_sampling_guards(config, fixed_instance):
    from data.generate_orders import _rollout_metrics
    capped = deepcopy(config)
    capped['environment'].update(max_decisions=1, max_zero_time_actions=1)
    normal, _ = _rollout_metrics(fixed_instance, config)
    limited, _ = _rollout_metrics(fixed_instance, capped)
    assert normal == limited
    assert limited['heuristic_completed'] and not limited['heuristic_failed']
    assert not limited['heuristic_truncated']


@pytest.mark.parametrize('corruption', ['global_width', 'global_order', 'global_nonfinite', 'relation'])
def test_rollout_buffer_validates_current_graph_contract(config, fixed_instance, corruption):
    observation = AssemblySchedulingEnv(config).reset(fixed_instance)
    invalid = observation.copy()
    if corruption == 'global_width':
        invalid = replace(invalid, global_features=invalid.global_features[:-1])
    elif corruption == 'global_order':
        invalid = replace(invalid, global_feature_names=tuple(reversed(GLOBAL_FEATURE_NAMES)))
    elif corruption == 'global_nonfinite':
        invalid.global_features[-1] = np.nan
    else:
        invalid.relations.pop(next(iter(invalid.relations)))
    buffer = RolloutBuffer()
    with pytest.raises(ValueError, match='(global feature|graph relations)'):
        buffer.add(invalid, np.array([False]), 0, 0.0, 0.0, 0.0, False)
    assert not buffer.transitions
