"""Episode stopping, independent validation accounting and request identity."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from configs import load_config
from agent.comparators import learning
from agent.comparators.collection import collect
from agent.comparators.ddqn import RuleDDQN
from agent.comparators.training_budget import training_budget


class Runner:
    def __init__(self, config=None, template=None, episode_count=2000, worker_count=3, **kwargs):
        self.worker_count, self.count = worker_count, episode_count
        self.states, self.resets, self.steps = {}, [], 0

    def __enter__(self): return self
    def __exit__(self, *args): pass

    def _exchange(self, commands):
        replies = {}
        for lane, (command, request) in commands.items():
            if command == 'reset_online':
                e = request.value
                assert e < self.count
                self.states[lane] = [e, 0]
                self.resets.append(e)
            elif command == 'step':
                self.states[lane][1] += 1
                self.steps += 1
            else:
                raise AssertionError('episode collection must not snapshot a live trajectory')
            e, n = self.states[lane]
            ended = n == e % 5 + 1
            truncated = ended and e % 7 == 0
            success = ended and not truncated and e % 11 != 0
            metrics = dict(task_succeeded=success, task_failed=ended and not success and not truncated,
                terminated=ended and not truncated, truncated=truncated, decision_count=n,
                maximum_worker_fatigue=.2, safe_fatigue_limit=.75, flow_excess_objective=10.,
                reconfiguration_cost=20., workload_objective_value=3., time=30.,
                terminal_reason='decision_limit' if truncated else 'completed' if success else 'unrecoverable_deadlock')
            replies[lane] = SimpleNamespace(observation={'episode':e}, action_mask=np.array([False]),
                terminated=ended and not truncated, truncated=truncated,
                environment_step_count=int(command == 'step'), metrics=metrics,
                instance_id=str(e), metadata={}, generation_time_seconds=0., record_sha256=str(e),
                reward_vector=SimpleNamespace(scalarize=lambda _: 0.))
        return replies


class Agent:
    def __init__(self):
        self.updates = self.sample_presentations = self.remembered = 0
        self.batch_epsilon = None
        self.epsilons = []

    set_episode_progress = RuleDDQN.set_episode_progress
    exploration_epsilon = RuleDDQN.exploration_epsilon

    def act_batch(self, observations, masks):
        return [0]*len(masks), [0.]*len(masks), [0.]*len(masks)

    def act_rules(self, observations, masks, explore=False):
        if self.batch_epsilon is not None:
            self.epsilons.append(self.batch_epsilon)
        return [0]*len(masks), [0]*len(masks)

    def remember(self, *args):
        self.remembered += 1
        self.updates += 1
        self.sample_presentations += 1

    def value(self, *args): return 2.
    def update(self, buffer):
        self.updates += 1
        return {'loss': 0.}

    def save(self, path, metadata):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'metadata': metadata}, path)


@pytest.fixture
def config():
    c = load_config('configs/default.json')
    c['device'] = 'cpu'
    return c


@pytest.mark.parametrize('workers', [1, 3, 20])
def test_collection_finishes_each_attempt(config, workers):
    runner, agent = Runner(worker_count=workers), Agent()
    result = collect(runner, agent, config, 9, episodes_per_batch=23)
    assert runner.resets == list(range(9, 32))
    assert [e['episode'] for e in result.episodes] == list(range(9, 32))
    assert result.steps == runner.steps == sum(e['steps'] for e in result.episodes)
    assert all(not e['budget_truncated'] for e in result.episodes)
    assert all(e['metrics']['terminated'] or e['metrics']['truncated'] for e in result.episodes)
    assert any(e['metrics']['truncated'] for e in result.episodes)
    assert any(e['metrics']['task_failed'] for e in result.episodes)
    for e in result.episodes:
        tail = [t for t in result.buffer.transitions if t.observation['episode'] == e['episode']][-1]
        assert tail.return_value == (2. if e['metrics']['truncated'] else 0.)


def test_collection_caps_training_lanes_in_shared_validation_pool(config):
    runner, agent = Runner(worker_count=20), Agent()
    result = collect(runner, agent, config, 0, episodes_per_batch=23, max_parallelism=3)
    assert result.episodes_started == 23
    assert max(runner.states) == 2
    assert result.steps == sum(e['steps'] for e in result.episodes)


def test_collection_progress_uses_completed_attempts_and_real_steps(config):
    runner, agent, progress = Runner(worker_count=3), Agent(), []
    result = collect(runner, agent, config, 0, episodes_per_batch=7, progress_callback=progress.append)
    assert progress[0] == dict(completed=0, total=7, environment_steps=0, active=3)
    assert progress[-1] == dict(completed=7, total=7, environment_steps=result.steps, active=0)
    assert [p['completed'] for p in progress] == sorted(p['completed'] for p in progress)
    assert [p['environment_steps'] for p in progress] == sorted(p['environment_steps'] for p in progress)


def test_training_releases_observations_before_validation_and_next_batch(config, tmp_path, monkeypatch):
    install_training_fakes(monkeypatch)
    original_collect, original_validation = learning.collect, learning.validation
    buffers = []
    def collect_checked(*args, **kwargs):
        assert all(len(buffer) == 0 for buffer in buffers)
        result = original_collect(*args, **kwargs)
        assert len(result.buffer) > 0
        buffers.append(result.buffer)
        return result
    def validation_checked(*args, **kwargs):
        assert buffers and all(len(buffer) == 0 for buffer in buffers)
        return original_validation(*args, **kwargs)
    monkeypatch.setattr(learning, 'collect', collect_checked)
    monkeypatch.setattr(learning, 'validation', validation_checked)
    learning.train_request(config, 'ppo', 11, 1e-4, None, None, [0], [(1.,0.,0.)],
                           tmp_path, 'release-test', episodes=101, workers=3)
    assert len(buffers) == 6 and all(len(buffer) == 0 for buffer in buffers)


def install_training_fakes(monkeypatch):
    agent, calls, saved = Agent(), [], {}
    monkeypatch.setattr(learning, 'ParallelEpisodeRunner', Runner)
    monkeypatch.setattr(learning, 'AssemblySchedulingEnv', lambda c: SimpleNamespace(reset=lambda i: object()))
    monkeypatch.setattr(learning, 'load_instance_yaml', lambda p: object())
    monkeypatch.setattr(learning, 'build_agent', lambda *a: agent)
    monkeypatch.setattr(learning, 'save_epoch', lambda *a: None)
    monkeypatch.setattr(learning, 'export_history', lambda *a: None)
    original = learning.atomic_json
    def save(path, value):
        saved[path.name] = deepcopy(value)
        if path.name not in ('episodes.json', 'training_log.json'):
            original(path, value)
    monkeypatch.setattr(learning, 'atomic_json', save)
    def validation(*args, **kwargs):
        calls.append((agent.updates, agent.remembered, agent.batch_epsilon))
        return {'summary': {'safe': True, 'validation_env_steps': 7, 'validation_seconds': .25},
                'rows': [], 'deliveries': []}, (True, 1., -.1)
    monkeypatch.setattr(learning, 'validation', validation)
    return agent, calls, saved


@pytest.mark.parametrize('method', ['ppo', 'a2c', 'dcan', 'rule_ddqn'])
def test_2000_attempts_100_epochs_20_validations(config, tmp_path, monkeypatch, method):
    agent, calls, saved = install_training_fakes(monkeypatch)
    result = learning.train_request(config, method, 11, 1e-4, None, None, [0], [(1.,0.,0.)],
                                    tmp_path, 'identity', episodes=2000, workers=3)
    assert result['episodes_started'] == 2000 and result['epochs'] == 100
    assert result['validation_count'] == len(calls) == 20
    assert [v['epoch'] for v in saved['validations.json']] == list(range(5,101,5))
    episodes = saved['episodes.json']
    assert [e['episode'] for e in episodes] == list(range(2000))
    assert result['training_steps'] == sum(e['steps'] for e in episodes) == 6000
    assert result['validation_env_steps'] == 140 and result['validation_seconds'] == 5.
    assert result['training_sufficiency'] == 'not_established'
    if method == 'rule_ddqn':
        assert agent.remembered == 6000 and agent.epsilons[0] == 1.
        assert agent.epsilons[-1] == pytest.approx(.05)
    else:
        assert agent.updates == 100
    checkpoint = torch.load(tmp_path/'last_checkpoint.pt', weights_only=False)
    assert checkpoint['metadata']['training_budget'] == training_budget(config)
    changed = deepcopy(config)
    changed['training']['validation_interval_epochs'] = 10
    with pytest.raises(ValueError, match='identity differs'):
        learning.train_request(changed, method, 11, 1e-4, None, None, [0], [(1.,0.,0.)],
                               tmp_path, 'identity', episodes=2000)


@pytest.mark.parametrize('episodes,expected', [(1,[1]), (21,[2]), (101,[5,6])])
def test_tail_batch_and_final_validation(config, tmp_path, monkeypatch, episodes, expected):
    _, _, saved = install_training_fakes(monkeypatch)
    result = learning.train_request(config, 'ppo', 11, 1e-4, None, None, [0], [],
                                    tmp_path, 'tail', episodes=episodes, workers=2)
    assert result['episodes_started'] == episodes
    assert [v['epoch'] for v in saved['validations.json']] == expected


def test_episode_exploration_is_fixed_within_batch():
    a = Agent()
    for e, epsilon in [(0,1.), (500,.525), (1000,.05), (2000,.05)]:
        a.set_episode_progress(e, 2000)
        assert a.exploration_epsilon == pytest.approx(epsilon)
        a.steps = 999999
        assert a.exploration_epsilon == pytest.approx(epsilon)


def test_budget_arguments_and_checkpoint_gate(config, tmp_path):
    from scripts.comparison_suite import check_learning_checkpoint, study_identity
    with pytest.raises(ValueError, match='mutually exclusive'):
        training_budget(config, steps=10, episodes=10)
    original = study_identity(config)
    for key, value in [('episodes',1000), ('episodes_per_update',10), ('validation_interval_epochs',10)]:
        alternate = deepcopy(config); alternate['training'][key] = value
        assert study_identity(alternate) != original
    path = tmp_path/'checkpoint.pt'
    torch.save({'metadata': {'comparison_identity': original, 'algorithm': 'ppo', 'seed':11,
                            'training_budget': training_budget(config)}}, path)
    check_learning_checkpoint(path, config, 'ppo', 11, original)
    with pytest.raises(ValueError, match='identity mismatch'):
        check_learning_checkpoint(path, alternate, 'ppo', 11, original)


def test_formal_cli_uses_episode_config_without_search_manifest(config, tmp_path, monkeypatch):
    import sys
    import scripts.comparison_suite as suite
    import scripts.run_experiments as launcher
    calls = []
    monkeypatch.setattr(suite, 'load_config', lambda _: config)
    monkeypatch.setattr(suite, 'study_identity', lambda _: 'cli_identity')
    monkeypatch.setattr(launcher, 'startup_preflight', lambda *a,**k: {'main': {'study_identity': 'cli_identity'}})
    monkeypatch.setattr(suite, 'train_request', lambda *a,**k: calls.append((a,k)))
    monkeypatch.setattr(suite, 'load_dataset_split', lambda *a: [])
    monkeypatch.setattr(sys, 'argv', ['suite', '--stage','train','--method','dcan','--remote-run',
                                     '--training-workers','7','--output-dir',str(tmp_path)])
    suite.main()
    args, kwargs = calls[0]
    assert args[4] is None and kwargs['episodes'] == 2000 and kwargs['workers'] == 7


@pytest.mark.parametrize('arguments', [
    ['--stage','calibrate-budget','--steps','10','--episodes','2'],
    ['--stage','train','--steps','10'],
    ['--stage','train','--budget-manifest','old.json'],
])
def test_cli_rejects_ambiguous_budget_arguments(arguments, monkeypatch):
    import sys
    from scripts.comparison_suite import main
    monkeypatch.setattr(sys, 'argv', ['suite', *arguments])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2


def test_same_seed_generates_same_cross_method_stream(config):
    from agent.comparators.common import method_config
    from data import OnlineInstanceDataset
    from data.dataset import template_sha256
    from environment import quality_preference_for_episode
    streams = []
    for method in ('ppo','dcan','rule_ddqn'):
        c = method_config(config, method, 23)
        dataset = OnlineInstanceDataset(config=c, episode_count=2000)
        streams.append([(template_sha256(dataset[i].instance),
                         quality_preference_for_episode(c, algorithm_seed=23, quality_episode_index=i).as_dict())
                        for i in (0,26,27,1999)])
    assert streams[0] == streams[1] == streams[2]


def test_worker_exception_leaves_request_incomplete(config, tmp_path, monkeypatch):
    install_training_fakes(monkeypatch)
    def disconnected(*args):
        raise RuntimeError('worker transport disconnected')
    monkeypatch.setattr(Runner, '_exchange', disconnected)
    with pytest.raises(RuntimeError, match='disconnected'):
        learning.train_request(config, 'ppo', 11, 1e-4, None, None, [0], [],
                               tmp_path, 'interrupted', episodes=20)
    assert not (tmp_path/'summary.json').exists()
    assert 'running' in (tmp_path/'status.json').read_text(encoding='utf-8')


def test_incomplete_identity_mismatch_is_preserved(config, tmp_path):
    status = tmp_path/'status.json'
    status.write_text('{"job_id": "other", "status": "running"}', encoding='utf-8')
    original = status.read_bytes()
    with pytest.raises(ValueError, match='incomplete training identity differs'):
        learning.train_request(config, 'ppo', 11, 1e-4, None, None, [0], [],
                               tmp_path, 'current', episodes=20)
    assert status.read_bytes() == original
