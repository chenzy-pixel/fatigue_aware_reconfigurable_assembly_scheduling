"""Check vanilla targets, real replay updates, and supplemental training scope."""
from copy import deepcopy
import contextlib
import io
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from agent.comparators.common import method_config
from agent.comparators.ddqn import RuleDDQN, double_q_targets
from agent.comparators.dqn import RuleDQN, dqn_targets
from agent.comparators.learning import build_agent
from configs import load_config
from data import load_dataset_split
from environment import AssemblySchedulingEnv
from scripts import run_dqn_a2c_training as launcher


def test_dqn_target_uses_target_max_and_masks_terminal_and_unavailable_actions():
    online = torch.tensor([[9., 1., 2.]] * 4)
    target = torch.tensor([[3., 8., 100.]] * 4)
    mask = torch.tensor([[False, False, True], [False, False, True],
                         [True, True, True], [False, False, True]])
    done = torch.tensor([False, True, False, False])
    reward = torch.tensor([1., 2., 3., 4.])
    assert torch.equal(dqn_targets(reward, done, target, mask, .5), torch.tensor([5., 2., 3., 8.]))
    assert double_q_targets(reward, done, online, target, mask, .5)[0] == 2.5


def test_dqn_real_replay_update_and_checkpoint_family(tmp_path):
    torch.set_num_threads(1)
    config = method_config(load_config('configs/default.json'), 'dqn', 11)
    config['device'] = 'cpu'
    config['comparison']['ddqn'] = {'warmup': 2, 'batch_size': 2, 'update_every': 1}
    env = AssemblySchedulingEnv(config)
    obs = env.reset(load_dataset_split(config, 'validation')[0].instance)
    agent = build_agent(config, obs, 'dqn')
    assert isinstance(agent, RuleDQN)
    before = deepcopy(agent.online.state_dict())
    for _ in range(3):
        action, rule = agent.act_rules([obs], [env.get_action_mask()])
        following, reward, done, _, _ = env.step(action[0])
        agent.remember(obs, rule[0], reward.scalarize(config['reward']), following, env.get_action_mask(), done)
        obs = following
    assert agent.updates >= 1
    assert any(not torch.equal(before[k], v) for k, v in agent.online.state_dict().items())
    path = tmp_path/'dqn.pt'
    agent.save(path)
    clone = RuleDQN(obs, config)
    assert clone.load(path)['algorithm'] == 'dqn'
    with pytest.raises(ValueError, match='family/spec mismatch'):
        RuleDDQN(obs, config).load(path)
    assert config['comparison']['delivery_protocol'] == 'greedy_single_v1'
    assert config['runtime_manifest']['encoder_variant'] == 'pooled_rule_state_v1'


def test_dqn_uses_off_policy_collection_for_full_episode_budget(tmp_path, monkeypatch):
    from test.test_episode_training import install_training_fakes
    from agent.comparators import learning
    agent, validations, saved = install_training_fakes(monkeypatch)
    config = load_config('configs/default.json')
    config['device'] = 'cpu'
    result = learning.train_request(config, 'dqn', 11, 1e-4, None, [0], [],
                                    tmp_path, 'dqn-test', episodes=2000, workers=3)
    assert result['method'] == 'dqn'
    assert result['episodes_started'] == 2000 and result['epochs'] == 100
    assert len(validations) == result['validation_count'] == 20
    assert agent.remembered == 6000 and agent.epsilons[-1] == pytest.approx(.05)


def fixture_project(tmp_path):
    project = tmp_path/'project with spaces'
    for name in launcher.DQN_OVERLAYS:
        path = project/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# source\n', encoding='utf-8')
    for root in ('experiment_suite_v1', 'experiment_suite_v1_parallel20'):
        output = project/'result'/root
        output.mkdir(parents=True)
        for name in ('preflight.json', 'source.zip'):
            (output/name).write_text('fixture', encoding='utf-8')
    return project


def test_launcher_selects_exact_seeds_and_training_only(tmp_path):
    project = fixture_project(tmp_path)
    frozen = tmp_path/'frozen'
    frozen.mkdir()
    (frozen/'result').mkdir()
    (frozen/'result/console_reporter.py').write_text('# frozen reporter\n', encoding='utf-8')
    with patch('scripts.resume_full_evaluation.prepare', return_value=(frozen, 'identity', [11,23,37,53,71])) as prepare, \
            patch('scripts.resume_full_evaluation.read', return_value={'configs': {'full': {'device': 'cuda'}}}), \
            patch.object(launcher, 'extend_dqn_runtime'), \
            patch.object(launcher.subprocess, 'run') as run, contextlib.redirect_stdout(io.StringIO()):
        launcher.main(['--project-root', str(project)])
    assert [call.args[2] for call in prepare.call_args_list] == [['rule_ddqn'], ['a2c']]
    assert all(call.kwargs == {'training': True} for call in prepare.call_args_list)
    commands = [call.args[0] for call in run.call_args_list]
    dqn, a2c = commands[-2:]
    assert dqn[dqn.index('--seeds')+1:] == ['11', '23', '37', '71']
    assert a2c[a2c.index('--seeds')+1:] == ['71', '--remote-run']
    assert a2c[a2c.index('--stage')+1] == 'train'
    assert a2c[a2c.index('--arms')+1] == 'a2c'


def test_missing_second_input_stops_before_any_preparation_or_launch(tmp_path):
    project = fixture_project(tmp_path)
    (project/'result/experiment_suite_v1/source.zip').unlink()
    with patch('scripts.resume_full_evaluation.prepare') as prepare, patch.object(launcher.subprocess, 'run') as run:
        with pytest.raises(SystemExit):
            launcher.main(['--project-root', str(project)])
    prepare.assert_not_called()
    run.assert_not_called()


def test_a2c_only_needs_its_own_inputs_and_preserves_frozen_source(tmp_path):
    project = fixture_project(tmp_path)
    (project/'result/experiment_suite_v1_parallel20/source.zip').unlink()
    frozen = tmp_path/'frozen'
    frozen.mkdir()
    with patch('scripts.resume_full_evaluation.prepare', return_value=(frozen, 'identity', [11,23,37,53,71])), \
            patch('scripts.resume_full_evaluation.read', return_value={'configs': {'full': {'device': 'cuda'}}}), \
            patch.object(launcher.subprocess, 'run') as run, patch.object(launcher.shutil, 'copyfile') as copy, \
            contextlib.redirect_stdout(io.StringIO()):
        launcher.main(['--project-root', str(project), '--methods', 'a2c'])
    assert run.call_count == 2
    copy.assert_not_called()


def test_training_preparation_accepts_a2c_without_completed_checkpoints():
    from test.test_resume_full_evaluation import FrozenEvaluationTests
    from scripts.resume_full_evaluation import prepare
    fixture = FrozenEvaluationTests('test_restores_missing_json_presets_and_exact_source_bytes')
    fixture.setUp()
    try:
        fixture.payload['suite']['arms']['a2c'] = {'method': 'a2c', 'network': {}}
        fixture.payload['configs']['a2c'] = deepcopy(fixture.payload['configs']['full'])
        fixture.write_snapshot()
        frozen, identity, seeds = prepare(fixture.project, fixture.output, ['a2c'], training=True)
        assert frozen.is_dir() and seeds == [11, 23, 37, 53, 71]
        assert not (fixture.output/'training/a2c').exists()
        with pytest.raises(ValueError, match='Select saved model arms'):
            prepare(fixture.project, fixture.output, ['a2c'])
    finally:
        fixture.doCleanups()


def test_supplemental_trainer_accepts_legacy_explicit_steps_argument(monkeypatch, tmp_path):
    from scripts.train_dqn import train_seed
    from agent.comparators import learning
    def legacy(base, method, seed, rate, steps, dataset, val_indices, prefs, output, identity, *, workers, episodes):
        assert steps is None and episodes == 2000
        assert method == 'dqn' and seed == 71 and workers == 20
        assert dataset == 'data' and val_indices == [0] and prefs == ['pref']
        assert output == tmp_path and identity == 'identity'
        return 'trained'
    monkeypatch.setattr(learning, 'train_request', legacy)
    assert train_seed({'ppo': {'learning_rate': 1e-4}}, 71, 'data', [0], ['pref'], tmp_path,
                      'identity', 20) == 'trained'
