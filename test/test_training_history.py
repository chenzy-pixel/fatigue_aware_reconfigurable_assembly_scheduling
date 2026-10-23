import csv
import json

import pytest
import torch

from configs import load_config
from data import load_dataset_split
from environment import AssemblySchedulingEnv
from agent.baselines import HeuristicPolicy
from agent.comparators.collection import collect
from agent.ppo.parallel import WorkerResponse
from result.training_history import instance_row, aggregate, export_history, save_epoch


def record(success=True, cutoff=False):
    return {"episode": 0, "instance_id": "test", "metadata": {"cell_id": "C01", "scale": "small"},
            "steps": 10, "reward_sum": -1., "budget_truncated": cutoff,
            "epoch": 1, "method": "ppo", "training_seed": 11, "training_steps": 10,
            "metrics": {"flow_excess_objective": 100., "reconfiguration_cost": 20.,
                        "workload_objective_value": 3., "task_succeeded": success,
                        "task_failed": not success and not cutoff, "terminated": not cutoff,
                        "schedule_violations": [], "maximum_worker_fatigue": .5,
                        "safe_fatigue_limit": .8, "decision_count": 10, "time": 70.,
                        "preference_key": "p", "worker_loads": {"W1": 5.}}}


def flat(r):
    return instance_row(r, epoch=r['epoch'], method='ppo', training_seed=11, training_steps=10)


def test_partial_and_failed_objectives_cannot_improve_final_curve():
    complete, partial, failed = record(), record(False, True), record(False)
    partial['metrics']['flow_excess_objective'] = 1.
    failed['metrics']['flow_excess_objective'] = 2.
    rows = list(map(flat, (complete, partial, failed)))
    assert rows[1]['objective_flow'] is None and rows[1]['snapshot_flow'] == 1.
    assert rows[1]['stop_reason'] == 'interaction_budget'
    summary = aggregate(rows)
    assert summary['flow_mean'] == 100. and summary['valid_objective_count'] == 1
    assert summary['decision_count'] == 30 and summary['valid_delivery_rate'] == 1/3
    assert aggregate(rows[1:])['flow_mean'] is None
    assert aggregate(rows[1:])['flow_count'] == 0


@pytest.mark.parametrize('violation', ['schedule', 'fatigue'])
def test_completed_but_unsafe_excluded(violation):
    r = record()
    if violation == 'schedule': r['metrics']['schedule_violations'] = ['overlap']
    else: r['metrics']['maximum_worker_fatigue'] = .9
    assert not flat(r)['objective_valid']


def test_epoch_export_rebuild_and_missing_validation(tmp_path):
    r = record(False, True)
    history = [{'epoch': 1, 'method': 'ppo', 'training_seed': 11, 'training_steps': 10,
                'losses': {'policy_loss': .1}, **aggregate([flat(r)])}]
    save_epoch(tmp_path, history[0], [r])
    export_history(tmp_path, history, [r])
    rows = list(csv.DictReader((tmp_path/'episodes.csv').open(encoding='utf-8-sig')))
    assert rows[0]['objective_flow'] == '' and rows[0]['snapshot_flow'] == '100.0'
    assert json.loads(rows[0]['metric_worker_loads']) == {'W1': 5.}
    assert rows[0]['epoch'] == '1' and rows[0]['episode'] == '1'
    for name, data in [('training_log',history), ('episodes',[r])]:
        (tmp_path/f'{name}.json').write_text(json.dumps(data), encoding='utf-8')
    from scripts.summarize_factorial27 import export_training
    export_training(tmp_path)
    val = {'epoch': 1, 'training_steps': 10, 'summary': {'quality': .2},
           'rows': [{'instance_id': 'v', 'cell_id': 'C01', 'scale': 'small', 'success': True,
                     'preference': [1,0,0], 'objectives': [100,20,3], 'decision_count': 10}]}
    export_history(tmp_path, history, [r], [val])
    curve = list(csv.DictReader((tmp_path/'validation_objectives.csv').open(encoding='utf-8-sig')))[0]
    assert curve['flow_mean'] == '100.0' and curve['epoch'] == '1'


class LocalRunner:
    """Real physics with in-process transport, keeping the logging test small."""
    worker_count = 1

    def __init__(self, config, instance):
        self.env = AssemblySchedulingEnv(config)
        self.instance = instance
        self.calls = []

    def _exchange(self, commands):
        result = {}
        for lane, (command, request) in commands.items():
            self.calls.append(command)
            reward = None
            if command == 'reset_online':
                obs = self.env.reset(self.instance, preference=request.preference)
            elif command == 'step':
                obs, reward, _, _, _ = self.env.step(request.action)
            elif command == 'snapshot':
                result[lane] = WorkerResponse(lane_id=lane, metrics=self.env.metrics())
                continue
            else: raise AssertionError(command)
            result[lane] = WorkerResponse(lane_id=lane, observation=obs,
                action_mask=self.env.get_action_mask(), reward_vector=reward,
                terminated=self.env.terminated, truncated=self.env.truncated,
                metrics=self.env.metrics() if self.env.terminated or self.env.truncated else None,
                instance_id=self.instance.instance_id, metadata={'cell_id': 'C01', 'scale': 'small'},
                environment_step_count=int(command == 'step'))
        return result


@pytest.mark.parametrize('allowance', [1, 10000])
def test_collection_snapshot_preserves_step_budget_and_complete_goals(allowance):
    torch.set_num_threads(1)
    config = load_config('configs/default.json'); config['device'] = 'cpu'
    runner = LocalRunner(config, load_dataset_split(config, 'validation')[0].instance)
    class Heuristic:
        def act_rules(self, observations, masks, explore=False):
            return [HeuristicPolicy().select_action(runner.env)], [0]
        def remember(self, *args): return None
    result = collect(runner, Heuristic(), config, 0, allowance, ddqn=True, episodes_per_batch=1)
    r = result.episodes[0]
    assert result.steps == runner.calls.count('step') == r['metrics']['decision_count']
    assert result.steps <= allowance
    assert (runner.calls.count('snapshot') == 1) == (allowance == 1)
    row = instance_row(r, epoch=1, method='rule_ddqn', training_seed=11, training_steps=result.steps)
    assert row['objective_valid'] == (allowance == 10000)
    assert row['snapshot_flow'] is not None


def test_direct_training_cannot_bypass_budget_manifest():
    from train import _train_single_stage
    with pytest.raises(ValueError, match='budget identity'):
        _train_single_stage({'training': {'budget_manifest': True}}, smoke=False,
                            run_name=None, episodes=1, parallel_envs=1, episodes_per_update=1, initial_checkpoint=None)


def test_direct_smoke_records_epoch_before_validation(tmp_path, monkeypatch):
    import train
    torch.set_num_threads(1)
    config = load_config('configs/default.json'); config['device'] = 'cpu'
    config['paths']['result_root'] = str(tmp_path)
    config['training']['smoke_rollout_steps'] = 2
    config['training']['validation_parallel_envs'] = 1
    class StopBeforeValidation(Exception): pass
    def stop(*args, **kwargs): raise StopBeforeValidation()
    monkeypatch.setattr(train, '_evaluate_policy', stop)
    with pytest.raises(StopBeforeValidation):
        train._train_single_stage(config, smoke=True, run_name='epoch_test', episodes=1,
                                  parallel_envs=1, episodes_per_update=1, initial_checkpoint=None)
    folder = tmp_path/'epoch_test'
    with (folder/'train_log.csv').open(encoding='utf-8-sig') as stream:
        row = next(csv.DictReader(stream))
    assert row['epoch'] == '1' and row['objective_flow'] == ''
    assert row['snapshot_flow'] != '' and row['stop_reason'] == 'episode_step_limit'
    assert (folder/'epochs/epoch_000001.json').exists()
