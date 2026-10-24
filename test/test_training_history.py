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
    from result.training_history import export_training
    export_training(tmp_path)
    val = {'epoch': 1, 'training_steps': 10, 'summary': {'quality': .2},
           'rows': [{'instance_id': 'v', 'cell_id': 'C01', 'scale': 'small', 'success': True,
                     'preference': [1,0,0], 'objectives': [100,20,3], 'decision_count': 10}]}
    export_history(tmp_path, history, [r], [val])
    curve = list(csv.DictReader((tmp_path/'validation_objectives.csv').open(encoding='utf-8-sig')))[0]
    assert curve['flow_mean'] == '100.0' and curve['epoch'] == '1'
