"""Reaggregate archived three-lambda records through the production entrypoint."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pandas as pd

from configs import load_config
from configs.runtime import attach_runtime_manifest
from result.provenance import effective_config_snapshot
from result.io import write_json, write_csv
from train import _aggregate_formal_rows
from training.protocol import LexicographicCheckpointSelector

ROOT=Path(__file__).resolve().parents[1]
RUNS={
    'lambda095': ('schema12_screen_seed11_20261012_201659_owa_log2',1800,.262400),
    'lambda097': ('schema12_screen_seed11_20261015_203938_owa_log2_lambda097',1900,.324509),
    'lambda099': ('schema12_screen_seed11_20261014_185917_owa_log2_lambda099',1400,.313682),
}


def records(frame):
    return frame.astype(object).where(pd.notna(frame),None).to_dict('records')


def main():
    out=ROOT/'result/analysis/best_of_three_protocol_verification'
    out.mkdir(exist_ok=True)
    summaries=[];curves=[];hashes={}
    for label,(name,expected_episode,expected_q) in RUNS.items():
        directory=ROOT/'result/runs'/name
        original=load_config(directory/'config.json')
        archived=json.loads((directory/'config.json').read_text(encoding='utf-8'))
        assert effective_config_snapshot(original)['sha256']==effective_config_snapshot(archived)['sha256']
        config=deepcopy(original);config.pop('runtime_manifest')
        config['training']['formal_evaluation']['sample_aggregation']='best_of_3_v1'
        attach_runtime_manifest(config)
        selector=LexicographicCheckpointSelector.from_config(config)
        for frame in pd.read_csv(directory/'sampled_validation_instance_metrics.csv',chunksize=1950):
            assert frame.validation_episode.nunique()==1
            ep=int(frame.validation_episode.iloc[0]);rows=records(frame)
            a=_aggregate_formal_rows(config,rows=rows,dataset_name='validation',manifest='archived',
                                     unique_instance_count=50,repeat_count=3,universal=True,strict_counts=True)
            selector.observe(a,completed_episodes=ep,physical_safety_pass=a['active_constraint_pass'])
            curves.append(dict(model=label,episode=ep,completion_rate=a['completion_rate'],Q=a['preference_balanced_quality_score']))
        assert selector.best_episode==expected_episode,(label,selector.best_episode)
        rows=records(pd.read_csv(directory/'final_sampled_instance_metrics.csv'))
        a=_aggregate_formal_rows(config,rows=rows,dataset_name='test',manifest='archived',
                                 unique_instance_count=20,repeat_count=3,universal=True,stage='final_test',strict_counts=True)
        assert a['delivery_count']==a['completed_cell_count']==1320
        assert abs(a['preference_balanced_quality_score']-expected_q)<5e-7
        assert a['request_count']==3960
        assert sum(a['success_count_distribution'].values())==1320
        assert a['total_solve_time_seconds']==a['raw_sampled']['total_solve_time_seconds']
        summaries.append(dict(model=label,selected_episode=selector.best_episode,original_best_test_Q=a['preference_balanced_quality_score'],
                              deliveries=a['delivery_count'],successes=a['completed_cell_count'],historical_config_hash_preserved=True))
        for file in ['config.json','sampled_validation_instance_metrics.csv','final_sampled_instance_metrics.csv']:
            path=directory/file;hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        print(summaries[-1],flush=True)
    write_csv(out/'summary.csv',summaries);write_csv(out/'validation_curve.csv',curves)
    write_json(out/'verification.json',dict(inputs=hashes,results=summaries,new_policy_rollouts=0,new_training=0,
                                          production_entrypoint='_aggregate_formal_rows',
                                          candidate_checkpoints_tested=False))


if __name__=='__main__':main()
