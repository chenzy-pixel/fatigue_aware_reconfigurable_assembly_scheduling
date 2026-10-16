"""Offline best-of-three decoding and checkpoint-selection sensitivity."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'result/analysis/schema12_lambda_threeway_20261016/bestof3'
RUNS={
    'lambda095':ROOT/'result/runs/schema12_screen_seed11_20261012_201659_owa_log2',
    'lambda097':ROOT/'result/runs/schema12_screen_seed11_20261015_203938_owa_log2_lambda097',
    'lambda099':ROOT/'result/runs/schema12_screen_seed11_20261014_185917_owa_log2_lambda099',
}
KEY=['instance_id','preference_key']

def aggregate(f):
    good=f[f.success]
    return dict(cells=len(f),successes=int(f.success.sum()),overall_completion=f.success.mean(),
                worst_preference_completion=f.groupby('preference_key').success.mean().min(),
                Q=good.groupby('preference_key').preference_quality_score.mean().mean())

def select(f):
    assert not f.duplicated(KEY+['sampling_repeat']).any()
    assert (f.groupby(KEY).size()==3).all()
    assert f.groupby(KEY).sampling_repeat.apply(lambda x:set(x)=={0,1,2}).all()
    chosen=f.sort_values(['success','preference_quality_score','sampling_repeat'],ascending=[False,True,True]).drop_duplicates(KEY)
    assert chosen.set_index(KEY).success.sort_index().equals(f.groupby(KEY).success.any().sort_index())
    # Reconstruct minimum quality independently for cells having at least one success.
    a=chosen[chosen.success].set_index(KEY).preference_quality_score.sort_index()
    b=f[f.success].groupby(KEY).preference_quality_score.min().sort_index()
    np.testing.assert_allclose(a,b,atol=1e-12)
    return chosen

def main():
    OUT.mkdir(exist_ok=True)
    curves=[];test=[];selection=[];sources={};chosen_rows=[]
    for model,path in RUNS.items():
        summary=json.loads((path/'summary.json').read_text(encoding='utf-8'))
        original_best=summary['checkpoint_selection']['best_episode']
        for stage,name in [('validation','sampled_validation_instance_metrics.csv'),('best_test','final_sampled_instance_metrics.csv')]:
            p=path/name;sources[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
            cols=KEY+['sampling_repeat','validation_episode','task_succeeded','physical_safety_pass','schedule_violation_count','sampling_truncated','preference_quality_score']
            f=pd.read_csv(p,usecols=lambda c:c in cols)
            assert f.physical_safety_pass.all() and (f.schedule_violation_count==0).all() and not f.sampling_truncated.any()
            f['success']=f.task_succeeded & f.physical_safety_pass & (f.schedule_violation_count==0)
            if stage=='validation':
                model_rows=[]
                for ep,g in f.groupby('validation_episode'):
                    assert len(g)==1950
                    chosen=select(g);assert len(chosen)==650
                    row=dict(model=model,episode=int(ep),**aggregate(chosen));curves.append(row);model_rows.append(row)
                    chosen_rows.extend(chosen.assign(model=model,stage=stage).to_dict('records'))
                c=pd.DataFrame(model_rows)
                best=c.sort_values(['worst_preference_completion','Q','episode'],ascending=[False,True,True]).iloc[0]
                old=c[c.episode==original_best].iloc[0]
                selection.append(dict(model=model,original_best_episode=original_best,new_candidate_episode=int(best.episode),
                                      original_best_of3_Q=old.Q,candidate_best_of3_Q=best.Q,
                                      candidate_worst_completion=best.worst_preference_completion,
                                      candidate_successes=int(best.successes),perfect_validations=int((c.worst_preference_completion==1).sum()),
                                      checkpoint_available=int(best.episode) in [original_best,2000]))
            else:
                assert len(f)==3960
                chosen=select(f);assert len(chosen)==1320
                test.append(dict(model=model,episode=original_best,**aggregate(chosen),raw_request_successes=int(f.success.sum()),raw_requests=len(f)))
                chosen_rows.extend(chosen.assign(model=model,stage=stage).to_dict('records'))
    for name,rows in [('validation_curve',curves),('selection_sensitivity',selection),('original_best_test',test),('selected_samples',chosen_rows)]:
        pd.DataFrame(rows).to_csv(OUT/f'{name}.csv',index=False,encoding='utf-8-sig')
    (OUT/'verification.json').write_text(json.dumps(dict(sources=sources,method='success first, minimum native preference Q second, repeat index third',
                                                     all_cells_have_three_distinct_repeats=True,selected_success_and_min_quality_verified=True,
                                                     new_policy_rollouts=0,new_training=0,selection_analysis='post-hoc sensitivity; new candidates not tested'),indent=2),encoding='utf-8')
    print(pd.DataFrame(selection).to_string(index=False))
    print(pd.DataFrame(test).to_string(index=False))

if __name__=='__main__':main()
