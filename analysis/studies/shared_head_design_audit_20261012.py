"""Audit historical/current heads, Flow modes, checkpoint choice and failure incentives."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import torch
from analysis.pareto_analysis import hypervolume_3d
from result.provenance import network_weights_sha256

ROOT=Path(__file__).resolve().parents[2]
RUNS=ROOT/'result/runs'
OUT=ROOT/'result/analysis/shared_head_design_audit_20261012'
PREFIX='remaining_flow_experiments_seed11_20261008_225956_'
OLD_SHARED='ablation_structural_shared_head_seed11_20261006_025649_012965'
OLD_EXPERT='v8_universal_seed11_20261005_004756_510'
KEYS=['instance_id','preference_key','sampling_repeat']
OBJ=['flow_time_objective','reconfiguration_cost','worker_load_variance']
PREF=['preference_flow','preference_cost','preference_variance']
SCALE=np.array([1089.15,353.27,2.2629])
SOURCES={Path(__file__)}


def csv(path):
    SOURCES.add(path)
    return pd.read_csv(path).copy()


def js(path):
    SOURCES.add(path)
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(name,rows):
    f=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    f.to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')


def safe(f):
    return f.task_succeeded & f.physical_safety_pass & (f.schedule_violation_count==0)


def quality(f):
    x=f[OBJ].to_numpy();w=f[PREF].to_numpy();y=x/(x+SCALE)*w
    return (y.max(axis=1)+.05*y.sum(axis=1))/1.05


def flatten(d,prefix=''):
    out={}
    for k,v in d.items():
        name=f'{prefix}.{k}' if prefix else k
        if isinstance(v,dict):out.update(flatten(v,name))
        else:out[name]=v
    return out


def serial(value):
    if isinstance(value,dict):return {str(k):serial(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [serial(v) for v in value]
    return value


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    runs={'old_experts':OLD_EXPERT,'old_shared':OLD_SHARED,'new_experts_raw':PREFIX+'main_universal',
          'new_shared_best':PREFIX+'shared_head_universal','new_shared_last':PREFIX+'shared_head_universal'}
    frames={};configs={};identities={};checks=[]
    for role,name in runs.items():
        run=RUNS/name;configs[role]=js(run/'config.json');summary=js(run/'summary.json')
        path=(RUNS/('ablation_eval_'+('universal' if role=='old_experts' else 'shared_head')+'_seed11_20261007_232505_102441')/'instance_metrics.csv'
              if role.startswith('old_') else ROOT/'result/analysis/shared_excess_last_20261012/last_test_instance_metrics.csv'
              if role=='new_shared_last' else run/'final_sampled_instance_metrics.csv')
        f=csv(path).set_index(KEYS).sort_index();assert len(f)==3960 and f.index.is_unique
        assert f.physical_safety_pass.all() and (f.schedule_violation_count==0).all();frames[role]=f
        ck_path=run/('last_checkpoint.pt' if role=='new_shared_last' else 'best_checkpoint.pt');SOURCES.add(ck_path)
        ck=torch.load(ck_path,map_location='cpu',weights_only=False)
        spec=ck['network_spec'];meta=ck['metadata'];prov=summary['provenance']
        assert network_weights_sha256(ck['network'])==meta['network_weights_sha256']
        identities[role]={'run':name,'episode':meta['checkpoint_episode'],'network_spec':serial(spec),
            'parameter_count':sum(t.numel() for t in ck['network'].values()),
            'global_weight_shape':list(ck['network']['global_encoder.0.weight'].shape),
            'network_weights_sha256':meta['network_weights_sha256'],'git':prov['git'],
            'source_state_sha256':prov.get('source_state_sha256'),'reward':configs[role]['reward'],
            'scales':configs[role]['objective_scalarizer']['scales']}
    first=frames['old_shared']
    for role,f in frames.items():
        assert first.index.equals(f.index)
        for field in ['derived_sampling_seed','dataset_manifest_sha256','subset_sha256',*PREF]:
            pd.testing.assert_series_equal(first[field],f[field],check_names=False)
        checks.append({'check':'matched_test_units_and_identity:'+role,'passed':True})
    train_plan=['episode','instance_id','seed','pressure_type','cost_profile','severity','preference_key']
    training={role:csv(RUNS/name/'train_log.csv') for role,name in runs.items() if role!='new_shared_last'}
    training_reward_checks=[]
    for role,f in training.items():
        pd.testing.assert_frame_equal(training['old_shared'][train_plan],f[train_plan])
        checks.append({'check':'matched_training_schedule:'+role,'passed':True})
        reconstructed=f.operation_progress-f.actual_preference_quality_score-f.initial_progress+f.initial_preference_quality_score-2*f.task_failed.astype(float)
        error=float(np.abs(reconstructed-f.reward).max())
        assert error<1e-9 and np.allclose(f.terminal_failure_penalty_applied,2*f.task_failed.astype(float))
        training_reward_checks.append({'role':role,'trajectories':len(f),'maximum_reward_identity_error':error,'failures':int(f.task_failed.sum())})
    save('training_reward_checks',training_reward_checks)
    differences=[]
    for a,b in [('old_experts','old_shared'),('old_shared','new_shared_best'),('new_experts_raw','new_shared_best')]:
        x,y=flatten(configs[a]),flatten(configs[b])
        for key in sorted(x.keys()|y.keys()):
            if x.get(key)!=y.get(key):differences.append({'comparison':a+' -> '+b,'field':key,'before':json.dumps(x.get(key)),'after':json.dumps(y.get(key))})
    save('config_differences',differences)
    overview=[];endpoints=[];volumes=[]
    for role,f in frames.items():
        good=f[safe(f)].copy();good['q_raw']=quality(good)
        request=f.groupby(level=KEYS[:2]).task_succeeded.sum()
        hvs=[]
        for iid,g in good.groupby(level='instance_id'):
            x=g[OBJ].to_numpy();h=hypervolume_3d(x/(x+SCALE));hvs.append(h)
            volumes.append({'role':role,'instance_id':iid,'raw_hv':h})
        overview.append({'role':role,'episode':identities[role]['episode'],'successes':len(good),
            'overall_completion':len(good)/len(f),'worst_preference_completion':f.groupby(level='preference_key').task_succeeded.mean().min(),
            'quality_raw':good.groupby(level='preference_key').q_raw.mean().mean(),'raw_hv':np.mean(hvs),
            'request_coverage':int((request>0).sum()),'all_three_success':int((request==3).sum()),
            'best_of_three_quality_raw':good.groupby(level=KEYS[:2]).q_raw.min().mean()})
        for p,obj in zip(PREF,OBJ):
            group=f[f[p]==1];success=group[safe(group)]
            endpoints.append({'role':role,'objective':obj,'successes':len(success),'requests':len(group),
                'mean':success[obj].mean(),'best_of_three':success.groupby(level='instance_id')[obj].min().mean()})
    save('universal_comparison',overview);save('universal_endpoints',endpoints);save('hypervolume_by_instance',volumes)
    paired=[]
    for a,b in [('old_experts','old_shared'),('old_shared','new_shared_best'),('old_shared','new_shared_last'),('old_experts','new_experts_raw'),('new_experts_raw','new_shared_last')]:
        x,y=frames[a],frames[b];both=safe(x)&safe(y)
        d=pd.Series(quality(y.loc[both])-quality(x.loc[both]),index=x.loc[both].index)
        paired.append({'reference':a,'candidate':b,'common_success':int(both.sum()),
            'gained':int((~safe(x)&safe(y)).sum()),'lost':int((safe(x)&~safe(y)).sum()),
            'both_failed':int((~safe(x)&~safe(y)).sum()),'paired_quality_raw_delta':d.groupby(level='preference_key').mean().mean(),
            'improved_instances':int((d.groupby(level='instance_id').mean()<0).sum()),
            'improved_preferences':int((d.groupby(level='preference_key').mean()<0).sum())})
    save('paired_comparisons',paired)
    x,y=frames['old_shared'],frames['new_shared_last'];both=safe(x)&safe(y)
    d=pd.Series(quality(y.loc[both])-quality(x.loc[both]),index=x.loc[both].index)
    ds=d.groupby(level=KEYS[:2]).sum().unstack().fillna(0);dc=d.groupby(level=KEYS[:2]).size().unstack().fillna(0)
    draws=np.random.default_rng(20261012).multinomial(20,np.full(20,.05),size=10000)
    quality_boot=((draws@ds.to_numpy())/(draws@dc.to_numpy())).mean(axis=1)
    h=pd.DataFrame(volumes).pivot(index='instance_id',columns='role',values='raw_hv').reindex(ds.index)
    hv_delta=h.new_shared_last-h.old_shared;hv_boot=draws@hv_delta.to_numpy()/20
    old_new_uncertainty={'paired_quality_raw_95_interval':np.quantile(quality_boot,[.025,.975]).tolist(),
        'raw_hv_delta_95_interval':np.quantile(hv_boot,[.025,.975]).tolist(),
        'new_hv_better_instances':int((hv_delta>0).sum()),'unit':'whole instance; 10000 resamples; no training-seed uncertainty'}
    pref_comparison=[]
    for key,aa in x.groupby(level='preference_key'):
        bb=y.xs(key,level='preference_key',drop_level=False);matched=safe(aa)&safe(bb)
        weights=aa[PREF].iloc[0].to_numpy()
        group='pure' if weights.max()==1 else 'interior' if weights.min()>0 else 'edge'
        qa=quality(aa.loc[matched]).mean();qb=quality(bb.loc[matched]).mean()
        pref_comparison.append({'preference_key':key,'group':group,**dict(zip(PREF,weights)),
            'old_successes':int(safe(aa).sum()),'new_successes':int(safe(bb).sum()),'common_success':int(matched.sum()),
            'old_matched_quality':qa,'new_matched_quality':qb,'quality_delta':qb-qa,
            **{'old_'+field:aa.loc[matched,field].mean() for field in OBJ},
            **{'new_'+field:bb.loc[matched,field].mean() for field in OBJ}})
    save('old_new_preference_comparison',pref_comparison)
    cost_tail=[];cost_common=both&(x.preference_cost==1)
    for role,f in [('old_shared',x),('new_shared_last',y)]:
        values=f.loc[cost_common,'reconfiguration_cost'];scale=353.27
        cost_tail.append({'role':role,'matched_successes':len(values),'mean':values.mean(),'std':values.std(),
            'median':values.median(),'p90':values.quantile(.9),'p95':values.quantile(.95),'maximum':values.max(),
            'mean_bounded_cost':(values/(scale+values)).mean(),'bounded_mean_cost':values.mean()/(scale+values.mean())})
    save('matched_cost_tail',cost_tail)

    validations={};penalties=[];identity_checks=[];ranking=[];ties=[]
    for role in ['old_shared','new_shared_best','new_experts_raw']:
        v=csv(RUNS/runs[role]/'sampled_validation_instance_metrics.csv');validations[role]=v
        initial=v.initial_progress-v.initial_preference_quality_score
        unpenalized=v.operation_progress-v.actual_preference_quality_score-initial
        error=float(np.abs(v.training_cumulative_reward-(unpenalized-2*v.task_failed.astype(float))).max())
        assert error<1e-9 and np.allclose(v.terminal_failure_penalty_applied,2*v.task_failed.astype(float))
        identity_checks.append({'role':role,'trajectories':len(v),'maximum_reward_identity_error':error,
            'failures':int(v.task_failed.sum()),'penalty_always_applied':True})
        for episode,g in v.groupby('validation_episode'):
            for subset,group in [('all',g),('pure_cost',g[g.preference_cost==1])]:
                base=group.base_cumulative_reward.mean();fail=group.task_failed.mean()
                for penalty in [0,2,5,8.42,10,20,50,120]:
                    penalties.append({'role':role,'episode':episode,'subset':subset,'penalty':penalty,
                        'failure_probability':fail,'base_return':base,'offline_mean_return':base-penalty*fail})
        for penalty in [2,5,10,20]:
            for subset in ['all','pure_cost']:
                eligible=[r for r in penalties if r['role']==role and r['penalty']==penalty and r['subset']==subset]
                ranking.append(max(eligible,key=lambda r:r['offline_mean_return']))
    save('reward_identity_checks',identity_checks);save('penalty_sensitivity',penalties);save('offline_reward_selected_episodes',ranking)
    for stage,first,second in [('validation',validations['new_shared_best'].query('validation_episode==400'),validations['new_shared_best'].query('validation_episode==2000')),
            ('test',frames['new_shared_best'],frames['new_shared_last']),('old_vs_new_last_test',frames['old_shared'],frames['new_shared_last'])]:
        for subset in ['all','pure_cost']:
            a,b=(first,second) if subset=='all' else (first[first.preference_cost==1],second[second.preference_cost==1])
            # Across old/raw and new/excess, common-raw return is a counterfactual score, not the original training reward.
            if stage=='old_vs_new_last_test':
                ba=(a.operation_progress-quality(a)).mean();bb=(b.operation_progress-quality(b)).mean()
            else:ba=a.base_cumulative_reward.mean();bb=b.base_cumulative_reward.mean()
            fa=a.task_failed.mean();fb=b.task_failed.mean();delta=fb-fa
            ties.append({'stage':stage,'subset':subset,'reference_base_return':ba,'candidate_base_return':bb,
                'reference_failure_probability':fa,'candidate_failure_probability':fb,
                'reference_return_at_2':ba-2*fa,'candidate_return_at_2':bb-2*fb,
                'penalty_tie':(bb-ba)/delta if abs(delta)>1e-12 else np.nan,
                'candidate_has_more_failures':bool(delta>0)})
    save('penalty_break_even',ties)
    # Linear global features at successful terminal states: exact values from saved objectives.
    globals_rows=[]
    for role in ['new_experts_raw','new_shared_best','new_shared_last']:
        f=frames[role];g=f[safe(f)];scale=identities[role]['scales']
        for name,field in [('flow','flow_time_objective' if role=='new_experts_raw' else 'flow_excess_objective'),('cost','reconfiguration_cost'),('variance','worker_load_variance')]:
            values=g[field]/scale[name]
            globals_rows.append({'role':role,'objective':name,'count':len(values),'mean_linear_global':values.mean(),
                'p95_linear_global':values.quantile(.95),'maximum_linear_global':values.max(),
                'fraction_above_one':(values>1).mean(),'fraction_above_ten':(values>10).mean()})
    save('terminal_global_feature_scales',globals_rows)
    # Direct same-version, same-mode Flow head comparison plus historical raw/excess evidence.
    singles=[];single_frames={}
    for role,name in [('experts_raw_flow','flow_seed11_20261008_130244'),('experts_excess_flow',PREFIX+'flow'),('shared_excess_flow',PREFIX+'shared_head_flow')]:
        f=csv(RUNS/name/'final_sampled_instance_metrics.csv').set_index(KEYS).sort_index();g=f[safe(f)]
        single_frames[role]=f
        singles.append({'role':role,'successes':len(g),'requests':len(f),'flow_mean':g.flow_time_objective.mean(),
            'flow_best_of_three':g.groupby(level='instance_id').flow_time_objective.min().mean()})
    save('single_flow_comparison',singles)
    x,y=single_frames['experts_excess_flow'],single_frames['shared_excess_flow'];assert x.index.equals(y.index)
    both=safe(x)&safe(y)
    single_head_pair={'common_success':int(both.sum()),'expert_flow':x.loc[both,'flow_time_objective'].mean(),
        'shared_flow':y.loc[both,'flow_time_objective'].mean()}
    result={'identities':identities,'checks':checks,'universal_comparison':overview,'paired':paired,
            'reward_identity':identity_checks,'training_reward_identity':training_reward_checks,'penalty_ties':ties,'single_flow':singles,
            'old_new_uncertainty':old_new_uncertainty,'single_head_pair':single_head_pair,
            'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(SOURCES)}}
    (OUT/'audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(pd.DataFrame(overview).to_string(index=False));print(pd.DataFrame(ties).to_string(index=False))


if __name__=='__main__':main()
