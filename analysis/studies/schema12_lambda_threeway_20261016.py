"""Audit saved lambda .95/.97/.99 runs. No training or new policy rollouts."""
from copy import deepcopy
from itertools import combinations
from pathlib import Path
import hashlib
import json
import subprocess

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

import analysis.studies.schema12_owa_comparison_20261014 as audit
from analysis.studies.schema12_gae099_convergence_20261015 import source_hash
from analysis.pareto_analysis import hypervolume_3d
from agent.ppo.network import infer_checkpoint_network_spec
from result.provenance import effective_config_snapshot, network_weights_sha256, dataset_manifest_snapshot
from utils import derive_evaluation_sampling_seed

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'result/analysis/schema12_lambda_threeway_20261016'
RUNS={
    'lambda095':ROOT/'result/runs/schema12_screen_seed11_20261012_201659_owa_log2',
    'lambda097':ROOT/'result/runs/schema12_screen_seed11_20261015_203938_owa_log2_lambda097',
    'lambda099':ROOT/'result/runs/schema12_screen_seed11_20261014_185917_owa_log2_lambda099',
}
CHECKS=[]
SOURCES={Path(__file__),Path(audit.__file__),ROOT/'analysis/schema12_gae099_convergence_20261015.py',ROOT/'analysis/pareto_analysis.py'}
KEY=audit.KEY

def check(name,condition):
    CHECKS.append({'check':name,'passed':bool(condition)})
    if not condition:raise AssertionError(name)

def js(path):
    SOURCES.add(path)
    return json.loads(path.read_text(encoding='utf-8-sig'))

def csv(path):
    SOURCES.add(path)
    return pd.read_csv(path)

def save(name,rows):
    pd.DataFrame(rows).to_csv(OUT/f'{name}.csv',index=False,encoding='utf-8-sig')

def main():
    OUT.mkdir(exist_ok=True);torch.set_num_threads(2);audit.OUT=OUT
    audit.EVAL.clear()
    for mode,scale in [('excess',368.3143),('raw',1089.15)]:
        audit.EVAL[mode+'_owa']={'fields':['flow_excess_objective' if mode=='excess' else 'flow_time_objective','reconfiguration_cost','owa_q05'],
                               'scales':[scale,353.27,21.03145448655653]}
    configs={};summaries={};tests={};vals={};diags={};updates={};trains={};subsets={};provenance=[]
    overview=[];curves=[];opt=[];conv=[];hvs=[];prefs=[];endpoints=[];milestones=[];lr=[]
    test_hash=dataset_manifest_snapshot(ROOT/'data/manifests/v2/test/manifest.json')['sha256']
    val_hash=dataset_manifest_snapshot(ROOT/'data/manifests/v2/validation/manifest.json')['sha256']
    for model,path in RUNS.items():
        config=js(path/'config.json');s=js(path/'summary.json');configs[model]=config;summaries[model]=s
        train=csv(path/'train_log.csv');u=csv(path/'update_log.csv');log=csv(path/'validation_log.csv')
        trains[model]=train;updates[model]=u;subsets[model]=js(path/'validation_subsets.json')
        check(model+':settings',config['seed']==11 and config['ppo']['gamma']==1 and config['ppo']['gae_lambda']==int(model[-3:])/100)
        check(model+':complete_training',s['episodes']==len(train)==2000 and train.episode.tolist()==list(range(1,2001)) and len(u)==100 and u.episode_count.sum()==2000)
        check(model+':reward',train.reward_identity_error.abs().max()<1e-8 and not train.sampling_truncated.any())
        check(model+':finite_updates',np.isfinite(u[['advantage_std','value_loss','pre_update_explained_variance','clip_fraction','approx_kl']]).all().all())
        check(model+':validation_schedule',log.episode.tolist()==list(range(100,2001,100)) and log.evaluation_complete.all())
        best=int(s['checkpoint_selection']['best_episode'])
        selected=log[log.checkpoint_eligible].sort_values(['completion_rate','preference_balanced_quality_score','episode'],ascending=[False,True,True]).iloc[0]
        check(model+':selection',int(selected.episode)==best)
        p=s['final_sampled']['provenance'];source=source_hash(p['git']['commit'])
        check(model+':source_hash',source['source_state_sha256']==p['source_state_sha256'] and source['python_file_count']==p['source_file_count'])
        check(model+':config_hash',effective_config_snapshot(config)['sha256']==p['effective_config_sha256'])
        provenance.append(dict(model=model,**source))
        for role in ['best','last']:
            file=path/f'{role}_checkpoint.pt';SOURCES.add(file);payload=torch.load(file,map_location='cpu',weights_only=False)
            meta=payload['metadata'];spec=infer_checkpoint_network_spec(payload);weights=network_weights_sha256(payload['network'])
            check(model+':'+role+':identity',meta['checkpoint_episode']==(best if role=='best' else 2000) and weights==meta['network_weights_sha256'] and spec['observation_schema_version']==12 and spec['workload_mode']=='geometric_owa' and spec['workload_q']==.5 and spec['global_objective_encoding']=='log2_v1')
            if role=='best':check(model+':test_checkpoint',hashlib.sha256(file.read_bytes()).hexdigest()==p['checkpoint_sha256'] and weights==p['network_weights_sha256'])
            del payload
        data={stage:audit.read(path/name,model,model+':'+stage) for stage,name in [('test','final_sampled_instance_metrics.csv'),('validation','sampled_validation_instance_metrics.csv'),('diagnostic','sampled_diagnostic_instance_metrics.csv')]}
        t,v,d=data['test'],data['validation'],data['diagnostic'];tests[model]=t;vals[model]=v;diags[model]=d
        check(model+':test_grid',len(t)==3960 and t.instance_id.nunique()==20 and t.preference_key.nunique()==66 and not t.duplicated(KEY).any() and (t.groupby(KEY[:2]).size()==3).all() and s['final_sampled']['evaluation_complete'])
        check(model+':validation_grid',len(v)==39000 and not v.duplicated(['validation_episode',*KEY]).any() and (v.groupby('validation_episode').size()==1950).all())
        check(model+':diagnostic_split',d.instance_id.nunique()==49 and not set(d.instance_id)&set(v.instance_id))
        check(model+':manifest',set(t.dataset_manifest_sha256)=={test_hash}=={p['dataset_manifest_sha256']} and set(v.dataset_manifest_sha256)=={val_hash})
        check(model+':test_seeds',all(int(r.derived_sampling_seed)==derive_evaluation_sampling_seed(int(r.sampling_seed),r.instance_id,r.preference_key) for r in t.itertuples()))
        check(model+':test_failures',int((~audit.safe(t)).sum())==s['final_sampled']['task_failed_count'])
        for stage,f in data.items():
            check(model+':'+stage+':safety',f.physical_safety_pass.all() and (f.schedule_violation_count==0).all() and not f.sampling_truncated.any() and (f.maximum_worker_fatigue<=f.safe_fatigue_limit+1e-9).all())
            check(model+':'+stage+':quality',np.allclose(audit.quality(f[audit.safe(f)],'excess_owa'),f.loc[audit.safe(f),'preference_quality_score'],atol=1e-8))
            check(model+':'+stage+':reward',np.allclose(f.training_cumulative_reward,f.operation_progress-f.initial_progress+f.initial_preference_quality_score-f.actual_preference_quality_score-f.terminal_failure_penalty_applied,atol=1e-8) and np.allclose(f.terminal_failure_penalty_applied,2*f.task_failed.astype(int)))
        for stage,f in [('best_test',t),('best_validation',v[v.validation_episode==best]),('last_validation',v[v.validation_episode==2000]),('last_diagnostic',d[d.validation_episode==2000])]:
            overview.append(dict(model=model,stage=stage,episode=best if stage.startswith('best') else 2000,**audit.aggregate(f)))
            for label,pref in zip(['Flow','Cost','OWA'],audit.PREF):
                endpoints.append(dict(model=model,stage=stage,endpoint=label,**audit.aggregate(f[np.isclose(f[pref],1)])))
        for ep,f in v.groupby('validation_episode'):
            curves.append(dict(model=model,episode=int(ep),**audit.aggregate(f)))
        for phase,f in [('early_1_400',u[u.episode_end<400]),('middle_401_1400',u[(u.episode_end>=400)&(u.episode_end<1400)]),('late_1401_2000',u[u.episode_end>=1400])]:
            opt.append(dict(model=model,phase=phase,updates=len(f),**f[['learning_rate','advantage_std','return_std','pre_update_explained_variance','value_loss','clip_fraction','approx_kl','gradient_norm','gradient_clipped_fraction']].mean().to_dict()))
        for r in log.itertuples():lr.append(dict(model=model,episode=r.episode,learning_rate=r.learning_rate,decays=r.learning_rate_decay_count,event=r.checkpoint_event))
        for coord,target in audit.EVAL.items():
            for iid,f in t.groupby('instance_id'):
                z=f.loc[audit.safe(f),target['fields']].to_numpy();hvs.append(dict(model=model,coordinate=coord,instance_id=iid,hv=hypervolume_3d(z/(z+target['scales']))))
        cells=t[audit.safe(t)].groupby(KEY[:2]).agg({**{p:'first' for p in audit.PREF},**{f:'mean' for f in ['flow_time_objective','reconfiguration_cost','owa_q05']}}).reset_index()
        for iid,f in cells.groupby('instance_id'):
            prefs.append(dict(model=model,instance_id=iid,covered_cells=len(f),flow_span=f.flow_time_objective.max()-f.flow_time_objective.min(),
                              **{name+'_spearman':spearmanr(f[pref],f[target]).statistic for name,pref,target in zip(['flow','cost','owa'],audit.PREF,['flow_time_objective','reconfiguration_cost','owa_q05'])}))
        print(model,'loaded; best',best,'test successes',int(audit.safe(t).sum()),flush=True)
    for model in ['lambda097','lambda099']:
        c=deepcopy(configs[model]);c['experiment_name']=configs['lambda095']['experiment_name'];c['ppo']['gae_lambda']=.95
        check(model+':only_lambda_config_change',effective_config_snapshot(c)['sha256']==effective_config_snapshot(configs['lambda095'])['sha256'])
        check(model+':same_training_inputs',trains[model][['instance_id','seed','preference_key','pressure_type','cost_profile']].equals(trains['lambda095'][['instance_id','seed','preference_key','pressure_type','cost_profile']]))
        check(model+':same_validation_subset',subsets[model]==subsets['lambda095'])
        commit=next(p['commit'] for p in provenance if p['model']==model)
        changed=subprocess.check_output(['git','diff','--name-only',provenance[0]['commit'],commit],text=True).splitlines()
        check(model+':only_launcher_python_changes',set(p for p in changed if p.endswith('.py'))<={'scripts/run_schema12_screening.py','scripts/run_flow_excess_training.py'})
    curve=pd.DataFrame(curves)
    for model,g in curve.groupby('model'):
        f=g[g.episode>=1400]
        conv.append(dict(model=model,points=len(f),quality_mean=f.quality_excess_owa.mean(),quality_std=f.quality_excess_owa.std(ddof=0),overall_completion_mean=f.overall_completion.mean(),worst_completion_mean=f.worst_preference_completion.mean(),worst_completion_min=f.worst_preference_completion.min()))
        for label,mask in [('Q_le_030',g.quality_excess_owa<=.30),('Q_le_028',g.quality_excess_owa<=.28),('worst_ge_098',(g.worst_preference_completion>=.98-1e-10)),('worst_ge_098_and_Q_le_030',(g.worst_preference_completion>=.98-1e-10)&(g.quality_excess_owa<=.30))]:
            f=g[mask];milestones.append(dict(model=model,criterion=label,first_episode=int(f.episode.min()) if len(f) else None,count=len(f)))
    paired=[];counts=[];cis=[]
    hv=pd.DataFrame(hvs)
    for left,right in combinations(RUNS,2):
        for stage,frames in [('best_test',tests),('last_validation',{k:f[f.validation_episode==2000] for k,f in vals.items()}),('last_diagnostic',{k:f[f.validation_episode==2000] for k,f in diags.items()})]:
            a,b=[frames[k].set_index(KEY).sort_index() for k in [left,right]]
            check(left+':'+right+':'+stage+':paired_keys',a.index.equals(b.index) and a.derived_sampling_seed.equals(b.derived_sampling_seed) and np.allclose(a[audit.PREF],b[audit.PREF]))
            mask=audit.safe(a)&audit.safe(b)
            counts.append(dict(left=left,right=right,stage=stage,requests=len(a),common_success=int(mask.sum()),gained=int((~audit.safe(a)&audit.safe(b)).sum()),lost=int((audit.safe(a)&~audit.safe(b)).sum())))
            x,y=a.loc[mask],b.loc[mask]
            fields={f:(x[f],y[f]) for f in audit.PHYSICAL}
            fields['quality_excess_owa']=(audit.quality(x,'excess_owa'),audit.quality(y,'excess_owa'))
            for field,(xx,yy) in fields.items():paired.append(dict(left=left,right=right,stage=stage,metric=field,left_mean=xx.mean(),right_mean=yy.mean(),percent_change=100*(yy.mean()/xx.mean()-1)))
            if stage=='best_test':
                clusters={field:(yy-xx).groupby(level='instance_id').mean() for field,(xx,yy) in fields.items()}
                clusters['completion']=(audit.safe(b).astype(float)-audit.safe(a).astype(float)).groupby(level='instance_id').mean()
                for coord in audit.EVAL:
                    hs=hv[hv.coordinate==coord].pivot(index='instance_id',columns='model',values='hv');clusters['hv_'+coord]=hs[right]-hs[left]
                rng=np.random.default_rng(20261016)
                for field,series in clusters.items():
                    z=series.sort_index().to_numpy();idx=rng.integers(0,len(z),size=(20000,len(z)));lo,hi=np.quantile(z[idx].mean(axis=1),[.025,.975]);cis.append(dict(left=left,right=right,metric=field,mean_delta=z.mean(),ci95_low=lo,ci95_high=hi,clusters=len(z)))
    # Matched cohort across all three for a directly comparable physical-quality table.
    aligned={k:v.set_index(KEY).sort_index() for k,v in tests.items()};mask=np.logical_and.reduce([audit.safe(f) for f in aligned.values()]);common=[]
    for model,f in aligned.items():common.append(dict(model=model,requests=int(mask.sum()),**{field:f.loc[mask,field].mean() for field in audit.PHYSICAL},quality_excess_owa=audit.quality(f.loc[mask],'excess_owa').mean()))
    for name,rows in [('overview',overview),('validation_curve',curves),('optimization_phases',opt),('convergence_phases',conv),('hypervolume_by_instance',hvs),('preference_by_instance',prefs),('endpoints',endpoints),('milestones',milestones),('learning_rate',lr),('paired_metrics',paired),('paired_counts',counts),('bootstrap',cis),('threeway_common_success',common),('checks',CHECKS+audit.CHECKS)]:save(name,rows)
    SOURCES.update(audit.SOURCES)
    result=dict(runs={k:str(p) for k,p in RUNS.items()},checks_passed=len(CHECKS+audit.CHECKS),source_reconstruction=provenance,
                elapsed_hours={k:s['elapsed_seconds']/3600 for k,s in summaries.items()},
                seed23_37_available=[p.name for p in (ROOT/'result/runs').glob('schema12_screen_seed*_owa_log2') if 'seed23_' in p.name or 'seed37_' in p.name],
                training_seeds=[11],new_training=False,new_policy_rollouts=0,
                input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(SOURCES)})
    (OUT/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    plot(curve,updates)
    print(pd.DataFrame(overview)[['model','stage','episode','successes','requests','worst_preference_completion','quality_excess_owa']].to_string(index=False))
    print(hv.groupby(['model','coordinate']).hv.mean().to_string())
    print(pd.DataFrame(conv).to_string(index=False))
    print(pd.DataFrame(common).to_string(index=False))
    print('CHECKS',result['checks_passed'])

def plot(curve,updates):
    colors={'lambda095':'#2563eb','lambda097':'#16a34a','lambda099':'#d97706'}
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for model,f in curve.groupby('model'):
        for ax,field,label in zip(axes.flat,['quality_excess_owa','worst_preference_completion','overall_completion','reconfiguration_cost'],['Quality loss Q (lower better)','Worst preference completion','Overall completion','Mean cost on successful requests']):
            ax.plot(f.episode,f[field],marker='.',label=model,color=colors[model]);ax.set(xlabel='Training episodes',ylabel=label);ax.grid(alpha=.2)
    axes[0,0].legend()
    for ext in ['png','svg']:fig.savefig(OUT/f'convergence.{ext}',dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for model,f in updates.items():
        for ax,field,label in zip(axes.flat,['advantage_std','pre_update_explained_variance','value_loss','clip_fraction'],['Raw advantage std','Pre-update explained variance','Value loss','Clip fraction']):
            ax.plot(f.episode_end+1,f[field].rolling(5,min_periods=1).mean(),label=model,color=colors[model]);ax.set(xlabel='Training episodes',ylabel=label);ax.grid(alpha=.2)
    axes[0,0].legend();axes[1,0].set_yscale('log')
    for ext in ['png','svg']:fig.savefig(OUT/f'optimization.{ext}',dpi=170)
    plt.close(fig)

if __name__=='__main__':main()
