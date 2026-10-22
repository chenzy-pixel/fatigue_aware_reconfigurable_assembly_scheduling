"""Read-only audit of completed Universal runs; reproduce from saved raw rows."""
from pathlib import Path
import sys, json, hashlib, math
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from train import _aggregate_formal_rows
from training.protocol import LexicographicCheckpointSelector
from result.provenance import network_weights_sha256, effective_config_snapshot

OUT = Path(__file__).parent
RUNS = sorted((ROOT/'result/runs').glob('experiment_seed*_01_mixed_instances'))
KEY = ['instance_id', 'preference_key', 'sampling_repeat']
OBJ = ['flow_excess_objective', 'reconfiguration_cost', 'workload_objective_value']
PREF = ['preference_flow', 'preference_cost', 'preference_workload']
checks, hashes, summaries, curves, diagnostics, trains, updates, finals = [], {}, [], [], [], [], [], []
windows, prefs, pressures, failures, endpoints, repeat_effects, checkpoints = [], [], [], [], [], [], []
configs, source_states, subsets = [], [], []

def check(name, ok):
    checks.append({'check': name, 'passed': bool(ok)})
    if not ok:
        raise AssertionError(name)

def sha(path):
    d=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): d.update(block)
    return d.hexdigest()

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def records(frame):
    return frame.astype(object).where(pd.notna(frame), None).to_dict('records')

def close(a,b):
    return math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-11)

def successful(df):
    return df.task_succeeded & ~df.sampling_truncated & df.schedule_violation_count.eq(0) & (df.maximum_worker_fatigue <= df.safe_fatigue_limit + 1e-9)

def verify_rows(raw, chosen, config, n, npref, label):
    check(label+': count and uniqueness',len(raw)==n*npref*3 and not raw.duplicated(KEY).any())
    check(label+': full cells',raw.groupby(['instance_id','preference_key']).sampling_repeat.agg(set).map(lambda x:x=={0,1,2}).all())
    raw=raw.copy();raw['_success']=successful(raw)
    check(label+': no incomplete requests',not raw.sampling_truncated.any())
    check(label+': physical safety',raw.physical_safety_pass.all() and raw.active_constraint_pass.all() and raw.schedule_violation_count.eq(0).all())
    vals=raw[OBJ].to_numpy();scales=np.array([config['objective_scalarizer']['scales'][k] for k in ['flow','cost','workload']])
    weighted=vals/(vals+scales)*raw[PREF].to_numpy();rho=config['objective_scalarizer']['rho']
    q=(weighted.max(axis=1)+rho*weighted.sum(axis=1))/(1+rho)
    check(label+': native quality formula',np.allclose(q,raw.actual_preference_quality_score,rtol=1e-10,atol=1e-12))
    check(label+': terminal failure score',np.allclose(np.where(raw.task_succeeded,q,1.0),raw.preference_quality_score,rtol=1e-10,atol=1e-12))
    expected=raw.sort_values(['_success','preference_quality_score','sampling_repeat'],ascending=[False,True,True]).drop_duplicates(['instance_id','preference_key'])
    check(label+': chosen rows',set(map(tuple,expected[KEY].to_numpy()))==set(map(tuple,chosen[KEY].to_numpy())))
    check(label+': stored flags',set(map(tuple,raw[raw.selected_for_aggregation][KEY].to_numpy()))==set(map(tuple,expected[KEY].to_numpy())))
    a=expected.set_index(KEY).sort_index();b=chosen.set_index(KEY).sort_index()
    check(label+': saved chosen contents',np.allclose(a[OBJ+['preference_quality_score']],b[OBJ+['preference_quality_score']],rtol=1e-11,atol=1e-11) and a.action_trace_sha256.equals(b.action_trace_sha256))
    return raw

for p in RUNS:
    c=read(p/'config.json');s=read(p/'summary.json');seed=c['seed'];label=f'seed{seed}'
    print(label,'start',flush=True)
    config_no_seed={k:v for k,v in c.items() if k!='seed'};configs.append(config_no_seed)
    source_states.append(s['provenance']['source_state_sha256']);subsets.append(read(p/'validation_subsets.json'))
    check(label+': mainline',c['runtime_manifest']['observation_schema']==12 and c['runtime_manifest']['global_objective_encoding']=='log2_v1' and c['ppo']['gamma']==1 and c['ppo']['gae_lambda']==.95 and c['workload_objective']['q']==.5 and c['training']['formal_evaluation']['sample_aggregation']=='best_of_3_v1')
    check(label+': saved config hash',effective_config_snapshot(c)['sha256']==s['provenance']['effective_config_sha256'])
    for name in ['config.json','summary.json','validation_subsets.json','train_log.csv','update_log.csv','validation_log.csv','diagnostic_log.csv','final_sampled_instance_metrics.csv','final_selected_instance_metrics.csv','sampled_validation_instance_metrics.csv','selected_validation_instance_metrics.csv','sampled_diagnostic_instance_metrics.csv','selected_diagnostic_instance_metrics.csv','best_checkpoint.pt','last_checkpoint.pt']:
        hashes[str((p/name).relative_to(ROOT))]=sha(p/name)
    for role in ['best','last']:
        ck=torch.load(p/f'{role}_checkpoint.pt',map_location='cpu',weights_only=False);meta=ck['metadata'];wh=network_weights_sha256(ck['network'])
        ep=s['checkpoint_selection']['best_episode'] if role=='best' else 2000
        check(label+': '+role+' checkpoint',meta['checkpoint_episode']==ep and meta['algorithm_seed']==seed and meta['sampling_budget']==3 and meta['network_weights_sha256']==wh)
        check(label+': '+role+' finite weights',all(torch.isfinite(t).all().item() for t in ck['network'].values()))
        if role=='best':
            check(label+': evaluated best checkpoint',s['final_sampled']['provenance']['checkpoint_sha256']==hashes[str((p/f'{role}_checkpoint.pt').relative_to(ROOT))] and s['final_sampled']['provenance']['network_weights_sha256']==wh)
        checkpoints.append(dict(seed=seed,role=role,episode=ep,network_weights_sha256=wh))
    t=pd.read_csv(p/'train_log.csv');u=pd.read_csv(p/'update_log.csv');v=pd.read_csv(p/'validation_log.csv');d=pd.read_csv(p/'diagnostic_log.csv')
    check(label+': completed budget',s['episodes']==2000 and t.episode.tolist()==list(range(1,2001)) and len(u)==100 and u.episode_count.sum()==2000 and u.sampling_truncated_count.sum()==0)
    check(label+': validation schedule',v.episode.tolist()==list(range(100,2001,100)) and d.episode.tolist()==[500,1000,1500,2000])
    check(label+': finite optimizer metrics',np.isfinite(u[['value_loss','policy_loss','entropy','approx_kl','gradient_norm','pre_update_explained_variance']]).all().all())
    selector=LexicographicCheckpointSelector.from_config(c)
    for stage, logfile, rawfile, chosenfile, n in [('validation',v,'sampled_validation_instance_metrics.csv','selected_validation_instance_metrics.csv',50),('diagnostic',d,'sampled_diagnostic_instance_metrics.csv','selected_diagnostic_instance_metrics.csv',49)]:
        selected=pd.read_csv(p/chosenfile)
        chunks=list(pd.read_csv(p/rawfile,chunksize=n*13*3))
        check(label+': '+stage+' chunk count',len(chunks)==len(logfile))
        for frame,(_,log) in zip(chunks,logfile.iterrows()):
            ep=int(log.episode);tag=f'{label}:{stage}:{ep}'
            check(tag+': episode',frame.validation_episode.nunique()==1 and frame.validation_episode.iloc[0]==ep)
            chosen=selected[selected.validation_episode==ep]
            verify_rows(frame,chosen,c,n,13,tag)
            agg=_aggregate_formal_rows(c,rows=records(frame),dataset_name='validation',manifest='archived',unique_instance_count=n,repeat_count=3,universal=True,strict_counts=True)
            for field in ['completion_rate','overall_completion_rate','preference_balanced_quality_score','raw_overall_completion_rate','raw_preference_balanced_quality_score']:
                check(tag+': '+field,close(agg[field],log[field]))
            if stage=='validation':
                event=selector.observe(agg,completed_episodes=ep,physical_safety_pass=agg['active_constraint_pass'])
                check(tag+': checkpoint decision',event==log.checkpoint_event and selector.best_episode==log.best_episode)
        print(label,stage,'raw reconstruction passed',flush=True)
    check(label+': best identity',selector.best_episode==s['checkpoint_selection']['best_episode'])
    raw=pd.read_csv(p/'final_sampled_instance_metrics.csv');f=pd.read_csv(p/'final_selected_instance_metrics.csv')
    raw=verify_rows(raw,f,c,20,66,label+': final')
    agg=_aggregate_formal_rows(c,rows=records(raw.drop(columns='_success')),dataset_name='test',manifest='archived',unique_instance_count=20,repeat_count=3,universal=True,stage='final_test',strict_counts=True)
    for key in ['completion_rate','overall_completion_rate','preference_balanced_quality_score','raw_overall_completion_rate','raw_preference_balanced_quality_score']:
        check(label+': final '+key,close(agg[key],s['final_sampled'][key]))
    check(label+': success distribution',agg['success_count_distribution']==s['final_sampled']['success_count_distribution'])
    for split,loghash in [('validation',v.dataset_manifest_sha256.iloc[0]),('test',f.dataset_manifest_sha256.iloc[0])]:
        check(label+': '+split+' manifest matches local',sha(ROOT/f'data/manifests/v2/{split}/manifest.json')==loghash)
    f['_success']=successful(f);f['algorithm_seed']=seed
    suc=f[f._success];bad=f[~f._success]
    for _,row in bad.iterrows():
        failures.append(dict(seed=seed,**row[['instance_id',*PREF,'termination_reason','operation_progress','unfinished_orders',*OBJ]].to_dict()))
    # Match quality comparison on cells where all 3 requests succeeded.
    cell_success=raw.groupby(['instance_id','preference_key'])._success.sum()
    all3=set(cell_success[cell_success==3].index)
    common=raw[[tuple(k) in all3 for k in raw[['instance_id','preference_key']].to_numpy()]]
    quality_gain=1-common.groupby(['instance_id','preference_key']).preference_quality_score.min().mean()/common.preference_quality_score.mean()
    repeat_effects.append(dict(seed=seed,all3_success_cells=len(all3),single_random_Q_common=common.preference_quality_score.mean(),best3_Q_common=common.groupby(['instance_id','preference_key']).preference_quality_score.min().mean(),relative_gain=quality_gain))
    for key,g in f.groupby('preference_key'):
        good=g[g._success]
        prefs.append(dict(seed=seed,preference_key=key,**g[PREF].iloc[0].to_dict(),completion=g._success.mean(),Q=good.preference_quality_score.mean(),**{k:good[k].mean() for k in OBJ}))
    for key,g in f.groupby('pressure_type'):
        good=g[g._success]
        pressures.append(dict(seed=seed,pressure_type=key,instances=g.instance_id.nunique(),completion=g._success.mean(),Q=good.preference_quality_score.mean(),**{k:good[k].mean() for k in OBJ}))
    for name,pref in zip(['Flow','Cost','Workload'],PREF):
        g=f[np.isclose(f[pref],1)];good=g[g._success]
        endpoints.append(dict(seed=seed,preference=name,completion=g._success.mean(),**{k:good[k].mean() for k in OBJ}))
    for lo,hi in [(1,500),(501,1000),(1001,1500),(1501,2000)]:
        tg=t[t.episode.between(lo,hi)];ug=u[(u.episode_end+1).between(lo,hi)];vg=v[v.episode.between(lo,hi)]
        windows.append(dict(seed=seed,start=lo,end=hi,mean_severity=tg.severity.mean(),train_completion=tg.task_succeeded.mean(),train_reward=tg.reward.mean(),validation_Q=vg.preference_balanced_quality_score.mean(),validation_Q_std=vg.preference_balanced_quality_score.std(),validation_min_completion=vg.completion_rate.min(),**{k:ug[k].mean() for k in ['entropy','value_loss','approx_kl','clip_fraction','pre_update_explained_variance','gradient_norm','gradient_clipped_fraction','learning_rate']}))
    stamps=[]
    with (p/'worker_progress.jsonl').open(encoding='utf-8') as log:
        for line in log:
            row=json.loads(line)
            if 'timestamp' in row:stamps.append(float(row['timestamp']))
    fmt=lambda x:datetime.fromtimestamp(x,timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')
    best=s['checkpoint_selection']['best_episode'];bq=s['checkpoint_selection']['best_preference_balanced_quality_score'];lastq=v.iloc[-1].preference_balanced_quality_score
    summaries.append(dict(seed=seed,run=p.name,start_utc8=fmt(min(stamps)),end_utc8=fmt(max(stamps)),elapsed_hours=s['elapsed_seconds']/3600,best_episode=best,best_val_Q=bq,last_val_Q=lastq,last_worse_percent=100*(lastq/bq-1),test_Q=agg['preference_balanced_quality_score'],test_completion=agg['overall_completion_rate'],worst_preference_completion=agg['completion_rate'],test_failures=len(bad),raw_completion=agg['raw_overall_completion_rate'],raw_Q=agg['raw_preference_balanced_quality_score'],all3_success_cells=len(all3),first_repeat_completion=raw[raw.sampling_repeat==0]._success.mean(),**{k:suc[k].mean() for k in OBJ},train_completion=t.task_succeeded.mean(),train_hours=(u.sampling_wall_time_seconds.sum()+u.ppo_update_time_seconds.sum())/3600,validation_hours=v.validation_wall_time_seconds.sum()/3600,diagnostic_hours=d.validation_wall_time_seconds.sum()/3600,test_hours=s['final_sampled']['wall_time_seconds']/3600,lr_decay_count=s['learning_rate_control'].get('learning_rate_decay_count'),source_state_sha256=source_states[-1]))
    for frame,container in [(v,curves),(d,diagnostics),(t,trains),(u,updates)]:
        frame=frame.copy();frame['algorithm_seed']=seed;container.append(frame)
    finals.append(f)
    print(label,'complete',flush=True)

check('exactly requested seeds',sorted(s['seed'] for s in summaries)==[11,23,37])
check('configs equal except seed',all(c==configs[0] for c in configs))
check('recorded source hash equal',len(set(source_states))==1)
check('validation cohorts equal',all(s==subsets[0] for s in subsets))
check('selection and diagnostic disjoint',not set(subsets[0]['target']['instance_indices']) & set(subsets[0]['diagnostic']['instance_indices']))

S=pd.DataFrame(summaries);V=pd.concat(curves);D=pd.concat(diagnostics);T=pd.concat(trains);U=pd.concat(updates);F=pd.concat(finals);W=pd.DataFrame(windows);P=pd.DataFrame(prefs);E=pd.DataFrame(endpoints)
tables={'run_summary':S,'validation_curves':V,'diagnostic_curves':D,'training_windows':W,'test_by_preference':P,'test_by_pressure':pd.DataFrame(pressures),'test_failures':pd.DataFrame(failures),'test_endpoints':E,'sampling_effect':pd.DataFrame(repeat_effects),'checkpoint_identity':pd.DataFrame(checkpoints)}
for name,df in tables.items():df.to_csv(OUT/f'{name}.csv',index=False,encoding='utf-8-sig')
metrics=['test_Q','test_completion','worst_preference_completion','raw_completion',*OBJ]
group={k:{'mean':float(S[k].mean()),'sample_sd':float(S[k].std(ddof=1))} for k in metrics}
agreement=F.pivot(index=['instance_id','preference_key'],columns='algorithm_seed',values='_success').all(axis=1)
common=F.set_index(['instance_id','preference_key']).loc[agreement[agreement].index].reset_index()
common_means=common.groupby('algorithm_seed')[OBJ+['preference_quality_score']].mean()
common_means.to_csv(OUT/'common_success_test_means.csv',encoding='utf-8-sig')

plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
colors={11:'#0072B2',23:'#D55E00',37:'#009E73'}
fig,axs=plt.subplots(2,2,figsize=(12,8))
for seed in colors:
    vv=V[V.algorithm_seed==seed];dd=D[D.algorithm_seed==seed];tt=T[T.algorithm_seed==seed];ss=S[S.seed==seed].iloc[0];co=colors[seed]
    axs[0,0].plot(vv.episode,vv.preference_balanced_quality_score,'o-',ms=3,color=co,label=f'Seed {seed}')
    axs[0,0].scatter([ss.best_episode],[ss.best_val_Q],s=140,marker='*',color=co,edgecolor='black',zorder=5)
    axs[0,1].plot(vv.episode,vv.completion_rate*100,'o-',ms=3,color=co,label=f'Seed {seed}: best-of-3')
    axs[0,1].plot(vv.episode,vv.raw_minimum_preference_completion_rate*100,'--',alpha=.6,color=co)
    axs[1,0].plot(dd.episode,dd.preference_balanced_quality_score,'o-',color=co,label=f'Seed {seed}')
    axs[1,1].plot(tt.episode,tt.reward.rolling(100,min_periods=100).mean(),color=co,label=f'Seed {seed}')
for ax,ylabel in zip(axs.ravel(),['Selection validation Q (lower is better)','Worst-preference completion (%)','Held-out diagnostic Q (lower is better)','Training return (100-episode moving mean)']):
    ax.set_xlabel('Training episodes');ax.set_ylabel(ylabel);ax.grid(alpha=.2)
severity_ax=axs[1,1].twinx()
severity_rows=T[T.algorithm_seed==11]
severity_ax.plot(severity_rows.episode,severity_rows.severity.rolling(100,min_periods=100).mean(),':',color='black',alpha=.6)
severity_ax.set_ylabel('Curriculum severity (dotted)',color='black')
axs[0,0].legend(frameon=False);axs[0,1].text(.03,.03,'Solid: best-of-3; dashed: raw samples',transform=axs[0,1].transAxes,fontsize=9)
fig.tight_layout();fig.savefig(OUT/'convergence.png',dpi=180);fig.savefig(OUT/'convergence.pdf');plt.close(fig)
fig,axs=plt.subplots(2,2,figsize=(12,7))
for seed,co in colors.items():
    uu=U[U.algorithm_seed==seed]
    for ax,k in zip(axs.ravel(),['entropy','value_loss','approx_kl','pre_update_explained_variance']):
        ax.plot(uu.episode_end+1,uu[k],color=co,label=f'Seed {seed}');ax.set_xlabel('Training episodes');ax.set_ylabel(k);ax.grid(alpha=.2)
axs[0,0].legend(frameon=False);fig.tight_layout();fig.savefig(OUT/'optimizer.png',dpi=180);fig.savefig(OUT/'optimizer.pdf');plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(13,3.8))
for seed,co in colors.items():
    pp=P[P.seed==seed]
    for ax,pref,obj in zip(axs,PREF,OBJ):
        g=pp.groupby(pref)[obj].mean();ax.plot(g.index,g.values,'o-',ms=3,color=co,label=f'Seed {seed}')
for ax,pref,obj in zip(axs,PREF,OBJ):ax.set_xlabel(pref);ax.set_ylabel(obj);ax.grid(alpha=.2)
axs[0].legend(frameon=False);fig.tight_layout();fig.savefig(OUT/'preference_response.png',dpi=180);fig.savefig(OUT/'preference_response.pdf');plt.close(fig)

report={'analysis_date':'2026-10-22','timezone':'UTC+8','runs':summaries,'seed_mean_sd':group,'checks':checks,'all_checks_passed':all(x['passed'] for x in checks),'input_sha256':hashes,'common_success_cells':int(agreement.sum()),'source_git':read(RUNS[0]/'summary.json')['provenance']['git'],'source_state_sha256':source_states[0],'new_training':0,'new_evaluation_rollouts':0}
(OUT/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('FINAL',json.dumps({'summary':summaries,'group':group,'checks':len(checks),'passed':report['all_checks_passed'],'common_success_cells':int(agreement.sum())},ensure_ascii=False),flush=True)
