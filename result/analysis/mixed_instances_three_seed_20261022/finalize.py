"""Combine preserved two-seed receipts with new seed37; publish aligned tables."""
from pathlib import Path
import sys,json,hashlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT));OUT=Path(__file__).parent
PRIOR=ROOT/'result/analysis/mixed_instances_review_20261021'
BASE=ROOT/'result/analysis/ppo_heuristic_backfill_20261019'
SEARCH=ROOT/'result/analysis/heuristics_seed11_20261019'
from data.challenge import read_sealed
from result.metrics import successful_row
from utils import action_trace_sha256

def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def quality(obj,pref):
    v=np.asarray(obj);z=np.asarray(pref)*v/(v+[368.3143,353.27,21.03145448655653])
    return float((z.max()+.05*z.sum())/1.05)
def boot(v):
    v=np.asarray(v);rng=np.random.default_rng(20261022)
    return np.quantile(v[rng.integers(len(v),size=(20000,len(v)))].mean(1),[.025,.975]).tolist()

formal=read(OUT/'audit.json');assert formal['all_checks_passed']
formal_old=read(ROOT/'result/analysis/three_seed_best3_20261019/audit.json')
for audit in [formal,formal_old]:
    for name,h in audit['input_sha256'].items():assert sha(ROOT/name)==h,name
current_source_changes={};overlap=[];totals=[];failures=[];frozen_files={}
def verify_bundle(folder):
    plan=read_sealed(folder/'plan.json');count=0;deliveries=0;verified=[]
    changes=[]
    for name,h in plan['source_hashes'].items():
        p=ROOT/name
        if not p.exists() or sha(p)!=h:changes.append(name)
    current_source_changes[folder.name]=changes
    if folder==OUT:assert not changes
    for name in ['agent/ppo/agent.py','agent/ppo/network.py','result/metrics.py','result/sample_selection.py','analysis/challenge_study.py','utils.py']:
        assert sha(ROOT/name)==plan['source_hashes'][name],name
    for name,h in plan['search_inputs_sha256'].items():assert sha(SEARCH/name)==h,name
    for model in plan['models']:
        run=ROOT/'result/runs'/model['run'];assert sha(run/'best_checkpoint.pt')==model['checkpoint_sha256'];assert sha(run/'config.json')==model['config_sha256']
        frozen_files[str((run/'best_checkpoint.pt').relative_to(ROOT))]=model['checkpoint_sha256']
        seed=model['seed'];f=folder/f'seed{seed}';identity=read_sealed(f/'identity.json');done=read_sealed(f/'completed.json')
        assert done['identity']==identity['content_sha256'] and len(done['paths'])==28
        val=pd.read_csv(run/'selected_validation_instance_metrics.csv');val=val[val.validation_episode==model['checkpoint_episode']]
        for ref in done['paths']:
            path=f/ref['path'];assert sha(path)==ref['sha256'];req=read_sealed(path)
            frozen_files[str(path.relative_to(ROOT))]=ref['sha256']
            assert req['identity_sha256']==identity['content_sha256']
            raw=req['raw'];assert len(raw)==3 and {x['sampling_repeat'] for x in raw}=={0,1,2}
            chosen=min(raw,key=lambda x:(not successful_row(x['metrics']),x['metrics']['preference_quality_score'],x['sampling_repeat']))
            assert req['selected_repeat']==chosen['sampling_repeat']
            for x in raw:
                m=x['metrics'];success=successful_row(m);q=quality([m['flow_excess_objective'],m['reconfiguration_cost'],m['workload_objective_value']],x['preference'])
                assert abs(q-m['actual_preference_quality_score'])<1e-10
                assert abs((q if success else 1)-m['preference_quality_score'])<1e-10
                assert action_trace_sha256(x['actions'])==m['action_trace_sha256']
                assert not m['schedule_violations'] and not m['sampling_truncated']
                assert m['maximum_worker_fatigue']<=m['safe_fatigue_limit']+1e-9
                assert m['sampling_seed']==model['sampling_seeds'][x['sampling_repeat']]
                assert x['selected_for_aggregation']==(x['sampling_repeat']==chosen['sampling_repeat'])
                label={(1.,0.,0.):'Flow',(0.,1.,0.):'Cost',(0.,0.,1.):'OWA'}.get(tuple(x['preference']),'Equal')
                verified.append(dict(ppo_seed=seed,instance_id=x['instance_id'],preference=label,sampling_repeat=x['sampling_repeat'],success=success,selected=x['selected_for_aggregation'],quality=m['preference_quality_score']))
                count+=1
                if not success:failures.append(dict(bundle=folder.name,seed=seed,instance=x['instance_id'],preference=label,repeat=x['sampling_repeat'],reason=m.get('termination_reason',m.get('terminal_reason')),unfinished_orders=m['unfinished_orders']))
            deliveries+=1
            old=val[(val.instance_id==chosen['instance_id'])&(val.preference_key==chosen['preference_key'])]
            if len(old):
                assert len(old)==1;v=old.iloc[0];m=chosen['metrics']
                delta=abs(v.preference_quality_score-m['preference_quality_score']);same=v.sampling_repeat==chosen['sampling_repeat'];actions=v.action_trace_sha256==m['action_trace_sha256']
                assert delta<1e-10 and same and actions
                overlap.append(dict(bundle=folder.name,seed=seed,instance=chosen['instance_id'],preference=label,Q_error=delta,same_repeat=bool(same),same_actions=bool(actions)))
    assert count==plan['expected_raw_trajectories'] and deliveries==plan['expected_deliveries']
    csv=pd.read_csv(folder/'raw_samples.csv');assert len(csv)==count
    keys=['ppo_seed','instance_id','preference','sampling_repeat']
    a=pd.DataFrame(verified).set_index(keys).sort_index();b=csv.set_index(keys).sort_index();assert a.index.equals(b.index)
    assert np.allclose(a.quality,b.quality,rtol=0,atol=1e-12) and (a.success==b.success).all() and (a.selected==b.selected).all()
    totals.append(dict(bundle=folder.name,raw=count,deliveries=deliveries))
    return plan

oldplan=verify_bundle(BASE);plan2=verify_bundle(PRIOR);plan37=verify_bundle(OUT)
assert oldplan['instance_ids']==plan2['instance_ids']==plan37['instance_ids']
assert oldplan['preferences']==plan2['preferences']==plan37['preferences']
replay=read(OUT/'previous_environment_replay.json');assert replay['replayed_trajectories']==84 and replay['all_objectives_match']
raw=pd.concat([pd.read_csv(PRIOR/'raw_samples.csv'),pd.read_csv(OUT/'raw_samples.csv')],ignore_index=True)
selected=pd.concat([pd.read_csv(PRIOR/'selected_samples.csv'),pd.read_csv(OUT/'selected_samples.csv')],ignore_index=True)
pairs=pd.concat([pd.read_csv(PRIOR/'paired_comparison.csv'),pd.read_csv(OUT/'paired_comparison.csv')],ignore_index=True)
assert len(raw)==252 and len(selected)==84 and len(pairs)==168
assert not raw.duplicated(['ppo_seed','instance_id','preference','sampling_repeat']).any()
assert not selected.duplicated(['ppo_seed','instance_id','preference']).any()
assert set(selected.ppo_seed)=={11,23,37}
assert all(selected.groupby(['ppo_seed','instance_id','preference']).size()==1)
for name,df in [('unified_raw_samples',raw),('unified_selected_samples',selected),('unified_paired_comparison',pairs),('backfill_overlap',pd.DataFrame(overlap))]:df.to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')

baseline=pd.read_csv(BASE/'selected_samples.csv');cells=selected.groupby(['instance_id','preference'],as_index=False).agg(mixed_Q=('quality','mean'))
cells=cells.merge(baseline.groupby(['instance_id','preference'],as_index=False).agg(progressive_Q=('quality','mean')),on=['instance_id','preference'],validate='one_to_one')
pop=read(SEARCH/'search_selection.json')['population_sizes'];search_verified=0
for path in sorted((SEARCH/'tune_search').glob('*/*/*.json')):
    h=read(path)['summary']
    if h['population_size']!=pop[h['method']]:continue
    assert abs(quality(h['objectives'],h['preference'])-h['quality'])<1e-10;search_verified+=1
assert search_verified==56
for method in pop:
    tab=pairs[pairs.search_method==method].drop_duplicates(['instance_id','preference'])[['instance_id','preference','search_Q']].rename(columns={'search_Q':method})
    assert len(tab)==28;cells=cells.merge(tab,on=['instance_id','preference'],validate='one_to_one')
pref=cells.groupby('preference')[['progressive_Q','mixed_Q','nsga2_ls','moead']].mean().reindex(['Flow','Cost','OWA','Equal']);pref.loc['Overall']=pref.mean()
pref.to_csv(OUT/'heuristic_preferences_overall.csv',encoding='utf-8-sig');cells.to_csv(OUT/'heuristic_cells.csv',index=False,encoding='utf-8-sig')
seedrows=[]
for seed in [11,23,37]:
    a=selected[selected.ppo_seed==seed];b=baseline[baseline.ppo_seed==seed];rr=raw[raw.ppo_seed==seed]
    row=dict(seed=seed,progressive_Q=b.quality.mean(),mixed_Q=a.quality.mean(),relative_change_percent=100*(a.quality.mean()/b.quality.mean()-1),raw_success=int(rr.success.sum()),raw_count=len(rr),delivery_success=int(a.success.sum()),delivery_count=len(a))
    for method in pop:
        p=pairs[(pairs.ppo_seed==seed)&(pairs.search_method==method)];row[method+'_wins']=int((p.ppo_Q<p.search_Q-1e-10).sum())
    seedrows.append(row)
pd.DataFrame(seedrows).to_csv(OUT/'heuristic_by_seed.csv',index=False,encoding='utf-8-sig')
stats=[]
for comparator in ['progressive_Q','nsga2_ls','moead']:
    delta=cells[comparator]-cells.mixed_Q;cl=delta.groupby(cells.instance_id).mean()
    stats.append(dict(comparator=comparator,comparator_minus_mixed=delta.mean(),mixed_wins=int((delta>1e-10).sum()),ties=int((delta.abs()<=1e-10).sum()),mixed_losses=int((delta<-1e-10).sum()),instance_bootstrap95=boot(cl)))
G=pd.read_csv(OUT/'group_summary.csv');S=pd.read_csv(OUT/'comparison_summary.csv')
test_group=[]
for arm,suffix in [('progressive','default'),('mixed','mixed_instances')]:
    frames=[]
    for seed in [11,23,37]:
        p=next((ROOT/'result/runs').glob(f'experiment_seed{seed}*_01_{suffix}'));f=pd.read_csv(p/'final_selected_instance_metrics.csv');f['seed']=seed;f['loss']=f.preference_quality_score.where(f.task_succeeded,1.0);frames.append(f)
    fs=pd.concat(frames);test_group.append(dict(arm=arm,deliveries=len(fs),successful=int(fs.task_succeeded.sum()),full_coverage_loss=fs.loss.mean()))
oldloss=test_group[0]['full_coverage_loss'];newloss=test_group[1]['full_coverage_loss']

plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(11,4.6));pos=np.arange(5)
for j,(col,label,color) in enumerate([('progressive_Q','Progressive PPO (3 seeds)','#0072B2'),('mixed_Q','Mixed PPO (3 seeds)','#009E73'),('nsga2_ls','NSGA-II + LS N=16','#D55E00'),('moead','MOEA/D N=32','#CC79A7')]):ax.bar(pos+(j-1.5)*.19,pref[col],width=.18,label=label,color=color)
ax.set_xticks(pos,pref.index);ax.set_ylabel('Mean Q (lower is better)');ax.set_ylim(0,.7);ax.legend(frameon=False,ncol=2,fontsize=9);ax.grid(axis='y',alpha=.2)
fig.tight_layout();fig.savefig(OUT/'heuristic_overall_comparison.png',dpi=180);fig.savefig(OUT/'heuristic_overall_comparison.pdf');plt.close(fig)
receipt=dict(analysis_date='2026-10-22',training_audit_checks=len(formal['checks']),training_all_checks_passed=formal['all_checks_passed'],formal_input_hashes_reverified=True,backfill_bundles=totals,mixed_backfill_raw_count=len(raw),mixed_backfill_raw_success=int(raw.success.sum()),mixed_backfill_delivery_count=len(selected),mixed_backfill_delivery_success=int(selected.success.sum()),newly_sampled_policy_trajectories=84,reused_policy_trajectories=168,previous_environment_replay=replay,search_selected_requests_verified=search_verified,backfill_overlaps=overlap,source_changes_since_archived_evaluations=current_source_changes,checkpoints_unchanged=True,preserved_files=frozen_files,failures=failures,heuristic_stats=stats,test_group=test_group,group_test_loss_difference=newloss-oldloss,new_training=0)
(OUT/'unified_verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
print('HEURISTIC',pref.round(9).to_json(orient='index'));print('BYSEED',json.dumps(seedrows));print('GROUP_TEST',json.dumps(test_group));print('STATS',json.dumps(stats));print('RAW_SUCCESS',int(raw.success.sum()),'DELIVERIES',int(selected.success.sum()))
