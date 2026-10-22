"""Matched-seed curriculum comparison from saved completed runs."""
from pathlib import Path
import json,hashlib,sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
OUT=Path(__file__).parent
from data.distribution import training_sampling_plan
from result.provenance import effective_config_snapshot

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def differences(a,b,path=''):
    if isinstance(a,dict) and isinstance(b,dict):
        return [x for k in sorted(a.keys()|b.keys()) for x in differences(a.get(k),b.get(k),path+'.'+k)]
    return [] if a==b else [path]
def boot(a):
    a=np.asarray(a);rng=np.random.default_rng(20261021)
    return np.quantile(a[rng.integers(len(a),size=(20000,len(a)))].mean(1),[.025,.975]).tolist()

summary=[];curves=[];diags=[];trains=[];update=[];paired=[];preference=[];endpoints=[];windows=[];failures=[];proof=[];config_diffs=[]
names=['flow_excess_objective','reconfiguration_cost','workload_objective_value']
for seed in [11,23,37]:
    old=next((ROOT/'result/runs').glob(f'experiment_seed{seed}*_01_default'));new=next((ROOT/'result/runs').glob(f'experiment_seed{seed}*_01_mixed_instances'))
    oc=read(old/'config.json');nc=read(new/'config.json');diff=differences(oc,nc)
    assert set(diff)=={'.experiment_name','.generator.curriculum.anchors','.generator.severity_curriculum.anchors'}
    assert read(old/'validation_subsets.json')==read(new/'validation_subsets.json')
    config_diffs.append(dict(seed=seed,changed_config_fields=diff,matched_validation_subsets=True))
    dfs={}
    for arm,p in [('progressive',old),('mixed',new)]:
        c=read(p/'config.json');s=read(p/'summary.json');t=pd.read_csv(p/'train_log.csv');u=pd.read_csv(p/'update_log.csv');v=pd.read_csv(p/'validation_log.csv');d=pd.read_csv(p/'diagnostic_log.csv');f=pd.read_csv(p/'final_selected_instance_metrics.csv')
        assert effective_config_snapshot(c)['sha256']==s['provenance']['effective_config_sha256']
        if arm=='mixed':
            # Current training rejects archived curriculum fields; audit recorded
            # per-window exposure instead of invoking a newer training sampler.
            from collections import Counter
            expected={k:round(v*100) for k,v in c['generator']['curriculum']['anchors'][0]['weights'].items()}
            assert t.severity.eq(1).all()
            assert all(Counter(g.pressure_type)==expected for _,g in t.groupby((t.episode-1)//100))
        dfs[arm]=f
        fs=s['final_sampled'];late=v[v.episode>=1300];tl=t[t.episode>1200]
        summary.append(dict(arm=arm,seed=seed,run=p.name,best_episode=s['checkpoint_selection']['best_episode'],best_val_Q=s['checkpoint_selection']['best_preference_balanced_quality_score'],last_val_Q=v.iloc[-1].preference_balanced_quality_score,last_val_worst_completion=v.iloc[-1].completion_rate,last_val_overall_completion=v.iloc[-1].overall_completion_rate,test_Q=fs['preference_balanced_quality_score'],test_completion=fs['overall_completion_rate'],raw_completion=fs['raw_overall_completion_rate'],test_worst_completion=fs['minimum_preference_completion_rate'],**{k:f[f.task_succeeded][k].mean() for k in names},train_failures=int(t.task_failed.sum()),late_train_failures=int(tl.task_failed.sum()),late_train_reward=tl.reward.mean(),late_train_sd=tl.reward.std(),late_validation_Q_mean=late.preference_balanced_quality_score.mean(),late_validation_Q_sd=late.preference_balanced_quality_score.std(),late_validation_Q_range=late.preference_balanced_quality_score.max()-late.preference_balanced_quality_score.min(),late_validation_failures=int(((1-late.overall_completion_rate)*650).round().sum()),training_steps=int(t.step_count.sum()),sampling_update_hours=(u.sampling_wall_time_seconds.sum()+u.ppo_update_time_seconds.sum())/3600,total_hours=s['elapsed_seconds']/3600,last_diagnostic_Q=d.iloc[-1].preference_balanced_quality_score,last_diagnostic_worst_completion=d.iloc[-1].completion_rate,last_diagnostic_overall_completion=d.iloc[-1].overall_completion_rate,source_hash=s['provenance']['source_state_sha256'],git=s['provenance']['git']['commit'],git_dirty=s['provenance']['git']['dirty']))
        for frame,container in [(v,curves),(d,diags),(t,trains),(u,update)]:
            frame=frame.copy();frame['arm']=arm;frame['algorithm_seed']=seed;container.append(frame)
        for pref,col in zip(['flow','cost','workload'],names):
            success=f[f.task_succeeded];corr=[g['preference_'+pref].corr(g[col],method='spearman') for _,g in success.groupby('instance_id')]
            preference.append(dict(arm=arm,seed=seed,objective=pref,mean_within_instance_spearman=np.mean(corr),negative_instances=int((np.array(corr)<0).sum())))
            ep=f[np.isclose(f['preference_'+pref],1)];endpoints.append(dict(arm=arm,seed=seed,preference=pref,success=int(ep.task_succeeded.sum()),**{k:ep[ep.task_succeeded][k].mean() for k in names}))
        for start,end in [(1,500),(501,1000),(1001,1500),(1501,2000)]:
            tt=t[t.episode.between(start,end)];vv=v[v.episode.between(start,end)]
            windows.append(dict(arm=arm,seed=seed,start=start,end=end,severity=tt.severity.mean(),reward=tt.reward.mean(),reward_without_penalty=(tt.reward-tt.reward_failure).mean(),training_failures=int(tt.task_failed.sum()),val_Q=vv.preference_balanced_quality_score.mean(),val_Q_sd=vv.preference_balanced_quality_score.std()))
        for split,filename in [('validation','selected_validation_instance_metrics.csv'),('diagnostic','selected_diagnostic_instance_metrics.csv')]:
            z=pd.read_csv(p/filename);z=z[(z.validation_episode==2000)&~z.task_succeeded]
            for _,r in z.iterrows():failures.append(dict(arm=arm,seed=seed,split=split,**r[['instance_id','preference_flow','preference_cost','preference_workload','termination_reason','unfinished_orders']].to_dict()))
    a=dfs['progressive'].set_index(['instance_id','preference_key']).sort_index();b=dfs['mixed'].set_index(['instance_id','preference_key']).sort_index();assert a.index.equals(b.index)
    ok=a.task_succeeded&b.task_succeeded
    for name in ['preference_quality_score',*names]:
        delta=b.loc[ok,name]-a.loc[ok,name];byinstance=delta.groupby(level='instance_id').mean()
        paired.append(dict(seed=seed,metric=name,common_success_cells=int(ok.sum()),old_mean=a.loc[ok,name].mean(),mixed_mean=b.loc[ok,name].mean(),mixed_minus_old=delta.mean(),relative_percent=100*(b.loc[ok,name].mean()/a.loc[ok,name].mean()-1),better_cells=int((delta<-1e-10).sum()),tied_cells=int((delta.abs()<=1e-10).sum()),worse_cells=int((delta>1e-10).sum()),instance_bootstrap95_low=boot(byinstance)[0],instance_bootstrap95_high=boot(byinstance)[1]))
    loss_a=a.preference_quality_score.where(a.task_succeeded,1.);loss_b=b.preference_quality_score.where(b.task_succeeded,1.)
    proof.append(dict(seed=seed,old_failures=int((~a.task_succeeded).sum()),mixed_failures=int((~b.task_succeeded).sum()),full_coverage_loss_old=loss_a.mean(),full_coverage_loss_mixed=loss_b.mean(),full_coverage_delta=float((loss_b-loss_a).mean()),exploratory_instance_ci=boot((loss_b-loss_a).groupby(level='instance_id').mean())))

S=pd.DataFrame(summary);V=pd.concat(curves);D=pd.concat(diags);T=pd.concat(trains);U=pd.concat(update)
for name,data in [('comparison_summary',S),('comparison_validation',V),('comparison_diagnostic',D),('comparison_training_windows',pd.DataFrame(windows)),('test_paired',pd.DataFrame(paired)),('preference_comparison',pd.DataFrame(preference)),('endpoints_comparison',pd.DataFrame(endpoints)),('last_failures',pd.DataFrame(failures))]:data.to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
colors={'progressive':'#0072B2','mixed':'#D55E00'}
fig,axs=plt.subplots(3,3,figsize=(16,10),sharex=True)
for j,seed in enumerate([11,23,37]):
    for arm in colors:
        vv=V[(V.arm==arm)&(V.algorithm_seed==seed)];dd=D[(D.arm==arm)&(D.algorithm_seed==seed)];tt=T[(T.arm==arm)&(T.algorithm_seed==seed)];ss=S[(S.arm==arm)&(S.seed==seed)].iloc[0]
        axs[0,j].plot(vv.episode,vv.preference_balanced_quality_score,'o-',color=colors[arm],ms=3,label=arm)
        axs[0,j].scatter([ss.best_episode],[ss.best_val_Q],marker='*',s=110,color=colors[arm],edgecolor='black',zorder=5)
        axs[1,j].plot(vv.episode,100*vv.completion_rate,'o-',ms=3,color=colors[arm])
        axs[2,j].plot(dd.episode,dd.preference_balanced_quality_score,'o-',color=colors[arm])
    axs[0,j].set_title(f'Seed {seed}');axs[0,j].legend(frameon=False);axs[2,j].set_xlabel('Training episodes')
    for ax,label in zip(axs[:,j],['Selection validation Q (lower is better)','Worst-preference completion (%)','Fixed diagnostic Q (lower is better)']):ax.set_ylabel(label);ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(OUT/'curriculum_comparison.png',dpi=180);fig.savefig(OUT/'curriculum_comparison.pdf');plt.close(fig)
fig,axs=plt.subplots(2,3,figsize=(16,7),sharex=True)
for j,seed in enumerate([11,23,37]):
    for arm in colors:
        tt=T[(T.arm==arm)&(T.algorithm_seed==seed)]
        axs[0,j].plot(tt.episode,tt.reward.rolling(100).mean(),color=colors[arm],label=arm)
        axs[1,j].plot(tt.episode,(tt.reward-tt.reward_failure).rolling(100).mean(),color=colors[arm])
    axs[0,j].set_title(f'Seed {seed}');axs[0,j].legend(frameon=False);axs[1,j].set_xlabel('Training episodes')
    for ax,label in zip(axs[:,j],['Actual return, rolling 100','Same returns before failure penalty']):ax.set_ylabel(label);ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(OUT/'reward_comparison.png',dpi=180);fig.savefig(OUT/'reward_comparison.pdf');plt.close(fig)
(OUT/'comparison_identity.json').write_text(json.dumps(dict(config_differences=config_diffs,test_full_coverage=proof,uncertainty='Two trained seeds; per-instance bootstrap is exploratory and not training-seed uncertainty.'),indent=2),encoding='utf-8')
print('SUMMARY',S.round(6).to_json(orient='records'));print('PAIRED',pd.DataFrame(paired).round(6).to_json(orient='records'));print('PREF',json.dumps(preference));print('FULL',json.dumps(proof))
