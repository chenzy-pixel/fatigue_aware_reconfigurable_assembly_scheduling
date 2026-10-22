"""Align reward and preference diagnostics across the six saved runs."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).parent
SEEDS=[11,23,37];OBJECTIVES=['flow','cost','workload']
COLS=['flow_excess_objective','reconfiguration_cost','workload_objective_value']
PREFS=['preference_'+x for x in OBJECTIVES]
corr=[];local=[];ends=[];means=[];reward=[];trains=[];cohorts=[]
for seed in SEEDS:
    frames={}
    for arm,suffix in [('progressive','default'),('mixed','mixed_instances')]:
        p=next((ROOT/'result/runs').glob(f'experiment_seed{seed}*_01_{suffix}'))
        frames[arm]=pd.read_csv(p/'final_selected_instance_metrics.csv')
        t=pd.read_csv(p/'train_log.csv');t['seed']=seed;t['arm']=arm;trains.append(t)
        predicted=t.operation_progress-t.initial_progress-t.actual_preference_quality_score+t.initial_preference_quality_score-2*t.task_failed
        assert np.allclose(predicted,t.reward,rtol=0,atol=1e-10)
        for lo,hi in [(1,500),(501,1000),(1001,1500),(1501,2000),(1201,2000)]:
            g=t[t.episode.between(lo,hi)]
            reward.append(dict(seed=seed,arm=arm,start=lo,end=hi,reward=g.reward.mean(),reward_std=g.reward.std(),failures=int(g.task_failed.sum()),reward_without_penalty=(g.reward-g.reward_failure).mean(),severity=g.severity.mean()))
    key=['instance_id','preference_key'];common=frames['progressive'].query('task_succeeded')[key].merge(frames['mixed'].query('task_succeeded')[key],on=key,validate='one_to_one')
    assert len(common)==(1316 if seed==23 else 1320)
    delta_frames={arm:f.merge(common,on=key,validate='one_to_one') for arm,f in frames.items()}
    for arm,f in frames.items():
        for obj,col,pref in zip(OBJECTIVES,COLS,PREFS):
            full_good=f[f.task_succeeded];matched=delta_frames[arm]
            rho=[g[pref].corr(g[col],method='spearman') for _,g in matched.groupby('instance_id')]
            corr.append(dict(seed=seed,arm=arm,objective=obj,common_success_cells=len(matched),rho_mean=np.mean(rho),negative_instances=int((np.array(rho)<0).sum())))
            for weight,g in full_good.groupby(pref):means.append(dict(seed=seed,arm=arm,objective=obj,weight=weight,value=g[col].mean(),successful_rows=len(g)))
        cost=f[np.isclose(f.preference_cost,1)].set_index('instance_id').sort_index();work=f[np.isclose(f.preference_workload,1)].set_index('instance_id').sort_index()
        assert cost.index.equals(work.index) and cost.task_succeeded.all() and work.task_succeeded.all()
        ends.append(dict(seed=seed,arm=arm,cost_at_cost=cost.reconfiguration_cost.mean(),cost_at_workload=work.reconfiguration_cost.mean(),workload_at_cost=cost.workload_objective_value.mean(),workload_at_workload=work.workload_objective_value.mean(),workload_drop_percent=100*(1-work.workload_objective_value.mean()/cost.workload_objective_value.mean()),cost_increase_percent=100*(work.reconfiguration_cost.mean()/cost.reconfiguration_cost.mean()-1)))
    for obj,col,pref in zip(OBJECTIVES,COLS,PREFS):
        rest=[x for x in PREFS if x!=pref];pairs={arm:{} for arm in frames}
        for arm,f in frames.items():
            for iid,g in f[np.isclose(f[rest[0]],f[rest[1]])].groupby('instance_id'):
                rs=g.sort_values(pref).to_dict('records')
                for a,b in zip(rs,rs[1:]):
                    if a['task_succeeded'] and b['task_succeeded']:pairs[arm][(iid,a['preference_key'],b['preference_key'])]=b[col]-a[col]
        common_steps=pairs['progressive'].keys()&pairs['mixed'].keys()
        for arm in frames:
            n=sum(pairs[arm][x]<=1e-8 for x in common_steps)
            local.append(dict(seed=seed,arm=arm,objective=obj,matched_steps=len(common_steps),nonworsening=n,rate=n/len(common_steps)))

P=pd.DataFrame(corr);M=pd.DataFrame(means);R=pd.DataFrame(reward);L=pd.DataFrame(local);T=pd.concat(trains)
for name,data in [('preference_correlations_matched',P),('preference_monotonicity_matched',L),('preference_cost_workload_endpoints',pd.DataFrame(ends)),('preference_weight_means',M),('reward_windows',R)]:data.to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(3,3,figsize=(14,10))
for i,seed in enumerate(SEEDS):
    for j,obj in enumerate(OBJECTIVES):
        ax=axs[i,j]
        for arm,color,style in [('progressive','#0072B2','--'),('mixed','#D55E00','-')]:
            g=M[(M.seed==seed)&(M.arm==arm)&(M.objective==obj)].sort_values('weight')
            ax.plot(g.weight,g.value,marker='o',ms=3,color=color,linestyle=style,label=arm)
        ax.set_xlabel(obj+' preference weight');ax.set_ylabel(COLS[j]);ax.set_title(f'Seed {seed}');ax.grid(alpha=.2)
axs[0,0].legend(frameon=False);fig.tight_layout();fig.savefig(OUT/'preference_comparison.png',dpi=160);fig.savefig(OUT/'preference_comparison.pdf');plt.close(fig)
fig,axs=plt.subplots(2,1,figsize=(11,7),sharex=True)
for seed,color in [(11,'#0072B2'),(23,'#D55E00'),(37,'#009E73')]:
    g=T[(T.seed==seed)&(T.arm=='mixed')]
    axs[0].plot(g.episode,g.reward.rolling(100,min_periods=100).mean(),color=color,label=f'Seed {seed}')
    axs[1].plot(g.episode,(g.reward-g.reward_failure).rolling(100,min_periods=100).mean(),color=color)
axs[0].legend(frameon=False)
for ax,title in zip(axs,['Actual return, rolling 100','Same trajectories before failure penalty']):ax.set_ylabel(title);ax.grid(alpha=.2)
axs[1].set_xlabel('Training episodes');fig.tight_layout();fig.savefig(OUT/'mixed_reward_three_seeds.png',dpi=160);fig.savefig(OUT/'mixed_reward_three_seeds.pdf');plt.close(fig)
S=pd.read_csv(OUT/'comparison_summary.csv')
group=S.groupby('arm').agg(Q_mean=('test_Q','mean'),Q_sample_sd=('test_Q','std'),delivery_completion=('test_completion','mean'),raw_completion=('raw_completion','mean'),mean_steps=('training_steps','mean'),mean_total_hours=('total_hours','mean')).reset_index()
group.to_csv(OUT/'group_summary.csv',index=False,encoding='utf-8-sig')
print('GROUP',group.to_json(orient='records'));print('PREFERENCE',P.round(6).to_json(orient='records'));print('MONOTONIC',L.round(6).to_json(orient='records'));print('ENDPOINTS',pd.DataFrame(ends).round(5).to_json(orient='records'));print('REWARD',R[(R.start==1201)].round(5).to_json(orient='records'))
