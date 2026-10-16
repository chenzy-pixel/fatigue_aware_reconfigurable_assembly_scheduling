"""Offline decision-horizon study of the frozen v4 diagnostic; never trains."""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.gae_lambda_diagnostic import estimates

SOURCE = ROOT / 'result/analysis/gae_lambda_v4_20261012'
TRACE = ROOT / 'result/analysis/shared_excess_last_20261012/trace'
OUT = SOURCE / 'decision_horizon_study'
LAMBDAS = (.9, .95, .97, .98, .99, .995, 1.)
GAMMAS = (.95, .97, .99, .995, .999, 1.)


def quantiles(x):
    x = np.asarray(x, dtype=float)
    return dict(zip(('min', 'p10', 'median', 'p90', 'max'), map(float, np.quantile(x, [0, .1, .5, .9, 1]))))


def main():
    OUT.mkdir(exist_ok=True)
    files = sorted(SOURCE.glob('*_repeat[012].json'))
    assert len(files) == 6
    episodes = [e for p in files for e in json.loads(p.read_text(encoding='utf-8'))]
    assert len(episodes) == 1200
    config = json.loads((SOURCE/'frozen_source_config.json').read_text(encoding='utf-8'))
    assert config['ppo']['gamma'] == 1.0
    assert config['training']['forced_action_compression'] is False
    assert config['reward']['terminal_failure_penalty'] == 2.0
    assert not config['reward']['feasibility_shaping']['enabled']
    by_key, ep_rows, gamma_rows, instance_rows = {}, [], [], []
    for e in episodes:
        m = e['metadata']
        key = (e['role'], m['instance_id'], e['region'], m['repeat'])
        assert key not in by_key
        by_key[key] = e
        e['r'] = np.asarray(e['rewards'])
        e['v'] = np.asarray(e['values'])
        e['choice'] = np.asarray(e['legal_counts']) > 1
        e['adv'], e['mc'] = estimates(e['r'], e['v'], lambdas=LAMBDAS)
        n = len(e['r'])
        ep_rows.append(dict(role=e['role'], instance_id=m['instance_id'], region=e['region'],
                            repeat=m['repeat'], success=e['success'], steps=n,
                            choice_steps=int(e['choice'].sum()), forced_steps=int((~e['choice']).sum()),
                            zero_reward_steps=int((np.abs(e['r']) < 1e-12).sum()),
                            terminal_reward=e['r'][-1], total_reward=e['mc'][0],
                            terminal_reason=m['terminal_reason']))
        for gamma in GAMMAS:
            weights = gamma ** np.arange(n)
            # Check the discounted telescoping identity independently of the return recursion.
            increments = e['r'].copy()
            if not e['success']:increments[-1] += 2.
            potential = np.r_[0.,np.cumsum(increments)]
            telescoped = weights[-1]*potential[-1] + (1-gamma)*np.dot(weights[:-1],potential[1:-1])
            if not e['success']:telescoped -= 2*weights[-1]
            np.testing.assert_allclose(weights @ e['r'],telescoped,atol=1e-12,rtol=1e-12)
            gamma_rows.append(dict(role=e['role'], instance_id=m['instance_id'], region=e['region'],
                                   repeat=m['repeat'], success=e['success'], gamma=gamma, steps=n,
                                   return_from_start=float(weights @ e['r']),
                                   terminal_weight=float(weights[-1]),
                                   discounted_failure_penalty=float(-2*weights[-1]) if not e['success'] else 0.))
    ep = pd.DataFrame(ep_rows)
    ep.to_csv(OUT/'episode_lengths.csv', index=False)
    gamma_df = pd.DataFrame(gamma_rows)
    gamma_df.to_csv(OUT/'discounted_returns.csv', index=False)
    for iid in sorted(ep.instance_id.unique()):
        path = ROOT/f'data/instances/v2/validation/instance_{iid.rsplit("_", 1)[1]}.json'
        data = json.loads(path.read_text(encoding='utf-8'))
        ins = data['instance']
        assert ins['instance_id'] == iid
        instance_rows.append(dict(instance_id=iid, orders=len(ins['orders']),
                                  operations=sum(len(o['operations']) for o in ins['orders']),
                                  machines=len(ins['machines']), workers=len(ins['workers']),
                                  source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    instances = pd.DataFrame(instance_rows)
    instances.to_csv(OUT/'instance_sizes.csv', index=False)
    length_rows = []
    for (role, region, success), group in ep.groupby(['role', 'region', 'success']):
        length_rows.append(dict(role=role, region=region, success=success, episodes=len(group),
                                **{'steps_'+k:v for k,v in quantiles(group.steps).items()},
                                choice_fraction=group.choice_steps.sum()/group.steps.sum()))
    pd.DataFrame(length_rows).to_csv(OUT/'length_summary.csv', index=False)

    metrics = []
    for role in ('best', 'last'):
        role_eps = [e for e in episodes if e['role'] == role]
        centers = {l:np.concatenate([e['adv'][l] for e in role_eps]).mean() for l in LAMBDAS}
        for region in ('all','flow','cost','variance','center'):
            for outcome in ('all','success','failure'):
                selected = [e for e in role_eps if (region == 'all' or e['region'] == region)
                            and (outcome == 'all' or e['success'] == (outcome == 'success'))]
                if not selected:
                    continue
                base = np.concatenate([e['adv'][1.][e['choice']] for e in selected])
                for lam in LAMBDAS:
                    a = np.concatenate([e['adv'][lam][e['choice']] for e in selected])
                    valid = (np.abs(a)>1e-8) & (np.abs(base)>1e-8)
                    ac, bc = a-centers[lam], base-centers[1.]
                    vc = (np.abs(ac)>1e-8) & (np.abs(bc)>1e-8)
                    error = a-base
                    metrics.append(dict(role=role, region=region, outcome=outcome, trajectories=len(selected),
                                        steps=len(a), lam=lam, variance=a.var(), bias=error.mean(),
                                        mae=np.abs(error).mean(), rmse=np.sqrt(np.mean(error**2)),
                                        same_sign=np.mean(np.sign(a[valid])==np.sign(base[valid])),
                                        centered_same_sign=np.mean(np.sign(ac[vc])==np.sign(bc[vc]))))
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(OUT/'lambda_tradeoffs.csv', index=False)
    # Regression against the completed diagnostic, not against a new definition.
    old = pd.read_csv(SOURCE/'gae_metrics.csv')
    for _, row in metrics_df[metrics_df.lam.isin([.9,.95,.97,.99,1.])].iterrows():
        match = old[(old.role==row.role)&(old.region==row.region)&(old.outcome==row.outcome)
                    &(old.scope=='choice_steps')&(old['lambda']==row.lam)].iloc[0]
        np.testing.assert_allclose([row.variance,row.bias,row.mae,row.rmse],
                                   [match.advantage_variance,match.bias,match.mae,match.rmse],atol=1e-12)

    ranking_rows = []
    for (role, iid, region), group in gamma_df.groupby(['role','instance_id','region']):
        pivot = group.pivot(index='repeat',columns='gamma',values='return_from_start')
        outcomes = group.drop_duplicates('repeat').set_index('repeat').success
        for a,b in combinations(pivot.index,2):
            ref = pivot.loc[a,1.]-pivot.loc[b,1.]
            for gamma in GAMMAS:
                diff = pivot.loc[a,gamma]-pivot.loc[b,gamma]
                mixed = outcomes[a] != outcomes[b]
                failed_beats_success = bool(diff > 1e-10) if not outcomes[a] else bool(diff < -1e-10)
                ranking_rows.append(dict(role=role,instance_id=iid,region=region,repeat_a=a,repeat_b=b,
                                         gamma=gamma,mixed_outcomes=mixed,
                                         both_success=bool(outcomes[a] and outcomes[b]),
                                         reference_difference=ref,discounted_difference=diff,
                                         reversal=bool(ref*diff<0 and abs(ref)>1e-10 and abs(diff)>1e-10),
                                         failed_beats_success=failed_beats_success if mixed else False))
    ranks = pd.DataFrame(ranking_rows)
    ranks.to_csv(OUT/'within_instance_return_rankings.csv',index=False)
    rank_summary=[]
    for (role,gamma),g in ranks.groupby(['role','gamma']):
        mixed=g[g.mixed_outcomes]; successful=g[g.both_success]
        rank_summary.append(dict(role=role,gamma=gamma,pairs=len(g),reversals=int(g.reversal.sum()),
                                 successful_pairs=len(successful),successful_reversals=int(successful.reversal.sum()),
                                 mixed_pairs=len(mixed),failed_beats_success=int(mixed.failed_beats_success.sum())))
    pd.DataFrame(rank_summary).to_csv(OUT/'discount_ranking_summary.csv',index=False)

    traces = pd.read_csv(TRACE/'decision_trace.csv')
    summaries = pd.read_csv(TRACE/'trace_summary.csv').set_index('case')
    detail_rows, case_rows, wait_rows = [], [], []
    for case, g in traces.groupby('case',sort=False):
        g = g.sort_values('step').copy()
        e = by_key[(g.role.iloc[0],g.instance_id.iloc[0],'cost',int(g.sampling_repeat.iloc[0]))]
        assert len(g)==len(e['r'])
        np.testing.assert_array_equal(g.action, e['actions'])
        np.testing.assert_allclose(g.reward,e['r'],atol=1e-12)
        np.testing.assert_allclose(g.critic_value,e['v'],atol=5e-7,rtol=0)
        g['mc']=e['mc']; g['critic_value']=e['v']
        g['remaining_steps']=len(g)-1-np.arange(len(g))
        g['time_advance']=g.after_time-g.time
        g['choice']=g.legal_count>1
        g['optional_wait']=g.choice & (g.action_type=='WAIT')
        g['success']=e['success']
        for lam in LAMBDAS:
            tag=str(lam).replace('.','')
            g[f'adv_{tag}']=e['adv'][lam]
            g[f'return_target_{tag}']=e['adv'][lam]+e['v']
            g[f'failure_contribution_{tag}']=-2*lam**g.remaining_steps if not e['success'] else 0.
        detail_rows.append(g)
        choice=g[g.choice]; waits=g[g.optional_wait]
        case_rows.append(dict(case=case,role=e['role'],instance_id=g.instance_id.iloc[0],repeat=int(g.sampling_repeat.iloc[0]),
                              success=e['success'],steps=len(g),choice_steps=len(choice),zero_time_steps=int((g.time_advance.abs()<1e-10).sum()),
                              optional_waits=len(waits),forced_waits=int(((g.action_type=='WAIT')&~g.choice).sum()),
                              terminal_time=g.after_time.iloc[-1],total_reward=e['mc'][0],initial_value=e['v'][0],
                              terminal_reward=e['r'][-1],last_preterminal_value=e['v'][-1],
                              terminal_operation_progress=summaries.loc[case,'operation_progress'],
                              **{f'count_{a}':int((g.action_type==a).sum()) for a in g.action_type.unique()}))
        for lam in LAMBDAS:
            col=f'adv_{str(lam).replace(".","")}'
            valid=(waits[col].abs()>1e-8)&(waits.adv_10.abs()>1e-8)
            wait_rows.append(dict(case=case,role=e['role'],success=e['success'],lam=lam,optional_waits=len(waits),
                                  positive=int((waits[col]>1e-8).sum()),
                                  opposite_mc=int(((waits[col]*waits.adv_10<0)&valid).sum()),
                                  mae=float((waits[col]-waits.adv_10).abs().mean()),
                                  mean_terminal_weight=float((lam**waits.remaining_steps).mean())))
    detail=pd.concat(detail_rows,ignore_index=True)
    detail.to_csv(OUT/'decision_details.csv',index=False)
    cases=pd.DataFrame(case_rows)
    cases.to_csv(OUT/'case_summary.csv',index=False)
    pd.DataFrame(wait_rows).to_csv(OUT/'optional_wait_advantages.csv',index=False)
    action_summary=detail.groupby(['role','success','action_type']).agg(
        steps=('step','size'),zero_time_fraction=('time_advance',lambda x:(x.abs()<1e-10).mean()),
        median_time_advance=('time_advance','median'),max_time_advance=('time_advance','max'),
        choice_fraction=('choice','mean')).reset_index()
    action_summary.to_csv(OUT/'action_semantics.csv',index=False)
    # Evidence rows selected by an explicit rule: largest positive .95 estimate among optional WAITs
    # with negative MC, for each failed trace; also provide evenly spaced trace points.
    examples=[]
    for case,g in detail[~detail.success].groupby('case'):
        candidates=g[g.optional_wait & (g.adv_095>1e-8) & (g.adv_10 < -1e-8)]
        indices=set(g.iloc[np.linspace(0,len(g)-1,6).astype(int)].index)
        if len(candidates):indices.add(candidates.adv_095.idxmax())
        for index in sorted(indices):
            row=g.loc[index].to_dict()
            row['selection_reason']='largest_positive_095_optional_wait' if len(candidates) and index==candidates.adv_095.idxmax() else 'evenly_spaced'
            examples.append(row)
    pd.DataFrame(examples).to_csv(OUT/'selected_decisions.csv',index=False)
    selfcheck = dict(complete_episodes=len(episodes),steps=int(ep.steps.sum()),
                    original_five_lambda_metrics_match=True,matched_detailed_traces=len(cases),
                    all_detailed_actions_rewards_values_match=True,
                    discounted_telescoping_verified=True, forced_action_compression=False,
                    source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in files+[TRACE/'decision_trace.csv',TRACE/'trace_summary.csv']},
                    lambdas=LAMBDAS,gammas=GAMMAS,
                    gamma_analysis='Recompute discounted rewards only; no reuse of gamma=1 critic as if trained for gamma<1.',
                    sample_limit='The 12 detailed traces are selected Cost cases, not a representative estimate of action proportions.',
                    no_training=True)
    (OUT/'verification.json').write_text(json.dumps(selfcheck,ensure_ascii=False,indent=2),encoding='utf-8')
    stats = dict(instance_sizes={c:quantiles(instances[c]) for c in ('orders','operations','machines','workers')},
                 all_step_lengths=quantiles(ep.steps),
                 roles={role:dict(steps=quantiles(g.steps),choice_steps=quantiles(g.choice_steps),
                                  total_steps=int(g.steps.sum()),choice_fraction=float(g.choice_steps.sum()/g.steps.sum()))
                        for role,g in ep.groupby('role')},
                 detailed_zero_time_fraction=float((detail.time_advance.abs()<1e-10).mean()),
                 detailed_steps=len(detail))
    (OUT/'summary.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
    print(json.dumps(stats,indent=2))
    print(metrics_df[(metrics_df.region=='all') & (metrics_df.outcome=='all')].to_string(index=False))
    print(pd.DataFrame(rank_summary).to_string(index=False))
    plot(detail, metrics_df)


def plot(detail,metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    case='last_validation_reconfiguration_bottleneck_2000177_1'
    g=detail[detail.case==case]
    ax=axes[0,0]
    ax.plot(g.step,g.critic_value,label='Critic V',color='black')
    ax.plot(g.step,g.mc,label='MC return G',color='#cc4444')
    ax.plot(g.step,g.return_target_095,label='GAE target, lambda=.95',color='#e69f00')
    ax.plot(g.step,g.return_target_099,label='GAE target, lambda=.99',color='#0072b2')
    ax.set(title='Actual failed Cost case: instance 2000177',xlabel='Decision index',ylabel='Reward units')
    ax.legend(fontsize=8)
    ax=axes[0,1]
    w=g[g.optional_wait]
    for l,c in [(.95,'#e69f00'),(.99,'#0072b2'),(1.,'#cc4444')]:
        ax.scatter(w.step,w[f'adv_{str(l).replace(".","")}'],s=15,label=f'lambda={l}',color=c)
    ax.axhline(0,c='black',lw=.8)
    ax.set(title='Advantages on optional WAIT decisions',xlabel='Decision index',ylabel='Unnormalized advantage')
    ax.legend(fontsize=8)
    ax=axes[1,0]
    for role,marker in [('best','s'),('last','o')]:
        rows=metrics[(metrics.role==role)&(metrics.region=='all')&(metrics.outcome=='all')]
        ax.plot(rows.variance,rows.rmse,marker=marker,label=role)
        for _,r in rows.iterrows():
            ax.annotate(f'{r.lam:g}',(r.variance,r.rmse),xytext=(3,3),textcoords='offset points',fontsize=8)
    ax.set(xscale='log',title='Fixed-data estimator tradeoff',xlabel='Advantage variance (choice steps)',ylabel='Return RMSE vs MC')
    ax.legend(fontsize=8)
    ax=axes[1,1]
    k=np.arange(301)
    for gamma in [.99,.995,.999,1.]:
        ax.plot(k,gamma**k,label=f'gamma={gamma:g}')
    ax.set(title='Weight on a terminal penalty k decisions away',xlabel='Remaining decision steps k',ylabel='Discount weight')
    ax.legend(fontsize=8)
    for ax in axes.flat:ax.grid(alpha=.2)
    for ext in ('png','pdf','svg'):fig.savefig(OUT/f'decision_horizon.{ext}',dpi=160)
    plt.close(fig)


if __name__=='__main__':main()
