"""Fixed-state sensitivity to objective globals; not a retrained global-vector ablation."""
from dataclasses import replace
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
RUN=ROOT/'result/runs/remaining_flow_experiments_seed11_20261008_225956_shared_head_universal'
OUT=ROOT/'result/analysis/shared_head_design_audit_20261012'


def main():
    from agent.ppo import PPOAgent,build_actor_critic
    from configs import load_config
    from data import load_dataset_split
    from environment import AssemblySchedulingEnv,PreferenceContext
    from result.io import write_csv,write_json
    from utils import set_seed
    c=load_config(RUN/'config.json');torch.set_num_threads(2);set_seed(11)
    records={r.instance.instance_id:r for r in load_dataset_split(c,'validation')}
    archived=pd.read_csv(ROOT/'result/analysis/shared_excess_last_20261012/trace/decision_trace.csv')
    bootstrap=AssemblySchedulingEnv(c).reset(next(iter(records.values())).instance)
    agents={}
    for role in ['best','last']:
        agent=PPOAgent(build_actor_critic(bootstrap,c['network']),c['ppo'],device=c['device'])
        agent.load(RUN/(role+'_checkpoint.pt'),load_optimizer=False);agent.network.eval();agents[role]=agent
    rows=[];controls=[]
    for iid in ['validation_reconfiguration_bottleneck_2000177','validation_reconfiguration_bottleneck_2000330']:
        trace=archived[(archived.instance_id==iid)&(archived.role=='last')&(archived.case_kind=='archived_failure')].sort_values('step')
        env=AssemblySchedulingEnv(c);obs=env.reset(records[iid].instance,preference=PreferenceContext.from_input((0,1,0)))
        for _,row in trace.iterrows():
            mask=env.get_action_mask();action=int(row.action);assert not mask[action]
            assert abs(env.current_time-row.time)<1e-9
            if (~mask).sum()>1 and (int(row.step)%20==0 or (row.action_type=='WAIT' and env.current_time>=180)):
                base=obs.global_features.copy();bound_var=base.copy();bound_var[8]=base[8]/(1+base[8])
                bound_all=base.copy();bound_all[6:9]=base[6:9]/(1+base[6:9])
                zero=base.copy();zero[6:9]=0
                names=['original','bounded_variance','bounded_objectives','zero_objectives']
                variants=[replace(obs,global_features=g) for g in [base,bound_var,bound_all,zero]]
                for role,agent in agents.items():
                    with torch.no_grad():logits,values=agent.network.forward_batch(variants,[mask]*4,device=agent.device)
                    agent.consume_policy_decision_diagnostics()
                    probs=torch.softmax(logits[:,:len(mask)],dim=1).cpu().numpy();values=values.cpu().numpy()
                    weight=agent.network.global_encoder[0].weight.detach().cpu().numpy()
                    contribution=np.abs(base)*np.linalg.norm(weight,axis=0)
                    if role=='last':
                        error=abs(float(probs[0,action])-float(row.chosen_probability))
                        assert error<1e-5 and abs(float(values[0])-row.critic_value)<1e-4
                        controls.append({'instance_id':iid,'step':int(row.step),'probability_error':error})
                    for index,name in enumerate(names):
                        rows.append({'instance_id':iid,'step':int(row.step),'time':env.current_time,'phase':env.decision_type.value,
                            'model':role,'variant':name,'global_flow':float(base[6]),'global_cost':float(base[7]),'global_variance':float(base[8]),
                            'sum_column_norm_share_objectives':float(contribution[6:9].sum()/max(1e-12,contribution.sum())),
                            'wait_probability':float(probs[index,-1]),'chosen_probability':float(probs[index,action]),
                            'value':float(values[index]),'distribution_l1_vs_original':float(np.abs(probs[index]-probs[0]).sum()),
                            'top_action_changed':bool(probs[index].argmax()!=probs[0].argmax())})
            obs,*_=env.step(action)
        assert env.task_failed and not env.truncated
        print(json.dumps({'completed_case':iid,'states_checked':sum(r['instance_id']==iid for r in controls)}),flush=True)
        write_csv(OUT/'global_fixed_state_probe.csv',rows);write_csv(OUT/'global_probe_controls.csv',controls)
    f=pd.DataFrame(rows)
    agg=f.groupby(['model','variant']).agg(states=('step','size'),mean_l1=('distribution_l1_vs_original','mean'),
        maximum_l1=('distribution_l1_vs_original','max'),top_changed_fraction=('top_action_changed','mean'),
        mean_wait_probability=('wait_probability','mean'),mean_value=('value','mean')).reset_index()
    write_csv(OUT/'global_probe_summary.csv',agg.to_dict('records'))
    write_json(OUT/'global_probe_verification.json',{'states':len(controls),'all_original_policy_probabilities_match':True,
        'maximum_probability_error':max(r['probability_error'] for r in controls),
        'interpretation':'Fixed physical states; changing trained input encoding measures sensitivity, not retraining effect or cause of regression.'})
    print(agg.to_string(index=False))


if __name__=='__main__':main()
