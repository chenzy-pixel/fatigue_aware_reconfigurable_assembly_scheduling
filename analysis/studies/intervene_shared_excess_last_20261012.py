"""Small diagnostic interventions on four reproduced Cost failures; not formal scores."""
from __future__ import annotations
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
RUN=ROOT/"result/runs/remaining_flow_experiments_seed11_20261008_225956_shared_head_universal"
OUT=ROOT/"result/analysis/shared_excess_last_20261012/interventions"


def main():
    import numpy as np
    import pandas as pd
    import torch
    from agent.ppo import PPOAgent,build_actor_critic
    from configs import load_config
    from data import load_dataset_split
    from environment import AssemblySchedulingEnv,PreferenceContext
    from eval import EvaluationPolicy
    from result.io import write_csv,write_json
    from utils import set_seed

    OUT.mkdir(parents=True,exist_ok=True);c=load_config(RUN/"config.json");torch.set_num_threads(2);set_seed(11)
    dataset=load_dataset_split(c,"validation");records={r.instance.seed:r for r in dataset} if False else {
        int(r.instance.instance_id.rsplit('_',1)[-1]):r for r in dataset}
    bootstrap=AssemblySchedulingEnv(c).reset(dataset[0].instance);models={}
    for role in ("best","last"):
        models[role]=PPOAgent(build_actor_critic(bootstrap,c["network"]),c["ppo"],device=c["device"])
        models[role].load(RUN/(role+"_checkpoint.pt"),load_optimizer=False);models[role].network.eval()
    results=[];events=[];counterfactuals=[];start=time.perf_counter();pref=PreferenceContext.from_input((0,1,0))
    for seed,repeat in ((2000121,0),(2000177,1),(2000182,2),(2000330,2)):
        record=records[seed]
        for variant in ("single_late_wait","late_wait_pairs","lower_fatigue_worker","greedy"):
            env=AssemblySchedulingEnv(c);obs=env.reset(record.instance,preference=pref)
            policy=EvaluationPolicy(c,policy_name="ppo",bootstrap_observation=obs,ppo_agent=models["last"],
                decode_mode="greedy" if variant=="greedy" else "sampled",sampling_seed=None if variant=="greedy" else 100011+repeat)
            policy.begin_episode(record.instance.instance_id,pref.key);policy.enter_evaluation_mode();changes=0;step=0
            while not(env.terminated or env.truncated):
                mask=env.get_action_mask();action=policy.select_action(obs,env);original=action;reason=None
                phase=env.decision_type.value
                if variant in {"single_late_wait","late_wait_pairs"} and phase=="PRODUCTION" and action==env.wait_action and env.current_time>=.75*record.instance.horizon:
                    cert=(env._last_action_mask_analysis or {}).get("wait") or {}
                    pair_actions=np.flatnonzero(~mask[:-1])
                    if cert.get("wait_ticks",0)>0 and len(pair_actions) and (variant=="late_wait_pairs" or changes==0):
                        action=int(policy._episode_policy_diagnostics[-1]["final_pair_top_action"])
                        assert action in pair_actions
                        reason="Choose the policy's top pair instead of a time-advancing late WAIT"
                if variant=="lower_fatigue_worker" and phase=="WORKER" and action!=env.wait_action:
                    mi,wi=env.decode_worker_action(action);fp=env.projected_worker_fatigue(mi,wi)
                    alternatives=[]
                    for candidate in np.flatnonzero(~mask[:-1]):
                        mm,ww=env.decode_worker_action(int(candidate))
                        if mm==mi:alternatives.append((env.projected_worker_fatigue(mm,ww),int(candidate)))
                    if fp>=.65 and alternatives and min(alternatives)[0]<=fp-.1:
                        action=min(alternatives)[1];reason="Assign the same stage to a legal worker with lower projected fatigue"
                if reason is not None:
                    # This intervention follows the same random draws up to the changed action.
                    if changes==0:
                        variants=[replace(obs,preference=np.array(p,dtype=np.float32)) for p in ((1,0,0),(0,1,0),(0,0,1))]
                        for role,agent in models.items():
                            with torch.no_grad():logits,_=agent.network.forward_batch(variants,[mask]*3,device=agent.device)
                            agent.consume_policy_decision_diagnostics();probs=torch.softmax(logits[:,:len(mask)],dim=1).cpu().numpy()
                            for j,label in enumerate(("flow","cost","variance")):
                                counterfactuals.append({"instance_id":record.instance.instance_id,"repeat":repeat,"variant":variant,
                                    "step":step,"time":env.current_time,"scoring_model":role,"preference":label,
                                    "probability_original_action":float(probs[j,original]),"probability_alternative":float(probs[j,action]),
                                    "probability_wait":float(probs[j,len(mask)-1]),"distribution_l1_vs_cost":float(np.abs(probs[j]-probs[1]).sum())})
                    changes+=1;events.append({"instance_id":record.instance.instance_id,"repeat":repeat,"variant":variant,
                        "step":step,"time":env.current_time,"phase":phase,"original_action":original,"alternative":action,"reason":reason})
                obs,_,_,_,_=env.step(action);step+=1
            m=env.metrics();row={"instance_id":record.instance.instance_id,"repeat":repeat,"variant":variant,"changes":changes,
                "success":m["task_succeeded"],"termination":m.get("terminal_reason"),"decisions":step,
                "flow":m["flow_time_objective"],"cost":m["reconfiguration_cost"],"variance":m["worker_load_variance"],
                "reconfigurations":m["completed_reconfigurations"],"fatigue_peak":m["maximum_worker_fatigue"],
                "progress":m["operation_progress"],"all_safe":not env.validate_schedule() and m["maximum_worker_fatigue"]<=m["safe_fatigue_limit"]+1e-9}
            results.append(row);write_csv(OUT/"outcomes.csv",results);write_csv(OUT/"intervention_events.csv",events)
            write_csv(OUT/"fixed_state_preference_responses.csv",counterfactuals)
            write_json(OUT/"progress.json",{"completed":len(results),"total":16,"elapsed_seconds":time.perf_counter()-start})
            print(json.dumps(row),flush=True)


if __name__=="__main__":main()
