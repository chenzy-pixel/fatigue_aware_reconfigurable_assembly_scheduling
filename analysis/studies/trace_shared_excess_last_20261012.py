"""Replay selected validation requests and inspect both policies at identical states."""
from __future__ import annotations
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
RUN=ROOT/"result/runs/remaining_flow_experiments_seed11_20261008_225956_shared_head_universal"
OUT=ROOT/"result/analysis/shared_excess_last_20261012/trace"


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
    from utils import set_seed,action_trace_sha256

    OUT.mkdir(parents=True,exist_ok=True)
    config=load_config(RUN/"config.json");torch.set_num_threads(2);set_seed(11)
    dataset=load_dataset_split(config,"validation")
    records={r.instance.instance_id:r for r in dataset}
    archived=pd.read_csv(RUN/"sampled_validation_instance_metrics.csv",usecols=["validation_episode","instance_id",
        "preference_key","sampling_repeat","action_trace_sha256","task_succeeded","flow_time_objective","reconfiguration_cost","worker_load_variance"])
    models={};captured={};head_scores={};hooks=[]
    bootstrap=AssemblySchedulingEnv(config).reset(dataset[0].instance)
    for role in ("best","last"):
        agent=PPOAgent(build_actor_critic(bootstrap,config["network"]),config["ppo"],device=config["device"])
        agent.load(RUN/(role+"_checkpoint.pt"),load_optimizer=False);agent.network.eval();models[role]=agent
        original=agent.network.forward_batch
        def forward(observations,masks,*,device,role=role,original=original):
            result=original(observations,masks,device=device)
            if len(observations)==1:
                captured[role]=(result[0][0,:len(masks[0])].detach().cpu().numpy(),float(result[1][0].detach().cpu()))
            return result
        agent.network.forward_batch=forward
        for name in ("production_shared","production_wait_shared","worker_shared","worker_wait_shared"):
            def hook(module,inputs,result,role=role,name=name):
                head_scores[(role,name)]=result.detach().cpu().numpy().reshape(-1)
            hooks.append(getattr(agent.network,name).score.register_forward_hook(hook))

    cases=[]
    for seed,repeat in ((2000121,0),(2000177,1),(2000182,2),(2000330,2)):
        iid=next(i for i in records if i.endswith(str(seed)))
        cases.extend([(iid,repeat,"last","archived_failure"),(iid,repeat,"best","matched_best")])
    for seed,repeat in ((2000121,1),(2000177,0),(2000182,0),(2000330,0)):
        iid=next(i for i in records if i.endswith(str(seed)));cases.append((iid,repeat,"last","matched_success_repeat"))
    all_rows=[];summary=[];counterfactual=[]
    pref=PreferenceContext.from_input((0,1,0));start=time.perf_counter()
    for iid,repeat,role,case_kind in cases:
        instance=records[iid].instance
        env=AssemblySchedulingEnv(config);observation=env.reset(instance,preference=pref)
        policy=EvaluationPolicy(config,policy_name="ppo",bootstrap_observation=observation,ppo_agent=models[role],
            decode_mode="sampled",sampling_seed=100011+repeat)
        policy.begin_episode(iid,pref.key);policy.enter_evaluation_mode()
        rows=[];selected_actions=[];snapshots=[]
        case=f"{role}_{instance.instance_id}_{repeat}"
        while not(env.terminated or env.truncated):
            mask=env.get_action_mask();legal=np.flatnonzero(~mask);phase=env.decision_type
            tick=env.current_tick;action=policy.select_action(observation,env);selected_actions.append(action)
            logits,value=captured[role]
            probs=torch.softmax(torch.as_tensor(logits),dim=0).numpy()
            types={int(a):env._action_type(phase,int(a)) for a in legal}
            masses={kind:sum(float(probs[a]) for a in legal if types[int(a)]==kind)
                    for kind in ("DIRECT_PROCESS","COMMIT_RECONFIG","WORKER_ASSIGN","WAIT")}
            name="production_shared" if phase.value=="PRODUCTION" else "worker_shared"
            scores=head_scores.get((role,name),np.array([]))
            pair_saturation=float((np.abs(np.tanh(scores))>.99).mean()) if len(scores) else np.nan
            pair_actions=legal[legal!=len(mask)-1]
            direct_count=sum(types[int(a)]=="DIRECT_PROCESS" for a in legal)
            row={"case":case,"case_kind":case_kind,"role":role,"instance_id":iid,"sampling_repeat":repeat,
                "step":len(rows),"tick":tick,"time":env.current_time,"phase":phase.value,"action":action,
                "action_type":types[action],"legal_count":len(legal),"legal_direct_count":direct_count,
                "chosen_probability":float(probs[action]),"chosen_logit":float(logits[action]),
                "maximum_probability":float(probs.max()),"top_action":int(np.argmax(probs)),
                "entropy":float(-(probs[legal]*np.log(np.maximum(probs[legal],1e-30))).sum()),"critic_value":value,
                "shared_pair_saturation_fraction":pair_saturation,
                **{"probability_mass_"+k:v for k,v in masses.items()},
                "operation_progress":env.operation_progress(),"peak_fatigue_so_far":max(w.peak_fatigue for w in env.workers),
                "worker_fatigue_max":max(w.fatigue for w in env.workers),
                "pending_reconfigurations":sum(m.locked_operation_id is not None for m in env.machines)}
            if row["action_type"] in ("DIRECT_PROCESS","COMMIT_RECONFIG"):
                oi,mi=env.decode_production_action(action);op=env.operations[oi];machine=env.machines[mi]
                row.update(operation_id=op.spec.id,machine_id=machine.spec.id,source_module=machine.current_module,
                    target_module=op.spec.required_module,processing_ticks=env.estimate_processing_ticks(oi,mi),
                    optimistic_reconfiguration_ticks=env.estimate_reconfiguration_ticks(oi,mi),
                    selected_minimum_finish_tick=tick+env.estimate_processing_ticks(oi,mi)+env.estimate_reconfiguration_ticks(oi,mi),
                    horizon_tick=env.horizon_tick)
            elif row["action_type"]=="WORKER_ASSIGN":
                mi,wi=env.decode_worker_action(action)
                row.update(machine_id=env.machines[mi].spec.id,worker_id=env.workers[wi].spec.id,
                    worker_projected_fatigue=env.projected_worker_fatigue(mi,wi))
            top=np.array(sorted(legal,key=lambda a:float(probs[a]),reverse=True)[:5])
            row["top_actions"]=json.dumps([{"action":int(a),"type":types[int(a)],"probability":float(probs[a])} for a in top])
            if len(legal)>1 and (len(rows)%25==0 or row["action_type"]=="COMMIT_RECONFIG" or (row["action_type"]=="WAIT" and direct_count>0)):
                # A preference is the only changed observation field; physics/features/mask stay fixed.
                observations=[replace(observation,preference=np.array(p,dtype=np.float32)) for p in ((1,0,0),(0,1,0),(0,0,1))]
                for model_name,agent in models.items():
                    with torch.no_grad():
                        hypothetical_logits,_=agent.network.forward_batch(observations,[mask]*3,device=agent.device)
                    agent.consume_policy_decision_diagnostics()
                    probabilities=torch.softmax(hypothetical_logits[:,:len(mask)],dim=1).cpu().numpy()
                    for index,label in enumerate(("flow","cost","variance")):
                        pr=probabilities[index];counterfactual.append({"case":case,"step":len(rows),"time":env.current_time,
                            "phase":phase.value,"scoring_model":model_name,"preference":label,
                            "probability_mass_reconfiguration":sum(float(pr[a]) for a in legal if types[int(a)]=="COMMIT_RECONFIG"),
                            "probability_mass_direct":sum(float(pr[a]) for a in legal if types[int(a)]=="DIRECT_PROCESS"),
                            "probability_wait":float(pr[len(mask)-1]),"top_action":int(np.argmax(pr)),
                            "pure_cost_distribution_l1":float(np.abs(pr-probabilities[1]).sum())})
            observation,reward,_,_,info=env.step(action)
            row["reward"]=reward.scalarize(config["reward"])
            row["after_time"]=env.current_time
            row["wait_reason"]=info.get("wait_reason")
            rows.append(row)
        metrics=env.metrics();digest=action_trace_sha256(selected_actions)
        expected=archived[(archived.validation_episode==(400 if role=="best" else 2000))&
            (archived.instance_id==iid)&(archived.preference_key==pref.key)&(archived.sampling_repeat==repeat)].iloc[0]
        record={"case":case,"case_kind":case_kind,"role":role,"instance_id":iid,"repeat":repeat,
            "action_hash_match":digest==expected.action_trace_sha256,"replayed_hash":digest,
            "task_succeeded":metrics["task_succeeded"],"termination_reason":metrics.get("termination_reason",metrics.get("terminal_reason")),
            "decisions":len(rows),"flow_time_objective":metrics["flow_time_objective"],
            "reconfiguration_cost":metrics["reconfiguration_cost"],"worker_load_variance":metrics["worker_load_variance"],
            "completed_reconfigurations":metrics["completed_reconfigurations"],
            "maximum_worker_fatigue":metrics["maximum_worker_fatigue"],"operation_progress":metrics["operation_progress"],
            "all_safe":not env.validate_schedule() and metrics["maximum_worker_fatigue"]<=metrics["safe_fatigue_limit"]+1e-9,
            "elapsed_seconds":time.perf_counter()-start}
        if not record["action_hash_match"]:
            record["interpretation"]="Serial replay differs; this is a new diagnostic sample, not the archived failure"
        summary.append(record);all_rows.extend(rows)
        write_csv(OUT/"decision_trace.csv",all_rows);write_csv(OUT/"trace_summary.csv",summary)
        write_csv(OUT/"preference_counterfactuals.csv",counterfactual)
        write_json(OUT/(case+"_terminal.json"),{"summary":record,"unfinished_operations":[
            {"id":o.spec.id,"state":o.state.value,"module":o.spec.required_module,"machine":o.machine_id,
             "start_tick":o.start_tick,"end_tick":o.end_tick} for o in env.operations if o.state.value!="DONE"],
            "machines":[{"id":m.spec.id,"state":m.state.value,"module":m.current_module,"locked":m.locked_operation_id} for m in env.machines],
            "workers":[{"id":w.spec.id,"state":w.state.value,"fatigue":w.fatigue,"load":w.load} for w in env.workers]})
        print(json.dumps(record),flush=True)
    for hook in hooks:hook.remove()


if __name__=="__main__":main()
