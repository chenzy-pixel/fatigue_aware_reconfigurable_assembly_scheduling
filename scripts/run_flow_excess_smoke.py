"""Actual decision-step audit plus short matched PPO training/evaluation."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
import torch

from agent.baselines import HeuristicPolicy
from configs import load_config
from data import load_dataset_split
from data.selection import select_validation_subsets
from environment import AssemblySchedulingEnv
from environment.time_context import project_wait_state
from result.io import write_csv,write_json
from train import train


def audit_fixed_validation(output):
    config=load_config("configs/flow_excess/universal.json")
    dataset=load_dataset_split(config,"validation")
    selection=select_validation_subsets(dataset,config["generator"]["dataset_pressure_weights"],
                                        target_count=50,diagnostic_count=0)["target"]
    write_json(output/"validation_subset.json",selection)
    waits,steps,episode_rows=[],[],[]
    policy=HeuristicPolicy()
    for count,index in enumerate(selection["instance_indices"],1):
        record=dataset[index]
        raw=AssemblySchedulingEnv(load_config("configs/default.json"))
        env=AssemblySchedulingEnv(config)
        raw.reset(record.instance,build_observation=False); env.reset(record.instance,build_observation=False)
        started=time.perf_counter(); selected_times=[]; total=0; step_index=0
        while not env.task_done:
            mask=env.get_action_mask()
            certificate=env._wait_certificate()
            if certificate["allowed"]:
                start=time.perf_counter()
                projected=project_wait_state(env,certificate["wait_ticks"],settle_terminal=True,certificate=certificate)
                elapsed=time.perf_counter()-start; selected_times.append(elapsed)
                delta=projected.flow_objective()-env.flow_objective()
                waits.append({"instance_id":record.instance.instance_id,"step":step_index,
                              "time":env.current_time,"phase":env.decision_type.value,
                              "wait_ticks":certificate["wait_ticks"],"flow_delta":delta,
                              "feature":delta/368.3143,"projected_failed":projected.task_failed,
                              "projected_succeeded":projected.task_succeeded,"projection_seconds":elapsed})
            action=policy.select_action(env)
            assert action==policy.select_action(raw)
            flow_before=env.flow_objective()
            _,reward,_,_,info=env.step(action,build_observation=False)
            raw.step(action,build_observation=False)
            value=reward.scalarize(config["reward"]); total+=value
            if action==len(mask)-1 and certificate["allowed"]:
                assert abs((env.flow_objective()-flow_before)-delta)<1e-8
            steps.append({"instance_id":record.instance.instance_id,"step":step_index,
                          "time":env.current_time,"action":action,"action_type":info["action_type"],
                          "flow_excess":env.flow_excess_objective(),"raw_flow":raw.flow_objective(),
                          "quality_reward":reward.quality,"progress_reward":reward.operation_progress,
                          "failure_reward":reward.failure,"reward":value})
            step_index+=1
        metrics=env.metrics()
        assert env.schedule_log==raw.schedule_log and env.reconfiguration_log==raw.reconfiguration_log
        episode_rows.append({"instance_id":record.instance.instance_id,"success":env.task_succeeded,
                             "flow":metrics["flow_time_objective"],"excess":metrics["flow_excess_objective"],
                             "return":total,"decisions":step_index,"wall_seconds":time.perf_counter()-started,
                             "projection_seconds":sum(selected_times),"schedule_violations":len(env.validate_schedule())})
        if count%10==0: print(f"Flow fixed validation audit {count}/50",flush=True)
    write_csv(output/"decision_steps.csv",steps)
    write_csv(output/"legal_wait_projections.csv",waits)
    write_csv(output/"fixed_validation_metrics.csv",episode_rows)
    features=np.array([row["feature"] for row in waits]); times=np.array([row["projection_seconds"] for row in waits])
    summary={"episodes":50,"successful":sum(row["success"] for row in episode_rows),
             "legal_wait_states":len(waits),"wait_terminal_failure_fraction":float(np.mean([row["projected_failed"] for row in waits])),
             "wait_feature_quantiles":dict(zip(("p50","p95","p99","max"),np.quantile(features,[.5,.95,.99,1]).tolist())),
             "projection_seconds_quantiles":dict(zip(("p50","p95","p99","max"),np.quantile(times,[.5,.95,.99,1]).tolist())),
             "scope":"50 fixed instances, deterministic heuristic, every legal WAIT; not converged PPO"}
    write_json(output/"wait_audit_summary.json",summary)
    return summary


def short_training(output,*,existing_raw=None):
    torch.set_num_threads(2)
    run_dirs={}
    if existing_raw is not None:
        config=load_config(Path(existing_raw)/"config.json")
        if config["objective_scalarizer"].get("flow_mode","raw_v1")!="raw_v1":
            raise ValueError("existing raw smoke must have raw Flow identity")
        run_dirs["raw"]=str(Path(existing_raw).resolve())
    for mode,path in (("raw","configs/v8/universal.json"),("excess","configs/flow_excess/universal.json")):
        if mode in run_dirs: continue
        config=load_config(path)
        config["device"]="cpu"
        config["network"].update({"hidden_dim":32,"message_passing_layers":1})
        config["training"].update({"smoke_episodes":2,"smoke_rollout_steps":16,
            "smoke_validation_instance_limit":1,"smoke_parallel_envs":2,
            "validation_parallel_envs":8,"torch_num_threads":2})
        config["training"]["formal_evaluation"]["validation_repeats"]=1
        config["training"]["formal_evaluation"]["final_test_repeats"]=1
        config.setdefault("logging",{})["save_rollout_decisions"]=True
        config["training"]["worker_local_physical_forced_actions"]=False
        directory=train(config,smoke=True,run_name=f"flow_{mode}_smoke_{datetime.now():%Y%m%d_%H%M%S}",
                        parallel_envs=2,validation_parallel_envs=8,visdom_enabled=False)
        run_dirs[mode]=str(directory)
        print(f"Flow {mode} PPO smoke completed: {directory}",flush=True)
    from analysis.v8_pareto_analysis import analyze_dual_flow_runs
    analysis=analyze_dual_flow_runs(run_dirs,output/"matched_hv")
    write_json(output/"smoke_runs.json",run_dirs)
    return analysis


def replay_native_states(output):
    """Check the native state function against every recorded historical event snapshot."""
    import pandas as pd
    from scripts.replay_flow_excess import build_replay
    from environment.types import OperationState
    config=load_config("configs/flow_excess/universal.json")
    instances={record.instance.instance_id:record.instance for record in load_dataset_split(config,"test")}
    original=ROOT/"result/analysis/flow_excess_replay_20261008"
    certificates=pd.read_csv(original/"terminal_certificates.csv")
    rows=[]
    for arm in certificates.arm.unique():
        directory=ROOT/f"result/runs/ablation_eval_{arm}_seed11_20261007_232505_102441"
        metrics=pd.read_csv(directory/"instance_metrics.csv"); schedules=pd.read_csv(directory/"schedule.csv")
        for metric in metrics.to_dict("records"):
            iid,repeat=metric["instance_id"],metric["sampling_repeat"]
            env=AssemblySchedulingEnv(config); env.reset(instances[iid],build_observation=False)
            schedule=schedules[(schedules.instance_id==iid)&(schedules.sampling_repeat==repeat)]
            replay=build_replay(instances[iid],metric,schedule,env)
            trace=pd.read_csv(original/"traces"/f"{arm}_{iid}_repeat{repeat}_events.csv")
            error=0.0
            for row in trace.to_dict("records"):
                tick=int(row["tick"]); env.current_tick=tick
                env._flow_integral=row["raw_flow"]; env._flow_penalty=row["failure_flow_penalty"]
                for op in env.operations:
                    op.state=OperationState.UNRELEASED; op.planned_duration_ticks=None; op.start_tick=None
                for interval in replay.intervals:
                    if tick<interval.start: continue
                    op=env.operations[env.instance.operation_index[interval.operation_id]]
                    op.start_tick=interval.start; op.planned_duration_ticks=interval.planned_duration
                    op.state=OperationState.DONE if interval.completed and tick>=interval.observed_end else OperationState.PROCESSING
                error=max(error,abs(env.flow_excess_objective()-row["excess_proportional"]))
            assert error<1e-8
            rows.append({"arm":arm,"instance_id":iid,"sampling_repeat":repeat,"states":len(trace),"max_error":error})
    write_csv(output/"native_historical_replay.csv",rows)
    print(f"Native historical replay passed: {len(rows)} trajectories",flush=True)


def pre_update_gradient_variance(agent,buffer):
    """Population trace of per-transition PPO-loss gradients at fixed parameters."""
    parameters=tuple(agent.network.parameters())
    advantages=torch.tensor([t.advantage for t in buffer.transitions])
    advantages=(advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-8)
    first_moment=[torch.zeros_like(p,dtype=torch.float64) for p in parameters]
    squared_norm=0.0
    for transition,advantage in zip(buffer.transitions,advantages):
        logits,value=agent.network.forward_batch([transition.observation],[transition.action_mask],device=agent.device)
        distribution=torch.distributions.Categorical(logits=logits)
        logp=distribution.log_prob(torch.tensor([transition.action]))
        ratio=torch.exp(logp-transition.log_probability)
        policy=-torch.minimum(ratio*advantage,torch.clamp(
            ratio,1-agent.config["clip_epsilon"],1+agent.config["clip_epsilon"])*advantage).mean()
        loss=(policy+agent.config["value_coefficient"]*(value-transition.return_value).square().mean()
              -agent.config["entropy_coefficient"]*distribution.entropy().mean())
        gradients=torch.autograd.grad(loss,parameters,allow_unused=True)
        for mean,gradient in zip(first_moment,gradients):
            if gradient is not None:
                gradient=gradient.detach().double()
                mean.add_(gradient)
                squared_norm+=float(gradient.square().sum())
    count=len(buffer)
    mean_norm=sum(float((value/count).square().sum()) for value in first_moment)
    return max(0.0,squared_norm/count-mean_norm)


def ppo_decision_audit(output):
    """Capture actual sampled physical decisions, bootstrap and GAE for a small PPO update."""
    from agent.ppo import PPOAgent,build_actor_critic
    from agent.ppo.buffer import RolloutBuffer
    torch.set_num_threads(2)
    rows=[]; updates=[]; waits=[]
    for label,path in (("raw","configs/v8/universal.json"),("excess","configs/flow_excess/universal.json")):
        torch.manual_seed(11)
        config=load_config(path); config["network"].update(hidden_dim=32,message_passing_layers=1)
        dataset=load_dataset_split(config,"validation")
        for episode in range(2):
            record=dataset[episode]; env=AssemblySchedulingEnv(config)
            obs=env.reset(record.instance,preference=(1,0,0))
            agent=PPOAgent(build_actor_critic(obs,config["network"]),config["ppo"],device="cpu") if episode==0 else agent
            buffer=RolloutBuffer(); total=0
            for step in range(32):
                mask=env.get_action_mask(); action,logp,value=agent.act(obs,mask)
                certificate=env._wait_certificate()
                if certificate["allowed"]:
                    started=time.perf_counter()
                    projected=project_wait_state(env,certificate["wait_ticks"],settle_terminal=True,certificate=certificate)
                    elapsed=time.perf_counter()-started
                    waits.append({"mode":label,"episode":episode,"step":step,
                                  "instance_id":record.instance.instance_id,
                                  "projected_failed":projected.task_failed,"projected_succeeded":projected.task_succeeded,
                                  "feature":(projected.flow_objective()-env.flow_objective())/config["objective_scalarizer"]["scales"]["flow"],
                                  "projection_seconds":elapsed})
                following,reward,_,_,info=env.step(action)
                scalar=reward.scalarize(config["reward"]); total+=scalar
                buffer.add(obs,mask,action,logp,value,scalar,env.task_done)
                obs=following
                if env.task_done: break
            buffer.compute_gae(last_value=0 if env.task_done else agent.value(obs,env.get_action_mask()),gamma=1,gae_lambda=config["ppo"]["gae_lambda"])
            for index,t in enumerate(buffer.transitions):
                rows.append({"mode":label,"episode":episode,"instance_id":record.instance.instance_id,
                             "step":index,"time_ratio":float(t.observation.global_features[0]),
                             "action":t.action,"reward":t.reward,"value":t.value,
                             "advantage":t.advantage,"return_value":t.return_value,"done":t.done})
            gradient_variance=pre_update_gradient_variance(agent,buffer)
            result=agent.update(buffer)
            assert all(np.isfinite(v) for v in result.values())
            updates.append({"mode":label,"episode":episode,"physical_steps":len(buffer),"reward":total,
                            "pre_update_per_transition_gradient_variance_trace":gradient_variance,
                            "decision_reward_variance":float(np.var([t.reward for t in buffer.transitions])),
                            "gae_advantage_variance":float(np.var([t.advantage for t in buffer.transitions])),**result})
    write_csv(output/"ppo_decision_steps.csv",rows); write_csv(output/"ppo_decision_updates.csv",updates)
    write_csv(output/"ppo_legal_wait_projections.csv",waits)
    wait_summary={}
    for mode in ("raw","excess"):
        selected=[row for row in waits if row["mode"]==mode]
        wait_summary[mode]={"legal_wait_states":len(selected),
            "failure_terminal_fraction":float(np.mean([row["projected_failed"] for row in selected])) if selected else None,
            "success_terminal_fraction":float(np.mean([row["projected_succeeded"] for row in selected])) if selected else None,
            "feature_quantiles":dict(zip(("p50","p95","p99","max"),np.quantile([row["feature"] for row in selected],[.5,.95,.99,1]).tolist())) if selected else {},
            "projection_seconds_quantiles":dict(zip(("p50","p95","p99","max"),np.quantile([row["projection_seconds"] for row in selected],[.5,.95,.99,1]).tolist())) if selected else {}}
    write_json(output/"ppo_wait_audit_summary.json",{"modes":wait_summary,"scope":"two untrained sampled PPO rollouts per mode, 32 decisions each"})
    print(f"Actual PPO decision audit completed: {len(rows)} physical steps",flush=True)


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--phase",choices=("all","audit","train","replay","decisions"),default="all")
    parser.add_argument("--raw-run",type=Path)
    args=parser.parse_args()
    output=ROOT/"result/analysis/flow_excess_integration_20261008"
    output.mkdir(parents=True,exist_ok=True)
    if args.phase in {"all","audit"}: audit_fixed_validation(output)
    if args.phase in {"all","train"}: short_training(output,existing_raw=args.raw_run)
    if args.phase in {"all","replay"}: replay_native_states(output)
    if args.phase in {"all","decisions"}: ppo_decision_audit(output)


if __name__=="__main__": main()
