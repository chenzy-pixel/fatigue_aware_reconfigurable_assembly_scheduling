"""Compare deterministic initial values with three saved returns per validation request."""
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
RUN=ROOT/"result/runs/remaining_flow_experiments_seed11_20261008_225956_shared_head_universal"
OUT=ROOT/"result/analysis/shared_excess_last_20261012"


def main():
    import numpy as np
    import pandas as pd
    import torch
    from configs import load_config
    from configs.formal_preferences import formal_preferences
    from data import load_dataset_split
    from environment import AssemblySchedulingEnv
    from agent.ppo import PPOAgent,build_actor_critic
    from result.io import write_csv

    c=load_config(RUN/"config.json");torch.set_num_threads(2)
    dataset=load_dataset_split(c,"validation");indices=json.loads((RUN/"validation_subsets.json").read_text())["target"]["instance_indices"]
    grid=formal_preferences(c,"validation");inputs=[];masks=[];keys=[]
    for index in indices:
        record=dataset[index];env=AssemblySchedulingEnv(c);obs=env.reset(record.instance);mask=env.get_action_mask()
        for pref in grid:
            inputs.append(replace(obs,preference=pref.as_array()));masks.append(mask)
            keys.append((record.instance.instance_id,pref.key,pref.preference.cost))
    raw=pd.read_csv(RUN/"sampled_validation_instance_metrics.csv",usecols=["validation_episode","instance_id","preference_key",
        "training_cumulative_reward","task_succeeded","pressure_type"])
    outputs=[];summaries=[]
    for role,episode in (("best",400),("last",2000)):
        agent=PPOAgent(build_actor_critic(inputs[0],c["network"]),c["ppo"],device=c["device"])
        agent.load(RUN/(role+"_checkpoint.pt"),load_optimizer=False);agent.network.eval();values=[]
        with torch.no_grad():
            for first in range(0,len(inputs),20):values.extend(agent.network.value_batch(inputs[first:first+20],masks[first:first+20],device=agent.device).cpu().tolist())
        observed=raw[raw.validation_episode==episode].groupby(["instance_id","preference_key"]).agg(
            observed_mean_return=("training_cumulative_reward","mean"),observed_completion=("task_succeeded","mean"),
            pressure_type=("pressure_type","first"))
        rows=[]
        for (iid,key,cost_weight),value in zip(keys,values):
            measured=observed.loc[(iid,key)];rows.append({"role":role,"instance_id":iid,"preference_key":key,
                "cost_weight":cost_weight,"value_prediction":value,**measured.to_dict(),
                "error":value-measured.observed_mean_return})
        frame=pd.DataFrame(rows);outputs.extend(rows)
        for label,subset in (("all",frame),("pure_cost",frame[frame.cost_weight==1]),
            ("pure_cost_failed_repeat",frame[(frame.cost_weight==1)&(frame.observed_completion<1)])):
            summaries.append({"role":role,"subset":label,"requests":len(subset),"prediction_mean":subset.value_prediction.mean(),
                "observed_return_mean":subset.observed_mean_return.mean(),"bias":subset.error.mean(),
                "mae":subset.error.abs().mean(),"rmse":np.sqrt((subset.error**2).mean())})
    write_csv(OUT/"critic_initial_calibration.csv",outputs);write_csv(OUT/"critic_initial_calibration_summary.csv",summaries)
    print(json.dumps(summaries,indent=2))


if __name__=="__main__":main()
