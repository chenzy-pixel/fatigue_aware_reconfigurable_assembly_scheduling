"""Compare inference implementations on identical schema-6 inputs and weights.

The saved mainline network receives the same expanded input dimensions. This
isolates sparse actor/value-only execution from order-estimation overhead.
"""
import importlib.util
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np
import torch
sys.dont_write_bytecode=True
PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
torch.set_num_threads(2)
from configs import load_config
from eval import load_configured_instance
from data import load_dataset_split
from environment import AssemblySchedulingEnv, DecisionType
from agent.baselines import HeuristicPolicy
from agent.ppo import build_actor_critic

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--baseline-ref",default="ba08892")
parser.add_argument("--output-dir",type=Path,default=PROJECT/"result/audits/inference_20261002")
args=parser.parse_args()
OUT=args.output_dir.resolve();OUT.mkdir(parents=True,exist_ok=True)
source=subprocess.check_output(["git","show",f"{args.baseline_ref}:agent/ppo/network.py"],cwd=PROJECT)
(OUT/"mainline_network.py").write_bytes(source)
spec=importlib.util.spec_from_file_location("before_sparse_network",OUT/"mainline_network.py")
legacy=importlib.util.module_from_spec(spec);sys.modules[spec.name]=legacy;spec.loader.exec_module(legacy)
config=load_config("configs/v8/universal.json")
instances=[load_configured_instance(config),load_dataset_split(config,"validation")[0].instance]
samples={DecisionType.PRODUCTION:[],DecisionType.WORKER:[]}
for instance in instances:
    env=AssemblySchedulingEnv(config);obs=env.reset(instance);policy=HeuristicPolicy()
    for _ in range(100):
        if len(samples[obs.decision_type])<8:samples[obs.decision_type].append((obs,env.get_action_mask()))
        if all(len(values)==8 for values in samples.values()):break
        obs,_,done,truncated,_=env.step(policy.select_action(env))
        if done or truncated:break
    if all(len(values)==8 for values in samples.values()):break
records=samples[DecisionType.PRODUCTION]+samples[DecisionType.WORKER]
observations,masks=zip(*records)
counts={"graphs":len(records),"dense_pair_rows":sum(len(mask)-1 for mask in masks),
        "legal_pair_rows":sum(int((~mask[:-1]).sum()) for mask in masks)}
results={}
for device in ["cpu"]+(["cuda"] if torch.cuda.is_available() else []):
    torch.manual_seed(11)
    old=legacy.build_actor_critic(observations[0],config["network"]).to(device).eval()
    new=build_actor_critic(observations[0],config["network"]).to(device).eval()
    with torch.no_grad():
        for prefix in ("production_experts","worker_experts","production_wait_experts","worker_wait_experts"):
            for expert in getattr(old,prefix).experts.values():expert.context[-1].weight.uniform_(-.05,.05)
    new.load_state_dict(old.state_dict(),strict=True)
    def sync():
        if device=="cuda":torch.cuda.synchronize()
    functions={"old_actor":lambda:old.forward_batch(observations,masks,device=device),
        "new_actor":lambda:new.forward_batch(observations,masks,device=device),
        "old_value":lambda:old.forward_batch(observations,masks,device=device)[1],
        "new_value":lambda:new.value_batch(observations,masks,device=device)}
    timings={}
    with torch.no_grad():
        old_logits,old_values=functions["old_actor"]();new_logits,new_values=functions["new_actor"]()
        legal=torch.zeros_like(old_logits,dtype=torch.bool)
        for i,mask in enumerate(masks):legal[i,:len(mask)]=~torch.as_tensor(mask,device=device)
        error=float((old_logits[legal]-new_logits[legal]).abs().max())
        value_error=float((old_values-new_values).abs().max())
        torch.testing.assert_close(old_logits,new_logits,atol=1e-5,rtol=1e-4)
        torch.testing.assert_close(old_values,functions["new_value"](),atol=1e-5,rtol=1e-4)
        for function in functions.values():
            for _ in range(2):function()
        samples_time={name:[] for name in functions}
        for _ in range(7):
            for name,function in functions.items():
                sync();start=time.perf_counter();function();sync()
                samples_time[name].append(1000*(time.perf_counter()-start))
        timings={name:statistics.median(values) for name,values in samples_time.items()}
    results[device]={"median_ms":timings,"actor_speedup":timings["old_actor"]/timings["new_actor"],
        "value_speedup":timings["old_value"]/timings["new_value"],"max_legal_logit_error":error,
        "max_value_error":value_error,"raw_ms":samples_time}
    if device=="cuda":results[device]["gpu"]=torch.cuda.get_device_name()
output={"comparison":"old dense phase-batched execution vs sparse execution; same schema-6 inputs/weights",
    "baseline_ref":args.baseline_ref,"baseline_source_sha256":hashlib.sha256(source).hexdigest(),
    "excludes":"environment construction, time estimation, multiprocessing and training updates",
    "counts":counts,"results":results,"torch":torch.__version__}
(OUT/"benchmark.json").write_text(json.dumps(output,indent=2)+"\n")
print(json.dumps(output,ensure_ascii=False),flush=True)
