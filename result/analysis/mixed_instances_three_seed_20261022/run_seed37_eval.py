"""Evaluate mixed-instance best checkpoints on the frozen seven search cases."""
import os
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"

import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from agent.ppo import PPOAgent, build_actor_critic
from analysis.challenge_study import _parallel_requests, atomic_json, clean, delivery_rows
from configs import load_config
from configs.config import public_config
from configs.runtime import assert_checkpoint_fatigue_mode
from data import load_dataset_split
from data.challenge import digest, read_sealed, seal
from data.dataset import sha256_file
from environment import AssemblySchedulingEnv, PreferenceContext
from result.io import write_csv
from result.metrics import successful_row
from utils import configured_formal_evaluation_sampling_seeds, action_trace_sha256, set_seed

OUT = Path(__file__).resolve().parent
SEARCH = ROOT / "result/analysis/heuristics_seed11_20261019"
WORKERS = 8
MODELS = ((37, 'experiment_seed37_20261021_141740_420983_01_mixed_instances', 'a8c6752fbf23f73fcd59cf693f7eefd9bbf3aaf08a489afff4f96e3d1252eefd', 1900),)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pref_key(pref):
    return tuple(round(float(x), 8) for x in pref)


def pref_label(pref):
    return {(1.,0.,0.): "Flow", (0.,1.,0.): "Cost", (0.,0.,1.): "OWA"}.get(pref_key(pref), "Equal")


def sources():
    paths = [Path(__file__)]
    for directory in ("agent", "environment", "configs", "data", "result"):
        paths.extend((ROOT/directory).glob("*.py"))
        if directory == "agent":
            paths.extend((ROOT/"agent/ppo").glob("*.py"))
    paths.extend([ROOT/"analysis/challenge_study.py", ROOT/"eval.py", ROOT/"utils.py"])
    return {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in paths}


def preflight():
    remote = read(SEARCH/"identity.json")
    population = read(SEARCH/"search_selection.json")["population_sizes"]
    search_jobs = [read(p) for p in sorted((SEARCH/"tune_search").glob("*/*/*.json"))]
    assert len(search_jobs) == 168
    requests = [j for j in search_jobs if j["summary"]["method"] == "nsga2_ls" and j["summary"]["population_size"] == population["nsga2_ls"]]
    ids = sorted({j["summary"]["instance_id"] for j in requests})
    prefs = [PreferenceContext.from_input(x) for x in ((1,0,0),(0,1,0),(0,0,1),(1/3,)*3)]
    assert len(ids) == 7 and len(requests) == 28
    expected_cells = {(iid, pref_key(p.as_tuple())) for iid in ids for p in prefs}
    for method, n in population.items():
        own = [j["summary"] for j in search_jobs if j["summary"]["method"] == method and j["summary"]["population_size"] == n]
        assert {(r["instance_id"],pref_key(r["preference"])) for r in own} == expected_cells
    models = []
    for seed, name, expected_sha, episode in MODELS:
        folder = ROOT/"result/runs"/name
        checkpoint = folder/"best_checkpoint.pt"
        assert sha256_file(checkpoint) == expected_sha, (seed,"checkpoint file changed")
        config = load_config(folder/"config.json")
        summary = read(folder/"summary.json")
        assert config["seed"] == seed and summary["checkpoint_selection"]["best_episode"] == episode
        for key in ("environment","reward","objective_scalarizer","workload_objective","network"):
            # Loading represents objective_scales as a tuple; saved JSON uses a list.
            assert json.loads(json.dumps(config[key])) == remote["config"][key], (seed,key)
        generator_keys = set(config["generator"]) - {"curriculum", "severity_curriculum"}
        assert {k:config["generator"][k] for k in generator_keys} == {k:remote["config"]["generator"][k] for k in generator_keys}
        sampling = configured_formal_evaluation_sampling_seeds(config,"validation")
        assert len(sampling) == 3 and config["training"]["formal_evaluation"]["sample_aggregation"] == "best_of_3_v1"
        dataset = load_dataset_split(config,"validation")
        manifest_hash = sha256_file(dataset.manifest_path)
        assert manifest_hash == read(SEARCH/"subsets.json")["tuning"]["dataset_manifest_sha256"]
        records = [r for iid in ids for r in dataset if r.instance.instance_id == iid]
        assert len(records) == 7
        models.append({"seed":seed,"run":name,"checkpoint_sha256":expected_sha,"checkpoint_episode":episode,
            "config_sha256":sha256_file(folder/"config.json"),"sampling_seeds":sampling})
    hardware = {"hostname":platform.node(),"platform":platform.platform(),"python":platform.python_version(),
        "torch":torch.__version__,"numpy":np.__version__,"device":"cuda","gpu":torch.cuda.get_device_name(0)}
    value = {"version":"existing_ppo_search_backfill_v1","instance_ids":ids,"preferences":[list(p.as_tuple()) for p in prefs],
        "models":models,"manifest_sha256":manifest_hash,"source_hashes":sources(),"hardware":hardware,
        "parallel_envs":WORKERS,"sampling_namespace":"validation","delivery_protocol":"best_of_3_v1",
        "expected_deliveries":28,"expected_raw_trajectories":84,"remote_search_identity":remote["sha256"],
        "search_inputs_sha256":{p.relative_to(SEARCH).as_posix():sha256_file(p) for p in sorted((SEARCH/"tune_search").glob("*/*/*.json"))},
        "interpretation":"post-hoc tuning-set comparison of frozen existing PPO; not independent confirmation or matched retraining; parallel timings are not isolated latency"}
    plan = seal(value)
    if (OUT/"plan.json").exists():
        assert read_sealed(OUT/"plan.json") == plan, "Frozen evaluation identity changed"
    else:
        atomic_json(OUT/"plan.json",plan)
    return plan, prefs


def evaluate(plan,prefs):
    for model in plan["models"]:
        seed = model["seed"]
        run = ROOT/"result/runs"/model["run"]
        config = load_config(run/"config.json")
        config["device"] = "cuda"
        config["training"]["validation_parallel_envs"] = WORKERS
        torch.set_num_threads(int(config["training"].get("torch_num_threads",4)))
        set_seed(seed)
        dataset = load_dataset_split(config,"validation")
        byid = {r.instance.instance_id:r for r in dataset}
        records = [byid[iid] for iid in plan["instance_ids"]]
        folder = OUT/f"seed{seed}";folder.mkdir(exist_ok=True)
        identity = seal({"content_plan":plan["content_sha256"],"parallel_envs":WORKERS,"model":model,
                         "effective_config":public_config(config)})
        if (folder/"identity.json").exists():
            assert read_sealed(folder/"identity.json") == identity
        atomic_json(folder/"identity.json",identity)
        atomic_json(folder/"config.json",public_config(config))
        observation = AssemblySchedulingEnv(config).reset(records[0].instance)
        agent = PPOAgent(build_actor_critic(observation,config["network"]),config["ppo"],device=config["device"])
        metadata = agent.load(run/"best_checkpoint.pt",load_optimizer=False)
        assert metadata["checkpoint_episode"] == model["checkpoint_episode"]
        assert_checkpoint_fatigue_mode(metadata,config)
        atomic_json(folder/"checkpoint_metadata.json",clean(metadata))
        began = time.perf_counter()
        print(f"START seed={seed} best_ep={model['checkpoint_episode']} requests=28 trajectories=84 workers={WORKERS}",flush=True)
        paths = _parallel_requests(config,agent,records,prefs,model["sampling_seeds"],identity,folder)
        assert len(paths) == 28 and len({p['path'] for p in paths}) == 28
        assert sha256_file(run/"best_checkpoint.pt") == model["checkpoint_sha256"]
        atomic_json(folder/"completed.json",seal({"identity":identity["content_sha256"],"paths":paths,
            "elapsed_seconds_this_invocation":time.perf_counter()-began,"checkpoint_unchanged":True}))
        print(f"DONE seed={seed} wall_seconds={time.perf_counter()-began:.1f}",flush=True)
        del agent
        torch.cuda.empty_cache()


def summarize(plan):
    raw_rows,selected,pairings = [],[],[]
    for model in plan["models"]:
        seed=model["seed"];folder=OUT/f"seed{seed}"
        identity=read_sealed(folder/"identity.json");completed=read_sealed(folder/"completed.json")
        assert completed["identity"] == identity["content_sha256"]
        for ref in completed["paths"]:
            path=folder/ref["path"]; assert sha256_file(path) == ref["sha256"]
            request=read_sealed(path);assert request["identity_sha256"]==identity["content_sha256"]
            assert len(request["raw"])==3
            raw=request["raw"]
            assert request["selected_repeat"]==delivery_rows(raw)
            for sample in raw:
                metrics=sample["metrics"]
                assert action_trace_sha256(sample["actions"])==metrics["action_trace_sha256"]
                assert not metrics.get("sampling_truncated",metrics.get("truncated"))
                assert not metrics["schedule_violations"] and metrics["maximum_worker_fatigue"]<=metrics["safe_fatigue_limit"]+1e-9
                row={"ppo_seed":seed,"checkpoint_episode":model["checkpoint_episode"],"instance_id":sample["instance_id"],
                    "pressure_type":request["metadata"]["pressure_type"],"preference":pref_label(sample["preference"]),
                    "preference_vector":sample["preference"],"sampling_repeat":sample["sampling_repeat"],
                    "selected":sample["selected_for_aggregation"],"success":successful_row(metrics),
                    "quality":metrics["preference_quality_score"] if successful_row(metrics) else 1.,
                    "flow":metrics["flow_excess_objective"],"cost":metrics["reconfiguration_cost"],"owa":metrics["workload_objective_value"],
                    "reconfigurations":metrics["completed_reconfigurations"],"solve_seconds":metrics["solve_time_seconds"],
                    "request_worker_solve_sum_seconds":sum(s["metrics"]["solve_time_seconds"] for s in raw),
                    "actions_sha256":metrics["action_trace_sha256"],"sampling_seed":metrics["sampling_seed"]}
                raw_rows.append(row)
                if row["selected"]:selected.append(row)
    assert len(raw_rows)==84 and len(selected)==28
    populations=read(SEARCH/"search_selection.json")["population_sizes"]
    lookup={(r["instance_id"],pref_key(r["preference_vector"]),r["ppo_seed"]):r for r in selected}
    assert len(lookup)==28
    for path in sorted((SEARCH/"tune_search").glob("*/*/*.json")):
        search=read(path)["summary"];method=search["method"]
        if search["population_size"]!=populations[method]:continue
        for model in plan["models"]:
            ppo=lookup[search["instance_id"],pref_key(search["preference"]),model["seed"]]
            pairings.append({"search_method":method,"population_size":search["population_size"],"ppo_seed":model["seed"],
                "instance_id":ppo["instance_id"],"pressure_type":ppo["pressure_type"],"preference":ppo["preference"],
                "ppo_success":ppo["success"],"search_success":search["success"],"ppo_Q":ppo["quality"],"search_Q":search["quality"],
                "search_minus_ppo_Q":search["quality"]-ppo["quality"],
                "search_better":(search["success"] and not ppo["success"]) or (search["success"]==ppo["success"] and search["quality"]<ppo["quality"]-1e-10),
                "ppo_flow":ppo["flow"],"search_flow":search["objectives"][0],"ppo_cost":ppo["cost"],"search_cost":search["objectives"][1],
                "ppo_owa":ppo["owa"],"search_owa":search["objectives"][2],
                "search_seconds":search["search_seconds"],"ppo_parallel_worker_seconds_sum":ppo["request_worker_solve_sum_seconds"]})
    assert len(pairings)==56
    write_csv(OUT/"raw_samples.csv",raw_rows);write_csv(OUT/"selected_samples.csv",selected);write_csv(OUT/"paired_comparison.csv",pairings)
    summary={"plan_sha256":plan["content_sha256"],"raw_trajectories":len(raw_rows),"deliveries":len(selected),"paired_rows":len(pairings),
        "ppo_by_seed":{str(m["seed"]):{"successes":sum(r["success"] for r in selected if r["ppo_seed"]==m["seed"]),
            "quality":float(np.mean([r["quality"] for r in selected if r["ppo_seed"]==m["seed"]]))} for m in plan["models"]},
        "all_raw_constraints_pass":True,"all_raw_complete_without_engineering_truncation":True,"checkpoint_files_unchanged":True,
        "interpretation":plan["interpretation"]}
    atomic_json(OUT/"summary.json",summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    plan,prefs=preflight()
    print(f"PREFLIGHT PASSED: {plan['content_sha256']} / 1 frozen best x 7 cases x 4 prefs x 3 samples",flush=True)
    evaluate(plan,prefs)
    summarize(plan)
