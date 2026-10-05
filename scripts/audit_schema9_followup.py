"""Read-only runtime probes for remaining schema-9 observation/pipeline gaps."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import csv
import hashlib
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from agent.baselines import HeuristicPolicy
from agent.ppo import PPOAgent
from agent.ppo.network import build_actor_critic, NODE_TYPES
from agent.ppo.parallel import ParallelEpisodeRunner
from configs import load_config
from configs.config import public_config
from data.models import load_instance_yaml, validate_instance, OrderSpec, OperationSpec, instance_to_dict
from environment import AssemblySchedulingEnv, CAPABLE_EDGE, MACHINE_MODULE_EDGE
from scripts.audit_observation_reward import build_alias_instance, reachable_histories, compare_observations
from scripts.audit_state_sufficiency import worker_alias_instance, worker_histories

OUTPUT = Path("result/analysis/schema9_followup")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def edge_times(env, operation, machine):
    obs = env.observe()
    store = obs.relations[CAPABLE_EDGE]
    op = env.instance.operation_index[operation]
    mi = env.instance.machine_index[machine]
    pos = np.flatnonzero((store.edge_index[0] == op) & (store.edge_index[1] == mi))[0]
    return {name: float(store.edge_features[pos, store.feature_names.index(name)])*env.instance.horizon
            for name in ("processing_time_norm", "earliest_start_time_norm", "resource_ready_time_norm", "predicted_finish_time_norm")}


def busy_predictions(config, template):
    processing = reachable_histories(config, build_alias_instance(template))[0][0]
    installing = worker_histories(config, worker_alias_instance(template))[0][0]
    return {
        "processing": {"time": processing.current_time, "operation": "D_1", "machine": "M1",
                       "machine_end": processing.machines[0].busy_until_tick*processing.resolution,
                       "edge": edge_times(processing, "D_1", "M1")},
        "installation": {"time": installing.current_time, "operation": "A_1", "machine": "M6",
                         "installation_end": installing.machines[5].busy_until_tick*installing.resolution,
                         "committed_processing_end": (installing.machines[5].busy_until_tick + installing.estimate_processing_ticks(0, 5))*installing.resolution,
                         "edge": edge_times(installing, "A_1", "M6")}}


def standard_busy_prediction(config, template):
    env = AssemblySchedulingEnv(config)
    env.reset(template)
    policy = HeuristicPolicy()
    path = []
    while not env.task_done:
        for op in env.operations:
            if op.state.value != "READY":
                continue
            for machine in env.machines:
                if machine.state.value == "PROCESSING" and machine.current_module == op.spec.required_module:
                    values = edge_times(env, op.spec.id, machine.spec.id)
                    bound = machine.busy_until_tick*env.resolution + values["processing_time_norm"]
                    if values["predicted_finish_time_norm"] < bound - 1e-5:
                        return {"instance": template.instance_id, "legal_history": path,
                                "time": env.current_time, "operation": op.spec.id, "machine": machine.spec.id,
                                "edge": values, "earliest_physical_finish": bound,
                                "candidate_currently_masked": bool(env.get_action_mask()[env.encode_production_action(
                                    env.instance.operation_index[op.spec.id], env.instance.machine_index[machine.spec.id])])}
        action = int(policy.select_action(env))
        path.append(action)
        env.step(action, build_observation=False)
    return {"found": False}


def attributed_edge_instances(template):
    orders = tuple(OrderSpec(name, "audit_"+name, 0, (OperationSpec(name+"_1", name, 1, module, 150),))
                   for name, module in (("A", "A1"), ("B", "A2")))
    result = []
    for swapped in (False, True):
        machines = []
        for i, machine in enumerate(template.machines):
            parameters = dict(machine.module_parameters)
            if i in (0, 3):
                for j, module in enumerate(("A1", "A2")):
                    fast = ((i == 0) != (j == 1)) != swapped
                    parameters[module] = replace(parameters[module], processing_speed_factor=.9 if fast else 1.1,
                                                 installation_base_time=15, disassembly_base_time=15)
            machines.append(replace(machine, module_parameters=parameters))
        instance = replace(template, instance_id=f"attribute_pairing_{int(swapped)}", instance_type="test",
                           machines=tuple(machines), workers=tuple(replace(w, initial_fatigue=0) for w in template.workers),
                           orders=orders, waves={"audit_"+name: {"dominant_module": module, "order_ids": [name], "release_interval": [0, 0]}
                                                 for name, module in (("A", "A1"), ("B", "A2"))})
        validate_instance(instance)
        result.append(instance)
    return result


def encoding_pair(config, template):
    instances = attributed_edge_instances(template)
    envs = [AssemblySchedulingEnv(config) for _ in instances]
    obs = [env.reset(instance, preference=(1, 0, 0)) for env, instance in zip(envs, instances)]
    equality = compare_observations(*obs)
    incident_errors = {}
    for kind in (CAPABLE_EDGE, MACHINE_MODULE_EDGE):
        stores = [item.relations[kind] for item in obs]
        worst = 0.0
        for side in (0, 1):
            for node in np.unique(stores[0].edge_index[side]):
                sums = [s.edge_features[s.edge_index[side] == node].astype(np.float64).sum(0) for s in stores]
                worst = max(worst, float(np.max(np.abs(sums[0]-sums[1]))))
        incident_errors["__".join(kind)] = worst
    encodings = []
    for seed in (0, 1, 11, 23, 37):
        torch.manual_seed(seed)
        model = build_actor_critic(obs[0], config["network"]).eval()
        with torch.no_grad():
            batch, _, _, context = model.encode_graph(obs, device="cpu")
            _, values = model.forward_batch(obs, [env.get_action_mask() for env in envs], device="cpu")
            double = deepcopy(model).double()
            nodes = {name: double.node_projectors[name](features.double()) for name, features in batch.node_features.items()}
            relations = {key: (indices, features.double(), direction) for key, (indices, features, direction) in batch.relations.items()}
            for layer in double.message_layers:
                nodes = layer(nodes, relations)
            ctx = torch.cat(tuple(double._pool_slices(nodes[name], batch.node_slices[name]) for name in NODE_TYPES)
                            + (double.global_encoder(batch.global_features.double()),), dim=-1)
            pref = double.preference_encoder(torch.as_tensor(np.stack([o.preference for o in obs]), dtype=torch.float64))
            dv = double._critic_values(ctx, pref)
        encodings.append({"seed": seed, "graph_float32": float((context[0]-context[1]).abs().max()),
                          "graph_float64": float((ctx[0]-ctx[1]).abs().max()),
                          "critic_float32": float((values[0]-values[1]).abs()),
                          "critic_float64": float((dv[0]-dv[1]).abs())})
    returns = []
    for env in envs:
        path = [env.encode_production_action(0, 0), env.encode_production_action(1, 2)]
        reward_sum = 0.0
        for action in path:
            reward_sum += env.step(action)[1].scalarize(config["reward"])
        while not env.task_done:
            action = env.wait_action
            path.append(action)
            reward_sum += env.step(action)[1].scalarize(config["reward"])
        returns.append({"actions": path, "flow": env._objective_vector()[0], "return": reward_sum, "succeeded": env.task_succeeded})
    for index, instance in enumerate(instances):
        write_json(OUTPUT/f"attribute_instance_{index}.json", instance_to_dict(instance))
    return {"raw_equality": equality, "incident_attribute_sum_error": incident_errors, "encoding": encodings,
            "fixed_dispatch_returns": returns, "scope": "validated synthetic instances; processing/stage times differ from benchmark generator"}


def rollout_cutoff(config, template):
    settings = deepcopy(config)
    settings["paths"]["training_instances_cache"] = "result/audits/s9_followup_cache"
    settings["training"]["worker_timeout_seconds"] = 120
    settings["device"] = "cpu"
    obs = AssemblySchedulingEnv(settings).reset(template)
    model = build_actor_critic(obs, {**settings["network"], "hidden_dim": 16})
    agent = PPOAgent(model, settings["ppo"], device="cpu")
    with ParallelEpisodeRunner(config=settings, template=template, episode_count=1, worker_count=1) as runner:
        batch = runner.collect_training_batch(agent, [0], gamma=1, gae_lambda=.95, step_limit=2)
    episode = batch.episodes[0]
    return {"step_count": episode.step_count, "buffer_length": len(episode.buffer),
            "last_done": episode.buffer.transitions[-1].done if len(episode.buffer) else None,
            "metrics": {name: episode.metrics.get(name) for name in ("terminated", "truncated", "task_done", "task_succeeded", "task_failed", "sampling_truncated", "objective_complete", "terminal_reason")}}


def sequential_worker_projection(config, template):
    from environment.time_context import OrderTimeEstimator, _Resources
    instance = replace(worker_alias_instance(template), instance_id="sequential_worker_projection",
                       workers=tuple(replace(w, initial_fatigue=.5 if w.id == "H5" else .74) for w in template.workers))
    # Start the target order immediately so the initial fatigue is retained.
    orders = tuple(replace(o, release_time=0 if o.id == "A" else o.release_time) for o in instance.orders)
    waves = deepcopy(instance.waves)
    waves["W1"]["release_interval"][0] = 0
    instance = replace(instance, orders=orders, waves=waves)
    validate_instance(instance)
    write_json(OUTPUT/"sequential_worker_instance.json", instance_to_dict(instance))
    env = AssemblySchedulingEnv(config)
    env.reset(instance)
    profile = env._production_resource_profile(5, "A2")
    dis = env._earliest_safe_stage_projection(5, 4, "A3", installation=False, earliest_tick=0)
    start, duration = dis
    ins = env._earliest_safe_stage_projection(5, 4, "A2", installation=True, earliest_tick=start+duration)
    estimator = OrderTimeEstimator(env)
    resources = _Resources(estimator.machines.copy(), estimator.workers.copy())
    sequential_end = estimator._transition_finish(5, "A3", "A2", 0, resources)
    path = [env.encode_production_action(0, 5), env.wait_action]
    for action in path:
        env.step(action, build_observation=False)
    action = env.encode_worker_action(5, 4)
    assert not env.get_action_mask()[action]
    path.append(action)
    env.step(action, build_observation=False)
    while env.current_tick < start+duration:
        action = env.wait_action
        path.append(action)
        env.step(action, build_observation=False)
    rec = env._pending_reconfiguration(env.machines[5].spec.id)
    can_install = [w.spec.id for w in env.workers if env._worker_can_start(rec, w)]
    revised = [env._earliest_safe_stage_projection(5, wi, "A2", installation=True, earliest_tick=env.current_tick)
               for wi in range(len(env.workers))]
    earliest_end = min(a+b for projection in revised if projection is not None for a,b in [projection])
    return {"old_profile_processing_start": profile.processing_start_tick*env.resolution,
            "same_worker_dis_projection": list(dis), "same_worker_ins_projection": list(ins),
            "sequential_time_estimator_end": sequential_end*env.resolution,
            "legal_history": path, "actual_disassembly_end": env.current_time,
            "actual_H5_fatigue": env.workers[4].fatigue, "immediately_safe_installers": can_install,
            "earliest_install_end_after_actual_dis": earliest_end*env.resolution}


def plotting_cutoff(template):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from analysis.single_objective_analysis import analyze_run, plot_run, load_plot_rows
    config = load_config("configs/e1/single_flow.json")
    config["environment"]["max_decisions"] = 1
    env = AssemblySchedulingEnv(config)
    env.reset(template)
    _, reward, _, _, _ = env.step(HeuristicPolicy().select_action(env))
    root = OUTPUT/"truncated_plot_run"
    write_json(root/"config.json", public_config(config))
    write_json(root/"summary.json", {})
    row = {**env.metrics(), "episode": 1, "reward": reward.scalarize(config["reward"])}
    with (root/"train_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    with (root/"validation_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("episode", "evaluation_complete", "completion_rate"))
        writer.writeheader()
        writer.writerow({"episode": 1, "evaluation_complete": False, "completion_rate": ""})
    error = None
    try:
        plot_run(root)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        plt.close("all")
    return {"plot_error": error, "summary": analyze_run(root), "loaded_row_keys": list(load_plot_rows(root)[0])}


def main():
    from environment.observation_schema import OBSERVATION_SCHEMA_VERSION
    if OBSERVATION_SCHEMA_VERSION != 9:
        raise RuntimeError('This historical finding probe requires schema 9; for the current repair use scripts/verify_sequential_projection.py. Saved schema-9 evidence remains in result/analysis/schema9_followup/.')
    torch.set_num_threads(2)
    config = load_config("configs/default.json")
    template = load_instance_yaml(config["paths"]["fixed_instance"])
    results = {}
    for name, function in (("busy_prediction", lambda: busy_predictions(config, template)),
                           ("standard_busy_prediction", lambda: standard_busy_prediction(config, template)),
                           ("edge_pairing", lambda: encoding_pair(config, template)),
                           ("sequential_projection", lambda: sequential_worker_projection(config, template)),
                           ("rollout_cutoff", lambda: rollout_cutoff(config, template)),
                           ("single_objective_plot", lambda: plotting_cutoff(template))):
        results[name] = function()
        write_json(OUTPUT/"findings.json", results)
        print(f"Completed {name}", flush=True)
    # Assert the reported witnesses, not a desired production behavior.
    standard = results["standard_busy_prediction"]
    assert standard["edge"]["predicted_finish_time_norm"] < standard["earliest_physical_finish"] - 1
    pairing = results["edge_pairing"]
    assert pairing["raw_equality"]["nodes"] and not pairing["raw_equality"]["relations"]
    assert all(error == 0 for error in pairing["incident_attribute_sum_error"].values())
    assert all(row["graph_float64"] < 1e-12 for row in pairing["encoding"])
    assert pairing["fixed_dispatch_returns"][0]["flow"] != pairing["fixed_dispatch_returns"][1]["flow"]
    sequential = results["sequential_projection"]
    assert not sequential["immediately_safe_installers"]
    assert sequential["earliest_install_end_after_actual_dis"] > sequential["old_profile_processing_start"] + 1
    assert results["rollout_cutoff"]["step_count"] == 2
    assert results["rollout_cutoff"]["metrics"]["sampling_truncated"] is False
    assert results["single_objective_plot"]["plot_error"] == "ValueError: could not convert string to float: ''"
    write_json(OUTPUT/"source_hashes.json", {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
               for name in ("environment/env.py", "agent/ppo/network.py", "agent/ppo/parallel.py",
                            "analysis/single_objective_analysis.py", "environment/observation_schema.py",
                            "environment/time_context.py", "configs/default.json", "scripts/audit_schema9_followup.py")})
    print("All five reported findings reproduced", flush=True)


if __name__ == "__main__":
    main()
