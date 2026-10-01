"""Continue a V2 single-objective experiment as an explicit fine-tuning stage.

Copy this file to the V2 project's scripts directory and run from its root:
  python scripts/continue_v2.py --check
  python scripts/continue_v2.py --run

The existing training engine, worker physics, PPO update and evaluation code are
reused. A runner adapter translates stage-local indices to unused training-data
indices; evaluation indices and the fixed dataset contract remain unchanged.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

sys.dont_write_bytecode = True
OBJECTIVES = ("flow", "cost", "variance")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def activate_project(project):
    if not (project / "data" / "distribution.py").is_file():
        raise ValueError("Use the V2 project checkout (with data/distribution.py).")
    sys.path.insert(0, str(project))
    import configs.config
    if configs.config.PROJECT_ROOT.resolve() != project:
        raise ValueError("Imported runtime belongs to a different checkout.")


def prepare(args, objective):
    import torch
    from agent.ppo import PPOAgent, build_actor_critic
    from configs import load_config
    from data import load_dataset_split
    from data.distribution import protocol_hashes, training_sampling_plan
    from data.selection import select_validation_subsets
    from environment import AssemblySchedulingEnv

    source = args.run_root / f"{objective}_v2_seed11_1000"
    checkpoint = source / "best_checkpoint.pt"
    original = read_json(source / "config.json")
    if original["generator"]["version"] != "2.0.0":
        raise ValueError("Source run is not a V2 experiment.")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = saved["metadata"]
    if metadata["objective_name"] != objective or int(metadata["algorithm_seed"]) != 11:
        raise ValueError("Checkpoint objective/algorithm seed disagrees with the requested run.")
    with (source / "train_log.csv").open(encoding="utf-8-sig", newline="") as handle:
        used_seeds = {int(row["seed"]) for row in csv.DictReader(handle)}
    seed_start = int(original["dataset"]["splits"]["train"]["seed_start"])
    offset = max(used_seeds) - seed_start + 1
    if seed_start + offset + args.episodes > int(original["dataset"]["splits"]["train"]["seed_end"]):
        raise ValueError("Additional training would exceed the training seed range.")

    config = deepcopy(original)
    config.pop("runtime_manifest", None)  # Generated on loading, never an input selector.
    config["training"].pop("initial_checkpoint", None)
    config["experiment_name"] = f"{objective}_v2_seed11_best_plus{args.episodes}"
    config["training"]["episodes"] = args.episodes
    config["ppo"]["learning_rate"] = args.learning_rate
    terminal_weights = deepcopy(config["generator"]["curriculum"]["anchors"][-1]["weights"])
    config["generator"]["curriculum"] = {
        "mode": "linear", "anchors": [{"at_fraction": fraction, "weights": terminal_weights}
                                        for fraction in (0, 1)]}
    config["generator"]["severity_curriculum"] = {
        "anchors": [{"at_fraction": fraction, "value": 1.0} for fraction in (0, 1)]}
    config["paths"]["result_root"] = str(args.output_root / "runs")
    # Atomic cache files append a 64-character fingerprint and a temporary name.
    # Keep this root short enough for Windows installations with MAX_PATH limits.
    config["paths"]["training_instances_cache"] = str(args.cache_root)
    # Rebase input paths from the machine that originally ran the experiment.
    for key in ("fixed_instance", "instances_root", "manifests_root"):
        raw = config["paths"][key].replace("\\", "/")
        relative = raw[raw.index("data/"):] if "data/" in raw else raw
        config["paths"][key] = str(args.project_root / relative)
    manifest_name = config["objective_scalarizer"]["normalization_manifest"].replace("\\", "/").split("/")[-1]
    config["objective_scalarizer"]["normalization_manifest"] = str(args.project_root / "configs" / "manifests" / manifest_name)
    lineage = {
        "mode": "best_checkpoint_fine_tuning_v1", "optimizer_restored": False,
        "rng_and_controller_restored": False, "source_run": str(source),
        "source_checkpoint": str(checkpoint), "source_checkpoint_sha256": file_hash(checkpoint),
        "source_checkpoint_episode": int(metadata["checkpoint_episode"]),
        "source_network_weights_sha256": metadata["network_weights_sha256"],
        "source_training_budget": int(original["training"]["episodes"]),
        "stage_episodes": args.episodes, "stage_episode_start": 1,
        "training_instance_index_offset": offset,
        "training_instance_seed_start": seed_start + offset,
        "training_instance_seed_end_inclusive": seed_start + offset + args.episodes - 1,
        "learning_rate": args.learning_rate, "severity": 1.0,
        "source_validation": metadata["validation"],
    }
    config["training"]["continuation"] = lineage
    path = args.output_root / f"{objective}_plus{args.episodes}.json"
    write_json(path, config)
    effective = load_config(path)
    if protocol_hashes(effective) != protocol_hashes(original):
        raise ValueError("Fixed dataset protocol changed.")
    for key in ("environment", "reward", "preference", "network"):
        if effective[key] != original[key]:
            raise ValueError(f"Continuation changed {key}.")
    source_runtime = original["runtime_manifest"]
    target_runtime = effective["runtime_manifest"]
    permitted_runtime = dict(source_runtime)
    if source_runtime.get("observation_schema") == 5:
        permitted_runtime.update(observation_schema=6, time_context="order_chain_action_context_v1")
    if target_runtime not in (source_runtime, permitted_runtime):
        raise ValueError("Runtime contract changed beyond the supported time-context migration.")
    if effective["objective_scalarizer"]["scales"] != original["objective_scalarizer"]["scales"]:
        raise ValueError("Frozen normalization scales changed.")
    validation = load_dataset_split(effective, "validation")
    selection = select_validation_subsets(validation, effective["generator"]["dataset_pressure_weights"],
                                         target_count=50, diagnostic_count=49)["target"]
    if selection["subset_sha256"] != metadata["validation_subset_sha256"]:
        raise ValueError("Validation subset no longer matches the source checkpoint.")
    observation = AssemblySchedulingEnv(effective).reset(validation[0].instance)
    agent = PPOAgent(build_actor_critic(observation, effective["network"]), effective["ppo"], device="cpu")
    loaded_metadata = agent.load(checkpoint, load_optimizer=False)
    migration = loaded_metadata.get("checkpoint_load_migration")
    if target_runtime != source_runtime and not migration:
        raise ValueError("Observation schema changed without a recorded checkpoint migration.")
    exact_load = True
    for key, tensor in agent.network.state_dict().items():
        source_tensor = saved["network"][key].cpu()
        tensor = tensor.cpu()
        if torch.equal(tensor, source_tensor):
            continue
        if (migration and tensor.ndim == source_tensor.ndim == 2
                and tensor.shape[0] == source_tensor.shape[0]
                and tensor.shape[1] > source_tensor.shape[1]
                and torch.equal(tensor[:, :source_tensor.shape[1]], source_tensor)
                and torch.count_nonzero(tensor[:, source_tensor.shape[1]:]).item() == 0):
            exact_load = False
            continue
        raise ValueError(f"Loaded network weights differ beyond zero-padded time inputs: {key}")
    plan = training_sampling_plan(effective, offset + args.episodes)[offset:]
    if len(plan) != args.episodes or any(severity != 1.0 for _, severity in plan):
        raise ValueError("Continuation sampling plan is not fixed at severity 1.")
    planned_seeds = set(range(seed_start + offset, seed_start + offset + args.episodes))
    if used_seeds & planned_seeds:
        raise ValueError("Training instances reuse the original run's seeds.")
    lineage.update(config_path=str(path), validation_subset_sha256=selection["subset_sha256"],
                   pressure_counts=dict(Counter(label for label, _ in plan)),
                   exact_network_load=exact_load, source_input_weights_preserved=True,
                   observation_migration=migration, fixed_dataset_protocol_verified=True,
                   training_seeds_disjoint=True)
    effective["training"]["continuation"] = dict(lineage)
    print(json.dumps({"objective": objective, "source_episode": lineage["source_checkpoint_episode"],
                      "stage_episodes": args.episodes, "learning_rate": args.learning_rate,
                      "training_seeds": [min(planned_seeds), max(planned_seeds)],
                      "pressure_counts": lineage["pressure_counts"], "checks": "passed"}, ensure_ascii=False), flush=True)
    return effective, checkpoint, lineage


def train_stage(args, objective, config, checkpoint, lineage):
    import train as training
    from result.terminal_log import capture_terminal_output

    original_runner = training.ParallelEpisodeRunner
    offset = int(lineage["training_instance_index_offset"])

    class ContinuationEpisodeRunner(original_runner):
        """Keep stage-local logs while collecting unused online-instance indices."""
        def __init__(self, *runner_args, episode_count, **runner_kwargs):
            super().__init__(*runner_args, episode_count=offset + episode_count, **runner_kwargs)

        def collect_training_batch(self, agent, episode_indices, **kwargs):
            batch = super().collect_training_batch(agent, [offset + index for index in episode_indices], **kwargs)
            for episode in batch.episodes:
                episode.episode_index -= offset
            return batch

    # This binding is scoped to this launcher invocation; source files are untouched.
    training.ParallelEpisodeRunner = ContinuationEpisodeRunner
    name = f"{objective}_v2_seed11_best_plus{args.episodes}" + ("_smoke" if args.smoke else "")
    if args.smoke:
        config = deepcopy(config)
        config["training"].update(smoke_episodes=2, smoke_parallel_envs=2,
                                  smoke_rollout_steps=32, smoke_validation_instance_limit=2,
                                  validation_parallel_envs=2)
        config["training"]["validation_selection"]["diagnostic_instance_limit"] = 0
    destination = args.output_root / "runs" / name
    if destination.exists():
        raise FileExistsError(f"New run already exists: {destination}")
    try:
        with capture_terminal_output(args.output_root / f"{name}.terminal.log"):
            run = training.train(config, smoke=args.smoke, run_name=name,
                                 initial_checkpoint=checkpoint, visdom_enabled=False)
    finally:
        training.ParallelEpisodeRunner = original_runner
    write_json(run / "continuation_lineage.json", lineage)
    if args.smoke:
        with (run / "train_log.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        expected = list(range(lineage["training_instance_seed_start"], lineage["training_instance_seed_start"] + 2))
        if [int(row["seed"]) for row in rows] != expected or any(float(row["severity"]) != 1.0 for row in rows):
            raise ValueError("Smoke rollout used unexpected instance seeds/severity.")
        print(f"Smoke passed: {run}", flush=True)
        return
    summary = read_json(run / "summary.json")
    previous = lineage["source_validation"]
    selected = summary["checkpoint_selection"]
    old_completion, old_quality = previous["completion_rate"], previous["preference_balanced_quality_score"]
    new_completion, new_quality = selected["best_completion_rate"], selected["best_preference_balanced_quality_score"]
    improved = selected["has_best"] and (new_completion > old_completion + 1e-12 or (
        math.isclose(new_completion, old_completion, rel_tol=0, abs_tol=1e-12)
        and new_quality < old_quality - 1e-12))
    comparison = {"objective": objective, "new_stage_improved": improved,
                  "old_completion_rate": old_completion, "new_completion_rate": new_completion,
                  "old_quality": old_quality, "new_quality": new_quality,
                  "recommended_checkpoint": str(run / "best_checkpoint.pt") if improved else str(checkpoint),
                  "selection": "completion_first_quality_second_ties_retain_source"}
    write_json(run / "continuation_comparison.json", comparison)
    print(json.dumps(comparison, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-root", type=Path, help="Original run root; default: PROJECT/result/runs")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--cache-root", type=Path, help="Short cache path; default: PROJECT/result/cc")
    parser.add_argument("--objective", nargs="+", choices=OBJECTIVES, default=list(OBJECTIVES))
    parser.add_argument("--episodes", type=int, default=500, help="Additional episodes in this new stage")
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Prepare configurations and verify them; no training")
    mode.add_argument("--run", action="store_true", help="Run full continuation stages sequentially")
    mode.add_argument("--smoke", action="store_true", help="Run two short episodes and a small validation")
    args = parser.parse_args()
    if args.episodes < 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("episodes and learning rate must be positive")
    args.project_root = args.project_root.resolve()
    args.run_root = (args.run_root or args.project_root / "result" / "runs").resolve()
    args.output_root = (args.output_root or args.project_root / "result" / "continuation" / f"v2_seed11_best_plus{args.episodes}").resolve()
    args.cache_root = (args.cache_root or args.project_root / "result" / "cc").resolve()
    if args.output_root == args.run_root or args.run_root in args.output_root.parents:
        parser.error("output-root must be outside the original run-root")
    activate_project(args.project_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    prepared = [(objective, *prepare(args, objective)) for objective in args.objective]
    write_json(args.output_root / "continuation_plan.json", [lineage for _, _, _, lineage in prepared])
    if args.check:
        print(f"Configurations and plan: {args.output_root}", flush=True)
        return
    for objective, config, checkpoint, lineage in prepared:
        train_stage(args, objective, config, checkpoint, lineage)


if __name__ == "__main__":
    main()
