"""Real CLI evaluation of a schema-10 checkpoint on a tiny fixed dataset."""
from __future__ import annotations

import csv
import json
import os
import runpy
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import torch
import yaml

from agent.baselines import HeuristicPolicy
from agent.ppo import PPOAgent
from agent.ppo.network import build_actor_critic
from configs import load_config, project_path
from data.dataset import GeneratedInstanceRecord, save_generated_record, template_sha256
from data.models import instance_to_dict, validate_instance
from data.distribution import protocol_hashes
from environment import AssemblySchedulingEnv
from result.io import write_config, write_json


def test_schema10_checkpoint_cli_saves_executed_hashes(fixed_instance, tmp_path):
    config = load_config("configs/v8/universal.json")
    config["device"] = "cpu"
    config["network"]["hidden_dim"] = 16
    config["training"]["torch_num_threads"] = 2
    config["training"]["validation_parallel_envs"] = 2
    config["paths"]["result_root"] = str(tmp_path / "runs")
    config["paths"]["instances_root"] = str(tmp_path / "instances")
    config["paths"]["manifests_root"] = str(tmp_path / "manifests")
    order = fixed_instance.orders[0]
    operation = replace(order.operations[0], base_processing_time=1.0)
    tiny = replace(fixed_instance, instance_id="test_tiny_3000000", instance_type="test",
                   orders=(replace(order, release_time=0.0, operations=(operation,)),),
                   waves={order.wave: {"dominant_module": operation.required_module,
                                       "order_ids": [order.id], "release_interval": [0.0, 0.0]}})
    validate_instance(tiny)
    template_path = tmp_path / "template.yaml"
    template_path.write_text(yaml.safe_dump(instance_to_dict(tiny)), encoding="utf-8")
    config["paths"]["fixed_instance"] = str(template_path)
    env = AssemblySchedulingEnv(config)
    observation = env.reset(tiny)
    heuristic = HeuristicPolicy()
    while not env.task_done:
        env.step(heuristic.select_action(env), build_observation=False)
    metrics = env.metrics()
    metadata = {
        "seed": 3000000, "split": "test", "generator_version": config["generator"]["version"],
        "template_instance": config["dataset"]["template_instance"], "template_sha256": template_sha256(tiny),
        **protocol_hashes(config), "severity": 1.0, "feasibility_status": "unknown",
        "diagnostic_status": "completed", "diagnostic_terminal_reason": metrics.get("terminal_reason"),
        "pressure_type": "easy", "cost_profile": "balanced_cost",
        "pressure_metrics": {"total_effective_load": 0.0, "max_module_load": 0.0},
        "heuristic_metrics": {
            "heuristic_completed": metrics["task_succeeded"], "heuristic_makespan": metrics["time"],
            "heuristic_flow_time": metrics["flow_time_objective"],
            "heuristic_reconfiguration_cost": metrics["reconfiguration_cost"],
            "worker_workload_variance": metrics["worker_load_variance"],
            "ready_configuration_gap_ratio": 0.0, "heuristic_reconfiguration_ratio": 0.0,
            "mean_wave_overlap_ratio": 0.0,
        },
    }
    digest = save_generated_record(GeneratedInstanceRecord(tiny, metadata), tmp_path / "instances/test/instance_3000000.json")
    manifest = {"schema_version": config["dataset"]["schema_version"],
                "generator_version": config["generator"]["version"],
                "template_instance": config["dataset"]["template_instance"], "template_sha256": template_sha256(tiny),
                **protocol_hashes(config), "generation_summary": {"pressure_counts": {"easy": 1}},
                "split": "test", "instance_count": 1, "seed_start": 3000000,
                "files": [{"path": "instance_3000000.json", "seed": 3000000, "sha256": digest}]}
    (tmp_path / "manifests/test").mkdir(parents=True)
    write_json(tmp_path / "manifests/test/manifest.json", manifest)
    old_network = build_actor_critic(observation, config['network'])
    checkpoint_path = tmp_path / 'schema10.pt'
    PPOAgent(old_network, config['ppo'], device='cpu').save(checkpoint_path, metadata={'runtime_manifest': config['runtime_manifest']})
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    source_hash = checkpoint['metadata']['network_weights_sha256']
    write_config(tmp_path, config)
    command = [sys.executable, str(project_path("eval.py")), "--config", str(tmp_path / "config.json"),
               "--policy", "ppo", "--checkpoint", str(checkpoint_path), "--dataset", "test", "--device", "cpu",
               "--allow-observation-migration"]
    environment = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
    for label, extra, cell_count in (
        ("grid", ["--preference-set", "final_test"], 198),
        ("ordinary", [], 3),
    ):
        result = subprocess.run(command + extra + ["--run-name", label], cwd=project_path("."),
                                env=environment, capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stdout + result.stderr
        output = tmp_path / "runs" / label
        saved = json.loads((output / "metrics.json").read_text())
        provenance = saved["provenance"]
        assert provenance["network_weights_sha256"] == source_hash
        assert provenance.get("checkpoint_load_migration") is None
        with (output / "instance_metrics.csv").open(encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == cell_count
        assert all(int(row["schedule_violation_count"]) == 0 for row in rows)
    from analysis.v8_pareto_analysis import analyze_runs
    summary = analyze_runs({"tiny": tmp_path / "runs/grid"}, tmp_path / "pareto")
    assert summary["preference_count"] == 66 and summary["repeat_count"] == 3
    from configs.ablation_protocol import validate_evaluation_run
    validate_evaluation_run(tmp_path / "runs/grid", config, checkpoint_path)
