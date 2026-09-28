"""Measure training and formal-validation throughput on the target machine.

The same 40 online episode indices, initial weights, and validation seeds are
used for every candidate. Run from the project root on the target machine.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import multiprocessing
import os
import platform
import statistics
import subprocess
import threading
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from agent.baselines import HeuristicPolicy
from agent.ppo import PPOAgent, build_actor_critic
from agent.ppo.parallel import ParallelEpisodeRunner
from configs import load_config, project_path
from configs.config import public_config
from data.dataset import OnlineInstanceDataset
from data.models import load_instance_yaml
from environment import AssemblySchedulingEnv, DecisionType
from result.io import write_json
from train import _evaluate_policy
from utils import configured_formal_evaluation_sampling_seeds, set_seed


CANDIDATES = (8, 12, 16, 20, 24, 32, 40)
GIB = 1024 ** 3


def _memory_bytes() -> tuple[int, int] | None:
    if os.name != "nt":
        return None

    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                   ("total_physical", ctypes.c_ulonglong),
                   ("available_physical", ctypes.c_ulonglong),
                   ("total_page_file", ctypes.c_ulonglong),
                   ("available_page_file", ctypes.c_ulonglong),
                   ("total_virtual", ctypes.c_ulonglong),
                   ("available_virtual", ctypes.c_ulonglong),
                   ("available_extended_virtual", ctypes.c_ulonglong)]

    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return int(status.total_physical), int(status.available_physical)


def _gpu_read(fields: str) -> list[str] | None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            text=True, timeout=10, stderr=subprocess.DEVNULL,
        )
        return [part.strip() for part in output.splitlines()[0].split(",")]
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, IndexError):
        return None


class HardwareMonitor:
    def __init__(self) -> None:
        self.samples: list[dict[str, float | None]] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample_loop, daemon=True)

    def __enter__(self) -> "HardwareMonitor":
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop_event.set()
        self.thread.join(timeout=12)

    def _sample_loop(self) -> None:
        while not self.stop_event.is_set():
            gpu = _gpu_read("utilization.gpu,utilization.memory,memory.used,memory.total")
            memory = _memory_bytes()
            def parse(part: str) -> float | None:
                try:
                    return float(part)
                except ValueError:
                    return None
            values = [parse(part) for part in gpu] if gpu and len(gpu) == 4 else None
            self.samples.append({
                "gpu_utilization_percent": values[0] if values else None,
                "gpu_memory_utilization_percent": values[1] if values else None,
                "gpu_memory_used_mib": values[2] if values else None,
                "gpu_memory_total_mib": values[3] if values else None,
                "available_memory_gib": memory[1] / GIB if memory is not None else None,
            })
            self.stop_event.wait(5.0)

    def summary(self) -> dict[str, float | None]:
        def collect(key: str) -> list[float]:
            return [float(row[key]) for row in self.samples if row[key] is not None]

        utilization = collect("gpu_utilization_percent")
        memory_utilization = collect("gpu_memory_utilization_percent")
        used = collect("gpu_memory_used_mib")
        total = collect("gpu_memory_total_mib")
        available = collect("available_memory_gib")
        return {
            "mean_gpu_utilization_percent": statistics.mean(utilization) if utilization else None,
            "mean_gpu_memory_utilization_percent": statistics.mean(memory_utilization) if memory_utilization else None,
            "peak_gpu_memory_mib": max(used) if used else None,
            "gpu_memory_total_mib": total[0] if total else None,
            "minimum_available_memory_gib": min(available) if available else None,
        }


def _safe(row: dict[str, Any]) -> bool:
    memory = row.get("minimum_available_memory_gib")
    used = row.get("peak_gpu_memory_mib")
    total = row.get("gpu_memory_total_mib")
    return (memory is None or memory >= 4.0) and (
        used is None or total is None or used <= 0.9 * total
    )


def _choose(rows: list[dict[str, Any]], metric: str) -> int | None:
    safe = [row for row in rows if row.get("status") == "ok" and _safe(row)]
    if not safe:
        return None
    fastest = min(float(row[metric]) for row in safe)
    return min(int(row["workers"]) for row in safe if float(row[metric]) <= fastest * 1.05)


def _preflight(config: dict[str, Any], template: Any) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    set_seed(int(config["seed"]))
    torch.set_num_threads(int(config["training"].get("torch_num_threads", 4)))
    environment = AssemblySchedulingEnv(config)
    observation = environment.reset(template)
    mask = environment.get_action_mask()
    current = observation
    heuristic = HeuristicPolicy()
    for _ in range(200):
        if current.decision_type == DecisionType.WORKER:
            worker_observation = current
            worker_mask = environment.get_action_mask()
            break
        current, _, terminated, truncated, _ = environment.step(
            heuristic.select_action(environment)
        )
        if terminated or truncated:
            raise RuntimeError("preflight instance did not reach a worker decision")
    else:
        raise RuntimeError("preflight worker decision was not reached in 200 steps")
    network = build_actor_critic(observation, config["network"]).to(config["device"])
    logits, values = network.forward_batch(
        [observation, worker_observation], [mask, worker_mask], device=config["device"]
    )
    loss = values.sum() + sum(
        logits[index, :len(action_mask)][~torch.as_tensor(action_mask, device=config["device"])].mean()
        for index, action_mask in enumerate((mask, worker_mask))
    )
    loss.backward()
    if not bool(torch.isfinite(loss)) or any(
        parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
        for parameter in network.parameters()
    ):
        raise FloatingPointError("GPU forward/backward preflight returned a non-finite value")
    state = {name: value.detach().cpu().clone() for name, value in network.state_dict().items()}
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.contiguous().numpy().tobytes())
    gpu = _gpu_read("name,driver_version,memory.total")
    memory = _memory_bytes()
    info = {
        "cpu": platform.processor() or platform.machine(), "logical_cpu_count": os.cpu_count(),
        "system_memory_total_gib": memory[0] / GIB if memory is not None else None,
        "system_memory_available_gib": memory[1] / GIB if memory is not None else None,
        "gpu": gpu[0] if gpu else None,
        "driver": gpu[1] if gpu else None,
        "gpu_memory_total_mib": float(gpu[2]) if gpu else None,
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "device": str(config["device"]),
        "device_capability": torch.cuda.get_device_capability() if torch.cuda.is_available() else None,
        "forward_backward": "passed",
        "initial_weights_sha256": digest.hexdigest(),
    }
    return info, state


def _agent(config: dict[str, Any], template: Any, state: dict[str, torch.Tensor]) -> PPOAgent:
    set_seed(int(config["seed"]))
    observation = AssemblySchedulingEnv(config).reset(template)
    network = build_actor_critic(observation, config["network"])
    network.load_state_dict(state)
    network.execution_mode = str(config["training"].get("policy_execution_version", "phase_batched_v1"))
    return PPOAgent(network, config["ppo"], device=config["device"])


def _warmup(agent: PPOAgent, config: dict[str, Any], template: Any) -> float:
    started = time.perf_counter()
    environment = AssemblySchedulingEnv(config)
    production = environment.reset(template)
    production_mask = environment.get_action_mask()
    heuristic = HeuristicPolicy()
    current = production
    for _ in range(200):
        if current.decision_type == DecisionType.WORKER:
            break
        current, _, terminated, truncated, _ = environment.step(
            heuristic.select_action(environment)
        )
        if terminated or truncated:
            raise RuntimeError("warmup instance did not reach a worker decision")
    else:
        raise RuntimeError("warmup worker decision was not reached")
    observations = [production, current]
    masks = [production_mask, environment.get_action_mask()]
    for _ in range(2):
        agent.act_batch(observations, masks, deterministic=True)
        agent.consume_policy_decision_diagnostics()
    return time.perf_counter() - started


def _trial(
    config: dict[str, Any], template: Any, state: dict[str, torch.Tensor],
    *, workers: int, full_training: bool, validation_instances: int,
    validation_repeats: int, quick_steps: int, validation_only: bool = False,
) -> dict[str, Any]:
    agent = _agent(config, template, state)
    warmup_seconds = _warmup(agent, config, template)
    local_config = deepcopy(config)
    local_config["training"]["validation_parallel_envs"] = workers
    training_count = 0 if validation_only else 40
    started = time.perf_counter()
    with HardwareMonitor() as monitor:
        with ParallelEpisodeRunner(
            config=local_config, template=template,
            episode_count=120, worker_count=workers,
        ) as runner:
            startup = time.perf_counter() - started
            rollout = None
            if training_count:
                set_seed(int(config["seed"]))
                rollout = runner.collect_training_batch(
                    agent, list(range(training_count)),
                    gamma=float(config["ppo"]["gamma"]),
                    gae_lambda=float(config["ppo"]["gae_lambda"]),
                    step_limit=None if full_training else quick_steps,
                )
            update_seconds = 0.0
            if full_training and rollout is not None:
                update_start = time.perf_counter()
                agent.update(rollout.buffer)
                update_seconds = time.perf_counter() - update_start
            validation_seconds = 0.0
            if validation_instances:
                seeds = configured_formal_evaluation_sampling_seeds(config, "validation")[:validation_repeats]
                validation_start = time.perf_counter()
                _evaluate_policy(
                    local_config, dataset_name=str(config["training"]["validation_split"]),
                    ppo_agent=agent, runner=runner,
                    instance_limit=validation_instances, sampling_seeds=seeds,
                )
                validation_seconds = time.perf_counter() - validation_start
    sampling_seconds = rollout.sampling_wall_time_seconds if rollout is not None else 0.0
    seconds = sampling_seconds + update_seconds
    phase_counts: dict[str, int] = {}
    graph_sizes: set[int] = set()
    if rollout is not None:
        for transition in rollout.buffer.transitions:
            observation = transition.observation
            phase = str(observation.decision_type.value)
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
            graph_sizes.add(sum(len(nodes) for nodes in observation.node_features.values()))
    return {
        "status": "ok", "workers": workers,
        "startup_seconds": startup,
        "warmup_seconds": warmup_seconds,
        "sampling_seconds": sampling_seconds,
        "policy_inference_seconds": rollout.policy_inference_time_seconds if rollout is not None else 0.0,
        "worker_generation_seconds_sum": sum(item.generation_time_seconds for item in rollout.episodes) if rollout is not None else 0.0,
        "worker_environment_step_seconds_sum": sum(item.environment_step_time_seconds for item in rollout.episodes) if rollout is not None else 0.0,
        "ppo_update_seconds": update_seconds,
        "validation_wall_seconds": validation_seconds,
        "episode_count": training_count,
        "transition_count": rollout.transition_count if rollout is not None else 0,
        "decision_type_counts": phase_counts,
        "distinct_graph_sizes": sorted(graph_sizes),
        "episodes_per_hour": training_count * 3600.0 / seconds if seconds else 0.0,
        "transitions_per_second": rollout.transition_count / seconds if rollout is not None and seconds else 0.0,
        "training_wall_seconds": seconds,
        **monitor.summary(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/e1/single_flow_relative_time.json")
    parser.add_argument("--output", default=None)
    parser.add_argument("--quick-steps", type=int, default=64)
    parser.add_argument("--quick-validation-instances", type=int, default=40)
    parser.add_argument("--candidates", default=",".join(map(str, CANDIDATES)),
                        help="comma-separated worker counts; default 8,12,16,20,24,32,40")
    parser.add_argument("--skip-full", action="store_true", help="only run the 7-candidate short sweep")
    args = parser.parse_args()
    if args.quick_steps < 1 or args.quick_validation_instances < 1:
        parser.error("quick steps and validation instances must be positive")
    try:
        candidates = tuple(int(part) for part in args.candidates.split(","))
    except ValueError:
        parser.error("--candidates must be comma-separated positive integers")
    if not candidates or any(value < 1 for value in candidates) or len(set(candidates)) != len(candidates):
        parser.error("--candidates must contain distinct positive integers")
    config = load_config(args.config)
    if str(config["device"]) != "cuda" or not torch.cuda.is_available():
        parser.error("benchmark requires the target CUDA GPU")
    config["training"]["policy_execution_version"] = "phase_batched_v1"
    config["training"]["policy_precision"] = "float32"
    output = Path(args.output or f"result/benchmarks/rtx5060ti_{datetime.now():%Y%m%d_%H%M%S}")
    output.mkdir(parents=True, exist_ok=False)
    template = load_instance_yaml(project_path(config["paths"]["fixed_instance"]))
    hardware, state = _preflight(config, template)
    print("[corpus] preparing the 120 fixed online instances", flush=True)
    corpus_started = time.perf_counter()
    corpus = OnlineInstanceDataset(config=config, template=template, episode_count=120)
    for index in range(120):
        corpus.get_with_cache_info(index)
        if (index + 1) % 20 == 0:
            print(f"[corpus] {index + 1}/120 ready", flush=True)
    corpus_seconds = time.perf_counter() - corpus_started
    report: dict[str, Any] = {
        "hardware": hardware, "quick": [], "full_training": [], "full_validation": [],
        "reference_training": [], "reference_validation": [],
        "corpus_materialization_seconds": corpus_seconds,
        "config": str(args.config), "seed": int(config["seed"]),
        "episodes_per_update": 40, "formal_validation_instances": 50,
        "formal_validation_repeats": 3,
        "selected_training_workers": None, "selected_validation_workers": None,
    }

    def save() -> None:
        write_json(output / "benchmark.json", report)
        lines = [
            "# RTX 5060 Ti 并行测速报告", "",
            f"设备：{hardware.get('gpu')}；驱动：{hardware.get('driver')}；"
            f"PyTorch：{hardware.get('torch')}；CUDA：{hardware.get('torch_cuda')}", "",
            "| 阶段 | 并行数 | 预热秒数 | 训练秒数 | 验证秒数 | transitions/s | GPU 利用率 | 显存带宽利用率 | 显存峰值 MiB | 最低可用内存 GiB | 状态 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for section in ("quick", "full_training", "full_validation", "reference_training", "reference_validation"):
            for row in report[section]:
                def cell(name: str) -> str:
                    value = row.get(name)
                    return f"{value:.2f}" if isinstance(value, (int, float)) else "—"
                lines.append(
                    f"| {section} | {row['workers']} | {cell('warmup_seconds')} | "
                    f"{cell('training_wall_seconds')} | "
                    f"{cell('validation_wall_seconds')} | {cell('transitions_per_second')} | "
                    f"{cell('mean_gpu_utilization_percent')} | "
                    f"{cell('mean_gpu_memory_utilization_percent')} | "
                    f"{cell('peak_gpu_memory_mib')} | "
                    f"{cell('minimum_available_memory_gib')} | {row['status']} |"
                )
        lines.extend((
            "", f"推荐训练并行数：{report['selected_training_workers']}",
            f"推荐验证并行数：{report['selected_validation_workers']}", "",
            "估计的 120 episode 耗时见 benchmark.json 的 estimated_120_seconds；"
            "实测完整运行耗时以训练 summary.json 为准。", "",
        ))
        if "estimated_120_seconds" in report:
            estimate = report["estimated_120_seconds"]
            ratio = estimate["reference_v8"] / estimate["phase_batched_v1"]
            lines.append(f"相同并行数下的参考实现估计加速比：{ratio:.2f}×。")
            lines.append("")
        if report.get("recommended_command"):
            lines.extend(("```powershell", report["recommended_command"], "```", ""))
        (output / "report.md").write_text("\n".join(lines), encoding="utf-8")

    save()
    for workers in candidates:
        print(f"[quick] workers={workers}", flush=True)
        try:
            row = _trial(
                config, template, state, workers=workers, full_training=False,
                validation_instances=args.quick_validation_instances,
                validation_repeats=1, quick_steps=args.quick_steps,
            )
        except Exception as error:
            row = {"status": "error", "workers": workers, "error": repr(error)}
        report["quick"].append(row)
        save()
    if not args.skip_full:
        valid = [row for row in report["quick"] if row["status"] == "ok" and _safe(row)]
        train_shortlist = sorted(valid, key=lambda row: row["training_wall_seconds"])[:2]
        train_candidates = {row["workers"] for row in train_shortlist}
        if any(row["workers"] == 40 for row in valid):
            train_candidates.add(40)
        train_candidates = sorted(train_candidates)
        val_candidates = sorted({row["workers"] for row in sorted(
            valid, key=lambda row: row["validation_wall_seconds"]
        )[:2]})
        for workers in train_candidates:
            for repeat in range(2):
                print(f"[full train] workers={workers} repeat={repeat + 1}/2", flush=True)
                try:
                    row = _trial(config, template, state, workers=workers, full_training=True,
                                 validation_instances=0, validation_repeats=0, quick_steps=args.quick_steps)
                except Exception as error:
                    row = {"status": "error", "workers": workers, "error": repr(error)}
                row["repeat"] = repeat
                report["full_training"].append(row)
                save()
        for workers in val_candidates:
            print(f"[full val] workers={workers} instances=50 repeats=3", flush=True)
            try:
                row = _trial(config, template, state, workers=workers, full_training=False,
                             validation_instances=50, validation_repeats=3,
                             quick_steps=args.quick_steps, validation_only=True)
            except Exception as error:
                row = {"status": "error", "workers": workers, "error": repr(error)}
            report["full_validation"].append(row)
            save()
        grouped = {}
        for row in report["full_training"]:
            if row["status"] == "ok" and _safe(row):
                grouped.setdefault(row["workers"], []).append(row)
        training_rows = [{
            "workers": workers, "status": "ok",
            "training_wall_seconds": statistics.median(item["training_wall_seconds"] for item in rows),
        } for workers, rows in grouped.items() if len(rows) == 2]
        report["selected_training_workers"] = _choose(training_rows, "training_wall_seconds")
        report["selected_validation_workers"] = _choose(
            report["full_validation"], "validation_wall_seconds"
        )
        selected_train = report["selected_training_workers"]
        selected_val = report["selected_validation_workers"]
        if selected_train is not None and selected_val is not None:
            reference_config = deepcopy(config)
            reference_config["training"]["policy_execution_version"] = "reference_v8"
            for repeat in range(2):
                print(f"[reference train] workers={selected_train} repeat={repeat + 1}/2", flush=True)
                try:
                    row = _trial(reference_config, template, state, workers=selected_train,
                                 full_training=True, validation_instances=0,
                                 validation_repeats=0, quick_steps=args.quick_steps)
                except Exception as error:
                    row = {"status": "error", "workers": selected_train, "error": repr(error)}
                row["repeat"] = repeat
                report["reference_training"].append(row)
                save()
            print(f"[reference val] workers={selected_val} instances=50 repeats=3", flush=True)
            try:
                reference_val_row = _trial(
                    reference_config, template, state, workers=selected_val,
                    full_training=False, validation_instances=50,
                    validation_repeats=3, quick_steps=args.quick_steps,
                    validation_only=True,
                )
            except Exception as error:
                reference_val_row = {"status": "error", "workers": selected_val, "error": repr(error)}
            report["reference_validation"].append(reference_val_row)
            save()
            command = (
                f"python -u train.py --config \"{args.config}\" "
                f"--episodes 120 --algorithm-seed {int(config['seed'])} --episodes-per-update 40 "
                f"--parallel-envs {selected_train} --validation-parallel-envs {selected_val} "
                "--run-name e1_flow_relative_time_seed11_ep120_optimized"
            )
            report["recommended_command"] = command
            recommended_config = deepcopy(config)
            recommended_config["training"].update({
                "episodes": 120, "episodes_per_update": 40,
                "parallel_envs": selected_train, "validation_parallel_envs": selected_val,
            })
            write_json(output / "recommended_config.json", public_config(recommended_config))
            optimized_train = next(row["training_wall_seconds"] for row in training_rows if row["workers"] == selected_train)
            optimized_val = next(row["validation_wall_seconds"] for row in report["full_validation"] if row["workers"] == selected_val)
            if all(row["status"] == "ok" for row in report["reference_training"] + report["reference_validation"]):
                reference_train = statistics.median(row["training_wall_seconds"] for row in report["reference_training"])
                reference_val = report["reference_validation"][0]["validation_wall_seconds"]
                report["estimated_120_seconds"] = {
                    "phase_batched_v1": 3 * optimized_train + 4 * optimized_val,
                    "reference_v8": 3 * reference_train + 4 * reference_val,
                    "estimated_speedup": (
                        (3 * reference_train + 4 * reference_val)
                        / (3 * optimized_train + 4 * optimized_val)
                    ),
                    "note": "Estimate: three 40-episode updates, three formal validations and one final test. Policy trajectories can change during an actual 120-episode run.",
                }
        save()
    print(json.dumps({
        "output": str(output.resolve()),
        "selected_training_workers": report["selected_training_workers"],
        "selected_validation_workers": report["selected_validation_workers"],
        "recommended_command": report.get("recommended_command"),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
