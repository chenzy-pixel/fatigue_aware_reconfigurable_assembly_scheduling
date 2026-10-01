"""Run the two 400-episode development courses and summarize their outcomes."""

from __future__ import annotations

from copy import deepcopy
import csv
import json
import time

from configs import load_config, project_path
from result.io import write_json
from result.provenance import effective_config_snapshot, source_state_snapshot
from scripts.prepare_data_v2 import main as prepare_data
from train import train


def main() -> None:
    prepare_data()
    config = load_config("configs/v8/universal.json")
    config["seed"] = 11
    config["training"].update(episodes=400, validation_interval_episodes=200,
                              validation_instance_limit=20, episodes_per_update=20)
    config["training"]["validation_selection"]["diagnostic_instance_limit"] = 7
    config["training"]["formal_evaluation"].update(validation_repeats=1, final_test_repeats=1)
    legacy = json.loads(project_path("configs/archive/data_v1.json").read_text(encoding="utf-8"))
    runs = {}
    for arm in ("original_course", "progressive_course"):
        current = deepcopy(config)
        if arm == "original_course":
            current["generator"]["curriculum"] = legacy["generator"]["curriculum"]
            current["generator"]["severity_curriculum"] = {"anchors": [
                {"at_fraction": 0.0, "value": 1.0}, {"at_fraction": 1.0, "value": 1.0}]}
        name = f"data_v2_{arm}_seed11_400"
        directory = project_path(current["paths"]["result_root"]) / name
        candidates = [directory] + sorted(directory.parent.glob(name + "_*"), reverse=True)
        reusable = False
        for candidate in candidates:
            if not (candidate / "summary.json").exists():
                continue
            provenance = json.loads((candidate / "summary.json").read_text(encoding="utf-8"))["provenance"]
            if (provenance["effective_config_sha256"] == effective_config_snapshot(current)["sha256"]
                    and provenance["source_state_sha256"] == source_state_snapshot()["sha256"]):
                directory, reusable = candidate, True
                break
        if not reusable:
            if directory.exists():
                name += f"_{time.time_ns()}"
            print(f"Running {arm}: 400 episodes", flush=True)
            directory = train(current, run_name=name)
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        diagnostic = json.loads((directory / "diagnostic_summary.json").read_text(encoding="utf-8"))
        runs[arm] = {
            "run_directory": str(directory), "elapsed_seconds": summary["elapsed_seconds"],
            "training_distribution": summary["training_distribution"],
            "checkpoint_selection": summary["checkpoint_selection"],
            "validation_subsets": summary["validation_subsets"],
            "diagnostic": diagnostic, "test": summary["final_sampled"],
            "effective_config_sha256": summary["provenance"]["effective_config_sha256"],
        }
    output = project_path("result/audits/data_v2/comparison.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    audit = json.loads(project_path("data/manifests/v2/audit.json").read_text(encoding="utf-8"))
    write_json(output, {"data_protocol_version": "2.0.0", "algorithm_seed": 11,
                        "data_audit": audit,
                        "episodes_per_arm": 400, "runs": runs})
    lines = ["# 算例协议 v2 开发对照", "", "算法种子 11；每组 400 episode。每 200 episode 验证 20 个固定实例 × 13 偏好 × 1 次采样；结束诊断 7 个实例；best checkpoint 在 20 个测试实例 × 66 偏好 × 1 次采样上评测。", "",
             f"开发数据生成及首次审计耗时：{audit['initial_generation_and_audit_seconds']:.1f} 秒；历史文件校验：{audit['legacy_files_verified']} 个。", "",
             "| 课程 | 总耗时（秒） | 训练实例生成/载入（累计秒） | 测试轨迹完成率 | 失败原因 |",
             "|---|---:|---:|---:|---|"]
    for arm, run in runs.items():
        test = run["test"]
        lines.append(f"| {arm} | {run['elapsed_seconds']:.1f} | {run['training_distribution']['generation_time_seconds']:.1f} | {test['completed_cell_count']/test['cell_count']:.2%} | {json.dumps(test['failure_reasons'], ensure_ascii=False)} |")
    lines += ["", "分场景测试轨迹统计；原始目标均值只统计成功轨迹。", "",
              "| 课程 | 场景 | 轨迹数 | 完成率 | Flow | Cost | Variance |", "|---|---|---:|---:|---:|---:|---:|"]
    for arm, run in runs.items():
        for scenario, group in run["test"]["by_pressure_type"].items():
            values = [group["completed_metrics"][name]["mean"] for name in ("flow_time_objective", "reconfiguration_cost", "worker_load_variance")]
            rendered = ["—" if value is None else f"{value:.3f}" for value in values]
            lines.append(f"| {arm} | {scenario} | {group['count']} | {group['completion_rate']:.2%} | {' | '.join(rendered)} |")
    lines += ["", "本轮用于验收协议、数值稳定性和可追溯性；性能结论需完整多种子实验。完整验证/诊断统计、失败进度、配置及轨迹位于 comparison.json 和各 run 目录。", ""]
    output.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    append_runtime_stats(output)
    print(f"Comparison report: {output}", flush=True)


def append_runtime_stats(output):
    report = json.loads(output.read_text(encoding="utf-8"))
    lines = ["耗时分解（秒）。训练采样与 PPO 更新为墙钟耗时；实例生成/载入另按各 episode 累计统计。", "",
             "| 课程 | 训练采样与更新 | 主验证 | 结束诊断 | 最终测试 |",
             "|---|---:|---:|---:|---:|"]
    for arm, run in report["runs"].items():
        directory = project_path(run["run_directory"])
        with (directory / "update_log.csv").open(encoding="utf-8-sig", newline="") as handle:
            updates = list(csv.DictReader(handle))
        with (directory / "validation_log.csv").open(encoding="utf-8-sig", newline="") as handle:
            validations = list(csv.DictReader(handle))
        run["training_wall_time_seconds"] = sum(float(row["sampling_wall_time_seconds"]) + float(row["ppo_update_time_seconds"]) for row in updates)
        run["validation_wall_time_seconds"] = sum(float(row["validation_wall_time_seconds"]) for row in validations)
        run["diagnostic_wall_time_seconds"] = run["diagnostic"]["wall_time_seconds"]
        run["test_wall_time_seconds"] = run["test"]["wall_time_seconds"]
        lines.append(f"| {arm} | {run['training_wall_time_seconds']:.1f} | {run['validation_wall_time_seconds']:.1f} | {run['diagnostic_wall_time_seconds']:.1f} | {run['test_wall_time_seconds']:.1f} |")
    write_json(output, report)
    markdown = output.with_suffix(".md")
    text = markdown.read_text(encoding="utf-8").split("\n\n耗时分解（秒）")[0]
    markdown.write_text(text.rstrip() + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
