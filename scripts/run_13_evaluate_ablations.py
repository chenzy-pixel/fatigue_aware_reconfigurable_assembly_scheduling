"""Evaluate matched checkpoints on the frozen test manifest."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "result" / "runs"
PAIRS = (
    ("universal", "configs/v8/universal.json", "universal_seed11_ep2000", True),
    ("no_graph", "configs/ablations/no_graph.json", "ablation_no_graph_seed11_ep2000", True),
    ("shared_head", "configs/ablations/shared_head.json", "ablation_shared_head_seed11_ep2000", True),
    ("full_flow", "configs/e1/single_flow_relative_time.json", "e1_flow_relative_time_seed11_ep1000_rerun_20260928_223621", False),
    ("neutral_flow", "configs/ablations/neutral_flow.json", "neutral_flow_seed11_ep1000", False),
    ("full_cost", "configs/e1/single_cost.json", "e1_cost_seed11_ep1000_rerun_20260928_223621", False),
    ("neutral_cost", "configs/ablations/neutral_cost.json", "neutral_cost_seed11_ep1000", False),
    ("full_variance", "configs/e1/single_variance.json", "e1_variance_seed11_ep1000_rerun_20260928_223621", False),
    ("neutral_variance", "configs/ablations/neutral_variance.json", "neutral_variance_seed11_ep1000", False),
)


def main() -> None:
    for label, config, source, universal in PAIRS:
        output = RUN_ROOT / f"ablation_eval_{label}_seed11"
        if (output / "instance_metrics.csv").is_file() and (output / "metrics.json").is_file():
            print(f"complete: {output}")
            continue
        if output.exists():
            raise RuntimeError(f"incomplete evaluation requires inspection: {output}")
        checkpoint = RUN_ROOT / source / "best_checkpoint.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"required best checkpoint: {checkpoint}")
        command = [sys.executable, str(ROOT / "eval.py"), "--config", config,
                   "--policy", "ppo", "--checkpoint", str(checkpoint),
                   "--dataset", "test", "--decode-mode", "sampled",
                   "--device", "cuda" if torch.cuda.is_available() else "cpu",
                   "--run-name", output.name]
        if universal:
            command += ["--preference-set", "final_test"]
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
