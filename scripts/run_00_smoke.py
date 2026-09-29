"""Run a small Universal PPO update, 13-point validation, and 66-point final test."""

from __future__ import annotations

import csv
import json
from datetime import datetime

import torch

from configs import load_config
from train import train


def main() -> None:
    config = load_config("configs/v8/universal.json")
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    config["training"].update({
        "smoke_episodes": 2,
        "smoke_rollout_steps": 16,
        "smoke_validation_instance_limit": 1,
        "smoke_parallel_envs": 2,
        "validation_parallel_envs": 20,
        "torch_num_threads": 2,
    })
    formal = config["training"]["formal_evaluation"]
    formal["validation_repeats"] = 1
    formal["final_test_repeats"] = 1
    run_name = f"universal_protocol_smoke_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = train(config, smoke=True, run_name=run_name, parallel_envs=2, validation_parallel_envs=20)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if not summary["checkpoint_selection"]["has_best"]:
        raise RuntimeError("smoke validation did not create a best checkpoint")
    with (run_dir / "sampled_validation_instance_metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
        validation_rows = list(csv.DictReader(handle))
    with (run_dir / "final_sampled_instance_metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
        final_rows = list(csv.DictReader(handle))
    if len(validation_rows) != 13 or len(final_rows) != 66:
        raise RuntimeError(f"unexpected smoke cell counts: validation={len(validation_rows)}, final={len(final_rows)}")
    print(f"Universal protocol smoke passed: {run_dir}")


if __name__ == "__main__":
    main()
