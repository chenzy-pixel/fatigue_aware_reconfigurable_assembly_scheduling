"""Small end-to-end update for every ablation variant."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs import load_config
from train import train

CONFIGS = ("no_graph", "shared_head", "neutral_flow", "neutral_cost", "neutral_variance")


def main() -> None:
    for name in CONFIGS:
        completed = [path for path in (ROOT / "result" / "runs").glob(f"ablation_smoke_{name}_*")
                     if (path / "best_checkpoint.pt").is_file() and (path / "summary.json").is_file()]
        if completed:
            print(f"complete: {max(completed)}", flush=True)
            continue
        config = load_config(f"configs/ablations/{name}.json")
        config["device"] = "cpu"
        config["training"].update({
            "smoke_episodes": 2, "smoke_rollout_steps": 16,
            "smoke_parallel_envs": 2, "validation_parallel_envs": 2,
            "smoke_validation_instance_limit": 1, "torch_num_threads": 2,
        })
        config["training"]["formal_evaluation"]["validation_repeats"] = 1
        config["training"]["formal_evaluation"]["final_test_repeats"] = 1
        config["training"]["formal_evaluation"]["validation_preferences"] = [
            [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        ]
        config["training"]["formal_evaluation"]["final_test_lattice_denominator"] = 1
        output = train(config, smoke=True,
                       run_name=f"ablation_smoke_{name}_{datetime.now():%Y%m%d_%H%M%S}",
                       parallel_envs=2, episodes_per_update=2, validation_parallel_envs=2)
        if not (output / "best_checkpoint.pt").is_file():
            raise RuntimeError(f"smoke did not select a checkpoint: {name}")
        print(f"{name}: {output}", flush=True)


if __name__ == "__main__":
    main()
