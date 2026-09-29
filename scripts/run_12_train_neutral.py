"""Train the three fatigue neutral single-objective policies."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs import load_config
from train import train

RUNS = (
    ("neutral_flow_seed11_ep1000", "configs/ablations/neutral_flow.json"),
    ("neutral_cost_seed11_ep1000", "configs/ablations/neutral_cost.json"),
    ("neutral_variance_seed11_ep1000", "configs/ablations/neutral_variance.json"),
)


def main() -> None:
    for name, path in RUNS:
        output = ROOT / "result" / "runs" / name
        if (output / "best_checkpoint.pt").is_file() and (output / "summary.json").is_file():
            print(f"complete: {output}")
            continue
        if output.exists():
            raise RuntimeError(f"incomplete run requires inspection: {output}")
        train(load_config(path), run_name=name, algorithm_seed=11)


if __name__ == "__main__":
    main()
