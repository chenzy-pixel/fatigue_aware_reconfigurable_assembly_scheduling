"""Train the matched Universal baseline and two structural ablations."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs import load_config
from train import train

RUNS = (
    ("universal_seed11_ep2000", "configs/v8/universal.json"),
    ("ablation_no_graph_seed11_ep2000", "configs/ablations/no_graph.json"),
    ("ablation_shared_head_seed11_ep2000", "configs/ablations/shared_head.json"),
)


def main() -> None:
    for name, path in RUNS:
        output = ROOT / "result" / "runs" / name
        if (output / "best_checkpoint.pt").is_file() and (output / "summary.json").is_file():
            print(f"complete: {output}")
            continue
        if output.exists():
            raise RuntimeError(f"incomplete run requires inspection: {output}")
        config = load_config(path)
        train(config, run_name=name, algorithm_seed=11)


if __name__ == "__main__":
    main()
