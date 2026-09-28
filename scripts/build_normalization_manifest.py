"""Build the frozen Universal scales from completed E1 run records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from configs.normalization import build_normalization_manifest, write_immutable_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default="result/runs")
    parser.add_argument("--validation-manifest", default="data/manifests/validation/manifest.json")
    parser.add_argument("--output", default="configs/manifests/e1_tail5_scales_20260928.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    runs = root / args.runs_root
    source_runs = {
        "flow": runs / "e1_flow_relative_time_seed11_ep1000",
        "cost": runs / "e1_cost_seed11_ep1000",
        "variance": runs / "e1_variance_seed11_ep1000",
    }
    manifest = build_normalization_manifest(
        source_runs,
        validation_dataset_path=root / args.validation_manifest,
        project_root=root,
    )
    destination = root / args.output
    digest = write_immutable_manifest(destination, manifest)
    print(json.dumps({"path": str(destination), "sha256": digest, "scales": manifest["scales"]}))


if __name__ == "__main__":
    main()
