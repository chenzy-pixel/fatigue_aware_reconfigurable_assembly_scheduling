"""Build the frozen Universal scales from completed E1 run records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from configs.normalization import build_normalization_manifest, write_immutable_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flow-run", required=True)
    parser.add_argument("--cost-run", required=True)
    parser.add_argument("--variance-run", required=True)
    parser.add_argument("--validation-manifest", default="data/manifests/validation/manifest.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source_runs = {
        "flow": root / args.flow_run,
        "cost": root / args.cost_run,
        "variance": root / args.variance_run,
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
