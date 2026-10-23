"""Prepare and audit the frozen development datasets for data protocol v2."""

from __future__ import annotations

import json
import argparse
from pathlib import Path
import time

from configs import load_config, project_path
from data.dataset import PERSISTED_SPLITS, build_dataset_split, dataset_profile_counts, load_dataset_split, sha256_file
from data.distribution import protocol_hashes, training_sampling_plan
from data.models import load_instance_yaml
from data.selection import select_validation_subsets
from environment.preference import quality_preference_for_episode
from result.io import write_json


def verify_legacy_files() -> int:
    source = project_path("configs/archive/data_v1_hashes.json")
    hashes = json.loads(source.read_text(encoding="utf-8"))
    for name, digest in hashes.items():
        if sha256_file(project_path(name)) != digest:
            raise RuntimeError(f"legacy data changed: {name}")
    return len(hashes)


def main(*, generation_workers: int = 4, overwrite: bool = False) -> None:
    config = load_config("configs/default.json")
    template = load_instance_yaml(project_path(config["paths"]["fixed_instance"]))
    started = time.perf_counter()
    protected_count = verify_legacy_files()
    datasets = {}
    for split in PERSISTED_SPLITS:
        manifest = project_path(config["paths"]["manifests_root"]) / split / "manifest.json"
        if overwrite or not manifest.exists():
            print(f"Generating {split}", flush=True)
            build_dataset_split(config=config, template=template, split=split, profile="dev", resume=True,
                generation_workers=generation_workers, overwrite=overwrite)
        dataset = load_dataset_split(config, split)
        if len(dataset) != dataset_profile_counts(config, "dev")[split]:
            raise RuntimeError(f"published {split} count differs from the development protocol")
        # Read every record to verify the file and protocol hashes.
        for record in dataset:
            if record.metadata["severity"] != 1:
                raise RuntimeError("fixed evaluation severity must equal 1")
        datasets[split] = dataset.manifest["generation_summary"]
        print(f"Verified {split}: {len(dataset)} instances", flush=True)
    pool = load_dataset_split(config, "validation")
    selections = select_validation_subsets(pool, config["generator"]["dataset_pressure_weights"])
    root = project_path(config["paths"]["manifests_root"])
    previous = json.loads((root / "audit.json").read_text(encoding="utf-8")) if (root / "audit.json").exists() else {}
    if previous.get("protocol_hashes") != protocol_hashes(config):
        previous = {}
    write_json(root / "validation/subsets.json", selections)
    plan = training_sampling_plan(config, 2000)
    coverage = {}
    for index, (pressure, _) in enumerate(plan):
        preference = quality_preference_for_episode(config, algorithm_seed=11, quality_episode_index=index)
        kind = preference.source.removeprefix("endpoint_") if preference.source.startswith("endpoint_") else "mixed"
        key = pressure + ":" + kind
        coverage[key] = coverage.get(key, 0) + 1
    write_json(root / "audit.json", {
        "data_protocol_version": "2.0.0", "datasets": datasets,
        "protocol_hashes": protocol_hashes(config),
        "legacy_files_verified": protected_count, "training_pressure_preference_counts": coverage,
        "generation_and_audit_seconds": time.perf_counter() - started,
        "initial_generation_and_audit_seconds": previous.get("initial_generation_and_audit_seconds",
            previous.get("generation_and_audit_seconds", time.perf_counter() - started)),
        "validation_subsets": selections,
    })
    verify_legacy_files()
    print(f"Data v2 ready: {root / 'audit.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main(generation_workers=args.generation_workers, overwrite=args.overwrite)
