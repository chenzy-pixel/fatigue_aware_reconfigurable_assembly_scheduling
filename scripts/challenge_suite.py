"""Build, audit and evaluate independent post-training challenge suites."""
import argparse
from pathlib import Path

from configs import load_config, project_path
from data.challenge import (atomic_json, audit_suites, build_suites, load_spec, load_suites,
                            protected_files, read_sealed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("build", "audit", "smoke", "evaluate", "report"), required=True)
    parser.add_argument("--suite-config", default="configs/challenge_suite.json")
    parser.add_argument("--suite-manifest")
    parser.add_argument("--output-dir", default="result/audits/challenge_v1")
    parser.add_argument("--generation-workers", type=int, default=4)
    parser.add_argument("--training-config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--policy", choices=("ppo", "heuristic"), default="ppo")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    if args.generation_workers < 1:
        parser.error("generation-workers must be positive")
    spec = load_spec(args.suite_config)
    manifest = project_path(args.suite_manifest or str(Path(spec["manifests_root"]) / "suite.json"))
    output = project_path(args.output_dir)
    if args.stage == "build":
        summary = build_suites(spec, workers=args.generation_workers)
    elif args.stage == "audit":
        datasets = load_suites(manifest)
        config = load_config(datasets[0].manifest["spec"]["base_config"])
        summary = audit_suites(datasets, config)
        index = read_sealed(manifest)
        if "datasets" in index:
            from data.dataset import sha256_file
            audit_path = manifest.parent / "audit.json"
            if sha256_file(audit_path) != index["audit_sha256"]:
                raise ValueError("frozen build audit mismatch")
            import json
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            if audit["protected_files"] != protected_files():
                raise ValueError("published legacy/v2 files changed")
    else:
        from analysis.challenge_study import evaluate_suites, report_evaluation
        if args.stage == "report":
            summary = report_evaluation(output)
        else:
            training = args.training_config or spec["default_training_config"]
            checkpoint = args.checkpoint or spec["default_checkpoint"] if args.policy == "ppo" else None
            summary = evaluate_suites(manifest, training, checkpoint, output,
                                      policy=args.policy, smoke=args.stage == "smoke", device=args.device)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / f"{args.stage}_summary.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
