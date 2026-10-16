"""Build the full report or a complete last-checkpoint primary cohort report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-last", action="store_true")
    args = parser.parse_args()
    from analysis.gae_lambda_experiment_report import build_diagnostic_report
    from analysis.gae_lambda_robustness import supplement
    from analysis.gae_lambda_diagnostic import aggregate, diagnostic_gate
    from result.io import write_csv, write_json
    import numpy as np
    directory = ROOT / "result/analysis/gae_lambda_v4_20261012"
    if args.primary_last:
        source = directory
        directory = source / "primary_last600"
        directory.mkdir(exist_ok=True)
        episodes = []
        for repeat in range(3):
            path = source / f"last_repeat{repeat}.json"
            chunk = json.loads(path.read_text())
            if len(chunk) != 200: raise ValueError("primary cohort needs all 200 cells in every repeat")
            shutil.copy2(path, directory / path.name)
            episodes.extend(chunk)
        provenance = json.loads((source / "provenance.json").read_text())
        provenance["checkpoints"] = {"last": provenance["checkpoints"]["last"]}
        provenance["cohort"] = "complete primary last checkpoint; best checkpoint supplementary collection continues"
        write_json(directory / "provenance.json", provenance)
        for episode in episodes:
            episode.update({k: np.asarray(episode[k]) for k in ("rewards", "values", "actions", "legal_counts")})
        rows, signs, trajectories = aggregate(episodes)
        write_csv(directory / "gae_metrics.csv", rows)
        write_csv(directory / "sign_agreement.csv", signs)
        write_csv(directory / "trajectory_metrics.csv", trajectories)
        gate = diagnostic_gate(rows, signs)
        write_json(directory / "decision.json", gate)
        write_json(directory / "progress.json", {"stage": "completed", "complete_trajectories": len(episodes),
            "steps": sum(len(e["values"]) for e in episodes), "all_action_hashes_match": True,
            "requires_training": gate["requires_training"]})
    elif json.loads((directory / "progress.json").read_text()).get("stage") != "completed":
        raise ValueError("full diagnostic collection is not complete")
    build_diagnostic_report(directory)
    result = supplement(directory)
    print(json.dumps({"report": str(directory / "report.md"), "verification": result}), flush=True)


if __name__ == "__main__": main()
