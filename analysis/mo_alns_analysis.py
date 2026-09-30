"""Compare PPO and MO-ALNS using the current formal experiment matrix."""
from __future__ import annotations
import argparse
import json
import math
from statistics import fmean, pstdev
from scipy.stats import wilcoxon
from configs import load_config
from analysis.pareto_analysis import analyze_rows as _analyze_rows, analyze_candidate_files as _analyze_files

PROTOCOL_VERSION = "ppo_mo_alns_solver_budget_v2"
ARMS = ("ppo", "mo_alns")


def _summary(values):
    values = [float(value) for value in values if math.isfinite(value)]
    return {"count": len(values), "mean": fmean(values) if values else None, "std": pstdev(values) if values else None}


def _paired(first, second, *, higher):
    pairs = [(left, right) for left, right in zip(first, second, strict=True) if math.isfinite(left) and math.isfinite(right)]
    delta = [right - left for left, right in pairs]
    oriented = delta if higher else [-value for value in delta]
    test = wilcoxon(delta, method="auto") if any(abs(value) > 1e-12 for value in delta) else None
    return {
        "ppo": _summary(left for left, _ in pairs),
        "mo_alns": _summary(right for _, right in pairs),
        "mo_alns_minus_ppo": _summary(delta),
        "win_tie_loss": {"wins": sum(value > 1e-12 for value in oriented), "ties": sum(abs(value) <= 1e-12 for value in oriented), "losses": sum(value < -1e-12 for value in oriented)},
        "wilcoxon": {"pair_count": len(pairs), "statistic": float(test.statistic) if test is not None else 0.0, "p_value": float(test.pvalue) if test is not None else 1.0},
    }


def analyze_rows(rows, config=None, *, stage="final_test"):
    annotated, instances, seeds, summary = _analyze_rows(rows, config, stage=stage, arms=ARMS)
    summary["analysis_protocol"] = PROTOCOL_VERSION
    statistics = {}
    for dataset in sorted({row["dataset"] for row in seeds}):
        values = [row for row in seeds if row["dataset"] == dataset]
        instance_sets = [{row["instance_id"] for row in instances if row["dataset"] == dataset and row["algorithm_seed"] == seed["algorithm_seed"]} for seed in values]
        if any(ids != instance_sets[0] for ids in instance_sets):
            raise ValueError("paired seed statistics require the same instance set at every seed")
        statistics[dataset] = {
            "algorithm_seed_count": len(values),
            "mo_alns_vs_ppo": {
                metric: _paired([row[f"mean_ppo_{metric}"] for row in values], [row[f"mean_mo_alns_{metric}"] for row in values], higher=higher)
                for metric, higher in (("hypervolume", True), ("completion_rate", True), ("preference_balanced_quality", False), ("union_contribution", True))
            },
        }
    summary["statistics"] = statistics
    return annotated, instances, seeds, summary


def analyze_candidate_files(paths, output_dir, config=None, *, stage="final_test"):
    return _analyze_files(paths, output_dir, config, stage=stage, analyzer=analyze_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppo-candidate-csv", action="append", required=True)
    parser.add_argument("--mo-alns-candidate-csv", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--stage", choices=("validation", "final_test"), default="final_test")
    args = parser.parse_args()
    summary = analyze_candidate_files(args.ppo_candidate_csv + args.mo_alns_candidate_csv, args.output_dir, load_config(args.config), stage=args.stage)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
