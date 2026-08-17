"""Equal-budget E1/E2 empirical Pareto evaluation and analysis.

The E2 arm contributes 22 deterministic greedy policies conditioned on the
denominator-five simplex lattice plus the canonical 0.5/0.3/0.2 preference.
The E1 control contributes one greedy and 21 fixed-seed sampled rollouts.  The
resulting fronts are empirical candidate sets, never certified true Pareto
fronts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr, wilcoxon

from agent.ppo import PPOAgent, build_actor_critic
from configs import load_config, project_path
from data import load_dataset_split
from environment import (
    AssemblySchedulingEnv,
    CANONICAL_PREFERENCE,
    PREFERENCE_NAMES,
    PreferenceVector,
    default_preference,
    simplex_lattice,
)
from eval import evaluate_dataset
from pareto_analysis import hypervolume_3d, nondominated_indices, normalize_objectives
from result.io import write_csv, write_json
from utils import set_seed


E2_ANALYSIS_PROTOCOL = "e2_preference_equal_budget_v1"
E1_SAMPLING_SEEDS = tuple(range(100001, 100022))
OBJECTIVE_FIELDS = (
    "flow_time_objective",
    "reconfiguration_cost",
    "worker_load_variance",
)
NORMALIZATION_SCALES = (1200.0, 1000.0, 50.0)


def e2_preference_grid() -> tuple[PreferenceVector, ...]:
    points = simplex_lattice(5, include=(CANONICAL_PREFERENCE,))
    if len(points) != 22:
        raise RuntimeError("the E2 evaluation grid must contain 22 preferences")
    return points


def equal_budget_design() -> dict[str, Any]:
    return {
        "protocol": E2_ANALYSIS_PROTOCOL,
        "candidate_budget_per_instance": 22,
        "e2_preferences": [point.as_dict() for point in e2_preference_grid()],
        "e1_greedy_count": 1,
        "e1_sampling_seeds": list(E1_SAMPLING_SEEDS),
        "canonical_preference": dict(
            zip(PREFERENCE_NAMES, CANONICAL_PREFERENCE, strict=True)
        ),
    }


def _checkpoint_agent(
    config: dict[str, Any],
    checkpoint: str | Path,
    dataset_name: str,
) -> PPOAgent:
    dataset = load_dataset_split(config, dataset_name)
    if not len(dataset):
        raise ValueError(f"dataset {dataset_name!r} is empty")
    environment = AssemblySchedulingEnv(config)
    observation = environment.reset(
        dataset[0].instance,
        preference=default_preference(config),
    )
    agent = PPOAgent(
        build_actor_critic(observation, config["network"]),
        config["ppo"],
        device=config["device"],
    )
    agent.load(project_path(checkpoint))
    agent.network.eval()
    return agent


def _candidate_id(
    arm: str,
    *,
    preference: PreferenceVector | None = None,
    sampling_seed: int | None = None,
) -> str:
    if preference is not None:
        return "e2_w_" + "_".join(f"{value:.12g}" for value in preference.as_tuple())
    if sampling_seed is None:
        return f"{arm}_greedy"
    return f"{arm}_sampled_{sampling_seed}"


def _tag_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    arm: str,
    algorithm_seed: int,
    dataset: str,
    candidate_id: str,
    candidate_source: str,
    sampling_seed: int | None,
) -> list[dict[str, Any]]:
    return [
        {
            **dict(row),
            "arm": arm,
            "algorithm_seed": int(algorithm_seed),
            "dataset": dataset,
            "candidate_id": candidate_id,
            "candidate_source": candidate_source,
            "sampling_seed": sampling_seed,
        }
        for row in rows
    ]


def collect_equal_budget_candidates(
    *,
    e1_config: dict[str, Any],
    e1_checkpoint: str | Path,
    e2_config: dict[str, Any],
    e2_checkpoint: str | Path,
    dataset_name: str,
    algorithm_seed: int,
    instance_limit: int | None = None,
) -> list[dict[str, Any]]:
    """Evaluate one algorithm-seed pair under the fixed 22-vs-22 budget."""

    e1 = deepcopy(e1_config)
    e2 = deepcopy(e2_config)
    e1["seed"] = int(algorithm_seed)
    e2["seed"] = int(algorithm_seed)
    set_seed(int(algorithm_seed))
    e1_agent = _checkpoint_agent(e1, e1_checkpoint, dataset_name)
    e2_agent = _checkpoint_agent(e2, e2_checkpoint, dataset_name)
    candidates: list[dict[str, Any]] = []
    manifest_hashes: set[str] = set()

    rows, _, _, aggregate = evaluate_dataset(
        e1,
        dataset_name=dataset_name,
        policy_name="ppo",
        ppo_agent=e1_agent,
        instance_limit=instance_limit,
        decode_mode="greedy",
    )
    manifest_hashes.add(str(aggregate["dataset_manifest_sha256"]))
    candidates.extend(
        _tag_rows(
            rows,
            arm="e1",
            algorithm_seed=algorithm_seed,
            dataset=dataset_name,
            candidate_id=_candidate_id("e1"),
            candidate_source="greedy",
            sampling_seed=None,
        )
    )
    for sampling_seed in E1_SAMPLING_SEEDS:
        rows, _, _, aggregate = evaluate_dataset(
            e1,
            dataset_name=dataset_name,
            policy_name="ppo",
            ppo_agent=e1_agent,
            instance_limit=instance_limit,
            decode_mode="sampled",
            sampling_seed=sampling_seed,
        )
        manifest_hashes.add(str(aggregate["dataset_manifest_sha256"]))
        candidates.extend(
            _tag_rows(
                rows,
                arm="e1",
                algorithm_seed=algorithm_seed,
                dataset=dataset_name,
                candidate_id=_candidate_id(
                    "e1", sampling_seed=sampling_seed
                ),
                candidate_source="sampled",
                sampling_seed=sampling_seed,
            )
        )

    for preference in e2_preference_grid():
        rows, _, _, aggregate = evaluate_dataset(
            e2,
            dataset_name=dataset_name,
            policy_name="ppo",
            ppo_agent=e2_agent,
            instance_limit=instance_limit,
            decode_mode="greedy",
            preference=preference,
        )
        manifest_hashes.add(str(aggregate["dataset_manifest_sha256"]))
        candidates.extend(
            _tag_rows(
                rows,
                arm="e2",
                algorithm_seed=algorithm_seed,
                dataset=dataset_name,
                candidate_id=_candidate_id("e2", preference=preference),
                candidate_source="preference_greedy",
                sampling_seed=None,
            )
        )
    if len(manifest_hashes) != 1:
        raise RuntimeError("E1/E2 evaluations did not use one fixed dataset manifest")
    return candidates


def _valid_candidate(row: Mapping[str, Any]) -> bool:
    def flag(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes"}:
                return True
            if normalized in {"false", "0", "no", ""}:
                return False
        return bool(value)

    return (
        flag(row.get("terminated"))
        and not flag(row.get("truncated"))
        and int(row.get("schedule_violation_count", 0)) == 0
        and all(math.isfinite(float(row[field])) for field in OBJECTIVE_FIELDS)
    )


def _objective_vector(row: Mapping[str, Any]) -> tuple[float, float, float]:
    return tuple(float(row[field]) for field in OBJECTIVE_FIELDS)  # type: ignore[return-value]


def _summary(values: Sequence[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"count": 0, "mean": None, "std": None, "minimum": None, "maximum": None}
    return {
        "count": len(finite),
        "mean": mean(finite),
        "std": pstdev(finite),
        "minimum": min(finite),
        "maximum": max(finite),
    }


def _safe_spearman(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 2 or len(set(x)) < 2 or len(set(y)) < 2:
        return math.nan
    value = float(spearmanr(x, y).statistic)
    return value if math.isfinite(value) else math.nan


def _canonical_row(rows: Sequence[dict[str, Any]], arm: str) -> dict[str, Any]:
    if arm == "e1":
        matches = [row for row in rows if row["candidate_source"] == "greedy"]
    else:
        matches = [
            row
            for row in rows
            if all(
                math.isclose(
                    float(row[f"w_{name}"]),
                    target,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                for name, target in zip(
                    PREFERENCE_NAMES, CANONICAL_PREFERENCE, strict=True
                )
            )
        ]
    if len(matches) != 1:
        raise ValueError(f"expected one canonical {arm} candidate, got {len(matches)}")
    return matches[0]


def _exact_wilcoxon(differences: Sequence[float]) -> dict[str, Any]:
    nonzero = [
        float(value)
        for value in differences
        if math.isfinite(float(value)) and not math.isclose(float(value), 0.0)
    ]
    if not nonzero:
        return {"nonzero_pairs": 0, "statistic": 0.0, "p_value": 1.0}
    result = wilcoxon(nonzero, alternative="two-sided", method="exact")
    return {
        "nonzero_pairs": len(nonzero),
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
    }


def _win_tie_loss(
    differences: Sequence[float],
    *,
    higher_is_better: bool,
    tolerance: float = 1e-12,
) -> dict[str, int]:
    finite = [float(value) for value in differences if math.isfinite(float(value))]
    oriented = finite if higher_is_better else [-value for value in finite]
    return {
        "wins": sum(value > tolerance for value in oriented),
        "ties": sum(abs(value) <= tolerance for value in oriented),
        "losses": sum(value < -tolerance for value in oriented),
    }


def _paired_seed_statistics(
    rows: Sequence[Mapping[str, Any]],
    *,
    e1_field: str,
    e2_field: str,
    higher_is_better: bool,
) -> dict[str, Any]:
    pairs = [
        (float(row[e1_field]), float(row[e2_field]))
        for row in rows
        if math.isfinite(float(row[e1_field]))
        and math.isfinite(float(row[e2_field]))
    ]
    e1_values = [pair[0] for pair in pairs]
    e2_values = [pair[1] for pair in pairs]
    differences = [e2 - e1 for e1, e2 in pairs]
    return {
        "higher_is_better": higher_is_better,
        "e1": _summary(e1_values),
        "e2": _summary(e2_values),
        "e2_minus_e1": _summary(differences),
        "win_tie_loss": _win_tie_loss(
            differences,
            higher_is_better=higher_is_better,
        ),
        "exact_wilcoxon": _exact_wilcoxon(differences),
    }


def analyze_candidate_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Compute per-instance fronts, responsiveness, and seed-level statistics."""

    candidates = [dict(row) for row in rows]
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        grouped[(str(row["dataset"]), int(row["algorithm_seed"]), str(row["instance_id"]))].append(row)
    annotated: list[dict[str, Any]] = []
    instance_rows: list[dict[str, Any]] = []

    for (dataset, algorithm_seed, instance_id), group in sorted(grouped.items()):
        raw_by_arm = {
            arm: [row for row in group if row["arm"] == arm]
            for arm in ("e1", "e2")
        }
        for arm in ("e1", "e2"):
            if len(raw_by_arm[arm]) != 22:
                raise ValueError(
                    f"{dataset}/{algorithm_seed}/{instance_id}/{arm} must "
                    f"contain 22 submitted candidates, got {len(raw_by_arm[arm])}"
                )
        valid = [row for row in group if _valid_candidate(row)]
        by_arm = {
            arm: [row for row in valid if row["arm"] == arm]
            for arm in ("e1", "e2")
        }
        for arm in ("e1", "e2"):
            if not by_arm[arm]:
                raise ValueError(
                    f"{dataset}/{algorithm_seed}/{instance_id}/{arm} has no "
                    "valid completed candidate"
                )
        arm_front_ids: dict[str, set[str]] = {}
        arm_hv: dict[str, float] = {}
        arm_front_spans: dict[str, tuple[float, float, float]] = {}
        arm_unique_traces: dict[str, int] = {}
        arm_unique_vectors: dict[str, int] = {}
        for arm, arm_rows in by_arm.items():
            normalized = [
                normalize_objectives(_objective_vector(row), NORMALIZATION_SCALES)
                for row in arm_rows
            ]
            front_indices = nondominated_indices(normalized)
            arm_front_ids[arm] = {
                str(arm_rows[index]["candidate_id"]) for index in front_indices
            }
            arm_hv[arm] = hypervolume_3d(
                [normalized[index] for index in front_indices]
            )
            front = [normalized[index] for index in front_indices]
            arm_front_spans[arm] = tuple(
                max(point[dimension] for point in front)
                - min(point[dimension] for point in front)
                for dimension in range(3)
            )
            arm_unique_traces[arm] = len(
                {str(row.get("action_trace_sha256")) for row in arm_rows}
            )
            arm_unique_vectors[arm] = len(
                {
                    tuple(round(value, 9) for value in _objective_vector(row))
                    for row in arm_rows
                }
            )
        union_normalized = [
            normalize_objectives(_objective_vector(row), NORMALIZATION_SCALES)
            for row in valid
        ]
        union_indices = set(nondominated_indices(union_normalized))
        union_hv = hypervolume_3d(
            [union_normalized[index] for index in sorted(union_indices)]
        )
        contributions = {"e1": 0, "e2": 0}
        for index, row in enumerate(valid):
            is_union = index in union_indices
            if is_union:
                contributions[str(row["arm"])] += 1
            annotated.append(
                {
                    **row,
                    "is_arm_pareto": str(row["candidate_id"])
                    in arm_front_ids[str(row["arm"])],
                    "is_union_pareto": is_union,
                    "valid_candidate": True,
                }
            )
        annotated.extend(
            {
                **row,
                "is_arm_pareto": False,
                "is_union_pareto": False,
                "valid_candidate": False,
            }
            for row in group
            if not _valid_candidate(row)
        )

        e2_rows = by_arm["e2"]
        correlations = {
            name: _safe_spearman(
                [float(row[f"w_{name}"]) for row in e2_rows],
                [float(row[field]) for row in e2_rows],
            )
            for name, field in zip(PREFERENCE_NAMES, OBJECTIVE_FIELDS, strict=True)
        }
        try:
            e1_canonical = _canonical_row(by_arm["e1"], "e1")
        except ValueError:
            e1_canonical = None
        try:
            e2_canonical = _canonical_row(by_arm["e2"], "e2")
        except ValueError:
            e2_canonical = None
        vertex_deltas: dict[str, float] = {}
        for name, field in zip(PREFERENCE_NAMES, OBJECTIVE_FIELDS, strict=True):
            vertices = [
                row
                for row in e2_rows
                if math.isclose(float(row[f"w_{name}"]), 1.0, abs_tol=1e-9)
            ]
            vertex_deltas[name] = (
                float(vertices[0][field]) - float(e2_canonical[field])
                if len(vertices) == 1 and e2_canonical is not None
                else math.nan
            )
        canonical_delta = (
            float(e2_canonical["quality_score"])
            - float(e1_canonical["quality_score"])
            if e1_canonical is not None and e2_canonical is not None
            else math.nan
        )
        instance_rows.append(
            {
                "dataset": dataset,
                "algorithm_seed": algorithm_seed,
                "instance_id": instance_id,
                "e1_hypervolume": arm_hv["e1"],
                "e2_hypervolume": arm_hv["e2"],
                "e2_minus_e1_hypervolume": arm_hv["e2"] - arm_hv["e1"],
                "union_hypervolume": union_hv,
                "e1_front_size": len(arm_front_ids["e1"]),
                "e2_front_size": len(arm_front_ids["e2"]),
                "e1_valid_candidates": len(by_arm["e1"]),
                "e2_valid_candidates": len(by_arm["e2"]),
                "e1_union_contribution": contributions["e1"],
                "e2_union_contribution": contributions["e2"],
                "e1_unique_action_traces": arm_unique_traces["e1"],
                "e2_unique_action_traces": arm_unique_traces["e2"],
                "e1_unique_objective_vectors": arm_unique_vectors["e1"],
                "e2_unique_objective_vectors": arm_unique_vectors["e2"],
                **{
                    f"{arm}_{name}_normalized_front_span": arm_front_spans[arm][index]
                    for arm in ("e1", "e2")
                    for index, name in enumerate(PREFERENCE_NAMES)
                },
                "flow_weight_objective_spearman": correlations["flow"],
                "cost_weight_objective_spearman": correlations["cost"],
                "variance_weight_objective_spearman": correlations["variance"],
                "flow_vertex_minus_canonical": vertex_deltas["flow"],
                "cost_vertex_minus_canonical": vertex_deltas["cost"],
                "variance_vertex_minus_canonical": vertex_deltas["variance"],
                "e1_canonical_quality_score": (
                    float(e1_canonical["quality_score"])
                    if e1_canonical is not None
                    else math.nan
                ),
                "e2_canonical_quality_score": (
                    float(e2_canonical["quality_score"])
                    if e2_canonical is not None
                    else math.nan
                ),
                "e2_minus_e1_canonical_quality_score": canonical_delta,
            }
        )

    seed_groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in instance_rows:
        seed_groups[(str(row["dataset"]), int(row["algorithm_seed"]))].append(row)
    seed_rows: list[dict[str, Any]] = []
    for (dataset, algorithm_seed), group in sorted(seed_groups.items()):
        fields = (
            "e1_hypervolume",
            "e2_hypervolume",
            "e2_minus_e1_hypervolume",
            "e1_union_contribution",
            "e2_union_contribution",
            "e1_unique_action_traces",
            "e2_unique_action_traces",
            "e1_unique_objective_vectors",
            "e2_unique_objective_vectors",
            "e1_canonical_quality_score",
            "e2_canonical_quality_score",
            "e2_minus_e1_canonical_quality_score",
            "flow_weight_objective_spearman",
            "cost_weight_objective_spearman",
            "variance_weight_objective_spearman",
            "flow_vertex_minus_canonical",
            "cost_vertex_minus_canonical",
            "variance_vertex_minus_canonical",
            *(
                f"{arm}_{name}_normalized_front_span"
                for arm in ("e1", "e2")
                for name in PREFERENCE_NAMES
            ),
        )
        seed_rows.append(
            {
                "dataset": dataset,
                "algorithm_seed": algorithm_seed,
                "instance_count": len(group),
                **{
                    f"mean_{field}": (
                        mean(finite)
                        if (
                            finite := [
                                float(row[field])
                                for row in group
                                if math.isfinite(float(row[field]))
                            ]
                        )
                        else math.nan
                    )
                    for field in fields
                },
            }
        )

    datasets = sorted({str(row["dataset"]) for row in seed_rows})
    statistics: dict[str, Any] = {}
    for dataset in datasets:
        selected = [row for row in seed_rows if row["dataset"] == dataset]
        statistics[dataset] = {
            "algorithm_seed_count": len(selected),
            "hypervolume": _paired_seed_statistics(
                selected,
                e1_field="mean_e1_hypervolume",
                e2_field="mean_e2_hypervolume",
                higher_is_better=True,
            ),
            "union_coverage_contribution": _paired_seed_statistics(
                selected,
                e1_field="mean_e1_union_contribution",
                e2_field="mean_e2_union_contribution",
                higher_is_better=True,
            ),
            "unique_action_traces": _paired_seed_statistics(
                selected,
                e1_field="mean_e1_unique_action_traces",
                e2_field="mean_e2_unique_action_traces",
                higher_is_better=True,
            ),
            "unique_objective_vectors": _paired_seed_statistics(
                selected,
                e1_field="mean_e1_unique_objective_vectors",
                e2_field="mean_e2_unique_objective_vectors",
                higher_is_better=True,
            ),
            "canonical_quality": _paired_seed_statistics(
                selected,
                e1_field="mean_e1_canonical_quality_score",
                e2_field="mean_e2_canonical_quality_score",
                higher_is_better=False,
            ),
            "normalized_front_span": {
                name: _paired_seed_statistics(
                    selected,
                    e1_field=f"mean_e1_{name}_normalized_front_span",
                    e2_field=f"mean_e2_{name}_normalized_front_span",
                    higher_is_better=True,
                )
                for name in PREFERENCE_NAMES
            },
            "preference_response_spearman": {
                name: _summary(
                    [
                        float(row[f"mean_{name}_weight_objective_spearman"])
                        for row in selected
                    ]
                )
                for name in PREFERENCE_NAMES
            },
            "vertex_minus_canonical": {
                name: _summary(
                    [
                        float(row[f"mean_{name}_vertex_minus_canonical"])
                        for row in selected
                    ]
                )
                for name in PREFERENCE_NAMES
            },
        }
    summary = {
        "analysis_protocol": E2_ANALYSIS_PROTOCOL,
        "true_pareto_claim": False,
        "candidate_design": equal_budget_design(),
        "candidate_row_count": len(candidates),
        "valid_candidate_row_count": sum(
            1 for row in annotated if bool(row["valid_candidate"])
        ),
        "invalid_candidate_row_count": sum(
            1 for row in annotated if not bool(row["valid_candidate"])
        ),
        "instance_count": len(instance_rows),
        "seed_count": len(seed_rows),
        "statistics": statistics,
        "interpretation": (
            "Fronts are empirical fixed-budget rollout sets. Canonical quality "
            "and preference controllability are reported separately."
        ),
    }
    return annotated, instance_rows, seed_rows, summary


def _read_candidate_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# E1/E2 equal-budget empirical Pareto analysis",
        "",
        "This report compares empirical rollout candidate sets, not certified true Pareto fronts.",
        "",
        f"- Candidate budget per arm and instance: {summary['candidate_design']['candidate_budget_per_instance']}.",
        f"- Valid candidate rows: {summary['valid_candidate_row_count']}.",
        f"- Invalid candidate rows excluded from fronts: {summary['invalid_candidate_row_count']}.",
        f"- Instance/seed summaries: {summary['instance_count']}/{summary['seed_count']}.",
        "- The canonical 0.5/0.3/0.2 score is reported independently of Pareto coverage.",
        "",
        "## Dataset-level seed statistics",
        "",
    ]
    for dataset, values in summary["statistics"].items():
        hv = values["hypervolume"]["e2_minus_e1"]
        test = values["hypervolume"]["exact_wilcoxon"]
        hv_wtl = values["hypervolume"]["win_tie_loss"]
        quality = values["canonical_quality"]["e2_minus_e1"]
        lines.extend(
            [
                f"### {dataset}",
                "",
                f"- Mean E2-E1 hypervolume: {hv['mean']} (std {hv['std']}).",
                f"- Hypervolume win/tie/loss: {hv_wtl['wins']}/{hv_wtl['ties']}/{hv_wtl['losses']}.",
                f"- Exact Wilcoxon p-value: {test['p_value']} over {test['nonzero_pairs']} nonzero seed pairs.",
                f"- Mean canonical-quality E2-E1: {quality['mean']} (lower is better).",
                "",
            ]
        )
    return "\n".join(lines)


def _render_plots(instance_rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.size": 12, "axes.spines.top": False, "axes.spines.right": False}
    )
    figure, axis = plt.subplots(figsize=(4.8, 4.2))
    x = np.asarray([float(row["e1_hypervolume"]) for row in instance_rows])
    y = np.asarray([float(row["e2_hypervolume"]) for row in instance_rows])
    lower = float(min(x.min(), y.min())) if len(x) else 0.0
    upper = float(max(x.max(), y.max())) if len(x) else 1.0
    axis.plot([lower, upper], [lower, upper], linestyle="--", color="#666666")
    axis.scatter(x, y, color="#2F6B9A", alpha=0.7)
    axis.set_xlabel("E1 normalized hypervolume")
    axis.set_ylabel("E2 normalized hypervolume")
    figure.tight_layout()
    figure.savefig(output / "hypervolume_comparison.pdf")
    figure.savefig(output / "hypervolume_comparison.png", dpi=300)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(5.4, 3.6))
    values = [
        [
            float(row[f"{name}_weight_objective_spearman"])
            for row in instance_rows
            if math.isfinite(float(row[f"{name}_weight_objective_spearman"]))
        ]
        for name in PREFERENCE_NAMES
    ]
    axis.boxplot(values, tick_labels=["Flow", "Cost", "Variance"])
    axis.axhline(0.0, linestyle="--", color="#666666")
    axis.set_ylabel("Spearman(weight, objective)")
    figure.tight_layout()
    figure.savefig(output / "preference_response.pdf")
    figure.savefig(output / "preference_response.png", dpi=300)
    plt.close(figure)


def analyze_candidates(
    candidates: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    render_plots: bool = True,
) -> dict[str, Any]:
    output = project_path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    annotated, instance_rows, seed_rows, summary = analyze_candidate_rows(candidates)
    write_csv(output / "candidates.csv", annotated)
    write_csv(
        output / "pareto_front.csv",
        [row for row in annotated if bool(row["is_union_pareto"])],
    )
    write_csv(output / "instance_summary.csv", instance_rows)
    write_csv(output / "seed_summary.csv", seed_rows)
    write_json(output / "summary.json", summary)
    (output / "report.md").write_text(_report(summary), encoding="utf-8")
    if render_plots and instance_rows:
        _render_plots(instance_rows, output)
    return summary


def run_manifest(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    instance_limit: int | None = None,
) -> dict[str, Any]:
    path = project_path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    runs = manifest.get("runs")
    datasets = manifest.get("datasets", ["test", "ood", "stress"])
    if not isinstance(runs, list) or not runs:
        raise ValueError("manifest.runs must be a non-empty list")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("manifest.datasets must be a non-empty list")
    candidates: list[dict[str, Any]] = []
    for run in runs:
        seed = int(run["algorithm_seed"])
        e1_config = load_config(run["e1_config"])
        e2_config = load_config(run["e2_config"])
        for dataset in datasets:
            candidates.extend(
                collect_equal_budget_candidates(
                    e1_config=e1_config,
                    e1_checkpoint=run["e1_checkpoint"],
                    e2_config=e2_config,
                    e2_checkpoint=run["e2_checkpoint"],
                    dataset_name=str(dataset),
                    algorithm_seed=seed,
                    instance_limit=instance_limit,
                )
            )
    return analyze_candidates(candidates, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run or analyze the E1/E2 equal-budget Pareto protocol"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest")
    source.add_argument("--candidate-csv", action="append")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--instance-limit", type=int)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if args.manifest:
        summary = run_manifest(
            args.manifest,
            args.output_dir,
            instance_limit=args.instance_limit,
        )
    else:
        candidates = [
            row
            for value in args.candidate_csv
            for row in _read_candidate_csv(project_path(value))
        ]
        summary = analyze_candidates(
            candidates,
            args.output_dir,
            render_plots=not args.no_plots,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"results: {project_path(args.output_dir)}")


if __name__ == "__main__":
    main()
