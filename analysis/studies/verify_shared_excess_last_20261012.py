"""Independent volume sweep and instance-cluster uncertainty for the completed test."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "result/analysis/shared_excess_last_20261012"
RUNS = ROOT / "result/runs"
PREFIX = "remaining_flow_experiments_seed11_20261008_225956_"
KEYS = ["instance_id", "preference_key", "sampling_repeat"]
OBJECTIVES = ["flow_time_objective", "reconfiguration_cost", "worker_load_variance"]
PREFS = ["preference_flow", "preference_cost", "preference_variance"]
SCALES = {"raw": np.array([1089.15, 353.27, 2.2629]),
          "excess": np.array([368.3143, 353.27, 2.2629])}


def successful(frame):
    return frame.task_succeeded & frame.physical_safety_pass & (frame.schedule_violation_count == 0)


def q_raw(frame):
    obj = frame[OBJECTIVES].to_numpy(dtype=float)
    weighted = (obj / (SCALES["raw"] + obj)) * frame[PREFS].to_numpy(dtype=float)
    return (weighted.max(axis=1) + .05 * weighted.sum(axis=1)) / 1.05


def box_union_volume(points):
    """Integrate xy rectangle unions along z; no project Pareto/HV calls."""
    assert np.isfinite(points).all() and ((points >= 0) & (points <= 1)).all()
    levels = np.unique(np.r_[points[:, 2], 1.0])
    volume = 0.0
    for z0, z1 in zip(levels[:-1], levels[1:]):
        active = points[points[:, 2] <= z0]
        x_levels = np.unique(np.r_[active[:, 0], 1.0])
        area = 0.0
        for x0, x1 in zip(x_levels[:-1], x_levels[1:]):
            y0 = active[active[:, 0] <= x0, 1].min()
            area += (x1 - x0) * (1.0 - y0)
        volume += (z1 - z0) * area
    return volume


def main():
    paths = {"best": RUNS / (PREFIX + "shared_head_universal") / "final_sampled_instance_metrics.csv",
             "last": OUT / "last_test_instance_metrics.csv",
             "experts_raw_best": RUNS / (PREFIX + "main_universal") / "final_sampled_instance_metrics.csv"}
    frames = {name: pd.read_csv(path).set_index(KEYS).sort_index() for name, path in paths.items()}
    last = frames["last"]
    assert len(last) == 3960 and last.index.is_unique
    assert last.index.get_level_values("instance_id").nunique() == 20
    for frame in frames.values():
        assert frame.index.equals(last.index)
        assert frame.derived_sampling_seed.equals(last.derived_sampling_seed)
        assert frame.physical_safety_pass.all() and (frame.schedule_violation_count == 0).all()
    saved_hv = pd.read_csv(OUT / "checkpoint_hypervolumes.csv").query("stage == 'final_test'")
    recomputed = []
    for role, frame in frames.items():
        for iid, group in frame[successful(frame)].groupby(level="instance_id"):
            for mode, scale in SCALES.items():
                fields = OBJECTIVES if mode == "raw" else ["flow_excess_objective", *OBJECTIVES[1:]]
                values = group[fields].to_numpy(dtype=float)
                value = box_union_volume(values / (scale + values))
                expected = saved_hv[(saved_hv.role == role) & (saved_hv.instance_id == iid)
                                    & (saved_hv['mode'] == mode)].pooled_hv.item()
                recomputed.append({"role": role, "instance_id": iid, "mode": mode,
                                   "sweep_hv": value, "project_hv": expected,
                                   "absolute_error": abs(value - expected)})
    volumes = pd.DataFrame(recomputed)
    assert len(volumes) == 120 and volumes.absolute_error.max() < 1e-10
    volumes.to_csv(OUT / "independent_hypervolume_checks.csv", index=False, encoding="utf-8-sig")

    # Resample whole instances, retaining all their preferences and repeated trajectories.
    # These intervals describe this fixed 20-instance ID sample, not training-seed uncertainty.
    instances = sorted(last.index.get_level_values("instance_id").unique())
    preferences = sorted(last.index.get_level_values("preference_key").unique())
    rng = np.random.default_rng(20261012)
    draws = rng.multinomial(len(instances), np.full(len(instances), 1 / len(instances)), size=10000)
    intervals = []
    for role in ("best", "experts_raw_best"):
        first = frames[role]
        common = successful(first) & successful(last)
        differences = pd.DataFrame({"delta": q_raw(last.loc[common]) - q_raw(first.loc[common])},
                                   index=last.loc[common].index)
        summed = differences.groupby(level=KEYS[:2]).delta.sum().unstack().reindex(index=instances, columns=preferences).fillna(0)
        counts = differences.groupby(level=KEYS[:2]).size().unstack().reindex(index=instances, columns=preferences).fillna(0)
        quality_boot = ((draws @ summed.to_numpy()) / (draws @ counts.to_numpy())).mean(axis=1)
        assert np.isfinite(quality_boot).all()
        quality_delta = (summed.sum(axis=0) / counts.sum(axis=0)).mean()
        completion_by_instance = (successful(last).astype(float) - successful(first).astype(float)).groupby(level="instance_id").mean().reindex(instances)
        completion_boot = draws @ completion_by_instance.to_numpy() / len(instances)
        raw = volumes[volumes['mode'] == "raw"].pivot(index="instance_id", columns="role", values="sweep_hv").reindex(instances)
        hv_difference = raw['last'] - raw[role]
        hv_boot = draws @ hv_difference.to_numpy() / len(instances)
        for name, estimate, distribution in (("completion_delta", completion_by_instance.mean(), completion_boot),
                                               ("paired_success_quality_raw_delta", quality_delta, quality_boot),
                                               ("pooled_hv_raw_delta", hv_difference.mean(), hv_boot)):
            lo, hi = np.quantile(distribution, [.025, .975])
            intervals.append({"reference": role, "metric": name, "last_minus_reference": estimate,
                              "instance_cluster_95_low": lo, "instance_cluster_95_high": hi,
                              "instance_count": len(instances), "resamples": len(draws)})
    pd.DataFrame(intervals).to_csv(OUT / "instance_cluster_intervals.csv", index=False, encoding="utf-8-sig")
    failures = last[~successful(last)].reset_index()
    reasons = failures.termination_reason.value_counts().to_dict()
    failures.groupby(["instance_id", "pressure_type", "termination_reason"]).size().rename("failures").reset_index().to_csv(
        OUT / "last_test_failure_groups.csv", index=False, encoding="utf-8-sig")
    pref = last.reset_index().assign(valid=successful(last).to_numpy()).groupby(PREFS).valid.agg(["sum", "count", "mean"])
    pref.sort_values("mean").to_csv(OUT / "last_test_completion_by_preference.csv", encoding="utf-8-sig")
    result = {"trajectories": len(last), "successful": int(successful(last).sum()),
              "failed": len(failures), "failure_reasons": reasons,
              "volume_cross_checks": len(volumes), "maximum_volume_error": volumes.absolute_error.max(),
              "uncertainty_method": "10000 paired whole-instance bootstrap resamples; fixed ID sample only",
              "intervals": intervals}
    (OUT / "independent_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
