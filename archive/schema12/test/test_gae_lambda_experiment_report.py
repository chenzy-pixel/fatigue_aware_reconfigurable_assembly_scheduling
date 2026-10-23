import numpy as np
import pandas as pd
import pytest

from analysis.gae_lambda_experiment_report import test_metrics as summarize_test


def cells():
    return pd.DataFrame([{"instance_id": str(i), "preference_key": str(p), "sampling_repeat": r,
        "task_succeeded": True, "schedule_violation_count": 0, "maximum_worker_fatigue": .2,
        "safe_fatigue_limit": .6, "flow_excess_objective": 1., "reconfiguration_cost": 2.,
        "worker_load_variance": 3., "flow_time_objective": 100+p+5*(r-1)}
        for i in range(20) for p in range(66) for r in range(3)])


def test_hv_is_per_instance_and_span_averages_repeats():
    values, instances = summarize_test(cells(), np.ones(3))
    assert values["mean_instance_pooled_hv"] == pytest.approx(.5*(1-2/3)*.25)
    assert values["mean_instance_single_repeat_hv"] == pytest.approx(values["mean_instance_pooled_hv"])
    assert values["mean_instance_flow_span"] == 65
    assert values["completion_rate"] == 1
    assert values["full_grid_all_repeats_successful_instances"] == 20


def test_failed_instance_stays_in_hv_denominator_and_coverage():
    frame = cells()
    frame.loc[frame.instance_id == "0", "task_succeeded"] = False
    values, instances = summarize_test(frame, np.ones(3))
    assert values["completion_rate"] == .95
    assert values["mean_instance_pooled_hv"] == pytest.approx(.95*.5*(1-2/3)*.25)
    failed = next(x for x in instances if x["instance_id"] == "0")
    assert failed["pooled_hv"] == 0
    assert failed["successful_preferences"] == 0
    assert failed["flow_span_successful_preference_means"] is None


def test_partial_or_duplicate_test_never_enters_final_summary():
    frame = cells()
    with pytest.raises(ValueError, match="all 20"):
        summarize_test(frame.iloc[:-1], np.ones(3))
    with pytest.raises(ValueError, match="duplicate"):
        summarize_test(pd.concat([frame, frame.iloc[:1]]), np.ones(3))
