"""Delivery selection, accounting and historical protocol compatibility."""
from copy import deepcopy
import json
import math
from pathlib import Path

import pytest

from configs import load_config
from configs.config import public_config
from configs.runtime import attach_runtime_manifest
from result.provenance import effective_config_snapshot
from result.sample_selection import select_best_of_three
from training.protocol import LexicographicCheckpointSelector
from train import _aggregate_formal_rows, _validation_log_row, LearningRatePlateauController
from configs.formal_preferences import formal_preferences
from test_preference_protocol import _row


def rows_for(config, qualities=(.7, .4, .2), successes=(True, True, True), instances=1):
    return [{**_row(p.key, q, succeeded=s, index=i), "sampling_repeat": r,
             "maximum_worker_fatigue": 0., "safe_fatigue_limit": .75,
             "physical_safety_pass": True, "active_constraint_pass": True,
             "action_trace_sha256": f"hash-{i}-{p.key}-{r}",
             "inference_time_seconds": 1., "solve_time_seconds": 2.}
            for p in formal_preferences(config, "validation") for i in range(instances)
            for r, (q,s) in enumerate(zip(qualities,successes))]


def aggregate(config, rows, universal=False, instances=1):
    return _aggregate_formal_rows(config, rows=rows, dataset_name="validation", manifest="m.json",
                                  unique_instance_count=instances, repeat_count=3, universal=universal,
                                  strict_counts=True)


@pytest.mark.parametrize("successes,qualities,repeat", [
    ((False,True,False),(.1,.8,.2),1),
    ((True,True,True),(.7,.4,.2),2),
    ((True,True,False),(.4,.4,.1),0),
    ((False,False,False),(.8,.7,.6),0),  # all native failure Q are 1, tie on repeat
])
def test_native_success_quality_and_repeat_order(successes,qualities,repeat):
    c=load_config("configs/e1/single_flow.json");rows=rows_for(c,qualities,successes)
    result=aggregate(c,rows)
    selected=[r for r in rows if r["selected_for_aggregation"]]
    assert len(selected)==1 and selected[0]["sampling_repeat"]==repeat
    assert result["completion_rate"]==float(any(successes))
    assert result["raw_overall_completion_rate"]==sum(successes)/3
    assert result["success_count_distribution"][str(sum(successes))]==1
    assert result["total_solve_time_seconds"]==6
    assert result["total_inference_time_seconds"]==3
    assert result["request_count"]==3 and result["delivery_count"]==1
    if any(successes):
        assert result["preference_balanced_quality_score"]==min(q for q,s in zip(qualities,successes) if s)
    else:assert math.isinf(result["preference_balanced_quality_score"])


def test_preference_balancing_and_worst_preference():
    c=load_config("configs/controls/raw.json");rows=rows_for(c,instances=2)
    key=formal_preferences(c,"validation")[0].key
    for row in rows:
        if row["preference_key"]==key:
            row["preference_quality_score"]=.6
            if row["instance_id"]=="instance_0":
                row.update(task_succeeded=False,task_failed=True)
    a=aggregate(c,rows,universal=True,instances=2)
    assert a["cell_count"]==26 and a["request_count"]==78
    assert a["completed_cell_count"]==25 and a["completion_rate"]==.5
    assert a["overall_completion_rate"]==25/26
    assert a["preference_balanced_quality_score"]==pytest.approx((.6+12*.2)/13)
    for row in rows:
        if row["preference_key"]==key:row.update(task_succeeded=False,task_failed=True)
    assert math.isinf(aggregate(c,rows,universal=True,instances=2)["preference_balanced_quality_score"])


@pytest.mark.parametrize("defect",["missing","duplicate","wrong_repeat","missing_preference"])
def test_reject_incomplete_triples(defect):
    c=load_config("configs/controls/raw.json");rows=rows_for(c)
    if defect=="missing":rows.pop()
    if defect=="duplicate":rows[-1]=dict(rows[-2])
    if defect=="wrong_repeat":rows[-1]["sampling_repeat"]=3
    if defect=="missing_preference":rows=rows[3:]
    with pytest.raises(ValueError):aggregate(c,rows,universal=True)


@pytest.mark.parametrize("defect",["truncation","violation","physical_flag","fatigue"])
def test_losing_bad_sample_cannot_be_hidden(defect):
    c=load_config("configs/e1/single_flow.json");rows=rows_for(c)
    if defect=="truncation":rows[0].update(sampling_truncated=True,truncated=True)
    elif defect=="violation":rows[0]["schedule_violation_count"]=1
    elif defect=="physical_flag":rows[0]["physical_safety_pass"]=False
    else:rows[0]["maximum_worker_fatigue"]=1
    a=aggregate(c,rows)
    if defect=="truncation":
        assert a["completed_metrics"]=={} and a["preference_balanced_quality_score"] is None
    selector=LexicographicCheckpointSelector.from_config(c)
    event=selector.observe(a,completed_episodes=100,physical_safety_pass=a["active_constraint_pass"])
    assert event=="ineligible" and not selector.has_best


def test_selector_and_plateau_follow_selected_quality():
    c=load_config("configs/e1/single_flow.json");selector=LexicographicCheckpointSelector.from_config(c)
    plateau=LearningRatePlateauController.from_config(c);plateau.stale_validations=5
    for episode,q in [(100,.4),(200,.2)]:
        a=aggregate(c,rows_for(c,qualities=(q+.2,q+.1,q)))
        event=selector.observe(a,completed_episodes=episode,physical_safety_pass=True)
        plateau.observe(event in {"best_initialized","best_improved"})
        assert plateau.stale_validations==0
    assert selector.best_episode==200
    b=aggregate(c,rows_for(c,successes=(False,False,False)))
    assert selector.observe(b,completed_episodes=300,physical_safety_pass=True)=="not_improved"
    tied=aggregate(c,rows_for(c,qualities=(.4,.3,.2)))
    assert selector.observe(tied,completed_episodes=400,physical_safety_pass=True)=="tied"
    assert selector.best_episode==200
    row=_validation_log_row(tied,episode=400)
    assert row["sample_aggregation"]=="best_of_3_v1" and row["request_count"]==3
    assert json.loads(row["success_count_distribution"])["3"]==1


def test_new_roundtrip_and_historical_missing_field(tmp_path):
    c=load_config("configs/default.json")
    assert c["runtime_manifest"]["training_protocol"].endswith("_best_of_3_v1")
    path=tmp_path/"config.json";path.write_text(json.dumps(public_config(c)),encoding="utf-8")
    assert effective_config_snapshot(load_config(path))["sha256"]==effective_config_snapshot(c)["sha256"]
    old=deepcopy(c);old.pop("runtime_manifest");old["training"]["formal_evaluation"].pop("sample_aggregation")
    attach_runtime_manifest(old)
    assert "sample_aggregation" not in old["runtime_manifest"]
    path.write_text(json.dumps(public_config(old)),encoding="utf-8")
    loaded=load_config(path)
    assert effective_config_snapshot(loaded)["sha256"]==effective_config_snapshot(old)["sha256"]
    # Historical snapshots which extend a newly changed default retain their hash too.
    snapshot=public_config(old);snapshot["extends"]=str(Path(__file__).resolve().parents[1]/"configs/default.json")
    path.write_text(json.dumps(snapshot),encoding="utf-8")
    assert effective_config_snapshot(load_config(path))["sha256"]==effective_config_snapshot(old)["sha256"]


def test_invalid_configuration_and_legacy_explicit(tmp_path):
    c=public_config(load_config("configs/default.json"));c.pop("runtime_manifest")
    for field,value in [("sample_aggregation","unknown"),("validation_repeats",1),("final_test_repeats",2)]:
        bad=deepcopy(c);bad["training"]["formal_evaluation"][field]=value
        path=tmp_path/"bad.json";path.write_text(json.dumps(bad),encoding="utf-8")
        with pytest.raises(ValueError):load_config(path)
    c["training"]["formal_evaluation"].update(sample_aggregation="per_sample_v1",validation_repeats=1,final_test_repeats=1)
    path.write_text(json.dumps(c),encoding="utf-8")
    assert load_config(path)["training"]["formal_evaluation"]["sample_aggregation"]=="per_sample_v1"


def test_historical_dispatch_keeps_all_requests():
    c=load_config("configs/e1/single_flow.json")
    c["training"]["formal_evaluation"].pop("sample_aggregation")
    rows=rows_for(c,successes=(True,False,True))
    a=aggregate(c,rows)
    assert a["completion_rate"]==2/3
    assert a["instance_count"]==3
    assert not any("selected_for_aggregation" in row for row in rows)


def test_selector_rejects_mixed_aggregation_protocols():
    c=load_config("configs/e1/single_flow.json")
    selector=LexicographicCheckpointSelector.from_config(c)
    a=aggregate(c,rows_for(c));a["sample_aggregation"]="per_sample_v1"
    with pytest.raises(ValueError,match="aggregation disagree"):
        selector.observe(a,completed_episodes=100,physical_safety_pass=True)
