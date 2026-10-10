from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from analysis.v8_pareto_analysis import analyze_dual_flow_runs
from configs import load_config
from data import load_dataset_split
from environment import AssemblySchedulingEnv
from result.io import write_config,write_csv,write_json
from result.provenance import build_provenance
from configs.config import project_path
from test_v8_pareto_analysis import _write_run


def test_dual_hv_accepts_distinct_training_modes_and_derives_old_success_fields(tmp_path,monkeypatch,fixed_instance):
    raw,metadata,rows=_write_run(tmp_path/"raw")
    dataset=load_dataset_split(raw,"test")
    records=[SimpleNamespace(instance=replace(fixed_instance,instance_id=iid)) for iid in ("test_a","test_b")]
    fake=type("Dataset",(list,),{})(records); fake.manifest=dataset.manifest
    monkeypatch.setattr("data.load_dataset_split",lambda config,name:fake)
    env=AssemblySchedulingEnv(raw); env.reset(fixed_instance,build_observation=False)
    lb=env._total_processing_lower_bound_ticks*env.resolution
    for row in rows: row["flow_time_objective"]=lb+100
    write_csv(tmp_path/"raw/instance_metrics.csv",rows)
    excess=load_config("configs/flow_excess/universal.json")
    directory=tmp_path/"excess"; directory.mkdir()
    write_config(directory,excess)
    newrows=deepcopy(rows)
    for row in newrows:
        row["flow_time_objective"]=lb+90
        row["flow_excess_objective"]=90
    write_csv(directory/"instance_metrics.csv",newrows)
    metadata=deepcopy(metadata)
    metadata["provenance"]=build_provenance(excess,dataset_manifest_path=project_path(excess["paths"]["manifests_root"])/"test/manifest.json",formal_evaluation_stage="final_test")
    write_json(directory/"metrics.json",metadata)
    summary=analyze_dual_flow_runs({"raw":tmp_path/"raw","excess":directory},tmp_path/"analysis")
    assert summary["primary"]["objective_scales"]["flow"]==1089.15
    assert summary["supplementary"]["objective_scales"]["flow"]==368.3143
    for metric in summary.values():
        assert metric["methods"]["excess"]["mean_pooled_hypervolume"]["mean"] > metric["methods"]["raw"]["mean_pooled_hypervolume"]["mean"]
    newrows[0]["flow_excess_objective"]=80
    write_csv(directory/"instance_metrics.csv",newrows)
    with pytest.raises(ValueError,match="excess report"):
        analyze_dual_flow_runs({"raw":tmp_path/"raw","excess":directory},tmp_path/"invalid")


def test_manifest_row_source_validation(tmp_path):
    import json
    from configs.normalization import load_normalization_manifest,canonical_json_sha256,file_sha256
    payload=json.loads(project_path("configs/manifests/flow_excess_scales_20261008.json").read_text())
    payload["sources"]["flow"]["baseline_rows"][0]["flow"]+=1
    payload.pop("content_sha256")
    payload["content_sha256"]=canonical_json_sha256(payload)
    path=tmp_path/"invalid.json"; path.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match="validation rows"):
        load_normalization_manifest(path,expected_sha256=file_sha256(path))


def test_solver_candidates_use_dual_metrics_and_enforce_matched_budget(tmp_path,monkeypatch,fixed_instance):
    from analysis.flow_evaluation import analyze_candidate_flow_runs
    from test_pareto_analysis import _formal_candidates
    from environment.types import bounded_quality_score
    from result.provenance import dataset_manifest_snapshot
    raw=load_config("configs/baselines/mo_alns.json")
    excess=load_config("configs/flow_excess/mo_alns.json")
    dataset=load_dataset_split(raw,"test")
    fake=type("Dataset",(list,),{})([SimpleNamespace(instance=replace(fixed_instance,instance_id="matrix_instance"))])
    fake.manifest_path=dataset.manifest_path
    monkeypatch.setattr("analysis.flow_evaluation.load_dataset_split",lambda config,name:fake)
    digest=dataset_manifest_snapshot(dataset.manifest_path)["sha256"]
    env=AssemblySchedulingEnv(raw); env.reset(fixed_instance,build_observation=False)
    lb=env._total_processing_lower_bound_ticks*env.resolution
    runs={}
    for label,config in (("raw",raw),("excess",excess)):
        directory=tmp_path/label; directory.mkdir(); runs[label]=directory
        write_config(directory,config)
        rows=_formal_candidates(config,stage="final_test",arm="mo_alns")
        for row in rows:
            row.update(flow_mode=config["objective_scalarizer"].get("flow_mode","raw_v1"),
                       flow_time_objective=lb+100,flow_excess_objective=100,dataset_manifest_sha256=digest)
            for name in ("flow","cost","variance"):
                row[f"preference_{name}"]=row[f"w_{name}"]
            row["preference_quality_score"]=bounded_quality_score(
                100 if label=="excess" else lb+100,300,3,config,
                preference=tuple(row[f"w_{name}"] for name in ("flow","cost","variance")))
        write_csv(directory/"instance_metrics.csv",rows)
    summary=analyze_candidate_flow_runs(runs,tmp_path/"matched")
    assert len(summary["instances"])==2
    for metric in ("raw_primary_hypervolume","excess_supplementary_hypervolume"):
        assert summary["instances"][0][metric]==summary["instances"][1][metric]
    excess["mo_alns"]["max_evaluations_per_preference"]+=1
    write_config(runs["excess"],excess)
    with pytest.raises(ValueError,match="evaluation budgets"):
        analyze_candidate_flow_runs(runs,tmp_path/"mismatched")
