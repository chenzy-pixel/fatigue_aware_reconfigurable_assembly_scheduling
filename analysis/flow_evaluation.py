"""Common raw/excess evaluation for PPO and solver candidate artifacts."""
from collections import defaultdict
import json
from pathlib import Path

from analysis.pareto_analysis import (hypervolume_3d, normalize_objectives, nondominated_indices,
                                      valid_candidate, _validate_row_protocol, _flag)
from configs import load_config
from configs.formal_preferences import formal_preferences
from data import load_dataset_split
from environment import AssemblySchedulingEnv, PreferenceContext
from environment.workload import objective_names, is_modern_workload, workload_value
from configs.formal_preferences import objective_scales
from analysis.workload_dominance import (actual_worker_loads, resolve_workload_evaluation_config,
    historical_load_sources, workload_identity, labelled_workload_dominance)
from result.io import write_csv,write_json
from result.provenance import dataset_manifest_snapshot


def analyze_candidate_flow_runs(runs,output_dir,*,stage="final_test",require_complete=True,
    evaluation_config=None, own_goal=False, require_matched_sampling_budget=True):
    """Validate each source's training protocol before a shared metric comparison.

    Candidate budgets remain distinct: PPO repeats and solver search budgets are
    reported rather than relabeled as matched sampling budgets.
    """
    import csv
    groups=defaultdict(list); identity=None; sources=[]; budgets={}; configs_by_method={}; sampling_cells={}
    loaded_configs = {method: load_config(Path(directory)/"config.json") for method, directory in runs.items()}
    workload_comparison = evaluation_config is not None or any(is_modern_workload(config) for config in loaded_configs.values())
    if not runs or stage not in {"validation", "final_test"}:
        raise ValueError("candidate evaluation requires runs and a known evaluation stage")
    for method,directory in runs.items():
        directory=Path(directory)
        config=loaded_configs[method]
        configs_by_method[method]=config
        csv_path=directory/"instance_metrics.csv"
        if not csv_path.is_file():
            csv_path=directory/"final_sampled_instance_metrics.csv"
        with csv_path.open(encoding="utf-8-sig") as stream:
            rows=list(csv.DictReader(stream))
        if not rows: raise ValueError("candidate artifact has no rows")
        datasets={row["dataset"] for row in rows}
        if len(datasets)!=1: raise ValueError("mixed dataset artifact")
        dataset_name=next(iter(datasets)); dataset=load_dataset_split(config,dataset_name)
        actual_hash=dataset_manifest_snapshot(dataset.manifest_path)["sha256"]
        instances={record.instance.instance_id:record.instance for record in dataset}
        ids={row["instance_id"] for row in rows}
        keys={point.key for point in formal_preferences(config,stage)}
        seeds={int(row["algorithm_seed"]) for row in rows}
        if seeds != {int(config["seed"])}:
            raise ValueError("candidate seed disagrees with its training configuration")
        current=(dataset_name,actual_hash,tuple(sorted(ids)),tuple(sorted(keys)),tuple(sorted(seeds)),
                 json.dumps(config["environment"],sort_keys=True))
        if identity is None: identity=current
        elif identity!=current: raise ValueError("unmatched physical instances or environment")
        observed=set(); bounds={}; env=AssemblySchedulingEnv(config)
        load_sources = {}
        if workload_comparison and any(row.get("worker_loads") in {None, ""} for row in rows if _flag(row["task_succeeded"])):
            load_sources=historical_load_sources(directory,config,dataset_name,[row for row in rows if _flag(row["task_succeeded"])])
        for iid in ids:
            if iid not in instances: raise ValueError("candidate references unverified instance")
            env.reset(instances[iid],build_observation=False)
            bounds[iid]=env._total_processing_lower_bound_ticks*env.resolution
        for row in rows:
            _validate_row_protocol(row,config)
            if _flag(row.get("sampling_truncated",row.get("truncated",False))):
                raise ValueError("externally truncated evaluations cannot enter Pareto/HV analysis")
            if row["dataset_manifest_sha256"]!=actual_hash: raise ValueError("candidate dataset hash mismatch")
            key=PreferenceContext.from_input([float(row[f"preference_{n}"]) for n in objective_names(config)]).key
            repeat=int(row.get("sampling_repeat") or 0)
            cell=(row["instance_id"],key,repeat)
            if cell in observed: raise ValueError("duplicate candidate cell")
            observed.add(cell)
            row={**row,"method":method,"verified_flow_lb":bounds[row["instance_id"]]}
            if valid_candidate(row):
                excess=float(row["flow_time_objective"])-bounds[row["instance_id"]]
                if excess < -1e-8: raise ValueError("successful Flow is below verified LB")
                if row.get("flow_excess_objective") not in {None,""} and abs(float(row["flow_excess_objective"])-excess)>1e-7:
                    raise ValueError("candidate excess disagrees with verified LB")
                row["flow_excess_objective"]=max(0,excess)
                if workload_comparison:
                    loads=actual_worker_loads(row, **load_sources.get((row["instance_id"],str(row.get("candidate_id","")),str(row.get("sampling_repeat",0))),{}))
                    row["worker_loads"]=json.dumps(loads,sort_keys=True)
            groups[(method,int(row["algorithm_seed"]),row["instance_id"])].append(row)
        repeats=int(config["training"]["formal_evaluation"]["final_test_repeats" if stage=="final_test" else "validation_repeats"]) if rows[0]["arm"]=="ppo" else 1
        arm=rows[0]["arm"]
        if {row["arm"] for row in rows} != {arm}:
            raise ValueError("mixed candidate arms")
        if arm=="ppo":
            from utils import configured_formal_evaluation_sampling_seeds,derive_evaluation_sampling_seed
            sampling_seeds=configured_formal_evaluation_sampling_seeds(config,stage)
            for row in rows:
                seed=sampling_seeds[int(row["sampling_repeat"])]
                if (int(row["sampling_seed"])!=seed or int(row["derived_sampling_seed"]) !=
                        derive_evaluation_sampling_seed(seed,row["instance_id"],row["preference_key"])):
                    raise ValueError("candidate sampling seed mismatch")
            budget=(repeats,tuple(sampling_seeds))
        else:
            budget=(config.get("mo_alns",{}).get("max_evaluations_per_preference"),)
        if arm in budgets and budgets[arm]!=budget:
            raise ValueError("matched candidate arms have different evaluation budgets")
        budgets[arm]=budget
        sampling_cells[method]=observed
        if require_complete and observed!={(iid,key,r) for iid in ids for key in keys for r in range(repeats)}:
            raise ValueError("incomplete preference/repeat candidate matrix")
        sources.append({"method":method,"directory":str(directory),"flow_mode":config["objective_scalarizer"].get("flow_mode","raw_v1"),
                        "training_scales":config["objective_scalarizer"]["scales"],"repeat_budget":repeats,
                        "training_workload": workload_identity(config),
                        "solver_evaluations_per_preference":config.get("mo_alns",{}).get("max_evaluations_per_preference") if rows[0]["arm"]!="ppo" else None})
    summaries=[]
    from environment.types import flow_mode
    requested_modes = {flow_mode(evaluation_config)} if evaluation_config is not None else {"raw_v1", "excess_proportional_lb_v1"}
    if workload_comparison and require_matched_sampling_budget:
        cells=list(sampling_cells.values())
        if any(candidate_cells != cells[0] for candidate_cells in cells[1:]):
            raise ValueError("methods use unmatched preference/repeat candidate sampling budgets")
    for (method,seed,iid),rows in groups.items():
        safe=[row for row in rows if valid_candidate(row)]
        record={"method":method,"algorithm_seed":seed,"instance_id":iid,"candidate_count":len(rows),
                "successful_safe_count":len(safe),"completion_rate":len(safe)/len(rows)}
        for name,field,scales in (("raw_primary","flow_time_objective",(1089.15,353.27,2.2629)),
                                  ("excess_supplementary","flow_excess_objective",(368.3143,353.27,2.2629))):
            mode = "raw_v1" if name=="raw_primary" else "excess_proportional_lb_v1"
            if mode not in requested_modes or own_goal and mode != flow_mode(configs_by_method[method]):
                continue
            scoring=None
            if workload_comparison:
                scoring=resolve_workload_evaluation_config(configs_by_method[method],evaluation_config=evaluation_config,
                    own_goal=own_goal,flow_mode=mode)
                scales=objective_scales(scoring)
            points=[normalize_objectives((float(r[field]),float(r["reconfiguration_cost"]),
                workload_value(list(actual_worker_loads(r).values()),scoring) if scoring else float(r["worker_load_variance"])),scales) for r in safe]
            record[f"{name}_hypervolume"]=hypervolume_3d(points)
            record[f"{name}_front_size"]=len(nondominated_indices(points))
        summaries.append(record)
    output=Path(output_dir); output.mkdir(parents=True,exist_ok=True)
    write_csv(output/"instance_summary.csv",summaries)
    summary={"primary_flow_mode":"raw_v1","supplementary_flow_mode":"excess_proportional_lb_v1",
             "reference_point":[1,1,1],"sources":sources,"instances":summaries,
             "complete_matrix_required":require_complete}
    if evaluation_config is not None:
        summary.update(primary_flow_mode=flow_mode(evaluation_config), supplementary_flow_mode=None)
    elif own_goal:
        summary.update(primary_flow_mode="per_source_training_goal", supplementary_flow_mode=None)
    if workload_comparison:
        summary.update(evaluation_goal="own_training_goals_separate" if own_goal else "explicit_frozen_goal" if evaluation_config is not None else "unified_q05",
            matched_sampling_budget_required=require_matched_sampling_budget,
            labelled_worker_dominance=labelled_workload_dominance([row for group in groups.values() for row in group]))
        summary["evaluation_metrics"]={mode: {"scales":objective_scales(resolve_workload_evaluation_config(next(iter(configs_by_method.values())),
            evaluation_config=evaluation_config,own_goal=own_goal,flow_mode=mode)),
            **workload_identity(resolve_workload_evaluation_config(next(iter(configs_by_method.values())),
            evaluation_config=evaluation_config,own_goal=own_goal,flow_mode=mode))} for mode in requested_modes
            if not own_goal or mode == flow_mode(next(iter(configs_by_method.values())))}
    write_json(output/"summary.json",summary)
    return summary
