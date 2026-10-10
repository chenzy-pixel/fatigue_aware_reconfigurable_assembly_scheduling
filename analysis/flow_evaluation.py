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
from result.io import write_csv,write_json
from result.provenance import dataset_manifest_snapshot


def analyze_candidate_flow_runs(runs,output_dir,*,stage="final_test",require_complete=True):
    """Validate each source's training protocol before a shared metric comparison.

    Candidate budgets remain distinct: PPO repeats and solver search budgets are
    reported rather than relabeled as matched sampling budgets.
    """
    import csv
    groups=defaultdict(list); identity=None; sources=[]; budgets={}
    if not runs or stage not in {"validation", "final_test"}:
        raise ValueError("candidate evaluation requires runs and a known evaluation stage")
    for method,directory in runs.items():
        directory=Path(directory)
        config=load_config(directory/"config.json")
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
        for iid in ids:
            if iid not in instances: raise ValueError("candidate references unverified instance")
            env.reset(instances[iid],build_observation=False)
            bounds[iid]=env._total_processing_lower_bound_ticks*env.resolution
        for row in rows:
            _validate_row_protocol(row,config)
            if _flag(row.get("sampling_truncated",row.get("truncated",False))):
                raise ValueError("externally truncated evaluations cannot enter Pareto/HV analysis")
            if row["dataset_manifest_sha256"]!=actual_hash: raise ValueError("candidate dataset hash mismatch")
            key=PreferenceContext.from_input([float(row[f"preference_{n}"]) for n in ("flow","cost","variance")]).key
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
        if require_complete and observed!={(iid,key,r) for iid in ids for key in keys for r in range(repeats)}:
            raise ValueError("incomplete preference/repeat candidate matrix")
        sources.append({"method":method,"directory":str(directory),"flow_mode":config["objective_scalarizer"].get("flow_mode","raw_v1"),
                        "training_scales":config["objective_scalarizer"]["scales"],"repeat_budget":repeats,
                        "solver_evaluations_per_preference":config.get("mo_alns",{}).get("max_evaluations_per_preference") if rows[0]["arm"]!="ppo" else None})
    summaries=[]
    for (method,seed,iid),rows in groups.items():
        safe=[row for row in rows if valid_candidate(row)]
        record={"method":method,"algorithm_seed":seed,"instance_id":iid,"candidate_count":len(rows),
                "successful_safe_count":len(safe),"completion_rate":len(safe)/len(rows)}
        for name,field,scales in (("raw_primary","flow_time_objective",(1089.15,353.27,2.2629)),
                                  ("excess_supplementary","flow_excess_objective",(368.3143,353.27,2.2629))):
            points=[normalize_objectives((float(r[field]),float(r["reconfiguration_cost"]),float(r["worker_load_variance"])),scales) for r in safe]
            record[f"{name}_hypervolume"]=hypervolume_3d(points)
            record[f"{name}_front_size"]=len(nondominated_indices(points))
        summaries.append(record)
    output=Path(output_dir); output.mkdir(parents=True,exist_ok=True)
    write_csv(output/"instance_summary.csv",summaries)
    summary={"primary_flow_mode":"raw_v1","supplementary_flow_mode":"excess_proportional_lb_v1",
             "reference_point":[1,1,1],"sources":sources,"instances":summaries,
             "complete_matrix_required":require_complete}
    write_json(output/"summary.json",summary)
    return summary
