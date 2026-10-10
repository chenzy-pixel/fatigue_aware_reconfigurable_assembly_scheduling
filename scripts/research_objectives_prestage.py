"""Recompute historical objectives and run a resource-feasible prestaging pilot."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from agent.baselines.policies import HeuristicPolicy
from configs import load_config
from configs.config import public_config
from data import load_dataset_split
from data.selection import select_validation_subsets
from environment import AssemblySchedulingEnv, DecisionType
from environment.dynamics import quantize_to_ticks
from environment.fatigue_monitor import audit_fatigue
from environment.types import MachineState, OperationState

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "result/analysis/objective_prestage_research_20261008"
SUFFIX = "seed11_20261007_232505_102441"
THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6)
WINDOWS = (5.0, 10.0, 20.0)
FLOW_SCALE = 1089.15


def lower_bounds(instance):
    return {
        op.id: min(max(1, quantize_to_ticks(
            op.base_processing_time * machine.module_parameters[op.required_module].processing_speed_factor,
            instance.resolution,
        )) * instance.resolution for machine in instance.machines
        if op.required_module in machine.module_parameters)
        for op in instance.operations
    }


def historical_audit(config):
    dataset = load_dataset_split(config, "test")
    instances = {record.instance.instance_id: record.instance for record in dataset}
    rows, checks, idle_rows, provenance = [], [], [], []
    for objective in ("flow", "cost", "variance"):
        directory = ROOT / f"result/runs/ablation_eval_full_{objective}_{SUFFIX}"
        metrics = pd.read_csv(directory / "instance_metrics.csv")
        schedule = pd.read_csv(directory / "schedule.csv")
        recs = pd.read_csv(directory / "reconfigurations.csv")
        assert len(metrics) == 60 and not metrics.duplicated(["instance_id", "sampling_repeat"]).any()
        assert set(metrics.instance_id) == set(instances)
        assert set(metrics.sampling_repeat) == {0, 1, 2}
        for name in ("instance_metrics.csv", "schedule.csv", "reconfigurations.csv", "config.json"):
            path = directory / name
            provenance.append({"path": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        for metric in metrics.to_dict("records"):
            iid, repeat = metric["instance_id"], metric["sampling_repeat"]
            instance = instances[iid]
            s = schedule[(schedule.instance_id == iid) & (schedule.sampling_repeat == repeat)]
            r = recs[(recs.instance_id == iid) & (recs.sampling_repeat == repeat)]
            loads = np.array([r.loc[r.worker_id == worker.id, "duration"].sum() for worker in instance.workers])
            variance = float(np.var(loads))
            assert np.isclose(variance, metric["worker_load_variance"], atol=1e-8)
            bounds = lower_bounds(instance)
            lb = sum(bounds.values())
            success = bool(metric["task_succeeded"])
            if success:
                assert metric["flow_time_objective"] >= lb - 1e-8
            physical, _ = audit_fatigue(instance, r.to_dict("records"), metric["makespan"])
            assert np.isclose(physical["fatigue_monitor_peak"], metric["fatigue_monitor_peak"], atol=1e-8)
            row = {"objective": objective, "instance_id": iid, "sampling_repeat": repeat,
                   "success": success, "flow": metric["flow_time_objective"], "cost": metric["reconfiguration_cost"],
                   "workers": len(loads), "total_load": loads.sum(), "variance": variance,
                   "h_variance": len(loads)*variance, "total_squared_over_h": loads.sum()**2/len(loads),
                   "sum_squared": np.square(loads).sum(), "max_load": loads.max(),
                   "lp4": np.power(loads,4).sum()**0.25, "lp8": np.power(loads,8).sum()**0.125,
                   "peak_fatigue": physical["fatigue_monitor_peak"], "flow_lb": lb,
                   "reported_maximum_worker_fatigue": metric["maximum_worker_fatigue"],
                   "flow_excess": metric["flow_time_objective"]-lb}
            for threshold in THRESHOLDS:
                shifted = replace(instance, fatigue=replace(instance.fatigue, maximum_safe_fatigue=threshold))
                exposure, _ = audit_fatigue(shifted, r.to_dict("records"), metric["makespan"])
                row[f"area_{threshold:.2f}"] = exposure["fatigue_monitor_over_limit_area"]
            rows.append(row)
            # Independent integral check at each operation completion, including failures.
            releases = {order.id: order.release_time for order in instance.orders}
            completion = {}
            for order in instance.orders:
                os = s[s.order_id == order.id]
                partial = os.get("truncated", pd.Series(False,index=os.index)).fillna(False).astype(bool)
                if len(os) == len(order.operations) and not partial.any():
                    completion[order.id] = float(os.end.max())
            minimum = 0.0
            partial = s.get("truncated",pd.Series(False,index=s.index)).fillna(False).astype(bool)
            done = s[~partial]
            for time in sorted(set(done.end.astype(float))):
                flow = sum(max(0.0, min(time, completion.get(order.id, time)) - releases[order.id]) for order in instance.orders)
                debit = sum(bounds[op] for op in done.loc[done.end <= time+1e-9, "operation_id"])
                minimum = min(minimum, flow-debit)
            assert minimum >= -1e-7
            checks.append({"objective": objective, "instance_id": iid, "sampling_repeat": repeat,
                           "minimum_completed_debit_excess": minimum})
            if objective == "flow":
                op_specs = {op.id: op for op in instance.operations}
                for rec_id, task in r.groupby("reconfiguration_id", sort=False):
                    opid, machine = task.iloc[0].operation_id, task.iloc[0].machine_id
                    op = op_specs[opid]
                    predecessors = s[(s.order_id == op.order_id) & (s.sequence < op.sequence)]
                    ready = max(releases[op.order_id], float(predecessors.end.max()) if len(predecessors) else 0.0)
                    dis_start, install_end = float(task.start.min()), float(task.end.max())
                    prior_work = pd.concat([s[s.machine_id == machine][["start", "end"]], r[(r.machine_id == machine)&(r.reconfiguration_id != rec_id)][["start", "end"]]])
                    prior = prior_work[prior_work.end <= dis_start+1e-9]
                    idle_since = float(prior.end.max()) if len(prior) else 0.0
                    overlap = min(max(0.0,ready-idle_since), install_end-dis_start)
                    successors = s[(s.order_id == op.order_id)&(s.sequence > op.sequence)]
                    next_start = float(successors.start.min()) if len(successors) else None
                    op_process = s[s.operation_id == opid]
                    absorption = max(0.0,next_start-float(op_process.end.max())) if next_start is not None and len(op_process) else 0.0
                    idle_rows.append({"instance_id": iid,"sampling_repeat": repeat,"rec_id": rec_id,
                                      "ready_time":ready,"idle_since":idle_since,"stage_span":install_end-dis_start,
                                      "overlap_proxy":overlap,"next_operation_wait":absorption})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "historical_trajectory_metrics.csv", index=False)
    numeric = frame.select_dtypes("number").columns.difference(["sampling_repeat"])
    frame.groupby("objective")[numeric].mean().to_csv(OUT / "historical_objective_means.csv")
    frame[frame.success].groupby("objective")[numeric].mean().to_csv(OUT / "historical_success_means.csv")
    (OUT / "historical_source_hashes.json").write_text(json.dumps(provenance,indent=2),encoding="utf-8")
    pd.DataFrame(checks).to_csv(OUT / "flow_completed_debit_checks.csv",index=False)
    pd.DataFrame(idle_rows).to_csv(OUT / "reconfiguration_idle_overlap.csv",index=False)
    print(frame.groupby("objective")[["total_load","h_variance","total_squared_over_h","sum_squared","max_load","lp4","lp8","peak_fatigue",*[f"area_{t:.2f}" for t in THRESHOLDS]]].mean().round(4).to_string(), flush=True)
    return instances, provenance


def flow_scale_reference(config):
    manifest = json.loads((ROOT / config["objective_scalarizer"]["normalization_manifest"]).read_text())
    # Record the exact reference structure; calibration is resolved separately from test analysis.
    (OUT / "normalization_reference.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")


def flow_grid_audit(config):
    dataset = load_dataset_split(config,"test")
    bounds = {record.instance.instance_id:sum(lower_bounds(record.instance).values()) for record in dataset}
    validation = pd.read_csv(OUT/"prestage_validation_metrics.csv")
    baseline = validation[(validation.arm == "baseline") & validation.success]
    scale = float((baseline.flow-baseline.flow_lb).mean())
    assert scale > 0
    (OUT/"exploratory_flow_scale.json").write_text(json.dumps({
        "flow_excess_scale":scale,"source":"successful baseline heuristic on 50 fixed validation instances",
        "successful_count":len(baseline),"test_used_for_calibration":False,
        "status":"diagnostic scale only; no production manifest or checkpoint migration"},indent=2),encoding="utf-8")
    directory = ROOT / f"result/runs/ablation_eval_universal_{SUFFIX}"
    data = pd.read_csv(directory/"instance_metrics.csv").copy()
    assert len(data) == 20*66*3 and not data.duplicated(["instance_id","preference_key","sampling_repeat"]).any()
    data["lb"] = data.instance_id.map(bounds)
    assert data.lb.notna().all()
    data["excess"] = data.flow_time_objective-data.lb
    success = data[data.task_succeeded].copy()
    assert (success.excess >= -1e-8).all()
    success["q_old"] = success.flow_time_objective/(FLOW_SCALE+success.flow_time_objective)
    success["q_excess"] = success.excess/(scale+success.excess)
    cells = success.groupby(["instance_id","preference_key"]).agg(
        flow=("flow_time_objective","mean"),q_old=("q_old","mean"),q_excess=("q_excess","mean"),
        success_repeats=("sampling_repeat","count"),lb=("lb","first"))
    cells.to_csv(OUT/"universal_flow_cells.csv")
    summaries = []
    for iid, group in cells.groupby("instance_id"):
        summaries.append({"instance_id":iid,"successful_cells":len(group),
                          "fully_successful_cells":int((group.success_repeats==3).sum()),
                          "lb_over_fmin":bounds[iid]/group.flow.min(),
                          "flow_range_fraction":(group.flow.max()-group.flow.min())/group.flow.min(),
                          "old_q_range":group.q_old.max()-group.q_old.min(),
                          "new_q_range":group.q_excess.max()-group.q_excess.min()})
    pd.DataFrame(summaries).to_csv(OUT/"universal_flow_normalization.csv",index=False)
    print("Flow diagnostic scale",scale,"grid successes",len(success),flush=True)
    print(pd.DataFrame(summaries).drop(columns="instance_id").agg(["min","median","max"]).to_string(),flush=True)


class PrestagingEnv(AssemblySchedulingEnv):
    """Pilot extension: native reconfiguration stages with an unbound IDLE finish.

    operation_id identifies the forecast used by the rule. The operation runtime
    remains untouched; it is never reserved and may run on another machine.
    """
    def reset(self, *args, **kwargs):
        self.prestage_ids = set()
        self.prestage_log = []
        self._suppress_forecast_start = False
        return super().reset(*args, **kwargs)

    def _complete_installation(self, reconfiguration_id, worker_id):
        prestage = reconfiguration_id in self.prestage_ids
        self._suppress_forecast_start = prestage
        try:
            super()._complete_installation(reconfiguration_id, worker_id)
        finally:
            self._suppress_forecast_start = False
        if prestage:
            machine = self._machine_by_id(self.reconfigurations[reconfiguration_id].machine_id)
            machine.locked_operation_id = machine.source_module = machine.target_module = None

    def _start_processing(self, operation, machine):
        if not self._suppress_forecast_start:
            super()._start_processing(operation, machine)

    def _wait_opportunity(self):
        original = super()._wait_opportunity()
        window = getattr(self,"wakeup_window",None)
        if window is None or self.decision_type != DecisionType.PRODUCTION:
            return original
        if original is not None and original[0] == self.current_tick:
            return original
        for wave in self.instance.waves:
            release = min(order.release_time for order in self.instance.orders if order.wave == wave)
            tick = quantize_to_ticks(max(0.0,release-window),self.resolution)
            if self.current_tick < tick <= self.horizon_tick and (original is None or tick < original[0]):
                original = (tick,"prestage_window_open")
        return original

    def try_prestage(self, window):
        if self.decision_type != DecisionType.PRODUCTION or self._has_pending_worker_task():
            return False
        if any(rec_id in self._machine_reconfiguration.values() for rec_id in self.prestage_ids):
            return False
        # Forecast the earliest wave that has not yet started.
        future = []
        for wave, data in self.instance.waves.items():
            orders = [order for order in self.instance.orders if order.wave == wave]
            release = min(order.release_time for order in orders)
            gap = release - self.current_time
            if 0.0 < gap <= window:
                future.append((release, wave, data["dominant_module"], orders))
        if not future:
            return False
        release, wave, target, orders = min(future, key=lambda item:item[0])
        coverage = sum(machine.current_module == target and machine.state in {MachineState.IDLE,MachineState.PROCESSING}
                       or machine.target_module == target and machine.state != MachineState.IDLE for machine in self.machines)
        if coverage >= 2:
            return False
        candidates = []
        for index, machine in enumerate(self.machines):
            if machine.state != MachineState.IDLE or machine.current_module == target or target not in machine.spec.module_parameters:
                continue
            if any(op.state == OperationState.READY and op.spec.required_module in machine.spec.module_parameters for op in self.operations):
                continue
            source_coverage = sum(other.current_module == machine.current_module and other.state in {MachineState.IDLE,MachineState.PROCESSING}
                                  for other in self.machines)
            if source_coverage <= 1:
                continue
            duration = self._idle_worker_reconfiguration_ticks_at(machine,target,self.current_tick)
            if duration is None or self.current_tick+duration > quantize_to_ticks(release,self.resolution):
                continue
            candidates.append((duration,index))
        if not candidates:
            return False
        _, machine_index = min(candidates)
        forecast = next((op for order in orders for op in order.operations if op.required_module == target),None)
        if forecast is None:
            return False
        operation_index = self.instance.operation_index[forecast.id]
        operation = self.operations[operation_index]
        before = (operation.state,operation.machine_id)
        self._invalidate_resource_snapshot()
        self._execute_production_action(operation_index,machine_index)
        operation.state,operation.machine_id = before
        machine = self.machines[machine_index]
        machine.locked_operation_id = None
        rec_id = self._machine_reconfiguration[machine.spec.id]
        self.prestage_ids.add(rec_id)
        self.prestage_log.append({"rec_id":rec_id,"machine_id":machine.spec.id,"target":target,
                                  "time":self.current_time,"release":release,"wave":wave,"window":window})
        self._invalidate_resource_snapshot()
        return True


def rollout(config, instance, window=None, *, wakeup=False):
    env = AssemblySchedulingEnv(config) if window is None else PrestagingEnv(config)
    if wakeup:
        env.wakeup_window = window
    env.reset(instance,build_observation=False)
    policy = HeuristicPolicy()
    while not env.task_done:
        if window is not None:
            env.try_prestage(window)
        env.step(policy.select_action(env),build_observation=False)
    metrics = env.metrics()
    violations = env.validate_schedule()
    # Native validation plus independent release/module checks for the extension.
    for op in env.operations:
        if op.start_tick is not None:
            assert op.start_tick >= quantize_to_ticks(env._order_by_id(op.spec.order_id).release_time,env.resolution)
    assert not violations, violations
    assert metrics["maximum_worker_fatigue"] <= instance.fatigue.maximum_safe_fatigue+1e-8
    costs = sum(row["fixed_cost"]+row["duration"]*env._worker_by_id(row["worker_id"]).spec.labor_cost_per_minute
                for row in env.reconfiguration_log)
    costs += sum((min(rec.installation_end_tick or env.current_tick,env.current_tick)-rec.lock_tick)*env.resolution
                 *env._machine_by_id(rec.machine_id).spec.downtime_cost_per_minute for rec in env.reconfigurations.values())
    assert np.isclose(costs,metrics["reconfiguration_cost"],atol=1e-7)
    flow = sum(max(0.0,env._order_completion_tick.get(order.id,env.current_tick)*env.resolution-order.release_time)
               for order in instance.orders)
    if env.task_failed:
        flow += (len(instance.orders)-len(env._order_completion_tick))*instance.unfinished_order_penalty
    assert np.isclose(flow,metrics["flow_time_objective"],atol=1e-7)
    by_machine = {machine.id:machine.initial_module for machine in instance.machines}
    timeline = [(row["start"],1,"process",row) for row in env.schedule_log]
    timeline += [(row["end"],0,row["stage"],row) for row in env.reconfiguration_log]
    for _,_,kind,row in sorted(timeline,key=lambda item:item[:2]):
        if kind == "DIS":
            assert by_machine[row["machine_id"]] == row["source_module"]
            by_machine[row["machine_id"]] = instance.no_module_state
        elif kind == "INS":
            assert by_machine[row["machine_id"]] == instance.no_module_state
            by_machine[row["machine_id"]] = row["target_module"]
        else:
            assert by_machine[row["machine_id"]] == row["required_module"]
    row = {"instance_id":instance.instance_id,"arm":"baseline" if window is None else f"prestage_{window:g}",
           "success":env.task_succeeded,"failed":env.task_failed,"truncated":env.sampling_truncated,
           "flow":metrics["flow_time_objective"],"cost":metrics["reconfiguration_cost"],
           "variance":metrics["worker_load_variance"],"peak_fatigue":metrics["maximum_worker_fatigue"],
           "makespan":env.current_time,"reconfigurations":metrics["completed_reconfigurations"],
           "flow_lb":sum(lower_bounds(instance).values()),
           "prestage_count":len(getattr(env,"prestage_ids",set())),"violations":len(violations)}
    if wakeup:
        row["arm"] += "_wakeup"
    return row, getattr(env,"prestage_log",[])


def prestage_pilot(config):
    dataset = load_dataset_split(config,"validation")
    selection = select_validation_subsets(dataset,config["generator"]["dataset_pressure_weights"],
                                        target_count=50,diagnostic_count=0)["target"]
    (OUT/"validation_subset.json").write_text(json.dumps(selection,indent=2),encoding="utf-8")
    rows,logs = [],[]
    for count,index in enumerate(selection["instance_indices"],1):
        record = dataset[index]
        for window in (None,*WINDOWS):
            row, trace = rollout(config,record.instance,window)
            row["pressure_type"] = record.metadata["pressure_type"]
            rows.append(row)
            logs.extend({"instance_id":record.instance.instance_id,**value} for value in trace)
        pd.DataFrame(rows).to_csv(OUT/"prestage_validation_metrics.csv",index=False)
        if count % 5 == 0:
            print(f"prestage validation {count}/50",flush=True)
    frame = pd.DataFrame(rows)
    pd.DataFrame(logs).to_csv(OUT/"prestage_injections.csv",index=False)
    paired, summaries = [],[]
    base = frame[frame.arm == "baseline"].set_index("instance_id")
    for window in WINDOWS:
        arm = frame[frame.arm == f"prestage_{window:g}"].set_index("instance_id")
        common = base.success & arm.success
        result = {"arm":f"prestage_{window:g}","baseline_success":int(base.success.sum()),
                  "arm_success":int(arm.success.sum()),"common_success":int(common.sum()),
                  "injections":int(arm.prestage_count.sum()),"changed_instances":int((arm.prestage_count>0).sum())}
        for objective in ("flow","cost"):
            delta = arm.loc[common,objective]-base.loc[common,objective]
            pct = 100*delta/base.loc[common,objective]
            result.update({f"{objective}_mean_delta":float(delta.mean()),f"{objective}_mean_pct":float(pct.mean()),
                           f"{objective}_median_pct":float(pct.median()),f"{objective}_wins":int((delta < -1e-8).sum()),
                           f"{objective}_losses":int((delta > 1e-8).sum()),f"{objective}_ties":int((abs(delta)<=1e-8).sum()),
                           f"{objective}_wilcoxon_p":float(wilcoxon(delta).pvalue) if (abs(delta)>1e-8).any() else 1.0})
        for iid in base.index:
            paired.append({"instance_id":iid,"arm":f"prestage_{window:g}","common_success":bool(common.loc[iid]),
                           "flow_delta":arm.loc[iid,"flow"]-base.loc[iid,"flow"],
                           "cost_delta":arm.loc[iid,"cost"]-base.loc[iid,"cost"],"prestage_count":arm.loc[iid,"prestage_count"]})
        summaries.append(result)
    pd.DataFrame(paired).to_csv(OUT/"prestage_paired.csv",index=False)
    pd.DataFrame(summaries).to_csv(OUT/"prestage_summary.csv",index=False)
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)
    return summaries


def wakeup_pilot(config):
    dataset = load_dataset_split(config,"validation")
    selection = json.loads((OUT/"validation_subset.json").read_text())
    base = pd.read_csv(OUT/"prestage_validation_metrics.csv")
    base = base[base.arm == "baseline"].set_index("instance_id")
    assert len(base) == 50 and base.index.is_unique
    rows, logs = [], []
    for count,index in enumerate(selection["instance_indices"],1):
        record = dataset[index]
        row, trace = rollout(config,record.instance,10.0,wakeup=True)
        row["pressure_type"] = record.metadata["pressure_type"]
        rows.append(row)
        logs.extend({"instance_id":record.instance.instance_id,**value} for value in trace)
        if count%10 == 0:
            print(f"prestage wakeup validation {count}/50",flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT/"prestage_wakeup_metrics.csv",index=False)
    pd.DataFrame(logs).to_csv(OUT/"prestage_wakeup_injections.csv",index=False)
    arm = frame.set_index("instance_id")
    assert set(base.index) == set(arm.index)
    common = base.success & arm.success
    summary = {"arm":"prestage_10_wakeup","baseline_success":int(base.success.sum()),
               "arm_success":int(arm.success.sum()),"common_success":int(common.sum()),
               "injections":int(arm.prestage_count.sum()),"changed_instances":int((arm.prestage_count>0).sum())}
    for name in ("flow","cost"):
        delta = arm.loc[common,name]-base.loc[common,name]
        pct = delta/base.loc[common,name]*100
        summary.update({f"{name}_mean_delta":float(delta.mean()),f"{name}_mean_pct":float(pct.mean()),
                        f"{name}_median_pct":float(pct.median()),f"{name}_wins":int((delta < -1e-8).sum()),
                        f"{name}_losses":int((delta > 1e-8).sum()),f"{name}_ties":int((abs(delta)<=1e-8).sum()),
                        f"{name}_wilcoxon_p":float(wilcoxon(delta).pvalue) if (abs(delta)>1e-8).any() else 1.0})
    pd.DataFrame([summary]).to_csv(OUT/"prestage_wakeup_summary.csv",index=False)
    print(json.dumps(summary,indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase",choices=("all","audit","pilot","wakeup"),default="all")
    args = parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    config = load_config("configs/default.json")
    (OUT/"pilot_effective_config.json").write_text(json.dumps(public_config(config),indent=2),encoding="utf-8")
    source_paths = ("scripts/research_objectives_prestage.py","environment/env.py","agent/baselines/policies.py",
                    "data/manifests/v2/test/manifest.json","data/manifests/v2/validation/manifest.json")
    provenance = {path:hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in source_paths}
    provenance["git_commit"] = subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    (OUT/"pilot_source_hashes.json").write_text(json.dumps(provenance,indent=2),encoding="utf-8")
    if args.phase in {"all","audit"}:
        historical_audit(config)
        flow_scale_reference(config)
    if args.phase in {"all","pilot"}:
        prestage_pilot(config)
    if args.phase in {"all","pilot","wakeup"}:
        wakeup_pilot(config)
    if (OUT/"prestage_validation_metrics.csv").exists():
        flow_grid_audit(config)


if __name__ == "__main__":
    main()
