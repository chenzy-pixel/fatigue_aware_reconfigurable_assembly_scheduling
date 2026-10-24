"""Epoch records for online training; partial objectives never enter final curves."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics

from result.io import write_csv, write_json

LOG_VERSION = "epoch_history_v2"
EPOCH_DEFINITION = "one_collection_and_learning_batch"
OBJECTIVES = {
    "flow": "flow_excess_objective",
    "cost": "reconfiguration_cost",
    "workload": "workload_objective_value",
}


def csv_rows(rows):
    return [{k: json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (dict, list, tuple)) else v
             for k, v in row.items()} for row in rows]


def instance_row(episode, *, epoch, method, training_seed, training_steps):
    metrics = episode.get("metrics") or {}
    meta = episode.get("metadata") or {}
    cutoff = bool(episode.get("budget_truncated") or metrics.get("truncated"))
    valid = bool(metrics.get("task_succeeded") and not cutoff
                 and not metrics.get("schedule_violations")
                 and metrics.get("maximum_worker_fatigue", float("inf")) <= metrics.get("safe_fatigue_limit", 0.) + 1e-9)
    index = int(episode["episode"])
    row = {
        "log_schema_version": LOG_VERSION, "epoch": epoch, "update_id": epoch,
        "epoch_definition": EPOCH_DEFINITION, "method": method, "training_seed": training_seed,
        "training_steps": training_steps, "episode": index + 1, "episode_index": index,
        "sampling_round": index // 27 + 1, "instance_id": episode.get("instance_id"),
        **{k: meta.get(k) for k in ("seed", "cell_id", "scale", "N", "K", "M", "W", "A")},
        "preference": metrics.get("preference", episode.get("preference")),
        "preference_key": metrics.get("preference_key"),
        "environment_steps": episode["steps"], "decision_count": metrics.get("decision_count"),
        "reward_sum": episode.get("reward_sum"), "budget_truncated": cutoff,
        "task_succeeded": bool(metrics.get("task_succeeded")),
        "task_failed": bool(metrics.get("task_failed")), "objective_valid": valid,
        "stop_reason": episode.get("cutoff_reason", "interaction_budget") if episode.get("budget_truncated") else metrics.get("terminal_reason"),
        "objective_status": "complete_valid" if valid else "partial" if cutoff or not metrics.get("terminated") else "failed",
        "cmax": metrics.get("time") if valid else None, "snapshot_time": metrics.get("time"),
        **{f"objective_{name}": metrics.get(key) if valid else None for name, key in OBJECTIVES.items()},
        **{f"snapshot_{name}": metrics.get(key) for name, key in OBJECTIVES.items()},
        **{k: v for k, v in episode.items() if k.endswith("_seconds")},
        **{f"metric_{k}": v for k, v in metrics.items()},
        "metadata": meta,
        "record_sha256": episode.get("record_sha256"),
    }
    return row


def aggregate(rows):
    good = [r for r in rows if r["objective_valid"]]
    result = {
        "instance_count": len(rows), "valid_objective_count": len(good),
        "valid_delivery_rate": len(good)/len(rows) if rows else None,
        "failed_count": sum(r.get("task_failed", False) for r in rows),
        "truncated_count": sum(r.get("budget_truncated", False) for r in rows),
        "environment_steps": sum(r["environment_steps"] for r in rows) if all(r.get("environment_steps") is not None for r in rows) else None,
        "decision_count": sum(r["decision_count"] for r in rows) if all(r.get("decision_count") is not None for r in rows) else None,
        "cell_counts": dict(Counter(r.get("cell_id") for r in rows)),
        "scale_counts": dict(Counter(r.get("scale") for r in rows)),
        "preference_counts": dict(Counter(r.get("preference_key") for r in rows)),
    }
    for name in OBJECTIVES:
        values = [r[f"objective_{name}"] for r in good if r.get(f"objective_{name}") is not None]
        result.update({f"{name}_{stat}": value for stat, value in {
            "count": len(values), "mean": statistics.fmean(values) if values else None,
            "std": statistics.pstdev(values) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None,
            "median": statistics.median(values) if values else None,
        }.items()})
        partial = [r.get(f"snapshot_{name}") for r in rows if r.get(f"snapshot_{name}") is not None]
        result[f"snapshot_{name}_mean"] = statistics.fmean(partial) if partial else None
    return result


def grouped_rows(rows, context):
    result = [{**context, "group_by": "all", "group": "all", **aggregate(rows)}]
    for names in (("scale",), ("cell_id",), ("preference_key",), ("cell_id", "preference_key")):
        groups = defaultdict(list)
        for row in rows:
            groups[tuple(row.get(k) for k in names)].append(row)
        for key, members in groups.items():
            result.append({**context, "group_by": "+".join(names),
                           "group": json.dumps(key, ensure_ascii=False), **aggregate(members)})
    return result


def export_history(output, history, episodes, validations=None):
    """Regenerable flat views; raw JSON remains the complete evidence."""
    output = Path(output)
    flat = [instance_row(e, epoch=e["epoch"], method=e["method"],
                         training_seed=e["training_seed"], training_steps=e["training_steps"]) for e in episodes]
    write_csv(output / "episodes.csv", csv_rows(flat))
    write_csv(output / "training_log.csv", csv_rows([
        {**r, **{"loss_"+k: v for k, v in r.get("losses", {}).items()}} for r in history]))
    groups = []
    by_epoch = defaultdict(list)
    for r in flat:
        by_epoch[r["epoch"]].append(r)
    for h in history:
        groups.extend(grouped_rows(by_epoch[h["epoch"]],
                                  {k: h.get(k) for k in ("epoch", "method", "training_seed", "training_steps")}))
    write_csv(output / "epoch_objectives.csv", csv_rows(groups))
    if validations is not None:
        records, curves = [], []
        for val in validations:
            context = {k: val.get(k) for k in ("epoch", "method", "training_seed", "training_steps")}
            batch = []
            for r in val["rows"]:
                valid = bool(r["success"])
                record = {**context, **r, "objective_valid": valid,
                          "preference_key": json.dumps(r["preference"]),
                          **{f"objective_{k}": r["objectives"][i] if valid else None
                             for i, k in enumerate(OBJECTIVES)}}
                batch.append(record)
            records.extend(batch)
            curves.extend(grouped_rows(batch, context))
        write_csv(output / "validation_instances.csv", csv_rows(records))
        write_csv(output / "validation_objectives.csv", csv_rows(curves))
        write_csv(output / "validation_log.csv", csv_rows([
            {**{k: v.get(k) for k in ("epoch", "training_steps")}, **v["summary"],
             **{f"cumulative_{k}": v[k] for k in ('validation_env_steps', 'validation_seconds') if k in v}}
            for v in validations]))


def save_epoch(output, context, episodes):
    """One atomic per-epoch record also survives an interrupted aggregate export."""
    folder = Path(output) / "epochs"
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / f"epoch_{context['epoch']:06d}.json", {"summary": context, "episodes": episodes})


def export_training(directory):
    """Rebuild current training CSV files from their saved epoch records."""
    from pathlib import Path
    import json
    for file in Path(directory).rglob('training_log.json'):
        folder = file.parent
        history = json.loads(file.read_text(encoding='utf-8'))
        episodes = json.loads((folder/'episodes.json').read_text(encoding='utf-8'))
        path = folder/'validations.json'
        validations = json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
        export_history(folder, history, episodes, validations)
