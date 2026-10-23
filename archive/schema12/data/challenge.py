"""Frozen post-training challenges, with candidate receipts and replayable witnesses."""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import random

from configs import load_config, project_path
from data.dataset import GeneratedInstanceRecord, canonical_json_bytes, sha256_file, template_sha256
from data.distribution import protocol_hashes
from data.feasibility import analyze_static_feasibility, cheap_feasibility_precheck
from data.generate_orders import (InstanceGenerator, _physical_precheck_config, _quantize,
                                  _rollout_metrics, _stable_seed)
from data.models import instance_to_dict, load_instance_yaml, parse_instance_dict, validate_instance

VERSION = "challenge_v1"
NAMES = ("test_hard", "reconfiguration_extrapolation")


def digest(value):
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(canonical_json_bytes(value))
    os.replace(temporary, path)


def seal(value):
    return {**value, "content_sha256": digest(value)}


def read_sealed(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = value.pop("content_sha256", None)
    if expected != digest(value):
        raise ValueError(f"content hash mismatch: {path}")
    return {**value, "content_sha256": expected}


def safe_file(root, name):
    if not isinstance(name, str) or Path(name).name != name or name in {"", ".", ".."} or "/" in name or "\\" in name:
        raise ValueError("unsafe challenge file name")
    return Path(root) / name


def physical_hash(instance):
    raw = instance_to_dict(instance) if not isinstance(instance, dict) else deepcopy(instance)
    raw.pop("instance_id", None)
    raw.pop("instance_type", None)
    return digest(raw)


def protected_files():
    """Snapshot published legacy and v2 files; exclude training caches."""
    paths = set()
    for folder in ("data/manifests", "data/instances"):
        root = project_path(folder)
        for split in ("validation", "test", "ood", "stress"):
            for prefix in (root, root / "v2"):
                paths.update((prefix / split).glob("*.json"))
    return {p.relative_to(project_path(".")).as_posix(): sha256_file(p) for p in sorted(paths)}


def existing_physical_hashes():
    hashes = set()
    for name in protected_files():
        if not name.startswith("data/instances/"):
            continue
        raw = json.loads(project_path(name).read_text(encoding="utf-8"))
        if "instance" in raw:
            hashes.add(physical_hash(raw["instance"]))
    return hashes


def load_spec(path="configs/challenge_suite.json"):
    spec = json.loads(project_path(path).read_text(encoding="utf-8"))
    if spec.get("version") != VERSION:
        raise ValueError("unsupported challenge version")
    if set(spec["seed_ranges"]) != set(NAMES):
        raise ValueError("challenge seed ranges must cover both suites")
    ranges = sorted(spec["seed_ranges"].values())
    if any(a < 6000000 or b <= a for a, b in ranges) or ranges[0][1] > ranges[1][0]:
        raise ValueError("invalid/overlapping challenge seed ranges")
    if not spec["pressures"] or len(set(spec["pressures"])) != len(spec["pressures"]):
        raise ValueError("pressure strata must be nonempty and unique")
    if any(type(q) is not int or q < 1 for q in spec["cost_quotas"].values()):
        raise ValueError("cost quotas must be positive integers")
    if not 1 <= spec["max_candidates_per_stratum"] <= 200:
        raise ValueError("candidate cap must be in [1,200]")
    if spec["scales"] != [1.4, 1.6, 1.8, 2.0] or spec["training_scale_upper"] != 1.6:
        raise ValueError("challenge_v1 requires the four declared scales and boundary")
    return spec


def physical_config(config):
    result = deepcopy(config)
    result["environment"]["fatigue_mode"] = "full"
    return result


def successful_witness(metrics):
    return bool(metrics.get("heuristic_completed") and not metrics.get("heuristic_truncated")
                and not metrics.get("schedule_violations")
                and metrics["heuristic_makespan"] <= metrics["horizon"] + 1e-9
                and metrics["maximum_worker_fatigue"] <= metrics["safe_limit"] + 1e-9)


def replay_certificate(instance, config, certificate):
    from environment import AssemblySchedulingEnv
    if certificate["physical_sha256"] != physical_hash(instance):
        raise ValueError("certificate instance mismatch")
    if certificate["action_sha256"] != digest(certificate["actions"]):
        raise ValueError("certificate action hash mismatch")
    env = AssemblySchedulingEnv(_physical_precheck_config(instance, physical_config(config)))
    env.reset(instance, build_observation=False)
    for action in certificate["actions"]:
        if env.terminated or env.truncated or type(action) is not int:
            raise ValueError("certificate has trailing/invalid actions")
        mask = env.get_action_mask()
        if not 0 <= action < len(mask) or mask[action]:
            raise ValueError("certificate contains illegal action")
        env.step(action, build_observation=False)
    metrics = env.metrics()
    if (not env.task_succeeded or env.truncated or env.validate_schedule()
            or metrics["maximum_worker_fatigue"] > instance.fatigue.maximum_safe_fatigue + 1e-9
            or metrics["time"] > instance.horizon + 1e-9):
        raise ValueError("certificate does not prove safe completion")
    if digest(env.schedule_log) != certificate["schedule_sha256"] or digest(env.reconfiguration_log) != certificate["reconfiguration_sha256"]:
        raise ValueError("certificate replay differs")
    return metrics


def diagnose(instance, config):
    actions = []
    metrics, env = _rollout_metrics(instance, physical_config(config), actions=actions)
    metrics.update(horizon=instance.horizon, safe_limit=instance.fatigue.maximum_safe_fatigue)
    certificate = None
    if successful_witness(metrics):
        certificate = {"version": "heuristic_witness_v1", "physical_sha256": physical_hash(instance),
                       "actions": actions, "action_sha256": digest(actions),
                       "schedule_sha256": digest(env.schedule_log),
                       "reconfiguration_sha256": digest(env.reconfiguration_log),
                       "schedule": env.schedule_log, "reconfigurations": env.reconfiguration_log}
        replay_certificate(instance, config, certificate)
    return {"feasibility_status": "observed_feasible" if certificate else "unknown",
            "diagnostic_terminal_reason": metrics["heuristic_terminal_reason"],
            "heuristic_metrics": metrics, "diagnostic_actions": actions, "certificate": certificate}


def apply_scale(base, latent, scale):
    times = {(x["machine"], x["module"], x["phase"]): x["unscaled"] for x in latent["unscaled_times"]}
    machines = tuple(replace(m, module_parameters={module: replace(
        p, installation_base_time=_quantize(times[m.id, module, "installation"] * scale, base.resolution),
        disassembly_base_time=_quantize(times[m.id, module, "disassembly"] * scale, base.resolution))
        for module, p in m.module_parameters.items()}) for m in base.machines)
    return replace(base, machines=machines, instance_id=f"{base.instance_id}_scale{scale:.1f}")


def build_candidate(config, suite, pressure, cost, seed):
    template = load_instance_yaml(project_path(config["paths"]["fixed_instance"]))
    generator = InstanceGenerator(template, config["generator"], config=config)
    latent = {}
    instance = generator._build_candidate(
        rng=random.Random(_stable_seed(VERSION, suite, pressure, cost, seed)), seed=seed,
        split=suite, pressure_type=pressure, cost_profile=cost, ood_factor=None, attempt=0,
        reconfiguration_scale_override=1.4 if suite == NAMES[1] else None, sampled_parameters=latent)
    validate_instance(instance, minimum_qualified_workers=3)
    analysis = analyze_static_feasibility(instance)
    reasons = generator._static_rejection_reasons(instance, analysis.load_metrics())
    precheck = cheap_feasibility_precheck(instance, analysis)
    reasons.extend(precheck.reason_codes)
    return instance, latent, precheck.to_dict(), sorted(set(reasons))


def _stratum_task(task):
    spec, config, identity, suite, pressure, cost, slot, quota = task
    root = project_path(spec["instances_root"]) / suite
    receipt_root = root / "candidates"
    start, end = spec["seed_ranges"][suite]
    accepted, receipts = [], []
    for index in range(spec["max_candidates_per_stratum"]):
        seed = start + slot * 1000 + index
        if seed >= end:
            raise ValueError("candidate seed range exhausted")
        path = receipt_root / f"candidate_{seed}.json"
        key = digest([identity, suite, pressure, cost, seed])
        if path.exists():
            receipt = read_sealed(path)
            if receipt["request_sha256"] != key:
                raise ValueError("candidate resume identity mismatch")
        else:
            receipt = {"request_sha256": key, "suite": suite, "pressure_type": pressure,
                       "cost_profile": cost, "seed": seed, "accepted": False}
            # A diagnostic or execution exception propagates; only structural rejection is recorded.
            instance, latent, precheck, reasons = build_candidate(config, suite, pressure, cost, seed)
            receipt.update(base_instance=instance_to_dict(instance), latent=latent,
                           precheck=precheck, rejection_reasons=reasons)
            if not reasons:
                members = []
                scales = spec["scales"] if suite == NAMES[1] else [latent["reconfiguration_scale"]]
                for scale in scales:
                    member = apply_scale(instance, latent, scale) if suite == NAMES[1] else instance
                    result = diagnose(member, config)
                    member_precheck = cheap_feasibility_precheck(member).to_dict()
                    metadata = {"suite": suite, "pressure_type": pressure, "cost_profile": cost,
                                "seed": seed, "severity": 1.0, "base_instance_id": instance.instance_id,
                                "reconfiguration_scale": scale, "latent_sha256": digest(latent["unscaled_times"]),
                                "scale_role": ("in_range" if scale < spec["training_scale_upper"] else
                                               "training_boundary" if scale == spec["training_scale_upper"] else "extrapolation"),
                                "feasibility_precheck": member_precheck, **result}
                    members.append({"instance": instance_to_dict(member), "metadata": metadata})
                receipt["members"] = members
                receipt["accepted"] = suite == NAMES[1] or members[0]["metadata"]["feasibility_status"] == "observed_feasible"
                if not receipt["accepted"]:
                    receipt["rejection_reasons"] = ["reference_witness_not_found"]
            receipt = seal(receipt)
            atomic_json(path, receipt)
        receipts.append({"path": path.name, "sha256": sha256_file(path)})
        if receipt["accepted"]:
            accepted.append(receipt)
        if len(accepted) == quota:
            break
    print(f"{suite}/{pressure}/{cost}: {len(accepted)}/{quota}, candidates={len(receipts)}", flush=True)
    return {"suite": suite, "pressure": pressure, "cost": cost, "quota": quota,
            "accepted": accepted, "receipts": receipts, "complete": len(accepted) == quota}


class ChallengeDataset:
    def __init__(self, manifest_path):
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = read_sealed(self.manifest_path)
        if self.manifest["version"] != VERSION or self.manifest["suite"] not in NAMES or self.manifest["status"] != "frozen":
            raise ValueError("challenge manifest is not frozen/compatible")
        self.instances_root = (self.manifest_path.parent / self.manifest["instances_root"]).resolve()
        self.records = []
        seen = set()
        for entry in self.manifest["files"]:
            path = safe_file(self.instances_root, entry["path"])
            if sha256_file(path) != entry["sha256"]:
                raise ValueError("challenge instance hash mismatch")
            raw = json.loads(path.read_text(encoding="utf-8"))
            instance = parse_instance_dict(raw["instance"])
            validate_instance(instance, minimum_qualified_workers=3)
            if instance.instance_id in seen or physical_hash(instance) != entry["physical_sha256"]:
                raise ValueError("duplicate ID/physical hash mismatch")
            seen.add(instance.instance_id)
            md = raw["metadata"]
            if md["suite"] != self.manifest["suite"]:
                raise ValueError("challenge suite mismatch")
            if md.get("certificate_file"):
                cp = safe_file(self.instances_root, md["certificate_file"])
                if sha256_file(cp) != md["certificate_sha256"]:
                    raise ValueError("challenge certificate hash mismatch")
                certificate = read_sealed(cp)
                if certificate["physical_sha256"] != physical_hash(instance) or digest(certificate["actions"]) != certificate["action_sha256"]:
                    raise ValueError("challenge certificate binding mismatch")
            elif md["feasibility_status"] == "observed_feasible":
                raise ValueError("observed feasibility requires a certificate")
            self.records.append(GeneratedInstanceRecord(instance, md))
        if len(self.records) != self.manifest["instance_count"] or not self.records:
            raise ValueError("challenge count mismatch")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


def load_suites(path):
    path = project_path(path)
    value = read_sealed(path)
    if "suite" in value:
        return [ChallengeDataset(path)]
    if value.get("version") != VERSION or value.get("status") != "frozen":
        raise ValueError("suite index is not frozen")
    result = []
    for entry in value["datasets"]:
        manifest = path.parent / entry["path"]
        if manifest.resolve().parent.parent != path.resolve().parent or sha256_file(manifest) != entry["sha256"]:
            raise ValueError("suite index manifest mismatch")
        result.append(ChallengeDataset(manifest))
    if {d.manifest["suite"] for d in result} != set(NAMES):
        raise ValueError("suite index missing datasets")
    return result


def audit_suites(datasets, config, *, replay=True, check_overlap=True):
    seen = existing_physical_hashes() if check_overlap else set()
    summary = {}
    for dataset in datasets:
        spec = dataset.manifest["spec"]
        expected = sum(spec["cost_quotas"].values()) * (len(spec["pressures"]) if dataset.manifest["suite"] == NAMES[0] else len(spec["scales"]))
        if len(dataset) != expected:
            raise ValueError("challenge target count mismatch")
        counts, statuses, groups, diagnostics = Counter(), Counter(), defaultdict(list), defaultdict(list)
        for record in dataset:
            instance, md = record.instance, record.metadata
            physical = physical_hash(instance)
            if physical in seen:
                raise ValueError("challenge overlaps published data or another instance")
            seen.add(physical)
            counts[(md["pressure_type"], md["cost_profile"]) ] += 1
            statuses[md["feasibility_status"]] += 1
            groups[md["base_instance_id"]].append(record)
            diagnostics[md["pressure_type"]].append(md["heuristic_metrics"])
            if replay and md.get("certificate_file"):
                replay_certificate(instance, config, read_sealed(safe_file(dataset.instances_root, md["certificate_file"])))
        if dataset.manifest["suite"] == NAMES[0]:
            expected_counts = {(p, c): q for p in spec["pressures"] for c, q in spec["cost_quotas"].items()}
            if counts != Counter(expected_counts) or statuses != Counter(observed_feasible=expected):
                raise ValueError("hard suite quotas/certificates incomplete")
        else:
            expected_counts = {("reconfiguration_bottleneck", c): q * len(spec["scales"]) for c, q in spec["cost_quotas"].items()}
            if counts != Counter(expected_counts) or len(groups) != sum(spec["cost_quotas"].values()):
                raise ValueError("extrapolation cost/group quotas incomplete")
            for group in groups.values():
                if sorted(r.metadata["reconfiguration_scale"] for r in group) != spec["scales"]:
                    raise ValueError("incomplete scale group")
                baseline = None
                for r in group:
                    raw = instance_to_dict(r.instance)
                    raw.pop("instance_id")
                    for m in raw["machines"]:
                        for p in m["module_parameters"].values():
                            p.pop("installation_base_time")
                            p.pop("disassembly_base_time")
                    if baseline is not None and raw != baseline:
                        raise ValueError("paired non-time physical fields differ")
                    baseline = raw
        receipts = dataset.manifest["candidate_receipts"]
        for receipt in receipts:
            p = safe_file(dataset.instances_root / "candidates", receipt["path"])
            if sha256_file(p) != receipt["sha256"]:
                raise ValueError("candidate ledger hash mismatch")
            saved = read_sealed(p)
            if saved["suite"] == NAMES[1] and saved["accepted"]:
                for member in saved["members"]:
                    actual = parse_instance_dict(member["instance"])
                    base = parse_instance_dict(saved["base_instance"])
                    target = apply_scale(base, saved["latent"], member["metadata"]["reconfiguration_scale"])
                    if instance_to_dict(actual) != instance_to_dict(target):
                        raise ValueError("paired scale/quantization mismatch")
        summary[dataset.manifest["suite"]] = {
            "instance_count": len(dataset), "feasibility_counts": dict(statuses),
            "candidate_count": len(receipts), "accepted_base_count": len(groups),
            "acceptance_rate": len(groups) / len(receipts),
            "mechanism_nonactivation": {p: {"count": len(rows),
                "no_worker_competition": sum(r["worker_competition_event_count"] == 0 for r in rows),
                "no_fatigue_mask": sum(r["fatigue_masked_action_count"] == 0 for r in rows)} for p, rows in diagnostics.items()},
            "all_certificates_replayed": replay}
    return summary


def build_suites(spec, *, workers=4):
    config = load_config(spec["base_config"])
    manifest_root = project_path(spec["manifests_root"])
    index_path = manifest_root / "suite.json"
    protected = protected_files()
    template = load_instance_yaml(project_path(config["paths"]["fixed_instance"]))
    source_hashes = {p: sha256_file(project_path(p)) for p in (
        "data/challenge.py", "data/generate_orders.py", "data/feasibility.py", "data/models.py",
        "environment/env.py", "agent/baselines/policies.py")}
    identity = digest({"spec": spec, "protocol": protocol_hashes(config), "template": template_sha256(template), "sources": source_hashes})
    if index_path.exists():
        index = read_sealed(index_path)
        if index["identity"] != identity:
            raise ValueError("frozen suite identity differs; use a new version directory")
        return audit_suites(load_suites(index_path), config)
    tasks = []
    for suite in NAMES:
        pressures = spec["pressures"] if suite == NAMES[0] else ["reconfiguration_bottleneck"]
        for slot, (pressure, cost) in enumerate((p, c) for p in pressures for c in spec["cost_quotas"]):
            tasks.append((spec, config, identity, suite, pressure, cost, slot, spec["cost_quotas"][cost]))
    if workers == 1:
        results = [_stratum_task(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_stratum_task, tasks))
    if not all(r["complete"] for r in results):
        atomic_json(manifest_root / "incomplete.json", {"status": "incomplete", "identity": identity,
                    "strata": [{k: v for k, v in r.items() if k != "accepted"} for r in results]})
        raise RuntimeError("challenge quotas incomplete; see incomplete.json")
    datasets, entries = [], []
    for suite in NAMES:
        root = project_path(spec["instances_root"]) / suite
        files, receipts = [], []
        for stratum in (r for r in results if r["suite"] == suite):
            receipts.extend(stratum["receipts"])
            for receipt in stratum["accepted"]:
                for member in receipt["members"]:
                    raw = deepcopy(member)
                    certificate = raw["metadata"].pop("certificate")
                    iid = raw["instance"]["instance_id"]
                    if certificate:
                        cp = root / f"{iid}.certificate.json"
                        atomic_json(cp, seal(certificate))
                        raw["metadata"].update(certificate_file=cp.name, certificate_sha256=sha256_file(cp))
                    path = root / f"{iid}.json"
                    atomic_json(path, raw)
                    files.append({"path": path.name, "sha256": sha256_file(path), "physical_sha256": physical_hash(raw["instance"])})
        mp = manifest_root / suite / "manifest.json"
        manifest = seal({"version": VERSION, "suite": suite, "status": "frozen", "identity": identity,
                         "spec": spec, "source_hashes": source_hashes, "protocol_hashes": protocol_hashes(config),
                         "physical_environment": config["environment"], "template_sha256": template_sha256(template),
                         "instances_root": os.path.relpath(root, mp.parent).replace("\\", "/"),
                         "instance_count": len(files), "files": files, "candidate_receipts": receipts})
        atomic_json(mp, manifest)
        datasets.append(ChallengeDataset(mp))
        entries.append({"path": mp.relative_to(manifest_root).as_posix(), "sha256": sha256_file(mp)})
    summary = audit_suites(datasets, config)
    if protected != protected_files():
        raise RuntimeError("published V1/V2 files changed during challenge generation")
    atomic_json(manifest_root / "audit.json", {"identity": identity, "datasets": summary,
                "protected_files": protected, "protected_file_count": len(protected), "protected_files_unchanged": True})
    atomic_json(index_path, seal({"version": VERSION, "status": "frozen", "identity": identity, "spec": spec,
                               "datasets": entries, "audit_sha256": sha256_file(manifest_root / "audit.json")}))
    return summary
