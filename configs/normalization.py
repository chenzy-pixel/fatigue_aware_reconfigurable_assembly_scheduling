"""Frozen factorial-27 objective scales and their calibration identity."""
from __future__ import annotations
import hashlib
import math
import json
from pathlib import Path, PureWindowsPath
from typing import Any

def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def apply_normalization_manifest(config: dict[str, Any], *, project_root: Path) -> None:
    scalarizer = config.get("objective_scalarizer")
    if not isinstance(scalarizer, dict):
        raise ValueError("config requires objective_scalarizer")
    source = str(scalarizer.get("scale_source", "frozen_manifest"))
    if source != "frozen_manifest":
        raise ValueError(f"unknown objective scalarizer scale_source {source!r}")
    manifest_path = scalarizer.get("normalization_manifest")
    expected = scalarizer.get("normalization_manifest_sha256")
    if not manifest_path or not expected:
        raise ValueError("frozen scales require manifest path and SHA256")
    path = Path(str(manifest_path))
    if (path.is_absolute() or PureWindowsPath(str(manifest_path)).is_absolute()) and not path.is_file():
        # Relocate snapshots from another checkout. The pinned digest is still
        # checked by load_normalization_manifest before this file is accepted.
        relocated = project_root / "configs" / "manifests" / PureWindowsPath(str(manifest_path)).name
        if relocated.is_file():
            path = relocated
    if not path.is_absolute():
        path = project_root / path
    if config.get("generator",{}).get("version")!="3.0.0":
        raise ValueError("generator schema 3 required")
    payload=path.read_bytes()
    import hashlib,json
    if hashlib.sha256(payload).hexdigest()!=expected:
        raise ValueError("normalization manifest SHA256 mismatch")
    manifest=json.loads(payload)
    if manifest.get("schema_version")!="factorial27_scales_v1":
        raise ValueError("new normalization schema required")
    if manifest.get("workload_objective")!=config["workload_objective"] or manifest.get("flow_mode")!=scalarizer.get("flow_mode"):
        raise ValueError("normalization objective identity mismatch")
    calibration = project_root / config['paths']['manifests_root'] / 'calibration' / 'manifest.json'
    if file_sha256(calibration) != manifest.get('calibration_manifest_sha256'):
        raise ValueError('normalization calibration manifest mismatch')
    from data.distribution import protocol_hashes
    calibration_identity = json.loads(calibration.read_text(encoding='utf-8'))
    if any(calibration_identity.get(k) != v for k, v in protocol_hashes(config).items()):
        raise ValueError('normalization generator protocol differs from calibration')
    scales=manifest["scales"]
    if set(scales)!={"flow","cost","workload"} or any(not math.isfinite(float(x)) or float(x)<=0 for x in scales.values()):
        raise ValueError("invalid objective scales")
    scalarizer["scales"]=scales
    from environment.workload import workload_config,objective_names
    identity=workload_config(config);network=config.setdefault("network",{})
    generated=dict(workload_mode=identity["mode"],workload_q=identity["q"],workload_order=identity["order"],workload_weight_rule=identity["weights"],workload_version=identity["version"],objective_names=objective_names(config),objective_scales=tuple(scales[k] for k in objective_names(config)),workload_scale=scales["workload"],flow_mode=scalarizer["flow_mode"],normalization_manifest_sha256=expected)
    for key,value in generated.items():
        if key in network:
            old=tuple(network[key]) if key in ("objective_scales","objective_names") else network[key]
            if old!=value: raise ValueError("network normalization identity differs: "+key)
        network[key]=value
    return
