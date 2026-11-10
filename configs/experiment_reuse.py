"""Audited DCAN execution refresh within an existing experiment root."""
import hashlib
import json
from pathlib import Path
import zipfile

from configs import project_path
from agent.comparators.common import atomic_json, digest


REGISTRATION = "execution_reuse.json"
MANIFEST = "configs/dcan_execution_reuse_v1.json"


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def archive_payload(path):
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read("experiment_identity.json"))


def compatibility_manifest():
    value = _read(project_path(MANIFEST))
    if value.get("version") != "dcan_execution_reuse_v1" or value.get("reset_arms") != ["dcan"]:
        raise ValueError("invalid DCAN execution compatibility manifest")
    for path, approved in value.get("report_sources", {}).items():
        if _sha(project_path(path)) not in approved:
            raise ValueError(f"report source revision is not approved for DCAN refresh: {path}")
    return value


def compare_payloads(previous, current, manifest):
    """Accept exactly the reviewed source change with identical experiment inputs."""
    if digest({k:v for k,v in previous.items() if k != "sources"}) != digest({k:v for k,v in current.items() if k != "sources"}):
        raise ValueError("DCAN refresh requires identical suite, configurations and fixed data")
    old, new = previous["sources"], current["sources"]
    changed = {p for p in old.keys() | new.keys() if old.get(p) != new.get(p)}
    unexpected = changed - manifest["sources"].keys()
    if unexpected:
        raise ValueError("unreviewed source changes for DCAN refresh: " + ", ".join(sorted(unexpected)))
    for path, rule in manifest["sources"].items():
        if old.get(path) not in rule["before"] or new.get(path) not in rule["after"]:
            raise ValueError(f"source revision is not approved for DCAN refresh: {path}")
    return {p:{"before":old.get(p), "after":new.get(p)} for p in sorted(changed)}


def prepare_refresh(root, current_payload):
    root = Path(root)
    original = _read(root/"preflight.json")
    if original.get("status") != "passed":
        raise ValueError("DCAN refresh requires the original passed preflight")
    previous = archive_payload(root/"source.zip")
    if digest(previous) != original.get("identity"):
        raise ValueError("original source snapshot does not match the original preflight")
    if original.get("source_hashes") != previous["sources"]:
        raise ValueError("original preflight source hashes differ from its snapshot")
    manifest = compatibility_manifest()
    changed = compare_payloads(previous, current_payload, manifest)
    for directory in (root/"training"/"dcan", *(root/"evaluation"/task/"dcan" for task in ("frontier", "independent_request"))):
        if directory.exists() and any(p.is_file() for p in directory.rglob("*")):
            raise ValueError(f"stop the old run and clear the DCAN results before refresh: {directory}")
    return {"version":"dcan_execution_reuse_v1", "comparison_identity":original["identity"],
            "execution_identity":digest(current_payload), "reset_arms":["dcan"],
            "original_preflight_sha256":_sha(root/"preflight.json"),
            "original_source_sha256":_sha(root/"source.zip"),
            "manifest":manifest, "changed_sources":changed}


def publish_refresh(root, plan):
    """Publish only after the new CUDA/CPU probes and source snapshot succeeded."""
    root = Path(root)
    execution = root/"executions"/plan["execution_identity"]
    passed = _read(execution/"preflight.json")
    if passed.get("status") != "passed" or passed.get("identity") != plan["execution_identity"]:
        raise ValueError("DCAN refresh requires a passed new-execution preflight")
    if digest(archive_payload(execution/"source.zip")) != plan["execution_identity"]:
        raise ValueError("new source snapshot differs from the execution identity")
    value = {**plan, "preflight_sha256":_sha(execution/"preflight.json"),
             "source_sha256":_sha(execution/"source.zip")}
    value["content_sha256"] = digest(value)
    atomic_json(execution/"compatibility.json", value)
    atomic_json(root/REGISTRATION, value)
    return value


def registered_execution(root):
    path = Path(root)/REGISTRATION
    if not path.exists():
        return None
    value = _read(path)
    if (value.get("version") != "dcan_execution_reuse_v1" or value.get("reset_arms") != ["dcan"]
            or value.get("content_sha256") != digest({k:v for k,v in value.items() if k != "content_sha256"})):
        raise ValueError("invalid experiment execution registration")
    execution_id = value.get("execution_identity", "")
    if len(execution_id) != 64 or any(c not in "0123456789abcdef" for c in execution_id):
        raise ValueError("invalid registered execution identity")
    return value


def resolve_comparison_identity(root, execution_identity):
    """Validate both immutable snapshots and return the shared comparison identity."""
    root = Path(root)
    original = _read(root/"preflight.json")
    value = registered_execution(root)
    if value is None:
        if original.get("status") != "passed" or original.get("identity") != execution_identity:
            raise ValueError("experiment identity differs; refresh DCAN explicitly or use a new output directory")
        return execution_identity
    if value["execution_identity"] != execution_identity or value["comparison_identity"] != original.get("identity"):
        raise ValueError("registered experiment execution identity differs")
    execution = root/"executions"/execution_identity
    for path, key in ((root/"preflight.json", "original_preflight_sha256"),
                      (root/"source.zip", "original_source_sha256"),
                      (execution/"preflight.json", "preflight_sha256"),
                      (execution/"source.zip", "source_sha256")):
        if _sha(path) != value[key]:
            raise ValueError(f"registered execution evidence differs: {path}")
    previous, current = archive_payload(root/"source.zip"), archive_payload(execution/"source.zip")
    if digest(previous) != value["comparison_identity"] or digest(current) != execution_identity:
        raise ValueError("registered snapshot payload differs")
    if compatibility_manifest() != value["manifest"] or compare_payloads(previous, current, value["manifest"]) != value["changed_sources"]:
        raise ValueError("registered source compatibility differs")
    return value["comparison_identity"]


def arm_execution_identity(root, arm):
    value = registered_execution(root)
    return value["execution_identity"] if value and arm in value["reset_arms"] else None


def execution_job(parts, execution_identity):
    return digest([*parts, {"execution_identity":execution_identity}]) if execution_identity else digest(parts)
