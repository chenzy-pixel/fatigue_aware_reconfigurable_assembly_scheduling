"""Build, verify, or restore the three-seed result archives (Python standard library)."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import zipfile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MANIFEST = HERE / "publication_manifest.json"
WEIGHTS = {".pt", ".pth", ".ckpt", ".safetensors"}
STALE_FIGURES = {f"04_mix_heuristic_comparison.{ext}" for ext in ("png", "pdf", "svg")}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def record(path):
    return {"path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size, "sha256": sha(path)}


def archive(name, files):
    target = HERE / "archives" / name
    target.parent.mkdir(exist_ok=True)
    files = sorted(set(files))
    members = []
    temporary = target.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6, allowZip64=True) as bundle:
        for path in files:
            if path.suffix.lower() in WEIGHTS:
                raise ValueError(f"Model weights cannot be published: {path}")
            item = record(path)
            bundle.write(path, item["path"])
            if sha(path) != item["sha256"]:
                raise RuntimeError(f"Input changed during packaging: {path}")
            members.append(item)
    os.replace(temporary, target)
    if target.stat().st_size >= 95 * 1024**2:
        raise ValueError(f"Archive exceeds the publication size budget: {target}")
    print(f"Built {name}: {len(members)} files, {target.stat().st_size / 1024**2:.2f} MiB",
          flush=True)
    return {**record(target), "members": members,
            "uncompressed_bytes": sum(item["bytes"] for item in members)}


def analysis_files():
    return sorted(path for path in HERE.rglob("*") if path.is_file()
                  and "archives" not in path.relative_to(HERE).parts
                  and "__pycache__" not in path.parts
                  and path.name not in STALE_FIGURES
                  and path.name not in {MANIFEST.name, "publication_verification.json"}
                  and not path.name.endswith(".tmp"))


def build():
    audit = json.loads((HERE / "audit.json").read_text(encoding="utf-8"))
    assert audit["all_checks_passed"] and len(audit["checks"]) == 1235
    for relative, expected in audit["input_sha256"].items():
        if sha(ROOT / Path(relative.replace("\\", "/"))) != expected:
            raise ValueError(f"Audited input changed: {relative}")
    print("Reverified all 45 audited input hashes.", flush=True)
    jobs = {}
    model_identity = []
    for run in audit["runs"]:
        folder = ROOT / "result" / "runs" / run["run"]
        files = [path for path in folder.rglob("*") if path.is_file()]
        jobs[f"seed{run['seed']}_logs.zip"] = [p for p in files if p.suffix.lower() not in WEIGHTS]
        model_identity.extend(record(path) for path in files if path.suffix.lower() in WEIGHTS)
        assert len(jobs[f"seed{run['seed']}_logs.zip"]) == 18

    # The seed 11/23 backfill and the frozen search requests are inputs of the unified tables.
    prior = ROOT / "result/analysis/mixed_instances_review_20261021"
    evidence = []
    for seed in (11, 23):
        evidence.extend(path for path in (prior / f"seed{seed}").rglob("*") if path.is_file())
    for filename in ("plan.json", "raw_samples.csv", "selected_samples.csv", "final_verification.json"):
        evidence.append(prior / filename)
    search = ROOT / "result/analysis/heuristics_seed11_20261019"
    for filename in ("identity.json", "execution_settings.json", "preflight.json",
                     "search_selection.json", "subsets.json"):
        evidence.append(search / filename)
    search_files = list((search / "tune_search/16/nsga2_ls").glob("*.json"))
    search_files += list((search / "tune_search/32/moead").glob("*.json"))
    assert len(search_files) == 56
    evidence.extend(search_files)
    assert all(path.is_file() for path in evidence)
    jobs["heuristic_evidence.zip"] = evidence

    bundles = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(archive, name, files) for name, files in jobs.items()]
        for future in as_completed(futures):
            bundles.append(future.result())
    manifest = {
        "schema": "hgnn_ppo_three_seed_results_v1",
        "analysis_date": "2026-10-22", "timezone": "UTC+8",
        "method": "HGNN-PPO", "training_seeds": [11, 23, 37],
        "training_base_commit": audit["source_git"],
        "source_state_sha256": audit["source_state_sha256"],
        "publication_base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "historical_audit_checks": len(audit["checks"]),
        "audited_input_hashes_reverified": len(audit["input_sha256"]),
        "model_identity_only": model_identity,
        "archives": sorted(bundles, key=lambda item: item["path"]),
        "analysis_files": [record(path) for path in analysis_files()],
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return verify()


def safe_path(relative):
    item = PurePosixPath(relative)
    if item.is_absolute() or ".." in item.parts or not item.parts or item.parts[0] != "result":
        raise ValueError(f"Invalid archive path: {relative}")
    target = ROOT.joinpath(*item.parts)
    target.resolve().relative_to((ROOT / "result").resolve())
    return target


def verify():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    member_count = 0
    members_by_path = {}
    for item in manifest["analysis_files"]:
        path = safe_path(item["path"])
        assert path.stat().st_size == item["bytes"] and sha(path) == item["sha256"], item["path"]
    for item in manifest["archives"]:
        path = safe_path(item["path"])
        assert path.stat().st_size == item["bytes"] and sha(path) == item["sha256"], item["path"]
        expected = {entry["path"]: entry for entry in item["members"]}
        with zipfile.ZipFile(path) as bundle:
            assert len(bundle.namelist()) == len(expected)
            assert set(bundle.namelist()) == set(expected)
            for entry in bundle.infolist():
                safe_path(entry.filename)
                assert Path(entry.filename).suffix.lower() not in WEIGHTS
                assert entry.file_size == expected[entry.filename]["bytes"]
                with bundle.open(entry) as stream:
                    assert hashlib.file_digest(stream, "sha256").hexdigest() == expected[entry.filename]["sha256"]
                members_by_path[entry.filename] = expected[entry.filename]["sha256"]
                member_count += 1
    audit = json.loads((HERE / "audit.json").read_text(encoding="utf-8"))
    for relative, expected in audit["input_sha256"].items():
        relative = relative.replace("\\", "/")
        if Path(relative).suffix.lower() not in WEIGHTS:
            assert members_by_path[relative] == expected
    result = {"passed": True, "archive_count": len(manifest["archives"]),
              "archive_members_verified": member_count,
              "analysis_files_verified": len(manifest["analysis_files"]),
              "published_model_weights": 0,
              "historical_audit_checks": manifest["historical_audit_checks"],
              "new_training": 0, "new_evaluation_rollouts": 0}
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def restore():
    verify()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    restored = 0
    for item in manifest["archives"]:
        with zipfile.ZipFile(safe_path(item["path"])) as bundle:
            for entry in item["members"]:
                target = safe_path(entry["path"])
                if target.exists():
                    if sha(target) != entry["sha256"]:
                        raise FileExistsError(f"Existing file has different bytes: {target}")
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(entry["path"]) as source, target.open("xb") as destination:
                    while chunk := source.read(1024 * 1024):
                        destination.write(chunk)
                assert sha(target) == entry["sha256"]
                restored += 1
    print(json.dumps({"restored_files": restored}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build", action="store_true", help="Package the original local results.")
    mode.add_argument("--verify", action="store_true", help="Verify the committed evidence without models or GPU.")
    mode.add_argument("--restore", action="store_true", help="Restore archived logs; preserve matching existing files.")
    args = parser.parse_args()
    if args.build:
        outcome = build()
        (HERE / "publication_verification.json").write_text(
            json.dumps(outcome, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    elif args.verify:
        verify()
    else:
        restore()
