"""Evaluate saved models using their original source, config and fixed data.

Run with the existing training Python from the project root:
    python -m scripts.resume_full_evaluation
    python -m scripts.resume_full_evaluation --arms no_graph no_direct no_residual

The helper prepares an isolated copy in .cache, checks its effective runtime
identity, and evaluates all saved seeds for the selected arms and both tasks in
the original result root. Current hardware is recorded separately in evaluation_runtime/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import zipfile


_ARM_METHODS = {"full": "ppo", "no_graph": "ppo", "no_direct": "ppo", "no_residual": "ppo",
                "rule_ddqn": "rule_ddqn", "dcan": "dcan"}


def _selected_arms(arms):
    selected = ["full"] if arms is None else list(arms)
    if not selected or any(arm not in _ARM_METHODS for arm in selected):
        raise ValueError("Select saved model arms from: " + ", ".join(_ARM_METHODS))
    if len(set(selected)) != len(selected):
        raise ValueError("Selected model arms must be unique.")
    return selected


def _arm_label(arms):
    return "Full" if arms == ["full"] else ", ".join(arms)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, ensure_ascii=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _relative_parts(name):
    name = str(name).replace("\\", "/")
    parts = name.split("/")
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                *(f"LPT{i}" for i in range(1, 10))}
    if any(not part or part in (".", "..") or part.endswith((".", " "))
           or any(ord(char) < 32 or char in '<>:"|?*' for char in part)
           or part.split(".")[0].upper() in reserved for part in parts):
        raise ValueError(f"Expected a safe project-relative path: {name!r}")
    return parts


def relative_file(root, name):
    root = Path(root)
    target = root.joinpath(*_relative_parts(name))
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes the selected directory: {name}")
    return target


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8"))


def copy_checked(project, frozen, name, expected_hash):
    original = relative_file(project, name)
    raw = original.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ValueError(f"Original fixed-data/normalization hash differs: {original}")
    destination = relative_file(frozen, name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(raw)


def _archive_members(archive):
    """Validate the complete member namespace before reading or extracting it."""
    members, folded = {}, {}
    for entry in archive.infolist():
        # ZipInfo truncates filename at NUL; inspect the original name as well.
        original = getattr(entry, "orig_filename", entry.filename)
        if "\x00" in original:
            raise ValueError("Source archive contains a NUL in a member path.")
        name = entry.filename[:-1] if entry.is_dir() else entry.filename
        name = "/".join(_relative_parts(name))
        mode = stat.S_IFMT(entry.external_attr >> 16)
        if mode not in (0, stat.S_IFREG, stat.S_IFDIR):
            raise ValueError(f"Source archive contains a link or special member: {name}")
        if name.casefold() in folded:
            raise ValueError(f"Source archive contains duplicate or aliased member paths: {name}")
        members[name] = entry
        folded[name.casefold()] = entry
    for name in members:
        parts = name.split("/")
        for index in range(1, len(parts)):
            parent = folded.get("/".join(parts[:index]).casefold())
            if parent is not None and not parent.is_dir():
                raise ValueError(f"Source archive has a file/directory path collision: {name}")
    return members


def _reserve_file(reserved, name, expected_hash):
    """Keep reconstruction and fixed-data copies from changing source files."""
    canonical = "/".join(_relative_parts(name))
    key = canonical.casefold()
    previous = reserved.get(key)
    if previous is not None and previous != (canonical, expected_hash):
        raise ValueError(f"Prepared files have conflicting paths or content: {canonical}")
    parts = key.split("/")
    if (any("/".join(parts[:index]) in reserved for index in range(1, len(parts)))
            or any(other.startswith(key + "/") for other in reserved)):
        raise ValueError(f"Prepared files have a file/directory path collision: {canonical}")
    reserved[key] = (canonical, expected_hash)


def prepare(project, output, arms=None):
    arms = _selected_arms(arms)
    project, output = Path(project).resolve(), Path(output).resolve()
    pre = read(output / "preflight.json")
    if pre.get("status") != "passed":
        raise ValueError("The original preflight did not pass.")
    if (output / "execution_reuse.json").exists():
        raise ValueError("Select an independent frozen result root; this root has an execution registration.")
    reserved = {}
    with zipfile.ZipFile(output / "source.zip") as archive:
        members = _archive_members(archive)
        if archive.testzip() is not None:
            raise ValueError("Original source archive CRC check failed.")
        if "experiment_identity.json" not in members:
            raise ValueError("Original source archive is missing experiment_identity.json.")
        payload = json.loads(archive.read(members["experiment_identity.json"]))
        if digest(payload) != pre["identity"] or payload["sources"] != pre["source_hashes"]:
            raise ValueError("Original source archive and preflight identities differ.")
        if payload["configs"] != pre["configs"] or payload["data"] != pre["data_manifests"]:
            raise ValueError("Original source archive and preflight inputs differ.")
        if "suite" in pre and payload["suite"] != pre["suite"]:
            raise ValueError("Original source archive and preflight suites differ.")
        for section in ("sources", "configs", "suite", "data"):
            if not isinstance(payload[section], dict):
                raise ValueError(f"Original identity section must be an object: {section}")
        for name, expected_hash in payload["sources"].items():
            canonical = "/".join(_relative_parts(name))
            entry = members.get(canonical)
            if entry is None or entry.is_dir():
                raise ValueError(f"Original source archive is missing a saved source: {name}")
            if hashlib.sha256(archive.read(entry)).hexdigest() != expected_hash:
                raise ValueError(f"Archived source hash differs: {name}")
            _reserve_file(reserved, name, expected_hash)
        seeds = payload["suite"]["seeds"]
        if (not isinstance(seeds, list) or not seeds
                or any(type(seed) is not int or seed < 1 for seed in seeds)
                or len(set(seeds)) != len(seeds)):
            raise ValueError("Saved seeds must be a nonempty list of unique positive integers.")
        definitions = payload["suite"].get("arms")
        if not isinstance(definitions, dict):
            raise ValueError("Saved suite arms must be an object.")
        for arm in arms:
            definition = definitions.get(arm)
            if (not isinstance(payload["configs"].get(arm), dict) or not isinstance(definition, dict)
                    or definition.get("method") != _ARM_METHODS[arm]):
                raise ValueError(f"Saved identity does not define the selected model arm: {arm}")
            for seed in seeds:
                run = output / "training" / arm / str(seed)
                summary = read(run / "summary.json")
                checkpoint = run / "best_checkpoint.pt"
                if not (summary.get("budget_completed") is True and summary.get("episodes_started") == 2000
                        and summary.get("epochs") == 100 and summary.get("validation_count") == 20
                        and checkpoint.is_file() and checkpoint.stat().st_size > 0):
                    raise ValueError(f"{_arm_label([arm])} seed {seed} has not completed formal training.")
                if ("seed" in summary and summary["seed"] != seed
                        or "method" in summary and summary["method"] != _ARM_METHODS[arm]):
                    raise ValueError(f"{_arm_label([arm])} seed {seed} training summary identifies a different request.")
        cache = relative_file(project, ".cache")
        cache.mkdir(exist_ok=True)
        frozen = Path(tempfile.mkdtemp(prefix="full_evaluation_", dir=cache)).resolve()
        for name, entry in members.items():
            destination = relative_file(frozen, name)
            if entry.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(entry))

    # Some historical snapshots omit JSON presets. Rebuild the effective base
    # and suite from the saved identity, then verify the original constructors
    # reproduce every arm before evaluation. Source hashes remain invariant.
    full = payload["configs"]["full"]
    configs = (("configs/default.json", full), (payload["suite"]["base_config"], full),
               ("configs/experiment_suite_v1.json", payload["suite"]))
    for name, value in configs:
        raw = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
        _reserve_file(reserved, name, hashlib.sha256(raw).hexdigest())
        write_json(relative_file(frozen, name), value)
    scalar = full["objective_scalarizer"]
    name, expected_hash = scalar["normalization_manifest"], scalar["normalization_manifest_sha256"]
    _reserve_file(reserved, name, expected_hash)
    copy_checked(project, frozen, name, expected_hash)
    instances, manifests = full["paths"]["instances_root"], full["paths"]["manifests_root"]
    for split, manifest in payload["data"].items():
        name = f"{manifests}/{split}/manifest.json"
        manifest_path = relative_file(project, name)
        raw = manifest_path.read_bytes()
        if json.loads(raw.decode("utf-8-sig")) != manifest:
            raise ValueError(f"Original {split} dataset manifest differs from frozen inputs.")
        _reserve_file(reserved, name, hashlib.sha256(raw).hexdigest())
        destination = relative_file(frozen, name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
        for item in manifest["files"]:
            for path_key, hash_key in (("path", "sha256"), ("certificate", "certificate_sha256")):
                name = f"{instances}/{item[path_key]}"
                _reserve_file(reserved, name, item[hash_key])
                copy_checked(project, frozen, name, item[hash_key])
    for name, expected_hash in payload["sources"].items():
        if hashlib.sha256(relative_file(frozen, name).read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"Prepared source hash differs: {name}")
    return frozen, pre["identity"], seeds


_VERIFY_RUNTIME = """
import datetime
import json
from pathlib import Path
import sys
import torch
from configs.experiment_suite import load_suite, suite_identity
from scripts.experiment_suite import check_preflight, hardware

expected, output, device, seeds_json, *selection = sys.argv[1:]
arms = json.loads(selection[0]) if selection else ['full']
if (len(selection) > 1 or not isinstance(arms, list) or not arms
        or any(arm not in ('full', 'no_graph', 'no_direct', 'no_residual', 'rule_ddqn', 'dcan') for arm in arms)
        or len(set(arms)) != len(arms)):
    raise ValueError('Invalid saved model arm selection.')
suite = load_suite()
torch.set_num_threads(suite['execution']['torch_threads'])
actual = suite_identity(suite, device)
if actual != expected:
    raise ValueError('Frozen effective identity differs: expected ' + expected + ', got ' + actual)
seeds = json.loads(seeds_json)
if suite['seeds'] != seeds:
    raise ValueError('Frozen suite seeds differ from the prepared training requests.')
if check_preflight(Path(output), actual) not in (None, actual):
    raise ValueError('Original preflight identity differs.')
stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
meta = {'source_identity': actual, 'started_at_utc': stamp, 'hardware': hardware(),
        'python_executable': sys.executable, 'frozen_source_directory': str(Path.cwd()),
        'stage': 'evaluate', 'arms': arms, 'seeds': seeds, 'task': 'both', 'device': device,
        'cache_policy': 'reuse_complete_matching_units'}
folder = Path(output) / 'evaluation_runtime'
if not folder.resolve().is_relative_to(Path(output).resolve()):
    raise ValueError('Runtime record directory escapes the original result root.')
folder.mkdir(exist_ok=True)
with (folder / ('_'.join(arms) + '_' + stamp + '_' + Path.cwd().name + '.json')).open('x', encoding='utf-8') as handle:
    json.dump(meta, handle, indent=2)
label = 'Full' if arms == ['full'] else ', '.join(arms)
print('Original ' + label + ' runtime identity: passed', flush=True)
"""


def _discover_output(project, arms=None):
    arms = _selected_arms(arms)
    project = Path(project)
    choices = set()
    for checkpoint in (project / "result").rglob("best_checkpoint.pt"):
        if checkpoint.parent.parent.name in arms and checkpoint.parent.parent.parent.name == "training":
            choices.add(checkpoint.parents[3].resolve())
    choices = sorted(choices)
    if len(choices) != 1:
        details = []
        for root in choices:
            missing = [name for name in ("preflight.json", "source.zip") if not (root / name).is_file()]
            details.append(f"  {root}" + (f" (missing {', '.join(missing)})" if missing else ""))
        listing = "\n".join(details) if details else (
            "  (none under result/**/training/<selected-arm>/<seed>/best_checkpoint.pt; selected: "
            + ", ".join(arms) + ")")
        raise ValueError(_arm_label(arms) + " result root is missing or ambiguous. Candidates:\n" + listing
                         + "\nSelect the original root with --output-dir PATH (the parent of training/).")
    return choices[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="Project containing original data/ and result/.")
    parser.add_argument("--output-dir", type=Path, help="Original model result root; relative paths use --project-root.")
    parser.add_argument("--arms", nargs="+", choices=tuple(_ARM_METHODS), default=["full"],
                        help="Saved model arms to evaluate (default: full).")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare and check files; skip runtime identity verification and evaluation.")
    args = parser.parse_args()
    project = args.project_root.resolve()
    if not (project / "data").is_dir() or not (project / "result").is_dir():
        parser.error(f"Project root must contain data/ and result/: {project}. Select it with --project-root.")
    try:
        arms = _selected_arms(args.arms)
        output = ((args.output_dir if args.output_dir.is_absolute() else project / args.output_dir).resolve()
                  if args.output_dir is not None else _discover_output(project, arms))
        missing = [name for name in ("preflight.json", "source.zip") if not (output / name).is_file()]
        if missing:
            raise ValueError(f"{_arm_label(arms)} result root {output} is missing {', '.join(missing)}. Select the parent of training/.")
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Original {_arm_label(arms)} output: {output}", flush=True)
    frozen, identity, seeds = prepare(project, output, arms)
    print(f"Frozen source prepared: {frozen}\nExpected identity: {identity}", flush=True)
    if args.prepare_only:
        print("File preparation checks passed. Runtime identity and inference were not executed.", flush=True)
        return
    device = read(frozen / "experiment_identity.json")["configs"]["full"].get("device", "cuda")
    subprocess.run([sys.executable, "-u", "-c", _VERIFY_RUNTIME, identity, str(output), device,
                    json.dumps(seeds), json.dumps(arms)],
                   cwd=frozen, check=True)
    command = [sys.executable, "-u", "-m", "scripts.experiment_suite", "--stage", "evaluate", "--device", device,
               "--arms", *arms, "--seeds", *[str(seed) for seed in seeds], "--task", "both",
               "--output-dir", str(output), "--remote-run"]
    print(f"Launching {_arm_label(arms)} evaluation. Results will be written under the original output/evaluation/.", flush=True)
    subprocess.run(command, cwd=frozen, check=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile, subprocess.CalledProcessError) as exc:
        print(f"STOP: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
