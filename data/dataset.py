from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from data.models import (
    AssemblyInstance,
    instance_to_dict,
    load_instance_yaml,
    parse_instance_dict,
)


PERSISTED_SPLITS = ("validation", "test", "ood", "stress")
ALL_SPLITS = ("train", *PERSISTED_SPLITS)
MANIFEST_REQUIRED_KEYS = {
    "schema_version",
    "split",
    "generator_version",
    "template_instance",
    "template_sha256",
    "seed_start",
    "instance_count",
    "files",
}
MANIFEST_FILE_KEYS = {"seed", "path", "sha256"}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _normalized_json_value(value: Any) -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical JSON does not support non-finite floats")
        rounded = round(value, 10)
        return 0.0 if rounded == 0.0 else rounded
    if isinstance(value, dict):
        return {
            str(key): _normalized_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_normalized_json_value(item) for item in value]
    return value


def canonical_json_bytes(value: Any) -> bytes:
    normalized = _normalized_json_value(value)
    text = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return (text + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def template_sha256(instance: AssemblyInstance) -> str:
    return sha256_bytes(canonical_json_bytes(instance_to_dict(instance)))


def validate_seed_configuration(config: dict[str, Any]) -> None:
    dataset = config.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("config.dataset must be an object")
    splits = dataset.get("splits")
    if not isinstance(splits, dict) or set(splits) != set(ALL_SPLITS):
        raise ValueError(
            "config.dataset.splits must define exactly "
            f"{', '.join(ALL_SPLITS)}"
        )
    ranges: list[tuple[int, int, str]] = []
    for split in ALL_SPLITS:
        settings = splits[split]
        start = int(settings["seed_start"])
        end = int(settings["seed_end"])
        if start < 0 or end <= start:
            raise ValueError(f"invalid seed range for {split}: [{start}, {end})")
        expected_persist = split in PERSISTED_SPLITS
        if bool(settings["persist"]) != expected_persist:
            raise ValueError(f"invalid persistence setting for {split}")
        ranges.append((start, end, split))
    ordered = sorted(ranges)
    for (_, previous_end, previous), (start, _, current) in zip(
        ordered, ordered[1:]
    ):
        if start < previous_end:
            raise ValueError(
                f"seed ranges overlap: {previous} and {current}"
            )
    algorithm_seeds = [int(value) for value in config["algorithm_seeds"]]
    if not algorithm_seeds or len(set(algorithm_seeds)) != len(algorithm_seeds):
        raise ValueError("algorithm_seeds must be non-empty and unique")
    for seed in algorithm_seeds:
        if any(start <= seed < end for start, end, _ in ranges):
            raise ValueError(
                f"algorithm seed {seed} overlaps an instance seed range"
            )
    if int(config["seed"]) not in algorithm_seeds:
        raise ValueError("config.seed must be listed in algorithm_seeds")


def split_seed_range(
    config: dict[str, Any],
    split: str,
) -> tuple[int, int]:
    validate_seed_configuration(config)
    if split not in ALL_SPLITS:
        raise ValueError(f"unknown split {split}")
    settings = config["dataset"]["splits"][split]
    return int(settings["seed_start"]), int(settings["seed_end"])


def validate_instance_seed(
    config: dict[str, Any],
    split: str,
    seed: int,
) -> None:
    start, end = split_seed_range(config, split)
    if not start <= int(seed) < end:
        raise ValueError(
            f"seed {seed} is outside the {split} range [{start}, {end})"
        )


def validate_algorithm_seed(config: dict[str, Any], seed: int) -> int:
    validate_seed_configuration(config)
    value = int(seed)
    allowed = [int(candidate) for candidate in config["algorithm_seeds"]]
    if value not in allowed:
        raise ValueError(
            f"algorithm seed {value} is not one of {allowed}"
        )
    return value


def dataset_profile_counts(
    config: dict[str, Any],
    profile: str,
) -> dict[str, int]:
    profiles = config["dataset"].get("profiles")
    if not isinstance(profiles, dict) or profile not in profiles:
        choices = sorted(profiles) if isinstance(profiles, dict) else []
        raise ValueError(f"unknown dataset profile {profile}; choices={choices}")
    raw = profiles[profile]
    if not isinstance(raw, dict) or set(raw) != set(PERSISTED_SPLITS):
        raise ValueError(
            f"dataset profile {profile} must define exactly "
            f"{', '.join(PERSISTED_SPLITS)}"
        )
    counts = {split: int(raw[split]) for split in PERSISTED_SPLITS}
    if any(value < 1 for value in counts.values()):
        raise ValueError(f"dataset profile {profile} has a non-positive count")
    return counts


def resolve_dataset_count(
    config: dict[str, Any],
    *,
    split: str,
    profile: str,
    count: int | None,
) -> int:
    if split not in PERSISTED_SPLITS:
        raise ValueError(
            f"only {', '.join(PERSISTED_SPLITS)} are persisted"
        )
    profile_count = dataset_profile_counts(config, profile)[split]
    effective_count = profile_count if count is None else int(count)
    if effective_count < 1:
        raise ValueError("count must be positive")
    start, end = split_seed_range(config, split)
    if effective_count > end - start:
        raise ValueError(
            f"count {effective_count} exceeds the {split} seed range"
        )
    return effective_count


@dataclass(frozen=True)
class GeneratedInstanceRecord:
    instance: AssemblyInstance
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "instance": instance_to_dict(self.instance),
        }


def save_generated_record(
    record: GeneratedInstanceRecord,
    path: str | Path,
) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(record.to_dict())
    output.write_bytes(payload)
    return sha256_bytes(payload)


def save_generated_record_atomic(
    record: GeneratedInstanceRecord,
    path: str | Path,
) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(record.to_dict())
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return sha256_bytes(payload)


def load_generated_record(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> GeneratedInstanceRecord:
    source = Path(path)
    payload = source.read_bytes()
    actual_hash = sha256_bytes(payload)
    if expected_sha256 is not None and actual_hash != expected_sha256:
        raise ValueError(
            f"SHA-256 mismatch for {source}: "
            f"expected {expected_sha256}, got {actual_hash}"
        )
    raw = json.loads(payload.decode("utf-8"))
    if set(raw) != {"metadata", "instance"}:
        raise ValueError(f"{source} is not a generated instance record")
    metadata = dict(raw["metadata"])
    is_sparse_ood_like = (
        metadata.get("distribution") in {"ood", "stress"}
        and metadata.get("ood_factor") == "worker_qualification_sparsity"
    )
    instance = parse_instance_dict(
        raw["instance"],
        minimum_qualified_workers=1 if is_sparse_ood_like else 3,
    )
    return GeneratedInstanceRecord(instance=instance, metadata=metadata)


def _validate_sha256(value: Any, label: str) -> str:
    digest = str(value)
    if not SHA256_PATTERN.fullmatch(digest):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _safe_instance_filename(value: Any) -> str:
    filename = str(value)
    path = Path(filename)
    if (
        not filename
        or path.is_absolute()
        or path.name != filename
        or filename in {".", ".."}
        or path.suffix != ".json"
    ):
        raise ValueError(f"unsafe instance path {filename!r}")
    return filename


class InstanceDataset(Sequence[GeneratedInstanceRecord]):
    def __init__(
        self,
        manifest_path: str | Path,
        *,
        instances_root: str | Path,
        expected_split: str | None = None,
        expected_schema_version: str | None = None,
        expected_generator_version: str | None = None,
        expected_template_instance: str | None = None,
        expected_template_sha256: str | None = None,
        expected_seed_range: tuple[int, int] | None = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.instances_root = Path(instances_root)
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            self.manifest: dict[str, Any] = json.load(handle)
        missing = MANIFEST_REQUIRED_KEYS - set(self.manifest)
        if missing:
            raise ValueError(f"manifest is missing fields: {sorted(missing)}")
        split = str(self.manifest["split"])
        if split not in PERSISTED_SPLITS:
            raise ValueError(f"manifest has invalid split {split}")
        if expected_split is not None and split != expected_split:
            raise ValueError(
                f"manifest split mismatch: expected {expected_split}, got {split}"
            )
        self.split = split
        checks = (
            ("schema_version", expected_schema_version),
            ("generator_version", expected_generator_version),
            ("template_instance", expected_template_instance),
            ("template_sha256", expected_template_sha256),
        )
        for field, expected in checks:
            if expected is not None and self.manifest[field] != expected:
                raise ValueError(
                    f"manifest {field} mismatch: "
                    f"expected {expected}, got {self.manifest[field]}"
                )
        self.template_sha256 = _validate_sha256(
            self.manifest["template_sha256"],
            "manifest.template_sha256",
        )
        files = self.manifest["files"]
        if not isinstance(files, list):
            raise ValueError("manifest.files must be a list")
        count = int(self.manifest["instance_count"])
        if count < 1 or count != len(files):
            raise ValueError("manifest instance_count does not match files")
        seed_start = int(self.manifest["seed_start"])
        if expected_seed_range is not None:
            lower, upper = expected_seed_range
            if seed_start < lower or seed_start + count > upper:
                raise ValueError(
                    f"manifest seeds are outside [{lower}, {upper})"
                )
        entries: list[dict[str, Any]] = []
        seen_paths: set[str] = set()
        seen_seeds: set[int] = set()
        for index, raw_entry in enumerate(files):
            if not isinstance(raw_entry, dict):
                raise ValueError("manifest file entries must be objects")
            if set(raw_entry) != MANIFEST_FILE_KEYS:
                raise ValueError(
                    "manifest file entries must contain only "
                    "seed, path, and sha256"
                )
            seed = int(raw_entry["seed"])
            expected_seed = seed_start + index
            if seed != expected_seed:
                raise ValueError(
                    f"manifest seeds must be ordered and contiguous; "
                    f"expected {expected_seed}, got {seed}"
                )
            filename = _safe_instance_filename(raw_entry["path"])
            digest = _validate_sha256(
                raw_entry["sha256"],
                f"manifest.files[{index}].sha256",
            )
            if seed in seen_seeds:
                raise ValueError(f"duplicate seed {seed}")
            if filename in seen_paths:
                raise ValueError(f"duplicate instance path {filename}")
            seen_seeds.add(seed)
            seen_paths.add(filename)
            entries.append(
                {"seed": seed, "path": filename, "sha256": digest}
            )
        self._entries = entries

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, index: int) -> GeneratedInstanceRecord:
        entry = self._entries[index]
        source = self.instances_root / self.split / entry["path"]
        record = load_generated_record(
            source,
            expected_sha256=entry["sha256"],
        )
        expected_metadata = {
            "seed": entry["seed"],
            "split": self.split,
            "generator_version": self.manifest["generator_version"],
            "template_instance": self.manifest["template_instance"],
            "template_sha256": self.template_sha256,
        }
        for field, expected in expected_metadata.items():
            if record.metadata.get(field) != expected:
                raise ValueError(
                    f"instance metadata {field} mismatch for {source}: "
                    f"expected {expected}, got {record.metadata.get(field)}"
                )
        return record

    def __iter__(self) -> Iterator[GeneratedInstanceRecord]:
        for index in range(len(self)):
            yield self[index]


class OnlineInstanceDataset(Sequence[GeneratedInstanceRecord]):
    """Deterministic, stateless training-instance stream.

    Instance seeds depend only on the episode index. Algorithm seeds therefore
    change policy randomness without changing the online training instances.
    """

    def __init__(
        self,
        *,
        config: dict[str, Any],
        template: AssemblyInstance,
        episode_count: int,
    ):
        from data.generate_orders import InstanceGenerator

        validate_seed_configuration(config)
        self.config = config
        self.episode_count = int(episode_count)
        if self.episode_count < 1:
            raise ValueError("episode_count must be positive")
        start, end = split_seed_range(config, "train")
        if self.episode_count > end - start:
            raise ValueError("episode_count exceeds the train seed range")
        self.seed_start = start
        self.generator = InstanceGenerator(
            template,
            config["generator"],
            config=config,
        )

    def __len__(self) -> int:
        return self.episode_count

    def __getitem__(self, index: int) -> GeneratedInstanceRecord:
        if isinstance(index, slice):
            raise TypeError("OnlineInstanceDataset does not support slicing")
        if index < 0:
            index += self.episode_count
        if index < 0 or index >= self.episode_count:
            raise IndexError(index)
        seed = self.seed_start + index
        progress = index / max(1, self.episode_count)
        stage = next(
            (
                value
                for value in self.config["generator"]["curriculum"]
                if progress < float(value["until_fraction"]) + 1e-12
            ),
            None,
        )
        if stage is None:
            raise ValueError("curriculum does not cover the full training run")
        weights = stage["weights"]
        chooser = random.Random(seed)
        pressure_type = chooser.choices(
            list(weights),
            weights=[float(weights[name]) for name in weights],
            k=1,
        )[0]
        return self.generator.generate(
            seed=seed,
            split="train",
            pressure_type=pressure_type,
        )


def _weighted_labels(
    count: int,
    weights: dict[str, float],
) -> list[str]:
    if count < 1:
        raise ValueError("count must be positive")
    if not weights or any(float(value) < 0 for value in weights.values()):
        raise ValueError("weights must be non-empty and non-negative")
    total = sum(float(value) for value in weights.values())
    if total <= 0:
        raise ValueError("weights must have positive total")
    names = list(weights)
    exact = {
        name: count * float(weights[name]) / total
        for name in names
    }
    quotas = {name: int(math.floor(exact[name])) for name in names}
    remaining = count - sum(quotas.values())
    ranked = sorted(
        names,
        key=lambda name: (-(exact[name] - quotas[name]), names.index(name)),
    )
    for name in ranked[:remaining]:
        quotas[name] += 1
    labels: list[str] = []
    while len(labels) < count:
        for name in names:
            if quotas[name] > 0:
                labels.append(name)
                quotas[name] -= 1
    return labels


def _record_matches_build(
    record: GeneratedInstanceRecord,
    *,
    generator: Any,
    split: str,
    seed: int,
    pressure_type: str,
    ood_factor: str | None,
) -> bool:
    expected = {
        "generator_version": generator.version,
        "template_instance": generator.template_instance,
        "template_sha256": generator.template_hash,
        "split": split,
        "seed": seed,
        "pressure_type": pressure_type,
        "ood_factor": ood_factor,
    }
    return all(record.metadata.get(key) == value for key, value in expected.items())


def _publish_split(
    *,
    build_instances: Path,
    target_instances: Path,
    build_manifest: Path,
    target_manifest: Path,
) -> None:
    token = uuid.uuid4().hex
    instance_backup = target_instances.parent / (
        f".{target_instances.name}.backup.{token}"
    )
    manifest_backup = target_manifest.parent / (
        f".{target_manifest.name}.backup.{token}"
    )
    moved_old_instances = False
    moved_old_manifest = False
    published_instances = False
    published_manifest = False
    try:
        if target_instances.exists():
            os.replace(target_instances, instance_backup)
            moved_old_instances = True
        if target_manifest.exists():
            os.replace(target_manifest, manifest_backup)
            moved_old_manifest = True
        os.replace(build_instances, target_instances)
        published_instances = True
        os.replace(build_manifest, target_manifest)
        published_manifest = True
    except BaseException:
        if published_manifest and target_manifest.exists():
            os.replace(target_manifest, build_manifest)
        if published_instances and target_instances.exists():
            os.replace(target_instances, build_instances)
        if moved_old_manifest and manifest_backup.exists():
            os.replace(manifest_backup, target_manifest)
        if moved_old_instances and instance_backup.exists():
            os.replace(instance_backup, target_instances)
        raise
    if moved_old_manifest:
        shutil.rmtree(manifest_backup)
    if moved_old_instances:
        shutil.rmtree(instance_backup)


def build_dataset_split(
    *,
    config: dict[str, Any],
    template: AssemblyInstance,
    split: str,
    profile: str = "dev",
    count: int | None = None,
    instances_root: str | Path | None = None,
    manifests_root: str | Path | None = None,
    overwrite: bool = False,
    resume: bool = True,
) -> Path:
    from configs import project_path
    from data.generate_orders import InstanceGenerator

    effective_count = resolve_dataset_count(
        config,
        split=split,
        profile=profile,
        count=count,
    )
    instance_root = (
        project_path(config["paths"]["instances_root"])
        if instances_root is None
        else Path(instances_root)
    )
    manifest_root = (
        project_path(config["paths"]["manifests_root"])
        if manifests_root is None
        else Path(manifests_root)
    )
    instance_root.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)
    target_instances = instance_root / split
    target_manifest = manifest_root / split
    final_manifest = target_manifest / "manifest.json"
    if (target_instances.exists() or target_manifest.exists()) and not overwrite:
        raise FileExistsError(
            f"{split} dataset already exists; pass --overwrite"
        )

    generator_config = config["generator"]
    generator = InstanceGenerator(template, generator_config, config=config)
    seed_start, _ = split_seed_range(config, split)
    build_key = (
        f"{split}_{generator.version}_{generator.template_hash[:12]}_"
        f"{seed_start}_{effective_count}"
    )
    build_instances = instance_root / ".build" / build_key
    if build_instances.exists() and not resume:
        shutil.rmtree(build_instances)
    build_instances.mkdir(parents=True, exist_ok=True)

    if split in {"ood", "stress"}:
        factors: list[str | None] = _weighted_labels(
            effective_count,
            {
                str(name): 1.0
                for name in generator_config["ood"]["factors"]
            },
        )
        profiles = ["balanced"] * effective_count
    else:
        profiles = _weighted_labels(
            effective_count,
            generator_config["dataset_pressure_weights"],
        )
        factors = [None] * effective_count

    entries: list[dict[str, Any]] = []
    truncated_count = 0
    for index, (pressure_type, ood_factor) in enumerate(
        zip(profiles, factors)
    ):
        seed = seed_start + index
        validate_instance_seed(config, split, seed)
        filename = f"instance_{seed}.json"
        destination = build_instances / filename
        record: GeneratedInstanceRecord | None = None
        if destination.exists() and resume:
            try:
                candidate = load_generated_record(destination)
                if _record_matches_build(
                    candidate,
                    generator=generator,
                    split=split,
                    seed=seed,
                    pressure_type=pressure_type,
                    ood_factor=ood_factor,
                ):
                    record = candidate
                else:
                    destination.unlink()
            except (OSError, UnicodeError, TypeError, ValueError):
                destination.unlink()
        if record is None:
            record = generator.generate(
                seed=seed,
                split=split,
                pressure_type=pressure_type,
                ood_factor=ood_factor,
            )
            digest = save_generated_record_atomic(record, destination)
        else:
            digest = sha256_file(destination)
        truncated_count += int(
            bool(record.metadata["heuristic_metrics"]["heuristic_truncated"])
        )
        entries.append(
            {"seed": seed, "path": filename, "sha256": digest}
        )

    if split == "stress":
        maximum = float(
            generator_config["stress"]["max_truncated_fraction"]
        )
        if truncated_count / effective_count > maximum:
            raise RuntimeError(
                "stress truncated fraction exceeds configured maximum: "
                f"{truncated_count}/{effective_count} > {maximum:.3f}"
            )

    manifest = {
        "schema_version": str(config["dataset"]["schema_version"]),
        "split": split,
        "generator_version": generator.version,
        "template_instance": generator.template_instance,
        "template_sha256": generator.template_hash,
        "seed_start": seed_start,
        "instance_count": effective_count,
        "files": entries,
    }
    build_manifest = manifest_root / ".build" / build_key
    if build_manifest.exists():
        shutil.rmtree(build_manifest)
    build_manifest.mkdir(parents=True, exist_ok=True)
    (build_manifest / "manifest.json").write_bytes(
        canonical_json_bytes(manifest)
    )
    target_instances.parent.mkdir(parents=True, exist_ok=True)
    target_manifest.parent.mkdir(parents=True, exist_ok=True)
    _publish_split(
        build_instances=build_instances,
        target_instances=target_instances,
        build_manifest=build_manifest,
        target_manifest=target_manifest,
    )
    for build_parent in (
        instance_root / ".build",
        manifest_root / ".build",
    ):
        try:
            build_parent.rmdir()
        except OSError:
            pass
    return final_manifest


def build_all_dataset_splits(
    *,
    config: dict[str, Any],
    template: AssemblyInstance,
    profile: str = "dev",
    count: int | None = None,
    instances_root: str | Path | None = None,
    manifests_root: str | Path | None = None,
    overwrite: bool = False,
    resume: bool = True,
) -> dict[str, Path]:
    return {
        split: build_dataset_split(
            config=config,
            template=template,
            split=split,
            profile=profile,
            count=count,
            instances_root=instances_root,
            manifests_root=manifests_root,
            overwrite=overwrite,
            resume=resume,
        )
        for split in PERSISTED_SPLITS
    }


def load_dataset_split(
    config: dict[str, Any],
    split: str,
) -> InstanceDataset:
    from configs import project_path

    if split not in PERSISTED_SPLITS:
        raise ValueError(f"unknown persisted split {split}")
    template = load_instance_yaml(
        project_path(config["paths"]["fixed_instance"])
    )
    manifest = (
        project_path(config["paths"]["manifests_root"])
        / split
        / "manifest.json"
    )
    return InstanceDataset(
        manifest,
        instances_root=project_path(config["paths"]["instances_root"]),
        expected_split=split,
        expected_schema_version=str(config["dataset"]["schema_version"]),
        expected_generator_version=str(config["generator"]["version"]),
        expected_template_instance=str(
            config["dataset"]["template_instance"]
        ),
        expected_template_sha256=template_sha256(template),
        expected_seed_range=split_seed_range(config, split),
    )
