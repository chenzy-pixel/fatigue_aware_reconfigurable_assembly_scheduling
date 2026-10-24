"""Weight fingerprints and the current execution source manifest."""
from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Mapping, Any
import torch
from configs.config import PROJECT_ROOT
from data.dataset import canonical_json_bytes

def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def network_weights_sha256(
    state_dict: Mapping[str, torch.Tensor],
) -> str:
    """Hash tensor names, dtypes, shapes, and exact CPU storage bytes."""
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        descriptor = canonical_json_bytes(
            {
                "name": name,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            }
        )
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(descriptor).to_bytes(8, "big"))
        digest.update(descriptor)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def provenance_with_network_weights(
    provenance: Mapping[str, Any],
    weights_sha256: str,
) -> dict[str, Any]:
    updated = dict(provenance)
    updated["network_weights_sha256"] = str(weights_sha256)
    components = {
        key: value
        for key, value in updated.items()
        if key not in {
            "provenance_schema_version",
            "experiment_fingerprint_sha256",
        }
    }
    updated["experiment_fingerprint_sha256"] = _sha256(
        canonical_json_bytes(components)
    )
    return updated


def source_files(root: str | Path = PROJECT_ROOT) -> list[Path]:
    base = Path(root).resolve()
    selected = set(base.glob("*.py"))
    for package in ("agent", "analysis", "configs", "data", "environment", "scripts", "test"):
        directory = base / package
        if directory.is_dir():
            selected.update(directory.rglob("*.py"))
    result_directory = base / "result"
    if result_directory.is_dir():
        selected.update(result_directory.glob("*.py"))
    return sorted(
        (path for path in selected if path.is_file()),
        key=lambda path: path.relative_to(base).as_posix(),
    )

def execution_source_files(root=PROJECT_ROOT):
    root = Path(root).resolve()
    selected = {p for package in ('agent','environment','data','configs')
                for p in (root/package).rglob('*.py') if p.is_file()}
    for name in ('utils.py','scripts/experiment_suite.py','analysis/pareto_analysis.py',
                 'result/io.py','result/provenance.py','result/training_history.py'):
        path = root/name
        if path.is_file(): selected.add(path)
    return sorted(selected, key=lambda p:p.relative_to(root).as_posix())
