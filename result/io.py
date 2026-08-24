from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from configs.config import public_config


def create_run_directory(
    root: str | Path,
    *,
    label: str,
    run_name: str | None = None,
) -> Path:
    root_path = Path(root)
    name = run_name or f"{datetime.now():%Y%m%d_%H%M%S}_{label}"
    output = root_path / name
    output.mkdir(parents=True, exist_ok=False)
    return output


def write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(temporary, output)


def write_config(run_directory: str | Path, config: dict[str, Any]) -> None:
    write_json(Path(run_directory) / "config.json", public_config(config))


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if not rows:
        temporary.write_text("", encoding="utf-8")
        os.replace(temporary, output)
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, output)


def write_evaluation_outputs(
    run_directory: str | Path,
    *,
    config: dict[str, Any],
    metrics: dict[str, Any],
    schedule: list[dict[str, Any]],
    reconfigurations: list[dict[str, Any]],
) -> None:
    run_path = Path(run_directory)
    write_config(run_path, config)
    write_json(run_path / "metrics.json", metrics)
    write_csv(run_path / "schedule.csv", schedule)
    write_csv(run_path / "reconfigurations.csv", reconfigurations)
