"""CSV and JSON result writers."""
import csv
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

def write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(temporary, output)


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
