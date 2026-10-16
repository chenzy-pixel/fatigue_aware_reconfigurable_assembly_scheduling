"""Choose delivery solutions from complete triples, retaining raw evidence."""
from collections import Counter, defaultdict
import math

from result.metrics import successful_row


def select_best_of_three(rows, *, unique_instance_count, preference_keys):
    groups = defaultdict(list)
    instances = {str(row["instance_id"]) for row in rows}
    if len(instances) != unique_instance_count:
        raise ValueError("best-of-three evaluation has missing or extra instances")
    expected = {(iid, key) for iid in instances for key in preference_keys}
    for row in rows:
        groups[(str(row["instance_id"]), str(row["preference_key"]))].append(row)
    if set(groups) != expected:
        raise ValueError("best-of-three evaluation has missing or extra instance/preference cells")
    chosen, success_counts = [], Counter({str(n): 0 for n in range(4)})
    for key in sorted(groups):
        candidates = groups[key]
        if len(candidates) != 3 or {row.get("sampling_repeat") for row in candidates} != {0, 1, 2}:
            raise ValueError("best-of-three requires three distinct sampling_repeat values 0, 1, 2 per cell")
        for row in candidates:
            q = float(row["preference_quality_score"])
            if not math.isfinite(q) or q < 0:
                raise ValueError("best-of-three requires finite nonnegative native preference quality")
            row["selected_for_aggregation"] = False
        success_counts[str(sum(successful_row(row) for row in candidates))] += 1
        selected = min(candidates, key=lambda row: (
            not successful_row(row), float(row["preference_quality_score"]), int(row["sampling_repeat"])))
        selected["selected_for_aggregation"] = True
        chosen.append(selected)
    return chosen, dict(success_counts)


def write_selected_samples(path, rows):
    """Write full chosen rows, including native action hashes and repeat IDs."""
    from result.io import write_csv
    selected = [row for row in rows if row.get("selected_for_aggregation")]
    if selected:
        write_csv(path, selected)
