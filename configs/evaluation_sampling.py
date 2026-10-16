"""Versioned formal sampling aggregation, independent of reward/network identity."""
from collections.abc import Mapping

PER_SAMPLE = "per_sample_v1"
BEST_OF_THREE = "best_of_3_v1"


def sample_aggregation(config: Mapping) -> str:
    # Missing fields in full historical snapshots retain the historical meaning.
    mode = config.get("training", {}).get("formal_evaluation", {}).get("sample_aggregation", PER_SAMPLE)
    if mode not in {PER_SAMPLE, BEST_OF_THREE}:
        raise ValueError(f"unknown formal evaluation sample_aggregation: {mode!r}")
    return mode


def validate_sample_aggregation(config: Mapping) -> None:
    if sample_aggregation(config) == BEST_OF_THREE:
        formal = config["training"]["formal_evaluation"]
        for name in ("validation_repeats", "final_test_repeats"):
            if type(formal.get(name)) is not int or formal[name] != 3:
                raise ValueError(f"best_of_3_v1 requires {name} = 3")
        if formal.get("decode_mode") != "sampled":
            raise ValueError("best_of_3_v1 requires sampled formal evaluation")
