"""Versioned formal sampling aggregation, independent of reward/network identity."""
from collections.abc import Mapping

BEST_OF_THREE = "best_of_3_v1"
PER_SAMPLE = "per_sample_v1"


def sample_aggregation(config: Mapping) -> str:
    mode = config["training"]["formal_evaluation"].get("sample_aggregation")
    if mode not in {BEST_OF_THREE, PER_SAMPLE}:
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
