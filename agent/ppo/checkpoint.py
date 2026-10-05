"""Checkpoint migration boundary for the current observation contract."""
from typing import Any, Mapping, Sequence

from environment.observation_schema import OBSERVATION_SCHEMA_VERSION


def migrate_observation_checkpoint(
    checkpoint: Mapping[str, Any], target_spec: Mapping[str, Any], parameter_names: Sequence[str],
    *, migrate_optimizer: bool = False,
) -> dict[str, Any]:
    raise ValueError(
        f"schema {OBSERVATION_SCHEMA_VERSION} requires retraining; "
        "older observation schemas cannot be migrated"
    )
