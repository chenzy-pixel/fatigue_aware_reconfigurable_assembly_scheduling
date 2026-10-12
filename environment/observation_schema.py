"""Ordered global observation contract shared by environment and networks."""

OBSERVATION_SCHEMA_VERSION = 11
LEGACY_OBSERVATION_SCHEMA_VERSION = 10
LEGACY_GLOBAL_FEATURE_NAMES = (
    "current_time_norm", "pending_reconfiguration_ratio", "completed_operation_ratio",
    "production_decision", "worker_matching_deficit_norm", "minimum_worker_alternative_ratio",
    "objective_flow_norm", "objective_cost_norm", "objective_committed_variance_norm",
)
GLOBAL_FEATURE_NAMES = LEGACY_GLOBAL_FEATURE_NAMES[:-1] + ("objective_committed_workload_norm",)


def global_feature_names(version: int) -> tuple[str, ...]:
    if version == 10:
        return LEGACY_GLOBAL_FEATURE_NAMES
    if version == 11:
        return GLOBAL_FEATURE_NAMES
    raise ValueError("unsupported observation schema version")
