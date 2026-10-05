"""Ordered global observation contract shared by environment and networks."""

OBSERVATION_SCHEMA_VERSION = 10
GLOBAL_FEATURE_NAMES = (
    "current_time_norm", "pending_reconfiguration_ratio", "completed_operation_ratio",
    "production_decision", "worker_matching_deficit_norm", "minimum_worker_alternative_ratio",
    "objective_flow_norm", "objective_cost_norm", "objective_committed_variance_norm",
)
