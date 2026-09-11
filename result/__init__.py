"""Experiment result persistence and metric comparison."""

from .io import create_run_directory, write_evaluation_outputs
from .metrics import (
    CANONICAL_QUALITY_METRIC,
    CURRENT_RUNTIME_DIAGNOSTIC_FIELDS,
    EVALUATION_SCHEMA_VERSION,
    QUALITY_METRIC_VERSION,
    aggregate_evaluation_rows,
    compare_lexicographic,
    evaluation_quality_metric,
    evaluation_selection_key,
    quality_metric_sha256,
    relative_gap_percent,
    result_schema_version,
    summarize_values,
)
from .provenance import (
    PROVENANCE_SCHEMA_VERSION,
    build_provenance,
    dataset_manifest_snapshot,
    effective_config_snapshot,
    network_weights_sha256,
    provenance_with_network_weights,
    source_state_snapshot,
)
from .terminal_log import capture_terminal_output
from .v8_promotion import (
    BootstrapInterval,
    compare_preference_conditioned_checkpoints,
    paired_instance_block_bootstrap,
)
from .visdom_dashboard import (
    TrainingDashboard,
    create_training_dashboard,
)

__all__ = [
    "EVALUATION_SCHEMA_VERSION",
    "QUALITY_METRIC_VERSION",
    "CANONICAL_QUALITY_METRIC",
    "CURRENT_RUNTIME_DIAGNOSTIC_FIELDS",
    "aggregate_evaluation_rows",
    "compare_lexicographic",
    "create_run_directory",
    "evaluation_selection_key",
    "evaluation_quality_metric",
    "quality_metric_sha256",
    "relative_gap_percent",
    "result_schema_version",
    "summarize_values",
    "TrainingDashboard",
    "create_training_dashboard",
    "write_evaluation_outputs",
    "PROVENANCE_SCHEMA_VERSION",
    "build_provenance",
    "dataset_manifest_snapshot",
    "effective_config_snapshot",
    "network_weights_sha256",
    "provenance_with_network_weights",
    "source_state_snapshot",
    "capture_terminal_output",
    "BootstrapInterval",
    "compare_preference_conditioned_checkpoints",
    "paired_instance_block_bootstrap",
]
