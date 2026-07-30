"""Experiment result persistence and metric comparison."""

from .io import create_run_directory, write_evaluation_outputs
from .metrics import (
    EVALUATION_SCHEMA_VERSION,
    aggregate_evaluation_rows,
    compare_lexicographic,
    evaluation_selection_key,
    relative_gap_percent,
    summarize_values,
)
from .visdom_dashboard import (
    TrainingDashboard,
    create_training_dashboard,
)

__all__ = [
    "EVALUATION_SCHEMA_VERSION",
    "aggregate_evaluation_rows",
    "compare_lexicographic",
    "create_run_directory",
    "evaluation_selection_key",
    "relative_gap_percent",
    "summarize_values",
    "TrainingDashboard",
    "create_training_dashboard",
    "write_evaluation_outputs",
]
