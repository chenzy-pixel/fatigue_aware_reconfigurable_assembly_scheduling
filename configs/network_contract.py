"""Generated identity of the active graph-message computation."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


MESSAGE_IDENTITY_FIELDS = ("message_function", "message_aggregation")


def message_identity(encoder_variant: str = "hetero_gnn") -> dict[str, str]:
    if encoder_variant == "hetero_gnn":
        return {
            "message_function": "attributed_joint_relu_v1",
            "message_aggregation": "total_degree_mean_v1",
        }
    if encoder_variant == "node_mlp_pool":
        return dict.fromkeys(MESSAGE_IDENTITY_FIELDS, "not_applicable")
    raise ValueError("unknown network.encoder_variant")


def validate_message_identity(
    config: Mapping[str, Any], *, require: bool = False,
) -> dict[str, str]:
    """Accept generated identities, never use them to select an implementation."""
    expected = message_identity(str(config.get("encoder_variant", "hetero_gnn")))
    for field, value in expected.items():
        if field not in config:
            if require:
                raise ValueError(f"checkpoint is missing {field}; retraining is required")
        elif config[field] != value:
            raise ValueError(
                f"network {field} is incompatible: expected={value!r}, "
                f"received={config[field]!r}; retraining is required"
            )
    return expected
