from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from environment.observation_schema import OBSERVATION_SCHEMA_VERSION, global_feature_names
from environment import (
    ASSEMBLY_EDGE_TYPES,
    HeterogeneousGraphObservation,
    Observation,
)


def validate_graph_schema(observation: Observation, schema_version: int | None = None) -> int | None:
    """Check the current graph contract before storing a rollout sample."""
    if not isinstance(observation, HeterogeneousGraphObservation) or not observation.relations:
        return
    if schema_version is None:
        schema_version = next((version for version in (10, OBSERVATION_SCHEMA_VERSION)
                               if tuple(observation.global_feature_names) == global_feature_names(version)), None)
    expected = () if schema_version is None else global_feature_names(schema_version)
    if (schema_version is None or observation.global_features.shape != (len(expected),)
            or tuple(observation.global_feature_names) != expected):
        raise ValueError(
            f"schema-{OBSERVATION_SCHEMA_VERSION} global feature names/order/dimensions are incompatible"
        )
    if not np.all(np.isfinite(observation.global_features)):
        raise ValueError("global features must be finite")
    if set(observation.relations) != set(ASSEMBLY_EDGE_TYPES):
        raise ValueError(f"schema-{OBSERVATION_SCHEMA_VERSION} graph relations are incompatible")
    return schema_version


@dataclass
class Transition:
    observation: Observation
    action_mask: np.ndarray
    action: int
    log_probability: float
    value: float
    reward: float
    done: bool
    advantage: float = 0.0
    return_value: float = 0.0


class RolloutBuffer:
    def __init__(self, *, preserve_graph: bool = True, observation_schema_version: int | None = None) -> None:
        if not preserve_graph:
            raise ValueError("rollout buffers require graph observations")
        self.preserve_graph = True
        self.observation_schema_version = observation_schema_version
        self.transitions: list[Transition] = []

    def __len__(self) -> int:
        return len(self.transitions)

    def add(
        self,
        observation: Observation,
        action_mask: np.ndarray,
        action: int,
        log_probability: float,
        value: float,
        reward: float,
        done: bool,
    ) -> None:
        stored_observation = observation.copy()
        selected_schema = validate_graph_schema(stored_observation, self.observation_schema_version)
        if selected_schema is not None:
            self.observation_schema_version = selected_schema
        self.transitions.append(
            Transition(
                observation=stored_observation,
                action_mask=action_mask.copy(),
                action=int(action),
                log_probability=float(log_probability),
                value=float(value),
                reward=float(reward),
                done=bool(done),
            )
        )

    def extend(self, other: "RolloutBuffer") -> None:
        if (self.observation_schema_version is not None and other.observation_schema_version is not None
                and self.observation_schema_version != other.observation_schema_version):
            raise ValueError("rollout buffers cannot mix observation schemas")
        if self.observation_schema_version is None:
            self.observation_schema_version = other.observation_schema_version
        self.transitions.extend(other.transitions)

    def compute_gae(
        self,
        *,
        last_value: float,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        gae = 0.0
        next_value = float(last_value)
        for transition in reversed(self.transitions):
            nonterminal = 0.0 if transition.done else 1.0
            delta = (
                transition.reward
                + gamma * next_value * nonterminal
                - transition.value
            )
            gae = delta + gamma * gae_lambda * nonterminal * gae
            transition.advantage = gae
            transition.return_value = gae + transition.value
            next_value = transition.value

    def clear(self) -> None:
        self.transitions.clear()
