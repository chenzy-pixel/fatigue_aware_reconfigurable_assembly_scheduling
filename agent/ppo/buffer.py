from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from environment.observation_schema import GLOBAL_FEATURE_NAMES, OBSERVATION_SCHEMA_VERSION
from environment import (
    ASSEMBLY_EDGE_TYPES,
    HeterogeneousGraphObservation,
    Observation,
)


def validate_graph_schema(observation: Observation) -> None:
    """Check the current graph contract before storing a rollout sample."""
    if not isinstance(observation, HeterogeneousGraphObservation) or not observation.relations:
        return
    if (observation.global_features.shape != (len(GLOBAL_FEATURE_NAMES),)
            or tuple(observation.global_feature_names) != GLOBAL_FEATURE_NAMES):
        raise ValueError(
            f"schema-{OBSERVATION_SCHEMA_VERSION} global feature names/order/dimensions are incompatible"
        )
    if not np.all(np.isfinite(observation.global_features)):
        raise ValueError("global features must be finite")
    if set(observation.relations) != set(ASSEMBLY_EDGE_TYPES):
        raise ValueError(f"schema-{OBSERVATION_SCHEMA_VERSION} graph relations are incompatible")


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
    def __init__(self, *, preserve_graph: bool = True) -> None:
        if not preserve_graph:
            raise ValueError("rollout buffers require graph observations")
        self.preserve_graph = True
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
        validate_graph_schema(stored_observation)
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
