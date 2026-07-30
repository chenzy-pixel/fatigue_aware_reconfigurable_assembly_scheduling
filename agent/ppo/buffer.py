from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from environment import (
    HeterogeneousGraphObservation,
    Observation,
    PolicyObservation,
)


@dataclass
class Transition:
    observation: Observation | PolicyObservation
    action_mask: np.ndarray
    action: int
    log_probability: float
    value: float
    reward: float
    done: bool
    advantage: float = 0.0
    return_value: float = 0.0


class RolloutBuffer:
    def __init__(self, *, preserve_graph: bool = False) -> None:
        self.preserve_graph = bool(preserve_graph)
        self.transitions: list[Transition] = []

    def __len__(self) -> int:
        return len(self.transitions)

    def add(
        self,
        observation: Observation | PolicyObservation,
        action_mask: np.ndarray,
        action: int,
        log_probability: float,
        value: float,
        reward: float,
        done: bool,
    ) -> None:
        if self.preserve_graph:
            if not isinstance(observation, HeterogeneousGraphObservation):
                raise TypeError(
                    "graph-preserving buffers require full heterogeneous "
                    "graph observations"
                )
            stored_observation = observation.copy()
        else:
            stored_observation = PolicyObservation.from_observation(
                observation
            )
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
        if not self.transitions:
            self.preserve_graph = other.preserve_graph
        elif self.preserve_graph != other.preserve_graph:
            raise ValueError(
                "cannot merge compact and graph-preserving rollout buffers"
            )
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
