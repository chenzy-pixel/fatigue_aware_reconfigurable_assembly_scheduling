from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.distributions import Categorical
from torch.nn import functional as functional

from agent.ppo.buffer import RolloutBuffer
from agent.ppo.network import (
    ActorCriticNetwork,
    assert_network_config_matches_spec,
    infer_checkpoint_network_spec,
)
from environment import Observation, PolicyObservation


def read_checkpoint_network_spec(path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(
        Path(path),
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint root must be a mapping")
    return infer_checkpoint_network_spec(checkpoint)


class PPOAgent:
    def __init__(
        self,
        network: ActorCriticNetwork,
        config: dict[str, Any],
        *,
        device: str = "cpu",
    ):
        self.network = network
        self.config = config
        self.device = torch.device(device)
        self.network.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=config["learning_rate"]
        )

    @property
    def requires_graph_observation(self) -> bool:
        return bool(self.network.requires_graph_observation)

    @torch.no_grad()
    def act(
        self,
        observation: Observation | PolicyObservation,
        action_mask: np.ndarray,
        *,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
    ) -> tuple[int, float, float]:
        actions, log_probabilities, values = self.act_batch(
            [observation],
            [action_mask],
            deterministic=deterministic,
            generator=generator,
        )
        return actions[0], log_probabilities[0], values[0]

    @torch.no_grad()
    def act_batch(
        self,
        observations: Sequence[Observation | PolicyObservation],
        action_masks: Sequence[np.ndarray],
        *,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
    ) -> tuple[list[int], list[float], list[float]]:
        logits, values = self.network.forward_batch(
            observations,
            action_masks,
            device=self.device,
        )
        distribution = Categorical(logits=logits)
        if deterministic:
            actions = torch.argmax(logits, dim=-1)
        elif generator is None:
            actions = distribution.sample()
        else:
            actions = torch.multinomial(
                distribution.probs,
                num_samples=1,
                generator=generator,
            ).squeeze(-1)
        log_probabilities = distribution.log_prob(actions)
        return (
            [int(value) for value in actions.cpu().tolist()],
            [
                float(value)
                for value in log_probabilities.cpu().tolist()
            ],
            [float(value) for value in values.cpu().tolist()],
        )

    @torch.no_grad()
    def value(
        self,
        observation: Observation | PolicyObservation,
        action_mask: np.ndarray,
    ) -> float:
        return self.value_batch([observation], [action_mask])[0]

    @torch.no_grad()
    def value_batch(
        self,
        observations: Sequence[Observation | PolicyObservation],
        action_masks: Sequence[np.ndarray],
    ) -> list[float]:
        _, values = self.network.forward_batch(
            observations,
            action_masks,
            device=self.device,
        )
        return [float(value) for value in values.cpu().tolist()]

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        if not buffer.transitions:
            raise ValueError("cannot update PPO with an empty buffer")
        raw_advantages = torch.as_tensor(
            [transition.advantage for transition in buffer.transitions],
            dtype=torch.float32,
            device=self.device,
        )
        return_values_all = torch.as_tensor(
            [
                transition.return_value
                for transition in buffer.transitions
            ],
            dtype=torch.float32,
            device=self.device,
        )
        value_predictions_before = torch.as_tensor(
            [transition.value for transition in buffer.transitions],
            dtype=torch.float32,
            device=self.device,
        )
        advantages = (raw_advantages - raw_advantages.mean()) / (
            raw_advantages.std(unbiased=False) + 1e-8
        )
        return_variance = return_values_all.var(unbiased=False)
        pre_update_explained_variance = (
            1.0
            - (
                return_values_all - value_predictions_before
            ).var(unbiased=False)
            / return_variance
            if float(return_variance) > 1e-8
            else torch.zeros((), device=self.device)
        )
        epochs = int(self.config["epochs"])
        batch_size = int(self.config["batch_size"])
        metric_names = (
            "policy_loss",
            "value_loss",
            "entropy",
            "loss",
            "approx_kl",
            "clip_fraction",
            "ratio_mean",
            "gradient_norm",
            "gradient_clipped_fraction",
        )
        metrics: list[tuple[float, ...]] = []
        for _ in range(epochs):
            permutation = torch.randperm(
                len(buffer.transitions), device=self.device
            ).tolist()
            for start in range(0, len(permutation), batch_size):
                indices = permutation[start : start + batch_size]
                transitions = [
                    buffer.transitions[index] for index in indices
                ]
                logits, value_prediction = self.network.forward_batch(
                    [
                        transition.observation
                        for transition in transitions
                    ],
                    [
                        transition.action_mask
                        for transition in transitions
                    ],
                    device=self.device,
                )
                distribution = Categorical(logits=logits)
                actions = torch.as_tensor(
                    [
                        transition.action
                        for transition in transitions
                    ],
                    dtype=torch.long,
                    device=self.device,
                )
                new_log_probability = distribution.log_prob(actions)
                entropy = distribution.entropy().mean()
                old_log_probability = torch.as_tensor(
                    [
                        transition.log_probability
                        for transition in transitions
                    ],
                    dtype=torch.float32,
                    device=self.device,
                )
                return_values = torch.as_tensor(
                    [
                        transition.return_value
                        for transition in transitions
                    ],
                    dtype=torch.float32,
                    device=self.device,
                )
                batch_advantages = advantages[
                    torch.as_tensor(indices, dtype=torch.long, device=self.device)
                ]
                log_ratios = new_log_probability - old_log_probability
                ratios = torch.exp(log_ratios)
                approximate_kl = (
                    (ratios - 1.0) - log_ratios
                ).mean()
                clip_fraction = (
                    (ratios - 1.0).abs()
                    > float(self.config["clip_epsilon"])
                ).float().mean()
                surrogate_one = ratios * batch_advantages
                surrogate_two = torch.clamp(
                    ratios,
                    1.0 - self.config["clip_epsilon"],
                    1.0 + self.config["clip_epsilon"],
                ) * batch_advantages
                policy_loss = -torch.minimum(
                    surrogate_one, surrogate_two
                ).mean()
                value_loss = functional.mse_loss(
                    value_prediction, return_values
                )
                loss = (
                    policy_loss
                    + self.config["value_coefficient"] * value_loss
                    - self.config["entropy_coefficient"] * entropy
                )
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("non-finite PPO loss")
                self.optimizer.zero_grad()
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    self.config["max_grad_norm"],
                )
                self.optimizer.step()
                metrics.append(
                    (
                        float(policy_loss.detach().item()),
                        float(value_loss.detach().item()),
                        float(entropy.detach().item()),
                        float(loss.detach().item()),
                        float(approximate_kl.detach().item()),
                        float(clip_fraction.detach().item()),
                        float(ratios.detach().mean().item()),
                        float(gradient_norm.detach().item()),
                        float(
                            (
                                gradient_norm
                                > float(self.config["max_grad_norm"])
                            ).item()
                        ),
                    )
                )
        metric_array = np.asarray(metrics, dtype=np.float64)
        means = np.mean(metric_array, axis=0)
        result = {
            name: float(value)
            for name, value in zip(metric_names, means)
        }
        result["gradient_norm_max"] = float(
            np.max(metric_array[:, metric_names.index("gradient_norm")])
        )
        result["return_mean"] = float(return_values_all.mean().item())
        result["return_std"] = float(
            return_values_all.std(unbiased=False).item()
        )
        result["advantage_mean"] = float(raw_advantages.mean().item())
        result["advantage_std"] = float(
            raw_advantages.std(unbiased=False).item()
        )
        result["value_prediction_mean"] = float(
            value_predictions_before.mean().item()
        )
        result["value_prediction_std"] = float(
            value_predictions_before.std(unbiased=False).item()
        )
        result["pre_update_explained_variance"] = float(
            pre_update_explained_variance.item()
        )
        result["learning_rate"] = float(
            self.optimizer.param_groups[0]["lr"]
        )
        if not all(math.isfinite(value) for value in result.values()):
            raise FloatingPointError("PPO returned non-finite metrics")
        return result

    @property
    def learning_rate(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def set_learning_rate(self, value: float) -> None:
        learning_rate = float(value)
        if not math.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning rate must be finite and positive")
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "network": self.network.state_dict(),
                "network_spec": self.network.network_spec(),
                "optimizer": self.optimizer.state_dict(),
                "ppo_config": self.config,
                "metadata": metadata or {},
            },
            output,
        )

    def load(self, path: str | Path, *, load_optimizer: bool = False) -> dict[str, Any]:
        checkpoint = torch.load(
            Path(path), map_location=self.device, weights_only=False
        )
        checkpoint_spec = infer_checkpoint_network_spec(checkpoint)
        assert_network_config_matches_spec(
            self.network.network_spec(),
            checkpoint_spec,
        )
        self.network.load_state_dict(checkpoint["network"])
        if load_optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        return dict(checkpoint.get("metadata", {}))
