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
from environment import Observation
from result.provenance import (
    network_weights_sha256,
    provenance_with_network_weights,
)


def summarize_policy_decision_diagnostics(
    rows: Sequence[dict[str, Any]],
) -> dict[str, float | int]:
    ranked = [row for row in rows if int(row.get("legal_pair_count", 0))]
    production = [
        row
        for row in rows
        if str(row.get("decision_type", "")).lower() == "production"
    ]
    worker = [
        row
        for row in rows
        if str(row.get("decision_type", "")).lower() == "worker"
    ]
    ranker_top_count = sum(
        bool(row.get("ranker_top_selected", False)) for row in ranked
    )
    context_override_count = sum(
        bool(row.get("context_overrode_top", False)) for row in ranked
    )
    production_terminal_count = sum(
        bool(row.get("terminal_legal", False)) for row in production
    )
    worker_terminal_count = sum(
        bool(row.get("terminal_legal", False)) for row in worker
    )
    result: dict[str, float | int] = {
        "ranker_top_decision_count": len(ranked),
        "ranker_top_selected_count": ranker_top_count,
        "ranker_top_selection_rate": (
            ranker_top_count / len(ranked) if ranked else 0.0
        ),
        "context_override_count": context_override_count,
        "context_override_rate": (
            context_override_count / len(ranked) if ranked else 0.0
        ),
        "production_pair_plus_wait_state_count": production_terminal_count,
        "production_decision_state_count": len(production),
        "production_pair_plus_wait_ratio": (
            production_terminal_count / len(production) if production else 0.0
        ),
        "worker_pair_plus_wait_state_count": worker_terminal_count,
        "worker_decision_state_count": len(worker),
        "worker_pair_plus_wait_ratio": (
            worker_terminal_count / len(worker) if worker else 0.0
        ),
    }
    component_prefixes = (
        "direct_",
        "context_",
        "expert_",
        "contribution_",
        "residual_base_rms_ratio",
    )
    for phase, phase_rows in (("production", production), ("worker", worker)):
        keys = sorted(
            {
                key
                for row in phase_rows
                for key, value in row.items()
                if isinstance(value, (int, float))
                and any(key.startswith(prefix) for prefix in component_prefixes)
            }
        )
        for key in keys:
            values = [float(row[key]) for row in phase_rows if key in row]
            if values:
                result[f"policy_component_{phase}_{key}"] = float(np.mean(values))
    return result


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
        for module in network.modules():
            if isinstance(module, torch.nn.Dropout) and module.p != 0.0:
                raise ValueError("PPO requires network.dropout = 0")
        self.config = config
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA training was requested, but torch.cuda.is_available() "
                "is False. Install CUDA-enabled PyTorch and a compatible "
                "NVIDIA driver, or explicitly configure device='cpu'."
            )
        if (
            self.device.type == "cuda"
            and self.device.index is not None
            and self.device.index >= torch.cuda.device_count()
        ):
            raise RuntimeError(
                f"CUDA device index {self.device.index} was requested, but "
                f"only {torch.cuda.device_count()} CUDA device(s) are visible."
            )
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
        observation: Observation,
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
        observations: Sequence[Observation],
        action_masks: Sequence[np.ndarray],
        *,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
        generators: Sequence[torch.Generator] | None = None,
    ) -> tuple[list[int], list[float], list[float]]:
        if generators is not None and (
            deterministic or generator is not None or len(generators) != len(observations)
        ):
            raise ValueError("generators must align with a sampled observation batch")
        logits, values = self.network.forward_batch(
            observations,
            action_masks,
            device=self.device,
        )
        distribution = Categorical(logits=logits)
        if generators is not None:
            # Slice padded logits before sampling so each RNG sees its original
            # action width, independent of the other observations in this batch.
            actions = torch.stack([
                torch.multinomial(
                    Categorical(logits=logits[index, :len(mask)]).probs,
                    num_samples=1,
                    generator=generators[index],
                ).squeeze(0)
                for index, mask in enumerate(action_masks)
            ])
        elif deterministic:
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
        sampled = torch.stack(
            (actions.to(dtype=values.dtype), log_probabilities, values),
            dim=-1,
        ).cpu().tolist()
        return (
            [int(row[0]) for row in sampled],
            [float(row[1]) for row in sampled],
            [float(row[2]) for row in sampled],
        )

    @torch.no_grad()
    def value(
        self,
        observation: Observation,
        action_mask: np.ndarray,
    ) -> float:
        return self.value_batch([observation], [action_mask])[0]

    @torch.no_grad()
    def value_batch(
        self,
        observations: Sequence[Observation],
        action_masks: Sequence[np.ndarray],
    ) -> list[float]:
        values = self.network.value_batch(
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
        metrics: list[torch.Tensor] = []
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
                metrics.append(torch.stack((
                    policy_loss.detach(), value_loss.detach(), entropy.detach(),
                    loss.detach(), approximate_kl.detach(), clip_fraction.detach(),
                    ratios.detach().mean(), gradient_norm.detach(),
                    (gradient_norm > float(self.config["max_grad_norm"])).float().detach(),
                )))
        # Read all minibatch metrics and summary statistics in one host transfer.
        summaries = torch.stack((
            return_values_all.mean(), return_values_all.std(unbiased=False),
            raw_advantages.mean(), raw_advantages.std(unbiased=False),
            value_predictions_before.mean(),
            value_predictions_before.std(unbiased=False),
            pre_update_explained_variance,
        ))
        packed = torch.cat((torch.stack(metrics).flatten(), summaries)).cpu().numpy()
        metric_array = packed[:-7].reshape(len(metrics), len(metric_names)).astype(np.float64)
        means = np.mean(metric_array, axis=0)
        result = {
            name: float(value)
            for name, value in zip(metric_names, means)
        }
        result["gradient_norm_max"] = float(
            np.max(metric_array[:, metric_names.index("gradient_norm")])
        )
        for name, value in zip((
            "return_mean", "return_std", "advantage_mean", "advantage_std",
            "value_prediction_mean", "value_prediction_std",
            "pre_update_explained_variance",
        ), packed[-7:]):
            result[name] = float(value)
        result["learning_rate"] = float(
            self.optimizer.param_groups[0]["lr"]
        )
        result.update(self.policy_head_diagnostics())
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

    def policy_head_diagnostics(self) -> dict[str, float]:
        diagnostics = getattr(self.network, "policy_head_diagnostics", None)
        if diagnostics is None:
            return {}
        values = diagnostics()
        if not isinstance(values, dict):
            raise TypeError("policy-head diagnostics must be a mapping")
        result = {str(name): float(value) for name, value in values.items()}
        if not all(math.isfinite(value) for value in result.values()):
            raise FloatingPointError("policy-head diagnostics are non-finite")
        return result

    def consume_policy_decision_diagnostics(self) -> list[dict[str, Any]]:
        consume = getattr(
            self.network, "consume_policy_decision_diagnostics", None
        )
        if consume is None:
            return []
        values = consume()
        if not isinstance(values, list):
            raise TypeError("policy decision diagnostics must be a list")
        return [dict(value) for value in values]

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        saved_metadata = dict(getattr(self, "loaded_checkpoint_metadata", {}) if metadata is None else metadata)
        network_state = self.network.state_dict()
        weights_hash = network_weights_sha256(network_state)
        saved_metadata["network_weights_sha256"] = weights_hash
        if isinstance(saved_metadata.get("provenance"), dict):
            saved_metadata["provenance"] = provenance_with_network_weights(
                saved_metadata["provenance"],
                weights_hash,
            )
        diagnostics = self.policy_head_diagnostics()
        if diagnostics:
            saved_metadata["policy_head_diagnostics"] = diagnostics
        torch.save(
            {
                "network": network_state,
                "network_spec": self.network.network_spec(),
                "optimizer": self.optimizer.state_dict(),
                "optimizer_parameter_names": [name for name, _ in self.network.named_parameters()],
                "ppo_config": self.config,
                "metadata": saved_metadata,
            },
            output,
        )

    def load(self, path: str | Path, *, load_optimizer: bool = False,
             allow_observation_migration: bool = False) -> dict[str, Any]:
        checkpoint = torch.load(
            Path(path), map_location=self.device, weights_only=False
        )
        source_weights_hash = network_weights_sha256(checkpoint["network"])
        saved_weights_hash = checkpoint.get("metadata", {}).get("network_weights_sha256")
        if saved_weights_hash is not None and saved_weights_hash != source_weights_hash:
            raise ValueError("checkpoint metadata network_weights_sha256 does not match the checkpoint network state")
        checkpoint_spec = infer_checkpoint_network_spec(checkpoint)
        assert_network_config_matches_spec(
            self.network.network_spec(),
            checkpoint_spec,
        )
        self.network.load_state_dict(checkpoint["network"])
        if load_optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        metadata = dict(checkpoint.get("metadata", {}))
        self.loaded_checkpoint_metadata = metadata
        return metadata
