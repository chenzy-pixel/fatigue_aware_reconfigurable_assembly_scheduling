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
    HIERARCHICAL_PRODUCTION_ACTION_SEMANTICS,
    STATE_ONLY_COUNTERFACTUAL_MONOTONE_FLOW_COMMIT_GATE_VERSION,
    STATE_ONLY_HIERARCHICAL_PRODUCTION_ACTION_SEMANTICS,
    assert_network_config_matches_spec,
    infer_checkpoint_network_spec,
)
from environment import DecisionType, Observation, PolicyObservation
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
        if str(row.get("decision_type", "")).strip().lower()
        == "production"
    ]
    worker = [
        row
        for row in rows
        if str(row.get("decision_type", "")).strip().lower() == "worker"
    ]
    commit_logits = [
        float(row.get("commit_set_logit", 0.0))
        for row in production
        if math.isfinite(float(row.get("commit_set_logit", 0.0)))
    ]
    ranker_top_count = sum(
        bool(row.get("ranker_top_selected", False)) for row in ranked
    )
    context_override_count = sum(
        bool(row.get("context_overrode_top", False)) for row in ranked
    )
    preference_override_count = sum(
        bool(row.get("preference_overrode_relative_top", False))
        for row in ranked
    )
    preference_logit_stds = [
        float(row.get("preference_logit_std", 0.0)) for row in ranked
    ]

    def preference_summary(
        selected_rows: Sequence[dict[str, Any]],
    ) -> dict[str, float | int]:
        selected_ranked = [
            row
            for row in selected_rows
            if int(row.get("legal_pair_count", 0))
        ]
        overrides = sum(
            bool(row.get("preference_overrode_relative_top", False))
            for row in selected_ranked
        )
        logit_stds = [
            float(row.get("preference_logit_std", 0.0))
            for row in selected_ranked
        ]
        return {
            "decision_count": len(selected_ranked),
            "override_count": overrides,
            "override_rate": (
                overrides / len(selected_ranked) if selected_ranked else 0.0
            ),
            "mean_logit_std": (
                sum(logit_stds) / len(logit_stds) if logit_stds else 0.0
            ),
        }

    production_preference = preference_summary(production)
    worker_preference = preference_summary(worker)
    production_terminal_count = sum(
        bool(row.get("terminal_legal", False)) for row in production
    )
    worker_terminal_count = sum(
        bool(row.get("terminal_legal", False)) for row in worker
    )
    production_conditional_overrides = sum(
        bool(
            row.get(
                "production_conditional_preference_overrode_relative_top",
                False,
            )
        )
        for row in production
        if int(row.get("legal_pair_count", 0))
    )
    worker_variance_overrides = sum(
        bool(
            row.get(
                "worker_variance_preference_overrode_relative_top", False
            )
        )
        for row in worker
        if int(row.get("legal_pair_count", 0))
    )
    gate_rows = [
        row
        for row in production
        if int(row.get("production_gate_state_count", 0))
    ]
    gate_commit_selected = sum(
        int(row.get("selected_action", -1))
        < int(row.get("action_count", 0) or 0) - 1
        for row in gate_rows
    )
    gate_defer_selected = sum(
        int(row.get("selected_action", -1))
        == int(row.get("action_count", 0) or 0) - 1
        for row in gate_rows
    )
    counterfactual_eligible_count = sum(
        int(row.get("counterfactual_eligible_state", 0) or 0)
        for row in gate_rows
    )
    counterfactual_flip_count = sum(
        int(row.get("counterfactual_high_flow_commit_flip", 0) or 0)
        for row in gate_rows
    )
    counterfactual_state_scales = [
        float(row.get("counterfactual_state_residual_scale", 0.0) or 0.0)
        for row in gate_rows
    ]
    worker_direct_flow_max_abs = max(
        (
            float(row.get("worker_direct_preference_flow_logit_max_abs", 0.0))
            for row in worker
        ),
        default=0.0,
    )
    worker_direct_cost_max_abs = max(
        (
            float(row.get("worker_direct_preference_cost_logit_max_abs", 0.0))
            for row in worker
        ),
        default=0.0,
    )
    worker_direct_variance_max_abs = max(
        (
            float(
                row.get(
                    "worker_direct_preference_variance_logit_max_abs", 0.0
                )
            )
            for row in worker
        ),
        default=0.0,
    )
    return {
        "ranker_top_decision_count": len(ranked),
        "ranker_top_selected_count": ranker_top_count,
        "ranker_top_selection_rate": (
            ranker_top_count / len(ranked) if ranked else 0.0
        ),
        "context_override_count": context_override_count,
        "context_override_rate": (
            context_override_count / len(ranked) if ranked else 0.0
        ),
        "preference_override_count": preference_override_count,
        "preference_override_rate": (
            preference_override_count / len(ranked) if ranked else 0.0
        ),
        "mean_preference_logit_std": (
            sum(preference_logit_stds) / len(preference_logit_stds)
            if preference_logit_stds
            else 0.0
        ),
        "production_ranker_top_decision_count": production_preference[
            "decision_count"
        ],
        "production_preference_override_count": production_preference[
            "override_count"
        ],
        "production_preference_override_rate": production_preference[
            "override_rate"
        ],
        "production_mean_preference_logit_std": production_preference[
            "mean_logit_std"
        ],
        "worker_ranker_top_decision_count": worker_preference[
            "decision_count"
        ],
        "worker_preference_override_count": worker_preference[
            "override_count"
        ],
        "worker_preference_override_rate": worker_preference[
            "override_rate"
        ],
        "worker_mean_preference_logit_std": worker_preference[
            "mean_logit_std"
        ],
        "production_pair_plus_defer_state_count": production_terminal_count,
        "production_decision_state_count": len(production),
        "production_pair_plus_defer_ratio": (
            production_terminal_count / len(production) if production else 0.0
        ),
        "worker_pair_plus_advance_state_count": worker_terminal_count,
        "worker_decision_state_count": len(worker),
        "worker_pair_plus_advance_ratio": (
            worker_terminal_count / len(worker) if worker else 0.0
        ),
        "mean_commit_set_logit": (
            sum(commit_logits) / len(commit_logits) if commit_logits else 0.0
        ),
        "production_conditional_preference_override_count": (
            production_conditional_overrides
        ),
        "production_conditional_preference_override_rate": (
            production_conditional_overrides
            / int(production_preference["decision_count"])
            if production_preference["decision_count"]
            else 0.0
        ),
        "worker_variance_preference_override_count": worker_variance_overrides,
        "worker_variance_preference_override_rate": (
            worker_variance_overrides / int(worker_preference["decision_count"])
            if worker_preference["decision_count"]
            else 0.0
        ),
        "worker_direct_preference_flow_logit_max_abs": (
            worker_direct_flow_max_abs
        ),
        "worker_direct_preference_cost_logit_max_abs": (
            worker_direct_cost_max_abs
        ),
        "worker_direct_preference_variance_logit_max_abs": (
            worker_direct_variance_max_abs
        ),
        "unsafe_worker_preference_selection_count": sum(
            bool(row.get("unsafe_worker_preference_selected", False))
            for row in worker
        ),
        "production_gate_state_count": len(gate_rows),
        "production_gate_commit_selected_count": gate_commit_selected,
        "production_gate_defer_selected_count": gate_defer_selected,
        "mean_production_gate_commit_probability": (
            sum(
                float(row.get("production_gate_commit_probability", 0.0))
                for row in gate_rows
            )
            / len(gate_rows)
            if gate_rows
            else 0.0
        ),
        "mean_production_gate_defer_probability": (
            sum(
                float(row.get("production_gate_defer_probability", 0.0))
                for row in gate_rows
            )
            / len(gate_rows)
            if gate_rows
            else 0.0
        ),
        "mean_production_gate_logit_margin": (
            sum(
                float(row.get("production_gate_logit_margin", 0.0))
                for row in gate_rows
            )
            / len(gate_rows)
            if gate_rows
            else 0.0
        ),
        "mean_production_gate_base_commit_probability": (
            sum(float(row.get("production_gate_base_commit_probability", 0.0)) for row in gate_rows) / len(gate_rows)
            if gate_rows else 0.0
        ),
        "mean_production_gate_base_defer_probability": (
            sum(float(row.get("production_gate_base_defer_probability", 0.0)) for row in gate_rows) / len(gate_rows)
            if gate_rows else 0.0
        ),
        "mean_production_gate_commit_logit_boost": (
            sum(float(row.get("production_gate_commit_logit_boost", 0.0)) for row in gate_rows) / len(gate_rows)
            if gate_rows else 0.0
        ),
        "production_gate_residual_active_count": sum(
            bool(row.get("production_gate_residual_active", False)) for row in gate_rows
        ),
        "production_gate_base_defer_to_final_commit_flip_count": sum(
            bool(row.get("production_gate_base_defer_to_final_commit_flip", False)) for row in gate_rows
        ),
        "counterfactual_eligible_state_count": counterfactual_eligible_count,
        "counterfactual_high_flow_commit_flip_count": counterfactual_flip_count,
        "counterfactual_high_flow_commit_flip_rate": (
            counterfactual_flip_count / counterfactual_eligible_count
            if counterfactual_eligible_count
            else 0.0
        ),
        "mean_counterfactual_state_residual_scale": (
            sum(counterfactual_state_scales) / len(counterfactual_state_scales)
            if counterfactual_state_scales
            else 0.0
        ),
        "max_counterfactual_state_residual_scale": max(
            counterfactual_state_scales, default=0.0
        ),
        "counterfactual_low_flow_identity_violation_count": sum(
            int(row.get("counterfactual_low_flow_identity_violation", 0) or 0)
            for row in gate_rows
        ),
        "counterfactual_monotonicity_violation_count": sum(
            int(row.get("counterfactual_monotonicity_violation", 0) or 0)
            for row in gate_rows
        ),
    }


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
        self.counterfactual_preference_consistency = (
            self._validate_counterfactual_preference_consistency(config)
        )

    @property
    def requires_graph_observation(self) -> bool:
        return bool(self.network.requires_graph_observation)

    def _validate_counterfactual_preference_consistency(
        self,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        raw = config.get("counterfactual_preference_consistency", {})
        if not isinstance(raw, dict):
            raise TypeError("ppo.counterfactual_preference_consistency must be an object")
        if not raw:
            return {"enabled": False}
        enabled = bool(raw.get("enabled", False))
        if not enabled:
            return {"enabled": False}
        expected = {
            "enabled",
            "version",
            "apply_during_phase",
            "low_preference",
            "high_preference",
            "loss_coefficient",
        }
        if set(raw) != expected:
            raise ValueError(
                "E2.6 counterfactual preference consistency has an invalid schema"
            )
        if str(raw["version"]) != "production_gate_cross_zero_v1":
            raise ValueError("E2.6 counterfactual preference consistency has an invalid version")
        if str(raw["apply_during_phase"]) != "quality":
            raise ValueError("E2.6 counterfactual loss must run only during quality")
        if tuple(float(value) for value in raw["low_preference"]) != (0.2, 0.4, 0.4):
            raise ValueError("E2.6 low counterfactual preference must be (0.2, 0.4, 0.4)")
        if tuple(float(value) for value in raw["high_preference"]) != (1.0, 0.0, 0.0):
            raise ValueError("E2.6 high counterfactual preference must be (1, 0, 0)")
        coefficient = float(raw["loss_coefficient"])
        if not math.isfinite(coefficient) or coefficient <= 0.0:
            raise ValueError("E2.6 counterfactual loss coefficient must be finite and positive")
        if getattr(self.network, "production_gate_version", None) != (
            STATE_ONLY_COUNTERFACTUAL_MONOTONE_FLOW_COMMIT_GATE_VERSION
        ):
            raise ValueError("E2.6 counterfactual loss requires the E2.6 production gate")
        return {
            "enabled": True,
            "loss_coefficient": coefficient,
            "apply_during_phase": "quality",
        }

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
            actions = self._deterministic_actions(
                observations,
                action_masks,
                logits,
            )
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

    def _deterministic_actions(
        self,
        observations: Sequence[Observation | PolicyObservation],
        action_masks: Sequence[np.ndarray],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        semantics = getattr(
            self.network,
            "production_action_semantics",
            None,
        )
        if semantics not in {
            HIERARCHICAL_PRODUCTION_ACTION_SEMANTICS,
            STATE_ONLY_HIERARCHICAL_PRODUCTION_ACTION_SEMANTICS,
        }:
            return torch.argmax(logits, dim=-1)
        actions: list[int] = []
        for index, (observation, action_mask) in enumerate(
            zip(observations, action_masks)
        ):
            action_count = int(np.asarray(action_mask).shape[0])
            row = logits[index, :action_count]
            if (
                getattr(observation, "decision_type", None)
                == DecisionType.PRODUCTION
            ):
                actions.append(
                    self._hierarchical_greedy_action(row, action_mask)
                )
            else:
                actions.append(int(torch.argmax(row).detach().cpu()))
        return torch.as_tensor(actions, dtype=torch.long, device=logits.device)

    @staticmethod
    def _hierarchical_greedy_action(
        joint_logits: torch.Tensor,
        action_mask: np.ndarray | torch.Tensor,
    ) -> int:
        """Decode the gate first, then the conditional pair distribution."""

        mask = torch.as_tensor(
            action_mask,
            dtype=torch.bool,
            device=joint_logits.device,
        )
        if joint_logits.ndim != 1 or mask.shape != joint_logits.shape:
            raise ValueError("hierarchical greedy logits and mask must align")
        pair_legal = ~mask[:-1]
        defer_legal = not bool(mask[-1])
        if not bool(pair_legal.any()):
            if not defer_legal:
                raise ValueError("hierarchical greedy action set is empty")
            return int(joint_logits.shape[0] - 1)
        feasible_pair_indices = torch.nonzero(
            pair_legal,
            as_tuple=False,
        ).flatten()
        pair_values = joint_logits[feasible_pair_indices]
        if defer_legal:
            commit_log_mass = torch.logsumexp(pair_values, dim=0)
            defer_value = joint_logits[-1]
            if bool(commit_log_mass < defer_value):
                return int(joint_logits.shape[0] - 1)
        best_pair = feasible_pair_indices[torch.argmax(pair_values)]
        return int(best_pair.detach().cpu())

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

    def _counterfactual_loss(
        self,
        transitions: Sequence[Any],
        *,
        reward_phase: str,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Return E2.6's quality-only cross-zero auxiliary loss.

        ``counterfactual_production_gate_batch`` is deliberately restricted to
        the frozen base gate and the E2.6 state residual head.  Consequently
        this term cannot create gradients in the pair scorer, worker scorer,
        critic, or graph encoder.
        """

        zero = next(self.network.parameters()).new_zeros(())
        disabled_metrics = {
            "counterfactual_eligible_count": 0.0,
            "counterfactual_high_flow_commit_flip_count": 0.0,
            "counterfactual_high_flow_commit_flip_rate": 0.0,
            "counterfactual_state_scale_mean": 0.0,
            "counterfactual_state_scale_max": 0.0,
            "counterfactual_low_flow_identity_violation_count": 0.0,
            "counterfactual_monotonicity_violation_count": 0.0,
        }
        settings = self.counterfactual_preference_consistency
        if not settings["enabled"] or reward_phase != "quality":
            return zero, disabled_metrics
        if not bool(
            getattr(self.network, "production_state_gate_frozen", False)
        ) or not bool(
            getattr(self.network, "production_flow_commit_residual_active", False)
        ):
            raise RuntimeError(
                "E2.6 counterfactual loss requires a frozen active quality gate"
            )
        evaluator = getattr(
            self.network, "counterfactual_production_gate_batch", None
        )
        if evaluator is None:
            raise RuntimeError("E2.6 network lacks counterfactual gate support")
        diagnostics = evaluator(
            [transition.observation for transition in transitions],
            [transition.action_mask for transition in transitions],
            device=self.device,
        )
        eligible = diagnostics["eligible"]
        if bool(eligible.any()):
            loss = torch.relu(-diagnostics["high_margin"][eligible]).square().mean()
        else:
            loss = zero
        production = torch.as_tensor(
            [
                getattr(transition.observation, "decision_type", None)
                == DecisionType.PRODUCTION
                for transition in transitions
            ],
            dtype=torch.bool,
            device=self.device,
        )
        state_scales = diagnostics["state_scale"][production]
        eligible_count = int(eligible.sum().detach().cpu())
        flip_count = int(
            diagnostics["high_flow_flip"].sum().detach().cpu()
        )
        return loss, {
            "counterfactual_eligible_count": float(eligible_count),
            "counterfactual_high_flow_commit_flip_count": float(flip_count),
            "counterfactual_high_flow_commit_flip_rate": (
                float(flip_count / eligible_count) if eligible_count else 0.0
            ),
            "counterfactual_state_scale_mean": (
                float(state_scales.detach().mean().cpu())
                if state_scales.numel()
                else 0.0
            ),
            "counterfactual_state_scale_max": (
                float(state_scales.detach().max().cpu())
                if state_scales.numel()
                else 0.0
            ),
            "counterfactual_low_flow_identity_violation_count": float(
                diagnostics["low_flow_identity_violation"].sum()
                .detach()
                .cpu()
            ),
            "counterfactual_monotonicity_violation_count": float(
                diagnostics["monotonicity_violation"].sum().detach().cpu()
            ),
        }

    def update(
        self,
        buffer: RolloutBuffer,
        *,
        reward_phase: str = "legacy",
    ) -> dict[str, float]:
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
            "counterfactual_loss",
            "counterfactual_eligible_count",
            "counterfactual_high_flow_commit_flip_count",
            "counterfactual_high_flow_commit_flip_rate",
            "counterfactual_state_scale_mean",
            "counterfactual_state_scale_max",
            "counterfactual_low_flow_identity_violation_count",
            "counterfactual_monotonicity_violation_count",
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
                counterfactual_loss, counterfactual_metrics = (
                    self._counterfactual_loss(
                        transitions,
                        reward_phase=reward_phase,
                    )
                )
                loss = (
                    policy_loss
                    + self.config["value_coefficient"] * value_loss
                    - self.config["entropy_coefficient"] * entropy
                    + float(
                        self.counterfactual_preference_consistency.get(
                            "loss_coefficient", 0.0
                        )
                    )
                    * counterfactual_loss
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
                        float(counterfactual_loss.detach().item()),
                        counterfactual_metrics["counterfactual_eligible_count"],
                        counterfactual_metrics[
                            "counterfactual_high_flow_commit_flip_count"
                        ],
                        counterfactual_metrics[
                            "counterfactual_high_flow_commit_flip_rate"
                        ],
                        counterfactual_metrics["counterfactual_state_scale_mean"],
                        counterfactual_metrics["counterfactual_state_scale_max"],
                        counterfactual_metrics[
                            "counterfactual_low_flow_identity_violation_count"
                        ],
                        counterfactual_metrics[
                            "counterfactual_monotonicity_violation_count"
                        ],
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
        saved_metadata = dict(metadata or {})
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
                "ppo_config": self.config,
                "metadata": saved_metadata,
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
        saved_gate = checkpoint_spec.get("production_gate", {})
        if (
            isinstance(saved_gate, dict)
            and hasattr(self.network, "set_production_state_gate_frozen")
        ):
            self.network.set_production_state_gate_frozen(
                bool(saved_gate.get("base_gate_frozen", False))
            )
        if (
            isinstance(saved_gate, dict)
            and hasattr(self.network, "set_production_flow_commit_residual_enabled")
            and "residual_active" in saved_gate
        ):
            self.network.set_production_flow_commit_residual_enabled(
                bool(saved_gate["residual_active"])
            )
        if load_optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        return dict(checkpoint.get("metadata", {}))
