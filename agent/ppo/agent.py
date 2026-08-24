from __future__ import annotations

import math
import hashlib
from copy import deepcopy
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.distributions import Categorical, kl_divergence
from torch.nn import functional as functional

from agent.ppo.buffer import RolloutBuffer
from agent.ppo.network import (
    ActorCriticNetwork,
    E1_CENTERED_HIERARCHICAL_PRODUCTION_ACTION_SEMANTICS,
    E1_CENTERED_THREE_OBJECTIVE_GATE_VERSION,
    HIERARCHICAL_PRODUCTION_ACTION_SEMANTICS,
    STATE_ONLY_COUNTERFACTUAL_MONOTONE_FLOW_COMMIT_GATE_VERSION,
    STATE_ONLY_HIERARCHICAL_PRODUCTION_ACTION_SEMANTICS,
    assert_network_config_matches_spec,
    build_actor_critic,
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
    centered_dual_legal_count = sum(
        int(row.get("centered_gate_dual_legal_state", 0) or 0)
        for row in gate_rows
    )
    centered_flow_cost_flip_count = sum(
        int(row.get("centered_gate_flow_cost_flip", 0) or 0)
        for row in gate_rows
    )
    centered_flow_variance_flip_count = sum(
        int(row.get("centered_gate_flow_variance_flip", 0) or 0)
        for row in gate_rows
    )
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
        "centered_gate_dual_legal_state_count": centered_dual_legal_count,
        "centered_gate_flow_cost_flip_count": centered_flow_cost_flip_count,
        "centered_gate_flow_variance_flip_count": (
            centered_flow_variance_flip_count
        ),
        "centered_gate_extreme_flip_rate": (
            max(centered_flow_cost_flip_count, centered_flow_variance_flip_count)
            / centered_dual_legal_count
            if centered_dual_legal_count
            else 0.0
        ),
        "centered_gate_monotonicity_violation_count": sum(
            int(row.get("centered_gate_monotonicity_violation", 0) or 0)
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
        self.canonical_teacher_kl = self._validate_canonical_teacher_kl(config)
        self.canonical_teacher: ActorCriticNetwork | None = None
        self.warm_start_report: dict[str, Any] | None = None
        self.centered_state_pools: dict[
            str, list[tuple[Observation | PolicyObservation, np.ndarray]]
        ] = {
            "gate": [],
            "production_pair": [],
            "worker_variance": [],
        }
        self.centered_state_pool_reports: dict[str, dict[str, Any]] = {}
        self._centered_pool_gradient_preflight_complete = {
            name: False for name in self.centered_state_pools
        }
        # Compatibility aliases retained for the E2.7 gate report and callers.
        self.safe_dual_legal_state_pool = self.centered_state_pools["gate"]
        self.safe_dual_legal_state_pool_report: dict[str, Any] | None = None
        self._safe_pool_gradient_preflight_complete = False

    def set_safe_dual_legal_state_pool(
        self,
        states: Sequence[
            tuple[Observation | PolicyObservation, np.ndarray]
        ],
        *,
        provenance: dict[str, Any],
    ) -> dict[str, Any]:
        return self.set_centered_state_pool("gate", states, provenance=provenance)

    def set_centered_state_pool(
        self,
        name: str,
        states: Sequence[tuple[Observation | PolicyObservation, np.ndarray]],
        *,
        provenance: dict[str, Any],
    ) -> dict[str, Any]:
        """Install one immutable E2.7 counterfactual state pool."""

        if name not in self.centered_state_pools:
            raise ValueError(f"unknown centered E2.7 state-pool kind {name!r}")
        objectives = self.counterfactual_preference_consistency.get("objectives", {})
        settings = objectives.get(name, {})
        minimum = int(settings.get("minimum_eligible_states", 0))
        if minimum < 1:
            raise RuntimeError("centered state pools require E2.7 v3 training")
        normalized = [
            (observation.copy(), np.asarray(mask, dtype=bool).copy())
            for observation, mask in states
        ]
        if len(normalized) < minimum:
            raise RuntimeError(
                f"E2.7 {name} state pool is too small: "
                f"eligible={len(normalized)}, required={minimum}"
            )
        for observation, mask in normalized:
            phase = getattr(observation, "decision_type", None)
            pair_count = int(np.count_nonzero(~mask[:-1])) if mask.ndim == 1 else 0
            if mask.ndim != 1 or mask.size < 2:
                raise ValueError("centered state pool contains an invalid action mask")
            if name == "gate" and (
                phase != DecisionType.PRODUCTION
                or pair_count < 1
                or bool(mask[-1])
            ):
                raise ValueError("gate state pool contains a non-dual-legal state")
            if name == "production_pair" and (
                phase != DecisionType.PRODUCTION or pair_count < 2
            ):
                raise ValueError("production-pair state pool contains an invalid state")
            if name == "worker_variance" and (
                phase != DecisionType.WORKER or pair_count < 2
            ):
                raise ValueError("worker-variance state pool contains an invalid state")
        if name in {"production_pair", "worker_variance"}:
            observations = [observation for observation, _ in normalized]
            masks = [mask for _, mask in normalized]
            with torch.no_grad():
                diagnostics = (
                    self.network.centered_production_pair_counterfactual_batch(
                        observations,
                        masks,
                        device=self.device,
                    )
                    if name == "production_pair"
                    else self.network.centered_worker_variance_counterfactual_batch(
                        observations,
                        masks,
                        device=self.device,
                    )
                )
            if int(diagnostics["eligible"].sum().detach().cpu()) < minimum:
                raise ValueError(
                    f"{name} state pool lacks distinct objective candidates"
                )
        digest = hashlib.sha256()

        def update_array(name: str, value: Any) -> None:
            array = np.ascontiguousarray(np.asarray(value))
            digest.update(name.encode("utf-8"))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(str(tuple(array.shape)).encode("ascii"))
            digest.update(array.tobytes())

        for index, (observation, mask) in enumerate(normalized):
            digest.update(f"{name}:state:{index}".encode("ascii"))
            digest.update(str(observation.decision_type.value).encode("utf-8"))
            update_array("mask", mask)
            update_array("global", observation.global_features)
            update_array("action_set", observation.action_set_features)
            update_array("preference", observation.preference)
            for node_type in sorted(observation.node_features):
                update_array(
                    f"node:{node_type}", observation.node_features[node_type]
                )
            for edge_type in sorted(observation.relations, key=str):
                edge = observation.relations[edge_type]
                update_array(f"edge_index:{edge_type}", edge.edge_index)
                update_array(f"edge_features:{edge_type}", edge.edge_features)
        self.centered_state_pools[name] = normalized
        self._centered_pool_gradient_preflight_complete[name] = False
        report = {
            "version": "fixed_e2_7_centered_counterfactual_pool_v2",
            "pool_kind": name,
            "state_count": len(normalized),
            "minimum_required_state_count": minimum,
            "fixed_sampling_count_per_auxiliary_update": minimum,
            "state_pool_sha256": digest.hexdigest(),
            "provenance": dict(provenance),
        }
        self.centered_state_pool_reports[name] = report
        if name == "gate":
            self.safe_dual_legal_state_pool = normalized
            self.safe_dual_legal_state_pool_report = report
            self._safe_pool_gradient_preflight_complete = False
        return dict(report)

    @staticmethod
    def _validate_canonical_teacher_kl(
        config: dict[str, Any],
    ) -> dict[str, Any]:
        raw = config.get("canonical_teacher_kl", {})
        if not isinstance(raw, dict):
            raise TypeError("ppo.canonical_teacher_kl must be an object")
        if not raw or not bool(raw.get("enabled", False)):
            return {"enabled": False, "coefficient": 0.0}
        expected = {"enabled", "version", "coefficient", "canonical_preference"}
        if set(raw) != expected:
            raise ValueError("canonical teacher KL has an invalid schema")
        if str(raw["version"]) != "e1_canonical_policy_kl_v1":
            raise ValueError("unsupported canonical teacher KL version")
        canonical = tuple(float(value) for value in raw["canonical_preference"])
        if canonical != (0.5, 0.3, 0.2):
            raise ValueError("canonical teacher preference must be (0.5, 0.3, 0.2)")
        coefficient = float(raw["coefficient"])
        if not math.isfinite(coefficient) or coefficient <= 0.0:
            raise ValueError("canonical teacher KL coefficient must be positive")
        return {
            "enabled": True,
            "coefficient": coefficient,
            "canonical_preference": canonical,
        }

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
        version = str(raw.get("version", ""))
        if version == "centered_gate_pair_worker_v3":
            expected = {
                "enabled",
                "version",
                "apply_during_phase",
                "gate",
                "production_pair",
                "worker_variance",
            }
            if set(raw) != expected:
                raise ValueError(
                    "E2.7 counterfactual preference consistency has an invalid schema"
                )
            if str(raw["apply_during_phase"]) != "all":
                raise ValueError("E2.7 counterfactual loss must run in all stages")
            objectives: dict[str, dict[str, float | int]] = {}
            for name in ("gate", "production_pair", "worker_variance"):
                objective = raw[name]
                if not isinstance(objective, dict) or set(objective) != {
                    "minimum_margin_gap",
                    "minimum_eligible_states",
                    "loss_coefficient",
                }:
                    raise ValueError(
                        f"E2.7 {name} counterfactual objective has an invalid schema"
                    )
                gap = float(objective["minimum_margin_gap"])
                minimum = int(objective["minimum_eligible_states"])
                coefficient = float(objective["loss_coefficient"])
                if not math.isfinite(gap) or gap <= 0.0:
                    raise ValueError(
                        f"E2.7 {name} minimum margin gap must be positive"
                    )
                if minimum < 1:
                    raise ValueError(
                        f"E2.7 {name} minimum eligible states must be positive"
                    )
                if not math.isfinite(coefficient) or coefficient <= 0.0:
                    raise ValueError(
                        f"E2.7 {name} counterfactual coefficient must be positive"
                    )
                objectives[name] = {
                    "minimum_margin_gap": gap,
                    "minimum_eligible_states": minimum,
                    "loss_coefficient": coefficient,
                }
            if getattr(self.network, "production_gate_version", None) != (
                E1_CENTERED_THREE_OBJECTIVE_GATE_VERSION
            ):
                raise ValueError("E2.7 loss requires the centered production gate")
            return {
                "enabled": True,
                "version": version,
                "objectives": objectives,
                "apply_during_phase": "all",
            }
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
        if version != "production_gate_cross_zero_v1":
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
            "version": version,
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
            E1_CENTERED_HIERARCHICAL_PRODUCTION_ACTION_SEMANTICS,
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

    def _centered_pool_objective(
        self,
        name: str,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Evaluate one E2.7 v3 fixed-pool objective and its diagnostics."""

        settings = self.counterfactual_preference_consistency["objectives"][name]
        pool = self.centered_state_pools[name]
        minimum = int(settings["minimum_eligible_states"])
        if len(pool) < minimum:
            raise RuntimeError(
                f"E2.7 {name} state pool is too small: "
                f"eligible={len(pool)}, required={minimum}"
            )
        observations = [observation for observation, _ in pool[:minimum]]
        masks = [mask for _, mask in pool[:minimum]]
        gap = float(settings["minimum_margin_gap"])
        zero = next(self.network.parameters()).new_zeros(())
        if name == "gate":
            diagnostics = self.network.centered_gate_counterfactual_batch(
                observations, masks, device=self.device
            )
            eligible = diagnostics["eligible"]
            if int(eligible.sum().detach().cpu()) < minimum:
                raise RuntimeError("E2.7 gate state pool lost eligible states")
            flow = diagnostics["flow_margin"][eligible]
            cost = diagnostics["cost_margin"][eligible]
            variance = diagnostics["variance_margin"][eligible]
            loss = 0.5 * (
                torch.relu(gap - (flow - cost)).square().mean()
                + torch.relu(gap - (flow - variance)).square().mean()
            )
            flow_cost_flips = int(diagnostics["flow_cost_flip"].sum().detach().cpu())
            flow_variance_flips = int(
                diagnostics["flow_variance_flip"].sum().detach().cpu()
            )
            eligible_count = int(eligible.sum().detach().cpu())
            return loss, {
                "counterfactual_eligible_count": float(eligible_count),
                "counterfactual_high_flow_commit_flip_count": float(
                    max(flow_cost_flips, flow_variance_flips)
                ),
                "counterfactual_high_flow_commit_flip_rate": (
                    max(flow_cost_flips, flow_variance_flips) / eligible_count
                ),
                "counterfactual_monotonicity_violation_count": float(
                    diagnostics["monotonicity_violation"].sum().detach().cpu()
                ),
                "counterfactual_flow_cost_flip_count": float(flow_cost_flips),
                "counterfactual_flow_variance_flip_count": float(
                    flow_variance_flips
                ),
                "counterfactual_coefficient_mean": float(
                    diagnostics["coefficients"][eligible].detach().mean().cpu()
                ),
            }
        if name == "production_pair":
            diagnostics = self.network.centered_production_pair_counterfactual_batch(
                observations, masks, device=self.device
            )
            eligible = diagnostics["eligible"]
            if int(eligible.sum().detach().cpu()) < minimum:
                raise RuntimeError("E2.7 production-pair pool lost eligible states")
            flow_gain = diagnostics["flow_gain"][eligible]
            cost_gain = diagnostics["cost_gain"][eligible]
            flow_margin = diagnostics["flow_margin"][eligible]
            cost_margin = diagnostics["cost_margin"][eligible]
            loss = 0.25 * (
                torch.relu(gap - flow_gain).square().mean()
                + torch.relu(-flow_margin).square().mean()
                + torch.relu(gap - cost_gain).square().mean()
                + torch.relu(-cost_margin).square().mean()
            )
            correct = int(
                diagnostics["flow_correct"].sum().detach().cpu()
                + diagnostics["cost_correct"].sum().detach().cpu()
            )
            eligible_count = int(eligible.sum().detach().cpu())
            return loss, {
                "production_pair_eligible_count": float(eligible_count),
                "production_pair_correct_count": float(correct),
                "production_pair_correct_rate": correct / (2.0 * eligible_count),
            }
        if name == "worker_variance":
            diagnostics = self.network.centered_worker_variance_counterfactual_batch(
                observations, masks, device=self.device
            )
            eligible = diagnostics["eligible"]
            if int(eligible.sum().detach().cpu()) < minimum:
                raise RuntimeError("E2.7 worker-variance pool lost eligible states")
            gain = diagnostics["variance_gain"][eligible]
            margin = diagnostics["variance_margin"][eligible]
            loss = 0.5 * (
                torch.relu(gap - gain).square().mean()
                + torch.relu(-margin).square().mean()
            )
            correct = int(diagnostics["variance_correct"].sum().detach().cpu())
            eligible_count = int(eligible.sum().detach().cpu())
            return loss, {
                "worker_variance_eligible_count": float(eligible_count),
                "worker_variance_correct_count": float(correct),
                "worker_variance_correct_rate": correct / float(eligible_count),
            }
        raise ValueError(f"unknown E2.7 centered objective {name!r}")

    def _counterfactual_loss(
        self,
        transitions: Sequence[Any],
        *,
        reward_phase: str,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Return the configured same-state preference consistency loss."""

        zero = next(self.network.parameters()).new_zeros(())
        disabled_metrics = {
            "counterfactual_eligible_count": 0.0,
            "counterfactual_high_flow_commit_flip_count": 0.0,
            "counterfactual_high_flow_commit_flip_rate": 0.0,
            "counterfactual_state_scale_mean": 0.0,
            "counterfactual_state_scale_max": 0.0,
            "counterfactual_low_flow_identity_violation_count": 0.0,
            "counterfactual_monotonicity_violation_count": 0.0,
            "counterfactual_flow_cost_flip_count": 0.0,
            "counterfactual_flow_variance_flip_count": 0.0,
            "counterfactual_coefficient_mean": 0.0,
            "counterfactual_gate_loss": 0.0,
            "production_pair_loss": 0.0,
            "production_pair_eligible_count": 0.0,
            "production_pair_correct_count": 0.0,
            "production_pair_correct_rate": 0.0,
            "worker_variance_loss": 0.0,
            "worker_variance_eligible_count": 0.0,
            "worker_variance_correct_count": 0.0,
            "worker_variance_correct_rate": 0.0,
        }
        settings = self.counterfactual_preference_consistency
        if not settings["enabled"]:
            return zero, disabled_metrics
        if settings.get("version") == "centered_gate_pair_worker_v3":
            gate_loss, gate_metrics = self._centered_pool_objective("gate")
            stage = str(getattr(self.network, "centered_preference_stage", "gate"))
            pair_loss = zero
            pair_metrics: dict[str, float] = {}
            worker_loss = zero
            worker_metrics: dict[str, float] = {}
            if stage in {"production_pair", "worker_variance"}:
                pair_loss, pair_metrics = self._centered_pool_objective(
                    "production_pair"
                )
            if stage == "worker_variance":
                worker_loss, worker_metrics = self._centered_pool_objective(
                    "worker_variance"
                )
            objectives = settings["objectives"]
            loss = (
                float(objectives["gate"]["loss_coefficient"]) * gate_loss
                + float(objectives["production_pair"]["loss_coefficient"]) * pair_loss
                + float(objectives["worker_variance"]["loss_coefficient"]) * worker_loss
            )
            return loss, {
                **disabled_metrics,
                **gate_metrics,
                **pair_metrics,
                **worker_metrics,
                "counterfactual_gate_loss": float(gate_loss.detach().cpu()),
                "production_pair_loss": float(pair_loss.detach().cpu()),
                "worker_variance_loss": float(worker_loss.detach().cpu()),
            }
        if reward_phase != "quality":
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
            **disabled_metrics,
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

    def _canonical_teacher_loss(
        self,
        transitions: Sequence[Any],
    ) -> torch.Tensor:
        zero = next(self.network.parameters()).new_zeros(())
        if not self.canonical_teacher_kl["enabled"]:
            return zero
        if self.canonical_teacher is None:
            raise RuntimeError(
                "canonical teacher KL is enabled before E1 warm-start"
            )
        canonical = np.asarray((0.5, 0.3, 0.2), dtype=np.float32)
        observations = [
            replace(
                transition.observation,
                preference=canonical.copy(),
            )
            for transition in transitions
        ]
        masks = [transition.action_mask for transition in transitions]
        current_logits, _ = self.network.forward_batch(
            observations, masks, device=self.device
        )
        with torch.no_grad():
            teacher_logits, _ = self.canonical_teacher.forward_batch(
                observations, masks, device=self.device
            )
        return kl_divergence(
            Categorical(logits=teacher_logits),
            Categorical(logits=current_logits),
        ).mean()

    def _run_centered_pool_gradient_preflight(self, name: str) -> None:
        """Verify one E2.7 v3 auxiliary path once per installed pool."""

        with torch.no_grad():
            probe_loss, _ = self._centered_pool_objective(name)
        if not bool(torch.isfinite(probe_loss)):
            raise RuntimeError(f"E2.7 {name} preflight loss is non-finite")
        probe_loss_value = float(probe_loss.detach().cpu())
        if probe_loss_value < 0.0:
            raise RuntimeError(f"E2.7 {name} preflight loss is negative")
        if probe_loss_value > 0.0:
            self.optimizer.zero_grad()
            probe_loss, _ = self._centered_pool_objective(name)
            probe_loss.backward()
            if name == "gate":
                module = getattr(self.network, "centered_gate_coefficients", None)
                parameters = tuple(module.parameters()) if module is not None else ()
            elif name == "production_pair":
                parameters = (
                    getattr(self.network, "centered_production_pair_scale_raw", None),
                )
            else:
                parameters = (
                    getattr(self.network, "centered_worker_variance_scale_raw", None),
                )
            gradient_total = sum(
                float(parameter.grad.detach().abs().sum().cpu())
                for parameter in parameters
                if parameter is not None and parameter.grad is not None
            )
            self.optimizer.zero_grad()
            if not math.isfinite(gradient_total) or gradient_total <= 0.0:
                raise RuntimeError(
                    f"E2.7 {name} adapter gradient is zero or non-finite"
                )
        self._centered_pool_gradient_preflight_complete[name] = True
        if name == "gate":
            self._safe_pool_gradient_preflight_complete = True

    def _verify_centered_canonical_identity(self) -> float:
        """Return the maximum finite canonical logit difference from E1."""

        if self.canonical_teacher is None:
            raise RuntimeError("E2.7 canonical identity is enabled before warm-start")
        observations: list[Observation | PolicyObservation] = []
        masks: list[np.ndarray] = []
        for pool in self.centered_state_pools.values():
            observations.extend(observation for observation, _ in pool)
            masks.extend(mask for _, mask in pool)
        if not observations:
            raise RuntimeError("E2.7 canonical identity has no fixed state pool")
        canonical = np.asarray((0.5, 0.3, 0.2), dtype=np.float32)
        canonical_observations = [
            replace(observation, preference=canonical.copy())
            for observation in observations
        ]
        with torch.no_grad():
            current_logits, _ = self.network.forward_batch(
                canonical_observations, masks, device=self.device
            )
            teacher_logits, _ = self.canonical_teacher.forward_batch(
                canonical_observations, masks, device=self.device
            )
        valid_actions = torch.zeros_like(current_logits, dtype=torch.bool)
        for row, action_mask in enumerate(masks):
            legal = ~torch.as_tensor(
                np.asarray(action_mask, dtype=bool),
                dtype=torch.bool,
                device=self.device,
            )
            valid_actions[row, : legal.numel()] = legal
        if not bool(valid_actions.any()):
            raise RuntimeError("E2.7 canonical identity has no legal logits")
        if not bool(torch.isfinite(current_logits[valid_actions]).all()) or not bool(
            torch.isfinite(teacher_logits[valid_actions]).all()
        ):
            raise RuntimeError("E2.7 canonical identity has non-finite legal logits")
        error = (
            current_logits[valid_actions] - teacher_logits[valid_actions]
        ).abs().max()
        if not bool(torch.isfinite(error)):
            raise RuntimeError("E2.7 canonical identity error is non-finite")
        result = float(error.detach().cpu())
        if result > 1e-8:
            raise RuntimeError(
                "E2.7 canonical policy logits drifted from E1: "
                f"max_abs_error={result:.12g}"
            )
        return result

    def update(
        self,
        buffer: RolloutBuffer,
        *,
        reward_phase: str = "legacy",
    ) -> dict[str, float | str]:
        if not buffer.transitions:
            raise ValueError("cannot update PPO with an empty buffer")
        e2_7_counterfactual = (
            self.counterfactual_preference_consistency.get("version")
            == "centered_gate_pair_worker_v3"
        )
        if e2_7_counterfactual:
            stage = str(getattr(self.network, "centered_preference_stage", "gate"))
            active_objectives = ["gate"]
            if stage in {"production_pair", "worker_variance"}:
                active_objectives.append("production_pair")
            if stage == "worker_variance":
                active_objectives.append("worker_variance")
            for name in active_objectives:
                minimum = int(
                    self.counterfactual_preference_consistency["objectives"][name][
                        "minimum_eligible_states"
                    ]
                )
                if len(self.centered_state_pools[name]) < minimum:
                    raise RuntimeError(
                        f"E2.7 fixed {name} state pool was not initialized or is too small"
                    )
                if not self._centered_pool_gradient_preflight_complete[name]:
                    self._run_centered_pool_gradient_preflight(name)
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
            "counterfactual_flow_cost_flip_count",
            "counterfactual_flow_variance_flip_count",
            "counterfactual_coefficient_mean",
            "counterfactual_gate_loss",
            "production_pair_loss",
            "production_pair_eligible_count",
            "production_pair_correct_count",
            "production_pair_correct_rate",
            "worker_variance_loss",
            "worker_variance_eligible_count",
            "worker_variance_correct_count",
            "worker_variance_correct_rate",
            "canonical_teacher_kl",
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
                canonical_teacher_loss = self._canonical_teacher_loss(
                    transitions
                )
                counterfactual_multiplier = (
                    1.0
                    if self.counterfactual_preference_consistency.get("version")
                    == "centered_gate_pair_worker_v3"
                    else float(
                        self.counterfactual_preference_consistency.get(
                            "loss_coefficient", 0.0
                        )
                    )
                )
                loss = (
                    policy_loss
                    + self.config["value_coefficient"] * value_loss
                    - self.config["entropy_coefficient"] * entropy
                    + counterfactual_multiplier * counterfactual_loss
                    + float(self.canonical_teacher_kl["coefficient"])
                    * canonical_teacher_loss
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
                        counterfactual_metrics[
                            "counterfactual_flow_cost_flip_count"
                        ],
                        counterfactual_metrics[
                            "counterfactual_flow_variance_flip_count"
                        ],
                        counterfactual_metrics[
                            "counterfactual_coefficient_mean"
                        ],
                        counterfactual_metrics["counterfactual_gate_loss"],
                        counterfactual_metrics["production_pair_loss"],
                        counterfactual_metrics["production_pair_eligible_count"],
                        counterfactual_metrics["production_pair_correct_count"],
                        counterfactual_metrics["production_pair_correct_rate"],
                        counterfactual_metrics["worker_variance_loss"],
                        counterfactual_metrics["worker_variance_eligible_count"],
                        counterfactual_metrics["worker_variance_correct_count"],
                        counterfactual_metrics["worker_variance_correct_rate"],
                        float(canonical_teacher_loss.detach().item()),
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
        if e2_7_counterfactual:
            for key in (
                "counterfactual_gate_loss",
                "production_pair_loss",
                "worker_variance_loss",
            ):
                if result[key] < 0.0:
                    raise FloatingPointError(f"E2.7 {key} cannot be negative")
            result["canonical_identity_max_abs_error"] = (
                self._verify_centered_canonical_identity()
            )
            gate_loss_value = result["counterfactual_gate_loss"]
            if gate_loss_value < 0.0:
                raise FloatingPointError(
                    "E2.7 gate counterfactual loss cannot be negative"
                )
            result["counterfactual_constraint_status"] = (
                "constraint_satisfied"
                if gate_loss_value == 0.0
                else "constraint_active"
            )
            stage = str(getattr(self.network, "centered_preference_stage", "gate"))
            result["production_pair_constraint_status"] = (
                "inactive"
                if stage == "gate"
                else (
                    "constraint_satisfied"
                    if result["production_pair_loss"] == 0.0
                    else "constraint_active"
                )
            )
            result["worker_variance_constraint_status"] = (
                "inactive"
                if stage != "worker_variance"
                else (
                    "constraint_satisfied"
                    if result["worker_variance_loss"] == 0.0
                    else "constraint_active"
                )
            )
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
        if self.warm_start_report:
            saved_metadata.setdefault(
                "warm_start", dict(self.warm_start_report)
            )
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
        saved_centered = checkpoint_spec.get(
            "centered_preference_adapter", {}
        )
        if (
            isinstance(saved_centered, dict)
            and "active_stage" in saved_centered
            and hasattr(self.network, "set_centered_preference_stage")
        ):
            self.network.set_centered_preference_stage(
                str(saved_centered["active_stage"])
            )
        if load_optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        return dict(checkpoint.get("metadata", {}))

    def restore_e1_teacher_from_warm_start_report(
        self,
        report: dict[str, Any],
    ) -> None:
        """Restore E1 provenance and the frozen canonical teacher on resume."""

        if not isinstance(report, dict):
            raise TypeError("warm-start report must be an object")
        source_path = Path(str(report.get("source_checkpoint", "")))
        if not source_path.is_file():
            raise ValueError("warm-start source checkpoint is unavailable on resume")
        actual_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if actual_hash != str(report.get("source_checkpoint_sha256", "")):
            raise ValueError("warm-start source checkpoint hash changed on resume")
        source_checkpoint = torch.load(
            source_path, map_location=self.device, weights_only=False
        )
        source = source_checkpoint.get("network")
        if not isinstance(source, dict):
            raise ValueError("warm-start source checkpoint has no network state")
        source_metadata = source_checkpoint.get("metadata", {})
        if "accepted_episode" not in source_metadata:
            raise ValueError("warm-start source is no longer an accepted checkpoint")
        if len(source) != int(report.get("loaded_shared_parameter_count", -1)):
            raise ValueError("warm-start shared parameter count changed on resume")
        teacher = deepcopy(self.network).to(self.device)
        teacher_state = teacher.state_dict()
        missing = sorted(set(source) - set(teacher_state))
        mismatched = sorted(
            key
            for key in set(source) & set(teacher_state)
            if tuple(source[key].shape) != tuple(teacher_state[key].shape)
        )
        if missing or mismatched:
            raise ValueError(
                "cannot reconstruct E1 canonical teacher: "
                f"missing={missing}, shape_mismatches={mismatched}"
            )
        for key in teacher_state:
            if key in source:
                teacher_state[key] = source[key]
            else:
                teacher_state[key] = torch.zeros_like(teacher_state[key])
        teacher.load_state_dict(teacher_state, strict=True)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        self.canonical_teacher = teacher
        self.warm_start_report = dict(report)

    def warm_start_from_e1(
        self,
        path: str | Path,
        *,
        expected_shared_parameter_count: int = 116,
    ) -> dict[str, Any]:
        """Load every E1 tensor exactly while leaving E2.7 adapters fresh."""

        checkpoint_path = Path(path)
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        if not isinstance(checkpoint, dict):
            raise ValueError("E1 warm-start checkpoint root must be a mapping")
        source = checkpoint.get("network")
        if not isinstance(source, dict):
            raise ValueError("E1 warm-start checkpoint has no network state")
        source_spec = infer_checkpoint_network_spec(checkpoint)
        if str(source_spec.get("production_action_semantics")) != (
            "pair_plus_defer_v1"
        ):
            raise ValueError("E1 warm-start source must use flat pair+defer semantics")
        if source_spec.get("preference_conditioning") not in {None, "none"}:
            raise ValueError("E1 warm-start source must be preference independent")
        metadata = dict(checkpoint.get("metadata", {}))
        if "accepted_episode" not in metadata:
            raise ValueError("E1 warm-start source is not an accepted checkpoint")
        if len(source) != int(expected_shared_parameter_count):
            raise ValueError(
                "E1 warm-start shared parameter count changed: "
                f"expected={expected_shared_parameter_count}, actual={len(source)}"
            )
        target = self.network.state_dict()
        missing = sorted(set(source) - set(target))
        shape_mismatches = sorted(
            key
            for key in set(source) & set(target)
            if tuple(source[key].shape) != tuple(target[key].shape)
        )
        if missing or shape_mismatches:
            raise ValueError(
                "E1 warm-start is not lossless: "
                f"missing={missing}, shape_mismatches={shape_mismatches}"
            )
        merged = dict(target)
        for key, value in source.items():
            merged[key] = value
        self.network.load_state_dict(merged, strict=True)
        loaded = self.network.state_dict()
        unequal = [
            key
            for key, value in source.items()
            if not torch.equal(loaded[key].detach().cpu(), value.detach().cpu())
        ]
        if unequal:
            raise RuntimeError(f"E1 warm-start tensor verification failed: {unequal}")
        # Warm-start is transfer, never optimizer resume.
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=self.config["learning_rate"]
        )
        self.canonical_teacher = deepcopy(self.network).to(self.device)
        self.canonical_teacher.eval()
        for parameter in self.canonical_teacher.parameters():
            parameter.requires_grad_(False)
        new_keys = sorted(set(target) - set(source))
        report = {
            "mode": "e1_to_e2_7_lossless_warm_start_v1",
            "source_checkpoint": str(checkpoint_path.resolve()),
            "source_checkpoint_sha256": hashlib.sha256(
                checkpoint_path.read_bytes()
            ).hexdigest(),
            "source_network_weights_sha256": network_weights_sha256(source),
            "source_accepted_episode": int(metadata["accepted_episode"]),
            "loaded_shared_parameter_count": len(source),
            "new_parameter_count": len(new_keys),
            "new_parameter_keys": new_keys,
            "shape_mismatch_count": 0,
            "missing_shared_parameter_count": 0,
            "optimizer_restored": False,
            "optimizer_state_entry_count": len(self.optimizer.state),
            "canonical_identity_contract": (
                "exact E1 shared tensors; centered residuals are zero at "
                "w=(0.5,0.3,0.2)"
            ),
        }
        self.warm_start_report = report
        return dict(report)

    def verify_warm_start_canonical_identity(
        self,
        observation: Observation | PolicyObservation,
        action_mask: np.ndarray,
        *,
        source_checkpoint: str | Path,
    ) -> dict[str, Any]:
        """Numerically prove E2.7 is exactly E1 at canonical preference."""

        checkpoint = torch.load(
            Path(source_checkpoint), map_location=self.device, weights_only=False
        )
        metadata = checkpoint.get("metadata", {})
        effective_config = metadata.get("effective_config", {})
        source_network_config = effective_config.get("network")
        if not isinstance(source_network_config, dict):
            raise ValueError(
                "E1 checkpoint lacks effective_config.network for identity check"
            )
        canonical = np.asarray((0.5, 0.3, 0.2), dtype=np.float32)
        canonical_observation = replace(
            observation,
            preference=canonical,
        )
        source_network = build_actor_critic(
            canonical_observation, source_network_config
        ).to(self.device)
        source_network.load_state_dict(checkpoint["network"], strict=True)
        source_network.eval()
        target_was_training = self.network.training
        self.network.eval()
        with torch.no_grad():
            source_logits, source_value = source_network.forward_batch(
                [canonical_observation], [action_mask], device=self.device
            )
            target_logits, target_value = self.network.forward_batch(
                [canonical_observation], [action_mask], device=self.device
            )
        self.network.train(target_was_training)
        action_count = int(np.asarray(action_mask).shape[0])
        source_row = source_logits[0, :action_count]
        target_row = target_logits[0, :action_count]
        source_probability = torch.softmax(source_row, dim=0)
        target_probability = torch.softmax(target_row, dim=0)
        raw_equal = torch.equal(source_row, target_row)
        probability_equal = torch.equal(source_probability, target_probability)
        value_equal = torch.equal(source_value, target_value)
        result = {
            "preference": [0.5, 0.3, 0.2],
            "raw_logits_exact": bool(raw_equal),
            "probabilities_exact": bool(probability_equal),
            "value_exact": bool(value_equal),
            "raw_logits_max_abs_error": float(
                (source_row - target_row).abs().max().cpu()
            ),
            "probability_max_abs_error": float(
                (source_probability - target_probability).abs().max().cpu()
            ),
            "value_max_abs_error": float(
                (source_value - target_value).abs().max().cpu()
            ),
            "pass": bool(raw_equal and probability_equal and value_equal),
        }
        if not result["pass"]:
            raise RuntimeError(
                "E1 warm-start canonical identity verification failed: "
                f"{result}"
            )
        self.warm_start_report["canonical_identity_result"] = result
        return dict(self.warm_start_report)
