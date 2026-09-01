from __future__ import annotations

import argparse
import copy
import hashlib
import math
import random
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs import load_config, project_path
from data.dataset import (
    ALL_SPLITS,
    PERSISTED_SPLITS,
    GeneratedInstanceRecord,
    build_all_dataset_splits,
    build_dataset_split,
    template_sha256,
    validate_instance_seed,
)
from data.feasibility import (
    StaticFeasibilityAnalysis,
    analyze_static_feasibility,
    cheap_feasibility_precheck,
)
from data.models import (
    AssemblyInstance,
    MachineModuleSpec,
    MachineSpec,
    ModuleCostSpec,
    OperationSpec,
    OrderSpec,
    WorkerSpec,
    load_instance_yaml,
    save_instance_pickle,
    validate_instance,
)


PRESSURE_TYPES = (
    "easy",
    "balanced",
    "machine_bottleneck",
    "reconfiguration_bottleneck",
    "worker_bottleneck",
    "fatigue_bottleneck",
    "high_arrival_pressure",
)
OOD_LIKE_SPLITS = frozenset({"ood", "stress"})


class GenerationError(RuntimeError):
    def __init__(
        self,
        *,
        seed: int,
        split: str,
        pressure_type: str,
        generator_version: str,
        attempts: int,
        failure_reasons: Counter[str],
        last_metrics: dict[str, Any] | None,
    ):
        self.seed = seed
        self.split = split
        self.pressure_type = pressure_type
        self.generator_version = generator_version
        self.attempts = attempts
        self.failure_reasons = dict(failure_reasons)
        self.last_metrics = last_metrics
        super().__init__(
            "instance generation failed: "
            f"seed={seed}, split={split}, pressure_type={pressure_type}, "
            f"generator_version={generator_version}, attempts={attempts}, "
            f"failure_reasons={dict(failure_reasons)}, "
            f"last_metrics={last_metrics}"
        )


def _stable_seed(*parts: Any) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _quantize(minutes: float, resolution: float) -> float:
    ticks = max(1, int(math.ceil((minutes - 1e-9) / resolution)))
    return round(ticks * resolution, 10)


def _weighted_choice(rng: random.Random, weights: dict[str, float]) -> str:
    names = list(weights)
    values = [float(weights[name]) for name in names]
    return rng.choices(names, weights=values, k=1)[0]


def _rollout_metrics(
    instance: AssemblyInstance,
    config: dict[str, Any],
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], Any]:
    from agent.baselines import HeuristicPolicy
    from environment import AssemblySchedulingEnv
    from environment.types import MachineState, OperationState

    environment = AssemblySchedulingEnv(config)
    environment.temporal_progress_callback = progress_callback
    environment.reset(instance, build_observation=False)
    policy = HeuristicPolicy()
    seen_ready: set[str] = set()
    ready_count = 0
    ready_mismatch_count = 0
    while not (environment.terminated or environment.truncated):
        for operation in environment.operations:
            if (
                operation.state == OperationState.READY
                and operation.spec.id not in seen_ready
            ):
                seen_ready.add(operation.spec.id)
                ready_count += 1
                matching_idle = any(
                    machine.state == MachineState.IDLE
                    and machine.current_module == operation.spec.required_module
                    and operation.spec.required_module
                    in machine.spec.module_parameters
                    for machine in environment.machines
                )
                ready_mismatch_count += int(not matching_idle)
        action = policy.select_action(environment)
        environment.step(action, build_observation=False)
    base_metrics = environment.metrics()
    wave_overlap_values = _wave_overlap_values(instance, environment.schedule_log)
    completed_reconfigurations = int(base_metrics["completed_reconfigurations"])
    total_operations = len(instance.operations)
    heuristic_metrics = {
        "heuristic_completed": bool(base_metrics["terminated"]),
        "heuristic_truncated": bool(base_metrics["truncated"]),
        "heuristic_terminal_reason": base_metrics["terminal_reason"],
        "heuristic_makespan": float(base_metrics["time"]),
        "heuristic_flow_time": base_metrics["total_flow_time"],
        "heuristic_reconfiguration_cost": float(
            base_metrics["reconfiguration_cost"]
        ),
        "heuristic_reconfiguration_ratio": (
            completed_reconfigurations / total_operations
            if total_operations
            else 0.0
        ),
        "ready_configuration_gap_ratio": (
            ready_mismatch_count / ready_count if ready_count else 0.0
        ),
        "ready_operation_count": ready_count,
        "ready_configuration_gap_count": ready_mismatch_count,
        "maximum_worker_fatigue": float(
            base_metrics["maximum_worker_fatigue"]
        ),
        "mean_peak_worker_fatigue": float(
            base_metrics.get(
                "mean_peak_worker_fatigue",
                base_metrics["maximum_worker_fatigue"],
            )
        ),
        "fatigue_masked_action_count": int(
            base_metrics["fatigue_masked_action_count"]
        ),
        "fatigue_masked_action_ratio": float(
            base_metrics["fatigue_masked_action_ratio"]
        ),
        "worker_competition_event_count": int(
            base_metrics["worker_competition_event_count"]
        ),
        "machine_waiting_for_worker_time": float(
            base_metrics["machine_waiting_for_worker_time"]
        ),
        "worker_workload_variance": float(
            base_metrics["worker_load_variance"]
        ),
        "mean_wave_overlap_ratio": (
            sum(wave_overlap_values) / len(wave_overlap_values)
            if wave_overlap_values
            else 0.0
        ),
        "max_wave_overlap_ratio": max(wave_overlap_values, default=0.0),
        "schedule_violations": environment.validate_schedule(),
        **{
            name: base_metrics.get(name)
            for name in (
                "temporal_oracle_call_count",
                "temporal_oracle_cache_hit_count",
                "temporal_subproblem_cache_hit_count",
                "temporal_oracle_searched_nodes",
                "temporal_oracle_option_evaluations",
                "temporal_frontier_options_before",
                "temporal_frontier_options_after",
                "temporal_dominated_option_count",
                "temporal_oracle_feasible_count",
                "temporal_oracle_infeasible_count",
                "temporal_oracle_unknown_count",
                "temporal_budget_termination_counts",
                "temporal_search_implementation",
            )
        },
    }
    return heuristic_metrics, environment


def _wave_overlap_values(
    instance: AssemblyInstance,
    schedule: list[dict[str, Any]],
) -> list[float]:
    by_operation = {str(row["operation_id"]): row for row in schedule}
    ordered_waves = list(instance.waves)
    values: list[float] = []
    for previous_wave, next_wave in zip(ordered_waves, ordered_waves[1:]):
        next_orders = [
            order for order in instance.orders if order.wave == next_wave
        ]
        if not next_orders:
            continue
        release = min(order.release_time for order in next_orders)
        rows = [
            by_operation[operation.id]
            for order in instance.orders
            if order.wave == previous_wave
            for operation in order.operations
            if operation.id in by_operation
        ]
        total = sum(float(row["duration"]) for row in rows)
        remaining = 0.0
        for row in rows:
            start = float(row["start"])
            end = float(row["end"])
            if end <= release:
                continue
            remaining += end - max(start, release)
        values.append(remaining / total if total else 0.0)
    return values


def _load_metrics(
    instance: AssemblyInstance,
    analysis: StaticFeasibilityAnalysis | None = None,
) -> dict[str, Any]:
    effective = analysis or analyze_static_feasibility(instance)
    return effective.load_metrics()


class InstanceGenerator:
    def __init__(
        self,
        template: AssemblyInstance,
        generator_config: dict[str, Any],
        *,
        config: dict[str, Any] | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.template = template
        self.settings = generator_config
        self.config = config or load_config("configs/default.json")
        self.version = str(generator_config["version"])
        self.template_instance = str(
            self.config.get("dataset", {}).get(
                "template_instance",
                template.instance_id,
            )
        )
        self.template_hash = template_sha256(template)
        self.progress_callback = progress_callback

    def _emit_progress(self, **payload: Any) -> None:
        if self.progress_callback is not None:
            self.progress_callback(payload)

    def generate(
        self,
        *,
        seed: int,
        split: str,
        pressure_type: str,
        ood_factor: str | None = None,
        classify_reconfiguration_value: bool = True,
    ) -> GeneratedInstanceRecord:
        if pressure_type not in PRESSURE_TYPES:
            raise ValueError(f"unknown pressure_type {pressure_type}")
        if split not in ALL_SPLITS:
            raise ValueError(f"unknown split {split}")
        validate_instance_seed(self.config, split, seed)
        if split not in OOD_LIKE_SPLITS and ood_factor is not None:
            raise ValueError(
                "ood_factor is only valid for the ood and stress splits"
            )
        if split in OOD_LIKE_SPLITS:
            allowed = set(self.settings["ood"]["factors"])
            if ood_factor is None:
                chooser = random.Random(
                    _stable_seed(self.version, split, seed, "ood-factor")
                )
                ood_factor = chooser.choice(sorted(allowed))
            if ood_factor not in allowed:
                raise ValueError(f"unknown ood_factor {ood_factor}")
        cost_rng = random.Random(
            _stable_seed(self.version, split, seed, pressure_type, "cost")
        )
        cost_profile = _weighted_choice(
            cost_rng, self.settings["cost_profile_weights"]
        )
        failure_reasons: Counter[str] = Counter()
        last_metrics: dict[str, Any] | None = None
        maximum_attempts = int(self.settings["max_generation_attempts"])
        for attempt in range(maximum_attempts):
            self._emit_progress(
                phase="candidate_build",
                seed=seed,
                generation_attempt=attempt,
                pressure_type=pressure_type,
            )
            attempt_seed = _stable_seed(
                self.version,
                self.template_hash,
                split,
                seed,
                pressure_type,
                cost_profile,
                ood_factor or "",
                attempt,
            )
            rng = random.Random(attempt_seed)
            try:
                instance = self._build_candidate(
                    rng=rng,
                    seed=seed,
                    split=split,
                    pressure_type=pressure_type,
                    cost_profile=cost_profile,
                    ood_factor=ood_factor,
                    attempt=attempt,
                )
                minimum_workers = (
                    1
                    if split in OOD_LIKE_SPLITS
                    and ood_factor == "worker_qualification_sparsity"
                    else 3
                )
                validate_instance(
                    instance,
                    minimum_qualified_workers=minimum_workers,
                )
                static_analysis = analyze_static_feasibility(instance)
                static_metrics = _load_metrics(instance, static_analysis)
                static_reasons = self._static_rejection_reasons(
                    instance, static_metrics
                )
                if static_reasons:
                    failure_reasons.update(static_reasons)
                    last_metrics = static_metrics
                    continue
                precheck = cheap_feasibility_precheck(
                    instance, static_analysis
                )
                if not precheck.passed:
                    failure_reasons.update(precheck.reason_codes)
                    last_metrics = {
                        **static_metrics,
                        "feasibility_precheck": precheck.to_dict(),
                    }
                    continue
                self._emit_progress(
                    phase="heuristic_rollout",
                    seed=seed,
                    generation_attempt=attempt,
                    pressure_type=pressure_type,
                )
                heuristic_metrics, environment = _rollout_metrics(
                    instance,
                    self.config,
                    progress_callback=self.progress_callback,
                )
                metrics = {**static_metrics, **heuristic_metrics}
                dynamic_reasons = self._dynamic_rejection_reasons(
                    instance=instance,
                    pressure_type=pressure_type,
                    split=split,
                    metrics=metrics,
                )
                if dynamic_reasons:
                    failure_reasons.update(dynamic_reasons)
                    last_metrics = metrics
                    continue
                if classify_reconfiguration_value:
                    value_class, counterfactual_count = (
                        self._classify_reconfiguration_value(instance)
                    )
                else:
                    value_class, counterfactual_count = None, 0
                metadata = {
                    "generator_version": self.version,
                    "template_instance": self.template_instance,
                    "template_sha256": self.template_hash,
                    "seed": seed,
                    "split": split,
                    "distribution": (
                        split if split in OOD_LIKE_SPLITS else "id"
                    ),
                    "pressure_type": pressure_type,
                    "cost_profile": cost_profile,
                    "ood_factor": ood_factor,
                    "generation_attempt": attempt,
                    "attempt_seed": attempt_seed,
                    "pressure_metrics": static_metrics,
                    "feasibility_precheck": precheck.to_dict(),
                    "heuristic_metrics": heuristic_metrics,
                    "reconfiguration_value_class": value_class,
                    "counterfactual_candidate_count": counterfactual_count,
                    "generation_rejection_reasons": dict(
                        sorted(failure_reasons.items())
                    ),
                }
                self._emit_progress(
                    phase="generation_complete",
                    seed=seed,
                    generation_attempt=attempt,
                    pressure_type=pressure_type,
                    oracle_calls=heuristic_metrics.get(
                        "temporal_oracle_call_count", 0
                    ),
                    search_nodes=heuristic_metrics.get(
                        "temporal_oracle_searched_nodes", 0
                    ),
                    option_evaluations=heuristic_metrics.get(
                        "temporal_oracle_option_evaluations", 0
                    ),
                    root_cache_hits=heuristic_metrics.get(
                        "temporal_oracle_cache_hit_count", 0
                    ),
                    subproblem_cache_hits=heuristic_metrics.get(
                        "temporal_subproblem_cache_hit_count", 0
                    ),
                )
                return GeneratedInstanceRecord(instance, metadata)
            except (RuntimeError, ValueError) as error:
                failure_reasons[f"candidate_error:{type(error).__name__}"] += 1
                last_metrics = {"error": str(error)}
        raise GenerationError(
            seed=seed,
            split=split,
            pressure_type=pressure_type,
            generator_version=self.version,
            attempts=maximum_attempts,
            failure_reasons=failure_reasons,
            last_metrics=last_metrics,
        )

    def _build_candidate(
        self,
        *,
        rng: random.Random,
        seed: int,
        split: str,
        pressure_type: str,
        cost_profile: str,
        ood_factor: str | None,
        attempt: int,
    ) -> AssemblyInstance:
        profile = self.settings["pressure_profiles"][pressure_type]
        order_low, order_high = profile["order_count"]
        operation_low, operation_high = profile["operations_per_order"]
        order_count = rng.randint(int(order_low), int(order_high))
        wave_ids = tuple(self.template.waves)
        if len(wave_ids) != int(self.settings["wave_count"]):
            raise ValueError("template wave count does not match generator config")
        modules = tuple(self.template.modules)
        dominants, bottleneck_module = self._dominant_modules(
            rng, pressure_type, modules
        )
        dominant_share = rng.uniform(*profile["dominant_module_share"])
        if ood_factor == "module_demand_imbalance":
            dominant_share = rng.uniform(0.78, 0.82)
        release_windows = [tuple(value) for value in profile["release_windows"]]
        if ood_factor == "arrival_overlap":
            release_windows = [(0.0, 0.03), (0.05, 0.08), (0.10, 0.14)]
        counts = [
            rng.randint(int(operation_low), int(operation_high))
            for _ in range(order_count)
        ]
        wave_indexes = [
            min(len(wave_ids) - 1, index * len(wave_ids) // order_count)
            for index in range(order_count)
        ]
        routes = self._routes(
            rng=rng,
            counts=counts,
            wave_indexes=wave_indexes,
            dominants=dominants,
            modules=modules,
            dominant_share=dominant_share,
            pressure_type=pressure_type,
        )
        processing_low, processing_high = self.settings["processing_time"]
        orders: list[OrderSpec] = []
        for index, (count, wave_index, route) in enumerate(
            zip(counts, wave_indexes, routes)
        ):
            wave_id = wave_ids[wave_index]
            low, high = release_windows[wave_index]
            release = _quantize(
                rng.uniform(float(low), float(high)) * self.template.horizon,
                self.template.resolution,
            )
            order_id = f"R{index + 1}"
            operations = []
            for sequence, module in enumerate(route, start=1):
                if (
                    pressure_type == "machine_bottleneck"
                    and module == bottleneck_module
                ):
                    base_time = rng.randint(12, 14)
                elif pressure_type == "easy":
                    base_time = rng.randint(8, 11)
                else:
                    base_time = rng.randint(
                        int(processing_low), int(processing_high)
                    )
                operations.append(
                    OperationSpec(
                        id=f"O_{order_id}_{sequence}",
                        order_id=order_id,
                        sequence=sequence,
                        required_module=module,
                        base_processing_time=float(base_time),
                    )
                )
            orders.append(
                OrderSpec(
                    id=order_id,
                    wave=wave_id,
                    release_time=release,
                    operations=tuple(operations),
                )
            )
        ordered_orders = tuple(
            sorted(orders, key=lambda order: (order.release_time, order.id))
        )
        waves: dict[str, dict[str, Any]] = {}
        for wave_index, wave_id in enumerate(wave_ids):
            wave_orders = [
                order for order in ordered_orders if order.wave == wave_id
            ]
            waves[wave_id] = {
                "order_ids": [order.id for order in wave_orders],
                "release_interval": [
                    min(order.release_time for order in wave_orders),
                    max(order.release_time for order in wave_orders),
                ],
                "dominant_module": dominants[wave_index],
            }
        reconfiguration_scale = rng.uniform(
            *profile["reconfiguration_scale"]
        )
        if ood_factor == "reconfiguration_time_scale":
            reconfiguration_scale = rng.uniform(1.5, 2.0)
        machine_specs = self._machines(
            rng=rng,
            pressure_type=pressure_type,
            cost_profile=cost_profile,
            first_dominant=dominants[0],
            bottleneck_module=bottleneck_module,
            reconfiguration_scale=reconfiguration_scale,
            speed_ood=ood_factor == "processing_time_heterogeneity",
        )
        worker_specs = self._workers(
            rng=rng,
            pressure_type=pressure_type,
            cost_profile=cost_profile,
            bottleneck_module=bottleneck_module,
            sparse_ood=ood_factor == "worker_qualification_sparsity",
            fatigue_ood=ood_factor == "initial_fatigue",
        )
        module_costs = self._module_costs(rng, cost_profile)
        instance_id = (
            f"{split}_{ood_factor}_{seed}"
            if split in OOD_LIKE_SPLITS
            else f"{split}_{pressure_type}_{seed}"
        )
        return replace(
            self.template,
            instance_id=instance_id,
            instance_type=(
                "generated_ood"
                if split == "ood"
                else (
                    "generated_stress"
                    if split == "stress"
                    else "generated_id"
                )
            ),
            machines=machine_specs,
            workers=worker_specs,
            orders=ordered_orders,
            waves=waves,
            module_costs=module_costs,
        )

    def _dominant_modules(
        self,
        rng: random.Random,
        pressure_type: str,
        modules: tuple[str, ...],
    ) -> tuple[tuple[str, ...], str]:
        machine_counts = {
            module: sum(
                module in machine.module_parameters
                for machine in self.template.machines
            )
            for module in modules
        }
        bottleneck = min(modules, key=lambda value: (machine_counts[value], value))
        alternatives = [module for module in modules if module != bottleneck]
        if pressure_type in {
            "machine_bottleneck",
            "worker_bottleneck",
            "fatigue_bottleneck",
        }:
            middle = rng.choice(alternatives)
            return (bottleneck, middle, bottleneck), bottleneck
        shuffled = list(modules)
        rng.shuffle(shuffled)
        dominants = []
        while len(dominants) < len(self.template.waves):
            for module in shuffled:
                if not dominants or module != dominants[-1]:
                    dominants.append(module)
                if len(dominants) == len(self.template.waves):
                    break
        return tuple(dominants), bottleneck

    def _routes(
        self,
        *,
        rng: random.Random,
        counts: list[int],
        wave_indexes: list[int],
        dominants: tuple[str, ...],
        modules: tuple[str, ...],
        dominant_share: float,
        pressure_type: str,
    ) -> list[tuple[str, ...]]:
        routes: list[tuple[str, ...] | None] = [None] * len(counts)
        for wave_index, dominant in enumerate(dominants):
            order_indexes = [
                index
                for index, value in enumerate(wave_indexes)
                if value == wave_index
            ]
            total = sum(counts[index] for index in order_indexes)
            desired = int(round(total * dominant_share))
            desired = min(desired, total - len(order_indexes))
            desired = max(desired, int(math.ceil(total * 0.60)))
            non_dominant_count = total - desired
            selected: set[tuple[int, int]] = set()
            for order_index in order_indexes:
                count = counts[order_index]
                if pressure_type in {
                    "reconfiguration_bottleneck",
                    "worker_bottleneck",
                    "fatigue_bottleneck",
                }:
                    position = 1 if count > 2 else count - 1
                else:
                    position = count - 1
                selected.add((order_index, position))
            available = [
                (order_index, position)
                for order_index in order_indexes
                for position in range(counts[order_index])
                if (order_index, position) not in selected
            ]
            rng.shuffle(available)
            selected.update(available[: non_dominant_count - len(selected)])
            for order_index in order_indexes:
                route = [dominant] * counts[order_index]
                other_modules = [module for module in modules if module != dominant]
                for position in range(counts[order_index]):
                    if (order_index, position) in selected:
                        route[position] = rng.choice(other_modules)
                routes[order_index] = tuple(route)
        return [route for route in routes if route is not None]

    def _machines(
        self,
        *,
        rng: random.Random,
        pressure_type: str,
        cost_profile: str,
        first_dominant: str,
        bottleneck_module: str,
        reconfiguration_scale: float,
        speed_ood: bool,
    ) -> tuple[MachineSpec, ...]:
        downtime_scale = 1.5 if cost_profile == "downtime_dominant" else (
            0.75 if cost_profile != "balanced_cost" else 1.0
        )
        compatible_bottleneck = [
            machine
            for machine in self.template.machines
            if bottleneck_module in machine.module_parameters
        ]
        kept_bottleneck_machine = compatible_bottleneck[0].id
        result = []
        for machine in self.template.machines:
            parameters: dict[str, MachineModuleSpec] = {}
            for module, values in machine.module_parameters.items():
                speed_local = (
                    rng.uniform(0.80, 1.20)
                    if speed_ood
                    else rng.uniform(*self.settings["processing_factor_local"])
                )
                parameters[module] = MachineModuleSpec(
                    installation_base_time=_quantize(
                        values.installation_base_time
                        * reconfiguration_scale
                        * rng.uniform(*self.settings["reconfiguration_factor_local"]),
                        self.template.resolution,
                    ),
                    disassembly_base_time=_quantize(
                        values.disassembly_base_time
                        * reconfiguration_scale
                        * rng.uniform(*self.settings["reconfiguration_factor_local"]),
                        self.template.resolution,
                    ),
                    processing_speed_factor=round(
                        values.processing_speed_factor * speed_local, 10
                    ),
                )
            initial_module = machine.initial_module
            compatible = tuple(parameters)
            if pressure_type == "easy" and first_dominant in compatible:
                initial_module = first_dominant
            elif (
                pressure_type == "machine_bottleneck"
                and bottleneck_module in compatible
            ):
                initial_module = (
                    bottleneck_module
                    if machine.id == kept_bottleneck_machine
                    else next(
                        module
                        for module in compatible
                        if module != bottleneck_module
                    )
                )
            elif (
                pressure_type
                in {
                    "reconfiguration_bottleneck",
                    "worker_bottleneck",
                    "fatigue_bottleneck",
                }
                and first_dominant in compatible
            ):
                initial_module = next(
                    module for module in compatible if module != first_dominant
                )
            elif pressure_type in {"balanced", "high_arrival_pressure"}:
                if rng.random() < 0.20:
                    initial_module = rng.choice(compatible)
            result.append(
                MachineSpec(
                    id=machine.id,
                    initial_module=initial_module,
                    downtime_cost_per_minute=round(
                        machine.downtime_cost_per_minute
                        * downtime_scale
                        * rng.uniform(*self.settings["cost_factor_local"]),
                        10,
                    ),
                    module_parameters=parameters,
                )
            )
        return tuple(result)

    def _workers(
        self,
        *,
        rng: random.Random,
        pressure_type: str,
        cost_profile: str,
        bottleneck_module: str,
        sparse_ood: bool,
        fatigue_ood: bool,
    ) -> tuple[WorkerSpec, ...]:
        labor_scale = 1.5 if cost_profile == "labor_dominant" else (
            0.75 if cost_profile != "balanced_cost" else 1.0
        )
        qualified_ids = [
            worker.id
            for worker in self.template.workers
            if bottleneck_module in worker.qualified_modules
        ]
        retained = set(qualified_ids[:2]) if sparse_ood else set(qualified_ids)
        result = []
        for worker in self.template.workers:
            qualifications = tuple(worker.qualified_modules)
            if (
                sparse_ood
                and bottleneck_module in qualifications
                and worker.id not in retained
            ):
                qualifications = tuple(
                    module
                    for module in qualifications
                    if module != bottleneck_module
                )
            if fatigue_ood:
                initial_fatigue = rng.uniform(0.25, 0.40)
            elif pressure_type == "fatigue_bottleneck":
                initial_fatigue = rng.uniform(0.10, 0.25)
            elif pressure_type == "easy":
                initial_fatigue = rng.uniform(0.05, 0.10)
            else:
                initial_fatigue = rng.uniform(*self.settings["initial_fatigue"])
            result.append(
                WorkerSpec(
                    id=worker.id,
                    qualified_modules=qualifications,
                    labor_cost_per_minute=round(
                        worker.labor_cost_per_minute
                        * labor_scale
                        * rng.uniform(*self.settings["cost_factor_local"]),
                        10,
                    ),
                    initial_fatigue=round(initial_fatigue, 10),
                )
            )
        return tuple(result)

    def _module_costs(
        self,
        rng: random.Random,
        cost_profile: str,
    ) -> dict[str, ModuleCostSpec]:
        fixed_scale = 1.5 if cost_profile == "fixed_cost_dominant" else (
            0.75 if cost_profile != "balanced_cost" else 1.0
        )
        return {
            module: ModuleCostSpec(
                fixed_disassembly_cost=round(
                    costs.fixed_disassembly_cost
                    * fixed_scale
                    * rng.uniform(*self.settings["cost_factor_local"]),
                    10,
                ),
                fixed_installation_cost=round(
                    costs.fixed_installation_cost
                    * fixed_scale
                    * rng.uniform(*self.settings["cost_factor_local"]),
                    10,
                ),
            )
            for module, costs in self.template.module_costs.items()
        }

    def _static_rejection_reasons(
        self,
        instance: AssemblyInstance,
        metrics: dict[str, Any],
    ) -> list[str]:
        reasons: list[str] = []
        for order in instance.orders:
            if len({operation.required_module for operation in order.operations}) < 2:
                reasons.append("order_module_diversity")
                break
        previous = None
        for wave_id, wave in instance.waves.items():
            dominant = str(wave["dominant_module"])
            operations = [
                operation
                for order in instance.orders
                if order.wave == wave_id
                for operation in order.operations
            ]
            share = sum(
                operation.required_module == dominant for operation in operations
            ) / len(operations)
            if share < 0.60:
                reasons.append("wave_dominant_share")
            if previous == dominant:
                reasons.append("adjacent_wave_module")
            previous = dominant
        if not math.isfinite(float(metrics["total_effective_load"])):
            reasons.append("invalid_load")
        return reasons

    def _dynamic_rejection_reasons(
        self,
        *,
        instance: AssemblyInstance,
        pressure_type: str,
        split: str,
        metrics: dict[str, Any],
    ) -> list[str]:
        reasons: list[str] = []
        if metrics["schedule_violations"]:
            reasons.append("schedule_infeasible")
        if int(metrics["ready_configuration_gap_count"]) < 1:
            reasons.append("no_ready_configuration_gap")
        if split != "stress" and not metrics["heuristic_completed"]:
            reasons.append("heuristic_truncated")
            return reasons
        if (
            split != "stress"
            and float(metrics["heuristic_makespan"])
            > float(
                self.settings["acceptance"][
                    "max_heuristic_makespan_minutes"
                ]
            )
        ):
            reasons.append("heuristic_makespan")
        if pressure_type == "easy":
            if metrics["heuristic_makespan"] > 0.85 * instance.horizon:
                reasons.append("easy_makespan")
            if metrics["fatigue_masked_action_count"] > 0:
                reasons.append("easy_fatigue")
            if metrics["worker_competition_event_count"] > 0:
                reasons.append("easy_worker_competition")
            if metrics["heuristic_reconfiguration_ratio"] > 0.20:
                reasons.append("easy_reconfiguration")
        elif pressure_type == "machine_bottleneck":
            if metrics["max_module_load"] < float(
                self.settings["acceptance"]["machine_min_module_load"]
            ):
                reasons.append("machine_load")
        elif pressure_type == "reconfiguration_bottleneck":
            if metrics["ready_configuration_gap_ratio"] < float(
                self.settings["acceptance"]["reconfiguration_min_ready_gap"]
            ):
                reasons.append("configuration_gap")
            if metrics["heuristic_reconfiguration_ratio"] < float(
                self.settings["acceptance"]["reconfiguration_min_ratio"]
            ):
                reasons.append("reconfiguration_ratio")
        elif pressure_type == "worker_bottleneck":
            if metrics["worker_competition_event_count"] < 1:
                reasons.append("worker_competition")
            if metrics["machine_waiting_for_worker_time"] <= 0:
                reasons.append("worker_wait")
        elif pressure_type == "fatigue_bottleneck":
            if metrics["fatigue_masked_action_count"] < 1:
                reasons.append("fatigue_mask")
        elif pressure_type == "high_arrival_pressure":
            if metrics["mean_wave_overlap_ratio"] < float(
                self.settings["acceptance"]["arrival_min_overlap"]
            ):
                reasons.append("arrival_overlap")
        return reasons

    def _classify_reconfiguration_value(
        self,
        instance: AssemblyInstance,
    ) -> tuple[str, int]:
        from agent.baselines import HeuristicPolicy
        from environment import AssemblySchedulingEnv, DecisionType

        environment = AssemblySchedulingEnv(self.config)
        environment.reset(instance, build_observation=False)
        policy = HeuristicPolicy()
        values: list[float] = []
        maximum = int(self.settings.get("max_counterfactual_candidates", 6))
        while not (environment.terminated or environment.truncated):
            action = policy.select_action(environment)
            if (
                len(values) < maximum
                and environment.decision_type == DecisionType.PRODUCTION
                and action != environment.advance_action
            ):
                operation_index, machine_index = (
                    environment.decode_production_action(action)
                )
                operation = environment.operations[operation_index]
                machine = environment.machines[machine_index]
                target_module = operation.spec.required_module
                is_reconfiguration = machine.current_module != target_module
                has_waiting_target_machine = any(
                    other.current_module == target_module
                    and target_module in other.spec.module_parameters
                    for other in environment.machines
                    if other.spec.id != machine.spec.id
                )
                mask = environment.get_action_mask()
                if (
                    is_reconfiguration
                    and has_waiting_target_machine
                    and not mask[environment.advance_action]
                ):
                    switch_environment = copy.deepcopy(environment)
                    wait_environment = copy.deepcopy(environment)
                    switch_environment.step(action, build_observation=False)
                    wait_environment.step(
                        wait_environment.advance_action,
                        build_observation=False,
                    )
                    switch_valid = self._finish_counterfactual(
                        switch_environment,
                        policy,
                    )
                    wait_valid = self._finish_counterfactual(
                        wait_environment,
                        policy,
                        avoid_operation_id=operation.spec.id,
                    )
                    if switch_valid and wait_valid:
                        switch_metrics = switch_environment.metrics()
                        wait_metrics = wait_environment.metrics()
                        delta = float(wait_metrics["total_flow_time"]) - float(
                            switch_metrics["total_flow_time"]
                        )
                        if abs(delta) <= 1e-9:
                            delta = float(
                                wait_metrics["reconfiguration_cost"]
                            ) - float(
                                switch_metrics["reconfiguration_cost"]
                            )
                        values.append(delta)
            environment.step(action, build_observation=False)
        if not values:
            return "mixed", 0
        positive_share = sum(value > 0 for value in values) / len(values)
        if positive_share >= 0.75:
            return "high_benefit", len(values)
        if positive_share <= 0.25:
            return "low_benefit", len(values)
        return "mixed", len(values)

    def _finish_counterfactual(
        self,
        environment: Any,
        policy: Any,
        *,
        avoid_operation_id: str | None = None,
    ) -> bool:
        from environment import DecisionType

        while not (environment.terminated or environment.truncated):
            if (
                avoid_operation_id is not None
                and environment.decision_type == DecisionType.PRODUCTION
            ):
                action = self._select_avoiding_reconfiguration(
                    environment,
                    avoid_operation_id,
                )
                if action is None:
                    return False
            else:
                action = policy.select_action(environment)
            environment.step(action, build_observation=False)
        return bool(environment.terminated)

    @staticmethod
    def _select_avoiding_reconfiguration(
        environment: Any,
        operation_id: str,
    ) -> int | None:
        import numpy as np

        mask = environment.get_action_mask().copy()
        for action in np.flatnonzero(~mask):
            action = int(action)
            if action == environment.advance_action:
                continue
            operation_index, machine_index = (
                environment.decode_production_action(action)
            )
            operation = environment.operations[operation_index]
            machine = environment.machines[machine_index]
            if (
                operation.spec.id == operation_id
                and machine.current_module != operation.spec.required_module
            ):
                mask[action] = True
        pair_actions = [
            int(value)
            for value in np.flatnonzero(~mask)
            if int(value) != environment.advance_action
        ]
        if not pair_actions:
            return (
                environment.advance_action
                if not mask[environment.advance_action]
                else None
            )
        scored = []
        for action in pair_actions:
            operation_index, machine_index = (
                environment.decode_production_action(action)
            )
            processing = environment.estimate_processing_ticks(
                operation_index, machine_index
            )
            reconfiguration = environment.estimate_reconfiguration_ticks(
                operation_index, machine_index
            )
            operation = environment.operations[operation_index]
            same_configuration = (
                environment.machines[machine_index].current_module
                == operation.spec.required_module
            )
            scored.append(
                (
                    processing + reconfiguration,
                    0 if same_configuration else 1,
                    operation.spec.sequence,
                    operation_index,
                    machine_index,
                    action,
                )
            )
        return min(scored)[-1]


def generate_instance(
    template: AssemblyInstance,
    *,
    generator_config: dict[str, Any],
    seed: int,
    split: str,
    pressure_type: str,
    config: dict[str, Any] | None = None,
) -> AssemblyInstance:
    return InstanceGenerator(
        template,
        generator_config,
        config=config,
    ).generate(
        seed=seed,
        split=split,
        pressure_type=pressure_type,
    ).instance


def generate_random_instance(
    template: AssemblyInstance,
    *,
    num_orders: int,
    seed: int,
) -> AssemblyInstance:
    """Backward-compatible small random order generator."""
    if num_orders < 3:
        raise ValueError("num_orders must be at least three")
    rng = random.Random(seed)
    wave_ids = tuple(template.waves)
    release_ranges = ((0.00, 0.20), (0.20, 0.45), (0.45, 0.70))
    orders: list[OrderSpec] = []
    for order_index in range(num_orders):
        wave_index = min(2, (order_index * 3) // num_orders)
        wave_id = wave_ids[wave_index]
        low, high = release_ranges[wave_index]
        release = round(rng.uniform(low, high) * template.horizon, 1)
        count = rng.randint(3, 5)
        dominant = str(template.waves[wave_id]["dominant_module"])
        dominant_count = int(math.ceil(count * 0.60))
        alternatives = [
            value for value in template.modules if value != dominant
        ]
        modules = [dominant] * dominant_count + [
            rng.choice(alternatives)
            for _ in range(count - dominant_count)
        ]
        rng.shuffle(modules)
        order_id = f"R{order_index + 1}"
        operations = tuple(
            OperationSpec(
                id=f"O_{order_id}_{sequence}",
                order_id=order_id,
                sequence=sequence,
                required_module=module,
                base_processing_time=float(rng.randint(8, 14)),
            )
            for sequence, module in enumerate(modules, start=1)
        )
        orders.append(
            OrderSpec(
                id=order_id,
                wave=wave_id,
                release_time=release,
                operations=operations,
            )
        )
    ordered_orders = tuple(
        sorted(orders, key=lambda order: (order.release_time, order.id))
    )
    random_waves = {}
    for wave_id, template_wave in template.waves.items():
        wave_orders = [order for order in ordered_orders if order.wave == wave_id]
        wave = dict(template_wave)
        wave["order_ids"] = [order.id for order in wave_orders]
        if wave_orders:
            wave["release_interval"] = [
                min(order.release_time for order in wave_orders),
                max(order.release_time for order in wave_orders),
            ]
        random_waves[wave_id] = wave
    random_instance = replace(
        template,
        instance_id=f"random_{num_orders}_seed_{seed}",
        instance_type="random",
        orders=ordered_orders,
        waves=random_waves,
    )
    validate_instance(random_instance)
    return random_instance


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reproducible assembly instances")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--random-orders", type=int, default=0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output")
    parser.add_argument(
        "--build-split",
        choices=PERSISTED_SPLITS,
    )
    parser.add_argument("--build-all", action="store_true")
    parser.add_argument(
        "--profile",
        choices=("dev", "publication"),
        default="dev",
    )
    parser.add_argument("--count", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--build-train-cache", action="store_true")
    parser.add_argument("--parallel-envs", type=int)
    parser.add_argument("--run-name")
    args = parser.parse_args()

    config = load_config(args.config)
    fixed = load_instance_yaml(project_path(config["paths"]["fixed_instance"]))
    build_modes = sum(
        bool(value)
        for value in (
            args.build_split,
            args.build_all,
            args.build_train_cache,
        )
    )
    if build_modes > 1:
        parser.error(
            "--build-split, --build-all and --build-train-cache are "
            "mutually exclusive"
        )
    if args.build_train_cache:
        from agent.ppo.parallel import ParallelEpisodeRunner
        from result import create_run_directory
        from result.io import write_config

        episode_count = (
            int(config["training"]["episodes"])
            if args.count is None
            else int(args.count)
        )
        if episode_count < 2:
            parser.error("--build-train-cache requires --count >= 2")
        worker_count = (
            int(config["training"]["parallel_envs"])
            if args.parallel_envs is None
            else int(args.parallel_envs)
        )
        worker_count = min(worker_count, episode_count)
        if worker_count < 2:
            parser.error("--parallel-envs must be at least 2")
        run_directory = create_run_directory(
            project_path(config["paths"]["result_root"]),
            label="training_instance_generation",
            run_name=args.run_name,
        )
        write_config(run_directory, config)
        with ParallelEpisodeRunner(
            config=config,
            template=fixed,
            episode_count=episode_count,
            worker_count=worker_count,
            diagnostic_directory=run_directory,
        ) as runner:
            runner.pre_generate_training_instances()
        print(f"saved training cache manifest: {run_directory}")
        return
    if args.build_all:
        manifests = build_all_dataset_splits(
            config=config,
            template=fixed,
            profile=args.profile,
            count=args.count,
            overwrite=args.overwrite,
            resume=not args.no_resume,
        )
        for split, manifest in manifests.items():
            print(f"saved {split} manifest: {manifest}")
        return
    if args.build_split:
        manifest = build_dataset_split(
            config=config,
            template=fixed,
            split=args.build_split,
            profile=args.profile,
            count=args.count,
            overwrite=args.overwrite,
            resume=not args.no_resume,
        )
        print(f"saved {args.build_split} manifest: {manifest}")
        return
    instance = fixed
    if args.random_orders:
        instance = generate_random_instance(
            fixed,
            num_orders=args.random_orders,
            seed=config["seed"] if args.seed is None else args.seed,
        )
    if args.output:
        output = project_path(args.output)
    elif args.random_orders:
        effective_seed = config["seed"] if args.seed is None else args.seed
        output = project_path(
            f"data/instances/random_{args.random_orders}_seed_{effective_seed}.pkl"
        )
    else:
        output = project_path(config["paths"]["instance_cache"])
    save_instance_pickle(instance, output)
    print(
        f"saved {instance.instance_id}: "
        f"{len(instance.machines)} machines, {len(instance.workers)} workers, "
        f"{len(instance.orders)} orders, {len(instance.operations)} operations -> {output}"
    )


if __name__ == "__main__":
    main()
