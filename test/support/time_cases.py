"""Reusable scheduling regression cases."""
from __future__ import annotations

from copy import deepcopy

from dataclasses import replace

import csv

import hashlib

import json

from pathlib import Path

import sys

import numpy as np

import torch

from agent.baselines import HeuristicPolicy

from agent.ppo import PPOAgent

from agent.ppo.network import build_actor_critic, NODE_TYPES

from agent.ppo.parallel import ParallelEpisodeRunner

from configs import load_config

from configs.config import public_config

from data.models import load_instance_yaml, validate_instance, OrderSpec, OperationSpec, instance_to_dict

from environment import AssemblySchedulingEnv, CAPABLE_EDGE, MACHINE_MODULE_EDGE

from support.observation_cases import build_alias_instance, reachable_histories, compare_observations

from support.state_cases import worker_alias_instance, worker_histories

OUTPUT = Path("result/analysis/schema9_followup")

def edge_times(env, operation, machine):

    obs = env.observe()

    store = obs.relations[CAPABLE_EDGE]

    op = env.instance.operation_index[operation]

    mi = env.instance.machine_index[machine]

    pos = np.flatnonzero((store.edge_index[0] == op) & (store.edge_index[1] == mi))[0]

    return {name: float(store.edge_features[pos, store.feature_names.index(name)])*env.instance.horizon

            for name in ("processing_time_norm", "earliest_start_time_norm", "resource_ready_time_norm", "predicted_finish_time_norm")}

def attributed_edge_instances(template):

    orders = tuple(OrderSpec(name, "audit_"+name, 0, (OperationSpec(name+"_1", name, 1, module, 150),))

                   for name, module in (("A", "A1"), ("B", "A2")))

    result = []

    for swapped in (False, True):

        machines = []

        for i, machine in enumerate(template.machines):

            parameters = dict(machine.module_parameters)

            if i in (0, 3):

                for j, module in enumerate(("A1", "A2")):

                    fast = ((i == 0) != (j == 1)) != swapped

                    parameters[module] = replace(parameters[module], processing_speed_factor=.9 if fast else 1.1,

                                                 installation_base_time=15, disassembly_base_time=15)

            machines.append(replace(machine, module_parameters=parameters))

        instance = replace(template, instance_id=f"attribute_pairing_{int(swapped)}", instance_type="test",

                           machines=tuple(machines), workers=tuple(replace(w, initial_fatigue=0) for w in template.workers),

                           orders=orders, waves={"audit_"+name: {"dominant_module": module, "order_ids": [name], "release_interval": [0, 0]}

                                                 for name, module in (("A", "A1"), ("B", "A2"))})

        validate_instance(instance)

        result.append(instance)

    return result
