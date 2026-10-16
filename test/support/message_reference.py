"""Reusable scheduling regression cases."""
from __future__ import annotations

from copy import deepcopy

from pathlib import Path

from statistics import median

import sys

import time

ROOT = Path(__file__).resolve().parents[1]

import torch

from torch import nn

from torch.nn import functional as F

from agent.baselines import HeuristicPolicy

from agent.ppo.network import build_actor_critic

from configs import load_config, project_path

from data.dataset import load_dataset_split

from data.models import load_instance_yaml

from environment import ASSEMBLY_EDGE_TYPES, AssemblySchedulingEnv

from result.io import write_json

class _LinearMessageReference(nn.Module):
    """Benchmark/test oracle only; not a selectable or saved network variant."""

    def __init__(self, layer):
        super().__init__()
        self.transforms, self.norms, self.dropout = layer.transforms, layer.norms, layer.dropout

    def forward(self, embeddings, relations):
        total = {name: torch.zeros_like(value) for name, value in embeddings.items()}
        degree = {name: value.new_zeros((len(value), 1)) for name, value in embeddings.items()}
        for edge_type in ASSEMBLY_EDGE_TYPES:
            source_type, _, target_type = edge_type
            index, features, bidirectional = relations[edge_type]
            if index.shape[1] == 0:
                continue
            source, target = index
            transform = self.transforms["__".join(edge_type)]
            message = transform(torch.cat((embeddings[source_type][source], features), -1))
            total[target_type].index_add_(0, target, message)
            degree[target_type].index_add_(0, target, message.new_ones((len(message), 1)))
            if bidirectional:
                reverse = transform(torch.cat((embeddings[target_type][target], features), -1))
                total[source_type].index_add_(0, source, reverse)
                degree[source_type].index_add_(0, source, reverse.new_ones((len(reverse), 1)))
        return {name: self.dropout(F.relu(self.norms[name](value+total[name]/degree[name].clamp_min(1))))
                for name, value in embeddings.items()}

def _linear_reference(network):
    reference = deepcopy(network)
    reference.message_layers = nn.ModuleList(_LinearMessageReference(layer) for layer in reference.message_layers)
    return reference
