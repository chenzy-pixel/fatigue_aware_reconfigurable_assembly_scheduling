"""DCAN adapted to the native operation / machine / worker event MDP.

Conditional operation attention follows the author implementation at commit
caf60d8da721382c79009539a9c8d9ccbc39e734. Service competition, auxiliary node
context and the scalar reward/critic are project adaptations.
"""
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from agent.ppo import PPOAgent
from environment import ASSEMBLY_NODE_TYPES, PRECEDES_EDGE, CAPABLE_EDGE, SERVICE_CANDIDATE_EDGE, DecisionType
from .common import observation_spec, physical_identity, candidates, digest


class ConditionalAttention(nn.Module):
    """Multihead additive attention with an updated preference token per node."""
    def __init__(self, hidden=128, heads=4):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        self.heads, self.dim = heads, hidden // heads
        self.transform = nn.Linear(hidden, hidden, bias=False)
        self.modulation = nn.Linear(hidden, 2*hidden, bias=False)
        self.score = nn.Parameter(torch.empty(heads, 2*self.dim))
        self.edge_score = nn.Parameter(torch.zeros(heads))
        nn.init.xavier_uniform_(self.score)

    def forward(self, values, preference, adjacency, intensity=None):
        n = len(values)
        if not n:
            return values, preference
        gamma, beta = self.modulation(preference).chunk(2, -1)
        h = self.transform(values*gamma+beta).reshape(n, self.heads, self.dim)
        p = self.transform(preference).reshape(n, self.heads, self.dim)
        key = torch.cat((h.unsqueeze(0).expand(n, -1, -1, -1), p[:, None]), dim=1)
        valid = torch.cat((adjacency, torch.ones((n, 1), dtype=torch.bool, device=values.device)), 1)
        score = ((h[:, None]*self.score[:, :self.dim]).sum(-1)
                 + (key*self.score[:, self.dim:]).sum(-1))
        if intensity is not None:
            score = score + torch.cat((intensity, intensity.mean(1, keepdim=True)), 1)[..., None]*self.edge_score
        scores = F.leaky_relu(score, .2).masked_fill(~valid[..., None], torch.finfo(h.dtype).min)
        result = (scores.softmax(1)[..., None]*key).sum(1).reshape(n, -1)
        # The author's preference update includes neighboring operations, self
        # and preference. It uses the preference as query and its own self slot.
        pref_key = key.clone()
        diagonal = torch.arange(n, device=values.device)
        pref_key[diagonal, diagonal] = p
        pref_key[:, -1] = h
        pref_score = F.leaky_relu((p[:, None]*self.score[:, :self.dim]).sum(-1)
                                  + (pref_key*self.score[:, self.dim:]).sum(-1), .2)
        pref_score = pref_score.masked_fill(~valid[..., None], torch.finfo(h.dtype).min)
        updated_pref = (pref_score.softmax(1)[..., None]*pref_key).sum(1).reshape(n, -1)
        return F.elu(result), F.elu(updated_pref)


class DCANActorCritic(nn.Module):
    requires_graph_observation = True

    def __init__(self, observation, config):
        super().__init__()
        self.spec = {"family": "dcan_assembly_v1", "observation": observation_spec(observation),
                     "hidden": 128, "layers": 2, "heads": 4,
                     "physics": physical_identity(config),
                     "source": "caf60d8da721382c79009539a9c8d9ccbc39e734"}
        hidden = 128
        self.project = nn.ModuleDict({k: nn.Linear(v.shape[1], hidden) for k, v in observation.node_features.items()})
        self.pref = nn.ModuleDict({k: nn.Linear(3, hidden) for k in ("operation", "machine", "worker")})
        self.context = nn.ModuleList([nn.ModuleDict({k: nn.Linear(hidden, hidden) for k in ASSEMBLY_NODE_TYPES}) for _ in range(2)])
        self.blocks = nn.ModuleList([nn.ModuleDict({k: ConditionalAttention(hidden, 4)
                                    for k in ("operation", "machine", "worker")}) for _ in range(2)])
        self.global_encoder = nn.Linear(len(observation.global_features), hidden)
        self.pair = nn.ModuleDict({k: nn.Sequential(nn.Linear(5*hidden+len(edge.feature_names)+3, hidden), nn.ReLU(), nn.Linear(hidden, 1))
                                  for k, edge in (("production", observation.relations[CAPABLE_EDGE]),
                                                  ("worker", observation.relations[SERVICE_CANDIDATE_EDGE]))})
        self.wait = nn.Sequential(nn.Linear(3*hidden+len(observation.action_set_features)+3, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.critic = nn.Sequential(nn.Linear(3*hidden+3, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def network_spec(self):
        return deepcopy(self.spec)

    def _forward(self, obs, mask, device):
        if observation_spec(obs) != self.spec["observation"]:
            raise ValueError("DCAN observation identity differs")
        tensor = lambda x: torch.as_tensor(np.asarray(x), device=device, dtype=torch.float32)
        nodes = {k: F.elu(self.project[k](tensor(v))) for k, v in obs.node_features.items()}
        preference = tensor(obs.preference)
        prefs = {k: self.pref[k](preference).expand(len(nodes[k]), -1) for k in self.pref}
        adjacency = {}
        op = torch.eye(len(nodes["operation"]), dtype=torch.bool, device=device)
        indices = torch.as_tensor(obs.relations[PRECEDES_EDGE].edge_index, device=device)
        if indices.numel():
            op[indices[0], indices[1]] = True
            op[indices[1], indices[0]] = True
        adjacency["operation"] = (op, None)
        for kind, relation, resource_side in (("machine", CAPABLE_EDGE, 1), ("worker", SERVICE_CANDIDATE_EDGE, 1)):
            edge = obs.relations[relation]
            other_count = len(nodes[relation[0]])
            incidence = torch.zeros((len(nodes[kind]), other_count), device=device)
            ix = torch.as_tensor(edge.edge_index, device=device)
            if ix.numel():
                incidence[ix[resource_side], ix[1-resource_side]] = 1
            overlap = incidence @ incidence.T
            adjacent = overlap > 0
            adjacent.fill_diagonal_(True)
            adjacency[kind] = (adjacent, overlap / max(1, other_count))
        for context, blocks in zip(self.context, self.blocks):
            additions = {k: torch.zeros_like(v) for k, v in nodes.items()}
            degrees = {k: v.new_zeros((len(v), 1)) for k, v in nodes.items()}
            for (source, _, target), edge in obs.relations.items():
                ix = torch.as_tensor(edge.edge_index, device=device)
                if not ix.numel():
                    continue
                additions[target].index_add_(0, ix[1], nodes[source][ix[0]])
                degrees[target].index_add_(0, ix[1], nodes[target].new_ones((ix.shape[1], 1)))
                if edge.bidirectional:
                    additions[source].index_add_(0, ix[0], nodes[target][ix[1]])
                    degrees[source].index_add_(0, ix[0], nodes[source].new_ones((ix.shape[1], 1)))
            nodes = {k: F.elu(v+context[k](additions[k]/degrees[k].clamp_min(1))) for k, v in nodes.items()}
            for kind in blocks:
                nodes[kind], prefs[kind] = blocks[kind](nodes[kind], prefs[kind], *adjacency[kind])
        pooled = torch.stack([v.mean(0) if len(v) else preference.new_zeros(128) for v in nodes.values()]).mean(0)
        glob = F.elu(self.global_encoder(tensor(obs.global_features)))
        pref_pool = torch.stack([v.mean(0) if len(v) else preference.new_zeros(128) for v in prefs.values()]).mean(0)
        context = torch.cat((pooled, glob, pref_pool, preference))
        value = self.critic(context).squeeze(-1)
        edge, left, right, action_ids = candidates(obs)
        ix = torch.as_tensor(edge.edge_index, device=device)
        local_pref = prefs[left][ix[0]] if left in prefs else prefs[right][ix[1]]
        features = torch.cat((nodes[left][ix[0]], nodes[right][ix[1]], local_pref,
                              pooled.expand(len(action_ids), -1), glob.expand(len(action_ids), -1),
                              tensor(edge.edge_features), preference.expand(len(action_ids), -1)), 1)
        label = "production" if obs.decision_type == DecisionType.PRODUCTION else "worker"
        logits = value.new_full((len(mask),), torch.finfo(value.dtype).min)
        logits = logits.index_copy(0, torch.as_tensor(action_ids, device=device), self.pair[label](features).squeeze(-1))
        logits[-1] = self.wait(torch.cat((context, tensor(obs.action_set_features)))).squeeze()
        mask = torch.as_tensor(mask, device=device, dtype=torch.bool)
        return logits.masked_fill(mask, torch.finfo(logits.dtype).min), value

    def forward_batch(self, observations, action_masks, *, device):
        pairs = [self._forward(obs, mask, device) for obs, mask in zip(observations, action_masks)]
        width = max(len(x[0]) for x in pairs)
        return torch.stack([F.pad(logits, (0, width-len(logits)), value=torch.finfo(logits.dtype).min)
                            for logits, _ in pairs]), torch.stack([v for _, v in pairs])

    def value_batch(self, observations, action_masks, *, device):
        return self.forward_batch(observations, action_masks, device=device)[1]


class DCANAgent(PPOAgent):
    """PPO optimization reused; checkpoint contract belongs to DCAN alone."""
    def assert_evaluation_config(self, config):
        if physical_identity(config) != self.network.spec["physics"]:
            raise ValueError("DCAN physical/objective configuration differs")

    def load(self, path, *, load_optimizer=False):
        from result.provenance import network_weights_sha256
        saved = torch.load(path, map_location=self.device, weights_only=False)
        if saved.get("network_spec") != self.network.network_spec():
            raise ValueError("DCAN checkpoint family/spec mismatch")
        expected = saved.get("metadata", {}).get("network_weights_sha256")
        if expected and expected != network_weights_sha256(saved["network"]):
            raise ValueError("DCAN checkpoint weights hash differs")
        self.network.load_state_dict(saved["network"])
        if load_optimizer:
            self.optimizer.load_state_dict(saved["optimizer"])
        self.loaded_checkpoint_metadata = saved.get("metadata", {})
        return self.loaded_checkpoint_metadata
