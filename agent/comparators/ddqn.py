"""Masked, deduplicated dispatch-rule Double DQN; greedy delivery."""
from copy import deepcopy
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from environment import ASSEMBLY_NODE_TYPES, OPERATION_ORDER_EDGE, DecisionType
from .common import candidates, feature_column, pooled_state, observation_spec, physical_identity, save_torch

RULE_NAMES = ("earliest_finish_or_shortest_stage", "shortest_total_or_lowest_fatigue",
              "most_remaining_work_or_least_workload", "shortest_remaining_work_or_least_cost",
              "least_cost_or_oldest_lock", "least_workload_or_scarcest_service", "wait")


def rule_actions(obs, native_mask):
    """Unavailable/alias rules map to -1; retained IDs have distinct native actions."""
    edge, left, right, ids = candidates(obs)
    legal = ~np.asarray(native_mask, dtype=bool)[ids]
    indices = np.flatnonzero(legal)
    result = np.full(7, -1, dtype=np.int64)
    if len(indices):
        col = lambda name: feature_column(edge, name)
        if obs.decision_type == DecisionType.PRODUCTION:
            remain = np.zeros(len(obs.node_features["operation"]))
            links = obs.relations[OPERATION_ORDER_EDGE].edge_index
            values = obs.node_features["order"][:, obs.node_feature_names["order"].index("remaining_workload_norm")]
            remain[links[0]] = values[links[1]]
            scores = (col("predicted_finish_time_norm"), col("processing_time_norm")+col("reconfiguration_time_norm"),
                      -remain[edge.edge_index[0]], remain[edge.edge_index[0]],
                      col("fixed_disassembly_cost_norm")+col("fixed_installation_cost_norm")+col("estimated_labor_cost_norm")+col("estimated_downtime_cost_norm"),
                      col("estimated_worker_load_objective_delta_norm"))
        else:
            # SERVICE_CANDIDATE contains only pending service pairs; legality
            # additionally restricts availability and the fatigue safety limit.
            counts = np.bincount(edge.edge_index[0, indices], minlength=len(obs.node_features["machine"]))
            scores = (col("stage_duration_norm"), col("projected_fatigue_ratio"),
                      col("incremental_workload_objective_norm"),
                      col("fixed_cost_norm")+col("incremental_labor_cost_norm")+col("incremental_downtime_cost_norm"),
                      -col("stage_wait_time_norm"), counts[edge.edge_index[0]])
        for rule, score in enumerate(scores):
            winner = min(indices, key=lambda i: ((0 if col("projection_valid")[i] else 1) if obs.decision_type == DecisionType.PRODUCTION else 0, float(score[i]), int(ids[i])))
            result[rule] = ids[winner]
    if not native_mask[-1]:
        result[6] = len(native_mask)-1
    seen = set()
    for i, action in enumerate(result):
        if action in seen:
            result[i] = -1
        elif action >= 0:
            seen.add(action)
    if not seen:
        raise ValueError("no legal dispatch rule")
    return result


def double_q_targets(reward, terminated, online, target, next_mask, gamma=1.):
    valid = (~next_mask).any(-1)
    best = online.masked_fill(next_mask, torch.finfo(online.dtype).min).argmax(-1)
    boot = target.gather(1, best[:, None]).squeeze(1)
    return reward + gamma * torch.where((~terminated) & valid, boot, torch.zeros_like(boot))


class RuleDDQN:
    requires_graph_observation = True

    def __init__(self, obs, config):
        self.config, self.device = config, torch.device(config["device"])
        settings = config.get("comparison", {}).get("ddqn", {})
        self.spec = {"family": "rule_ddqn_v2", "observation": observation_spec(obs),
                     "physics": physical_identity(config), "rules": RULE_NAMES, "state_dim": len(pooled_state(obs)),
                     "hidden": [256, 256]}
        self.online = nn.Sequential(nn.Linear(self.spec["state_dim"], 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 7)).to(self.device)
        self.target = deepcopy(self.online).eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.get("comparison", {}).get("learning_rate", 3e-4))
        self.capacity = int(settings.get("capacity", 100000))
        self.replay, self.replay_cursor = [], 0
        self.batch_size = int(settings.get("batch_size", 128))
        self.warmup = int(settings.get("warmup", 1024))
        self.update_every = int(settings.get("update_every", 4))
        self.sync_every = int(settings.get("sync_every", 1000))
        self.steps, self.updates, self.sample_presentations = 0, 0, 0
        self.rng = random.Random(config["seed"])
        self.batch_epsilon = None

    def set_episode_progress(self, completed, total):
        if total <= 0 or not 0 <= completed <= total:
            raise ValueError("invalid episode exploration progress")
        self.batch_epsilon = max(.05, 1-.95*completed/(.5*total))

    @property
    def exploration_epsilon(self):
        if self.batch_epsilon is not None:
            return self.batch_epsilon
        return max(.05, 1-.95*self.steps/(.5*int(self.config.get("comparison", {}).get("training_steps", 1))))

    def act_rules(self, observations, masks, *, explore=False):
        state = np.stack([pooled_state(o) for o in observations])
        maps = [rule_actions(o, m) for o, m in zip(observations, masks)]
        legal = np.stack([m >= 0 for m in maps])
        with torch.no_grad():
            q = self.online(torch.as_tensor(state, device=self.device))
            rules = q.masked_fill(~torch.as_tensor(legal, device=self.device), torch.finfo(q.dtype).min).argmax(1).cpu().tolist()
        epsilon = self.exploration_epsilon
        for i in range(len(rules)):
            if explore and self.rng.random() < epsilon:
                rules[i] = self.rng.choice(np.flatnonzero(legal[i]).tolist())
        return [int(m[r]) for m, r in zip(maps, rules)], rules

    def remember(self, obs, rule, reward, next_obs, next_mask, terminated):
        if observation_spec(obs) != self.spec["observation"]:
            raise ValueError("DDQN observation identity differs")
        next_state = pooled_state(next_obs) if next_obs is not None else np.zeros_like(pooled_state(obs))
        next_rules = rule_actions(next_obs, next_mask) >= 0 if next_obs is not None else np.zeros(7, dtype=bool)
        transition = (pooled_state(obs), rule, reward, next_state, terminated, ~next_rules)
        if len(self.replay) < self.capacity:
            self.replay.append(transition)
        else:
            self.replay[self.replay_cursor] = transition
        self.replay_cursor = (self.replay_cursor+1) % self.capacity
        self.steps += 1
        if self.steps >= self.warmup and self.steps % self.update_every == 0 and len(self.replay) >= self.batch_size:
            return self.update()
        return None

    def update(self):
        batch = self.rng.sample(self.replay, self.batch_size)
        state, action, reward, following, done, mask = zip(*batch)
        state = torch.as_tensor(np.stack(state), device=self.device)
        following = torch.as_tensor(np.stack(following), device=self.device)
        reward = torch.tensor(reward, device=self.device, dtype=torch.float32)
        done = torch.tensor(done, device=self.device, dtype=torch.bool)
        mask = torch.as_tensor(np.stack(mask), device=self.device)
        with torch.no_grad():
            target = double_q_targets(reward, done, self.online(following), self.target(following), mask)
        selected = self.online(state).gather(1, torch.tensor(action, device=self.device)[:, None]).squeeze(1)
        loss = F.smooth_l1_loss(selected, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 5.)
        self.optimizer.step()
        self.updates += 1
        self.sample_presentations += self.batch_size
        if self.updates % self.sync_every == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.detach())

    def save(self, path, metadata=None):
        from result import network_weights_sha256
        metadata = {**(metadata or {}), "network_weights_sha256": network_weights_sha256(self.online.state_dict())}
        save_torch(path, {"network_spec": self.spec, "network": self.online.state_dict(), "target": self.target.state_dict(),
                          "optimizer": self.optimizer.state_dict(), "steps": self.steps, "updates": self.updates,
                          "sample_presentations": self.sample_presentations, "metadata": metadata})

    def load(self, path):
        from result import network_weights_sha256
        saved = torch.load(path, map_location=self.device, weights_only=False)
        if saved.get("network_spec") != self.spec:
            raise ValueError("DDQN checkpoint family/spec mismatch")
        if saved.get("metadata", {}).get("network_weights_sha256") != network_weights_sha256(saved["network"]):
            raise ValueError("DDQN checkpoint weights hash differs")
        self.online.load_state_dict(saved["network"])
        self.target.load_state_dict(saved["target"])
        self.steps, self.updates = saved["steps"], saved["updates"]
        self.sample_presentations = saved["sample_presentations"]
        return saved["metadata"]
