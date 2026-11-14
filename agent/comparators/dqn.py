"""Vanilla DQN over the shared masked dispatch-rule action space."""
import torch

from .ddqn import RuleDDQN


def dqn_targets(reward, terminated, target, next_mask, gamma=1.):
    """Select and evaluate the next legal action with the target network."""
    valid = (~next_mask).any(-1)
    boot = target.masked_fill(next_mask, torch.finfo(target.dtype).min).max(-1).values
    return reward + gamma * torch.where((~terminated) & valid, boot, torch.zeros_like(boot))


class RuleDQN(RuleDDQN):
    family = "rule_dqn_v1"

    def bootstrap_targets(self, reward, done, following, mask):
        return dqn_targets(reward, done, self.target(following), mask)

    def save(self, path, metadata=None):
        super().save(path, {**(metadata or {}), "algorithm": "dqn", "update_rule": "target_max_dqn_v1"})
