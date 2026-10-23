"""Synchronous on-policy A2C with one accumulated gradient step per rollout."""
import torch
from torch.distributions import Categorical
from torch.nn import functional as F

from agent.ppo import PPOAgent


class A2CAgent(PPOAgent):
    def save(self, path, metadata=None):
        super().save(path, {**(metadata or {}), 'algorithm':'a2c', 'update_rule':'synchronous_a2c_v1'})

    def load(self, path, *, load_optimizer=False):
        saved = torch.load(path,map_location='cpu',weights_only=False)
        if saved.get('metadata',{}).get('algorithm') != 'a2c':
            raise ValueError('A2C checkpoint algorithm differs')
        return super().load(path,load_optimizer=load_optimizer)

    def update(self, buffer):
        n = len(buffer.transitions)
        if not n:
            raise ValueError("cannot update A2C with an empty buffer")
        advantages = torch.tensor([t.advantage for t in buffer.transitions], device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        totals = torch.zeros(4, device=self.device)
        self.optimizer.zero_grad(set_to_none=True)
        for start in range(0, n, int(self.config['batch_size'])):
            batch = buffer.transitions[start:start + int(self.config['batch_size'])]
            logits, values = self.network.forward_batch(
                [t.observation for t in batch], [t.action_mask for t in batch], device=self.device)
            distribution = Categorical(logits=logits)
            actions = torch.tensor([t.action for t in batch], device=self.device)
            returns = torch.tensor([t.return_value for t in batch], device=self.device)
            policy_loss = -(distribution.log_prob(actions) * advantages[start:start+len(batch)].detach()).mean()
            value_loss = F.mse_loss(values, returns)
            entropy = distribution.entropy().mean()
            loss = policy_loss + self.config['value_coefficient'] * value_loss - self.config['entropy_coefficient'] * entropy
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite A2C loss")
            fraction = len(batch) / n
            (loss * fraction).backward()
            totals += fraction * torch.stack((policy_loss.detach(), value_loss.detach(), entropy.detach(), loss.detach()))
        gradient = torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config['max_grad_norm'])
        if not torch.isfinite(gradient):
            raise FloatingPointError("non-finite A2C gradient")
        self.optimizer.step()
        result = dict(zip(('policy_loss', 'value_loss', 'entropy', 'loss'), totals.cpu().tolist()))
        result.update(gradient_norm=float(gradient), optimizer_updates=1, sample_presentations=n)
        return result
