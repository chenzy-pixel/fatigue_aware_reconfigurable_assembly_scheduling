from copy import deepcopy
import pytest
import numpy as np
import torch
from configs import load_config
from environment.types import FLOW_EXCESS
SHARED = {
    "flow": "configs/e1/single_flow.json",
    "cost": "configs/e1/single_cost.json",
    "workload": "configs/e1/single_workload.json",
    "variance": "configs/controls/single_variance.json",
    "universal": "configs/default.json",
}


@pytest.mark.parametrize("role", tuple(SHARED))
def test_shared_head_excess_physical_update_and_checkpoint(role, fixed_instance, tmp_path):
    from agent.ppo import PPOAgent, build_actor_critic
    from agent.ppo.buffer import RolloutBuffer
    from environment import AssemblySchedulingEnv
    from environment.types import proxy_return_from_metrics
    torch.set_num_threads(2)
    torch.manual_seed(11)
    config = load_config(SHARED[role])
    config["device"] = "cpu"
    env = AssemblySchedulingEnv(config)
    observation = env.reset(fixed_instance)
    agent = PPOAgent(build_actor_critic(observation, config["network"]), config["ppo"], device="cpu")
    buffer = RolloutBuffer()
    total = 0.0
    for _ in range(16):
        mask = env.get_action_mask()
        action, logp, value = agent.act(observation, mask)
        after, reward, _, _, _ = env.step(action)
        scalar = reward.scalarize(config["reward"])
        buffer.add(observation, mask, action, logp, value, scalar, env.task_done)
        total += scalar
        observation = after
        if env.task_done:
            break
    assert total == pytest.approx(proxy_return_from_metrics(env.metrics(), config))
    buffer.compute_gae(last_value=0 if env.task_done else agent.value(observation, env.get_action_mask()),
                       gamma=1, gae_lambda=config["ppo"]["gae_lambda"])
    assert all(np.isfinite(value) for value in agent.update(buffer).values())
    checkpoint = tmp_path / "shared_excess.pt"
    agent.save(checkpoint)
    agent.load(checkpoint)
    assert agent.network.network_spec()["actor_head_variant"] == "shared_preference"
    assert agent.network.network_spec()["flow_mode"] == FLOW_EXCESS
