"""Exact globally reserved env.step collection over existing worker transport."""
from dataclasses import dataclass
import time

from agent.ppo.buffer import RolloutBuffer
from agent.ppo.parallel import _WorkerResetRequest, _WorkerStepRequest
from environment import quality_preference_for_episode


@dataclass
class Collected:
    buffer: RolloutBuffer
    steps: int
    episodes_started: int
    episodes: list
    seconds: float


def collect(runner, agent, config, start_episode, allowance, *, ddqn=False, episodes_per_batch=20):
    """Every issued command reserves exactly one real step, including WAIT.

    Policy freezes during each PPO batch. At the final global boundary, active
    trajectories bootstrap from their current observation; no fake failure.
    DDQN learns online at its configured raw-interaction cadence.
    """
    started = time.perf_counter()
    total = RolloutBuffer()
    active, histories, summary = {}, {}, []
    indices, used = 0, 0

    def fill(lanes):
        nonlocal indices
        commands = {}
        for lane in lanes:
            if indices >= episodes_per_batch or allowance-used <= len(active)+len(commands):
                break
            episode = start_episode+indices
            pref = quality_preference_for_episode(config, algorithm_seed=config["seed"], quality_episode_index=episode)
            commands[lane] = ("reset_online", _WorkerResetRequest(episode, preference=pref))
            histories[lane] = {"episode": episode, "steps": 0, "reward_sum": 0., "buffer": RolloutBuffer()}
            indices += 1
        if commands:
            for lane, response in runner._exchange(commands).items():
                if response.environment_step_count:
                    raise RuntimeError("comparison reset must not compress forced actions")
                if response.terminated or response.truncated:
                    raise RuntimeError("training instance terminated at reset")
                active[lane] = response
                histories[lane]["instance_id"] = response.instance_id
                histories[lane]["metadata"] = response.metadata

    def finish(lane, response, cutoff=False):
        context = histories[lane]
        if not ddqn:
            bootstrap = 0. if response.terminated else agent.value(response.observation, response.action_mask)
            context["buffer"].compute_gae(last_value=bootstrap, gamma=1., gae_lambda=config["ppo"]["gae_lambda"])
            total.extend(context["buffer"])
        summary.append({"episode": context["episode"], "steps": context["steps"],
                        "reward_sum": context["reward_sum"], "instance_id": context["instance_id"], "metadata": context["metadata"],
                        "budget_truncated": cutoff, "terminated": response.terminated,
                        "metrics": response.metrics if response.terminated or response.truncated else None})
        active.pop(lane, None)

    fill(range(runner.worker_count))
    while active and used < allowance:
        lanes = sorted(active)[:allowance-used]
        observations = [active[lane].observation for lane in lanes]
        masks = [active[lane].action_mask for lane in lanes]
        if ddqn:
            actions, rules = agent.act_rules(observations, masks, explore=True)
        else:
            actions, logps, values = agent.act_batch(observations, masks)
        # No compressed suffixes: a reservation cannot expand inside a worker.
        results = runner._exchange({lane: ("step", _WorkerStepRequest(action, max_environment_steps=1)) for lane, action in zip(lanes, actions)})
        used += len(lanes)
        completed = []
        for i, lane in enumerate(lanes):
            response = results[lane]
            if response.environment_step_count != 1:
                raise RuntimeError("worker violated reserved one-step allowance")
            reward = response.reward_vector.scalarize(config["reward"])
            if ddqn:
                agent.remember(observations[i], rules[i], reward, response.observation, response.action_mask, response.terminated)
            else:
                histories[lane]["buffer"].add(observations[i], masks[i], actions[i], logps[i], values[i], reward, response.terminated)
            histories[lane]["steps"] += 1
            histories[lane]["reward_sum"] += reward
            active[lane] = response
            if response.terminated or response.truncated:
                finish(lane, response)
                completed.append(lane)
        if used < allowance:
            fill(completed)
    for lane, response in list(active.items()):
        finish(lane, response, cutoff=True)
    return Collected(total, used, indices, summary, time.perf_counter()-started)
