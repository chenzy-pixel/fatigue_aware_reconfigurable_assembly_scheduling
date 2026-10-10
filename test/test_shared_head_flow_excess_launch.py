from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import numpy as np
import torch

from configs import load_config
from configs.formal_preferences import formal_preferences
from environment.types import FLOW_RAW, FLOW_EXCESS, flow_mode
import scripts.run_flow_excess_training as launch
from scripts.run_remaining_seed11_experiments import CONFIGS as REMAINING
from scripts.run_shared_head_flow_excess import CONFIGS as SHARED


def test_confirmed_remaining_matrix_has_seven_correct_experiments():
    assert tuple(REMAINING) == ("main_variance", "main_universal", "flow", "shared_head_flow", "shared_head_cost",
                                "shared_head_variance", "shared_head_universal")
    configs = {role: load_config(path) for role, path in REMAINING.items()}
    assert [c["training"]["episodes"] for c in configs.values()] == [1000, 2000, 1000, 1000, 1000, 1000, 2000]
    assert flow_mode(configs["main_variance"]) == FLOW_RAW
    assert flow_mode(configs["main_universal"]) == FLOW_RAW
    assert configs["main_universal"]["objective_scalarizer"]["scales"]["flow"] == 1089.15
    assert configs["main_universal"]["reward"]["mode"] == "single_stage_progress_quality_failure_v3"
    assert configs["main_universal"]["runtime_manifest"]["actor_head_variant"] == "objective_experts"
    assert len(formal_preferences(configs["main_universal"], "final_test")) == 66
    assert all(flow_mode(configs[role]) == FLOW_EXCESS for role in tuple(REMAINING)[2:])
    for role in SHARED:
        assert configs[role]["network"]["actor_head_variant"] == "shared_preference"
        assert configs[role]["runtime_manifest"]["actor_head_variant"] == "shared_preference"
        assert configs[role]["reward"]["mode"] == "single_stage_progress_quality_failure_v4"
    for role, weights in (("shared_head_flow", [1, 0, 0]), ("shared_head_cost", [0, 1, 0]),
                          ("shared_head_variance", [0, 0, 1])):
        assert configs[role]["preference"]["quality"]["fixed"] == weights
        assert len(formal_preferences(configs[role], "final_test")) == 1
    assert len(formal_preferences(configs["shared_head_universal"], "final_test")) == 66


def test_remaining_launcher_saves_and_runs_the_confirmed_order(monkeypatch, tmp_path):
    from scripts.run_remaining_seed11_experiments import main
    originals = {path: load_config(path) for path in REMAINING.values()}
    monkeypatch.setattr(launch, "ROOT", tmp_path)
    monkeypatch.setattr(launch, "load_config", lambda path: deepcopy(originals[path.relative_to(tmp_path).as_posix()]))
    monkeypatch.setattr(launch, "startup_preflight", lambda configs, **kwargs: {})
    monkeypatch.setattr("sys.argv", ["run_remaining_seed11_experiments.py"])
    calls = []

    def child(command, **kwargs):
        with open(command[command.index("--config") + 1], encoding="utf-8") as stream:
            config = json.load(stream)
        calls.append(config)
        assert config["seed"] == 11 and config["training"]["episodes_per_update"] == 20
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(launch.subprocess, "run", child)
    main()
    assert [c["experiment_name"] for c in calls] == [originals[p]["experiment_name"] for p in REMAINING.values()]
    status = json.loads(next(tmp_path.rglob("launch.json")).read_text())
    assert [job["experiment"] for job in status["jobs"]] == list(REMAINING)
    assert all(job["status"] == "completed" for job in status["jobs"])


def test_remaining_checker_runs_no_training(monkeypatch, tmp_path):
    from scripts.run_remaining_seed11_experiments import main
    originals = {path: load_config(path) for path in REMAINING.values()}
    monkeypatch.setattr(launch, "ROOT", tmp_path)
    monkeypatch.setattr(launch, "load_config", lambda path: deepcopy(originals[path.relative_to(tmp_path).as_posix()]))
    checks = []
    monkeypatch.setattr(launch, "startup_preflight", lambda configs, **kwargs: checks.append(kwargs) or {})
    monkeypatch.setattr(launch.subprocess, "run", lambda *args, **kwargs: pytest.fail("checker launched training"))
    monkeypatch.setattr("sys.argv", ["check_remaining_seed11_experiments.py"])
    main(preflight_only_default=True)
    assert checks[0]["probe_network"]
    assert set(checks[0]["allowed_flow_modes"]) == {FLOW_RAW, FLOW_EXCESS}
    assert not (tmp_path / "result/runs").exists()


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
