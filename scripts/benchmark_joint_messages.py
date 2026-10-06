"""Compare current joint messages with an isolated pre-change linear reference."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from statistics import median
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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


def _states(config):
    instances = [load_instance_yaml(project_path(config["paths"]["fixed_instance"])),
                 load_dataset_split(config, "validation")[0].instance,
                 load_dataset_split(config, "stress")[0].instance]
    observations, masks = [], []
    policy = HeuristicPolicy()
    for instance in instances:
        env = AssemblySchedulingEnv(config)
        env.reset(instance, build_observation=False)
        for step in range(31):
            if env.task_done:
                break
            if step % 5 == 0:
                observations.append(env.observe())
                masks.append(env.get_action_mask())
            if step < 30:
                env.step(policy.select_action(env), build_observation=False)
    return observations[:20], masks[:20]


def benchmark(config):
    torch.set_num_threads(int(config["training"]["torch_num_threads"]))
    observations, masks = _states(config)
    torch.manual_seed(11)
    current = build_actor_critic(observations[0], config["network"])
    reference = _linear_reference(current)
    assert tuple(current.state_dict()) == tuple(reference.state_dict())
    for name, value in current.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[name], atol=0, rtol=0)
    rows = []
    for device in ["cpu"]+(["cuda"] if torch.cuda.is_available() else []):
        models = {"linear_reference": deepcopy(reference).to(device), "joint_relu": deepcopy(current).to(device)}
        for mode in ("critic", "actor_critic_forward_backward"):
            samples = {name: [] for name in models}
            for iteration in range(12):
                order = list(models) if iteration % 2 == 0 else list(reversed(models))
                for name in order:
                    model = models[name]
                    if device == "cuda":
                        torch.cuda.synchronize()
                    started = time.perf_counter()
                    if mode == "critic":
                        model.eval()
                        with torch.no_grad():
                            values = model.value_batch(observations, masks, device=device)
                    else:
                        model.train()
                        model.zero_grad(set_to_none=True)
                        logits, values = model.forward_batch(observations, masks, device=device)
                        (values.square().mean()+torch.logsumexp(logits, -1).mean()).backward()
                    if device == "cuda":
                        torch.cuda.synchronize()
                    duration = (time.perf_counter()-started)*1000
                    assert torch.isfinite(values).all()
                    if iteration >= 3:
                        samples[name].append(duration)
            baseline = median(samples["linear_reference"])
            for name, values in samples.items():
                rows.append({"device": device, "mode": mode, "variant": name,
                             "median_ms": median(values), "relative_to_linear": median(values)/baseline,
                             "overhead_above_15_percent": median(values)/baseline > 1.15,
                             "samples_ms": values})
    return {"scope": "component timing; excludes environment, optimizer and full PPO update",
            "network_spec": current.network_spec(), "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "cpu_threads": torch.get_num_threads(), "batch_size": len(observations),
            "parameter_count": sum(p.numel() for p in current.parameters()),
            "state_dict_matches_linear_reference": True,
            "phases": [item.decision_type.value for item in observations], "measurements": rows}


def main():
    output = project_path("result/audits/joint_messages_20261006")
    output.mkdir(parents=True, exist_ok=True)
    results = benchmark(load_config("configs/default.json"))
    # Tuple relation keys are valid in torch checkpoint specs, not JSON keys.
    results["network_spec"]["edge_feature_dimensions"] = {
        "__".join(key): value for key, value in results["network_spec"]["edge_feature_dimensions"].items()}
    write_json(output / "performance.json", results)
    for row in results["measurements"]:
        print({key: value for key, value in row.items() if key != "samples_ms"}, flush=True)


if __name__ == "__main__":
    main()
