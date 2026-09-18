# Fatigue-aware reconfigurable assembly scheduling

This repository contains the accepted E1 research implementation for
fatigue-aware reconfigurable assembly scheduling.

The executable stack is fixed to:

- pair-plus-WAIT actions in both decision phases;
- V7 HGNN actor-critic with bounded ranker-scale context residual;
- instantaneous physical legality for production and worker pairs;
- progress-certified WAIT transitions with soft completion diagnostics;
- single-objective guarded promotion protocol v5.

Implementation identities are generated in `runtime_manifest`; configuration
files cannot select alternative implementations.

## Layout

For a file-by-file component map, public contracts, and the training/evaluation
call graph, see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

- `agent/networks/`: V7 policy/value building blocks.
- `agent/ppo/`: rollout buffer, PPO update, and shared parallel collector.
- `configs/default.json`: complete current E1 configuration.
- `configs/e1/`: flow, cost, and variance objective overrides.
- `configs/baselines/`: current MO-ALNS comparison settings.
- `data/`: instance models, generators, fixed datasets, and manifests.
- `environment/`: facade, runtime state, action codec, dynamics, graph
  observation, reward, and diagnostics.
- `training/`: single-objective promotion protocol.
- `result/`: persistence, provenance, metrics, and dashboards.

Historical experiment sources are available at Git tag
`archive/pre-latest-only-20260908`. Existing run artifacts under `result/`
remain untouched. A compact findings summary is in
`docs/HISTORICAL_EXPERIMENT_FINDINGS.md`.

## Environment

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Train

The same `TrainingEngine` is used for all worker counts. A serial run is
`--parallel-envs 1`.

```powershell
.\.venv\Scripts\python.exe train.py --config configs\e1\single_flow.json --smoke --parallel-envs 1 --run-name flow_smoke
.\.venv\Scripts\python.exe train.py --config configs\e1\single_cost.json --smoke --run-name cost_smoke
.\.venv\Scripts\python.exe train.py --config configs\e1\single_variance.json --smoke --run-name variance_smoke
```

Resume a checkpoint produced by the current architecture with:

```powershell
.\.venv\Scripts\python.exe train.py --config configs\e1\single_flow.json --initial-checkpoint result\runs\v7_2000_e1_seed11\accepted_checkpoint.pt --run-name flow_resume
```

Checkpoint loading is strict and requires the pair-plus-WAIT/schema-4 network spec.

## Evaluate

```powershell
.\.venv\Scripts\python.exe eval.py --config configs\e1\single_flow.json --dataset validation --policy ppo --checkpoint result\runs\v7_2000_e1_seed11\accepted_checkpoint.pt
```

PPO evaluation defaults to the formal stochastic policy (`sampled`,
temperature 1.0). Validation, checkpoint promotion, independent audit, and
final test use disjoint deterministic seed namespaces derived from the
algorithm seed: `+100000`, `+200000`, and `+300000`, respectively. Specialist
promotion objectives are computed only from completed, non-truncated sampled
rollouts; failures and truncations are enforced by the completion/safety gate.
Use `--decode-mode greedy` only for diagnostic or ablation runs.

## MO-ALNS and offline Pareto analysis

```powershell
.\.venv\Scripts\python.exe mo_alns.py --config configs\baselines\mo_alns_smoke.json --dataset validation --instance-limit 1 --parallel-envs 1
.\.venv\Scripts\python.exe mo_alns_benchmark.py --manifest configs\baselines\mo_alns_manifest.json --output-dir result\analysis\mo_alns
.\.venv\Scripts\python.exe pareto_analysis.py --help
.\.venv\Scripts\python.exe mo_alns_analysis.py --help
```

Pareto and hypervolume computations are offline evaluation tools; they do not
control PPO checkpoint promotion.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The latest-only audit checks configuration identity, environment golden
outputs, pair and WAIT contracts, strict checkpoint loading, PPO updates,
serial/parallel reproducibility, and the MO-ALNS/offline Pareto path.
