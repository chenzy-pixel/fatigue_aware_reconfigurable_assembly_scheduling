# Fatigue-aware reconfigurable assembly scheduling

This repository contains the accepted E1 research implementation for
fatigue-aware reconfigurable assembly scheduling.

The executable stack is fixed to:

- pair-plus-defer production actions;
- V7 HGNN actor-critic with bounded ranker-scale context residual;
- deterministic temporal matching admission/recovery v3;
- completion-viability defer shield v2;
- single-objective guarded promotion protocol v4.

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
- `environment/`: facade, runtime state, action codec, dynamics, temporal
  contracts, graph observation, reward, and diagnostics.
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

Resume the current accepted architecture with:

```powershell
.\.venv\Scripts\python.exe train.py --config configs\e1\single_flow.json --initial-checkpoint result\runs\v7_2000_e1_seed11\accepted_checkpoint.pt --run-name flow_resume
```

Checkpoint loading is strict. The current accepted checkpoint is supported;
historical network specs should be run from the archive tag.

## Evaluate

```powershell
.\.venv\Scripts\python.exe eval.py --config configs\e1\single_flow.json --dataset validation --policy ppo --checkpoint result\runs\v7_2000_e1_seed11\accepted_checkpoint.pt
```

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
outputs, temporal oracle boundaries, strict checkpoint loading, PPO updates,
serial/parallel reproducibility, and the MO-ALNS/offline Pareto path.
