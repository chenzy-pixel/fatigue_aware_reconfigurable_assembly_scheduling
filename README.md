# Fatigue-aware reconfigurable assembly scheduling

This repository trains a preference-conditioned HGNN policy with PPO for
fatigue-aware reconfigurable assembly scheduling. Production and worker
decisions use pair-plus-WAIT actions, exact action masks, deterministic event
simulation, and schema-5 heterogeneous graph observations.

## Reward contract

Every PPO entry point uses one reward throughout training:

\[
r_t=(P_{t+1}-P_t)-(Q_{t+1}-Q_t).
\]

For an instance with all orders fixed at reset,

\[
P_t=\frac{1}{N}\sum_{i=1}^{N}\frac{c_i(t)}{n_i},
\]

where only `DONE` operations contribute to `c_i(t)`. Unreleased orders remain
in the denominator, and order release does not change progress. `Q_t` is the
existing bounded preference quality score. A successful task uses its measured
terminal quality; a failed task uses `Q_T=1`.

With `gamma=1`, every trajectory is checked against the general identity

\[
G=P_T-P_0-Q_T+Q_0.
\]

Standard from-scratch resets assert `P_0=Q_0=0`. The feasibility potential and
its resource calculations remain available for experiments and diagnostics;
the published configurations set its reward coefficient path to disabled.

## Termination and PPO bootstrap

| Outcome | Terminal quality | Transition `done` | Critic bootstrap |
|---|---:|---:|---:|
| All operations complete | measured quality | yes | no |
| Deadlock, task horizon, environment decision limit | `1` | yes | no |
| Rollout collection cutoff | current quality | no | yes |

At a horizon boundary, the simulator first processes every event at the target
tick and checks completion. A final operation completing exactly at the horizon
is therefore a successful task.

## Checkpoint selection

Formal validation uses sampled decoding at temperature `1.0`; greedy decoding
is recorded as a diagnostic. Checkpoints are ranked lexicographically:

1. higher sampled completion rate;
2. lower preference-balanced quality when completion ties within `1e-12`.

For the Universal policy, completion is the minimum over the fixed 66-point
preference grid. Quality is averaged within each preference over successful
trajectories, then averaged equally across preferences. A preference without a
successful trajectory gives aggregate quality `+inf`.

The first physically safe, violation-free validation initializes
`best_checkpoint.pt`. Exact ties retain the existing file. Every run writes
`last_checkpoint.pt`; if no validation is safe, `best_checkpoint` is reported
as `null`. Validation manifests, instance order, repeats, preference set,
sampling seeds, seed derivation version, and temperature are stored in
checkpoint metadata and provenance.

## Layout

- `agent/ppo/`: V8 HGNN actor-critic, rollout buffer, PPO update, and parallel collector.
- `configs/`: the Universal configuration plus Flow, Cost, and Variance one-hot overrides.
- `data/`: instance models, deterministic online generation, fixed datasets, and manifests.
- `environment/`: action codec, runtime state, masks, event simulation, reward, and metrics.
- `training/`: completion-first checkpoint selector.
- `result/`: result schema, provenance, logs, checkpoints, and dashboards.
- `docs/ARCHITECTURE.md`: component boundaries and end-to-end call paths.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Train

All worker counts use the same `TrainingEngine`; `--parallel-envs 1` selects a
serial collector.

```powershell
.\.venv\Scripts\python.exe train.py --config configs\e1\single_flow.json --smoke --parallel-envs 1 --run-name flow_smoke
.\.venv\Scripts\python.exe train.py --config configs\e1\single_flow.json --algorithm-seed 11 --parallel-envs 20 --run-name flow_seed11
```

An explicit compatible checkpoint can initialize network weights:

```powershell
.\.venv\Scripts\python.exe train.py --config configs\e1\single_flow.json --initial-checkpoint result\runs\source\best_checkpoint.pt --run-name flow_initialized
```

E1 Flow, Cost, and Variance validate every 40 episodes. The default formal run
contains 2,000 training episodes.

## Evaluate

```powershell
.\.venv\Scripts\python.exe eval.py --config configs\e1\single_flow.json --dataset test --policy ppo --checkpoint result\runs\flow_seed11\best_checkpoint.pt
```

Sampled validation seeds use `algorithm_seed + 100000 + repeat`; independent
final-test seeds use `algorithm_seed + 300000 + repeat`. Each instance and
preference receives its own SHA256-derived Torch generator seed.

## Tests and analysis

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe single_objective_analysis.py result\runs\flow_seed11 --plots
```

For a hash-verified sampled trajectory replay with per-decision environment
snapshots and optional bounded branch continuations, use
`deadlock_replay.py`. The investigated Flow failure and the evidence standard
for branch results are documented in [DEADLOCK_REPLAY_FINDINGS.md](DEADLOCK_REPLAY_FINDINGS.md).

The test suite covers reward telescoping, fixed progress denominators,
termination/bootstrap semantics, horizon-boundary completion, deterministic
serial/parallel rollout, checkpoint ranking, preference-balanced aggregation,
manifest integrity, and checkpoint compatibility.
