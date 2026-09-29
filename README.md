# Fatigue-aware reconfigurable assembly scheduling

This repository trains a preference-conditioned HGNN policy with PPO for
fatigue-aware reconfigurable assembly scheduling. Production and worker
decisions use pair-plus-WAIT actions, exact action masks, deterministic event
simulation, and schema-5 heterogeneous graph observations.

## Confirmed main-experiment protocol (2026-09-28)

The agreed objective scales are **Flow=1152.2093959731544,
Cost=386.674652792805, worker-load variance=4.937746913580247**.
Each scale is the median of the corresponding objective's successful-trajectory
means at the final five validations (episodes 840, 880, 920, 960, and 1000).
Flow uses the latest relative-worker-time run; Cost and variance use their
respective 1000-episode single-objective runs. All main-method and structural ablation
runs share these frozen scales.

Universal training validation runs **every 100 episodes** on **50 fixed
instances × 3 sampled repeats × 13 fixed preferences = 1950 trajectories**.
The 13 preferences comprise three endpoints, three edge midpoints, the equal-weight
center, and all six permutations of `(0.6, 0.3, 0.1)`.
Final evaluation reloads the selected best checkpoint and uses the **66-point
step-0.1 simplex grid** on the test set with three repeats per instance/preference.

The full ordered preference list and scale provenance are in
[the experiment protocol](docs/experiment_protocol.md). The Universal model uses
`candidate_zscore_v1` for worker Flow time with a standard-deviation floor of
`0.001`; its scales are loaded from the checked manifest in `configs/manifests/`.

## Reward contract

Every PPO entry point uses one reward throughout training:

\[
r_t=(P_{t+1}-P_t)+(Q_t-Q_{t+1})-\lambda I_t.
\]

For an instance with all orders fixed at reset,

\[
P_t=\frac{1}{N}\sum_{i=1}^{N}\frac{c_i(t)}{n_i},
\]

where only `DONE` operations contribute to `c_i(t)`. Unreleased orders remain
in the denominator, and order release does not change progress. `Q_t` is the
existing bounded preference quality score computed from the actual objectives.
`I_t=1` only on a task-failure terminal step and is zero otherwise.
`lambda` is the finite, non-negative `reward.terminal_failure_penalty`:
the default template uses `1.0`; all E1 configs and Universal use `5.0`. Evaluation
continues to expose the formal failure quality bound separately; it is not used
by the failure-v2 training reward.

With `gamma=1`, every trajectory is checked against the general identity

\[
G=P_T-P_0+Q_0-Q_T-\lambda I.
\]

Standard from-scratch resets assert `P_0=Q_0=0`. The feasibility potential and
its resource calculations remain available for experiments and diagnostics;
the published configurations set its reward coefficient path to disabled.

## Termination and PPO bootstrap

| Outcome | Training quality | Failure penalty | Transition `done` | Critic bootstrap |
|---|---:|---:|---:|---:|
| All operations complete | measured quality | 0 | yes | no |
| Deadlock, task horizon, environment decision limit | measured quality | configured lambda once | yes | no |
| Rollout collection cutoff | current quality | 0 | no | yes |

At a horizon boundary, the simulator first processes every event at the target
tick and checks completion. A final operation completing exactly at the horizon
is therefore a successful task.

Runs persist `reward.mode=single_stage_progress_quality_failure_v2` and
the configured `reward.terminal_failure_penalty` in the effective config and runtime
manifest. Episode/evaluation rows record actual quality, failure penalty, base
cumulative reward, and scalar training reward so returns can be reconstructed
without reading the evaluation-only failure quality bound.

## Checkpoint selection

Formal validation uses sampled decoding at temperature `1.0`. Checkpoints are
ranked lexicographically:

1. higher sampled completion rate;
2. lower preference-balanced quality when completion ties within `1e-12`.

Under the confirmed main-experiment protocol, Universal checkpoint selection
uses the minimum completion over the fixed 13-point validation set. Quality is averaged within each preference over successful
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
- `analysis/`: reusable Pareto and experiment analysis.
- `scripts/`: baseline, audit, diagnostic, and batch-run entry points.
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
Use `--episodes-per-update` to keep the PPO episode count fixed while changing
training workers, and `--validation-parallel-envs` to tune validation separately.
The relative-worker-time Flow configuration uses 40 training workers, 40
validation workers, and 40 episodes per PPO update.

```powershell
.\.venv\Scripts\python.exe train.py --config configs\e1\single_flow.json --smoke --parallel-envs 1 --run-name flow_smoke
.\.venv\Scripts\python.exe train.py --config configs\e1\single_flow.json --algorithm-seed 11 --parallel-envs 20 --run-name flow_seed11
.\.venv\Scripts\python.exe -m scripts.run_00_smoke
```

An explicit compatible checkpoint can initialize network weights:

```powershell
.\.venv\Scripts\python.exe train.py --config configs\e1\single_flow.json --initial-checkpoint result\runs\source\best_checkpoint.pt --run-name flow_initialized
```

E1 Flow, Cost, and Variance validate every 40 episodes. The default formal run
contains 2,000 training episodes.
Universal validates every 100 episodes on the ordered 13-point set and runs
the 66-point final test after reloading its selected best checkpoint.
Its training and validation worker counts are both 20 in the effective config.
The five-seed batch entry point is `scripts/run_v8_universal.ps1`, which loads
`configs/v8/universal.json` by default.

## Evaluate

```powershell
.\.venv\Scripts\python.exe eval.py --config configs\e1\single_flow.json --dataset test --policy ppo --checkpoint result\runs\flow_seed11\best_checkpoint.pt
.\.venv\Scripts\python.exe eval.py --config configs\v8\universal.json --dataset validation --policy ppo --checkpoint result\runs\v8_universal_seed11\best_checkpoint.pt --preference-set validation
.\.venv\Scripts\python.exe eval.py --config configs\v8\universal.json --dataset test --policy ppo --checkpoint result\runs\v8_universal_seed11\best_checkpoint.pt --preference-set final_test
```

Sampled validation seeds use `algorithm_seed + 100000 + repeat`; independent
final-test seeds use `algorithm_seed + 300000 + repeat`. Each instance and
preference receives its own SHA256-derived Torch generator seed.

## Tests and analysis

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pytest -q --runslow -m slow
.\.venv\Scripts\python.exe -m analysis.single_objective_analysis result\runs\flow_seed11 --plots
```

For a hash-verified sampled trajectory replay with per-decision environment
snapshots and optional bounded branch continuations, use
`python -m scripts.deadlock_replay`. The investigated Flow failure and the evidence standard
for branch results are documented in [DEADLOCK_REPLAY_FINDINGS.md](docs/DEADLOCK_REPLAY_FINDINGS.md).

The test suite covers reward telescoping, fixed progress denominators,
termination/bootstrap semantics, horizon-boundary completion, deterministic
serial/parallel rollout, checkpoint ranking, preference-balanced aggregation,
manifest integrity, and checkpoint compatibility.
