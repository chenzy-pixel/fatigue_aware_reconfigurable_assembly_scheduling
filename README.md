# Fatigue-aware reconfigurable assembly scheduling

## Ablation runs

The seed11 ablation matrix has two Universal network variants (node MLP with
pooling; shared preference-conditioned actor head) and three matched full-fatigue /
fatigue-neutral single-objective pairs (Flow, Cost, Variance). The fatigue-neutral simulator
uses base reconfiguration durations and does not restrict actions by fatigue;
an independent audit reconstructs fatigue exposure from executed worker tasks.
`configs/manifests/ablation_seed11.json` defines all nine training runs and five
comparisons. `run_12_train_neutral.py` trains both members of each fatigue pair,
using frozen scales, failure penalty 2, and matching PPO/validation budgets.
Evaluation and summarization validate the training snapshots, selected checkpoints,
dataset hashes, and complete sampling cells before reusing results. See
[the experiment protocol](docs/experiment_protocol.md) for the comparison rules.

```powershell
python scripts/run_10_ablation_smoke.py
python scripts/run_11_train_structural.py
python scripts/run_12_train_neutral.py
python scripts/run_13_evaluate_ablations.py
python scripts/run_14_summarize_ablations.py
```

The last command writes paired results to `result/analysis/ablation_seed11/`
and the three structural methods' V8 Pareto/HV results to its `pareto/` subdirectory.
The full training commands are long-running and can be launched separately.

This repository trains a preference-conditioned HGNN policy with PPO for
fatigue-aware reconfigurable assembly scheduling. Production and worker
decisions use pair-plus-WAIT actions, exact action masks, deterministic event
simulation, and schema-10 heterogeneous graph observations. Order slack, worker-task
waiting age, and event-projected WAIT slack enter each objective expert's context
score through the action encoder. See [the time-context definition](docs/order_time_context.md).

The nine global features encode time, pending reconfigurations, completed
operations, decision phase, worker matching deficit, minimum worker alternatives,
and cumulative Flow, cost and committed-load variance divided by their frozen
scales. PPO requires `network.dropout=0`.

Attributed HGNN messages apply `ReLU(Linear([neighbor, edge]))` before
aggregation in both directions. Messages are averaged over the total incoming
degree; residual updates and per-node-type graph pooling follow the current
network contract. The generated computation identity is
`message_function=attributed_joint_relu_v1` and
`message_aggregation=total_degree_mean_v1`. The node-MLP ablation records both
fields as `not_applicable`. See [the implementation and acceptance record](docs/joint_messages_20261006.md).

## Confirmed main-experiment protocol (2026-10-02)

The agreed objective scales are **Flow=1089.15, Cost=353.27,
worker-load variance=2.2629**. They are the user-selected rounded V2 continuation
validation references at stage episodes 500, 500, and 360. Current configurations
share the versioned manifest in `configs/manifests/`. See the
[time-context contract](docs/order_time_context.md) for soft-estimate semantics
and the schema-10 checkpoint boundary under a matching normalization hash.

Universal training validation runs **every 100 episodes** on **50 fixed
instances × 3 sampled repeats × 13 fixed preferences = 1950 trajectories**.
The 13 preferences comprise three endpoints, three edge midpoints, the equal-weight
center, and all six permutations of `(0.6, 0.3, 0.1)`.
Final evaluation reloads the selected best checkpoint and uses the **66-point
step-0.1 simplex grid** on the test set with three repeats per instance/preference.

`configs/default.json` defines this protocol; `configs/v8/universal.json` is its
public alias. The full ordered preference list and scale provenance are in
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
`I_t=1` only on a task-failure terminal step and is zero otherwise. The global
default is `lambda=2.0`, inherited by E1, Universal, and MO-ALNS configurations.
The configured failure penalty must be finite and non-negative. Evaluation
continues to expose the formal failure quality bound separately; it is not used
by the failure-v3 training reward.

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
| Deadlock, task horizon | measured quality | configured lambda once | yes | no |
| Engineering decision guard or rollout cutoff | current quality | 0 | no | yes |

At a horizon boundary, the simulator first processes every event at the target
tick and checks completion. A final operation completing exactly at the horizon
is therefore a successful task.

Runs persist `reward.mode=single_stage_progress_quality_failure_v3` and
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

Historical experiment outputs were compacted on 2026-10-03. Their CSV logs and
evaluation tables remain in the original directories; `result/runs_summary.csv`
provides a compact overview of recorded run settings, selected validations, and
final sampled metrics. These directories are result archives; checkpoint-based
evaluation or resumed training requires a newly generated checkpoint.

## Setup

```powershell
& C:\Users\chenz\myenv\Scripts\Activate.ps1
python -m pip check
```

The project `.venv` directory links to the shared environment at
`C:\Users\chenz\myenv`. Existing `.venv\Scripts\python.exe` commands use
PyTorch 2.14.1 with CUDA 13.2. Install package updates in that shared environment.

For a fresh checkout, create a Python 3.12 environment and install the pinned
dependencies with `python -m pip install -r requirements.txt`. Tests read the
committed fixed-instance YAML directly.

## Train

All worker counts use the same `TrainingEngine`; `--parallel-envs 1` selects a
serial collector.
Use `--episodes-per-update` to keep the PPO episode count fixed while changing
training workers, and `--validation-parallel-envs` to tune validation separately.
All three single-objective configurations use 20 training workers, 20
validation workers, and 20 episodes per PPO update.

```powershell
.\.venv\Scripts\python.exe train.py --config configs\e1\single_flow.json --smoke --parallel-envs 1 --run-name flow_smoke
.\.venv\Scripts\python.exe train.py --config configs\e1\single_flow.json --algorithm-seed 11 --run-name flow_seed11
.\.venv\Scripts\python.exe -m scripts.run_00_smoke
```

The smoke entry point also supports `python scripts/run_00_smoke.py`.
Saved `config.json` files can be passed back through `--config`; generated runtime
identity is checked when loading a snapshot. In-repository normalization paths
remain relative to the checkout and their pinned hashes are verified.

An explicit compatible checkpoint can initialize network weights:

```powershell
.\.venv\Scripts\python.exe train.py --config configs\e1\single_flow.json --initial-checkpoint result\runs\source\best_checkpoint.pt --run-name flow_initialized
```

Training initialization and evaluation require schema-10 checkpoints. Schema
5/6/7/8/9 models and effective-config snapshots require retraining; the retained
`--allow-observation-migration` CLI option cannot bypass this boundary.
Checkpoint loading also requires the current message-computation identity.
Schema-10 models and snapshots created before the joint-message change require
retraining, even though parameter names, shapes, and counts are the same.
New checkpoints restore compatible network and optimizer state normally.

Engineering decision guards stop sampling with `terminated=false` and
`truncated=true`, preserving the physical observation and action mask. PPO
bootstraps the final value and starts a new instance. Physical success and
failure use `terminated=true`, with explicit `task_succeeded` / `task_failed`
fields. An externally truncated evaluation records all cells and coverage,
and is ineligible for checkpoint selection or Pareto/HV analysis.
See [schema-8 behavior and validation](docs/schema8_global_observation.md).
Schema 9 adds bidirectional `processing_on` and `served_by` relations for actual
allocations, with zero edge attributes and the same nine global features.
Schema 10 derives candidate time, cost, and load variance from one safe sequential
worker path, honoring existing machine commitments and continuous service fatigue.
See [the current projection contract and verification](docs/schema10_sequential_projection.md).

E1 Flow, Cost, and Variance train for 1,000 episodes and validate every 40
episodes. Each uses 20 training/validation workers and 20 episodes per update.
The default Universal run contains 2,000 training episodes.
Universal validates every 100 episodes on the ordered 13-point set and runs
the 66-point final test after reloading its selected best checkpoint.
Its training and validation worker counts are both 20 in the effective config.
The Universal launcher `scripts/run_v8_universal.ps1` defaults to seed 11 and
`configs/v8/universal.json`. Pass `-Seeds 11,23,37` for a selected seed list; runs
execute sequentially. Each batch appends a timestamp to its run names.

```powershell
.\scripts\run_v8_universal.ps1 -DryRun
.\scripts\run_v8_universal.ps1
.\scripts\run_v8_universal.ps1 -Seeds 11,23,37
.\scripts\run_v8_universal.ps1 -Smoke
```

The launcher uses the project virtual environment when present, otherwise the
active `python` command. Pass `-Python` with an executable path to select an
interpreter. `-DryRun` prints the launch commands, and `-Smoke` forwards the
training smoke option. A failed run stops the batch. Run directories follow
`result/runs/v8_universal_seed11_<batch timestamp>`; use that generated directory
for the checkpoint paths in the evaluation commands below.

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
.\.venv\Scripts\python.exe -m scripts.mo_alns --config configs\baselines\mo_alns.json --dataset test --algorithm-seed 11
.\.venv\Scripts\python.exe -m analysis.pareto_analysis --candidate-csv result\runs\v8_universal_seed11\final_sampled_instance_metrics.csv --output-dir result\analysis\universal
```

V8 final evaluation and training-run directories can be analyzed directly:

```powershell
python -m analysis.pareto_analysis --run-dir result/runs/universal_seed11_ep2000 --output-dir result/analysis/v8_pareto
python -m analysis.pareto_analysis --run-dir main=result/runs/ablation_eval_universal_seed11 --run-dir no_graph=result/runs/ablation_eval_no_graph_seed11 --run-dir shared_head=result/runs/ablation_eval_shared_head_seed11 --output-dir result/analysis/v8_comparison
```

The first command uses the run produced by `run_11_train_structural.py`; for launcher
runs, substitute their generated directory name. Each V8 run must contain
the complete 66-point final grid and its recorded sampled repeats. The analysis
checks dataset, scales, repeat budgets, training seeds, and sampling identity;
it writes per-instance fronts, pooled and per-repeat HV, completion rates, paired
coverage, and training-seed summaries. Repeated `--run-dir main=...` arguments group
independent training seeds for one method. All methods use `J/(scale+J)` with the
recorded frozen scales and HV reference `(1,1,1)`.

For a hash-verified sampled trajectory replay with per-decision environment
snapshots and optional bounded branch continuations, use
`python -m scripts.deadlock_replay --help`.

The test suite covers reward telescoping, fixed progress denominators,
termination/bootstrap semantics, horizon-boundary completion, deterministic
serial/parallel rollout, checkpoint ranking, preference-balanced aggregation,
manifest integrity, and checkpoint compatibility.

## Ablation experiments

The five matched variants and their training, evaluation, and pairing contracts are
documented in [the ablation protocol](docs/ablation_protocol.md). Structural variants
use the Universal budget; fatigue-neutral variants use their single-objective parent
budgets. All inherit the current V2 data, frozen scales, and failure penalty 2.0.

```powershell
python scripts/run_10_ablation_smoke.py
python scripts/run_11_train_structural.py
python scripts/run_12_train_neutral.py
python scripts/run_13_evaluate_ablations.py
python scripts/run_14_summarize_ablations.py
```

The training entry points default to seed 11. Evaluation requires the matched Universal
and three full-fatigue single-objective baselines, checks protocol compatibility, and
saves an evaluation manifest. Summarization uses that manifest to pair identical
instance/preference/repeat cells.

Completed reconfiguration densities use actual DONE reconfigurations and actual completed operations. Partial installations contribute executed work and fatigue exposure; unfinished installations contribute no completed reconfiguration. Zero denominators are reported as null.
