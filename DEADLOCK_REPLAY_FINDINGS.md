# Formal sampled failure attribution

## Reproduction scope

- Run: `result/runs/single_stage_flow_seed11_500_20260922_013711`
- Checkpoint SHA-256: `2ad18327583372b8f895b3bf9ee8e3b971eded3c0116b5973afdaa1e7a550b41`
- Evaluation role: formal final sampled evaluation
- Decode temperature: `1.0`
- Failure count: 5 of 60 trajectories

All five CUDA replays match the recorded instance ID, sampling seed, derived
seed, decision count, progress, and action-trace SHA-256. The diagnostic keeps
two separate facts:

1. whether the current schedule can make its next deterministic progress by
   the horizon; and
2. whether all remaining work is structurally recoverable after the horizon.

A post-horizon event establishes the first fact only. Structural
recoverability is reported as `not_assessed_beyond_horizon` unless separately
proved by a continuation.

## Five-failure attribution

| Instance | Seed | First inability (min) | Progress | Remaining | Next deterministic events (min) | Formal label | Primary cause | Structural status |
|---|---:|---:|---:|---:|---|---|---|---|
| `test_reconfiguration_bottleneck_3000009` | 300011 | 237.7 | 0.8639 | 10 ops / 6 orders | process completions at 243.7, 247.4, 248.3 | `unrecoverable_deadlock` | `horizon` | Three additional reconfigurations are waiting for fatigue-safe workers; later completion is not established. |
| `test_reconfiguration_bottleneck_3000002` | 300012 | 230.7 | 0.9375 | 4 ops / 1 order | `O_R12_1` completes at 242.7 | `unrecoverable_deadlock` | `horizon` | No pending reconfiguration at detection; successors remain precedence-blocked. |
| `test_reconfiguration_bottleneck_3000009` | 300012 | 234.9 | 0.9861 | 1 op / 1 order | `O_R14_4` completes at 247.5 | `unrecoverable_deadlock` | `horizon` | No pending reconfiguration at detection. |
| `test_balanced_3000019` | 300012 | 226.8 | 0.9833 | 1 op / 1 order | `O_R15_4` completes at 241.2 | `unrecoverable_deadlock` | `horizon` | No pending reconfiguration at detection. |
| `test_reconfiguration_bottleneck_3000002` | 300013 | 236.7 | 0.9250 | 5 ops / 3 orders | process completions at 241.5, 244.5, 246.8 | `unrecoverable_deadlock` | `horizon` | No pending reconfiguration at detection; two successor operations remain precedence-blocked. |

The primary label is `horizon` for all five because no event remains within
the time budget and the next deterministic event occurs after it. This does
not collapse the mixed case in seed 300011: its three waiting
reconfigurations remain visible as concurrent resource/fatigue blockers.

## Remaining-order time decomposition

Times below are measured through the horizon. `Remaining process LB` is the
sum of active-operation overrun plus the fastest compatible-machine duration
for operations not yet started. It excludes future reconfiguration and worker
waiting, so it is a lower bound rather than a feasibility certificate.

| Instance / seed | Order | Remaining ops | Processing elapsed | Reconfiguration active | Waiting for worker | Remaining process LB |
|---|---|---:|---:|---:|---:|---:|
| 3000009 / 300011 | R11 | 1 | 34.6 | 24.6 | 119.3 | 3.7 |
| 3000009 / 300011 | R12 | 2 | 23.0 | 11.1 | 126.9 | 15.9 |
| 3000009 / 300011 | R15 | 1 | 26.6 | 7.8 | 26.1 | 7.1 |
| 3000009 / 300011 | R16 | 1 | 42.7 | 10.5 | 2.5 | 8.3 |
| 3000009 / 300011 | R14 | 2 | 22.2 | 24.6 | 48.7 | 20.3 |
| 3000009 / 300011 | R18 | 3 | 13.7 | 11.1 | 103.0 | 30.3 |
| 3000002 / 300012 | R12 | 4 | 9.3 | 12.4 | 127.6 | 35.5 |
| 3000009 / 300012 | R14 | 1 | 40.1 | 14.5 | 49.6 | 7.5 |
| 3000019 / 300012 | R15 | 1 | 39.0 | 11.7 | 0.0 | 1.2 |
| 3000002 / 300013 | R10 | 2 | 28.7 | 23.4 | 121.3 | 15.8 |
| 3000002 / 300013 | R16 | 1 | 55.4 | 9.8 | 59.1 | 6.8 |
| 3000002 / 300013 | R12 | 2 | 33.4 | 11.1 | 43.1 | 13.4 |

Four failures occur on reconfiguration-bottleneck instances: 3000002 fails
in two of three sampled repeats and 3000009 fails in two of three. Across the
fixed test manifest, the reconfiguration-bottleneck class fails 4 of 9
sampled trajectories (44.4%). Balanced fails 1 of 21 (4.8%); every other
pressure class completes all sampled trajectories. The recurring signature
is accumulated worker/fatigue waiting followed by processing that starts too
late, rather than a common terminal resource cycle.

## R12 wait chain in 3000002 / seed 300012

At decision 159 (tick 907, 90.7 minutes), the policy commits `O_R12_1` to
`M2`, requiring `A1 -> A3` reconfiguration. This is the only legal R12 action
at that state. The complete chain is:

| Segment | Tick interval | Time |
|---|---|---:|
| Wait for a fatigue-safe A1 disassembly worker | 907–2183 | 127.6 |
| Disassembly by H3 | 2183–2246 | 6.3 |
| Installation by H6 | 2246–2307 | 6.1 |
| Processing executed before horizon | 2307–2400 | 9.3 |
| Processing overrun to planned completion | 2400–2427 | 2.7 |

At commitment time, H1 and H3 are idle but their predicted post-task fatigue
is 0.8046 and 0.8036, above the 0.75 safety limit; H2 is busy. During every
worker decision before tick 2183, assigning a worker to R12 remains masked by
fatigue safety. At tick 2183, H3 reaches a predicted fatigue of exactly 0.75
and becomes the first legal assignment.

The policy does assign H1/H2/H3 to other reconfigurations during this
interval, but those assignments are not alternatives for R12 at the same
worker decision: R12 is still fatigue-masked. The actionable alternatives
occur earlier, when production commitments determine later fatigue demand.
At decision 159 the legal choices are commitments for R8, R9, R14, R16, the
R12 commitment, or WAIT; there is no already-configured machine available for
direct R12 processing.

## Decision-160 paired continuation

The paired experiment starts from the identical snapshot at decision 160,
tick 907, progress 0.403125. Each row uses the same continuation seed for both
branches:

- Original: `COMMIT_RECONFIG O_R8_2 -> M1`
- Replacement: `COMMIT_RECONFIG O_R14_1 -> M1`
- Continuation seeds: 960000 through 960007

| Outcome | Original | Replacement |
|---|---:|---:|
| Completed | 2 / 8 | 1 / 8 |
| Mean final progress | 0.9559 | 0.9629 |
| Mean environment wait time | 231.29 | 234.33 |
| Mean machine waiting-for-worker time | 204.85 | 232.00 |
| Mean Flow among successful continuations | 1348.15 | 1475.90 |

Pair outcomes are: one original-only success, zero replacement-only
successes, one both-success, and six both-fail. The original branch succeeds
under seeds 960001 and 960006; the replacement succeeds only under 960006.
Decision 160 is therefore a proven recoverable state, but this experiment
does not identify it as a last recoverable state or establish the replacement
as an improvement. Common random-number seeds reduce avoidable sampling
variation, while the two post-action state distributions still diverge.

## Flow reward audit

The environment accumulates Flow continuously. During every time advance,
each released order that has not completed contributes one unit per minute to
`_flow_integral`. In the R12 trace, R12 is released and committed at 90.7
minutes, so it contributes continuously through the 240-minute horizon. At
failure, the environment also adds the configured 240-point unfinished-order
penalty.

For 3000002 / seed 300012:

- flow integral: 1527.7;
- unfinished-order penalty: 240.0;
- diagnostic Flow objective: 1767.7;
- sum of raw `flow` reward component: -1767.7;
- recorded v1 sum of quality deltas after terminal failure settlement: -1.0;
- operation-progress return: +0.9375;
- recorded v1 scalar training return: -0.0625.

The general return identity has zero numerical residual:

`0.9375 - 0.0 - 1.0 + 0.0 = -0.0625`.

Thus R12 delay was fully visible in the raw Flow metric and in intermediate
quality potentials, but the recorded v1 terminal overwrite made every failed
trajectory telescope to quality `-1`. The failure-v2 reward now retains the
actual terminal quality and applies a separate one-shot penalty of 1. Replay
output reports both the recorded v1 return and the failure-v2 reconstruction,
including base return, scalar training return, and identity residual.

## Diagnostic implementation

The environment records both the primary failure label and the independent
evidence fields: within-horizon event count, post-horizon event count, next
event tick/type, horizon-overrun evidence, and structural-recoverability
status. `deadlock_replay.py` additionally records full snapshots, schedule and
reconfiguration logs, per-order timing, reward-identity audit, and paired
continuations.

Example paired command:

```powershell
python deadlock_replay.py `
  --config result/runs/single_stage_flow_seed11_500_20260922_013711/config.json `
  --checkpoint result/runs/single_stage_flow_seed11_500_20260922_013711/best_checkpoint.pt `
  --dataset test `
  --instance-id test_reconfiguration_bottleneck_3000002 `
  --sampling-seed 300012 `
  --device cuda `
  --expected-action-trace-sha256 73dee8995402f0e8ed36a22096261b6e5a040f473e81635ed5e373beaaf3e46d `
  --paired-decision-index 160 `
  --paired-alternative-action 376 `
  --paired-seed-start 960000 `
  --paired-seed-count 8 `
  --output result/analysis/formal_failure_test_reconfiguration_bottleneck_3000002_seed300012_paired.json
```

Successful bounded continuations prove recoverability at their branch
snapshot. Unsuccessful bounded searches remain inconclusive.
