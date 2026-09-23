# Deadlock replay findings

## Reproduction target

- Run: `result/runs/single_stage_flow_seed11_500_20260922_013711`
- Checkpoint SHA-256: `2ad18327583372b8f895b3bf9ee8e3b971eded3c0116b5973afdaa1e7a550b41`
- Dataset instance: `test_reconfiguration_bottleneck_3000002`
- Formal sampling seed: `300012`
- Derived sampling seed: `5410848691043856757`
- Expected action trace SHA-256: `73dee8995402f0e8ed36a22096261b6e5a040f473e81635ed5e373beaaf3e46d`

The CUDA replay matches all of the recorded identifiers above and contains 376
decisions. The failure progress is `0.9375`.

## Terminal-state attribution

The state previously labeled `unrecoverable_deadlock` is not a structural
resource deadlock:

- The all-actions-masked state is first reached at tick 2307 (230.7 minutes).
- The horizon is tick 2400 (240.0 minutes).
- There are no pending reconfigurations and no resource cycle.
- The only active event is completion of `O_R12_1` on `M2` at tick 2427.
- `O_R12_2`, `O_R12_3`, and `O_R12_4` are blocked by that predecessor.
- WAIT is masked because the next real event is after the horizon.

This is a horizon failure: the selected schedule cannot complete within the
time budget. Advancing to the horizon and applying failure quality remains
correct, but the termination reason must be `horizon`, not
`unrecoverable_deadlock`.

## Causal replay

In the failed trace, decision 159 at tick 907 commits `O_R12_1` to `M2`, which
requires reconfiguration from `A1` to `A3`. The task then waits until tick 2183
for a safe `A1` disassembly worker. Disassembly completes at tick 2246,
installation completes at tick 2307, and processing would complete at tick
2427.

The action was not provably impossible when selected. At tick 907 the
environment's optimistic resource projection was:

- resource-ready tick: 991
- processing-start tick: 1115
- predicted finish tick: 1235
- horizon slack: 1165 ticks
- safe disassembly workers immediately available: 0
- matching deficit after the commitment: 1

Subsequent worker commitments repeatedly consume or re-fatigue the qualified
`A1` workers, causing the large gap between the optimistic projection and the
actual start.

The successful formal trace for sampling seed `300011` provides a useful
contrast. It directly processes `O_R12_1` on an already configured `A3`
machine (`M3`) at 98.3 minutes. Order `R12` completes at 184.6 minutes and the
whole instance completes at 187.9 minutes.

## Branch evidence

A bounded branch search found successful continuations:

1. At decision 159, replacing `COMMIT_RECONFIG O_R12_1 -> M2` with
   `COMMIT_RECONFIG O_R8_2 -> M1`, followed by sampled continuation seed
   `2532002`, completes the instance.
2. Even after the original decision 159, decision 160 is still recoverable.
   Replacing `COMMIT_RECONFIG O_R8_2 -> M1` with
   `COMMIT_RECONFIG O_R14_1 -> M1`, followed by sampled continuation seed
   `2557601`, completes the instance.

Therefore the last **proven** recoverable point in this replay is decision 160
at tick 907. Bounded searches at decisions 164, 291, 299, 328, and 349 did not
find a successful continuation. That negative result is not proof that every
continuation from those states is infeasible.

## State-information audit

The current observation already exposes the relevant information:

- released and future remaining demand and workload per module;
- installed and target module ratios;
- machine current/target modules and locked reconfiguration edges;
- all capable operation-machine alternatives;
- resource-ready time, predicted finish time, horizon slack, safe-worker
  ratios, and matching deficit for production candidates;
- worker qualifications, fatigue, busy state, and projected assignment
  fatigue.

The replay therefore does not support adding a state feature at this stage.
It also does not justify masking a particular action solely because it appears
in the failed trajectory. The proven defect is the terminal-reason
classification.

## Implemented change

When an all-actions-masked state still has a real event after the horizon, the
environment now classifies the failure as `horizon`. Reward settlement,
failure quality, action masks, event ordering, and the action trajectory are
unchanged.

Verification:

- Focused WAIT/mask/horizon tests: 10 passed.
- Full test suite: 189 passed, 2 skipped.
- Post-fix replay: the same 376 actions and action-trace hash are preserved;
  progress remains `0.9375`; only the reason changes to `horizon`.

## Replay command

```powershell
python deadlock_replay.py `
  --config result/runs/single_stage_flow_seed11_500_20260922_013711/config.json `
  --checkpoint result/runs/single_stage_flow_seed11_500_20260922_013711/best_checkpoint.pt `
  --dataset test `
  --instance-id test_reconfiguration_bottleneck_3000002 `
  --sampling-seed 300012 `
  --device cuda `
  --expected-action-trace-sha256 73dee8995402f0e8ed36a22096261b6e5a040f473e81635ed5e373beaaf3e46d `
  --output result/analysis/deadlock_replay_3000002_seed300012.json
```

Use `--branch-decision-index`, `--branch-ppo-samples`, and
`--branch-ppo-seed-start` for bounded continuation searches. A successful
continuation proves recoverability; absence of one does not prove
infeasibility.
