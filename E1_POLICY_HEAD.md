# E1 bounded-residual policy head

The mainline v7 policy head keeps policy-head v6 checkpoint loading intact and
adds the E1 context-exception mechanism only. It does not enable E2--E6 or
forced-action compression.

E1 combines the monotone production and worker rankers with a bounded learned
residual:

```text
sigmoid(gate) * residual_scale_ratio
* max(std(relative_logits), 1e-3)
* tanh(raw_residual)
```

The default E1 configuration uses `gate=-2`, `residual_scale_ratio=2`, and
disables the worker common offset when it cannot affect pair-vs-advance
competition. C0 retains the v6 policy head under the same guarded checkpoint
protocol.

## Configurations

- `configs/v7/c0_v6_control.json`: same-protocol v6 control.
- `configs/v7/e1_context_exception.json`: complete E1 treatment.

Both configurations keep `training.forced_action_compression=false` and
`non_delay_worker_dispatch=true`.

## Evaluation protocol v2

`v7_e1_protocol_v2` reports paper quality with the immutable
`canonical_bounded_quality_v1` metric: flow/cost/variance scales are
`1200/1000/50` and weights are `0.5/0.3/0.2`. Training reward scales remain
available as `reward_quality_score` diagnostics and cannot change the reported
`quality_score`. Results use schema `4.1.0` and reject mixed quality-metric
hashes.

Sampled evaluation uses `per_instance_sha256_v1`. Each instance owns a Torch
generator derived from its stable instance ID and the requested sampling seed,
so `parallel_envs=1/2/10` consumes identical random streams. Instance rows
include `action_trace_sha256` for direct audit.

Checkpoint metadata, training summaries, and evaluation metrics carry the
same provenance fields for source, effective config, dataset manifest,
fixed/template data, Git state, protocol, evaluator, and checkpoint hashes.
Legacy checkpoints remain loadable; their embedded protocol is reported
separately from the current evaluator protocol.

## Smoke checks

```powershell
.\.venv\Scripts\python.exe -m pytest test/test_v7_policy.py test/test_v7_experiments.py test/test_e1_protocol_v2.py -q
.\.venv\Scripts\python.exe train.py --config configs/v7/e1_context_exception.json --smoke --run-name e1_mainline_smoke
```

The fixed runtime instance is stored at `data/instances/fixed_instance.yaml`.
Development-only notes live under `docs/dev_context/` and are not runtime
inputs.
