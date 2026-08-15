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

## Smoke checks

```powershell
python -m pytest test/test_v7_policy.py test/test_v7_experiments.py -q
python train.py --config configs/v7/e1_context_exception.json --smoke --run-name e1_mainline_smoke
```

The fixed runtime instance is stored at `data/instances/fixed_instance.yaml`.
Development-only notes live under `docs/dev_context/` and are not runtime
inputs.
