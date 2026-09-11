# V8 preference-conditioned multi-objective policy

V8 keeps the six node types, twelve relations, and two HGNN message-passing layers. The episode preference is a separate three-value observation field and is encoded by `3 -> 32 -> ReLU -> 32`; it is not appended to graph global features.

Production, Worker, and WAIT each use Flow, Cost, and Variance experts. Every direct ranker is bias-free and fixed-sign, with `softplus(theta)` weights normalized to a simplex. Direct and context outputs are bounded to `[-1, 1]`; expert outputs are therefore bounded to `[-2, 2]`. The base logit is `lambda dot z`. A phase-level preference-conditioned residual is scaled by the standard deviation of legal base logits and uses a gate initialized at zero logit.

Quality reward is the exact telescoping difference of the normalized augmented Tchebycheff scalarizer. A completed trajectory uses its measured terminal scalarized objective; any truncated trajectory uses the common terminal failure bound `T_terminal = 1`, so every endpoint receives the same hard-feasibility signal. Feasibility shaping remains observable in diagnostics, but quality PPO return contains only the scalarizer difference. `gamma` remains one.

## Reproducible workflow

1. Run `run_v8_specialists.ps1` to train Flow, Cost, and Variance specialists at seeds `11/23/37/53/71`.
2. Every specialist accepted checkpoint is produced only after its independent 200-instance audit and records the endpoint raw-objective mean plus audit-manifest hash.
3. Run `v8_normalization.py --specialist-runs-root result/runs --audit-dataset-manifest <manifest.json> --output <normalization.json>`. The command verifies all 15 V8 checkpoint identities, seeds, endpoint names, and the shared `offset=50,count=200` audit selection and manifest hash. `--specialist-audits <rows.json>` remains available for explicitly assembled records that contain the same audit provenance fields. The destination is created exclusively and cannot overwrite an existing manifest.
4. Copy `configs/v8/universal.json.template` to a run config and replace the manifest path and file SHA256. Loading verifies the hash, freezes all three scales, records the content hash, and imports the specialist endpoint prediction bounds.
5. Run `run_v8_universal.ps1 -Config <frozen-universal-config.json>` to train seeds `11/23/37/53/71`. Formal runs fail before training unless the manifest, 50/200 instance protocol, 66-point grid, and endpoint bounds are verified. Quality preferences follow deterministic 20-episode blocks: two copies of every endpoint plus 14 seeded scrambled-Sobol exponential-simplex points.

Validation uses manifest rows `0:50` times all 66 step-0.1 simplex preferences. Promotion first applies completion and safety gates, then endpoint prediction bounds, an instance-block paired bootstrap of mean scalarized score, and—only when its confidence interval contains zero—an instance-block paired bootstrap of per-instance hypervolume. Both bootstraps use 10,000 deterministic resamples. A validation winner is first saved and reloaded, then audited on the non-overlapping manifest rows `50:250` times 66 preferences before an accepted checkpoint is written.
