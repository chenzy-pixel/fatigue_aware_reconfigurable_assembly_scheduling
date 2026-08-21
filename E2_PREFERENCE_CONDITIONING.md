# E2 preference-conditioned multi-objective policy

E2 extends the E1 bounded-residual policy with one episode-level preference
vector `w=(w_flow,w_cost,w_variance)`.  The vector is validated on the
probability simplex, stored in every observation, and encoded by a dedicated
two-layer MLP.  Its embedding is supplied to both pair-scoring actor paths,
the defer/advance heads, and the critic.

## Training distribution

`configs/v7/e2_preference_conditioned.json` uses a reproducible mixture:

- 70% `Dirichlet(1,1,1)` continuous simplex samples;
- 30% uniform samples from the three vertices, the equal-weight centre, and
  the canonical `0.5/0.3/0.2` point.

The local RNG seed is the SHA-256 derivation of the algorithm seed and episode
index.  It does not consume the global Python, NumPy, or Torch RNG and is
independent of `parallel_envs`.  E2 policy sampling uses a separate
episode-derived Torch generator as well, so changing `parallel_envs` does not
change an episode's preference/action pairing.

The quality phase minimizes the bounded episode preference score

```text
Q_w = w_flow * F / (1200 + F)
    + w_cost * C / (1000 + C)
    + w_variance * V / (50 + V)
```

while the feasibility phase remains unchanged.  The paper metric is still the
immutable canonical `0.5/0.3/0.2` score.  Periodic validation and checkpoint
promotion always evaluate E2 at that canonical preference.

## Compatibility

E2 retains policy-head version 7 and records
`preference_conditioning=separate_encoder_v1`, the objective order, and
observation schema 5 in `network_spec`.  E1/E2 checkpoints are intentionally
incompatible.  E2 is trained from scratch; partial E1 weight loading is not a
supported workflow.

## Evaluation

One requested preference can be evaluated directly:

```powershell
.\.venv\Scripts\python.exe eval.py `
  --config configs\v7\e2_preference_conditioned.json `
  --dataset test --policy ppo `
  --checkpoint result\runs\v7_2000_e2_seed11\accepted_checkpoint.pt `
  --preference 1 0 0
```

E2 results use schema `4.2.0` and add `w_flow/w_cost/w_variance` plus
`preference_quality_score`.  `quality_score` remains canonical.

The equal-budget protocol evaluates E2 at the 21 denominator-five simplex
points plus the canonical point.  E1 receives the same budget: one greedy plus
sampled seeds 100001--100021.  Run all five algorithm seeds and fixed
test/OOD/stress splits with:

```powershell
.\.venv\Scripts\python.exe e2_preference_analysis.py `
  --manifest configs\v7\e2_analysis_manifest.example.json `
  --output-dir result\analysis\e1_e2_equal_budget
```

The generated fronts are empirical rollout fronts, not certified true Pareto
fronts.  Hypervolume/diversity/controllability and canonical single-point
quality are reported separately.

## E2.5 safe monotone gate

`configs/v7/e2_5_safe_monotone_flow_gate.json` is a from-scratch E2.5
configuration (schema `4.6.0`, `parallel_envs=10`).  Its state-only base gate
is exactly E2.4 for `w_flow <= 0.2`.  Only after three canonical feasibility
successes is that base gate frozen and a positive-only
`kappa * max(w_flow - 0.2, 0)` commit-logit residual enabled.  The defer logit
and legal-action masks are unchanged.

The preregistered calibration configurations are
`e2_4_calibration_control_seed101.json` and
`e2_5_calibration_k{1,2,3}_seed101.json`; run each at 500 episodes with the
same final full-grid audit.  Select the smallest kappa satisfying both final
audits, 440-candidate safety/coverage, 8/8/4 diversity, all three Spearman
limits, and the 1% canonical-quality guard.  Do not expose test/OOD/stress
sets during this selection.  Summary checkpoint references are run-relative;
`train.resolve_summary_checkpoint` also accepts legacy absolute values.
