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

## E2.5 stop record

E2.5 calibration has stopped under the preregistered rule.  No kappa was
selected, no intermediate kappa may be added, and neither seed 11 nor the five
formal seeds may be run for E2.5.

| kappa | final two full-grid audits | result |
| --- | --- | --- |
| 1 | traces 8.00/8.45; objectives 7.45/7.60; nondominated 3.40/3.40 | fails 8/8/4 |
| 2 | traces 9.55/9.20; objectives 8.95/8.55; nondominated 3.90/3.40; flow Spearman 0.061/0.018 | fails nondominated and flow direction |
| 3 | traces 6.50/6.40; objectives 6.00/5.95; nondominated 2.85/3.25 | fails diversity |

All three completed 440/440 candidates with zero schedule violations and zero
truncations; all 220 low-flow candidates were complete and safe; coverage and
fatigue safety passed; kappa 1 and 3 passed the three Spearman limits.  The
mechanistic diagnosis was decisive: mean commit boost increased
`0.195 -> 0.394 -> 0.587` and final commit probability increased roughly
`0.251 -> 0.259 -> 0.271`, but every run recorded zero
`base-defer -> final-commit` greedy flips.  E2.4 control artifacts are not
required to establish this stop because every candidate failed before the 1%
canonical-quality guard.

## E2.6 counterfactual preference consistency

E2.6 is a from-scratch E2.4-derived experiment, not a continuation or partial
load of E2.4/E2.5.  It retains the E2.4 state-only base gate during feasibility.
After the three canonical feasibility successes, the base gate is frozen and
the quality phase enables

```text
commit_boost(s, w) = a(s) * max(w_flow - 0.2, 0)
a(s) = 8 * sigmoid(r(s))
```

where `r(s)` is a dedicated action-set state head initialized so that
`a(s)=2.0`.  Thus low-flow (`w_flow <= 0.2`) gate logits are exactly the E2.4
base logits, and the high-flow commit margin is structurally monotone.  The
state scale is hard bounded in `[0, 8]`.

Only in the quality phase, E2.6 evaluates the same production state under
`w_low=(0.2,0.4,0.4)` and `w_high=(1,0,0)`.  If commit and defer are legal and
the base gate greedily defers, it adds

```text
L_cf = mean(ReLU(-m_high)^2)
L = L_PPO + lambda * L_cf
```

where `m_high` is the counterfactual high-flow commit-minus-defer margin.
This auxiliary path touches only the frozen base gate/state residual pathway;
it does not directly update pair, worker, critic, or graph-encoder branches.

The only calibration candidates are
`e2_6_calibration_l005_seed101.json`,
`e2_6_calibration_l010_seed101.json`, and
`e2_6_calibration_l020_seed101.json`, corresponding to
`lambda={0.05,0.10,0.20}`.  Each runs seed 101 for 500 episodes using
validation data only.  All three must finish before selection.  For each
candidate, both final `full_grid_22` audits must pass the existing
440/440 safety/coverage, 220 low-flow safety, 8/8/4, and three Spearman
requirements.  E2.6 additionally requires 20/20 instances to contain an
eligible counterfactual state and a pooled high-flow greedy flip rate of at
least 25%, with zero low-flow identity and monotonicity violations.

`e2_6_calibration_audit.py` is a read-only auditor.  It validates the fixed
seed/config/provenance protocol, compares each of the last two canonical
qualities individually against the corresponding E2.4 control audit with a
1% limit, and emits only `selected`, `stopped`, or `guard_pending`.  It cannot
launch training.  If selected, it chooses the smallest passing lambda; no
interpolation or new lambda is permitted.  Test/OOD/stress data stay hidden
until selection, after which the formal from-scratch seed set is
`[11,23,37,53,71]` at 2000 episodes each.
