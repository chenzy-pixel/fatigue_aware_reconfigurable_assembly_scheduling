# V8 preference-conditioned policy

V8 uses six node types, twelve relations, and two HGNN message-passing layers.
The episode preference is a separate three-value observation field encoded by
`3 → 32 → ReLU → 32`.

Production, Worker, and WAIT actions each have Flow, Cost, and Variance experts.
Direct ranker weights use fixed-sign normalized `softplus(theta)` parameters.
The base logit is the preference-weighted expert output; a bounded
preference-conditioned residual is scaled by the standard deviation of legal
base logits. The actor and critic receive the same episode preference.

## Single-stage return

Training uses

\[
r_t=(P_{t+1}-P_t)-(Q_{t+1}-Q_t),
\]

where `P` is order-balanced completed-operation progress and `Q` is the bounded
augmented Tchebycheff score under the trajectory's preference. Every order is
present in the progress denominator from reset. Successful tasks use measured
terminal quality; failed tasks use the common terminal bound `1`.

With `gamma=1`, the collector verifies

\[
\sum_t r_t=P_T-P_0-Q_T+Q_0.
\]

Flow, Cost, and Variance remain in `RewardVector` and result rows as raw
diagnostics. The configured scalar reward is `operation_progress + quality`.

## Preference schedules

Flow, Cost, and Variance specialists use fixed one-hot preferences from the
first episode. The Universal policy uses deterministic 20-episode blocks:
two copies of every endpoint followed by 14 seeded scrambled-Sobol simplex
points. Formal Universal validation evaluates the fixed 66-point step-0.1
simplex grid.

## Formal selection

Validation evaluates each fixed manifest instance with three sampled repeats at
temperature `1.0`. Greedy evaluation is emitted alongside it as a diagnostic.
The rank is:

1. maximum sampled completion; Universal uses the minimum completion across the
   66 preferences;
2. minimum preference-balanced quality at equal completion.

Preference-balanced quality first averages successful trajectories inside each
preference, then equally averages the fixed preferences. Missing success in one
preference yields `+inf`.

The first safe validation creates `best_checkpoint.pt`. Later writes require a
strict lexicographic improvement. `last_checkpoint.pt` records the final online
state independently. Final evaluation reloads the best file and uses independent
sampled roots plus a greedy diagnostic pass.

## Reproducibility record

Checkpoint metadata includes:

- runtime and result schema versions;
- effective configuration and algorithm seed;
- validation manifest hash and ordered members;
- repeat count and full fixed preference set;
- sampled root seeds, temperature, RNG version, and derived-seed rule;
- the sampled validation row that selected the checkpoint.

Final provenance adds source state, environment information, effective-config
hash, checkpoint hash, and network-weight hash.
