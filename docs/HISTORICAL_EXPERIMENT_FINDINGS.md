# Historical experiment findings

This document is a read-only summary of experiments removed by the
latest-only refactor. The complete source, configurations, tests, and original
audit reports remain recoverable from Git tag
`archive/pre-latest-only-20260908`.

## Retained conclusion

E1 is the only accepted delivery line. Its bounded V7 ranker preserves the
monotone candidate score as the main signal and permits only a scale-bounded
context residual. The accepted seed-11 checkpoint reached 100% completion on
the fixed development validation set and improved canonical bounded quality
over the heuristic by 12.67% in the recorded protocol-v2 audit. These are
development results, not a five-seed publication claim.

## Why the preference-conditioned line was retired

The removed experiments formed a consistent failure chain:

- The first conditioned policy produced accepted canonical checkpoints but
  collapsed to roughly 3.6–4.8 unique trajectories out of 22 candidates and
  lost hypervolume to E1 on test, OOD, and stress splits.
- Direct preference terms in worker selection broke matching/liveness; two
  successors stopped at 95% validation completion.
- Restricting the preference signal to production recovered canonical
  feasibility, but low-flow preferences over-deferred and caused ten horizon
  truncations in the 440-candidate grid.
- The state-only safety gate reached 440/440 completion by removing the main
  controllable commit/defer degree of freedom, after which diversity fell
  below the acceptance threshold.
- Later gate scaling produced no greedy action flips. The auxiliary
  consistency loss then had an empty eligible-state set, zero effective loss,
  and identical final weight hashes across all tested coefficients.
- The final warm-start variants showed promising development hypervolume but
  did not satisfy the combined safety, response-direction, and canonical
  quality requirements; no accepted result was established.

The failure was therefore architectural and protocol-level rather than a lack
of model parameters. The repository now keeps the experimentally supported E1
path executable and retains Pareto/HV only as offline rollout analysis and the
MO-ALNS comparison protocol.

## Archival boundary

No historical run artifacts were deleted. Removed executable paths can be
inspected or restored from the archive tag without carrying them in the active
source tree.
