# Forced-action classification and non-delay ablation

Algorithm seed: 11; episodes per arm: 600; both arms use the same episode-indexed instance seeds.

## Training rollouts

| Metric | non-delay on | non-delay off | off - on |
|---|---:|---:|---:|
| Completion rate | 0 | 0 | +0 |
| Mean flow-time objective | 139.025 | 129.715 | -9.31 |
| Mean reconfiguration cost | 188.09 | 175.69 | -12.4001 |
| Mean worker-load variance | 3.83304 | 2.90779 | -0.92525 |
| Forced-action ratio | 0.645312 | 0.636719 | -0.00859375 |
| P95 episode longest forced chain | 10.05 | 8.4 | -1.65 |
| Maximum forced chain | 11 | 35 | +24 |
| WORKER unique-pair blocked by non-delay | 21 | 0 | -21 |

## Paired episode statistics

Differences are `non-delay off - non-delay on`; minimization metrics are better when negative.

| Metric | Mean difference | Median difference | Wilcoxon p |
|---|---:|---:|---:|
| flow_time_objective | -9.31 | -10.05 | 0.0400265 |
| reconfiguration_cost | -12.4001 | -7.26332 | 0.20245 |
| worker_load_variance | -0.92525 | -0.664444 | 0.153646 |
| forced_action_ratio | -0.00859375 | -0.015625 | 0.215616 |
| longest_forced_action_chain | -0.15 | -1.5 | 0.050808 |
| forced_worker_pair_non_delay_count | -1.05 | +0 | 0.00705833 |

## Fixed validation

No formal final checkpoint evaluation was available.
