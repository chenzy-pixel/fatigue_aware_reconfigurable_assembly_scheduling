# Forced-action classification and non-delay ablation

Algorithm seed: 11; episodes per arm: 600; both arms use the same episode-indexed instance seeds.

## Training rollouts

| Metric | non-delay on | non-delay off | off - on |
|---|---:|---:|---:|
| Completion rate | 0.993333 | 0.993333 | +0 |
| Mean flow-time objective | 1036.31 | 1031.36 | -4.94583 |
| Mean reconfiguration cost | 796.882 | 820.373 | +23.491 |
| Mean worker-load variance | 4.87184 | 4.32394 | -0.547899 |
| Forced-action ratio | 0.648815 | 0.659932 | +0.0111172 |
| P95 episode longest forced chain | 29.15 | 47.3 | +18.15 |
| Maximum forced chain | 3762 | 3762 | +0 |
| WORKER unique-pair blocked by non-delay | 2010 | 0 | -2010 |

## Paired episode statistics

Differences are `non-delay off - non-delay on`; minimization metrics are better when negative.

| Metric | Mean difference | Median difference | Wilcoxon p |
|---|---:|---:|---:|
| flow_time_objective | -4.94583 | -11.5 | 3.03225e-08 |
| reconfiguration_cost | +23.491 | +10.9234 | 0.00562106 |
| worker_load_variance | -0.547899 | -0.495694 | 0.00156875 |
| forced_action_ratio | +0.00455038 | +0.00767573 | 0.00250179 |
| longest_forced_action_chain | +4.48833 | +0 | 0.0010756 |
| forced_worker_pair_non_delay_count | -3.35 | -1 | 4.03569e-57 |

## Fixed validation

| Metric | non-delay on | non-delay off |
|---|---:|---:|
| Completion rate | 1 | 1 |
| Mean flow-time objective | 1080.28 | 1102.87 |
| Mean reconfiguration cost | 629.226 | 758.702 |
| Mean worker-load variance | 2.80643 | 2.66189 |
| Forced-action ratio | 0.614958 | 0.577639 |
| Mean longest forced chain | 8.15 | 16.15 |
