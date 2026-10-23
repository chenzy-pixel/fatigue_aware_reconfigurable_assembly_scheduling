"""Current batch protocol preflight and experiment launcher."""
import json
from configs import load_config
from data import load_dataset_split
from data.distribution import validate_training_distribution

def startup_preflight(configs, *, probe_network=False, **kwargs):
    result = {}
    for name, c in configs.items():
        validate_training_distribution(c)
        pools = {s: load_dataset_split(c, s) for s in ('calibration', 'validation', 'test')}
        from scripts.comparison_suite import study_identity
        from agent.comparators.training_budget import training_budget
        row = {'study_identity':study_identity(c),'schema': 13, 'counts': {s: len(d) for s, d in pools.items()}, 'scales': c['objective_scalarizer']['scales'],
               'training_budget': training_budget(c), 'budget_status': 'configured_episode_training',
               'search_budget_status': 'requires_calibration_manifest', 'runtime': c['runtime_manifest']}
        if probe_network:
            import torch
            from environment import AssemblySchedulingEnv
            from agent.ppo import build_actor_critic
            e = AssemblySchedulingEnv(c)
            o = e.reset(pools['validation'][0].instance)
            n = build_actor_critic(o, c['network']).to(c['device'])
            with torch.no_grad():
                logits, v = n(o, e.get_action_mask(), device=c['device'])
            row['network_probe_finite'] = bool(torch.isfinite(v).all())
            row['feature_dimensions'] = o.feature_dimensions
        result[name] = row
    return result

def main():
    from scripts.comparison_suite import main as run
    run()
if __name__ == '__main__':
    main()
