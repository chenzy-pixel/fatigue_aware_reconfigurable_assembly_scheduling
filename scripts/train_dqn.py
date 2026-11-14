"""Train supplemental vanilla DQN in an isolated, reconstructed runtime."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile


def train_seed(config, seed, dataset, indices, prefs, output, identity, workers, *, episodes=2000):
    """Use explicit episode budgets across the historical trainer signatures."""
    import inspect
    from agent.comparators.learning import train_request
    options = dict(dataset=dataset, val_indices=indices, prefs=prefs, output=output,
                   identity=identity, workers=workers, episodes=episodes)
    if 'steps' in inspect.signature(train_request).parameters:
        options['steps'] = None
    return train_request(config, 'dqn', seed, config['ppo']['learning_rate'], **options)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[11, 23, 37, 71])
    parser.add_argument('--base-identity', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(argv)

    import torch
    from configs.config import PROJECT_ROOT, public_config
    from configs.experiment_suite import load_suite, arm_config, identity_payload
    from configs.formal_preferences import formal_preferences
    from data import load_dataset_split
    from agent.comparators.common import method_config, digest, atomic_json
    from agent.comparators.learning import build_agent
    from scripts.experiment_suite import execution_lock, hardware
    from utils import set_seed

    suite = load_suite()
    if (not args.seeds or len(set(args.seeds)) != len(args.seeds)
            or any(seed not in suite['seeds'] for seed in args.seeds)):
        parser.error('Select distinct formal seeds.')
    torch.set_num_threads(suite['execution']['torch_threads'])
    base = arm_config(suite, 'rule_ddqn', suite['seeds'][0], args.device)
    config = method_config(base, 'dqn', suite['seeds'][0], base['ppo']['learning_rate'])
    config['comparison']['arm'] = 'dqn'
    payload = identity_payload(suite, args.device)
    # Include every archived Python module, including package initializers,
    # logging helpers and the supplemental launcher, in the new identity.
    import hashlib
    from result.provenance import source_files
    payload['sources'] = {path.relative_to(PROJECT_ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in source_files(PROJECT_ROOT)}
    payload['additional_training'] = {'method': 'dqn', 'base_identity': args.base_identity,
                                      'config': public_config(config), 'target_rule': 'target_max_dqn_v1'}
    identity = digest(payload)
    output = args.output_dir.resolve()
    dataset = load_dataset_split(config, 'validation')
    if len(dataset) != 27:
        raise ValueError('Formal DQN validation requires the original 27 instances.')
    indices = list(range(27))
    prefs = [p.as_tuple() for p in formal_preferences(config, 'validation')]
    # Exercise network construction and native action legality before a long run.
    from environment import AssemblySchedulingEnv
    set_seed(suite['seeds'][0])
    env = AssemblySchedulingEnv(config)
    obs = env.reset(dataset[0].instance)
    agent = build_agent(config, obs, 'dqn')
    action = agent.act_rules([obs], [env.get_action_mask()])[0][0]
    env.step(action)
    del agent, env, obs

    with execution_lock(output):
        preflight = output/'preflight.json'
        if preflight.exists():
            saved = json.loads(preflight.read_text(encoding='utf-8'))
            if saved.get('identity') != identity or saved.get('status') != 'passed':
                raise ValueError('DQN identity differs; select a new DQN output directory.')
        else:
            if any((output/name).exists() for name in ('training', 'source.zip')):
                raise ValueError('DQN output contains artifacts without a matching preflight.')
            archive_path = output/'source.zip'
            with zipfile.ZipFile(archive_path.with_suffix('.zip.tmp'), 'w', zipfile.ZIP_DEFLATED) as archive:
                for name in payload['sources']:
                    archive.write(PROJECT_ROOT/name, name)
                # Include JSON presets needed to load the archived runtime.
                for path in (PROJECT_ROOT/'configs').rglob('*.json'):
                    archive.write(path, path.relative_to(PROJECT_ROOT).as_posix())
                archive.writestr('experiment_identity.json', json.dumps(payload, ensure_ascii=False, indent=2))
            archive_path.with_suffix('.zip.tmp').replace(archive_path)
            atomic_json(preflight, {'status': 'passed', 'identity': identity,
                        'source_hashes': payload['sources'], 'suite': suite,
                        'configs': payload['configs'], 'data_manifests': payload['data'],
                        'additional_training': payload['additional_training'], 'hardware': hardware()})
        if not (output/'source.zip').is_file():
            raise ValueError('DQN source archive is missing.')
        with zipfile.ZipFile(output/'source.zip') as archive:
            if digest(json.loads(archive.read('experiment_identity.json'))) != identity:
                raise ValueError('DQN source archive identity differs.')
            for name, expected in payload['sources'].items():
                if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                    raise ValueError(f'DQN archived source differs: {name}')
        print(f'DQN identity: {identity}\nDQN output: {output}', flush=True)
        if args.check_only:
            print('DQN runtime checks passed; training was not started.', flush=True)
            return
        for seed in args.seeds:
            current = method_config(config, 'dqn', seed, config['ppo']['learning_rate'])
            train_seed(current, seed, dataset, indices, prefs, output/'training/dqn'/str(seed), identity,
                       suite['execution']['training_workers'])


if __name__ == '__main__':
    main()
