"""Two-attempt CPU console integration probe, outside formal experiments."""
import argparse
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json

from configs.experiment_suite import load_suite, arm_config, suite_identity
from data import load_dataset_split
from agent.comparators.common import atomic_json, digest
from agent.comparators.learning import train_request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='result/verification/console_progress_20261024/live')
    args = parser.parse_args()
    config = arm_config(load_suite(), 'full', 11, 'cpu')
    config['network'].update(hidden_dim=16, message_passing_layers=1)
    config['training'].update(episodes=2, episodes_per_update=2,
                              validation_interval_epochs=1, validation_parallel_envs=2)
    record = load_dataset_split(config, 'calibration')[0]
    tiny = replace(record.instance, instance_id='console_probe_validation', orders=record.instance.orders[:1])
    dataset = [SimpleNamespace(instance=tiny, metadata=record.metadata)]
    output = Path(args.output)
    identity = digest(['console_progress_probe', suite_identity(load_suite(), 'cpu')])
    summary = train_request(config, 'ppo', 11, config['ppo']['learning_rate'], None,
                            dataset, [0], [(1., 0., 0.)], output, identity,
                            episodes=2, workers=2, smoke_instance=tiny)
    assert summary['epochs'] == summary['validation_count'] == 1
    assert summary['episodes_started'] == 2
    validations = json.loads((output/'validations.json').read_text(encoding='utf-8'))
    raw = json.loads((output/validations[0]['deliveries_path']).read_text(encoding='utf-8'))
    assert len(raw['deliveries'][0]['raw']) == 3
    assert sum(len(sample['actions']) for d in raw['deliveries'] for sample in d['raw']) == summary['validation_env_steps']
    atomic_json(output/'console_probe.json', {'purpose': 'local_console_integration', 'formal_experiment': False,
                'episodes': summary['episodes_started'], 'epochs': summary['epochs'],
                'validation_count': summary['validation_count'], 'training_steps': summary['training_steps'],
                'validation_env_steps': summary['validation_env_steps'], 'validation_trajectories': 3})


if __name__ == '__main__':
    main()
