"""Bounded real worker/update/checkpoint verification of epoch artifacts."""
import argparse
import csv
import json
from pathlib import Path

from configs import load_config
from data import load_dataset_split
from agent.comparators.learning import train_request
from scripts.comparison_suite import study_identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    config = load_config('configs/default.json'); config['device'] = 'cpu'
    config['comparison'] = {'ddqn': {'warmup': 2, 'batch_size': 2, 'update_every': 1}}
    dataset = load_dataset_split(config, 'validation')
    identity = study_identity(config)
    report = {'identity': identity, 'purpose': 'bounded_logging_integration', 'methods': {}}
    for method in ('ppo', 'dcan', 'rule_ddqn'):
        folder = output/method
        summary = train_request(config, method, 11, 1e-4, 24, dataset, [0], [(1/3,1/3,1/3)],
                                folder, identity, workers=2)
        episodes = json.loads((folder/'episodes.json').read_text(encoding='utf-8'))
        history = json.loads((folder/'training_log.json').read_text(encoding='utf-8'))
        assert sum(e['steps'] for e in episodes) == summary['training_steps'] == 24
        assert summary['epochs'] == len(history) == len(list((folder/'epochs').glob('*.json')))
        assert all(e['metrics'] is not None and e['metrics']['decision_count'] == e['steps'] for e in episodes)
        with (folder/'episodes.csv').open(encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        assert all(r['snapshot_flow'] != '' and r['objective_flow'] == '' for r in rows if r['budget_truncated'] == 'True')
        assert (folder/'validation_instances.csv').stat().st_size > 0
        assert summary['optimizer_updates'] > 0
        report['methods'][method] = {'training_steps': 24, 'epochs': summary['epochs'],
                                     'episode_records': len(rows), 'optimizer_updates': summary['optimizer_updates']}
    (output/'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
