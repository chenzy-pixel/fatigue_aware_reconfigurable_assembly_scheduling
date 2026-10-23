"""Small real episode runs, validation isolation, and checkpoint reload checks."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import torch

from configs import load_config
from data import load_dataset_split
from environment import AssemblySchedulingEnv
from agent.comparators.common import method_config, atomic_json
from agent.comparators.learning import train_request, build_agent, validation
from scripts.comparison_suite import study_identity, check_learning_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--episodes', type=int, default=3)
    args = parser.parse_args()
    output = Path(args.output)
    if not 1 <= args.episodes <= 5:
        parser.error('local integration expects 1 to 5 attempts per method')
    c = load_config('configs/default.json')
    c['device'] = 'cpu'
    c['training'].update(episodes=args.episodes, episodes_per_update=2, validation_interval_epochs=1)
    c['comparison'] = {'ddqn': {'warmup': 2, 'batch_size': 2, 'update_every': 1}}
    dataset = load_dataset_split(c, 'validation')
    identity = study_identity(c)
    report = {'purpose': 'small_real_episode_integration', 'identity': identity, 'methods': {}}
    streams = []
    for method in ('ppo', 'dcan', 'rule_ddqn'):
        folder = output/method
        summary = train_request(c, method, 11, c['ppo']['learning_rate'], None, dataset,
                                [0], [(1/3,1/3,1/3)], folder, identity, workers=2, episodes=args.episodes)
        rows = json.loads((folder/'episodes.json').read_text(encoding='utf-8'))
        vals = json.loads((folder/'validations.json').read_text(encoding='utf-8'))
        assert summary['epochs'] == summary['validation_count'] == (args.episodes+1)//2
        assert summary['episodes_started'] == args.episodes
        assert [r['episode'] for r in rows] == list(range(args.episodes))
        assert all(not r['budget_truncated'] and (r['metrics']['terminated'] or r['metrics']['truncated']) for r in rows)
        assert sum(r['steps'] for r in rows) == summary['training_steps']
        assert sum(len(sample['actions']) for v in vals for d in v['deliveries'] for sample in d['raw']) == summary['validation_env_steps']
        assert summary['optimizer_updates'] > 0
        streams.append([(r['record_sha256'], r['preference']) for r in rows])
        torch.set_num_threads(1)
        agent = build_agent(method_config(c, method, 11), AssemblySchedulingEnv(c).reset(dataset[0].instance), method)
        check_learning_checkpoint(folder/'best_checkpoint.pt', c, method, 11, identity)
        agent.load(folder/'best_checkpoint.pt')
        network = agent.online if method == 'rule_ddqn' else agent.network
        weights = {k:v.clone() for k,v in network.state_dict().items()}
        if method == 'rule_ddqn':
            agent.set_episode_progress(1,3)
            before = (agent.steps, agent.updates, agent.sample_presentations,
                      deepcopy(agent.replay), agent.rng.getstate(), agent.exploration_epsilon)
        v, _ = validation(c, agent, dataset, [0], [(1/3,1/3,1/3)], method, seed=11)
        assert all(torch.equal(t, network.state_dict()[k]) for k,t in weights.items())
        assert v['summary']['validation_env_steps'] == sum(len(s['actions']) for d in v['deliveries'] for s in d['raw'])
        if method == 'rule_ddqn':
            assert before == (agent.steps, agent.updates, agent.sample_presentations,
                              agent.replay, agent.rng.getstate(), agent.exploration_epsilon)
        report['methods'][method] = summary
        print(f"{method}: {args.episodes} attempts / {summary['epochs']} epochs / {summary['validation_count']} validations; reload and validation isolation passed", flush=True)
    assert streams[0] == streams[1] == streams[2]
    report['identical_instance_hash_and_preference_stream'] = True
    atomic_json(output/'verification.json', report)


if __name__ == '__main__':
    main()
