"""Capture physical and unaffected observation contracts around projection changes."""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from agent.baselines import HeuristicPolicy, RandomPolicy
from configs import load_config
from data.dataset import load_generated_record
from data.models import load_instance_yaml
from environment import AssemblySchedulingEnv, EdgeStore, CAPABLE_EDGE, SERVICE_CANDIDATE_EDGE
from scripts.audit_observation_reward import observation_sha256


def stable_projection(obs):
    nodes = {k: v.copy() for k, v in obs.node_features.items()}
    names = dict(obs.node_feature_names)
    position = names['order'].index('estimated_order_slack_norm')
    nodes['order'] = np.delete(nodes['order'], position, axis=1)
    names['order'] = tuple(n for n in names['order'] if n != 'estimated_order_slack_norm')
    relations = dict(obs.relations)
    stable_capable = ('processing_time_norm', 'configuration_match')
    for kind in (CAPABLE_EDGE, SERVICE_CANDIDATE_EDGE):
        store = relations[kind]
        selected = [i for i, n in enumerate(store.feature_names)
                    if (n in stable_capable if kind == CAPABLE_EDGE else n != 'estimated_order_slack_norm')]
        relations[kind] = EdgeStore(store.edge_index, store.edge_features[:, selected],
                                    tuple(store.feature_names[i] for i in selected), store.bidirectional)
    selected = [i for i, n in enumerate(obs.action_set_feature_names)
                if n not in ('minimum_order_slack_after_wait_norm', 'minimum_order_slack_delta_if_wait_norm')]
    return replace(obs, node_features=nodes, node_feature_names=names, relations=relations,
                   action_set_features=obs.action_set_features[selected],
                   action_set_feature_names=tuple(obs.action_set_feature_names[i] for i in selected))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('capture', 'verify'))
    parser.add_argument('--baseline', default='result/audits/schema10_behavior_reference.json')
    args = parser.parse_args()
    config = load_config('configs/default.json')
    instances = [load_instance_yaml(config['paths']['fixed_instance'])] + [
        load_generated_record(str(Path(config['paths']['instances_root'])/'validation'/f'instance_{2000000+i}.json')).instance for i in range(7)]
    results = []
    for instance in instances:
        for mode in ('full', 'neutral'):
            for label, policy in (('heuristic', HeuristicPolicy()), ('random11', RandomPolicy(11))):
                settings = deepcopy(config)
                settings['environment']['fatigue_mode'] = mode
                env = AssemblySchedulingEnv(settings)
                env.reset(instance)
                digest = hashlib.sha256()
                actions, rewards = [], []
                while not env.task_done:
                    obs = env.observe()
                    digest.update(observation_sha256(stable_projection(obs)).encode())
                    digest.update(env.get_action_mask().tobytes())
                    action = int(policy.select_action(env))
                    actions.append(action)
                    rewards.append(env.step(action, build_observation=False)[1].as_dict())
                digest.update(observation_sha256(stable_projection(env.observe())).encode())
                results.append(dict(instance=instance.instance_id, mode=mode, policy=label, stable_digest=digest.hexdigest(),
                                    actions=actions, rewards=rewards, objectives=env._objective_vector(),
                                    reason=env.terminal_reason, schedule=env.schedule_log, reconfigurations=env.reconfiguration_log))
        print(f'{len(results)} trajectories checked', flush=True)
    target = Path(args.baseline)
    results = json.loads(json.dumps(results))
    if args.mode == 'capture':
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(results), encoding='utf-8')
    else:
        expected = json.loads(target.read_text(encoding='utf-8'))
        for actual, before in zip(results, expected, strict=True):
            assert actual == before, (actual['instance'], actual['mode'], actual['policy'])
    print(f'{args.mode}: {len(results)} trajectories / {sum(len(r["actions"]) for r in results)} steps matched', flush=True)

if __name__ == '__main__':
    main()
