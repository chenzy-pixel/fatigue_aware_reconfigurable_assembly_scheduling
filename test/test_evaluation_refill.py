from types import SimpleNamespace

import numpy as np
import pytest
import torch

from agent.ppo.parallel import ParallelEpisodeRunner, WorkerResponse


class SamplingAgent:
    device = 'cpu'
    network = SimpleNamespace(execution_mode='phase_batched_v1')

    def assert_evaluation_config(self, config):
        pass

    def act_batch(self, observations, masks, *, deterministic=False, generators=None):
        actions = ([0] * len(observations) if deterministic else
                   [int(torch.randint(0, 2, (), generator=g)) for g in generators])
        return actions, [0.0] * len(actions), [0.0] * len(actions)

    def consume_policy_decision_diagnostics(self):
        return []


def run_records(parallelism, deterministic):
    records = [SimpleNamespace(instance=SimpleNamespace(instance_id=f'job{i}',
                steps=steps, cutoff=i == 2)) for i, steps in enumerate((1, 4, 1, 2))]
    preferences = [(1., 0., 0.), (0., 1., 0.), (0., 0., 1.), (1., 0., 0.)]
    runner = object.__new__(ParallelEpisodeRunner)
    runner.worker_count = 2
    runner.config = {}
    states = {}
    events = []

    def exchange(requests):
        responses = {}
        for lane, (command, payload) in requests.items():
            if command == 'reset_instance':
                states[lane] = [payload.value, 0]
                events.append(('reset', payload.value.instance_id, payload.preference))
            else:
                states[lane][1] += 1
            record, count = states[lane]
            done = count == record.steps
            if done:
                events.append(('finish', record.instance_id, None))
            responses[lane] = WorkerResponse(lane_id=lane,
                observation=(record.instance_id, count), action_mask=np.zeros(2, dtype=np.bool_),
                terminated=done and not record.cutoff, truncated=done and record.cutoff,
                metrics={'task_succeeded': not record.cutoff, 'sampling_truncated': record.cutoff} if done else None)
        return responses

    runner._exchange = exchange
    results = runner.evaluate_records(SamplingAgent(), records, max_parallelism=parallelism,
        deterministic=deterministic, sampling_seed=None if deterministic else 100011,
        preferences=preferences)
    return results, events


@pytest.mark.parametrize('deterministic', [True, False])
def test_finished_evaluation_lane_refills_before_longer_task_ends(deterministic):
    serial, _ = run_records(1, deterministic)
    parallel, events = run_records(2, deterministic)
    assert [r.record_index for r in parallel] == list(range(4))
    assert [r.decisions for r in parallel] == [1, 4, 1, 2]
    assert events.index(('reset', 'job2', (0., 0., 1.))) < events.index(('finish', 'job1', None))
    for a, b in zip(serial, parallel):
        assert a.metrics == b.metrics
        assert a.action_trace_sha256 == b.action_trace_sha256
        assert a.derived_sampling_seed == b.derived_sampling_seed
        assert a.sampling_evaluation_key == b.sampling_evaluation_key
    assert parallel[2].metrics['sampling_truncated'] is True
