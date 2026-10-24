"""Workflow routing, shared pools, prefix reuse and completed-work skipping."""
from contextlib import contextmanager
from types import SimpleNamespace

from configs.experiment_suite import load_suite
from scripts import experiment_suite as executor


def test_frontier_pool_independent_latency_and_cached_restart(tmp_path, monkeypatch):
    suite = load_suite()
    calls, pools, loads, reuse = [], [], [], []
    records = [SimpleNamespace(instance=SimpleNamespace(instance_id=f'case_{i}'),
                               metadata={'cell_id': f'C0{i+1}', 'scale': 'small'}) for i in range(2)]
    config = {'training': {'validation_parallel_envs': 20}}
    monkeypatch.setattr(executor, 'arm_config', lambda *a: config)
    monkeypatch.setattr(executor, 'load_dataset_split', lambda *a: records)
    monkeypatch.setattr(executor, 'preferences', lambda c, task: [(1., 0., 0.)])
    def load(*a, **kw):
        loads.append(1)
        return config, object(), {}
    monkeypatch.setattr(executor, 'load_agent', load)
    @contextmanager
    def pool(*a, **kw):
        pools.append(1)
        yield 'shared_pool'
    monkeypatch.setattr(executor, 'evaluation_pool', pool)
    def unit(c, agent, record, method, prefs, seed, **kw):
        calls.append((record.instance.instance_id, kw['mode'], kw['parallel'], kw['runner']))
        return {'requests': [], 'deliveries': [], 'seconds': 1.}
    monkeypatch.setattr(executor, 'learning_unit', unit)
    def first(c, source):
        assert source['seconds'] == 1.
        reuse.append(1)
        return {'requests': [], 'deliveries': [], 'seconds': 1., 'cost_reference': 'full',
                'source_job_id': source['job_id'], 'source_content_sha256': source['content_sha256']}
    monkeypatch.setattr(executor, 'reuse_first_sample', first)
    executor.evaluate(suite, tmp_path, 'id', 'cpu', ['full'], [11], ['frontier', 'independent_request'])
    assert len(pools) == len(loads) == 1 and len(reuse) == 2
    assert len(calls) == 10
    assert all(pool == ('shared_pool' if parallel else None) for _, _, parallel, pool in calls)
    assert [m for _, m, parallel, _ in calls if parallel] == [None, 'greedy'] * 2
    assert [m for _, m, parallel, _ in calls if not parallel] == [None, 'greedy', 'sampled_single'] * 2
    # A completed restart must not reload the model or start idle workers.
    executor.evaluate(suite, tmp_path, 'id', 'cpu', ['full'], [11], ['frontier', 'independent_request'])
    assert len(pools) == len(loads) == 1 and len(calls) == 10
    # Missing derived output can be rebuilt from its saved parent, with no solve.
    path = tmp_path/'evaluation'/'frontier'/'full_sample1'/'11'/'case_0.json'
    path.unlink()
    executor.evaluate(suite, tmp_path, 'id', 'cpu', ['full'], [11], ['frontier'])
    assert len(reuse) == 3 and len(calls) == 10
    assert len(pools) == len(loads) == 1
    # Rewriting a parent under the same job changes its content identity. The
    # derived result must be refreshed even though its own file is intact.
    parent_path = tmp_path/'evaluation'/'frontier'/'full'/'11'/'case_0.json'
    parent = executor.read_json(parent_path)
    parent['model_timing']['warmup_seconds'] = 3.
    parent['content_sha256'] = executor.digest({k:v for k,v in parent.items() if k != 'content_sha256'})
    executor.atomic_json(parent_path, parent)
    executor.evaluate(suite, tmp_path, 'id', 'cpu', ['full'], [11], ['frontier'])
    rebuilt = executor.read_json(path)
    assert rebuilt['source_content_sha256'] == parent['content_sha256']
    assert len(reuse) == 4 and len(calls) == 10 and len(loads) == len(pools) == 1


def test_independent_only_never_starts_pool(tmp_path, monkeypatch):
    suite = load_suite()
    record = SimpleNamespace(instance=SimpleNamespace(instance_id='one'), metadata={'cell_id': 'C01', 'scale': 'small'})
    monkeypatch.setattr(executor, 'arm_config', lambda *a: {})
    monkeypatch.setattr(executor, 'load_dataset_split', lambda *a: [record])
    monkeypatch.setattr(executor, 'load_agent', lambda *a, **k: ({}, object(), {}))
    monkeypatch.setattr(executor, 'preferences', lambda *a: [(1., 0., 0.)])
    def pool(*a, **kw):
        raise AssertionError('standalone latency must not start evaluation workers')
    monkeypatch.setattr(executor, 'evaluation_pool', pool)
    def unit(*a, **kw):
        assert kw['parallel'] is False and kw['runner'] is None
        return {'requests': [], 'deliveries': [], 'seconds': 1.}
    monkeypatch.setattr(executor, 'learning_unit', unit)
    executor.evaluate(suite, tmp_path, 'id', 'cpu', ['a2c'], [11], ['independent_request'])
