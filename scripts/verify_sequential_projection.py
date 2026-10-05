"""Verify the three audited time corrections and save schema-10 evidence."""
from __future__ import annotations
from dataclasses import replace
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs import load_config
from data.models import load_instance_yaml
from environment import AssemblySchedulingEnv
from environment.resource_projection import ResourceProjector
from scripts.audit_schema9_followup import edge_times
from scripts.audit_state_sufficiency import worker_alias_instance, worker_histories


def main():
    config = load_config('configs/default.json')
    template = load_instance_yaml(config['paths']['fixed_instance'])
    env = AssemblySchedulingEnv(config)
    env.reset(template)
    env.step(0, build_observation=False)
    env.step(env.wait_action, build_observation=False)
    busy = edge_times(env, 'O_J2_1', 'M1')
    assert abs(busy['predicted_finish_time_norm']-18.9) < 1e-5
    installer = worker_histories(config, worker_alias_instance(template))[0][0]
    install = edge_times(installer, 'A_1', 'M6')
    assert abs(install['predicted_finish_time_norm']-46.4) < 1e-5
    instance = worker_alias_instance(template)
    waves = deepcopy(instance.waves)
    waves['W1']['release_interval'][0] = 0
    instance = replace(instance, orders=tuple(replace(o, release_time=0 if o.id=='A' else o.release_time)
                       for o in instance.orders), waves=waves,
                       workers=tuple(replace(w, initial_fatigue=.5 if w.id=='H5' else .74) for w in template.workers))
    sequential = AssemblySchedulingEnv(config)
    sequential.reset(instance)
    route = sequential._candidate_resource_projection(0, 5).path
    assert route.end_tick == 136 and abs(route.stages[0].end_fatigue-.6505) < 1e-12
    dis, ins, labor, downtime, variance = ResourceProjector(sequential).path_costs(5, route)
    result = dict(observation_schema=config['runtime_manifest']['observation_schema'],
                  time_context=config['runtime_manifest']['time_context'], busy_machine=busy,
                  active_installation=install, sequential=dict(processing_start=route.end_tick*sequential.resolution,
                    worker_indices=[s.worker_index for s in route.stages], starts=[s.start_tick for s in route.stages],
                    ends=[s.end_tick for s in route.stages], fatigue_after=[s.end_fatigue for s in route.stages],
                    fixed_disassembly=dis, fixed_installation=ins, labor=labor, downtime=downtime,
                    variance_delta=variance, planned_loads=route.resources.loads.tolist()),
                  source_hashes={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (
                    'environment/resource_projection.py','environment/time_context.py','environment/env.py',
                    'environment/observation_schema.py','configs/runtime.py','agent/ppo/network.py')})
    root = Path('result/analysis/schema10_projection_verification')
    root.mkdir(parents=True, exist_ok=True)
    (root/'acceptance.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('observation_schema','time_context','sequential')}, indent=2))

if __name__ == '__main__':
    main()
