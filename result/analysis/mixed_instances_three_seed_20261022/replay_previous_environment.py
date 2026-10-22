"""Replay every new seed37 action trace through the pre-cleanup environment."""
import json,subprocess,types,sys,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT));OUT=Path(__file__).parent
from configs import load_config
from data import load_dataset_split
from utils import action_trace_sha256

source=subprocess.check_output(['git','show','58d5845:environment/env.py'],cwd=ROOT)
(OUT/'previous_env_snapshot.py').write_bytes(source)
module=types.ModuleType('environment._previous_env');module.__package__='environment';sys.modules[module.__name__]=module
exec(compile(source,'58d5845:environment/env.py','exec'),module.__dict__)
config=load_config(next((ROOT/'result/runs').glob('experiment_seed37*_01_mixed_instances'))/'config.json')
data={r.instance.instance_id:r.instance for r in load_dataset_split(config,'validation')}
count=0;errors=[];deltas=[]
for path in sorted((OUT/'seed37/requests').glob('*.json')):
    req=json.loads(path.read_text())
    for raw in req['raw']:
        env=module.AssemblySchedulingEnv(config);env.reset(data[raw['instance_id']],preference=raw['preference'],build_observation=False)
        for action in raw['actions']:
            assert not env.terminated and not env.truncated
            mask=env.get_action_mask();assert not mask[action],(path,action)
            env.step(action,build_observation=False)
        m=env.metrics();expected=raw['metrics']
        assert m['task_succeeded']==expected['task_succeeded']
        assert not env.validate_schedule()
        for key in ['flow_excess_objective','reconfiguration_cost','workload_objective_value','preference_quality_score','maximum_worker_fatigue']:
            delta=abs(m[key]-expected[key]);deltas.append(delta);assert delta<1e-9,(path,key,delta)
        assert action_trace_sha256(raw['actions'])==expected['action_trace_sha256']
        count+=1
    print('replayed',count,flush=True)
assert count==84
receipt=dict(source_git='58d5845',source_sha256=hashlib.sha256(source).hexdigest(),replayed_trajectories=count,all_actions_legal=True,all_objectives_match=True,max_metric_absolute_error=max(deltas),new_policy_sampling=0)
(OUT/'previous_environment_replay.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(receipt)
