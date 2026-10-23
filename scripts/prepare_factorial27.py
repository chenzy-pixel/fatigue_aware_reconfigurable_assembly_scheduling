"""Build the fixed factorial corpus, replay witnesses, then freeze scales."""
import csv
import hashlib
import json
import statistics
from pathlib import Path

from data.dataset import canonical_json_bytes, save_generated_record_atomic, sha256_file
from data.distribution import CELLS, stable_seed, protocol_hashes
from data.generate_orders import InstanceGenerator
from environment import AssemblySchedulingEnv
from agent.baselines import HeuristicPolicy


ROOT=Path(__file__).resolve().parents[1]


def reference(instance, config, actions=None):
    env=AssemblySchedulingEnv(config)
    env.reset(instance,build_observation=False)
    policy=HeuristicPolicy(); trace=[]
    if actions is None:
        while not env.task_done:
            action=policy.select_action(env);trace.append(int(action))
            env.step(action,build_observation=False)
    else:
        for action in actions:
            if env.task_done: raise ValueError('trailing certificate actions')
            env.step(action,build_observation=False)
        trace=list(actions)
    metrics=env.metrics()
    if not env.task_succeeded or env.validate_schedule() or metrics['maximum_worker_fatigue']>instance.fatigue.maximum_safe_fatigue+1e-9:
        raise ValueError({'instance':instance.instance_id,'reason':env.terminal_reason})
    return metrics,trace


def write_table(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    config_path=ROOT/'configs/default.json'
    config=json.loads(config_path.read_text(encoding='utf-8'))
    if all((ROOT/config['paths']['manifests_root']/split/'manifest.json').is_file() for split in ('calibration','validation','test')) and (ROOT/'configs/factorial27_normalization.json').is_file():
        from configs import load_config
        from scripts.audit_factorial27 import audit
        print(json.dumps(audit(load_config(config_path)),indent=2));return
    # Reference decisions do not use scalar rewards. Unit scales bootstrap calibration only.
    config['objective_scalarizer']['scales']={k:1. for k in ('flow','cost','workload')}
    generator=InstanceGenerator(config);diagnostics=[];calibration=[]
    for split,repeats in (('calibration',2),('validation',1),('test',1)):
        entries=[]
        for cell in CELLS:
            for replicate in range(repeats):
                seed=stable_seed('factorial27_v1',split,cell.cell_id,replicate)
                record=generator.generate(seed=seed,cell=cell,split=split)
                metrics,actions=reference(record.instance,config)
                replay,_=reference(record.instance,config,actions)
                fields=('flow_excess_objective','reconfiguration_cost','workload_objective_value','time')
                if any(metrics[k]!=replay[k] for k in fields): raise ValueError('witness replay differs')
                name=cell.cell_id+(f'_{replicate}' if repeats>1 else '')
                relative=f'{split}/{name}.json'
                record.metadata.update(feasibility_status='observed_feasible',diagnostic_status='completed',diagnostic_terminal_reason='completed',heuristic_metrics={**metrics,'heuristic_completed':True,'heuristic_truncated':False,'heuristic_makespan':metrics['time'],'heuristic_flow_time':metrics['total_flow_time'],'heuristic_reconfiguration_cost':metrics['reconfiguration_cost'],'heuristic_worker_load_variance':metrics['worker_load_variance']})
                digest=save_generated_record_atomic(record,ROOT/config['paths']['instances_root']/relative)
                certificate={'version':'batch_witness_v1','instance_sha256':digest,'actions':actions,'schedule_sha256':hashlib.sha256(canonical_json_bytes(metrics)).hexdigest()}
                cp=ROOT/config['paths']['instances_root']/f'{split}/{name}.certificate.json'
                cp.write_bytes(canonical_json_bytes(certificate))
                entries.append({'path':relative,'sha256':digest,'seed':seed,'cell_id':cell.cell_id,'certificate':cp.relative_to(ROOT/config['paths']['instances_root']).as_posix(),'certificate_sha256':sha256_file(cp)})
                row={'split':split,'instance_id':record.instance.instance_id,'cell_id':cell.cell_id,'scale':cell.scale,'N':cell.N,'K':cell.K,'M':cell.M,'W':cell.W,'A':cell.A,'operations':cell.N*cell.K,**{k:metrics[k] for k in fields},'decisions':metrics['decision_count'],'reconfigurations':metrics['completed_reconfigurations'],'maximum_fatigue':metrics['maximum_worker_fatigue']}
                diagnostics.append(row)
                if split=='calibration': calibration.append(row)
                print(split,name,'completed',round(metrics['time'],1),flush=True)
        path=ROOT/config['paths']['manifests_root']/split/'manifest.json';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(canonical_json_bytes({'schema_version':'3.0.0','split':split,'instance_count':len(entries),'files':entries,**protocol_hashes(config)}))
    scales={'flow':statistics.fmean(r['flow_excess_objective'] for r in calibration),'cost':statistics.fmean(r['reconfiguration_cost'] for r in calibration),'workload':statistics.fmean(r['workload_objective_value'] for r in calibration)}
    manifest={'schema_version':'factorial27_scales_v1','selection':'calibration_54_reference_mean','scales':scales,'flow_mode':config['objective_scalarizer']['flow_mode'],'workload_objective':config['workload_objective'],'calibration_manifest_sha256':sha256_file(ROOT/config['paths']['manifests_root']/'calibration/manifest.json')}
    target=ROOT/'configs/factorial27_normalization.json';target.write_bytes(canonical_json_bytes(manifest))
    config=json.loads(config_path.read_text(encoding='utf-8'));config['objective_scalarizer'].update(scales=scales,normalization_manifest_sha256=sha256_file(target))
    config_path.write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
    write_table(ROOT/'data/tables/factorial27_diagnostics.csv',diagnostics)
    write_table(ROOT/'data/tables/factorial27_parameters.csv',[{k:r[k] for k in ('cell_id','scale','N','K','M','W','A','operations')} for r in diagnostics if r['split']=='test'])
    write_table(ROOT/'data/tables/factorial27_common_parameters.csv',[{'parameter':k,'value':json.dumps(v)} for k,v in config['generator'].items()])
    print('Frozen scales:',scales,flush=True)


if __name__=='__main__': main()
