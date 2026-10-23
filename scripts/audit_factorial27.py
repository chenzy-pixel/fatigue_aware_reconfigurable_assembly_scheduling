"""Replay every frozen witness and verify protected historical artifacts."""
import hashlib,json
from pathlib import Path
from configs import load_config
from data import load_dataset_split
from data.dataset import sha256_file,canonical_json_bytes
from scripts.prepare_factorial27 import reference


def audit_history(hashes, *, require_all=False):
    report = {'protected_verified': 0, 'protected_unavailable': 0}
    for path, expected in hashes.items():
        if not Path(path).is_file():
            if require_all:
                raise FileNotFoundError('historical artifact unavailable: '+path)
            report['protected_unavailable'] += 1
            continue
        if sha256_file(path) != expected:
            raise ValueError('historical artifact changed: '+path)
        report['protected_verified'] += 1
    report['history_status'] = 'complete' if not report['protected_unavailable'] else 'partial_archive'
    return report


def audit(config,*,replay=True,require_history=False):
    report={'counts':{},'certificates_replayed':0,'protected_verified':0}
    for split in ('calibration','validation','test'):
        dataset=load_dataset_split(config,split);report['counts'][split]=len(dataset)
        for row,record in zip(dataset.manifest['files'],dataset):
            path=(dataset.instances_root/row['certificate']).resolve()
            if not path.is_relative_to(dataset.instances_root.resolve()): raise ValueError('unsafe certificate path')
            if sha256_file(path)!=row['certificate_sha256']: raise ValueError('certificate hash mismatch')
            cert=json.loads(path.read_text(encoding='utf-8'))
            if cert['instance_sha256']!=row['sha256']: raise ValueError('witness instance mismatch')
            if replay:
                metrics,_=reference(record.instance,config,cert['actions'])
                old=record.metadata['heuristic_metrics']
                for key in ('time','flow_excess_objective','reconfiguration_cost','worker_loads','maximum_worker_fatigue','completed_operations'):
                    if metrics[key]!=old[key]: raise ValueError(f'witness replay differs: {key}')
                report['certificates_replayed']+=1
    before=json.loads(Path('result/migration/factorial27_20261023/before.json').read_text(encoding='utf-8'))
    report.update(audit_history(before['protected_hashes'], require_all=require_history))
    return report


def main():
    report=audit(load_config('configs/default.json'))
    path=Path('result/migration/factorial27_20261023/audit.json')
    path.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))


if __name__=='__main__': main()
