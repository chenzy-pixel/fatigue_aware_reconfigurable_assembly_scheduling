"""Export paid delivery requests, with instance-first aggregation."""
import csv,json,statistics,argparse
from pathlib import Path
from collections import defaultdict


def export_training(directory):
    from result.io import write_csv
    for file in Path(directory).rglob('training_log.json'):
        folder=file.parent
        history=json.loads(file.read_text(encoding='utf-8'))
        if history and 'epoch' in history[0]:
            from result.training_history import export_history
            episodes=json.loads((folder/'episodes.json').read_text(encoding='utf-8'))
            val_path=folder/'validations.json'
            validations=json.loads(val_path.read_text(encoding='utf-8')) if val_path.exists() else None
            export_history(folder,history,episodes,validations)
            continue
        for source,target in (('training_log.json','training_log.csv'),('episodes.json','episodes.csv')):
            rows=json.loads((folder/source).read_text(encoding='utf-8'))
            flattened=[]
            for r in rows:
                row={k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items() if k!='losses'}
                row.update({'loss_'+k:v for k,v in r.get('losses',{}).items()})
                flattened.append(row)
            write_csv(folder/target,flattened)
        if not (folder/'validations.json').exists():
            continue
        validations=json.loads((folder/'validations.json').read_text(encoding='utf-8'))
        write_csv(folder/'validation_log.csv',[{'training_steps':v['training_steps'],**v['summary']} for v in validations])


def summarize(directory):
    directory=Path(directory);export_training(directory);rows=[];seen=set()
    for p in directory.rglob('*.json'):
        payload=json.loads(p.read_text(encoding='utf-8'))
        r=payload.get('summary',{}) if isinstance(payload,dict) else {}
        if not isinstance(r,dict) or 'instance_id' not in r: continue
        key=(r['method'],r['instance_id'],tuple(r['preference']),r.get('seed'))
        if key in seen: raise ValueError('duplicate delivery request')
        seen.add(key);objectives=r.get('objectives')
        rows.append({'method':r['method'],'instance_id':r['instance_id'],'cell_id':r['cell_id'],'scale':r['scale'],'seed':r.get('seed'),'preference':json.dumps(r['preference']),'success':r['success'],'quality':r['quality'],**{k:objectives[i] if objectives is not None and r['success'] else None for i,k in enumerate(('flow','cost','workload'))},'seconds':r['search_seconds']})
    if not rows: return []
    with (directory/'deliveries.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    groups=defaultdict(list)
    for r in rows: groups[(r['method'],r['instance_id'])].append(r)
    per_instance=[{'method':k[0],'instance_id':k[1],'cell_id':v[0]['cell_id'],'scale':v[0]['scale'],'requests':len(v),'success_rate':statistics.fmean(float(r['success']) for r in v),'mean_successful_quality':statistics.fmean(r['quality'] for r in v if r['success']) if any(r['success'] for r in v) else None} for k,v in groups.items()]
    with (directory/'instance_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(per_instance[0]));w.writeheader();w.writerows(per_instance)
    return per_instance


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');args=p.parse_args();summarize(args.directory)
