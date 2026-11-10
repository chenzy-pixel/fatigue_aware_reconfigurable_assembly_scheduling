"""Audit and package the recorded experiment without loading model weights.

Build: python -m scripts.publish_results build --source-root <original checkout> --assets <directory>
Recompute: python -m scripts.publish_results recompute
Verify downloaded assets: python -m scripts.publish_results verify-assets --assets <directory>
"""
from __future__ import annotations
import argparse
import collections
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/current'
SEEDS = [11,23,37,53,71]
SOURCES = {
    'ablations': ('result/experiment_suite_v1', ['no_graph','no_direct','no_residual']),
    'main': ('result/experiment_suite_v1_parallel20', ['full','full_greedy','full_sample1','rule_ddqn']),
    'dcan': ('result/experiment_suite_v1_dcan_optimized', ['dcan']),
    'rules_search': ('result/received_20261103/experiment_suite_v1_parallel20', ['eft_lf','srw_lf','mc_mc','mwl_mwl','existing_heuristic','nsga2_ls','moead']),
}
SECRET = re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-(?:proj-)?[A-Za-z0-9_-]{35,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)')

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,default=str,ensure_ascii=False).encode()).hexdigest()

def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def portable(value):
    """Remove machine-specific absolute paths while preserving numeric evidence."""
    if isinstance(value,dict): return {k:portable(v) for k,v in value.items()}
    if isinstance(value,list): return [portable(v) for v in value]
    if isinstance(value,str):
        if SECRET.search(value): raise ValueError('credential-like value in publication input')
        if re.match(r'^[A-Za-z]:[\\/]',value) or value.startswith(('/root/','/home/','/workspace/')):
            path = value.replace('\\','/')
            for anchor in ('/result/','/data/','/configs/','/agent/','.cache/'):
                if anchor in path: return '<WORKSPACE>/'+path.split(anchor,1)[1] if anchor=='.cache/' else '<WORKSPACE>'+anchor+path.split(anchor,1)[1]
            return '<LOCAL>/'+path.rsplit('/',1)[-1]
    return value

def compact(unit):
    result = {k:v for k,v in unit.items() if k not in ('requests','searches','raw_rule','content_sha256')}
    result['original_content_sha256'] = unit['content_sha256']
    fields = ('success','objectives','quality','sampling_repeat','sampling_seed')
    result['requests'] = [{**{k:v for k,v in r.items() if k!='raw'},'raw':[{k:v for k,v in x.items() if k in fields} for x in r['raw']]} for r in unit.get('requests',[])]
    if 'raw_rule' in unit: result['raw_rule'] = {k:v for k,v in unit['raw_rule'].items() if k in (*fields,'seconds')}
    if 'searches' in unit:
        result['searches'] = [{k:([{kk:vv for kk,vv in x.items() if kk in ('objectives','action_trace_sha256')} for x in v] if k=='archive' else v)
            for k,v in s.items() if k not in ('candidate_history','search_log')} for s in unit['searches']]
    result = portable(result)
    result['content_sha256'] = digest(result)
    return result

def validate_selection(unit):
    """Verify the recorded best-of-k decision from the retained numeric evidence."""
    for delivery,request in zip(unit['deliveries'],unit.get('requests',[])):
        raw=request['raw']
        expected=3 if unit['arm'] in ('full','no_graph','no_direct','no_residual','dcan','a2c') else 1
        if len(raw)!=expected: raise ValueError('sampling repeat coverage differs')
        chosen=min(raw,key=lambda r:(not r['success'],r['quality'] if r['success'] else 1.,r['sampling_repeat']))
        if request['selected_repeat']!=chosen['sampling_repeat'] or delivery['success']!=chosen['success'] or delivery['objectives']!=chosen['objectives']:
            raise ValueError('selected candidate differs')
        if not math.isclose(delivery['quality'],chosen['quality'] if chosen['success'] else 1.,rel_tol=1e-8,abs_tol=1e-10):raise ValueError('selected quality differs')

class Parts:
    def __init__(self, directory): self.directory=directory; self.groups={}; self.files=[]
    def add(self, group, name, raw):
        if group not in self.groups or self.groups[group][1]+len(raw)>512*1024**2:
            previous=self.groups.get(group)
            if previous: previous[0].close()
            number=previous[2]+1 if previous else 1
            path=self.directory/f'{group}-{number:03d}.zip'
            self.groups[group]=[zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6),0,number]
            self.files.append(path)
        z=self.groups[group]; z[0].writestr(name,raw); z[1]+=len(raw)
    def close(self):
        for z,_,_ in self.groups.values(): z.close()

def build(source, assets):
    from analysis.experiment_suite import audit_evaluations
    from data import load_dataset_split
    assets.mkdir(parents=True,exist_ok=True)
    if list(assets.glob('*.zip')): raise ValueError('use an empty assets directory')
    OUT.mkdir(parents=True,exist_ok=True)
    details=OUT/'details'; details.mkdir(exist_ok=True)
    packs=Parts(assets); files=[]; origins=[]; all_compact=[]; training=[]; streams={}
    for label,(relative,methods) in SOURCES.items():
        base=source/relative
        pre=json.loads((base/'preflight.json').read_bytes())
        config=pre['configs']['full']
        # Frozen records are loaded from this cleaned checkout and checked by its manifests.
        records=list(load_dataset_split(config,'test'))
        meta={'label':label,'path':relative,'identity':pre['identity'],'preflight':portable(pre)}
        origins.append(meta)
        snap=base/'source.zip'
        with zipfile.ZipFile(snap) as z:
            identity=json.loads(z.read('experiment_identity.json'))
            if digest(identity)!=pre['identity'] or identity['sources']!=pre['source_hashes']:
                raise ValueError(f'source identity differs: {label}')
            for name,h in identity['sources'].items():
                if hashlib.sha256(z.read(name)).hexdigest()!=h: raise ValueError('frozen source hash differs')
            if z.testzip(): raise ValueError('source archive CRC')
        unique=assets/f'source-{pre["identity"][:12]}.zip'
        if not unique.exists():
            # Source snapshots are deduplicated, with machine paths removed from metadata.
            with zipfile.ZipFile(snap) as original, zipfile.ZipFile(unique,'w',zipfile.ZIP_DEFLATED) as target:
                for name in original.namelist():
                    raw=original.read(name)
                    if SECRET.search(raw.decode('utf-8',errors='replace')): raise ValueError('credential in source snapshot')
                    if name.endswith('.json'):
                        value=json.loads(raw); sanitized=portable(value)
                        if value!=sanitized: raw=json.dumps(sanitized,ensure_ascii=False,indent=2).encode()
                    target.writestr(name,raw)
            files.append({'source_path':relative+'/source.zip','archive':unique.name,'member':'*',
                          'original_sha256':sha(snap),'published_sha256':sha(unique),'kind':'source_snapshot'})
        configs = [m for m in methods if m in pre['suite']['arms']]
        for arm in configs:
            for seed in SEEDS:
                run=base/'training'/arm/str(seed)
                summary=json.loads((run/'summary.json').read_bytes())
                episodes=json.loads((run/'episodes.json').read_bytes())
                history=json.loads((run/'training_log.json').read_bytes())
                validations=json.loads((run/'validations.json').read_bytes())
                if not (summary.get('budget_completed') and summary['episodes_started']==2000 and summary['epochs']==100 and summary['validation_count']==20): raise ValueError('incomplete training')
                if [e['episode'] for e in episodes]!=list(range(2000)) or [e['epoch'] for e in history]!=list(range(1,101)) or [v['epoch'] for v in validations]!=list(range(5,101,5)): raise ValueError('training sequence differs')
                stream=[(e['record_sha256'],e['preference']) for e in episodes]
                if seed in streams:
                    for (h,p),(rh,rp) in zip(stream,streams[seed]):
                        if h!=rh or any(p.get(k)!=rp.get(k) for k in ('key','source','sample_index')) or any(abs(p['weights'][k]-rp['weights'][k])>1e-12 for k in ('flow','cost','workload')): raise ValueError('training streams differ')
                else: streams[seed]=stream
                training.append({'arm':arm,'seed':seed,'source':label,'source_identity':pre['identity'],**summary})
                folder=OUT/'training'/arm/str(seed);folder.mkdir(parents=True,exist_ok=True)
                # CSV is the canonical per-episode/objective record; avoid duplicate JSON exports.
                for name in ('episodes.csv','epoch_objectives.csv','training_log.csv','validation_instances.csv','validation_log.csv','validation_objectives.csv'):
                    raw=(run/name).read_bytes()
                    with gzip.GzipFile(filename='',mode='wb',fileobj=(folder/(name+'.gz')).open('wb'),mtime=0) as gz: gz.write(raw)
                write_json(folder/'summary.json',portable(summary)); write_json(folder/'config.json',portable(json.loads((run/'config.json').read_bytes())))
                # Preserve unique detailed validation evidence in Release, rather than duplicate CSV/JSON exports.
                for path in sorted((run/'validation_requests').glob('*.json')):
                    raw=path.read_bytes();value=json.loads(raw);new=portable(value)
                    published=raw if value==new else json.dumps(new,ensure_ascii=False,indent=2).encode()
                    member=path.relative_to(source).as_posix();group=f'validation-{arm}'
                    packs.add(group,member,published)
                    files.append({'source_path':member,'archive_group':group,'original_sha256':hashlib.sha256(raw).hexdigest(),'published_sha256':hashlib.sha256(published).hexdigest(),'bytes':len(published),'kind':'validation'})
        for path in sorted((base/'development').rglob('*.json')) if (base/'development').exists() else []:
            raw=path.read_bytes();v=portable(json.loads(raw));published=json.dumps(v,ensure_ascii=False,indent=2).encode()
            member=path.relative_to(source).as_posix();packs.add('search-development',member,published)
            files.append({'source_path':member,'archive_group':'search-development','original_sha256':hashlib.sha256(raw).hexdigest(),'published_sha256':hashlib.sha256(published).hexdigest(),'bytes':len(published),'kind':'search_development'})
        if (base/'search_budget.json').exists(): write_json(OUT/'search_budget.json',portable(json.loads((base/'search_budget.json').read_bytes())))
        compact_group=[]
        def units():
            paths=sorted((base/'evaluation').rglob('*.json'))
            for i,path in enumerate(paths,1):
                raw=path.read_bytes();u=json.loads(raw)
                if u['arm'] not in methods: raise ValueError('unexpected method')
                expected_seed='deterministic' if u['seed'] is None else str(u['seed'])
                if path.relative_to(base/'evaluation').parts[:3]!=(u['task'],u['arm'],expected_seed): raise ValueError('path metadata differs')
                if u.get('content_sha256')!=digest({k:v for k,v in u.items() if k!='content_sha256'}): raise ValueError('content hash differs: '+str(path))
                c=compact(u);compact_group.append(c)
                new=portable(u)
                # The original embedded content hash is retained as an original-evidence identifier.
                # The published byte hash in files.csv verifies the path-redacted artifact.
                published=raw if new==u else json.dumps(new,ensure_ascii=False,indent=2,allow_nan=False).encode()
                group=f'evaluation-{u["arm"]}-{u["task"]}'
                member=path.relative_to(source).as_posix();packs.add(group,member,published)
                files.append({'source_path':member,'archive_group':group,'original_sha256':hashlib.sha256(raw).hexdigest(),'original_content_sha256':u['content_sha256'],'published_sha256':hashlib.sha256(published).hexdigest(),'bytes':len(published),'kind':'evaluation'})
                if i%135==0: print(label,i,'/',len(paths),flush=True)
                yield u
        rows=audit_evaluations(units(),pre['suite'],config,records,pre['identity'],methods=methods)
        # Cross-check compact evidence through the same numerical and coverage audit.
        # Parent sample1 references use original content IDs, so restore them only during audit.
        for c in compact_group:
            if c.get('cost_reference')=='full':
                parent=next(x for x in compact_group if x['arm']=='full' and x['task']==c['task'] and x['seed']==c['seed'] and x['instance_id']==c['instance_id'])
                c['source_content_sha256']=parent['content_sha256'];c['content_sha256']=digest({k:v for k,v in c.items() if k!='content_sha256'})
        again=audit_evaluations(compact_group,pre['suite'],config,records,pre['identity'],methods=methods)
        if rows!=again: raise ValueError('compact delivery data differs')
        for arm in methods:
            for task in ('frontier','independent_request'):
                with gzip.GzipFile(filename='',mode='wb',fileobj=(details/f'{arm}-{task}.jsonl.gz').open('wb'),mtime=0) as gz:
                    for u in compact_group:
                        if u['arm']==arm and u['task']==task: gz.write((json.dumps(u,ensure_ascii=False,allow_nan=False)+'\n').encode())
        all_compact.extend(compact_group)
    packs.close()
    # Resolve each member to its actual independently extractable ZIP part.
    member_parts={}
    for path in packs.files:
        with zipfile.ZipFile(path) as z:
            if z.testzip(): raise ValueError('archive CRC differs')
            for name in z.namelist(): member_parts[name]=path.name
    for record in files:
        if 'archive_group' in record: record['archive']=member_parts[record['source_path']];record['member']=record['source_path'];del record['archive_group']
    from result.io import write_csv
    write_csv(OUT/'files.csv',files)
    write_json(OUT/'origins.json',origins);write_json(OUT/'training_summary.json',portable(training))
    inventory=[{'name':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(assets.glob('*.zip'))]
    write_json(OUT/'assets.json',{'tag':'experiments-20261110','repository':'chenzy-pixel/fatigue_aware_reconfigurable_assembly_scheduling','assets':inventory})
    write_json(OUT/'audit.json',{'status':'passed','evaluation_files':len(all_compact),'training_runs':len(training),'instances':27,'methods':sum(len(x[1]) for x in SOURCES.values()),'source_batches':len(origins),'checks':['source_snapshot_hashes','original_content_hashes','identity_status','instance_metadata','delivery_matrix','quality_from_objectives','paid_and_delivery_hv','sample1_parent_reuse','compact_equivalence','training_budget_sequences','training_stream_semantics','archive_crc'], 'path_redaction':'Original embedded content hashes identify original artifacts; files.csv contains original and published byte hashes. Numeric evidence is unchanged.','training_sufficiency':'not_established','search_sufficiency':'not_established','model_weights':'external local files required to rerun learned evaluations'})
    recompute()
    print('build complete',len(all_compact),'units',len(inventory),'assets',flush=True)

def recompute():
    from analysis.experiment_suite import instance_tables,descriptive_tables,paired_statistics,common_success_statistics,audit_evaluations
    from data import load_dataset_split
    from result.io import write_csv
    from configs.experiment_suite import load_suite
    units=[];rows=[]
    metadata={p.stem:json.loads(p.read_bytes())['metadata'] for p in (ROOT/'data/instances/v3/test').glob('C*.json') if '.certificate.' not in p.name}
    for p in sorted((OUT/'details').glob('*.jsonl.gz')):
        with gzip.open(p,'rt',encoding='utf-8') as f:
            for line in f:
                u=json.loads(line)
                if u['content_sha256']!=digest({k:v for k,v in u.items() if k!='content_sha256'}): raise ValueError('compact content hash differs')
                validate_selection(u)
                units.append(u)
                for d in u['deliveries']:
                    # N/K/M come from the frozen factorial cell metadata.
                    meta=metadata[u['cell_id']]
                    rows.append({'task':u['task'],'arm':u['arm'],'seed':u['seed'],'instance_id':u['instance_id'],'cell_id':u['cell_id'],'scale':u['scale'],**{k:meta[k] for k in ('N','K','M')},'preference':json.dumps(d['preference']),'success':d['success'],'quality':d['quality'],**{name:d['objectives'][i] if d['success'] else None for i,name in enumerate(('flow','cost','workload'))},'seconds':d['search_seconds'],'timing_basis':d.get('timing_basis','measured_request')})
    origins=json.loads((OUT/'origins.json').read_bytes())
    for origin in origins:
        methods=SOURCES[origin['label']][1];pre=origin['preflight'];config=pre['configs']['full']
        audit_evaluations((u for u in units if u['arm'] in methods),pre['suite'],config,list(load_dataset_split(config,'test')),origin['identity'],methods=methods)
    instances=instance_tables(rows,units);summary,seeds=descriptive_tables(instances,rows);suite=load_suite()
    out=OUT/'tables';out.mkdir(exist_ok=True)
    for name,table in [('per_instance',instances),('summary',summary),('seed_results',seeds),('paired_statistics',paired_statistics(instances,suite)),('common_success',common_success_statistics(rows,suite))]:write_csv(out/(name+'.csv'),table)
    plot(instances,summary)
    print('recomputed',len(units),'units',len(rows),'deliveries',flush=True)

def plot(instances,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=OUT/'figures';out.mkdir(exist_ok=True)
    methods=sorted({r['arm'] for r in instances})
    fig,ax=plt.subplots(figsize=(10,4.8))
    vals=[next(r['quality'] for r in summary if r['arm']==m and r['task']=='independent_request' and r['factor']=='all') for m in methods]
    ax.bar(methods,vals);ax.tick_params(axis='x',rotation=55);ax.set_ylabel('Mean public Q (lower is better)');fig.tight_layout()
    for ext in ('svg','png','pdf'):fig.savefig(out/f'independent_quality.{ext}',dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.8))
    vals=[next(r['paid_hv'] for r in summary if r['arm']==m and r['task']=='frontier' and r['factor']=='all') for m in methods]
    ax.bar(methods,vals);ax.tick_params(axis='x',rotation=55);ax.set_ylabel('Mean paid HV (higher is better)');fig.tight_layout()
    for ext in ('svg','png','pdf'):fig.savefig(out/f'frontier_paid_hv.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4.5))
    for arm in ['full','no_graph','no_direct','no_residual','rule_ddqn','dcan']:
        histories=[]
        for seed in SEEDS:
            with gzip.open(OUT/'training'/arm/str(seed)/'training_log.csv.gz','rt',encoding='utf-8-sig') as f:histories.append(list(csv.DictReader(f)))
        for ax,key in zip(axes,('flow_mean','cost_mean','workload_mean')):
            ax.plot(range(1,101),[sum(float(h[i][key] or 'nan') for h in histories)/len(histories) for i in range(100)],label=arm)
            ax.set(xlabel='Epoch',ylabel=key)
    axes[-1].legend(fontsize=8);fig.tight_layout()
    for ext in ('svg','png','pdf'):fig.savefig(out/f'training_objectives.{ext}',dpi=180)
    plt.close(fig)

def verify_assets(directory):
    manifest=json.loads((OUT/'assets.json').read_bytes())
    for a in manifest['assets']:
        path=directory/a['name']
        if path.stat().st_size!=a['bytes'] or sha(path)!=a['sha256']:raise ValueError('asset differs: '+a['name'])
    print('verified',len(manifest['assets']),'assets')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['build','recompute','verify-assets']);p.add_argument('--source-root',type=Path);p.add_argument('--assets',type=Path)
    a=p.parse_args()
    if a.action=='build':build(a.source_root.resolve(),a.assets.resolve())
    elif a.action=='recompute':recompute()
    else:verify_assets(a.assets.resolve())

if __name__=='__main__':main()
