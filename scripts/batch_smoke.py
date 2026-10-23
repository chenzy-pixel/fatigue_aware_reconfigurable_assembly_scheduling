"""Bounded local integration: exact 24-step learning arms and paid search probes."""
import json
from pathlib import Path
import torch
from configs import load_config
from data import load_dataset_split
from agent.comparators.learning import train_request
from scripts.comparison_suite import search_job,study_identity


def main():
    torch.set_num_threads(1)
    config=load_config('configs/default.json');config['device']='cpu'
    config['comparison']={'ddqn':{'warmup':8,'batch_size':8,'update_every':4}}
    data=load_dataset_split(config,'validation');identity=study_identity(config)
    output=Path('result/migration/factorial27_20261023/smoke_final')
    rows={}
    for method in ('ppo','dcan','rule_ddqn'):
        rows[method]=train_request(config,method,11,1e-4,24,data,[0],[(1/3,1/3,1/3)],output/method,identity,workers=2)
        from agent.comparators.learning import build_agent
        from agent.comparators.common import method_config
        from environment import AssemblySchedulingEnv
        agent=build_agent(method_config(config,method,11),AssemblySchedulingEnv(config).reset(data[0].instance),method)
        agent.load(output/method/'best_checkpoint.pt')
        print(method,'24 paid steps and checkpoint reload complete',flush=True)
    for method in ('mo_alns','nsga2_ls','moead'):
        rows[method]=search_job((config,data[0],(1/3,1/3,1/3),method,11,None,8,output/f'{method}.json',identity))
        print(method,'search probe complete',flush=True)
    (output/'summary.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')


if __name__=='__main__': main()
