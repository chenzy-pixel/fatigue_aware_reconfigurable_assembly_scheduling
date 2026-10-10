"""Matched raw/excess MO-ALNS candidate generation and dual Flow evaluation."""
import argparse
from datetime import datetime
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.mo_alns import main as run_cli
from analysis.flow_evaluation import analyze_candidate_flow_runs


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--seed",type=int,default=11)
    args=parser.parse_args(); original=sys.argv[:]; runs={}
    prefix=f"flow_comparison_mo_alns_seed{args.seed}_{datetime.now():%Y%m%d_%H%M%S}"
    for mode,path in (("raw","configs/baselines/mo_alns.json"),("excess","configs/flow_excess/mo_alns.json")):
        name=f"{prefix}_{mode}"
        sys.argv=[original[0],"--config",path,"--dataset","test","--run-name",name,
                  "--algorithm-seed",str(args.seed)]
        if args.smoke: sys.argv += ["--smoke","--parallel-envs","1"]
        try: run_cli()
        finally: sys.argv=original
        runs[mode]=ROOT/"result/runs"/name
    analyze_candidate_flow_runs(runs,ROOT/"result/analysis"/prefix)


if __name__=="__main__": main()
