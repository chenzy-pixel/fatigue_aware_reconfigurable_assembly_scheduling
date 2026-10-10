"""Matched full raw/excess training and evaluation entrypoint."""
from pathlib import Path
from datetime import datetime
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from configs import load_config
from train import train
from analysis.v8_pareto_analysis import analyze_dual_flow_runs
from analysis.flow_evaluation import analyze_candidate_flow_runs


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--objective",choices=("universal","flow"),default="universal")
    parser.add_argument("--seed",type=int,default=11)
    args=parser.parse_args()
    paths=(("raw","configs/v8/universal.json"),("excess","configs/flow_excess/universal.json")) if args.objective=="universal" else (
        ("raw","configs/e1/single_flow.json"),("excess","configs/flow_excess/single_flow.json"))
    runs={}
    name=f"flow_comparison_{args.objective}_seed{args.seed}_{datetime.now():%Y%m%d_%H%M%S}"
    for mode,path in paths:
        runs[mode]=train(load_config(path),algorithm_seed=args.seed,
                        run_name=f"{name}_{mode}")
    output=ROOT/"result/analysis"/name
    if args.objective=="universal":
        analyze_dual_flow_runs(runs,output)
    else:
        analyze_candidate_flow_runs(runs,output)


if __name__=="__main__": main()
