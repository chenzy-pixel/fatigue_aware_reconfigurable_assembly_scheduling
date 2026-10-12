"""Preflight and launch the geometric OWA workload experiments."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from environment.types import FLOW_RAW, FLOW_EXCESS
from scripts.run_flow_excess_training import launch_argument_parser, launch_configurations


def main() -> None:
    parser = launch_argument_parser()
    parser.add_argument("--experiment", choices=("workload", "universal", "variance", "all"), default="all")
    parser.add_argument("--q", type=float, choices=(.25, .5, .75), default=.5)
    parser.add_argument("--flow-mode", choices=("raw", "excess"), default="excess")
    args = parser.parse_args()
    folder = { .25: "q025", .5: "q05", .75: "q075" }[args.q]
    base = "configs/workload/" + ("raw/" if args.flow_mode == "raw" else "") + folder
    choices = {"workload": f"{base}/single_workload.json", "universal": f"{base}/universal.json",
               "variance": "configs/workload/raw/single_variance.json" if args.flow_mode == "raw" else "configs/flow_excess/shared_head/single_variance.json"}
    selected = choices if args.experiment == "all" else {args.experiment: choices[args.experiment]}
    launch_configurations(selected, args, prefix=f"workload_{args.flow_mode}_{folder}",
                          allowed_flow_modes=(FLOW_RAW if args.flow_mode == "raw" else FLOW_EXCESS,))


if __name__ == "__main__":
    main()
