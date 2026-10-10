"""Run new-normalization shared-head endpoints and Universal sequentially."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_flow_excess_training import launch_argument_parser, launch_configurations

CONFIGS = {
    "shared_head_flow": "configs/flow_excess/shared_head/single_flow.json",
    "shared_head_cost": "configs/flow_excess/shared_head/single_cost.json",
    "shared_head_variance": "configs/flow_excess/shared_head/single_variance.json",
    "shared_head_universal": "configs/flow_excess/shared_head/universal.json",
}


def main(*, preflight_only_default=False):
    parser = launch_argument_parser(preflight_only_default=preflight_only_default)
    parser.add_argument("--experiment", choices=("flow", "cost", "variance", "universal", "all"), default="all")
    args = parser.parse_args()
    roles = tuple(CONFIGS) if args.experiment == "all" else (f"shared_head_{args.experiment}",)
    launch_configurations({role: CONFIGS[role] for role in roles}, args, prefix="shared_head_flow_excess")


if __name__ == "__main__":
    main()
