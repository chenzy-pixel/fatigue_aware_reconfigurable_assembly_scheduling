"""Run the remaining raw-Flow mainline Variance endpoint, seed11/1000 episodes."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from environment.types import FLOW_RAW
from scripts.run_flow_excess_training import launch_argument_parser, launch_configurations

CONFIGS = {"main_variance": "configs/e1/single_variance.json"}


def main(*, preflight_only_default=False):
    args = launch_argument_parser(preflight_only_default=preflight_only_default).parse_args()
    launch_configurations(CONFIGS, args, prefix="main_raw", allowed_flow_modes=(FLOW_RAW,))


if __name__ == "__main__":
    main()
