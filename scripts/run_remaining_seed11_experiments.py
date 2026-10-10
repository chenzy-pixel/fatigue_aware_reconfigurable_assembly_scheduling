"""Run the seven experiments following the current remote mainline Cost run."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from environment.types import FLOW_RAW, FLOW_EXCESS
from scripts.run_flow_excess_training import launch_argument_parser, launch_configurations
from scripts.run_shared_head_flow_excess import CONFIGS as SHARED_CONFIGS

CONFIGS = {
    "main_variance": "configs/e1/single_variance.json",
    "main_universal": "configs/v8/universal.json",
    "flow": "configs/flow_excess/single_flow.json",
    **SHARED_CONFIGS,
}


def main(*, preflight_only_default=False):
    args = launch_argument_parser(preflight_only_default=preflight_only_default).parse_args()
    launch_configurations(CONFIGS, args, prefix="remaining_flow_experiments",
                          allowed_flow_modes=(FLOW_RAW, FLOW_EXCESS))


if __name__ == "__main__":
    main()
