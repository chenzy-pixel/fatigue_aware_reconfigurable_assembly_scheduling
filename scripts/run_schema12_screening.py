"""Launch the initial Variance/OWA pair or a selected encoding/GAE control."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_flow_excess_training import launch_argument_parser, launch_configurations


CONFIGS = {
    "variance_log2": "configs/observation/screening/variance_log2.json",
    "owa_log2": "configs/observation/screening/owa_log2.json",
    "variance_bounded": "configs/observation/screening/variance_bounded.json",
    "owa_bounded": "configs/observation/screening/owa_bounded.json",
    "owa_log2_lambda099": "configs/observation/screening/owa_log2_lambda099.json",
    "owa_log2_lambda097": "configs/observation/screening/owa_log2_lambda097.json",
}
INITIAL_PAIR = ("variance_log2", "owa_log2")


def main():
    parser = launch_argument_parser()
    parser.description = "Schema12 shared-head excess screening: initial Universal pair or one encoding/GAE control."
    parser.add_argument("--experiment", choices=("pair", *CONFIGS), default="pair",
                        help="pair runs Variance/log2 and OWA/log2 sequentially")
    args = parser.parse_args()
    labels = INITIAL_PAIR if args.experiment == "pair" else (args.experiment,)
    launch_configurations({label: CONFIGS[label] for label in labels}, args,
                          prefix="schema12_screen")


if __name__ == "__main__":
    main()
