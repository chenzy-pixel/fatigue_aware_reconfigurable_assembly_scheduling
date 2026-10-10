"""Check both modified experiments without starting formal training."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_flow_excess_training import main


if __name__ == "__main__":
    main(preflight_only_default=True)
