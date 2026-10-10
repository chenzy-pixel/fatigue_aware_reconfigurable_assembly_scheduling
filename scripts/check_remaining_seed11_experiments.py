"""Validate the seven remote experiments without launching training."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_remaining_seed11_experiments import main


if __name__ == "__main__":
    main(preflight_only_default=True)
