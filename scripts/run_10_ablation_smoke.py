"""Run a bounded PPO update and sampled checks for all five ablation variants."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.ablation_protocol import VARIANTS, train_group

if __name__ == "__main__":
    train_group(VARIANTS, smoke=True)
