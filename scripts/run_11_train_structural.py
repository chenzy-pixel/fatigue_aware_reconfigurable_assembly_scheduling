"""Train the two structural ablations on the current Universal protocol."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.ablation_protocol import STRUCTURAL, train_group

if __name__ == "__main__":
    train_group(STRUCTURAL)
