"""Evaluate matched baseline and ablation best checkpoints."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.ablation_protocol import evaluate_group

if __name__ == "__main__":
    evaluate_group()
