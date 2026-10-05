"""Train three matched full-fatigue and neutral single-objective pairs."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.ablation_protocol import FATIGUE, train_group

def main() -> None:
    train_group(FATIGUE)


if __name__ == "__main__":
    main()
