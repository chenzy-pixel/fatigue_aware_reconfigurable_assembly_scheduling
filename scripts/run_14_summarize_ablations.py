"""Summarize matched evaluation cells for the five ablation comparisons."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.ablation_analysis import main

if __name__ == "__main__":
    main()
