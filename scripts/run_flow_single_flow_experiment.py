"""Run the matched single-objective raw/excess Flow experiment."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_flow_normalization_experiment import main


if __name__ == "__main__":
    sys.argv[1:1] = ["--objective", "flow"]
    main()
