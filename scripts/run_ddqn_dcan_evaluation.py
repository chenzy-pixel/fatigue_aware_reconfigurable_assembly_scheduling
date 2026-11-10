"""Evaluate the saved Rule-DDQN and DCAN models in their original result roots.

Run with the training Python:
    python scripts/run_ddqn_dcan_evaluation.py
    python scripts/run_ddqn_dcan_evaluation.py --prepare-only

Each method delegates frozen-source checks and both evaluation tasks to
scripts/resume_full_evaluation.py, using the saved seeds and device.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIRS = {
    "rule_ddqn": Path("result/experiment_suite_v1_parallel20"),
    "dcan": Path("result/experiment_suite_v1_dcan_optimized"),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=tuple(DEFAULT_OUTPUT_DIRS),
                        default=list(DEFAULT_OUTPUT_DIRS),
                        help="Methods in launch order (default: rule_ddqn dcan).")
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT,
                        help="Project containing scripts/, original data/ and result/.")
    parser.add_argument("--rule-ddqn-output-dir", type=Path, default=DEFAULT_OUTPUT_DIRS["rule_ddqn"],
                        help="Original Rule-DDQN result root; relative paths use --project-root.")
    parser.add_argument("--dcan-output-dir", type=Path, default=DEFAULT_OUTPUT_DIRS["dcan"],
                        help="Original DCAN result root; relative paths use --project-root.")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Prepare and check frozen files for each selected method.")
    args = parser.parse_args(argv)
    if len(set(args.methods)) != len(args.methods):
        parser.error("Selected methods must be unique.")

    project = args.project_root.resolve()
    helper = project / "scripts/resume_full_evaluation.py"
    configured = {"rule_ddqn": args.rule_ddqn_output_dir, "dcan": args.dcan_output_dir}
    outputs = [(method, (path if path.is_absolute() else project / path).resolve())
               for method in args.methods for path in [configured[method]]]
    required = [helper]
    required.extend(output / name for _, output in outputs for name in ("preflight.json", "source.zip"))
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error("Missing evaluation inputs:\n  " + "\n  ".join(missing))

    for index, (method, output) in enumerate(outputs, 1):
        print(f"[{index}/{len(outputs)}] {method} evaluation: {output / 'evaluation'}", flush=True)
        command = [sys.executable, "-u", str(helper), "--project-root", str(project),
                   "--output-dir", str(output), "--arms", method]
        if args.prepare_only:
            command.append("--prepare-only")
        subprocess.run(command, cwd=project, check=True, stdout=sys.stdout, stderr=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"STOP: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
