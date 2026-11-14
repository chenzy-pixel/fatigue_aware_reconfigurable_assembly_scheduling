"""Train vanilla DQN seeds 11/23/37/71, then frozen A2C seed 71.

Run with the remote training Python: python scripts/run_dqn_a2c_training.py
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DQN_SEEDS = (11, 23, 37, 71)
A2C_SEEDS = (71,)
DQN_OVERLAYS = (
    'agent/comparators/dqn.py', 'scripts/train_dqn.py',
)

VERIFY = """
import sys
from pathlib import Path
import torch
from configs.experiment_suite import load_suite, suite_identity
from scripts.experiment_suite import check_preflight
expected, output, device = sys.argv[1:]
suite = load_suite()
torch.set_num_threads(suite['execution']['torch_threads'])
actual = suite_identity(suite, device)
if actual != expected:
    raise ValueError('Frozen runtime identity differs: ' + actual + ' != ' + expected)
check_preflight(Path(output), actual)
print('Original frozen training identity: passed', flush=True)
"""


def extend_dqn_runtime(project, frozen):
    """Extend the original source rather than importing newer training dependencies."""
    exact = {
        'agent/comparators/learning.py': [
            ('from .ddqn import RuleDDQN', 'from .ddqn import RuleDDQN\nfrom .dqn import RuleDQN'),
            ('"dcan", "rule_ddqn")', '"dcan", "rule_ddqn", "dqn")'),
            ('return RuleDDQN(obs, config)', 'return (RuleDQN if method == "dqn" else RuleDDQN)(obs, config)'),
        ],
        'agent/comparators/ddqn.py': [
            ('class RuleDDQN:\n', 'class RuleDDQN:\n    family = "rule_ddqn_v2"\n'),
            ('"family": "rule_ddqn_v2"', '"family": self.family'),
            ('target = double_q_targets(reward, done, self.online(following), self.target(following), mask)',
             'target = self.bootstrap_targets(reward, done, following, mask)'),
            ('    def save(self, path, metadata=None):',
             '    def bootstrap_targets(self, reward, done, following, mask):\n'
             '        return double_q_targets(reward, done, self.online(following), self.target(following), mask)\n\n'
             '    def save(self, path, metadata=None):'),
        ],
    }
    names = ('agent/comparators/__init__.py', 'agent/comparators/common.py',
             'agent/comparators/ddqn.py', 'agent/comparators/learning.py',
             'agent/comparators/parallel_evaluation.py', 'configs/runtime.py', 'result/console_reporter.py')
    for name in names:
        path = frozen/name
        text = path.read_text(encoding='utf-8')
        for old, new in exact.get(name, []):
            if text.count(old) != 1:
                raise ValueError(f'Unsupported frozen DQN extension point: {name}: {old}')
            text = text.replace(old, new)
        text = text.replace('"rule_ddqn", "nsga2_ls"', '"rule_ddqn", "dqn", "nsga2_ls"')
        text = text.replace('{"dcan", "rule_ddqn"}', '{"dcan", "rule_ddqn", "dqn"}')
        for quote in ("'", '"'):
            text = text.replace(f'method == {quote}rule_ddqn{quote}', f'method in ({quote}rule_ddqn{quote}, {quote}dqn{quote})')
            text = text.replace(f'method != {quote}rule_ddqn{quote}', f'method not in ({quote}rule_ddqn{quote}, {quote}dqn{quote})')
        path.write_text(text, encoding='utf-8')
    for name in DQN_OVERLAYS:
        destination = frozen/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(project/name, destination)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--methods', nargs='+', choices=('dqn', 'a2c'), default=['dqn', 'a2c'])
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--dqn-base-output-dir', type=Path, default=Path('result/experiment_suite_v1_parallel20'))
    parser.add_argument('--dqn-output-dir', type=Path, default=Path('result/experiment_suite_v1_dqn'))
    parser.add_argument('--a2c-output-dir', type=Path, default=Path('result/experiment_suite_v1'))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare-only', action='store_true', help='Check archived files; skip Torch and training.')
    mode.add_argument('--check-only', action='store_true', help='Verify runtime and DQN preflight; skip training.')
    args = parser.parse_args(argv)
    if len(set(args.methods)) != len(args.methods):
        parser.error('Methods must be distinct.')
    project = args.project_root.resolve()
    sys.path.insert(0, str(project))
    from scripts.resume_full_evaluation import prepare, read

    def resolve(path):
        return (path if path.is_absolute() else project/path).resolve()

    sources = {'dqn': resolve(args.dqn_base_output_dir), 'a2c': resolve(args.a2c_output_dir)}
    dqn_output = resolve(args.dqn_output_dir)
    if 'dqn' in args.methods and any(dqn_output == path or dqn_output.is_relative_to(path)
                                   or path.is_relative_to(dqn_output) for path in sources.values()):
        parser.error('DQN output must be separate from both original result roots.')
    required = [project/name for name in DQN_OVERLAYS] if 'dqn' in args.methods else []
    required += [sources[method]/name for method in args.methods for name in ('preflight.json', 'source.zip')]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error('Missing training inputs:\n  ' + '\n  '.join(missing))

    # Prepare every selected input before starting the first long-running job.
    jobs = []
    for method in args.methods:
        arm = 'rule_ddqn' if method == 'dqn' else 'a2c'
        frozen, identity, seeds = prepare(project, sources[method], [arm], training=True)
        requested = DQN_SEEDS if method == 'dqn' else A2C_SEEDS
        if any(seed not in seeds for seed in requested):
            raise ValueError(f'Saved {method} suite does not contain the requested seeds.')
        device = read(frozen/'experiment_identity.json')['configs']['full'].get('device', 'cuda')
        jobs.append((method, frozen, identity, device, requested))
        print(f'{method}: seeds={requested} | frozen={frozen}', flush=True)
    if args.prepare_only:
        print('Archive/data checks passed. Runtime checks and training were not executed.', flush=True)
        return
    # Verify all original identities before changing the DQN runtime or training.
    for method, frozen, identity, device, seeds in jobs:
        subprocess.run([sys.executable, '-u', '-c', VERIFY, identity, str(sources[method]), device],
                       cwd=frozen, check=True)
    for method, frozen, identity, device, seeds in jobs:
        if method == 'dqn':
            extend_dqn_runtime(project, frozen)
            command = [sys.executable, '-u', '-m', 'scripts.train_dqn', '--output-dir', str(dqn_output),
                       '--base-identity', identity, '--device', device, '--seeds', *map(str, seeds)]
            if args.check_only:
                command.append('--check-only')
        elif args.check_only:
            continue
        else:
            command = [sys.executable, '-u', '-m', 'scripts.experiment_suite', '--stage', 'train',
                       '--output-dir', str(sources[method]), '--device', device,
                       '--arms', 'a2c', '--seeds', *map(str, seeds), '--remote-run']
        print(f'Launching {method}: seeds={seeds}', flush=True)
        subprocess.run(command, cwd=frozen, check=True)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        print(f'STOP: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1)
