param(
    [string]$Python = 'python',
    [switch]$PreflightOnly
)
$ErrorActionPreference = 'Stop'
# Run from the repository root. Uses the existing launcher on the remote machine.
$taskCode = @'
import hashlib, json, sys
from pathlib import Path
from configs import load_config
from configs.config import public_config
from scripts.run_flow_excess_training import launch_argument_parser, launch_configurations

base = 'configs/observation/screening/owa_log2.json'
config = public_config(load_config(base))
digest = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
if digest != 'b6e537c1484d3efd4b7ca16ac6f0e7dd2c836b65c1c606547382a62e0791be22':
    raise SystemExit('Baseline configuration differs from the completed OWA/log2 experiment; stopped before training.')
preset = {'extends': 'owa_log2.json', 'experiment_name': 'schema12_screen_shared_excess_owa_log2_lambda097', 'ppo': {'gamma': 1.0, 'gae_lambda': 0.97}}
path = Path('configs/observation/screening/owa_log2_lambda097.json')
if path.exists() and json.loads(path.read_text(encoding='utf-8-sig')) != preset:
    raise SystemExit('An existing lambda097 preset differs; stopped without overwriting it.')
if not path.exists():
    path.write_text(json.dumps(preset, indent=2) + '\n', encoding='utf-8')
jobs = [('owa_log2_lambda097', str(path), 11, .97), ('owa_log2', base, 23, .95), ('owa_log2', base, 37, .95)]
parser = launch_argument_parser()
for preflight in ([True] if '--preflight-only' in sys.argv else [True, False]):
    for label, preset_path, seed, lam in jobs:
        args = parser.parse_args(['--seed', str(seed)] + (['--preflight-only'] if preflight else []))
        print(f'FOLLOWUP: seed={seed}, gamma=1.0, lambda={lam}, episodes=2000, preflight_only={preflight}', flush=True)
        launch_configurations({label: preset_path}, args, prefix='schema12_screen')
'@
$taskArgs = @('-c', $taskCode)
if ($PreflightOnly) { $taskArgs += '--preflight-only' }
& $Python @taskArgs
if ($LASTEXITCODE -ne 0) { throw "Experiment queue stopped with exit code $LASTEXITCODE" }
