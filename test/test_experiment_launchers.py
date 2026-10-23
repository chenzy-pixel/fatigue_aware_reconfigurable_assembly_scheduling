"""Exercise the three real PowerShell entrypoints without launching experiments."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
SHELL = shutil.which('pwsh') or shutil.which('powershell')
pytestmark = pytest.mark.skipif(SHELL is None, reason='PowerShell unavailable')

GROUPS = {
    'run_main_experiment.ps1': (['full','rule_ddqn','dcan'], ['preflight','train','evaluate']),
    'run_ablations_a2c.ps1': (['no_graph','no_direct','no_residual','a2c'], ['preflight','train','evaluate']),
    'run_rules_search.ps1': ([], ['preflight','rules','develop-search','freeze-budget','search']),
}


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def invoke(script, arguments, *, cwd=None, env=None, json_output=False):
    command = f"[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false); & {literal(ROOT/'scripts'/script)} {arguments}"
    if json_output:
        command += ' | ConvertTo-Json -Depth 8 -Compress'
    return subprocess.run([SHELL,'-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-Command',command],
                          capture_output=True,text=True,encoding='utf-8',errors='replace',
                          cwd=cwd or ROOT,env=env,timeout=45)


@pytest.mark.parametrize('script',GROUPS)
def test_dry_run_groups_and_stage_order(script,tmp_path):
    result = invoke(script, '-DryRun', cwd=tmp_path, json_output=True)
    assert result.returncode == 0, result.stderr
    jobs = json.loads(result.stdout)
    arms, stages = GROUPS[script]
    assert [j['Stage'] for j in jobs] == stages
    for job in jobs:
        assert job['Parameters'].get('Arms',[]) == arms
        assert job['Parameters']['OutputDir'] == 'result/experiment_suite_v1'
        assert job['Parameters']['Task'] == 'both'
        assert job['Parameters']['TrainingWorkers'] == 20


def capture_interpreter(tmp_path):
    folder = tmp_path/"launcher's folder with spaces"
    folder.mkdir()
    interpreter, log = folder/'capture.ps1', folder/'arguments.jsonl'
    interpreter.write_text('''$stageIndex = [Array]::IndexOf($args, '--stage')
$record = @{ Arguments=@($args); Directory=(Get-Location).Path }
Add-Content -LiteralPath $env:SUITE_LAUNCH_CAPTURE -Value ($record | ConvertTo-Json -Depth 5 -Compress) -Encoding utf8
if ($env:SUITE_LAUNCH_FAIL -and $args[$stageIndex + 1] -eq $env:SUITE_LAUNCH_FAIL) { exit 9 }
$global:LASTEXITCODE = 0
''',encoding='utf-8')
    return interpreter,log


def option(arguments,key):
    return arguments[arguments.index(key)+1]


def array_option(arguments,key):
    if key not in arguments: return []
    start=arguments.index(key)+1
    end=next((i for i in range(start,len(arguments)) if arguments[i].startswith('--')),len(arguments))
    return arguments[start:end]


@pytest.mark.parametrize('script',GROUPS)
def test_real_parameter_forwarding_with_capture_interpreter(script,tmp_path):
    interpreter,log = capture_interpreter(tmp_path)
    output = tmp_path/"results with spaces and 'quote'"
    manifest = tmp_path/'manifest with spaces.json'
    env = {**os.environ,'SUITE_LAUNCH_CAPTURE':str(log)}
    result = invoke(script, f'-RemoteRun -Python {literal(interpreter)} -OutputDir {literal(output)} '
                    f'-Manifest {literal(manifest)} -Device cpu -Seeds 11,37 -TrainingWorkers 4 -Task independent_request',
                    cwd=tmp_path,env=env)
    assert result.returncode == 0, result.stderr
    records=[json.loads(line) for line in log.read_text(encoding='utf-8-sig').splitlines()]
    arms,stages=GROUPS[script]
    assert [option(r['Arguments'],'--stage') for r in records] == stages
    for row in records:
        args=row['Arguments']
        assert args[:3] == ['-u','-m','scripts.experiment_suite']
        assert option(args,'--output-dir') == str(output)
        assert option(args,'--manifest') == str(manifest)
        assert option(args,'--device') == 'cpu'
        assert option(args,'--training-workers') == '4'
        assert option(args,'--task') == 'independent_request'
        assert array_option(args,'--seeds') == ['11','37']
        assert array_option(args,'--arms') == arms
        assert '--remote-run' in args
        assert Path(row['Directory']).resolve() == ROOT


@pytest.mark.parametrize('script',GROUPS)
def test_formal_execution_requires_remote_flag_before_any_stage(script,tmp_path):
    interpreter,log = capture_interpreter(tmp_path)
    result=invoke(script,f'-Python {literal(interpreter)}',env={**os.environ,'SUITE_LAUNCH_CAPTURE':str(log)})
    assert result.returncode != 0 and '-RemoteRun' in result.stderr
    assert not log.exists()


def test_failed_training_stops_before_evaluation(tmp_path):
    interpreter,log=capture_interpreter(tmp_path)
    env={**os.environ,'SUITE_LAUNCH_CAPTURE':str(log),'SUITE_LAUNCH_FAIL':'train'}
    result=invoke('run_main_experiment.ps1',f'-RemoteRun -Python {literal(interpreter)}',env=env)
    assert result.returncode != 0
    records=[json.loads(line) for line in log.read_text(encoding='utf-8-sig').splitlines()]
    assert [option(r['Arguments'],'--stage') for r in records] == ['preflight','train']


def test_report_is_explicit_and_group_training_matrix_complete():
    result=invoke('run_rules_search.ps1','-Stage report -DryRun',json_output=True)
    assert result.returncode == 0,result.stderr
    assert json.loads(result.stdout)['Stage'] == 'report'
    from configs.experiment_suite import load_suite
    arms=GROUPS['run_main_experiment.ps1'][0]+GROUPS['run_ablations_a2c.ps1'][0]
    suite=load_suite()
    assert len(arms)==len(set(arms))==7 and set(arms)==set(suite['arms'])
    assert len(arms)*len(suite['seeds'])==35


def test_launchers_in_frozen_source_manifest():
    from configs.experiment_suite import load_suite, identity_payload
    payload=identity_payload(load_suite(),'cpu')
    assert all('scripts/'+name in payload['sources'] for name in GROUPS)
    assert 'scripts/ExperimentLaunch.psm1' in payload['sources']


def test_source_identity_tracks_entry_modules_and_ignores_result_caches(tmp_path,monkeypatch):
    import configs.experiment_suite as suite
    (tmp_path/'utils.py').write_text('RNG_VERSION=1\n',encoding='utf-8')
    (tmp_path/'result').mkdir()
    (tmp_path/'result/runtime.py').write_text('VERSION=1\n',encoding='utf-8')
    monkeypatch.setattr(suite,'PROJECT_ROOT',tmp_path)
    spec=suite.load_suite()
    before=suite.identity_payload(spec,'cpu')
    cache=tmp_path/'result/analysis/cached_main';cache.mkdir(parents=True)
    (cache/'old.py').write_text('OLD=1\n',encoding='utf-8')
    after=suite.identity_payload(spec,'cpu')
    assert before==after
    assert set(after['sources'])=={'utils.py','result/runtime.py'}
    (tmp_path/'utils.py').write_text('RNG_VERSION=2\n',encoding='utf-8')
    assert suite.identity_payload(spec,'cpu')!=after
