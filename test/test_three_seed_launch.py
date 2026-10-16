"""Exercise the PowerShell batch boundary without starting training processes."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from configs import load_config, project_path


PWSH = shutil.which("pwsh") or shutil.which("powershell")
pytestmark = pytest.mark.skipif(os.name != "nt" or not PWSH, reason="Windows PowerShell launcher")
SCRIPT = project_path("scripts/run_three_seeds.ps1")


def quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def invoke(tmp_path, arguments, *, fail_seed=None):
    # Intercept Python's -m dispatch at startup; never import the training module.
    (tmp_path / "sitecustomize.py").write_text(
        "import json,os,runpy,sys\n"
        "def dispatch(module, alter_argv=True):\n"
        "    assert module == 'scripts.run_experiments', module\n"
        "    args = sys.argv[1:]\n"
        "    with open(os.environ['LAUNCH_TEST_LOG'], 'a', encoding='utf-8') as stream:\n"
        "        stream.write(json.dumps({'args':args,'cwd':os.getcwd()})+'\\n')\n"
        "    seed = args[args.index('--seed')+1]\n"
        "    raise SystemExit(7 if seed == os.environ.get('LAUNCH_TEST_FAIL_SEED') else 0)\n"
        "runpy._run_module_as_main = dispatch\n",
        encoding="utf-8",
    )
    log = tmp_path / "calls.jsonl"
    env = dict(os.environ, PYTHONPATH=str(tmp_path), LAUNCH_TEST_LOG=str(log),
               LAUNCH_TEST_FAIL_SEED=str(fail_seed or ""))
    command = (
        "$originalLocation = (Get-Location).Path; "
        f"try {{ & {quote(SCRIPT)} -Python {quote(sys.executable)} {arguments} }} "
        "catch { Write-Output $_.Exception.Message; "
        "if ((Get-Location).Path -ne $originalLocation) { exit 99 }; exit 1 }; "
        "if ((Get-Location).Path -ne $originalLocation) { exit 99 }"
    )
    result = subprocess.run([PWSH, "-NoProfile", "-Command", command], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=40)
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []
    return result, rows


def test_three_seed_jobs_keep_order_paths_and_runtime_overrides(tmp_path):
    result, rows = invoke(tmp_path, "-Experiment universal,single_flow -Device cpu -ParallelEnvs 4 -ValidationParallelEnvs 2")
    assert result.returncode == 0, result.stdout + result.stderr
    assert [row["args"][row["args"].index("--seed") + 1] for row in rows] == ["11", "23", "37"] * 2
    assert [row["args"][row["args"].index("--config") + 1] for row in rows] == ["configs/default.json"] * 3 + ["configs/e1/single_flow.json"] * 3
    for row in rows:
        assert Path(row["cwd"]) == project_path(".")
        assert row["args"][-6:] == ["--device", "cpu", "--parallel-envs", "4", "--validation-parallel-envs", "2"]
        assert "--episodes-per-update" not in row["args"]


def test_failed_seed_stops_later_seeds_and_experiments(tmp_path):
    result, rows = invoke(tmp_path, "-Experiment universal,single_flow", fail_seed=23)
    assert result.returncode == 1
    assert "exit 7" in result.stdout
    assert len(rows) == 2
    assert all("configs/default.json" in row["args"] for row in rows)


def test_preflight_and_dry_run_modes(tmp_path):
    result, rows = invoke(tmp_path, "-PreflightOnly")
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(rows) == 3 and all("--preflight-only" in row["args"] for row in rows)
    (tmp_path / "calls.jsonl").unlink()
    result, rows = invoke(tmp_path, "-DryRun")
    assert result.returncode == 0 and rows == []
    assert all(f"seed={seed}" in result.stdout for seed in (11, 23, 37))


@pytest.mark.parametrize("names", ["universal,unknown", "universal,universal"])
def test_invalid_selection_starts_nothing(tmp_path, names):
    result, rows = invoke(tmp_path, f"-Experiment {names}")
    assert result.returncode == 1 and rows == []


def test_catalog_covers_each_current_ppo_configuration_once():
    result = subprocess.run([PWSH, "-NoProfile", "-Command", f"& {quote(SCRIPT)} -List | ConvertTo-Json"],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    rows = json.loads(result.stdout)
    expected = {p.relative_to(project_path(".")).as_posix() for p in project_path("configs").rglob("*.json")
                if not {"archive", "manifests", "baselines"}.intersection(p.parts)}
    assert len(rows) == len(expected) == 26
    assert {row["Config"] for row in rows} == expected
    assert len({row["Experiment"] for row in rows}) == 26
    for row in rows:
        config = load_config(row["Config"])
        assert all(seed in config["algorithm_seeds"] for seed in (11, 23, 37))
        assert config["training"]["episodes_per_update"] == 20
