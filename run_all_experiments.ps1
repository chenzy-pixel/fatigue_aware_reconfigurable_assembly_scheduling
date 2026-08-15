param(
    [ValidateSet("Smoke", "Screen", "Formal", "Audit", "All")]
    [string]$Stage = "All",
    [ValidateRange(1, 256)]
    [int]$ParallelEnvs = 10,
    [ValidateRange(0, 10)]
    [int]$MaxRetries = 1,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project Python environment not found: $Python"
}

$Arguments = @(
    (Join-Path $ProjectRoot "v7_experiments.py"),
    "--stage", $Stage,
    "--parallel-envs", $ParallelEnvs,
    "--max-retries", $MaxRetries
)
if ($DryRun) {
    $Arguments += "--dry-run"
}

& $Python @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "v7 experiment suite failed with exit code $LASTEXITCODE"
}
