param(
    [string]$Config = "configs/v8/universal.json"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$configCandidatePath = if ([System.IO.Path]::IsPathRooted($Config)) {
    $Config
} else {
    Join-Path $projectRoot $Config
}
$configPath = Resolve-Path -LiteralPath $configCandidatePath
$seeds = @(11, 23, 37, 53, 71)

foreach ($seed in $seeds) {
    & $python "$projectRoot\train.py" `
        --config $configPath.Path `
        --algorithm-seed $seed `
        --run-name "v8_universal_seed${seed}"
    if ($LASTEXITCODE -ne 0) {
        throw "V8 universal training failed: seed=$seed"
    }
}
