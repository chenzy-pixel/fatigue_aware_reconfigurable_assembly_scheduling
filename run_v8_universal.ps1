param(
    [Parameter(Mandatory = $true)]
    [string]$Config
)

$ErrorActionPreference = "Stop"
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$configPath = Resolve-Path -LiteralPath $Config
$seeds = @(11, 23, 37, 53, 71)

foreach ($seed in $seeds) {
    & $python "$PSScriptRoot\train.py" `
        --config $configPath.Path `
        --algorithm-seed $seed `
        --run-name "v8_universal_seed${seed}"
    if ($LASTEXITCODE -ne 0) {
        throw "V8 universal training failed: seed=$seed"
    }
}
