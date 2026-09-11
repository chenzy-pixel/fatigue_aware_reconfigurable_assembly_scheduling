$ErrorActionPreference = "Stop"
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$seeds = @(11, 23, 37, 53, 71)
$objectives = @("flow", "cost", "variance")

foreach ($objective in $objectives) {
    foreach ($seed in $seeds) {
        & $python "$PSScriptRoot\train.py" `
            --config "$PSScriptRoot\configs\v8\specialist_$objective.json" `
            --algorithm-seed $seed `
            --run-name "v8_specialist_${objective}_seed${seed}"
        if ($LASTEXITCODE -ne 0) {
            throw "V8 specialist failed: objective=$objective seed=$seed"
        }
    }
}
