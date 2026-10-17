param(
    [ValidateSet('preflight','smoke','tune-search','tune-learning','train','curves','compare','report','formal','formal-evaluate','front-search','all')]
    [string]$Stage = 'preflight',
    [string]$Python = 'python',
    [string]$OutputDir = 'result/analysis/comparison_suite_remote_v1',
    [int]$Workers = 8,
    [int]$TrainingWorkers = 20,
    [double]$FrontSeconds = 0,
    [switch]$RemoteRun
)
$ErrorActionPreference = 'Stop'
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    $suiteArguments = @('-u','-m','scripts.comparison_suite','--stage',$Stage,'--output-dir',$OutputDir,'--workers',"$Workers",'--training-workers',"$TrainingWorkers")
    if ($RemoteRun) { $suiteArguments += '--remote-run' }
    if ($FrontSeconds -gt 0) { $suiteArguments += @('--front-seconds', "$FrontSeconds") }
    & $Python @suiteArguments
    if ($LASTEXITCODE -ne 0) { throw "Comparison suite failed with exit code $LASTEXITCODE" }
} finally { Pop-Location }
