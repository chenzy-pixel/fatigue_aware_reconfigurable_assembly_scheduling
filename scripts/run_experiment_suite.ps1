param(
    [ValidateSet('preflight','train','evaluate','rules','develop-search','freeze-budget','search','report','smoke')]
    [string]$Stage = 'preflight',
    [string]$Python = 'python',
    [string]$OutputDir = 'result/experiments',
    [string]$Manifest = 'configs/experiment_suite_v1.json',
    [string]$Device = 'cuda',
    [string[]]$Arms,
    [int[]]$Seeds,
    [ValidateSet('frontier','independent_request','both')]
    [string]$Task = 'both',
    [int]$TrainingWorkers = 20,
    [switch]$RemoteRun
)
$ErrorActionPreference = 'Stop'
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    $suiteArgs = @('-u','-m','scripts.experiment_suite','--stage',$Stage,'--output-dir',$OutputDir,
        '--manifest',$Manifest,'--device',$Device,'--task',$Task,'--training-workers',"$TrainingWorkers")
    if ($RemoteRun) { $suiteArgs += '--remote-run' }
    if ($Arms) { $suiteArgs += @('--arms') + $Arms }
    if ($Seeds) { $suiteArgs += @('--seeds') + @($Seeds | ForEach-Object { "$_" }) }
    & $Python @suiteArgs
    if ($LASTEXITCODE -ne 0) { throw "Experiment suite failed with exit code $LASTEXITCODE" }
} finally { Pop-Location }
