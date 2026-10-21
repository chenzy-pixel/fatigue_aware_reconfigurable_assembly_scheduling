param(
    [ValidateSet('Preflight','Benchmark','Run','Report')]
    [string]$Stage = 'Preflight',
    [string]$Python = 'python',
    [string]$SourceDir = 'result/analysis/heuristics_seed11_20261019',
    [string]$OutputDir = 'result/analysis/heuristics_1800_v1',
    [ValidateSet('seven','tuning21')]
    [string]$Subset = 'seven',
    [string]$Workers = 'auto',
    [int[]]$WorkerGrid = @(4,8,12),
    [int[]]$Seeds = @(11,23,37),
    [double]$Seconds = 1800
)
$ErrorActionPreference = 'Stop'
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    function Invoke-SearchStage([string]$Step) {
        $launchArguments = @('-u','-m','scripts.heuristic_search','--stage',$Step,
            '--source-dir',$SourceDir,'--output-dir',$OutputDir,'--subset',$Subset,
            '--workers',$Workers,'--seconds',$Seconds.ToString([Globalization.CultureInfo]::InvariantCulture),
            '--seeds') + @($Seeds | ForEach-Object { "$_" }) + @('--worker-grid') + @($WorkerGrid | ForEach-Object { "$_" })
        if ($Step -eq 'run') { $launchArguments += '--remote-run' }
        & $Python @launchArguments
        if ($LASTEXITCODE -ne 0) { throw "Heuristic $Step failed with exit code $LASTEXITCODE" }
    }
    if ($Stage -eq 'Run') {
        if ($Workers -eq 'auto') { Invoke-SearchStage 'benchmark' }
        Invoke-SearchStage 'preflight'
        Invoke-SearchStage 'run'
    } else {
        Invoke-SearchStage $Stage.ToLowerInvariant()
    }
} finally { Pop-Location }
