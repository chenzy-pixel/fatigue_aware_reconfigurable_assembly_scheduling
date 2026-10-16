<#
.SYNOPSIS
Run selected PPO experiments sequentially with seeds 11, 23, and 37.
.EXAMPLE
.\scripts\run_three_seeds.ps1 -Experiment universal
.EXAMPLE
.\scripts\run_three_seeds.ps1 -Experiment single_flow,single_cost -PreflightOnly
.EXAMPLE
.\scripts\run_three_seeds.ps1 -List
#>
[CmdletBinding()]
param(
    [string[]]$Experiment = @('universal'),
    [string]$Python,
    [ValidateSet('cpu', 'cuda')]
    [string]$Device,
    [ValidateRange(1, 1024)]
    [int]$ParallelEnvs,
    [ValidateRange(1, 1024)]
    [int]$ValidationParallelEnvs,
    [switch]$PreflightOnly,
    [switch]$DryRun,
    [switch]$List
)

$ErrorActionPreference = 'Stop'
$experimentRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$experimentSeeds = @(11, 23, 37)
$experimentConfigs = [ordered]@{
    universal              = 'configs/default.json'
    single_flow            = 'configs/e1/single_flow.json'
    single_cost            = 'configs/e1/single_cost.json'
    single_workload        = 'configs/e1/single_workload.json'
    no_graph               = 'configs/ablations/no_graph.json'
    neutral_flow           = 'configs/ablations/neutral_flow.json'
    neutral_cost           = 'configs/ablations/neutral_cost.json'
    neutral_workload       = 'configs/ablations/neutral_workload.json'
    raw                    = 'configs/controls/raw.json'
    variance               = 'configs/controls/variance.json'
    bounded                = 'configs/controls/bounded.json'
    lambda097              = 'configs/controls/lambda097.json'
    lambda099              = 'configs/controls/lambda099.json'
    owa_q025               = 'configs/controls/owa_q025.json'
    owa_q075               = 'configs/controls/owa_q075.json'
    raw_bounded            = 'configs/controls/raw_bounded.json'
    variance_bounded       = 'configs/controls/variance_bounded.json'
    single_variance        = 'configs/controls/single_variance.json'
    single_owa_q025        = 'configs/controls/single_owa_q025.json'
    single_owa_q075        = 'configs/controls/single_owa_q075.json'
    raw_owa_q025           = 'configs/controls/raw_owa_q025.json'
    raw_owa_q075           = 'configs/controls/raw_owa_q075.json'
    raw_single_variance    = 'configs/controls/raw_single_variance.json'
    raw_single_owa_q025    = 'configs/controls/raw_single_owa_q025.json'
    raw_single_owa_q05     = 'configs/controls/raw_single_owa_q05.json'
    raw_single_owa_q075    = 'configs/controls/raw_single_owa_q075.json'
}

if ($List) {
    $experimentConfigs.GetEnumerator() | ForEach-Object {
        [pscustomobject]@{ Experiment = $_.Key; Seeds = '11,23,37'; Config = $_.Value }
    }
    return
}

if (-not $Experiment -or $Experiment.Count -eq 0) {
    throw 'Specify at least one experiment. Use -List for available names.'
}
$selectedExperiments = @()
foreach ($name in $Experiment) {
    if ([string]::IsNullOrWhiteSpace($name) -or -not $experimentConfigs.Contains($name)) {
        throw "Unknown experiment '$name'. Use -List for available names."
    }
    if ($selectedExperiments -contains $name) {
        throw "Duplicate experiment '$name'."
    }
    $selectedExperiments += $name
    if (-not (Test-Path -LiteralPath (Join-Path $experimentRoot $experimentConfigs[$name]) -PathType Leaf)) {
        throw "Missing configuration for '$name': $($experimentConfigs[$name])"
    }
}

if (-not $Python) {
    $Python = Join-Path $experimentRoot '.venv/Scripts/python.exe'
}
$pythonCommand = Get-Command $Python -CommandType Application -ErrorAction Stop
$pythonExecutable = $pythonCommand.Source

Push-Location -LiteralPath $experimentRoot
try {
    foreach ($name in $selectedExperiments) {
        foreach ($experimentSeed in $experimentSeeds) {
            $launchArguments = @(
                '-B', '-u', '-m', 'scripts.run_experiments',
                '--config', $experimentConfigs[$name], '--seed', "$experimentSeed"
            )
            if ($Device) { $launchArguments += @('--device', $Device) }
            if ($ParallelEnvs) { $launchArguments += @('--parallel-envs', "$ParallelEnvs") }
            if ($ValidationParallelEnvs) {
                $launchArguments += @('--validation-parallel-envs', "$ValidationParallelEnvs")
            }
            if ($PreflightOnly) { $launchArguments += '--preflight-only' }

            Write-Host "[$name] seed=$experimentSeed; config=$($experimentConfigs[$name])"
            if ($DryRun) {
                $displayArguments = @($pythonExecutable) + $launchArguments
                Write-Output ('& ' + (($displayArguments | ForEach-Object { "'" + $_.Replace("'", "''") + "'" }) -join ' '))
                continue
            }

            & $pythonExecutable @launchArguments
            $childExitCode = $LASTEXITCODE
            if ($childExitCode -ne 0) {
                throw "Experiment '$name' failed at seed $experimentSeed (exit $childExitCode). Remaining jobs were not started."
            }
        }
    }
}
finally {
    Pop-Location
}
