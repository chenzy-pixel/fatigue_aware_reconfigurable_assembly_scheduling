function Invoke-ExperimentGroup {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [ValidateSet('main','ablations_a2c','rules_search')]
        [string]$Group,
        [string]$Stage = 'all',
        [string]$Python = '',
        [string]$OutputDir = 'result/experiment_suite_v1',
        [string]$Manifest = 'configs/experiment_suite_v1.json',
        [string]$Device = 'cuda',
        [int[]]$Seeds,
        [ValidateSet('frontier','independent_request','both')]
        [string]$Task = 'both',
        [ValidateRange(1,2147483647)]
        [int]$TrainingWorkers = 20,
        [switch]$RemoteRun,
        [switch]$DryRun
    )
    $ErrorActionPreference = 'Stop'
    $projectRoot = Split-Path -Parent $PSScriptRoot
    if ([string]::IsNullOrWhiteSpace($Python)) {
        $Python = 'python'
        foreach ($candidate in @('.venv/Scripts/python.exe', '.venv/bin/python')) {
            $interpreter = Join-Path $projectRoot $candidate
            if (Test-Path -LiteralPath $interpreter -PathType Leaf) {
                $Python = $interpreter
                break
            }
        }
    }
    $arms = @()
    switch ($Group) {
        'main' {
            $arms = @('full','rule_ddqn','dcan')
            $pipeline = @('preflight','train','evaluate')
        }
        'ablations_a2c' {
            $arms = @('no_graph','no_direct','no_residual','a2c')
            $pipeline = @('preflight','train','evaluate')
        }
        'rules_search' {
            $pipeline = @('preflight','rules','develop-search','freeze-budget','search')
        }
    }
    $allowed = @($pipeline)
    if ($Group -eq 'rules_search') { $allowed += 'report' }
    if ($Stage -ne 'all' -and $Stage -notin $allowed) {
        throw "Stage '$Stage' is unavailable for '$Group'. Allowed: all, $($allowed -join ', ')."
    }
    $stages = if ($Stage -eq 'all') { $pipeline } else { @($Stage) }
    $heavy = @($stages | Where-Object { $_ -in @('train','evaluate','rules','develop-search','search') })
    if ($heavy.Count -gt 0 -and -not $RemoteRun -and -not $DryRun) {
        throw 'Execution requires -RemoteRun. Use -DryRun to inspect the sequence, or -Stage preflight for checks.'
    }
    $runner = Join-Path $PSScriptRoot 'run_experiment_suite.ps1'
    foreach ($currentStage in $stages) {
        $options = @{
            Stage = $currentStage
            Python = $Python
            OutputDir = $OutputDir
            Manifest = $Manifest
            Device = $Device
            Task = $Task
            TrainingWorkers = $TrainingWorkers
            RemoteRun = [bool]$RemoteRun
        }
        if ($arms.Count -gt 0) { $options.Arms = $arms }
        if ($Seeds) { $options.Seeds = $Seeds }
        if ($DryRun) {
            [pscustomobject]@{ Group=$Group; Stage=$currentStage; Parameters=$options }
        } else {
            Write-Host "[$Group] $currentStage"
            & $runner @options
        }
    }
}

Export-ModuleMember -Function Invoke-ExperimentGroup
