# Five rules, two evolution algorithms, and both search tasks.
param(
    [ValidateSet('all','preflight','rules','develop-search','freeze-budget','search','report')]
    [string]$Stage = 'all',
    [string]$Python = '',
    [string]$OutputDir = 'result/experiments',
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
Import-Module (Join-Path $PSScriptRoot 'ExperimentLaunch.psm1') -Force
Invoke-ExperimentGroup -Group rules_search @PSBoundParameters
