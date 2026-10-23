# Three structural ablations and HGNN-A2C: 4 configurations x 5 seeds.
param(
    [ValidateSet('all','preflight','train','evaluate')]
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
Import-Module (Join-Path $PSScriptRoot 'ExperimentLaunch.psm1') -Force
Invoke-ExperimentGroup -Group ablations_a2c @PSBoundParameters
