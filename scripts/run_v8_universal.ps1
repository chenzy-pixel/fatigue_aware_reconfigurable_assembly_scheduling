[CmdletBinding()]
param(
    [ValidateNotNullOrEmpty()]
    [string]$Config = "configs/v8/universal.json",
    [ValidateNotNullOrEmpty()]
    [int[]]$Seeds = @(11),
    [string]$Python = "",
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_.-]*$')]
    [string]$RunPrefix = "v8_universal",
    [switch]$Smoke,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 2.0

if (@($Seeds | Where-Object { $_ -lt 0 }).Count -gt 0) {
    throw "Seeds must be non-negative integers."
}
if (@($Seeds | Select-Object -Unique).Count -ne $Seeds.Count) {
    throw "Seeds must be unique within a batch."
}

$projectRoot = Split-Path -Parent $PSScriptRoot
$trainPath = Join-Path $projectRoot "train.py"
$configCandidatePath = if ([System.IO.Path]::IsPathRooted($Config)) {
    $Config
} else {
    Join-Path $projectRoot $Config
}
$configPath = (Resolve-Path -LiteralPath $configCandidatePath).Path
if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
    throw "Config must point to a file: $configPath"
}

if ([string]::IsNullOrWhiteSpace($Python)) {
    $venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
    $Python = if (Test-Path -LiteralPath $venvPython -PathType Leaf) {
        $venvPython
    } else {
        "python"
    }
}
$pythonPath = (Get-Command -Name $Python -CommandType Application -ErrorAction Stop).Source
$batchStamp = Get-Date -Format "yyyyMMdd_HHmmss_fff"

Write-Host ("Universal batch: seeds={0}; config={1}" -f ($Seeds -join ','), $configPath)
foreach ($seed in $Seeds) {
    $runName = "${RunPrefix}_seed${seed}_${batchStamp}"
    if ($Smoke) {
        $runName += "_smoke"
    }
    $arguments = @(
        $trainPath,
        "--config", $configPath,
        "--algorithm-seed", [string]$seed,
        "--run-name", $runName
    )
    if ($Smoke) {
        $arguments += "--smoke"
    }

    $commandPreview = @($pythonPath) + $arguments | ForEach-Object {
        "'" + $_.Replace("'", "''") + "'"
    }
    Write-Host ($commandPreview -join ' ')
    if ($DryRun) {
        continue
    }

    & $pythonPath @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Universal training failed: seed=$seed, exit=$LASTEXITCODE, run=$runName"
    }
}
