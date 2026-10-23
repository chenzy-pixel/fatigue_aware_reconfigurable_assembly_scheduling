param(
    [ValidateSet('preflight','train','evaluate','calibrate-budget')]
    [string]$Stage = 'preflight',
    [string]$Python = 'python',
    [string]$OutputDir = 'result/factorial27_episode_study',
    [ValidateSet('ppo','a2c','dcan','rule_ddqn','mo_alns','nsga2_ls','moead')]
    [string]$Method = 'ppo',
    [string]$Config = 'configs/default.json',
    [int]$Seed = 11,
    [string]$Device = 'cuda',
    [int]$TrainingWorkers = 20,
    [int]$Episodes,
    [int]$Steps,
    [double]$Seconds,
    [string]$BudgetManifest,
    [string]$Checkpoint,
    [switch]$RemoteRun
)
$ErrorActionPreference = 'Stop'
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    $suiteArguments = @('-u','-m','scripts.comparison_suite','--stage',$Stage,'--output-dir',$OutputDir,
        '--method',$Method,'--config',$Config,'--seed',"$Seed",'--device',$Device,'--training-workers',"$TrainingWorkers")
    if ($RemoteRun) { $suiteArguments += '--remote-run' }
    foreach ($entry in @(@('Episodes','--episodes'), @('Steps','--steps'), @('Seconds','--seconds'),
                         @('BudgetManifest','--budget-manifest'), @('Checkpoint','--checkpoint'))) {
        if ($PSBoundParameters.ContainsKey($entry[0])) {
            $suiteArguments += @($entry[1], [string](Get-Variable -Name $entry[0] -ValueOnly))
        }
    }
    & $Python @suiteArguments
    if ($LASTEXITCODE -ne 0) { throw "Comparison suite failed with exit code $LASTEXITCODE" }
} finally { Pop-Location }
