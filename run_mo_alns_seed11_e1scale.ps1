Set-Location -LiteralPath 'C:\Users\chenz\Desktop\fatigue_aware_reconfigurable_assembly_scheduling'

Write-Output "[$(Get-Date -Format o)] Starting test"
& 'C:\Users\chenz\Desktop\fatigue_aware_reconfigurable_assembly_scheduling\.venv\Scripts\python.exe' -u mo_alns.py --config configs\baselines\mo_alns.json --dataset test --algorithm-seed 11 --instance-limit 20 --parallel-envs 20 --run-name mo_alns_seed11_e1scale_test
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Output "[$(Get-Date -Format o)] Starting ood"
& 'C:\Users\chenz\Desktop\fatigue_aware_reconfigurable_assembly_scheduling\.venv\Scripts\python.exe' -u mo_alns.py --config configs\baselines\mo_alns.json --dataset ood --algorithm-seed 11 --instance-limit 20 --parallel-envs 20 --run-name mo_alns_seed11_e1scale_ood
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Output "[$(Get-Date -Format o)] Starting stress"
& 'C:\Users\chenz\Desktop\fatigue_aware_reconfigurable_assembly_scheduling\.venv\Scripts\python.exe' -u mo_alns.py --config configs\baselines\mo_alns.json --dataset stress --algorithm-seed 11 --instance-limit 20 --parallel-envs 20 --run-name mo_alns_seed11_e1scale_stress
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Output "[$(Get-Date -Format o)] All datasets completed"

