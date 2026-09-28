# RTX 5060 Ti 并行训练测速

先结束远端正在运行的训练，让测速独占 GPU。随后在远端 PowerShell 中进入已同步到最新代码的项目目录，并激活原来的 `drl` 环境：

```powershell
cd C:\Users\Administrator\Projects\fatigue_aware_reconfigurable_assembly_scheduling
conda activate drl
python -u benchmark_parallel.py --config configs/e1/single_flow_relative_time.json
```

脚本先检查 GPU 前向及反向计算，随后短测 8、12、16、20、24、32、40 个工作进程。它复测训练候选两次，并用完整的 50 个验证实例 × 3 次采样复测验证候选。最后输出 `result/benchmarks/rtx5060ti_*/report.md`、`benchmark.json`、`recommended_config.json` 和一条 120 episode 启动命令。脚本会运行较长时间；进度写在终端，已完成的测速结果会逐项保存。

测速先准备完整 120 个训练实例，避免首个候选独自承担实例生成；这段时间单独记录。每档并行数还会单独预热生产和工人分配动作。各候选随后使用相同的前 40 个 episode 索引、初始权重及验证采样种子。训练和验证分别选择并行数：在内存约束内，取距离最快耗时不超过 5% 的最小档位。训练速度以包含 PPO 更新的墙钟耗时衡量；GPU 利用率用于解释结果。`worker_generation_seconds_sum` 与 `worker_environment_step_seconds_sum` 是各进程耗时之和，不应当与墙钟耗时相加。

完整实验按推荐命令运行，关键参数为：

```powershell
python -u train.py --config configs/e1/single_flow_relative_time.json --episodes 120 --algorithm-seed 11 --episodes-per-update 40 --parallel-envs <推荐训练数> --validation-parallel-envs <推荐验证数> --run-name e1_flow_relative_time_seed11_ep120_optimized
```

完成后检查运行目录的 `summary.json`、`update_log.csv`、`validation_log.csv`。验证应在 episode 40、80、120 后各出现一次；每次正式验证仍为 50 个实例、3 次采样。`benchmark.json` 中的 `reference_v8` 使用原来的逐图装配、逐环境动作头和逐条验证采样路径；双方使用相同训练循环、实例和初始权重。120 episode 数值按阶段实测外推，完整运行的 `elapsed_seconds` 才是实测总耗时。训练和验证期间，每 30 秒打印一次进度。

只想先检查 CUDA 和短测流程时，可在测速命令后加 `--skip-full --quick-validation-instances 10`；这不会产生正式推荐并行数。
