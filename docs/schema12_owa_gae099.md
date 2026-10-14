# OWA/log2 的 GAE lambda=0.99 对照

基线为已完成的 `schema12_screen_seed11_20261012_201659_owa_log2`，其 gamma=1.0、gae_lambda=0.95。新配置 `configs/observation/screening/owa_log2_lambda099.json` 继承同一 OWA/log2 预设，显式设置 gamma=1.0、gae_lambda=0.99。有效配置仅实验名称与 gae_lambda 不同。

固定 shared_head、excess、OWA(q=0.5)、schema12、log2、有符号全局完工裕量、原冻结尺度与失败罚项；seed11，从头训练2000 episodes，每20 episodes进行一次PPO更新。保持相同实例生成与Universal偏好流，采用原validation/test和completion-first选模。

## 远端运行

在远端仓库目录、原训练Python环境中执行：

```powershell
python scripts/run_schema12_screening.py --experiment owa_log2_lambda099 --preflight-only
```

检查通过后启动唯一的一组实验：

```powershell
python scripts/run_schema12_screening.py --experiment owa_log2_lambda099
```

启动器保存最终配置、preflight和launch状态，随后执行训练及best模型的完整测试。检查run内config.json的`ppo.gamma=1.0`、`ppo.gae_lambda=0.99`以及网络/尺度身份。默认pair入口仍用于原两组筛选。

## 判读

gamma仍为1.0，改变lambda只改变优势估计中后续TD残差的传播权重。距当前k个决策的残差权重由0.95^k变为0.99^k；例如k=100时约为0.00592与0.36603。这是决策步数，不是仿真分钟。更长传播可能改善远期信用分配，也可能提高估计方差，需要以实测判断。

按相同实例、偏好和采样种子比较两组的best完整测试、同为ep2000的last验证及独立验证：整体和最差偏好完成率、纯Cost/Workload端点、超时失败、统一OWA坐标Q/HV、Flow、Cost、实际工人负担和偏好响应。同时检查后期波动、KL、clip fraction、gradient norm、value loss和explained variance。2000轮预算与更新批次不变，单seed结果只作为筛选证据。

本实验的归一化尺度保持Flow=368.3143、Cost=353.27、OWA=21.03145448655653；按保存配置读取两组模型，用共同坐标报告质量。lambda对照不预先承诺完成率或质量收益。
