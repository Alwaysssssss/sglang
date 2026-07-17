# Vivid-VR VAE SP4 与 CFG2×SP2 对比设计

## 目标

在已经引入 CFG 并行的四卡场景中，公平比较纯 `SP4` 与
`CFG2×SP2` 两种拓扑的端到端推理性能，回答哪种并行方式更快。

## 固定实验条件

两组实验固定使用同一模型、输入视频、caption sidecar、reference、
`seed=42`、`130 frames`、`20 inference steps`、FA-SP、
`torch.compile`、modulation/residual fusion、VAE spatial tile parallel、
FlowCut 服务生命周期和一次 `1 step` warmup。

唯一主动变量是四卡并行拓扑：

- 新 treatment `R101_VAE_SP4`：纯 `SP4`，CFG 并行关闭，VAE SP group
  world size 为 4。
- 已验收 comparator `R100_VAE_SP`：`CFG2×SP2`，CFG 并行开启，VAE SP
  subgroup world size 为 2。

`R101_VAE_SP4` 只注册到隔离的 `VAE_SP_TREATMENTS`，不进入默认
`run-all` 矩阵，也不改变 Phase E 的正式默认配置。

## 对照与判定

runner 需要一个历史 control 才能生成 VAE SP 派生记录，因此
`R101_VAE_SP4` 使用同为纯 SP4 的 `R4` 作为辅助 control。由于 `R4`
未开启 modulation fusion，这个派生对比只用于检查结果完整性，不作为
本问题的主结论。

主结论直接比较本次 `R101_VAE_SP4` formal record 与已验收的
`R100_VAE_SP` formal record：

- 第一指标：`total_runtime_seconds`。
- 第二指标：`model_inference_runtime_seconds`。
- 原因分析：denoise、Decode/Trim、VAE tile decode/gather/merge 分项。
- 正确性：runner 的 effective config 必须证明 SP world size、CFG 开关、
  VAE SP world size 与方案一致，并保留原有质量指标。

总耗时更低者判定为本次四卡配置下更快的拓扑，同时报告速度比与各阶段
的收益/损失，避免只看 VAE decode 局部指标。
