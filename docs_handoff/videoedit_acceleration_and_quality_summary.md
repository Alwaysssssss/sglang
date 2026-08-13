# VideoEdit 加速与质量总结

## 1. 测试口径

- 硬件：双 NVIDIA L20，`SP=2`、`Ulysses=2`。
- 场景：1080p、80 个输出帧、81 帧推理窗口、4 个去噪 step。
- 正式性能数据：每个方案运行 3 次，表中使用中位数。
- BF16 基线已经使用双卡和 layerwise offload；表格只统计后续量化与算子优化的增量收益。

## 2. 加速手段逐项叠加

| 累积阶段 | 本阶段增加的手段 | 主要加速对象（占原始去噪时间） | 去噪时间 | 相对上一步 | 累计加速 |
|---|---|---:|---:|---:|---:|
| BF16 基线 | FlashAttention + BF16 Linear | - | 298.457 s | - | 1.000x |
| 在线 FP8 | Linear 权重/激活 W8A8 | Linear，约 21%～22% | 283.143 s | 1.054x | 1.054x |
| Triton FP8 | 用 Triton Scaled GEMM 替换原 FP8 kernel | 同上 | 266.580 s | 1.062x | 1.120x |
| QKV/KV 融合 | 合并投影，减少重复激活量化和 kernel launch | Linear 子集，不增加覆盖率 | 267.359 s | 0.997x | 1.116x |
| SageAttention | Self-Attention 使用 QK INT8、PV FP8 | Self-Attention，约 68% | **178.895 s** | **1.495x** | **1.668x** |
| 离线 FP8 权重 | 推理前直接加载 FP8 权重和 scale | 不增加 forward 覆盖率 | 约 178.9 s | 约 1.000x | 约 1.668x |
| 静态激活 scale | 使用校准 scale，省去运行时 absmax | 仅激活量化开销 | 178.499 s | 1.002x | 1.672x |

结论：最终推荐的动态激活方案，DiT 去噪从 `298.457 s` 降到
`178.895 s`，加速 **1.668x**；请求总耗时从 `348.060 s` 降到
`228.223 s`，端到端加速 **1.525x**。

QKV/KV 融合在本场景中没有单独产生速度收益，但减少了重复量化并使执行路径更紧凑。离线权重主要改善模型分发、启动量化和加载峰值，不会改变 forward kernel。静态激活只额外提升约 `0.22%`，未达到 `5%` 的上线收益门槛，因此动态 per-token 激活仍是更稳妥的默认方案。

## 3. 原始推理时间覆盖情况

| 原始模块 | 占 BF16 去噪时间 | 当前处理 |
|---|---:|---|
| Self-Attention 核心 | 约 68% | SageAttention：QK INT8、PV FP8 |
| 所有目标 Linear | 约 21%～22% | FP8 E4M3FN W8A8 Triton GEMM |
| SP 双卡通信 | 约 4% | 未量化 |
| Norm、RoPE、激活函数、残差等 | 约 5%～6% | BF16/FP32 |
| Cross-Attention 核心 | 不到 1% | FlashAttention BF16 |

Linear 与 Self-Attention 合计覆盖约 **89%～90%** 的原始去噪 GPU 时间。
其中只有 Linear 属于严格的“权重量化”，约占 `21%～22%`；Self-Attention
没有权重，低精度化的是 Q/K/V 激活之间的 Attention 计算。

原始请求中，去噪阶段占总耗时约 **85.75%**。因此当前低精度路径覆盖整次请求
约 `76%～77%`，VAE、预处理、视频编码及未量化 kernel 会限制端到端加速。

## 4. 当前量化配置

- Linear 权重：FP8 E4M3FN，对称、per-output-channel，FP32 scale。
- Linear 激活：FP8 E4M3FN，对称、动态 per-token max-abs scale。
- Self-Attention：SageAttention `qk_int8_pv_fp8_cuda`。
- Cross-Attention：FlashAttention BF16，因为其原始耗时不到 1%，继续低精度化的收益很小。
- 328/328 个目标 Transformer Linear 已进入真实 W8A8 路径，没有 BF16 反量化回退。

## 5. 40 步视频质量

以下指标比较固定 seed 下的 FP8 + SageAttention 输出与 BF16 输出。二者都是生成结果，
因此这些指标衡量的是“相对 BF16 的保真度”，不是相对真实目标视频的绝对质量。
Mask 编辑区域平均占整帧 `7.44%`。

| 评估区域 | PSNR | SSIM 均值 | SSIM P05 | MAE | 判断 |
|---|---:|---:|---:|---:|---|
| 全画面 | 41.20 dB | 0.9717 | 0.9687 | 0.00352 | 通过 |
| Mask 编辑区 | 34.93 dB | 0.9148 | 0.9101 | 0.01210 | 通过初筛 |
| Mask 外背景 | 42.52 dB | 0.9764 | 0.9735 | 0.00283 | 稳定 |
| Mask 16 px 边界 | 36.62 dB | 0.8758 | 0.8697 | 0.00879 | 需要重点目检 |

时间方向的连续帧残差 MAE 为 `0.000629`，未显示明显的大幅时序偏移。


全画面指标会被贴回后的非编辑区域抬高，因此质量判断必须同时看 Mask 内和边界。
目前结果支持继续使用该量化方案，但只有一个 40 步案例，正式结论仍需在同一代码版本下
对多案例进行 BF16/FP8 成对评估，并补充 LPIPS、时序指标和人工盲测。

## 6. 数据来源

- 性能汇总：`docs_tyx/videoedit_l20_w8a8_experiment_summary.md`
- 质量原始结果：`videoedit_quality_evaluation/fp8_sage_vs_bf16_40steps.json`
- 质量评估脚本：`scripts/videoedit_evaluate_quality.py`
