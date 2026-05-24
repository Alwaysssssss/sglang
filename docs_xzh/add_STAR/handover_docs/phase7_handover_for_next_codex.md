# STAR 接入 SGLang 交接文档

本文档用于给下一位 Codex 快速接棒当前 `STAR -> SGLang` 集成工作，重点覆盖：

- 之前各阶段已经完成了什么
- 当前 phase7 的真实状态
- 已经接入了哪些 SGLang 底层加速
- 哪些底层加速还没有真正吃到
- 下一位 Codex 在 phase7 剩余阶段需要完成的目标

相关核心参考文件：

- 总体对比记录：[compare.json](/sgl-workspace/sglang/docs_xzh/add_STAR/compare.json:1)
- phase7 规划文档：[phase_7_full_sglang_acceleration_completion.md](/sgl-workspace/sglang/docs_xzh/add_STAR/detail_plan/phase_7_full_sglang_acceleration_completion.md:1)
- STAR pipeline：[star_cogvideox_sr_pipeline.py](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/pipelines/star_cogvideox_sr_pipeline.py:1)
- STAR transformer：[star_cogvideox_sr.py](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/models/dits/star_cogvideox_sr.py:1)

## 1. 当前总状态

截至本交接时点：

- `phase1` 到 `phase5` 已完成
- `phase5 baseline parity` 已通过
- `phase6` 已做过大量加速接入与实验，但没有完成“所有底层加速全部接入”的最终目标
- `phase7` 已经开始，做过 fused norm / compile 稳定性 / FlashInfer RoPE / local enhancer 热路径 / 显存治理等实验
- 当前 `phase7` 仍未最终验收通过，原因是：
  - 单卡 exact 路线还没有达到目标速度
  - 仍有部分 SGLang 底层加速没有完全落地或没有形成质量可接受的可发布配置

当前最重要的共识：

- 双卡质量通过配置已经超过 `1.8x`
- 但 phase7 不能因此结束
- 后续必须继续补完单卡主线，以及剩余底层加速接入

## 2. 已完成的阶段概览

### 2.1 phase1

已完成 STAR 权重离线转换工具，支持把 STAR 资产导出为 SGLang 可加载目录，主要包括：

- transformer
- VAE
- text encoder / tokenizer
- scheduler 元信息
- manifest / conversion report / key mapping report

后续运行时不再依赖 `STAR_mg` 原仓库路径。

### 2.2 phase2

已完成 STAR 专用请求契约与条件视频输入：

- 新增 `condition_video_path`
- 支持条件视频帧数、采样 FPS、stride 等参数
- 新增条件视频加载 stage
- 新增 `[B, T, C, H, W]` 条件视频张量契约

### 2.3 phase3

已完成 STAR 组合式 pipeline 接线，主链路为：

`InputValidation -> STARConditionVideoLoading -> TextEncoding -> STARConditionVideoVAEEncoding -> STARLatentPreparation -> TimestepPreparation -> Denoising -> STARCogVideoXSRDecoding`

这一阶段重点是让 STAR 真正进入 SGLang pipeline，而不是外挂脚本调用。

### 2.4 phase4

已完成 STAR 三大核心组件接入：

- STAR transformer runtime adapter
- STAR VAE adapter
- STAR scheduler adapter

这一步把模型从“能被 registry 发现”推进到了“能跑起来”。

### 2.5 phase5

已完成语义对齐与 baseline 验收：

- 修正了 STAR 初始 latent 噪声语义
- 修正了条件视频帧数契约
- 对齐了 decode 窗口语义
- 完成 reference 逐帧对齐验收

phase5 通过时的关键产物目录：

- [/sgl-workspace/sglang/outputs/star_phase5_eval_seed1234_cpu_init_rng](/sgl-workspace/sglang/outputs/star_phase5_eval_seed1234_cpu_init_rng)

关键 baseline 指标：

- `ssim_mean = 0.936193`
- `ssim_min = 0.930056`
- `mse_mean = 30.326740`
- `mae_mean = 2.943940`
- `failed = 0/25`

### 2.6 phase6

已完成一批底层加速接入与大量性能实验，主要方向包括：

- SGLang attention 抽象接入
- FlashAttention 路径验证
- compile 路径接入
- cache-dit / TeaCache 接入
- CFG-parallel / batched CFG 实验
- 并行 linear 抽象迁移
- worker 侧 compile 稳定性治理

但 phase6 没有“完全收口”，原因是：

- 单卡 exact 速度仍不够高
- 量化路径未拿到质量通过配置
- 仍有底层 fused 化和显存常驻 fast path 没彻底打通

### 2.7 phase7

phase7 已经做过这些工作：

- fused layernorm / modulation 路径
- compile-safe qk layernorm 处理
- local enhancer 5D 热路径实验
- FlashInfer RoPE 实验接入
- 条件视频 VAE encode 前的显存治理
- pre-encode transformer 临时下放治理

但 phase7 目前仍处于“部分完成”状态。

## 3. 当前已接入的 SGLang 底层加速

以下能力已经有代码落点，且至少做过真实实验：

### 3.1 attention 抽象

STAR 已接入 SGLang attention 抽象：

- 单卡路径走 `LocalAttention`
- 并行路径支持 `USPAttention`

主要位置：

- [star_cogvideox_sr.py](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/models/dits/star_cogvideox_sr.py:1)

### 3.2 attention backend 元数据

已声明并实验过的 backend 包括：

- `TORCH_SDPA`
- `FA`
- `SAGE`
- `SAGE_ATTN_3`

其中：

- `FA` 已经做过真实速度验证
- `FlashInfer RoPE` 已集成，但实验表明不适合作为默认主路径

### 3.3 并行 linear 抽象

STAR 不再完全依赖裸 `nn.Linear`，已经迁移到 SGLang 的并行 linear 体系，包括：

- `ReplicatedLinear`
- `ColumnParallelLinear`
- `RowParallelLinear`
- `MergedColumnParallelLinear`

### 3.4 compile

`torch.compile` 已接入并验证有效，是当前单卡加速的关键收益来源之一。

同时已补：

- compile 场景下 worker 侧清理逻辑
- qk layernorm 非连续张量兼容

### 3.5 cache 相关

已接入：

- `TeaCache`
- `cache-dit`

但当前收益有限，仍需进一步调参或结构适配。

### 3.6 CFG 并行相关

已做过：

- `cfg-parallel`
- `batched CFG`

其中：

- 双卡 `cfg-parallel` 已取得质量通过且超过 `1.8x`
- `batched CFG` 在 STAR 上收益很小

### 3.7 fused norm / modulation 的初步接入

当前已做过一轮 fused 化尝试：

- input/post/final/qk layernorm 统一封装
- `norm + modulation`
- `residual + norm + modulation`

但这还不是 phase7 的最终状态，后续仍需要继续往更深的 fused kernel 路径推进。

## 4. 已接入但实验结论是“默认不建议开启”的部分

这些能力已经做过实现或验证，但目前不建议作为默认主路径：

### 4.1 FlashInfer RoPE

状态：

- 已接入
- 已完成实验
- 当前结论：默认关闭

原因：

- 在 STAR 当前主路径上，速度没有正收益，反而出现回退

### 4.2 local enhancer 5D 高性能路径

状态：

- 已做实验
- 质量可以对齐
- 速度没有超过当前 best exact 单卡主线

当前结论：

- 保留代码和实验记录
- 暂不作为默认配置

### 4.3 batched CFG

状态：

- 已接入并测试

当前结论：

- 对 STAR 的收益不明显，不是当前 phase7 的主攻方向

## 5. 当前仍未完全吃到的 SGLang 底层加速

这是下一位 Codex 最需要关注的部分。

### 5.1 parity-safe quantization 仍未打通

已经有规则和试验入口，但没有形成“质量通过 + 速度显著收益”的最终配置。

仍需重点推进：

- `FP8`
- `AWQ`
- `Nunchaku`
- `SVDQuant`

当前已知问题：

- `FP8` 虽然更快，但 baseline parity 不通过

这是 phase7 剩余阶段最有希望继续拉升单卡速度的主线之一。

### 5.2 单卡 resident fast path 仍未打通

当前最大现实瓶颈之一：

- 去掉某些保护性 offload 后，会在 `STARConditionVideoVAEEncodingStage` OOM
- compile 预热后，measured request 需要额外的显存治理

目前为了稳定：

- 条件视频 VAE encode 前会释放 text encoder residency
- 必要时会临时下放 transformer

但代价是：

- 单卡 exact 速度被拉低到约 `1.27x`

因此，下一位 Codex 必须解决：

- 如何在不依赖整模临时 CPU 下放的情况下稳定跑通 measured request
- 如何重排显存占用，让 transformer 尽量常驻 GPU

### 5.3 更深层 fused kernel 化还没有彻底完成

虽然已经做过 fused layernorm / modulation 的一轮接入，但还没有把 STAR 中这些热点完全迁到更底层的 SGLang 高性能路径：

- `SATLayerNorm`
- `RoPE`
- `AdaLN`
- `local enhancer`

尤其是：

- `SATLayerNorm` 仍然是一个非常值得继续优化的点
- `RoPE / AdaLN / local enhancer` 仍保留较多自定义 PyTorch 路径

这部分是 phase7 后续单卡性能提升的重点。

### 5.4 attention backend 适配矩阵还没有彻底收口

当前已经试过一部分 backend，但还没有形成完整的“适用 / 不适用 / 默认启用”结论矩阵。

后续仍需补齐：

- 更严格的 `FA / SAGE` 对比
- 评估 `AITER` 是否适用
- 与 compile / quant / cache 组合时的稳定性验证

### 5.5 cache 加速路径没有形成强收益

`TeaCache` 和 `cache-dit` 已接入，但当前表现还不够理想。

仍需继续：

- 调整 cache 触发策略
- 校准与 STAR block 结构的适配
- 评估在 exact 质量门槛下的可行空间

### 5.6 TP / SP / USP / CFG-parallel 还没有最终收口

目前双卡 `cfg-parallel` 已经质量通过并超过 `1.8x`，但 phase7 还需要更完整的结论：

- 单卡 exact 主线先达到 `1.6x` 或 `1.8x` 以上
- 再正式整理双卡推荐配置
- 对 `TP / SP / USP / CFG-parallel` 给出最终适用性判断

## 6. 当前 benchmark 结论

请始终以 [compare.json](/sgl-workspace/sglang/docs_xzh/add_STAR/compare.json:1) 为准。

本交接时点需要特别记住这几个结果：

### 6.1 phase5 质量基准

- baseline parity 已通过
- 后续所有 phase7 实验都不能破坏这条底线

### 6.2 双卡最好结果

- `dual_cfg_parallel`
- `warm_e2e_speedup ≈ 1.8628x`
- baseline 通过

这是当前唯一明确超过 `1.8x` 且质量通过的主结果，但它不是 phase7 的结束条件。

### 6.3 当前 HEAD 上稳定可复现的单卡 exact 结果

- `single_fa_compile_stable_v5`
- `warm_e2e_speedup ≈ 1.2726x`
- baseline 通过

这是当前 HEAD 上“稳定可复现”的单卡 exact 配置。

### 6.4 之前出现过的更高单卡 exact 结果

- `single_fa_compile_fusedln_v2`
- `warm_e2e_speedup ≈ 1.4314x`
- baseline 通过

需要注意：

- 这是之前实验阶段拿到过的更好结果
- 但当前 HEAD 在显存稳定性治理后的稳定主线不是这个数
- 下一位 Codex 可以把它作为重要回看对象，分析为什么历史 best 没能在当前主线上保持

### 6.5 本轮 phase7 新增实验的重要结论

从最近几轮实验看：

- `single_fa_compile_fusedln_local5d_v1`：`1.3358x`，通过 baseline，但比历史 best 慢
- `single_fa_compile_ropefused_v4`：`1.2606x`，通过 baseline，但证明 FlashInfer RoPE 不适合当前默认主线
- `single_fa_compile_mainline_v6`：warmup 成功，但 measured request 在 `STARConditionVideoVAEEncodingStage` OOM

结论非常明确：

- 当前最大阻碍不是“模型不能跑”
- 而是“单卡高性能主线在 compile 后 measured request 的显存稳定性和量化质量”

## 7. 下一位 Codex 在 phase7 必须完成的目标

以下目标必须作为下一阶段的正式任务来完成。

### 7.1 第一目标：把单卡 exact 路线继续推高

优先级最高。

目标：

- 先把单卡 exact 路线提升到 `>= 1.6x`
- 最理想是进一步达到 `>= 1.8x`

注意：

- 双卡已经过 `1.8x` 不代表 phase7 可以结束
- 单卡主线必须先尽可能做强

### 7.2 第二目标：完成 parity-safe quantization

这是当前最可能继续拉升单卡性能的大头。

必须做的事情：

- 系统性实验 `FP8`
- 系统性实验 `AWQ`
- 系统性实验 `Nunchaku`
- 系统性实验 `SVDQuant`

要求：

- 不能只看速度
- 必须同时满足 phase5 baseline parity

### 7.3 第三目标：解决 compile 后 measured request 的 OOM

要求：

- 尽量不依赖“pre-encode transformer 临时下放”
- 让 measured request 可以稳定通过
- 同时尽量让 transformer 常驻 GPU

这是 phase7 剩余部分最关键的工程问题之一。

### 7.4 第四目标：完成更深层 fused 化

需要继续推进：

- `SATLayerNorm` 热路径 fused 化
- `RoPE` 热路径 fused 化或明确不适用结论
- `AdaLN` 热路径 fused 化
- `local enhancer` 更深层高性能化

这部分不是“可选优化”，而是 phase7 设定中非常重要的一条主线。

### 7.5 第五目标：补全底层 backend 适用性矩阵

下一位 Codex 需要对这些结论做收口：

- `FA`
- `SAGE`
- `AITER`
- compile 组合
- quant 组合
- cache 组合

输出必须是明确结论，而不是停留在“理论可做”。

### 7.6 第六目标：收口 cache 路径

需要继续验证：

- `TeaCache`
- `cache-dit`

要求：

- 不是只证明“能开”
- 而是要证明“在 STAR 上是否值得默认开启”

### 7.7 第七目标：最终收口双卡推荐方案

单卡主线推进后，再对双卡方案做最终整理：

- `cfg-parallel`
- `TP / SP / USP`

需要给出最终推荐：

- 哪个配置最适合质量通过
- 哪个配置最适合高吞吐
- 哪个配置最适合作为文档默认 benchmark

## 8. 强制执行的实验记录规则

下一位 Codex 必须继续遵守以下规则：

### 8.1 每次实验都写回 compare.json

必须记录：

- 配置名
- 使用的 backend / compile / quant / cache / offload 组合
- 单卡还是双卡
- 用时
- speedup
- 是否通过 baseline parity
- 如果失败，失败点是什么

### 8.2 不能只记录成功实验

失败实验也必须留下，尤其是：

- OOM
- compile unstable
- parity broken
- load incompatibility

因为这些失败记录对下一轮决策非常重要。

### 8.3 不能为了速度降低 phase5 质量门槛

所有加速都必须建立在：

- `phase5 baseline parity` 持续通过

的前提下。

## 9. 建议的接棒顺序

如果下一位 Codex 要继续推进 phase7，建议按下面顺序执行：

1. 回读 [compare.json](/sgl-workspace/sglang/docs_xzh/add_STAR/compare.json:1)，先确认当前 HEAD 的稳定主线和历史 best
2. 回看 [star_cogvideox_sr.py](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/models/dits/star_cogvideox_sr.py:1) 里 fused norm、RoPE、local enhancer、quant 相关分支
3. 先解决 compile 后 measured request 的显存稳定性，目标是不靠整模临时下放也能稳定跑
4. 立刻进入 parity-safe quantization 主线
5. 在单卡 exact 至少达到 `1.6x` 后，再回到双卡方案做最终收口

## 10. 一句话交接结论

当前项目已经完成了 STAR 在 SGLang 上的正确性集成，并做了大量底层加速接入；但 phase7 还没有最终结束。下一位 Codex 的核心任务，是继续完成剩余底层加速接入，特别是 `parity-safe quantization`、`compile 后单卡显存稳定性`、以及 `SATLayerNorm / RoPE / AdaLN / local enhancer` 的更深层 fused 化，把单卡 exact 路线继续推高，并在全过程中持续维护 [compare.json](/sgl-workspace/sglang/docs_xzh/add_STAR/compare.json:1)。
