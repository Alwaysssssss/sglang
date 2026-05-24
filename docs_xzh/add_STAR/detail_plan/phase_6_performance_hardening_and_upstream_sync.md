# Phase 6：性能优化接入、底层加速对齐与长期维护方案

## 1. 文档定位

本阶段不是“再做一点收尾优化”，而是把当前已经**功能对齐、结果过 baseline parity** 的 STAR 接入，从：

1. `能在 SGLang runtime 上正确运行`

推进到：

2. `真正吃到 SGLang 底层加速能力`

本阶段的目标不是单纯记录 benchmark，而是系统回答以下问题：

1. 当前 STAR 集成版已经接入了哪些 SGLang 能力
2. 当前 STAR 集成版还缺少哪些底层加速路径
3. 应该按什么顺序接入这些加速能力
4. 每一种加速改造的风险、代码落点和验收口径是什么
5. 如何在优化过程中不破坏 phase 5 已经完成的推理语义对齐

---

## 2. 当前状态结论

## 2.1 当前已经完成的部分

截至 phase 5 结束，STAR CogVideoX-SR 已经完成：

1. SGLang native pipeline 接入
2. 条件视频独立输入契约
3. STAR DiT / VAE / scheduler / decode 运行时适配
4. phase 5 baseline parity 通过
5. 端到端真实权重 smoke run 可复现

## 2.2 当前还没有真正接入的加速能力

需要明确：当前 STAR 集成版虽然已经运行在 SGLang 框架内，但**并没有完整吃到 SGLang 的底层高性能栈**。

当前缺口主要包括：

1. STAR transformer 只走 `torch SDPA`，没有接 SGLang attention backend 抽象
2. STAR transformer 仍大量使用 `nn.Linear` / `SATLayerNorm`，没有迁到 SGLang 原生并行/量化/融合层
3. STAR 当前没有接入 `USPAttention / TP / SP / CFG-parallel`
4. STAR 当前没有接入 `TeaCache`
5. STAR 当前没有接入 `cache-dit`
6. STAR 当前没有接入 SGLang 的量化路径，如 AWQ / FP8 / Nunchaku
7. 当前通过 parity 的配置依然使用了 `dit_cpu_offload=true` 和 `text_encoder_cpu_offload=true`
8. 当前未建立正式的分阶段 benchmark 和性能回归基线

## 2.3 本阶段总目标

phase 6 的总目标是：

1. 在不破坏 phase 5 baseline parity 的前提下，逐步接入 SGLang 支持的可用加速能力
2. 先拿低风险、可验证的时延收益
3. 再推进需要重构 transformer 结构的高收益优化
4. 建立性能回归测试、配置矩阵和 upstream 维护规范

---

## 3. 本阶段范围

### 本阶段处理

1. 建立 STAR 当前性能瓶颈画像
2. 接入低风险性能优化项
3. 规划并实施 STAR transformer 的结构性加速改造
4. 评估并逐步接入 TP / SP / CFG-parallel / attention backend / cache / quant
5. 建立正式 benchmark、性能回归和验收规范
6. 清理 phase 5 期间为对齐保留的非最佳性能路径
7. 沉淀后续 upstream 同步和升级边界

### 本阶段不处理

1. 回退 phase 1-5 的总体接口设计
2. 为了性能破坏 baseline parity
3. 在没有 benchmark 的情况下做拍脑袋优化
4. 把 STAR 直接退化成“只要快，不再严格对齐原语义”

---

## 4. 总体策略：两条线并行推进

本阶段拆成两条线：

1. **线路 A：低风险收益优先**
   目标是尽快拿到稳定、可复现、不会大改模型结构的性能收益。
2. **线路 B：结构性重构**
   目标是把 STAR transformer / runtime 热路径真正迁到 SGLang 的原生高性能抽象上。

建议执行原则：

1. 先启动线路 A，建立基线、拿到第一轮收益
2. 在线路 A 结果稳定后，启动线路 B 的核心改造
3. 线路 B 每完成一个里程碑，都必须回跑 phase 5 baseline parity
4. 线路 A 与线路 B 共用同一套 benchmark 样例、日志口径和验收标准

---

## 5. 加速能力现状矩阵

下表用于明确“哪些已经接入、哪些还没有、phase 6 应该怎么做”。

| 能力项 | 当前状态 | 备注 | phase 6 目标 |
|---|---|---|---|
| SGLang native pipeline | 已接入 | 已替代原版脚本主调度 | 保持 |
| torch SDPA | 已接入 | 仅是 PyTorch 级 attention kernel | 保持 |
| SGLang attention backend 抽象 | 未接入 | STAR 仅支持 `TORCH_SDPA` | 接入 |
| FlashAttention / AITER / SageAttention | 未接入 | 当前 STAR 不能选择这些 backend | 接入并 benchmark |
| TP / SP / USPAttention | 未接入 | 当前 transformer 不使用 SGLang attention/linear abstraction | 接入 |
| CFG-parallel | 未接入 | 当前默认 serial cond/uncond 双前向 | 评估并接入 |
| torch.compile | 未启用 | 框架支持，默认未开 | 评估并接入 |
| TeaCache | 未接入 | 当前 STAR 不在内置 CFG cache 支持前缀内 | 评估并接入 |
| cache-dit | 未接入 | 当前 STAR 未注册 BlockAdapter | 评估并接入 |
| AWQ / FP8 / Nunchaku | 未接入 | `quant_config` 当前未使用 | 评估并接入 |
| DiT / text encoder CPU offload | 已接入 | 当前主要作为可运行保障，不是性能最优配置 | 重新调优 |
| VAE tiling | 已接入 | 当前主要是显存安全策略 | 保持并 benchmark |

---

## 6. 统一 benchmark 与性能回归规范

所有性能工作必须建立在固定基准之上。

## 6.1 固定 benchmark case

性能基线与回归统一使用 phase 5 的 reference case：

1. prompt：`/sgl-workspace/STAR_mg/input/cogvideox_test/text/023_klingai_reedit.txt`
2. condition video：`/sgl-workspace/STAR_mg/input/cogvideox_test/lq/023_klingai_reedit.mp4`
3. reference：`/sgl-workspace/STAR_mg/cogvideox-based/sat/output/ref_seed1234/.../000000.mp4`
4. seed：`1234`
5. resolution：`480x720`
6. sampling num frames：`7`
7. output frames：`25`
8. num inference steps：`50`
9. guidance scale：`6.0`

## 6.2 统一统计口径

每次 benchmark 必须至少产出：

1. 总 wall-clock
2. text encode 时延
3. condition video loading 时延
4. condition video VAE encode 时延
5. denoise 总时延
6. denoise 平均 step 时延
7. decode 总时延
8. 峰值显存
9. parity 结果

## 6.3 benchmark 产物目录

建议每轮 benchmark 输出到：

1. `outputs/star_phase6_bench/<run_id>/summary.json`
2. `outputs/star_phase6_bench/<run_id>/candidate.mp4`
3. `outputs/star_phase6_bench/<run_id>/compare_baseline.json`
4. `outputs/star_phase6_bench/<run_id>/compare_strict.json`
5. `outputs/star_phase6_bench/<run_id>/profile.json`

## 6.4 必须新增的脚本

建议新增：

1. `python/sglang/multimodal_gen/test/manual/profile_star_cogvideox_sr.py`
2. `python/sglang/multimodal_gen/test/manual/benchmark_star_cogvideox_sr_matrix.py`

### `profile_star_cogvideox_sr.py` 负责：

1. 跑单个固定 case
2. 输出分 stage 和分模块时延
3. 记录峰值显存
4. 可选择是否回跑 parity

### `benchmark_star_cogvideox_sr_matrix.py` 负责：

1. 扫配置矩阵
2. 对比不同 backend / offload / compile / parallel 配置
3. 统一写出对比报告

## 6.5 质量硬门槛

phase 6 的所有性能优化都必须先满足质量门槛，否则该优化结果无效。

### 最低质量门槛

1. 必须继续满足 phase 5 的 `baseline parity`
2. 不允许为了加速而放宽 phase 5 已定义的 baseline 验收标准

### 推荐的更强质量门槛

为了避免“虽然还过 baseline，但质量已经明显回退”的情况，建议在 phase 6 的内部 benchmark 中额外记录并优先满足：

1. `ssim_mean >= 0.93`
2. `ssim_min >= 0.92`
3. `mse_mean <= 35`
4. `mae_mean <= 3.5`
5. `failed_frames = 0 / 25`

说明：

1. 这组阈值不是替代 phase 5 baseline，而是 phase 6 的推荐内部质量红线
2. 如果某项优化只能在放松这些指标后才获得收益，则默认不进入推荐配置

## 6.6 性能主指标定义

phase 6 不允许只看单一总时间，也不允许混用冷启动和热启动口径。

建议固定以下指标：

1. `T_warm_e2e`
   - 主指标
   - 定义为模型与权重已经加载完成后，单次完整请求从进入 pipeline 到 `candidate.mp4` 落盘的时延
2. `T_denoise`
   - 核心子指标
   - 定义为 denoising 主循环总时延
3. `T_step_avg`
   - 定义为 denoising 平均每 step 时延
4. `T_cold_e2e`
   - 补充指标
   - 定义为包含加载和初始化开销的端到端 wall-clock

### 统一 speedup 定义

建议写死：

`speedup = 原生 STAR 同口径时间 / SGLang STAR 同口径时间`

其中：

1. phase 6 主 speedup 指标默认使用 `warm_e2e_speedup`
2. phase 6 次级指标使用 `denoise_speedup`
3. `cold_e2e_speedup` 只做补充，不作为主 gate

### 统计规则

1. 所有 speedup 必须在同一台机器、同一 GPU 型号、同一输入 case、同一 prompt、同一 seed、同一步数下测量
2. 如果启用了 `torch.compile`，主比较必须使用第 2 次和第 3 次运行的平均值，不使用第 1 次编译热身结果
3. 原生 STAR 与 SGLang 都必须记录完整命令行和运行配置

## 6.7 性能目标区间

### phase 6 最低通过线

在满足本文件 6.5 节质量门槛的前提下：

1. `warm_e2e_speedup >= 1.8x`

这是 phase 6 的最低通过线，不是理想目标。

### phase 6 推荐目标区间

建议把正式目标区间定义为：

1. `1.8x <= warm_e2e_speedup <= 3.0x`

解释：

1. `1.8x` 以下，说明本阶段虽然做了优化，但底层 acceleration 接入的工程收益仍然偏弱
2. `1.8x ~ 3.0x` 是当前阶段最合理的目标区间
3. `> 3.0x` 不是上限；如果能做到并保持质量，应视为额外收益

### 核心子指标建议

除了 `warm_e2e_speedup`，建议同时追踪：

1. `denoise_speedup >= 1.8x`

如果总时延达标但 `denoise_speedup` 很低，通常说明收益主要来自外围配置优化，而不是 SGLang 底层加速真正接上。

## 6.8 “提前达到 1.8x” 的处理规则

必须明确：`提前达到 1.8x speedup 不能自动视为 phase 6 完成。`

如果出现以下情况：

1. 当前版本已经达到 `warm_e2e_speedup >= 1.8x`
2. 但仍有 SGLang 底层加速能力没有接入

则 phase 6 仍然必须继续推进。

原因：

1. phase 6 的目标不是“碰巧快了”，而是“系统接入 SGLang 的底层 acceleration 栈”
2. 只靠少量配置收益提前达到 `1.8x`，不代表集成已经完成
3. 后续可维护性、可扩展性和多卡能力也依赖这些底层接入工作

## 6.9 “全部接入”的判定原则

本阶段默认目标是把 **SGLang 支持且对 STAR 结构适用的底层加速能力全部接入**。

因此，以下能力不能在没有结论的情况下被跳过：

1. attention backend abstraction
2. 非 `TORCH_SDPA` attention backend
3. linear / norm 的 SGLang 原生层迁移
4. TP
5. SP / USP
6. CFG-parallel
7. `torch.compile`
8. TeaCache
9. cache-dit
10. quantization

如果某项能力最终没有接入，只允许有两种合法结论：

1. **已接入**
2. **经验证不适用，并给出明确技术论证和实验记录**

不允许的结论是：

1. “先不做”
2. “暂时没空”
3. “已经够快了所以不需要”

---

## 7. 线路 A：低风险收益优先

这条线不先大改 STAR transformer 结构，而是优先吃掉那些：

1. 代码改动小
2. 风险可控
3. 很容易 benchmark
4. 很容易回滚

的优化项。

## 7.1 A1：建立 phase 6 起始性能基线

### 目标

把“当前 baseline parity 通过版本”的性能测准，作为后续所有优化的比较起点。

### 涉及文件

1. `python/sglang/multimodal_gen/test/manual/run_star_cogvideox_sr_smoke.py`
2. `python/sglang/multimodal_gen/test/manual/profile_star_cogvideox_sr.py`
3. `python/sglang/multimodal_gen/test/manual/compare_star_sglang_outputs.py`

### 实施内容

1. 给现有 smoke 脚本补齐 wall-clock 统计
2. 拆出 encode / denoise / decode 的分段计时
3. 记录 offload 配置、attention backend、compile 状态
4. 把原版 STAR 的 wall-clock 作为对照一并落盘

### 验收方式

1. 生成一份固定格式 `summary.json`
2. 结果可和原版 STAR 在同一 case 下对比
3. baseline parity 仍然通过

## 7.2 A2：offload 配置矩阵调优

### 目标

先不改模型结构，找到当前硬件上最合理的 resident/offload 组合。

### 优先测试矩阵

1. `dit_cpu_offload=true, text_encoder_cpu_offload=true, vae_cpu_offload=false`
2. `dit_cpu_offload=false, text_encoder_cpu_offload=true, vae_cpu_offload=false`
3. `dit_cpu_offload=false, text_encoder_cpu_offload=false, vae_cpu_offload=false`
4. `dit_cpu_offload=false, text_encoder_cpu_offload=false, vae_cpu_offload=true`

### 涉及文件

1. `python/sglang/multimodal_gen/runtime/server_args.py`
2. `python/sglang/multimodal_gen/test/manual/benchmark_star_cogvideox_sr_matrix.py`
3. 如有必要，STAR pipeline config 文档说明处同步更新

### 关键判断

1. 在 L20 48GB 上是否可以稳定关闭 `dit_cpu_offload`
2. text encoder 是否可以常驻，不再反复 offload
3. 关闭 offload 后，phase 5 baseline parity 是否仍稳定

### 验收方式

1. 产出一份矩阵表，记录各配置：
   - wall-clock
   - peak memory
   - parity 结果
2. 明确 phase 6 默认推荐配置

## 7.3 A3：启用 `torch.compile`

### 目标

验证在不改模型结构的情况下，`torch.compile` 能否给 STAR transformer 带来稳定收益。

### 涉及文件

1. `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py`
2. `python/sglang/multimodal_gen/runtime/models/dits/star_cogvideox_sr.py`
3. `python/sglang/multimodal_gen/test/manual/benchmark_star_cogvideox_sr_matrix.py`

### 实施内容

1. 在固定 case 下测试 `enable_torch_compile=false/true`
2. 记录首次编译开销与第二次运行开销
3. 判断是否需要为 STAR 明确设置 compile 黑名单或 guard

### 风险点

1. 编译首次开销可能很高
2. 动态 shape 或 list/tensor 混用可能破坏 compile 收益
3. 数值或 parity 可能出现波动

### 验收方式

1. 明确 compile on/off 的净收益
2. 如果开启 compile 后第二次运行显著更快且 parity 不变，则纳入默认优化集

## 7.4 A4：CFG 双前向优化

### 目标

评估 STAR 当前 serial cond/uncond 两次前向是否应该改为单次 batch doubling 前向。

### 背景

当前 `DenoisingStage` 中 STAR 默认走 serial CFG，两次调用 `_predict_noise()`。
这对语义对齐是稳妥的，但对吞吐不一定最优。

### 涉及文件

1. `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py`
2. `python/sglang/multimodal_gen/configs/pipeline_configs/star_cogvideox_sr.py`
3. `python/sglang/multimodal_gen/test/unit/test_star_dynamic_cfg.py`
4. `python/sglang/multimodal_gen/test/manual/profile_star_cogvideox_sr.py`

### 实施内容

1. 设计 STAR 专用的 batched CFG 开关
2. 把 cond/uncond latent、text cond、condition latent 合批一次前向
3. 输出后再分拆 cond/uncond 噪声
4. 比较和 serial CFG 的数值一致性

### 前提

1. 必须先在少量 step / 单步 tensor 上做数值对比
2. 只有在误差仍满足 phase 5 baseline parity 时，才能进入默认路径候选

### 验收方式

1. 单步 tensor 对比可解释
2. baseline parity 通过
3. denoise 总时延明显下降

## 7.5 A5：VAE encode/decode 热点收尾

### 目标

在不改变 STAR 语义的前提下，减少 encode/decode 阶段的额外开销。

### 涉及文件

1. `python/sglang/multimodal_gen/runtime/models/vaes/star_cogvideox_vae.py`
2. `python/sglang/multimodal_gen/runtime/pipelines_core/stages/video_condition_vae_encoding.py`
3. `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/star_cogvideox_sr_decoding.py`

### 实施内容

1. 评估 VAE encode 是否存在不必要的 dtype 往返
2. 评估 decode 窗口拼接前后的 device / contiguous / cast 开销
3. 检查 VAE tiling 对当前 case 是否真正有收益
4. 明确“显存优先模式”和“时延优先模式”的默认配置

### 验收方式

1. encode/decode 分段时延下降
2. baseline parity 不变

---

## 8. 线路 B：结构性重构

这条线的目标不是微调现有代码，而是让 STAR 的核心热路径真正进入 SGLang 的原生高性能抽象。

## 8.1 B1：把 STAR attention 接入 SGLang attention abstraction

### 目标

把当前 `_StarAttention -> F.scaled_dot_product_attention` 的实现，迁移到 SGLang `USPAttention` / attention backend 体系。

### 当前问题

当前 STAR attention：

1. 只支持 `TORCH_SDPA`
2. 不能切换到 `FA / AITER / SAGE`
3. 不能自然接入 SP / USP / Ring
4. 不能吃到 SGLang 的 attention metadata builder 体系

### 涉及文件

1. `python/sglang/multimodal_gen/runtime/models/dits/star_cogvideox_sr.py`
2. `python/sglang/multimodal_gen/configs/models/dits/star_cogvideox_sr.py`
3. 如有需要：
   - `python/sglang/multimodal_gen/runtime/layers/attention/...`
   - `python/sglang/multimodal_gen/runtime/layers/rotary_embedding/...`

### 实施要点

1. 重构 `_StarAttention`
2. 显式拆出 `q/k/v` projection、RoPE、qk norm、attention impl
3. 改为使用 `USPAttention`
4. 支持 `supported_attention_backends` 配置
5. 先完成 `TORCH_SDPA` 等价实现
6. 再逐步打开 `FA / AITER / SAGE`

### 推荐里程碑

1. 里程碑 1：STAR + USPAttention + TORCH_SDPA 等价版
2. 里程碑 2：STAR + FlashAttention 后端
3. 里程碑 3：STAR + SP/USP smoke

### 验收方式

1. 单层 attention 输出与旧实现数值接近
2. phase 5 baseline parity 通过
3. 至少一种非 SDPA backend 可用

## 8.2 B2：把 STAR Linear/Norm 迁到 SGLang 原生层

### 目标

把 STAR 当前的大量 `nn.Linear / SATLayerNorm` 改为 SGLang runtime 层。

### 当前问题

当前 STAR transformer 中：

1. projection 基本都是 `nn.Linear`
2. 没有接并行 linear
3. 没有接 quant linear
4. 仍保留 `SATLayerNorm`

### 涉及文件

1. `python/sglang/multimodal_gen/runtime/models/dits/star_cogvideox_sr.py`
2. `python/sglang/multimodal_gen/runtime/layers/linear.py`
3. `python/sglang/multimodal_gen/runtime/layers/layernorm.py`
4. 必要时新增 STAR 专用 fused helper

### 实施内容

1. 把 qkv / out proj / mlp proj 迁到：
   - `ReplicatedLinear`
   - `ColumnParallelLinear`
   - 必要时 `MergedColumnParallelLinear`
2. 评估 AdaLN modulation 层是否也应迁到 SGLang linear
3. 逐步替换 `SATLayerNorm`
4. 评估是否可接 fused scale-shift/gate kernel

### 验收方式

1. 单层输出与旧实现一致
2. 支持 TP shard 的最小 smoke
3. 为量化和 parallel 打开前置条件

## 8.3 B3：接入 TP / SP / USP / CFG-parallel

### 目标

在 transformer 已经迁到 SGLang attention/linear abstraction 后，正式接入并行能力。

### 子任务 B3.1：TP

#### 目标

让 STAR 的 qkv / mlp / out proj 支持 tensor parallel。

#### 涉及文件

1. `python/sglang/multimodal_gen/runtime/models/dits/star_cogvideox_sr.py`
2. `python/sglang/multimodal_gen/configs/models/dits/star_cogvideox_sr.py`
3. `python/sglang/multimodal_gen/test/unit/test_star_tp_smoke.py`

#### 验收

1. `tp_size=2` 单 case smoke 通过
2. baseline parity 通过

### 子任务 B3.2：SP / USP

#### 目标

让 STAR 支持 sequence parallel / Ulysses path。

#### 重点风险

1. STAR 是 joint text+video token attention
2. text tokens 与 image tokens 的 replicated / sharded 关系要重新理清
3. RoPE 和 local enhancer 的 token layout 必须与 SP 一致

#### 涉及文件

1. `python/sglang/multimodal_gen/runtime/models/dits/star_cogvideox_sr.py`
2. `python/sglang/multimodal_gen/runtime/layers/attention/layer.py`
3. `python/sglang/multimodal_gen/test/unit/test_star_sp_smoke.py`

#### 验收

1. `sp_degree=2` smoke 通过
2. baseline parity 通过

### 子任务 B3.3：CFG-parallel

#### 目标

让 cond/uncond 分支在多卡上真正并行。

#### 涉及文件

1. `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py`
2. `python/sglang/multimodal_gen/runtime/models/dits/star_cogvideox_sr.py`
3. `python/sglang/multimodal_gen/test/unit/test_star_cfg_parallel_smoke.py`

#### 验收

1. 多卡 CFG parallel smoke 通过
2. baseline parity 不退化

## 8.4 B4：接入 TeaCache

### 目标

让 STAR 支持基于 timestep 相似性的 step skipping。

### 当前问题

虽然 STAR 继承了 `CachableDiT`，但没有现成可用的 STAR TeaCache 实现。

### 涉及文件

1. `python/sglang/multimodal_gen/runtime/models/dits/star_cogvideox_sr.py`
2. `python/sglang/multimodal_gen/runtime/cache/teacache.py`
3. `python/sglang/multimodal_gen/configs/sample/star_cogvideox_sr.py`

### 实施内容

1. 为 STAR 定义 modulated input 提取逻辑
2. 实现缓存命中时的 residual 复用
3. 明确 STAR 是否支持 CFG cache separation
4. 增加 STAR 专用 TeaCache 推荐参数

### 验收方式

1. `enable_teacache=true` 时无错误
2. baseline parity 在容忍范围内
3. denoise step 数学意义上的有效 skip 可记录
4. wall-clock 有实际收益

## 8.5 B5：接入 cache-dit

### 目标

让 STAR 进入 SGLang 的 cache-dit block caching 体系。

### 当前问题

当前 STAR 不在 cache-dit 预注册模型列表内，因此需要自定义 BlockAdapter。

### 涉及文件

1. `python/sglang/multimodal_gen/runtime/models/dits/star_cogvideox_sr.py`
2. `python/sglang/multimodal_gen/runtime/cache/cache_dit_integration.py`
3. 如有需要，新增 STAR 专用 BlockAdapter 注册文件

### 实施内容

1. 分析 STAR block 结构是否符合 cache-dit block 识别要求
2. 定义 STAR 的 block adapter
3. 先做 transformer-only cache-dit smoke
4. 再做正式 benchmark

### 验收方式

1. `cache_dit_config` 打开后可运行
2. 无 block mismatch 错误
3. baseline parity 在容忍范围内
4. wall-clock 有收益

## 8.6 B6：接入量化能力

### 目标

让 STAR 支持 SGLang 的可用量化路径。

### 推荐顺序

1. 先做 `AWQ / FP8` 的 feasibility
2. 再评估 `Nunchaku / SVDQ`

### 涉及文件

1. `python/sglang/multimodal_gen/runtime/models/dits/star_cogvideox_sr.py`
2. `python/sglang/multimodal_gen/configs/models/dits/star_cogvideox_sr.py`
3. 量化 loader / quant config 相关代码

### 实施内容

1. 让 STAR 不再忽略 `quant_config`
2. 为 STAR 补 `get_nunchaku_quant_rules()`
3. 把关键 linear 层迁到 quant-compatible linear
4. 对不同量化路径做单独 benchmark

### 验收方式

1. 量化模型可以加载
2. baseline parity 不显著退化
3. 显存或时延至少有一项收益

---

## 9. 推荐执行顺序

为了避免同时引入太多变量，推荐按以下顺序推进：

1. `A1` 建基线
2. `A2` offload 矩阵
3. `A3` torch.compile
4. `A4` batched CFG
5. `A5` VAE 热点收尾
6. `B1` attention abstraction
7. `B2` linear / norm 迁移
8. `B3` TP / SP / CFG-parallel
9. `B4` TeaCache
10. `B5` cache-dit
11. `B6` quantization

原因：

1. A 线可以先拿到确定收益
2. B1/B2 是 B 线前置条件
3. TP/SP/CFG-parallel 必须依赖 attention/linear 抽象重构
4. cache 和 quant 必须建立在新的模型结构抽象之上，不能太早做

---

## 10. 每项任务的通用验收模板

每个子任务完成后，必须至少执行以下验收：

1. `unit test`
2. `single-case smoke`
3. `phase 5 baseline parity`
4. `benchmark against previous best`

建议统一输出：

1. 任务前配置
2. 任务后配置
3. 时延变化
4. 峰值显存变化
5. parity 是否通过
6. 是否进入默认推荐配置

---

## 11. 需要新增的测试和脚本清单

建议 phase 6 至少新增以下文件：

1. `python/sglang/multimodal_gen/test/manual/profile_star_cogvideox_sr.py`
2. `python/sglang/multimodal_gen/test/manual/benchmark_star_cogvideox_sr_matrix.py`
3. `python/sglang/multimodal_gen/test/unit/test_star_tp_smoke.py`
4. `python/sglang/multimodal_gen/test/unit/test_star_sp_smoke.py`
5. `python/sglang/multimodal_gen/test/unit/test_star_cfg_parallel_smoke.py`
6. `python/sglang/multimodal_gen/test/unit/test_star_attention_backend_selection.py`
7. `python/sglang/multimodal_gen/test/unit/test_star_teacache_smoke.py`
8. `python/sglang/multimodal_gen/test/unit/test_star_cache_dit_smoke.py`
9. `python/sglang/multimodal_gen/test/unit/test_star_quantized_load_smoke.py`

---

## 12. 运行时代码清理清单

phase 6 优化完成后，需要清理并沉淀以下内容：

1. phase 5 为数值对齐保留的临时实验分支
2. 所有非默认 benchmark 参数硬编码
3. 仅用于人工实验的路径常量
4. 不再使用的 profile 日志片段
5. 旧版 slow path 的注释和 dead code

可以保留但必须整理为正式机制的内容：

1. smoke / compare / benchmark 脚本
2. phase 5 参考 case manifest
3. 性能对比结果样例

---

## 13. Upstream 同步与长期维护策略

## 13.1 STAR upstream 更新时

分两类处理：

1. **权重级更新**
   - 优先只改转换脚本和 integration metadata
2. **结构级更新**
   - 重新评估 transformer / vae / scheduler / decode 语义
   - 重新执行 phase 5 baseline parity
   - 重新执行 phase 6 benchmark

## 13.2 SGLang upstream 更新时

优先检查：

1. `DenoisingStage` hook 签名是否变化
2. attention backend 抽象是否变化
3. linear / quant / compile loader 是否变化
4. cache-dit / teacache 接口是否变化
5. distributed parallel state 相关接口是否变化

## 13.3 必须维护的版本信息

建议写入：

1. `star_integration_config.json`
2. `manifests/source_assets.json`
3. phase 6 benchmark 结果目录中的 `summary.json`

至少包含：

1. STAR upstream 版本或 commit
2. SGLang 版本或 commit
3. 当前默认性能配置
4. 当前支持的 attention backend
5. 当前支持的 parallel 模式
6. 当前支持的 quant 模式

---

## 14. 阶段验收标准

phase 6 完成时，至少应满足：

1. 已建立正式 benchmark 体系
2. 已明确当前 STAR 接入版的默认推荐性能配置
3. 在满足 6.5 节质量门槛的前提下，`warm_e2e_speedup >= 1.8x`
4. 已建立 `warm_e2e_speedup` 和 `denoise_speedup` 的统一统计口径
5. baseline parity 在默认优化配置下仍通过
6. 所有 SGLang 底层加速能力都已得到“已接入”或“经验证不适用”的最终结论
7. 已明确 TP / SP / CFG-parallel 的当前支持范围
8. 已明确 TeaCache / cache-dit / quant 的支持状态
9. 文档能说明未来升级时的边界与回归检查流程

## 14.1 推荐的更强完成标准

如果资源允许，建议把“真正完成”定义为：

1. `dit_cpu_offload=false` 可稳定运行
2. `1.8x <= warm_e2e_speedup <= 3.0x`
3. `denoise_speedup >= 1.8x`
4. STAR 已支持至少一个非 `TORCH_SDPA` attention backend
5. STAR 已支持至少一种并行能力：
   - TP
   - SP/USP
   - CFG-parallel
6. STAR 已支持至少一种缓存或量化能力：
   - TeaCache
   - cache-dit
   - AWQ / FP8 / Nunchaku
7. 对于未接入的底层加速项，文档中已有明确的“不适用证明”和实验记录

---

## 15. 止损条件

出现以下情况时，应暂停继续堆优化项，先回到设计层复盘：

1. baseline parity 在多个优化项下持续不稳定
2. 结构性重构后 attention/linear 输出难以建立可靠数值对照
3. 多卡并行 smoke 长时间无法稳定
4. 引入高复杂度优化但 wall-clock 无实际收益
5. phase 6 文档定义的 benchmark 口径已经无法解释结果变化

止损原则：

1. 先保留“正确且略快”的版本
2. 不为了接入某个特性而把维护成本抬到不可接受
3. 默认配置必须是团队可复现、可维护、可继续升级的版本

---

## 16. 最终交付判断

phase 6 结束后，STAR 接入才可以被视为：

1. `功能上完成`
2. `结果上对齐`
3. `性能上达到至少 1.8x 的同口径加速收益`
4. `底层 acceleration 能力已系统接入，而不是只靠少量配置收益提速`
5. `维护上具备长期演进边界`

如果 phase 6 结束时仍只有“在 SGLang 里能跑、结果也对”，但底层 acceleration 几乎都没接上，那么只能视为：

1. `phase 5 完成`
2. `phase 6 未完成`

不能视为“STAR 已经完整接入了 SGLang 的加速能力”。
