# Phase 7：STAR 剩余 SGLang 底层加速能力补全计划

## 1. 文档定位

phase 7 不是 phase 6 的简单续集，而是一次**补全剩余底层 acceleration 栈**的收口阶段。

phase 6 已经证明了三件事：

1. STAR 已经能在 SGLang native runtime 中稳定运行
2. phase 5 `baseline parity` 可以持续通过
3. 部分 SGLang 加速已经接入，但单卡 exact 路线还没有吃满底层收益

phase 7 的目标很明确：

1. 把 STAR 当前还没有真正吃到的 SGLang 底层加速能力补齐
2. 把“代码里有挂钩”推进到“真实跑通并形成可复现收益”
3. 优先把单卡 exact 路线推到 `>= 1.6x`，目标冲到 `>= 1.8x`
4. 单卡稳定后，再做双卡与并行路径的最终收口

---

## 2. phase 7 的前置事实

以下结论以 [compare.json](/sgl-workspace/sglang/docs_xzh/add_STAR/compare.json:1) 为准：

1. 当前最好单卡 exact 结果约为 `1.4186x`
2. 当前最好双卡质量通过结果约为 `1.8628x`
3. `FlashAttention + torch.compile` 已经有效
4. `TeaCache + cache-dit` 已接入，但当前参数下增益很小
5. `FP8` 路线更快，但目前严重破坏 baseline parity
6. 单卡关闭 offload 或让 transformer 常驻 GPU 时，会在条件视频 VAE encode 前 OOM

这意味着 phase 7 的重点不是再去证明“某个开关有一点收益”，而是集中攻克下列仍未真正吃满的底层加速：

1. fused norm / fused modulation / fused rope
2. local enhancer 的 kernel 化或 fused 化
3. parity-safe quantization
4. 单卡常驻与显存路径重排
5. TP / SP / USP / backend 扩展的最终实证接入

---

## 3. 当前仍未真正吃到的 SGLang 加速

## 3.1 已经有挂接，但还不能算完成

1. `FlashAttention`
   已有真实收益，但 STAR 仍保留较多自定义前后处理，attention 前后的 kernel 边界还比较碎。
2. `TeaCache / cache-dit`
   已能启用，但当前参数组合没有带来显著单卡收益。
3. `Quantization rules`
   规则声明已经存在，但还没有得到质量通过的量化运行配置。
4. `USPAttention / parallel linear`
   代码路径存在，但仍缺少完整的 phase 7 级 benchmark、稳定性和收益结论。

## 3.2 仍未真正吃到的核心底层加速

1. `SATLayerNorm` 仍大量存在
   STAR 目前没有充分迁到 SGLang 原生 norm/fused norm 路径。
2. `RoPE` 仍是自定义 PyTorch 路径
   目前 rotary cache 构造和旋转应用还是手写张量逻辑，尚未使用更底层实现。
3. `AdaLN / modulation` 仍是逐段 PyTorch 运算
   `shift/scale/gate` 的切分与调制仍是多次独立 kernel。
4. `local enhancer` 仍是纯 PyTorch 组合
   空间局部增强和时间局部增强仍是自定义卷积/线性拼接，没有融入更高效执行路径。
5. `single-card resident fast path` 没打通
   当前单卡 exact 仍依赖 offload，导致底层 kernel 优势没有完全释放。
6. `AITER` 等更多 backend 未落地
   当前 STAR 支持集合里没有把全部可用 backend 都变成真实可测能力。
7. `Parity-safe quantization` 未完成
   这是单卡继续提速的最大潜在空间之一。

---

## 4. phase 7 总目标

phase 7 的总目标拆成四层：

1. **结构层**
   把 STAR transformer 热路径中仍然停留在自定义 PyTorch 实现的部分，尽可能迁到 SGLang 原生高性能抽象。
2. **算子层**
   让 norm、rope、modulation、attention、local enhancer 这些热点路径获得 fused kernel、backend 或更少 kernel 边界。
3. **显存层**
   打通单卡更高常驻度的运行模式，让 compile、FA、quant 的收益不再被 offload 抵消。
4. **验收层**
   每完成一种底层加速，都必须重新过 phase 5 baseline parity，并把结果记录到 `compare.json`。

---

## 5. phase 7 范围

### 本阶段处理

1. STAR transformer 热路径的 fused 化改造
2. STAR 单卡显存/常驻策略改造
3. STAR quantization 真正落地
4. STAR attention backend 扩展与实证 benchmark
5. STAR cache 路径重新调优
6. STAR TP / SP / USP / CFG-parallel 的最终闭环
7. benchmark、profile、`compare.json` 持续记录与最终验收

### 本阶段不处理

1. 回退 phase 1-6 的接口设计
2. 改 prompt/输入样例来“做出更快结果”
3. 放宽 parity 标准换性能
4. 为了速度把 STAR 的原始推理语义改坏

---

## 6. phase 7 核心原则

1. **先单卡，后双卡**
   单卡 exact 路线优先达到 `>= 1.6x`，目标 `>= 1.8x`，再进入双卡对比和并行策略收口。
2. **每个优化都要留痕**
   所有测试配置、耗时、质量指标都必须写回 [compare.json](/sgl-workspace/sglang/docs_xzh/add_STAR/compare.json:1)。
3. **先解决热路径，后扫外围**
   phase 7 的最高优先级不是改脚本，而是改 transformer 内核热点。
4. **“已挂接”不等于“已完成”**
   只有满足“代码接入 + benchmark 通过 + 质量通过 + 有记录”的能力，才算完成。

---

## 7. 工作流总览

phase 7 建议拆成 7 条主线：

1. `P7-A`：Norm / RoPE / AdaLN fused 化
2. `P7-B`：Local enhancer 高性能化
3. `P7-C`：Single-card resident fast path
4. `P7-D`：Quantization 真正落地
5. `P7-E`：Attention backend 扩展与对比
6. `P7-F`：Cache 路径重调优
7. `P7-G`：Parallel 路径最终收口

建议执行顺序：

1. `P7-A`
2. `P7-B`
3. `P7-C`
4. `P7-D`
5. `P7-E`
6. `P7-F`
7. `P7-G`

原因：

1. `P7-A/P7-B` 直接打在单卡最热内核上
2. `P7-C` 决定这些优化能否真正释放收益
3. `P7-D` 是单卡进一步拉开差距的最大机会
4. `P7-E/P7-F/P7-G` 适合作为后续放大器与收尾项

---

## 8. P7-A：Norm / RoPE / AdaLN fused 化

## 8.1 目标

把 STAR transformer 中最碎、最频繁的前后处理 kernel 收敛到 SGLang 更原生的高性能实现上。

## 8.2 当前问题

当前 STAR 中以下路径仍较“散”：

1. `SATLayerNorm`
2. `_apply_rotary`
3. `_modulate`
4. `adaLN_modulations` 输出后的大量 `chunk + mul + add`
5. final layer 中的 modulation 路径

这些路径会造成：

1. kernel 数量偏多
2. compile 难以形成最佳 fusion
3. attention 前后边界不够紧
4. 单卡时间主要花在 attention 之外的准备/收尾上

## 8.3 计划改动文件

1. [star_cogvideox_sr.py](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/models/dits/star_cogvideox_sr.py:1)
2. `python/sglang/multimodal_gen/runtime/layers/` 下新增或复用 fused norm / fused modulation helper
3. 必要时新增 `python/sglang/multimodal_gen/runtime/layers/rotary/` 或相邻目录中的 STAR 适配 helper

## 8.4 实施任务

1. `P7-A1`
   盘点 SGLang 已有 fused norm / modulation / rope 能力，能直接复用的优先复用，不能复用的再为 STAR 新增 helper。
2. `P7-A2`
   把 `_StarTransformerLayer.input_layernorm` 和 `post_attention_layernorm` 从 `SATLayerNorm` 迁到 SGLang 优先路径。
3. `P7-A3`
   把 `query_layernorm_list` / `key_layernorm_list` 的 QK-LN 路径统一到同一套高性能实现。
4. `P7-A4`
   把 `_apply_rotary` 重构为更少的 reshape/cat/临时张量。
5. `P7-A5`
   把 `_modulate`、final layer modulation 与 AdaLN gating 改成更适合 compile/fusion 的实现。
6. `P7-A6`
   对比改造前后的 `T_denoise`、`T_step_avg` 和 kernel 画像。

## 8.5 验收标准

1. phase 5 baseline parity 继续通过
2. 单卡 exact 的 `T_step_avg` 明显下降
3. profiler 中 norm/modulation/rope 相关 kernel 数量下降
4. 结果写入 `compare.json`

---

## 9. P7-B：Local enhancer 高性能化

## 9.1 目标

把 STAR 特有的 `spa_local + temp_local` 从当前的纯 PyTorch 组合实现，改造成更适合 GPU 执行和 compile fusion 的路径。

## 9.2 当前问题

当前 local enhancer 会产生大量：

1. `view/permute/reshape`
2. `amax/mean/stack/cat`
3. 小张量 conv / linear 调用

这部分虽然单次不如 attention 大，但在每层、每步重复，会持续侵蚀单卡收益。

## 9.3 计划改动文件

1. [star_cogvideox_sr.py](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/models/dits/star_cogvideox_sr.py:1)
2. 必要时新增 STAR local enhancer helper，如：
   `python/sglang/multimodal_gen/runtime/models/dits/star_local_enhancer.py`

## 9.4 实施任务

1. `P7-B1`
   先 profile local enhancer 的独立耗时和 kernel 画像。
2. `P7-B2`
   优先减少中间 `permute/reshape` 次数，保证数据 layout 更稳定。
3. `P7-B3`
   评估是否把空间增强和时间增强合并为更少的运算阶段。
4. `P7-B4`
   若现有 SGLang 无对应 helper，则为 STAR 增加专用 fused helper，但接口形式应尽量贴合 runtime/layers 风格。
5. `P7-B5`
   对比 compile 前后收益，确认这块优化不是“纸面重构”。

## 9.5 验收标准

1. baseline parity 继续通过
2. 单卡 exact 下 local enhancer 自身耗时下降
3. 总 `T_denoise` 有可观改善

---

## 10. P7-C：Single-card resident fast path

## 10.1 目标

解决当前单卡 exact 仍强依赖 offload 的问题，尽量让 transformer、text encoder、VAE 在更合理的时机常驻或分段常驻，从而释放 compile、FA、quant 的收益。

## 10.2 当前问题

根据 `compare.json`，以下配置目前会失败：

1. 全部关闭 offload
2. transformer 常驻 GPU

失败位置集中在：

1. `STARConditionVideoVAEEncodingStage`

这说明当前瓶颈不是纯 denoise，而是**请求生命周期中的显存峰值组织方式**。

## 10.3 计划改动文件

1. [video_condition_vae_encoding.py](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/video_condition_vae_encoding.py:1)
2. [star_cogvideox_sr_decoding.py](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/star_cogvideox_sr_decoding.py:1)
3. [gpu_worker.py](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/managers/gpu_worker.py:1)
4. 必要时调整 pipeline 内的组件装载/释放节奏

## 10.4 实施任务

1. `P7-C1`
   增加更细粒度的 stage 级峰值显存记录，明确真正的峰值来自哪里。
2. `P7-C2`
   评估条件视频 VAE encode 后，能否更早释放不再需要的中间张量。
3. `P7-C3`
   评估 decode 前能否更积极回收 text encoder / encode 路径残留显存。
4. `P7-C4`
   设计“半常驻”策略：
   transformer 常驻，text encoder 按需下放，VAE 分阶段上下 GPU。
5. `P7-C5`
   针对 compile 路线复测单卡常驻收益。

## 10.5 验收标准

1. 单卡 exact 至少出现一条更高常驻度且不 OOM 的配置
2. 该配置下 `warm_e2e_speedup` 优于当前单卡 best
3. 质量继续通过 baseline parity

---

## 11. P7-D：Quantization 真正落地

## 11.1 目标

把 STAR 的 quantization 从“规则存在、probe 失败”推进到“至少一条质量通过且显著提速的量化配置”。

## 11.2 当前问题

当前 `FP8 probe` 更快，但质量严重失配，说明还不能直接作为 phase 7 合格解。

## 11.3 计划改动文件

1. [star_cogvideox_sr.py](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/models/dits/star_cogvideox_sr.py:1)
2. `python/sglang/multimodal_gen/configs/quantization/`
3. `python/sglang/multimodal_gen/tools/convert_hf_to_fp8.py`
4. phase 6/7 benchmark 脚本

## 11.4 实施任务

1. `P7-D1`
   先确认 FP8 失配是出在：
   权重转换、模块覆盖范围、scale 策略，还是特定层不应量化。
2. `P7-D2`
   做分组量化策略：
   attention-only、MLP-only、partial-final-layer、exclude-AdaLN、exclude-local-enhancer。
3. `P7-D3`
   对 `AWQ / SVDQuant / FP8` 分别建立最小矩阵。
4. `P7-D4`
   为 STAR 单独形成“允许量化层 / 禁止量化层”白名单，不强行套通用规则。
5. `P7-D5`
   找到至少一条 `baseline parity` 通过的量化配置。

## 11.5 验收标准

1. 至少一条单卡量化配置质量通过
2. 该配置的 `warm_e2e_speedup` 高于当前单卡 exact best
3. 若所有量化方案都失败，必须给出分层实验记录和明确技术结论

---

## 12. P7-E：Attention backend 扩展与最终对比

## 12.1 目标

把 STAR 的 attention backend 从“FA 已验证、SAGE 仅声明”推进到“所有适用 backend 都有实验结论”。

## 12.2 计划改动文件

1. [star_cogvideox_sr.py](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/models/dits/star_cogvideox_sr.py:1)
2. benchmark/profile 脚本
3. 必要时补 server args / backend config 对接

## 12.3 实施任务

1. `P7-E1`
   明确 STAR 当前设备环境中哪些 backend 真实可用。
2. `P7-E2`
   增加 `FA / SAGE / SAGE_ATTN_3 / AITER` 的统一 benchmark。
3. `P7-E3`
   如果 AITER 对 STAR 适用，则正式接入支持集合和测试矩阵。
4. `P7-E4`
   给出 backend 推荐顺序，而不是只保留“能跑哪个用哪个”。

## 12.4 验收标准

1. 每个 backend 都有“已接入”或“经验证不适用”的结论
2. 最优 backend 被写入推荐配置
3. 结果同步记录到 `compare.json`

---

## 13. P7-F：TeaCache / cache-dit 重新调优

## 13.1 目标

让已接入但当前收益偏小的 cache 路径真正发挥作用。

## 13.2 当前问题

当前 cache 组合只比纯 compile 略快，说明：

1. 参数可能不合适
2. STAR 的 CFG / residual 语义虽然兼容，但不一定是最佳 cache 切入点
3. 高噪声 / 低噪声步数掩码可能还没调到合适区间

## 13.3 计划改动文件

1. [star_cogvideox_sr.py](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/models/dits/star_cogvideox_sr.py:1)
2. `python/sglang/multimodal_gen/runtime/cache/cache_dit_integration.py`
3. `python/sglang/multimodal_gen/runtime/cache/teacache.py`
4. benchmark/profile 脚本

## 13.4 实施任务

1. `P7-F1`
   扫 TeaCache 阈值、边界步、CFG 正负缓存策略。
2. `P7-F2`
   扫 cache-dit 的 warmup、mask、block 选择参数。
3. `P7-F3`
   检查 STAR 的 residual 结构是否允许更激进的 cache 粒度。
4. `P7-F4`
   分析 cache 是否与 compile/quant 配置存在更优组合。

## 13.5 验收标准

1. 至少获得一条明显优于“纯 compile”的 cache 配置，或者
2. 给出充分证据说明 STAR 当前结构对 cache 收益有限

---

## 14. P7-G：Parallel 路径最终收口

## 14.1 目标

在单卡优先完成后，把 STAR 的多卡/并行能力补成正式可交付状态。

## 14.2 计划改动文件

1. [star_cogvideox_sr.py](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/models/dits/star_cogvideox_sr.py:1)
2. [denoising.py](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py:1)
3. distributed / parallel config 相关文件
4. benchmark/profile 脚本

## 14.3 实施任务

1. `P7-G1`
   在单卡路线稳定后，重新验证 `CFG-parallel`。
2. `P7-G2`
   逐项验证 `TP / SP / USPAttention / Ulysses / Ring` 对 STAR 的适用性。
3. `P7-G3`
   给出单卡最佳配置和双卡最佳配置，不混用结论。
4. `P7-G4`
   若某些并行方式不适用，必须给出结构性原因。

## 14.4 验收标准

1. 双卡推荐配置明确
2. 每种并行策略有清晰结论
3. 双卡速度、质量、显存结果全部记录到 `compare.json`

---

## 15. 统一 benchmark 与记录要求

phase 7 延续 phase 6 的 benchmark 口径，并额外要求：

1. 每一轮测试必须把以下内容写入 [compare.json](/sgl-workspace/sglang/docs_xzh/add_STAR/compare.json:1)
   - `label`
   - `scope`
   - `config`
   - `avg_wall_clock_s`
   - `avg_denoise_s` 或 `denoise_s_log`
   - `warm_e2e_speedup`
   - `baseline_passed`
   - `ssim_mean`
   - `mse_mean`
   - `mae_mean`
   - `notes`
2. 所有重要 benchmark 都必须落 `summary.json`
3. 若测试失败，也必须记录失败类型：
   - `oom`
   - `quality_failed`
   - `compile_unstable`
   - `backend_unsupported`

---

## 16. 推荐实施顺序

phase 7 的推荐落地顺序如下：

1. `P7-A` fused norm / rope / modulation
2. `P7-B` local enhancer 高性能化
3. `P7-C` 单卡常驻与显存路径
4. `P7-D` 量化矩阵和 parity-safe 配置
5. `P7-E` backend 扩展与最终 benchmark
6. `P7-F` cache 重调优
7. `P7-G` 多卡并行收口

如果单卡在 `P7-A + P7-B + P7-C` 后已经稳定达到 `>= 1.8x`，仍然不能提前结束 phase 7；后续任务必须继续完成，直到所有适用的底层加速都得到“已接入”或“经验证不适用”的最终结论。

---

## 17. 最终验收标准

phase 7 最终完成必须同时满足：

1. phase 5 `baseline parity` 持续通过
2. 单卡 exact 推荐配置达到：
   - 最低目标：`warm_e2e_speedup >= 1.6x`
   - 正式通过线：`warm_e2e_speedup >= 1.8x`
3. 双卡推荐配置有明确结果与记录
4. 所有适用的 SGLang 底层加速能力都得到：
   - `已接入`，或
   - `经验证不适用`
5. 所有测试配置、耗时和质量结果都已写回 `compare.json`

---

## 18. 结束条件

只有当以下问题都回答清楚，phase 7 才能结束：

1. STAR 的 norm / rope / AdaLN / local enhancer 是否已经吃到 SGLang 更底层的高性能路径
2. 单卡为什么能或不能达到 `>= 1.8x`
3. quantization 是否已经形成可发布方案
4. cache 与 backend 哪组组合最优
5. 哪些 SGLang acceleration 对 STAR 结构真正有效，哪些不适用

如果这些问题还有任何一个没有结论，就不应结束 phase 7。
