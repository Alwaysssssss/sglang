# VividVR Phase E 加速现状、完成度与 E3 后续执行计划交接

## 1. 文档目的

这份交接文档用于统一收口本轮对话中已经完成的分析结论、当前项目完成情况，以及后续工作的实际执行顺序。

本文档的作用不是替代已有 handover，而是把下面几类信息汇总到一处：

- 当前 `Phase E` 的正式结论到底是什么。
- `VividVR` 现在已经接入了哪些 `sglang` 加速。
- 它与 `Wan VideoEdit` 这类 `sglang` 原生视频模型相比还差什么。
- 为什么 `VividVR` 不能像 `Wan` 一样直接自然吃满 `sglang` 底层加速。
- `E3` 后续该怎么推进，应该先做什么、后做什么。
- 当前阶段哪些事情已经完成，哪些还只是文档层面的计划。

本轮只做了代码与文档分析，以及实施文档细化，没有做新的代码修改，也没有新增 benchmark 验收。


## 2. 本轮完成了什么

本轮对话已经完成的工作包括：

1. 仔细阅读并消化了：
   - `docs_xzh/hand_over/phase_e_e3_manual_fusion_priority_and_qk_norm_rope_handover.md`
2. 在此基础上，重新判断了：
   - 当前哪些 `sglang` 底层融合能力值得继续复用
   - 哪些路径更可能在 DiT 主耗时阶段带来实质收益
3. 对比分析了：
   - `VividVR` 当前接入的底层加速
   - `wan_videoedit_pipeline.py` 接入的底层加速
   - 二者在 runtime 骨架、attention backend、算子融合和辅助运行时能力上的差异
4. 明确回答了两个关键问题：
   - `VividVR` 不能直接照搬 `Wan` 的 `DenoisingStage.forward()` 主循环
   - 但 `VividVRDenoisingStage` 可以逐步复用 `DenoisingStage` helper
5. 对 `docs_xzh/add_strategy/11_phase_e_acceleration_implementation.md` 做了细化和纠偏：
   - 补了当前正式结论锚点
   - 纠正了过时的 `E2 / E3` 口径
   - 把后续 `E3` 计划拆成了 `E3.1 / E3.2 / E3.3`


## 3. 当前正式结论

这部分是后续所有工作都必须保护的正式口径。

### 3.1 已经冻结的正式结论

- `Phase D` 长视频主语义已经完成验收。
- 当前单卡最好正式结果仍是：
  - `Phase E2 = FA + torch.compile`
  - `model_inference_runtime_seconds = 923.9699`
- 当前正式通过的 `Phase E3` 只有：
  - `modulation / residual fusion`
  - `model_inference_runtime_seconds = 1007.328337`

### 3.2 不能写成当前正式默认配置的路径

下面这些路径都已经实现、接线或实验过，但当前不能写成现行正式 `E3` 默认方案：

- 浅层 `QKV fusion`
- `QK norm + RoPE`

原因分别是：

- 浅层 `QKV fusion` 现在还只是 diffusers 风格的 `to_qkv -> split -> flash_attn` 路线，不是更深层 packed-QKV 融合。
- `QK norm + RoPE` 虽然已经端到端跑通，并有小幅局部收益，但正式 compare 失败，当前存在语义回归风险。

### 3.3 一个必须持续强调的现实

当前单卡正式最优仍是 `E2`，不是“所有 fusion 全开”。

这意味着：

- 后续 `E3` 新工作应被视为“对当前正式结果的扩展候选”。
- 后续文档和代码都不应擅自改写 `E2` 仍是正式单卡最优这一事实。


## 4. 当前 VividVR 已接入的加速能力

`VividVR` 现在并不是“完全没接上 `sglang` 加速”，而是已经接入了一批能力，只是整体形态更像“diffusers 模型上的局部 patch”，还不是 `sglang` 原生化模型。

### 4.1 已正式通过并可视为当前稳定路径的能力

- `FA` attention backend
- `FA + torch.compile`
- `modulation / residual fusion`

### 4.2 已实现但当前不属于正式默认路径的能力

- 浅层 `QKV fusion`
- `QK norm + RoPE`

### 4.3 运行时层面已经存在的能力

- pipeline 初始化阶段对 `transformer / controlnet` 尝试 `torch.compile`
- `text encoder` CPU offload
- `VAE` CPU offload
- `VividVR` 自己的 spatial / temporal tiling 推理路径
- pipeline 外层已有部分 `StageProfiler` 记录

### 4.4 当前还没自然吃到的能力

- `DenoisingStage` 标准 `autocast` 路径
- `DenoisingStage` 标准 `_manage_device_placement(...)`
- `DenoisingStage` 标准 `attn_metadata builder`
- `cache-dit`
- metadata 驱动的更原生 attention/backend 接线
- decode 侧 `vae.enable_tiling()` 的实际接线


## 5. 与 Wan VideoEdit 的差异

`Wan VideoEdit` 之所以能更自然地吃到 `sglang` 底层加速，不是因为多了一个开关，而是因为它从模型到 runtime 结构都更接近 `sglang` 原生抽象。

### 5.1 Wan 已经具备的条件

- denoise 结构更接近 `DenoisingStage` 的默认假设
- transformer 是 `sglang` 原生 `CachableDiT / OffloadableDiTMixin` 路线
- denoise 主循环已接上：
  - `autocast`
  - `_manage_device_placement(...)`
  - `_build_attn_metadata(...)`
  - `cache-dit`
- decode 路径显式支持 `vae.enable_tiling()`
- block 内部大量使用 `sglang` 原生融合件：
  - `LayerNormScaleShift`
  - `ScaleResidualLayerNormScaleShift`
  - `MulAdd`
  - 原生 attention backend

### 5.2 VividVR 当前的不同点

- 仍然是 diffusers 风格 `CogVideoX` 扩展
- denoise 走自定义 `VividVRDenoisingStage`
- transformer 调用时目前仍显式传 `attn_metadata=None`
- 当前结构不是 `CachableDiT`
- 目前很多能力是通过 pipeline 初始化时手动 patch 模型组件实现

### 5.3 结论

`VividVR` 不能像 `Wan VideoEdit` 一样“天然挂进 `sglang` 原生骨架后就把底层加速基本吃上”。

它需要的是：

- 逐步复用已有 runtime helper
- 逐步把 attention/runtime 接线拉回 `sglang` 原生通路
- 再在此基础上推进更深的热点融合


## 6. 为什么 VividVR 不能直接改成通用 DenoisingStage.forward()

这个问题本轮已经分析清楚，结论如下。

### 6.1 不是不能复用 DenoisingStage

`VividVRDenoisingStage` 不是完全不能复用 `DenoisingStage`。

它的问题在于：

- 不能直接替换成通用 `DenoisingStage.forward()`
- 但可以逐步复用 `DenoisingStage` helper

### 6.2 不能直接套通用 forward 的主要原因

`VividVR` 当前 denoise 语义和通用模板差得比较大：

- 它是 `controlnet -> transformer` 两段式
- 每个 timestep 内部有 spatial / temporal tiling
- tile 内有 overlap merge 逻辑
- CFG 也不是通用两次 forward 形式，而是 tile 内拼 batch 一次过模型
- scheduler 带 `old_pred_original_sample`、`prev_timestep`、`restoration_guidance_scale` 等特有状态

这些都决定了：

- 不能直接把主循环切成通用 `DenoisingStage.forward()`
- 否则容易破坏 `Phase D` 已验收的长视频主语义

### 6.3 现有 VividVRDenoisingStage 还能不能继续用

能继续用，而且短期内应该继续作为语义基线使用。

但它更适合作为：

- 当前稳定实现

而不是：

- 所有后续底层加速都继续手工外挂的终局形态

后续更合理的方向是：

1. 保留它的自定义主语义
2. 逐步收敛它对 `DenoisingStage` helper 的复用


## 7. 对后续可复用融合方向的判断

本轮对“哪些底层算子融合值得继续做”已经有比较明确的优先级判断。

### 7.1 第一优先级：大投影入口融合

最值得继续投入的方向不是继续打磨浅层小 fusion，而是打到 DiT 主耗时的更深层入口：

- `packed-QKV`
- `merged projection`
- 进一步看 `QKV + MLP in-proj`

原因：

- 这类融合命中的是每层每步的大矩阵乘与 attention 入口
- 更接近 `sglang` 其他原生 DiT 模型已经验证过的成熟方向

`sglang` 内已有类似先例：

- `Flux`
- `Flux 2`
- `Hunyuan3D`
- `HunyuanVideo`

### 7.2 第二优先级：更深的 norm / residual / gate 链式融合

当前正式通过的 `modulation / residual fusion` 已经说明这条线能成立，但收益上限不高。

如果继续做，应该朝更深的链式路径推进，例如：

- `layernorm + scale/shift`
- `residual add + gate + next norm`
- `residual + gate * ff_output`

而不是仅停在当前浅层版本。

### 7.3 候选支线：QK norm + RoPE

这条路径不是完全没价值，但当前不能作为主线：

- 已端到端跑通
- 局部有小幅收益
- 但正式 compare 失败

因此当前只能保持为候选支线，不应升格为正式默认方案。

### 7.4 低优先级：当前这种浅层 QKV fusion

当前 `QKV fusion` 的价值有限，原因是它不是深层 packed-QKV，只是把三次线性改成一次后又立刻拆回去。

所以它可以保留，但不应被视为后续主战场。


## 8. 本轮对 E3 后续规划的重置

本轮最大的结论之一，是把后续 `E3` 的组织方式彻底改掉了。

### 8.1 旧理解的问题

旧文档里更像是在按：

1. `QK norm + RoPE`
2. `layernorm / residual / scale_shift / gate`
3. 其他 norm 热点

这样的思路排优先级。

这个顺序已经不适合当前项目现实，因为：

- `QK norm + RoPE` 已经证明当前不是正式可用主路径
- 当前单卡最优仍是 `E2`
- `VividVR` 更大的短板不在“还少一个 kernel”，而在 runtime/helper 与 native backend 接线没有收敛

### 8.2 新的组织方式

新的 `E3` 规划已经被重写为三个顺序依赖的子阶段：

- `E3.1`：runtime helper 收敛
- `E3.2`：attention/runtime 原生化与低风险补线
- `E3.3`：深层热点融合

这是本轮交接里最重要的后续执行框架。


## 9. 已经落地到文档的产物

本轮已完成的实际文档修改是：

- 细化并更新：
  - `docs_xzh/add_strategy/11_phase_e_acceleration_implementation.md`

这份文档现在已经补了：

- 当前正式结论锚点
- `VividVR` 当前接线现实
- `E3.1 / E3.2 / E3.3` 执行清单
- `E1 / E2` 的冻结口径

也就是说：

- 后续代码工作虽然还没开始
- 但后续执行计划已经从“宽泛方向”收束成了可落地任务序列


## 10. 当前项目完成情况

这里按“正式验收 / 文档规划 / 后续代码工作”三层来描述当前完成度。

### 10.1 已正式完成

- `Phase D`：已完成长视频主语义验收
- `Phase E1`：`FA` 作为单卡正式 backend 基线已经成立
- `Phase E2`：`FA + torch.compile` 已形成当前单卡最佳正式结果
- 当前正式 `Phase E3`：`modulation / residual fusion` 已独立验收通过

### 10.2 已完成分析与规划，但还未进入新的代码落地

- `VividVR` 与 `Wan VideoEdit` 的差异分析
- `VividVR` 为什么不能直接套通用 `DenoisingStage.forward()` 的判断
- `VividVR` 当前已接入与未接入加速能力的盘点
- 后续 `E3` 执行顺序重构
- 实施文档 `11_phase_e_acceleration_implementation.md` 的细化

### 10.3 仍未开始或未完成的工作

- `E3.1` 代码级 helper 收敛
- `E3.2` metadata/backend 原生化与 decode 低风险补线
- `E3.3` 深层热点融合
- `E4` 多卡并行
- `E5` 最终组合回归


## 11. 后续工作安排

后续建议按下面顺序推进，不建议再回到“多个变量一起开”的节奏。

### 11.1 第一阶段：E3.1 runtime helper 收敛

目标：

- 不碰主语义，先补 runtime 基础设施

具体任务：

1. 审计 `VividVRDenoisingStage` 与 `DenoisingStage` helper 的差异面
2. 接 `autocast`
3. 评估并接 `_manage_device_placement(...)`
4. 对齐 profiling / report helper
5. 为 metadata builder 预留统一接口

这一步的核心不是拿速度数字，而是为后面所有加速收口铺路。

### 11.2 第二阶段：E3.2 attention/runtime 原生化与低风险补线

目标：

- 让 `VividVR` 不再长期停留在 `attn_metadata=None`

具体任务：

1. 建 metadata builder 接口
2. 审计 backend 在 transformer / controlnet 两条支路的真实生效性
3. 补充 report 中的 runtime/backend 记录
4. 补 decode 侧 `vae.enable_tiling()`，或明确记录为什么不接

### 11.3 第三阶段：E3.3 深层热点融合

目标：

- 在 runtime 和 backend 路径收敛之后，才真正开始追求更大的 DiT 主耗时优化

具体任务：

1. 重新 profile DiT 主耗时
2. 以已通过的 `modulation / residual fusion` 为参考线
3. 优先评估：
   - `packed-QKV`
   - `merged projection`
   - 必要时 `QKV + MLP in-proj`
4. `QK norm + RoPE` 只保留为候选支线

### 11.4 第四阶段：E4 多卡

当前 benchmark 是单视频 latency 场景，后续多卡优先级建议为：

1. `SP`
2. `TP`
3. 可选 `CFG parallel`

`DP` 不是当前主线。

### 11.5 第五阶段：E5 最终组合回归

只有在所有单项都各自通过独立验收后，才进入最终组合回归。


## 12. 执行纪律

后续工作必须继续坚持下面这些规则：

1. 一次只改一个主要变量
2. 每个单项都必须有 `control / treatment`
3. 日常验收固定用 `130f / 20 step`
4. `pass_compare=true` 仍是硬门槛
5. 任何新路径都必须有明确回退开关
6. 不能用最终组合结果反推某个单项技术“有效”


## 13. 当前主要风险与开放问题

### 13.1 runtime/helper 收敛时不要冲掉现有 E2

`FA + torch.compile` 当前是单卡正式最优，因此后续 `E3.1` 做 helper 收敛时，必须避免误伤现有 compile 稳态路径。

### 13.2 attention backend 的“名义支持”不等于“真实生效”

当前文档和 CLI 上的 backend 参数暴露，不自动等于 transformer 和 controlnet 两条支路都真实吃到了 backend 选择。

### 13.3 QK norm + RoPE 的 parity 问题还没真正解决

当前最可疑的点仍然是 fast path 数值语义对齐，尤其是 image-token RoPE 路径。

### 13.4 decode 侧 VAE tiling 虽然低风险，但仍要做一次验收

即使这项更像漏接补齐，也仍然应该按单变量方式验证，不要直接混进别的更大改动里。


## 14. 后续建议重点关注的文件

- `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py`
- `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py`
- `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/videoedit_wan.py`
- `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py`
- `python/sglang/multimodal_gen/configs/pipeline_configs/vividvr.py`
- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox.py`
- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr.py`
- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr_controlnet.py`
- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_attention_backend.py`
- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_operator_fusion.py`
- `python/sglang/multimodal_gen/runtime/models/dits/wanvideo.py`
- `python/sglang/multimodal_gen/runtime/models/dits/flux_2.py`
- `python/sglang/multimodal_gen/runtime/models/dits/hunyuanvideo.py`
- `docs_xzh/add_strategy/11_phase_e_acceleration_implementation.md`


## 15. 交接结论

一句话总结当前状态：

`VividVR` 现在已经接上了一部分 `sglang` 加速，但整体仍是“diffusers 模型上的局部接线”，还没有像 `Wan` 那样自然站到 `sglang` 原生 runtime 骨架上；因此后续最合理的路线不是继续堆一个个浅层 fusion，而是先做 `E3.1` 和 `E3.2` 的 runtime/helper 与 backend 收敛，再进入 `E3.3` 的深层热点融合。

另一句必须保护的结论是：

当前正式单卡最优仍是 `E2 = FA + torch.compile`，后续所有工作都应在不破坏这个锚点的前提下推进。
