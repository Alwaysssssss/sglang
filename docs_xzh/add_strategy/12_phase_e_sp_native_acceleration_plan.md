# 12. Phase E: VividVR 原生 SP 加速实施计划

本文档用于把 `Phase E4.1` 从“多卡 runtime 接通”推进到“真正依靠 `SP` 降低单视频 latency”的实现路线。目标不是重新设计 `Vivid-VR` 语义，而是在已经冻结的 `Phase C / Phase D` 语义基线上，把 `VividVR` 的 `SP` 路径补到接近 `Wan` 当前的原生 `sequence parallel` 形态。

对应背景与上游约束：
- `Phase C`：单 clip 已验收语义基线，必须保护。
- `Phase D`：长视频 `clip split / timestep orchestration / latent merge / trim / stitch` 已验收，必须保护。
- `Phase E`：当前目标是性能收口、并行化收口与回归验收，不应混入新的语义试验。
- 本文档是 [11_phase_e_acceleration_implementation.md](./11_phase_e_acceleration_implementation.md) 中 `E4.1 SP` 分支的细化实施合同。

## 1. 文档目标

本文档回答三个问题：

1. 为什么当前 `E4.1` 的双卡 `SP` 几乎没有带来实质性加速。
2. `Wan` 当前能实现“真实多卡 `SP` 加速”的关键策略是什么。
3. `VividVR` 后续应如何补齐到类似 `Wan` 的原生 `SP` 路径，并据此指导后续代码修改。

本文档只定义后续实现方案，不在这里引入新的 backend、fusion、compile、offload 变量。

## 2. 当前问题诊断

`E4.1` 当前已经完成了真实双卡 runtime 接通，但正式验收结果说明它还不是“高质量的原生 `SP` 加速”。

### 2.1 现有正式结果

单卡 control `E3.2`：
- 指标文件：`Vivid_Acceptance/indicator/phase_e32_runtime_e2_align_130f_20step_compile_metrics_seed42_20260609T025514Z.json`
- `model_inference_runtime_seconds = 935.243947`
- `vividvr_long_video_denoising_loop = 771121.1627759039 ms`
- `avg_step_ms = 38555.42590525001`

双卡 `E4.1 SP-only`：
- 指标文件：`Vivid_Acceptance/indicator/phase_e41_sp_only_130f_20step_compile_metrics_seed42_20260611T041018Z.json`
- `model_inference_runtime_seconds = 933.725862`
- `vividvr_long_video_denoising_loop = 772165.95941782 ms`
- `avg_step_ms = 38607.76289589703`
- `world_size = 2`
- `tp_size = 1`
- `sp_degree = 2`
- `ulysses_degree = 2`
- `ring_degree = 1`

对比结果：
- 加速倍率仅 `1.0016x`
- 推理时延仅下降 `1.518085s`
- 百分比提升仅 `0.1623%`
- 最核心的 `denoising_loop` 没有下降，反而略慢

### 2.2 结论

这说明当前 `E4.1` 的状态是：
- 它不是“假双卡”，因为 distributed / model-parallel runtime 的确已经接通。
- 但它也不是“真实命中主热点的 `SP` 加速”，因为最核心的 denoise 主链没有显著变快。

因此，当前 `E4.1` 更准确的定义应是：
- 已完成 `SP` 运行时接线
- 尚未完成 `SP` 主链加速

## 3. 当前 VividVR 的 SP 为什么几乎没有加速

根本原因不是 world size 没生效，而是当前 `VividVR` 的 `SP` 更多停留在 runtime/metadata 层，没有真正演化成 `Wan` 那样的“模型原生 sequence shard”。

### 3.1 当前 VividVR 的实现特征

当前实现主要表现为：
- [vividvr_pipeline.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py:251) 只在 pipeline 初始化时接通 distributed / model-parallel runtime。
- [vividvr.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py:652) 的 `VividVRDenoisingStage` 是自定义 `PipelineStage`，没有复用通用 `DenoisingStage` 的 `SP` 预处理/后处理契约。
- [vividvr.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py:694) 基于完整 `raw_latent_shape` 构建 `attn_metadata`。
- [vividvr.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py:737) 仍然把完整 `latents / control_latents / prompt_embeds` 搬到每个 rank 上。
- [vividvr.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py:860) 在 step 内直接把完整 tile 喂给 `controlnet` 和 `transformer`。
- [cogvideox_vividvr.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr.py:174) 和 [cogvideox_vividvr_controlnet.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr_controlnet.py:247) 仍是 diffusers 风格 block loop，没有显式 `sequence shard -> local compute -> gather`。

### 3.2 这条路径为什么效果弱

因为当前路径很可能只做到：
- runtime 知道 `SP` 存在
- attention backend 能接收到部分 `SP` 上下文
- 但大部分主干计算仍然不是严格的 rank-local sequence 计算

所以实际结果会是：
- 通信和同步开销增加
- 每个 rank 仍持有完整或接近完整的主干张量
- 真正被分摊的热点很少
- 最终 `denoising_loop` 几乎不降

## 4. Wan 的 SP 策略是什么

要做“真实可加速”的 `SP`，应以 `Wan` 现有路径作为参考，而不是继续停留在当前 `VividVR E4.1` 这种浅层接线形态。

### 4.1 Pipeline 层

[wan_videoedit_pipeline.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/pipelines/wan_videoedit_pipeline.py:96) 本身并不承载 `SP` 核心算法，它只负责把 `VideoEditDenoisingStage` 组进 pipeline。

### 4.2 Stage 层

[videoedit_wan.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/videoedit_wan.py:326) 的 `VideoEditDenoisingStage` 继承的是通用 [denoising.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py:786) 里的 `DenoisingStage`，因此天然接上了通用 `SP` 生命周期：
- 预处理 latent shard
- 后处理 latent gather
- 与 pipeline config 中的 `SP` 语义保持一致

### 4.3 Config / Sampling 契约

[sampling_params.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/configs/sample/sampling_params.py:440) 会给 `wan/helios` 自动打开 `enable_sequence_shard`。

[base.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/configs/pipeline_configs/base.py:377) 明确规定：
- 如果 `enable_sequence_shard=True` 且 `sp_world_size > 1`
- 外层不再先沿时间维硬切 latent
- 因为模型内部会自己做 sequence shard

### 4.4 Model 层

[wanvideo.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/models/dits/wanvideo.py:958) 才是 `Wan` 真正能加速的关键：
- 读取 `forward_batch.enable_sequence_shard`
- 如有必要先对序列长度 pad 到 `sp_size` 的整数倍
- 按 `sp_rank` 切出 `local_seq_len`
- block 只对本 rank 的局部序列计算
- `timestep_proj` 等序列相关张量做相同切分
- 在末尾 `sequence_model_parallel_all_gather` 并 unpad

本质上，`Wan` 是：
- 模型原生 sequence shard
- rank-local block compute
- 明确 gather/unpad

这才是“能真实降低单视频 latency”的 `SP` 形态。

## 5. VividVR 与 Wan 当前的本质差异

差异不在于“有没有设置 `sp_degree=2`”，而在于并行发生在哪一层。

`Wan` 的并行层级：
- 模型内部显式 sequence shard
- block 内局部计算
- gather 作为算法的一部分

`VividVR E4.1` 的并行层级：
- distributed runtime 接通
- attention metadata 接通
- 但没有把主链显式切成 rank-local sequence compute

更具体地说，当前 `VividVR` 存在以下缺口：

1. `VividVRDenoisingStage` 没有接入通用 `DenoisingStage` 的 `SP` 契约。
2. `VividVR` 目前不在 `enable_sequence_shard` 的启用策略内。
3. `transformer` 没有显式 sequence shard。
4. `controlnet` 没有显式 sequence shard。
5. `image_rotary_emb`、`timestep`、`attn_metadata` 仍以全局视角组织，而不是围绕 local shard 组织。
6. rank 上仍保留完整主干张量的概率很高。

## 6. VividVR 原生 SP 的目标架构

后续 `VividVR` 的 `SP` 目标架构应满足以下原则：

### 6.1 核心目标

- 让 `SP` 直接作用于 denoise 主链的真实热点。
- 让每个 rank 只处理本 rank 的 sequence shard，而不是完整主干张量。
- 在不破坏 `Phase C / D` 语义的前提下，真正降低单视频 `model_inference_runtime_seconds`。

### 6.2 必须保护的语义

以下语义不能因为 `SP` 改造而回退：
- `Phase C` 单 clip 语义
- `Phase D` 长视频 `clip split`
- 多 clip 的 timestep 级同步推进
- overlap latent merge
- trim / stitch
- decode / postprocess
- `drop first 3 frames + crop padding + AdaIN/reference color fix`

### 6.3 明确禁止的错误方向

以下方向不是本文档允许的 `SP` 实现：
- 把 `clip0` 放到 `GPU0`，`clip1` 放到 `GPU1`，最后再拼接
- 把多卡阶段和新 backend / 新 fusion / 新 compile 混在同一次验收中
- 为了并行方便而改写 `Phase D` 已验收时序语义

## 7. 推荐的实施分阶段

建议把原生 `SP` 补齐分为七个阶段推进。

### 阶段 0：冻结基线与补全观测

目标：
- 把当前 `E4.1 SP-only` 冻结为“接线 control”
- 为后续原生 `SP` 改造增加更细的诊断指标

建议补充的观测字段：
- `enable_sequence_shard`
- `sp_world_size`
- `sp_rank`
- `sp_local_seq_len`
- `sp_seq_pad`
- `sp_shard_mode`
- `denoise_loop_local_compute_ms`
- `denoise_loop_sp_comm_ms`（若能精确拆出）

验收要求：
- 此阶段不追求提速
- 只追求让后续每一步能确认到底有没有真正进入 local-shard 路径

### 阶段 1：Sampling / Pipeline 契约对齐

目标：
- 让 `VividVR` 正式接入 `enable_sequence_shard` 语义，而不是只靠 `attn_metadata`

建议工作：
- 在 `VividVR` 对应 sampling / defaults / pipeline config 里增加显式 `enable_sequence_shard` 入口
- 不要把这个开关无差别扩散到所有模型
- 保证：
  - `enable_sequence_shard=False` 时行为退回当前 control
  - `enable_sequence_shard=True` 且 `sp_world_size > 1` 时，外层不再先做错误的 latent 硬切

目标文件：
- [vividvr.py sample config](/home/zhiheng/sglang/python/sglang/multimodal_gen/configs/sample/vividvr.py)
- [vividvr pipeline config](/home/zhiheng/sglang/python/sglang/multimodal_gen/configs/pipeline_configs/vividvr.py)
- 如有需要，补充 [sampling_params.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/configs/sample/sampling_params.py)

### 阶段 2：Stage 层 SP 生命周期对齐

目标：
- 让 `VividVRDenoisingStage` 进入与 `Wan` 一致的 `SP` 生命周期

两种可行方向：

方向 A：
- 让 `VividVRDenoisingStage` 尽量复用 `DenoisingStage`
- 把自定义长视频编排逻辑叠加在通用 stage 生命周期上

方向 B：
- 保留自定义 stage
- 但显式复制等价的 `SP preprocess / postprocess` 契约

这一步的重点不是“继承关系必须和 `Wan` 一模一样”，而是必须保证：
- `enable_sequence_shard=True` 时，不错误地做外层 latent 时间切片
- `enable_sequence_shard=False` 时，仍能回到既有 control 行为
- `SP` 前后的张量组织和 gather 边界是确定的

主文件：
- [vividvr.py stage](/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py)
- [denoising.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py)

### 阶段 3：Transformer 原生 sequence shard

目标：
- 让 `cogvideox_vividvr.py` 真正像 `wanvideo.py` 一样在模型内部做 local sequence compute

建议路径：

1. 在 patch/embed 后拿到视频 token 序列。
2. 在进入 block 前判断：
   - `forward_batch.enable_sequence_shard`
   - `get_sp_world_size() > 1`
3. 对视频 token 序列长度做 pad，使其可被 `sp_size` 整除。
4. 按 `sp_rank` 切出 `local_seq_len`。
5. 本 rank 只保留局部 `hidden_states`。
6. 与视频 token 对齐的 `image_rotary_emb` 也做 local 切片。
7. block loop 仅在 local shard 上运行。
8. 在需要恢复全局视图的位置再 `sequence_model_parallel_all_gather` 并 unpad。

第一版可以接受的保守策略：
- text encoder hidden states 先保持 replicated
- patch embed / 某些轻量预处理先保持 replicated
- 只要 block 主算力切到 local shard，就已经比当前状态前进很多

高风险点：
- `CogVideoX` / `VividVR` 当前是 diffusers 风格实现，补 sequence shard 时要非常谨慎，不能破坏输出 token 与原始时空布局的对应关系

主文件：
- [cogvideox_vividvr.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr.py)
- 必要时同步补 [cogvideox_vividvr_common.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr_common.py)

### 阶段 4：ControlNet 原生 sequence shard

目标：
- 让 `controlnet` 路径与主 transformer 保持同样的 local shard 语义

原因：
- 如果 transformer 已经 local shard，但 controlnet 仍在全局张量上运行，主热点仍会被大量保留
- control feature 若以全局形态产生，再回喂本地 transformer，也会引入额外 gather/scatter 开销

建议路径：
- 对 `controlnet` 也沿相同 token 维度做 pad / shard / local compute / gather
- 保持 `controlnet_hidden_states` 在局部 shard 语义下传递
- 让 transformer 消费局部 control residual，而不是每步都退回全局 residual

主文件：
- [cogvideox_vividvr_controlnet.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr_controlnet.py)

### 阶段 5：RoPE / Timestep / Forward Context 收口

目标：
- 清理当前“全局 metadata + 局部并行”的不一致点

建议关注：
- `image_rotary_emb` 应从“基于完整序列预构建”调整为“可按 local shard 切分或局部生成”
- 若 `timestep_proj` 或其他序列相关张量沿 token 维广播，也要保持和 local shard 一致
- `attn_metadata` 的角色应降级为辅助上下文，而不是当前 `SP` 唯一承载机制

主文件：
- [vividvr.py stage](/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py)
- [cogvideox_vividvr.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr.py)
- [cogvideox_vividvr_controlnet.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr_controlnet.py)

### 阶段 6：与 Phase D 长视频编排重新对齐

目标：
- 确认 `SP` 只作用在“每个 clip/tile 内部的 denoise 主链”，而不是改写长视频 orchestration 语义

必须守住：
- 多 clip 仍在 timestep 级同步推进
- overlap latent merge 仍按原有时序发生
- trim / stitch 逻辑保持不变
- 当前 benchmark 即使只有 `tile_count=1`，后续也不能把 `tile` 语义破坏掉

这里的正确理解是：
- `SP` 是单个 denoise compute 的内部并行
- 不是 long-video orchestration 的替代品

### 阶段 7：分阶段验收

建议每个里程碑都先 smoke，再 formal。

smoke：
- `2 GPU`
- `SP=2`
- `2 step`
- 程序自然退出
- 无 deadlock / hang / rank mismatch

formal：
- 固定 `130f / 20 step / seed=42`
- 当前 attention backend 不变
- 当前 compile 开关不变
- 连续至少两次稳定运行

formal 通过要求：
- `pass_compare = true`
- 并行配置记录完整
- `model_inference_runtime_seconds` 下降
- 更关键的是 `vividvr_long_video_denoising_loop` 要有实质下降，而不是只有总 runtime 噪声波动

## 8. 文件级改动清单

后续真正进入实现时，优先关注以下文件：

配置与契约：
- [vividvr defaults](/home/zhiheng/sglang/python/sglang/multimodal_gen/configs/vividvr_defaults.py)
- [vividvr sample config](/home/zhiheng/sglang/python/sglang/multimodal_gen/configs/sample/vividvr.py)
- [vividvr pipeline config](/home/zhiheng/sglang/python/sglang/multimodal_gen/configs/pipeline_configs/vividvr.py)
- [sampling_params.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/configs/sample/sampling_params.py)

pipeline / stage：
- [vividvr_pipeline.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py)
- [vividvr.py stage](/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py)
- [denoising.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py)

模型主链：
- [cogvideox.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/models/dits/cogvideox.py)
- [cogvideox_vividvr.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr.py)
- [cogvideox_vividvr_controlnet.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr_controlnet.py)
- [cogvideox_vividvr_common.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr_common.py)

测试：
- `python/sglang/multimodal_gen/test/unit/`
- 后续应新增覆盖：
  - `enable_sequence_shard` 开关语义
  - seq pad / unpad
  - local seq len / rank mapping
  - transformer/controlnet shard 一致性
  - gather 后输出 shape 与原路径一致

## 9. 推荐的实现顺序

建议顺序如下：

1. 先补 `enable_sequence_shard` 契约与观测字段。
2. 再处理 stage 层 `SP` 生命周期。
3. 然后优先改 `transformer` 主链的原生 sequence shard。
4. 再改 `controlnet` 的原生 sequence shard。
5. 最后统一清理 `RoPE / timestep / metadata / gather`。
6. 每一步都跑 smoke，关键里程碑再跑 formal。

这样做的原因是：
- 能尽早区分“没有进入原生 `SP` 路径”和“进入了但性能仍不够好”
- 能减少一次性大改造成的定位困难
- 能确保任何一步回归都能回到上一个可解释 control

## 10. 非目标与边界

本文档不包含以下事项：
- `TP` 设计
- `CFG parallel` 设计
- `DP` 设计
- 新 attention backend 试验
- 新 fusion / compile / offload 试验

这些事项应继续留在后续 `E4.2 / E4.3` 或其他专门文档中，不应混入当前 `VividVR` 原生 `SP` 改造阶段。

## 11. 风险与决策点

后续实现前，建议先明确两个关键决策：

### 决策点 A：Stage 是否要强行继承 `DenoisingStage`

两个可接受答案：
- 是：尽量向通用 stage 体系收敛，减少重复 `SP` 契约
- 否：保留自定义 `VividVRDenoisingStage`，但显式补齐等价 `SP` 生命周期

这里不应追求“继承关系好看”，而应优先保证：
- 长视频语义不回归
- `SP` 生命周期清晰
- 调试与验收口径可追踪

### 决策点 B：Transformer 和 ControlNet 是否必须同 patch 落地

建议：
- 可以分两步落地
- 但默认对外开关不应在“只改 transformer、controlnet 仍全局”的半成品状态下长期暴露为正式优化路径

换句话说：
- 允许阶段性开发拆分
- 不允许把半成品直接宣称为“原生 `SP` 加速完成”

## 12. 最终验收口径

当本文档对应实现真正完成时，验收结论应满足：

1. `pass_compare = true`
2. 无 deadlock / hang / rank mismatch
3. `Phase C / D` 已验收语义无回归
4. 并行配置与 shard 诊断字段记录完整
5. 相比当前 `E4.1 SP-only control`，`vividvr_long_video_denoising_loop` 有明确下降
6. 相比单卡 `E3.2 control`，`model_inference_runtime_seconds` 有清晰、可重复的下降

如果只满足 1-4，但不满足 5-6，则应把结果定义为：
- 原生 `SP` 路径接通成功
- 但未形成有效加速

而不是直接宣称 `E4.1` 已实现高质量多卡加速。
