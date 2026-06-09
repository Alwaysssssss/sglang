# Phase E: 直接复用 SGLang 底层加速的实施细化

## 1. 文档目标

这份文档用于把 `Phase E` 从“性能收口 + 回归验收”的泛目标，进一步细化成后续可执行的实现路线。

本阶段默认前提已经变化为：

- `Phase D` 长视频主语义已完成验收。
- 后续不再把主要精力放在 `caption fairness` 或 clip orchestration 语义补丁上。
- `Phase E` 的重点是让当前 `VividVR` 接入尽可能直接吃到 `sglang` 现有底层加速工程，而不是继续堆积只对单模型有效的一次性优化。

本阶段的核心要求是：

1. attention backend 接入 `FA / SageAttention` 等 `sglang` 已有后端。
2. 冻结并保护当前单卡最佳正式路径：`FA + torch.compile`。
3. 让 `VividVRDenoisingStage` 逐步复用 `sglang` 现有 denoise runtime helper，而不是继续长期平行维护一套特例路径。
4. 对后续高热度算子优先复用 `sglang` 现有 fused kernel，并把重点放在真正命中 DiT 主耗时的深层热点上。
5. 实现多卡并行推理加速版本，优先复用 `sglang` 现有分布式并行基础设施。


## 2. 固定 benchmark 与验收口径

`Phase E` 后续所有日常性能迭代，默认都固定在下面这组口径上，不再自由漂移：

- 输入视频：
  - `/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.mp4`
- caption sidecar：
  - `/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.txt`
- reference 视频：
  - `/home/zhiheng/Vivid-VR/result/720p_long_up1_result_vivid_ori_20step/videos/test_video_long_960x720_130f.mp4`
- `seed=42`
- `num_temporal_process_frames=121`
- `num_inference_steps=20`
- `guidance_scale=6`
- `restoration_guidance_scale=-1.0`

补充约束：

- `20 step` 是 `Phase E` 日常 profile 和收口档位。
- `50 step` 不再作为日常迭代必跑项，只保留给阶段性最终回归。
- 质量 reference 对象与 `Phase D` 相同，不重新换 reference 视频。

当前可作为 `Phase E` 质量基线的已验收 `sglang` 指标文件是：

- `/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_d_130f_20step_metrics_seed42_20260605T083022Z.json`

其中当前基线核心值为：

- `pass_compare = true`
- `ssim_mean = 0.984745`
- `ssim_min = 0.979066`
- `mse_mean = 12.207597`
- `mse_max = 20.391026`
- `mae_mean = 2.663361`
- `mae_max = 2.909005`
- `failed_frame_ratio = 0.0`
- `model_inference_runtime_seconds = 1075.420882`

当前原版 `20 step` runtime report 为：

- `/home/zhiheng/Vivid-VR/result/720p_long_up1_result_vivid_ori_20step/videos/test_video_long_960x720_130f_report.json`

其中当前原版核心时间为：

- `model_inference_runtime_seconds = 1047.001905`

后续 `Phase E` 默认以该字段计算主加速倍数：

```text
speedup_vs_original = original_model_inference_runtime_seconds / phase_e_model_inference_runtime_seconds
```

即：

```text
speedup_vs_original = 1047.001905 / phase_e_model_inference_runtime_seconds
```

可选的内部跟踪值：

```text
speedup_vs_phase_d_sglang = 1075.420882 / phase_e_model_inference_runtime_seconds
```

质量验收要求分两层：

- 硬门槛：继续满足现有 compare 通过条件，即 `pass_compare=true`。
- 软门槛：质量结果应和上述 `Phase D 20 step` 已验收 JSON 保持接近；如果 summary 明显漂移，即使宽松阈值仍过线，也按回归处理。

### 2.1 单项加速的验收方法学

`Phase E` 必须明确区分“单项加速验收”和“最终组合回归”。

单项加速验收阶段的硬规则是：

- 每次只允许引入一个新的主要加速变量。
- 不允许在同一轮结果验收里，同时新开两个或以上加速方法。
- 做某一项加速的收益测量时，除该项加速开关外，其余配置必须保持一致。

推荐采用下面的 A/B 口径：

- `control`：当前上一阶段已冻结的稳定配置，不开启本轮新增加速项。
- `treatment`：仅在 `control` 基础上打开本轮新增加速项。

对应的单项增益定义为：

```text
incremental_speedup_of_feature = control_model_inference_runtime_seconds / treatment_model_inference_runtime_seconds
```

同时仍保留相对原版的累计口径：

```text
cumulative_speedup_vs_original = 1047.001905 / treatment_model_inference_runtime_seconds
```

需要特别强调：

- `incremental_speedup_of_feature` 用于回答“这一项新加速本身带来了多少收益”。
- `cumulative_speedup_vs_original` 用于回答“当前阶段累计相比原版快了多少”。
- 组合实验只能放在所有单项加速都各自验收通过之后，作为最终集成回归的一部分。
- 组合实验的结果不能反过来充当某个单项加速的独立收益结论。

### 2.2 当前正式 Phase E 结论锚点

在继续细化后续实施计划前，需要先冻结当前已经成立的正式结论：

- 当前单卡最好正式结果仍是 `Phase E2 = FA + torch.compile`。
- 当前单卡最好正式 `model_inference_runtime_seconds = 923.9699`。
- 当前正式通过的 `Phase E3` 只包含 `modulation / residual fusion`，其正式 `model_inference_runtime_seconds = 1007.328337`。
- 当前已接线但不能写成正式 `E3` 默认配置的路径包括：
  - 浅层 `QKV fusion`
  - `QK norm + RoPE`

这意味着后续 `E3` 新工作应被视为“在现有正式结论之上的扩展候选”，而不是改写上述锚点本身。


## 3. 当前代码现实与 Phase E 的真实缺口

在开始 `Phase E` 代码修改前，需要先明确当前代码并不是“只差打开开关”。

### 3.1 已有基础设施

`sglang` 底层已经具备的能力包括：

- attention backend 抽象与多种后端实现：
  - `python/sglang/multimodal_gen/runtime/layers/attention/backends/*`
- attention backend 选择器：
  - `python/sglang/multimodal_gen/runtime/layers/attention/selector.py`
- `ServerArgs` 中的 backend / compile / parallelism 参数：
  - `python/sglang/multimodal_gen/runtime/server_args.py`
- 通用 profiler / stage timer：
  - `python/sglang/multimodal_gen/runtime/utils/perf_logger.py`
  - `python/sglang/multimodal_gen/runtime/utils/profiler.py`
- 分布式并行状态与 TP/SP/DP/CFG 基础设施：
  - `python/sglang/multimodal_gen/runtime/distributed/parallel_state.py`
- 已有 diffusion fused kernel：
  - `python/sglang/jit_kernel/diffusion/*`

### 3.2 当前 VividVR 路径的关键缺口

当前 `VividVR` 路径至少有下面几个现实问题：

- `run_vividvr_inference.py` 已经暴露了 `--attention-backend`、`--attention-backend-config`、`--enable-torch-compile`，但 `build_server_args()` 仍然把：
  - `num_gpus=1`
  - `tp_size=1`
  - `dp_size=1`
  - `sp_degree=1`
  写死成单卡。
- `VividVR` 走的是自定义 `VividVRDenoisingStage`，当前没有复用通用 `DenoisingStage` 里的 `autocast`、`_manage_device_placement(...)`、`_build_attn_metadata(...)`、`cache-dit` 入口等 helper，因此后续 runtime 能力仍在分叉维护。
- `VividVR` 当前已经在 pipeline 初始化阶段对 `transformer / controlnet` 尝试 `torch.compile`，并且 `FA + torch.compile` 已经是当前单卡最好正式结果；因此 `Phase E` 后续不应再把 compile 误写成“尚未接线”，而应把重点放在保护现有 E2 成果、减少与通用 helper 路径的分叉。
- `CogVideoXVividVRTransformer3DModel` 和 `CogVideoXVividVRControlNetModel` 当前仍大量直接依赖 `diffusers` 的 `CogVideoXBlock`；因此 `CogVideoXConfig` 上声明了 `_supported_attention_backends`，不等于 `VividVR` 当前主链已经真正吃到了 `sglang` attention backend。
- 当前 `VividVR` denoise 调 transformer 时显式传的是 `attn_metadata=None`，说明 metadata 驱动的 backend 路径还没有真正接到主链上。
- 配置里已经有 `vae_tiling=True`，但 decode 路径没有像 `Wan VideoEdit` 那样显式 `enable_tiling()`，这是一个低风险但真实存在的漏接项。
- 当前 `runtime/entrypoints/cli/utils.py` 里的 `launch_distributed()` 目标脚本仍是旧路径，不能直接作为 `VividVR` 多卡入口拿来即用。

`Phase E` 的真实任务不是“再加几个 CLI 参数”，而是把这些已经存在的底层能力，确实接到 `VividVR` 的真实执行路径上。


## 4. 实施原则

### 4.1 总原则

- 先复用 `sglang` 现有底层能力，再考虑新写 `VividVR` 专属实现。
- 先把加速路径“接通”，再做更深的局部替换。
- 一次只推进一个主要加速变量，避免多个变量叠加后无法归因。
- 每个阶段都要保留可直接回退到上一稳定配置的能力。
- 所有优化都以 `Phase D` 已验收长视频语义为边界，不允许为了速度重新定义主语义。

### 4.2 语义红线

以下事项在 `Phase E` 默认不允许被“性能优化”顺手改掉：

- `Phase C` 单 clip 已验收链路语义
- `Phase D` 长视频的 `clip split / timestep orchestration / latent merge / trim / stitch`
- prompt/caption 口径
- `drop first 3 frames + crop padding + AdaIN/reference color fix`

### 4.3 多卡工作的优先级原则

当前 benchmark 是单视频、`batch size = 1` 的 latency 场景，因此：

- `DP` 不是 `Phase E` 单视频加速的优先路径。
- 单视频多卡的首选方向应是：
  - `SP`
  - `TP`
  - 可选 `CFG parallel`
- 只有在要扩展到吞吐型多请求场景时，`DP` 才应作为主要目标。

### 4.4 不推荐的实现方向

- 不推荐把多 clip 直接拆成“每个 clip 各自完整跑完再拼回去”的多卡方案。
- 不推荐为了 `VividVR` 单模型先写一套完全平行于 `sglang` 现有 attention/backend/distributed 的专用框架。
- 不推荐先写全新 Triton kernel，再回头验证 `sglang` 现有 fused kernel 是否其实已经够用。


## 5. Phase E 的实施分解

## 5.0 前置工作：先补 Phase E 可观测性

在正式进入四条加速线前，建议先完成一轮“观测与记录”补强。虽然当前 `VividVR` pipeline 已经有部分 `StageProfiler` 包装，但 denoise 内层的记录粒度和 runtime 配置落盘仍不完整；如果不先补这一步，后续很难判断加速到底来自哪里。

优先要补的内容：

- 在 `VividVR` 路径上尽量对齐 `sglang` 现有 `StageProfiler / SGLDiffusionProfiler` 的使用方式
- 在验收 JSON 或附属 report 里补充以下配置字段：
  - `attention_backend`
  - `attention_backend_config`
  - `enable_torch_compile`
  - `torch_compile_mode`
  - `enable_cogvideox_modulation_fusion`
  - `enable_cogvideox_qkv_fusion`
  - `enable_cogvideox_qk_norm_rope_fusion`
  - `num_gpus`
  - `tp_size`
  - `sp_degree`
  - `ulysses_degree`
  - `ring_degree`
  - `dp_size`
  - `enable_cfg_parallel`
  - `attn_metadata_enabled`
  - `vae_tiling`
  - `cuda_visible_devices` 或等价设备列表
- 对 `denoise / decode / postprocess / merge` 至少输出可对比的阶段耗时

建议优先改动位置：

- `python/sglang/multimodal_gen/tools/run_vividvr_inference.py`
- `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py`
- 如有必要，再补 pipeline/report 写出逻辑

这一步不是最终性能优化本身，但它是后续所有 Phase E 结论可复现的前置条件。


## 5.1 工作线 A：attention backend 接入

### 目标

让 `VividVR` 主 denoise 路径真正接入 `sglang` 现有 attention backend，而不是停留在 config/CLI 名义上支持。

### 直接复用的现有工程

- backend 枚举与选择：
  - `runtime/platforms/interface.py`
  - `runtime/layers/attention/selector.py`
- 已有后端实现：
  - `flash_attn.py`
  - `flash_attn_2.py`
  - `sage_attn.py`
  - `sage_attn3.py`
  - `sdpa.py`
- `ServerArgs.attention_backend`

### 当前主要问题

`CogVideoX / VividVR` 代码当前仍主要基于 `diffusers` 的 `CogVideoXBlock`。

因此 `Phase E` 这里首先要确认两件事：

1. 当前 `attention_backend` 传参是否真的影响了 `VividVR` 主 denoise attention 计算。
2. 如果没有，应该用最小修改把 `CogVideoX` block 内 attention 调用绑到 `sglang` attention 抽象，而不是重写整条 pipeline。

### 实施建议

建议按下面顺序推进：

1. 先做“真实生效性”审计。
2. 如果 `CogVideoXBlock` 当前没有吃到 `sglang` backend，优先做局部 attention adapter。
3. 在单卡上先验证：
   - `fa`
   - `sage_attn`
   - `torch_sdpa`
4. 默认后端只在质量和稳定性都通过后再收口。

优先级建议：

- 第一优先：`fa`
- 第二优先：`sage_attn`
- 第三优先：`sage_attn_3`
- 保底回退：`torch_sdpa`

需要特别注意：

- 若后续多卡使用 `ring_degree > 1`，`ServerArgs` 已要求 backend 必须是 `fa` 或 `sage_attn`。
- backend 选择必须同时覆盖 transformer 和 controlnet 路径，不能只优化其中一支。

建议重点改动位置：

- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox.py`
- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr.py`
- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr_controlnet.py`
- 如需新增 adapter，再放到 `runtime/layers/attention/` 或相邻模型私有目录

阶段验收：

- `step=20` 长视频 compare 继续 `pass_compare=true`
- 质量 summary 与 `phase_d_130f_20step_metrics_seed42_20260605T083022Z.json` 保持接近
- report 中明确记录实际 backend
- 若 backend 无法稳定生效或引入 NaN / 质量回退，必须可回退到当前稳态路径


## 5.2 工作线 B：torch.compile 接入

### 目标

让 `torch.compile` 在 `VividVR` 路径上真正生效，并尽量复用 `sglang` 已有 compile 约定与环境变量控制方式。

### 直接复用的现有工程

- `ServerArgs.enable_torch_compile`
- 通用 compile 约定：
  - `SGLANG_TORCH_COMPILE_MODE`
- 现有 compile 参考实现：
  - `runtime/pipelines_core/stages/denoising.py`

### 当前主要问题

`VividVR` 当前已经在 pipeline 初始化阶段对 `transformer / controlnet` 尝试 `torch.compile`，并且该路径已经形成正式 `E2` 成果。

因此这里的重点不再是“从零补 compile 接线”，而是：

- 冻结当前 `E2 = FA + torch.compile` 作为单卡正式最佳基线
- 避免后续 `E3` 修改把现有 compile 稳态路径冲掉
- 在时机合适时减少 `VividVR` 与通用 `DenoisingStage` compile/helper 约定的分叉

### 实施建议

- compile 只优先作用于：
  - `transformer`
  - `controlnet`
- 不建议一开始就把整条 pipeline、I/O、视频保存、后处理一起 compile
- 先围绕当前固定 benchmark 形状做 compile：
  - `130f`
  - `2 clips`
  - `20 step`
  - 固定 tiling 参数

要特别处理的问题：

- compile 首次运行存在冷启动开销
- `Phase E` 的计时结论必须区分：
  - cold compile
  - warm steady-state

推荐口径：

- 日常性能验收使用 warm 路径
- 可以先跑一次非正式 warmup，再记录正式 report
- 若保留冷启动报告，需要和 warm 报告分开，不要混淆

建议重点改动位置：

- `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py`
- 如需共用 helper，可抽到 `runtime/pipelines_core/stages/` 公共层
- `python/sglang/multimodal_gen/tools/run_vividvr_inference.py`

阶段验收：

- compile 开启后仍通过 `step=20` compare
- report 中记录 compile 是否开启、compile mode、是否 warmup
- 稳态 `model_inference_runtime_seconds` 相比未 compile 版本有明确收益，或至少没有明显倒退


## 5.3 工作线 C：高热度算子融合

### 目标

对 `VividVR / CogVideoX` 的后续 `E3` 加速，优先复用 `sglang` 仓内已有 runtime helper、attention/backend 基础设施和 fused kernel，而不是继续把精力主要花在 `VividVR` 私有的小型 patch 上。

### 直接复用的现有工程

可优先评估的现成能力包括：

- `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py`
- `python/sglang/jit_kernel/diffusion/qknorm_rope.py`
- `python/sglang/jit_kernel/diffusion/triton/scale_shift.py`
- `python/sglang/jit_kernel/diffusion/triton/norm.py`
- `python/sglang/jit_kernel/diffusion/triton/rmsnorm_onepass.py`

现成参考接法可优先看：

- `python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py`
- `python/sglang/multimodal_gen/runtime/models/dits/wanvideo.py`
- `python/sglang/multimodal_gen/runtime/models/dits/flux_2.py`
- `python/sglang/multimodal_gen/runtime/models/dits/hunyuanvideo.py`

### 优先级重置

在继续推进 `E3` 前，需要先把当前正式口径写死：

- 当前单卡最好正式结果仍是 `E2 = FA + torch.compile`，不是“所有手工 fusion 全开”。
- 当前正式通过的 `E3` 只有 `modulation / residual fusion`。
- 浅层 `QKV fusion` 和 `QK norm + RoPE` 都属于已接线候选路径，不是当前正式 `E3` 默认配置。

因此后续 `E3` 不应继续按“先做 `QK norm + RoPE`，再看别的”推进，而应改成：

1. 先补会影响后续所有加速复用能力的 runtime helper 缺口。
2. 再把 attention/runtime 接线尽量往 `sglang` 原生通路上拉。
3. 最后才做更深的热点融合，并把重点放到真正命中 DiT 主耗时的大投影入口。

### 实施原则

- 先冻结现有正式结论，再在稳定基线上做后续 `E3` 扩展。
- 先补 runtime/helper 前置条件，再决定具体做哪些 fusion。
- 先复用已有 helper / kernel，再考虑新写 kernel。
- 先做局部替换，再考虑是否需要 `CogVideoXBlock` 原生化。
- `QK norm + RoPE` 默认只保留为候选支线，不再作为当前 `E3` 主攻方向。

### 建议优先排查的后续热点

1. `VividVRDenoisingStage` 与通用 `DenoisingStage` 之间可直接复用的 runtime helper
2. attention metadata builder 与 backend 真实生效链路
3. `layernorm / residual / scale_shift / gate` 的已通过融合路径是否还能继续加深
4. `packed-QKV / merged projection / QKV+MLP in-proj` 这类真正命中 DiT 主耗时的大投影入口
5. 只有在前面几项都稳定后，才重新审视 `QK norm + RoPE` 是否还值得为 fast path parity 单独投入

### 当前主要问题

由于 `VividVR` 当前 block 仍是 `diffusers` 风格实现，后续 `E3` 并不会像已有 `sglang` 原生 DiT 那样天然吃满底层加速。

因此这条线的优先策略应是：

1. 先让 runtime/helper 和 attention metadata 这类基础设施尽量收敛。
2. 再判断当前 `CogVideoXBlock` 内部哪些数学结构已经能和现有 fused kernel 或 merged projection 对齐。
3. 用最小局部 adapter 插入这些能力。
4. 只有在“局部接法不通”时，再评估是否需要把 `CogVideoXBlock` 替成 `sglang` 原生等价块。

不建议直接做的事：

- 先把 `VividVR` 改写成通用 `DenoisingStage.forward()` 的完全等价实现
- 先整体重写 `CogVideoX` block
- 先为 `VividVR` 自己写一套完全新的 Triton kernel

建议重点改动位置：

- `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py`
- `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/videoedit_wan.py`
- `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py`
- `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py`
- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox.py`
- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr.py`
- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr_controlnet.py`
- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_attention_backend.py`
- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_operator_fusion.py`
- 必要时新增模型私有 fused adapter

阶段验收：

- 只允许对 profile 证明是热点、且与当前 `E3.1 / E3.2 / E3.3` 顺序一致的路径做修改
- 每次只引入一类主要变化，单独记录收益
- `step=20` compare 继续通过
- 若 fused path 质量回退或平台兼容性差，必须保留原始回退开关


## 5.4 工作线 D：多卡并行推理

### 目标

实现 `VividVR` 的多卡并行加速版本，但优先接入 `sglang` 已有 distributed / model parallel 基础设施，而不是写 `VividVR` 独立多卡调度器。

### 直接复用的现有工程

- `runtime/distributed/parallel_state.py`
- `ServerArgs` 中已有的：
  - `num_gpus`
  - `tp_size`
  - `sp_degree`
  - `ulysses_degree`
  - `ring_degree`
  - `dp_size`
  - `enable_cfg_parallel`

### 当前主要问题

当前 `run_vividvr_inference.py` 把多卡相关参数全部写死为 `1`，而现有 `launch_distributed()` 目标脚本又不是 `VividVR` 入口。

因此多卡这条线至少要补齐：

1. `VividVR` 推理入口的 distributed 参数暴露
2. 多进程启动方式
3. `VividVR` denoise 主链与并行组初始化的真实接线

### 多卡路线优先级

对当前单视频 latency benchmark，推荐优先顺序为：

1. `SP`
2. `TP`
3. 可选 `CFG parallel`
4. `DP` 仅作为吞吐扩展，不是首要目标

原因：

- 当前输入是单视频，`DP` 对单请求 latency 几乎没有直接帮助
- `SP / TP` 更符合“让当前模型吃到底层并行加速”的目标

### 语义约束

多卡实现必须保留 `Phase D` 的长视频主语义，尤其是：

- timestep 级多 clip 同步推进
- overlap latent merge
- trim / stitch 语义

默认不建议把多卡第一版实现成：

- clip0 在 GPU0 独立完整跑完
- clip1 在 GPU1 独立完整跑完
- 最后再粗拼接

这种方式虽然工程上看似简单，但很容易偏离 `Phase D` 已验收的时序主语义，也不符合“尽量复用 `sglang` 底层并行”的方向。

### 实施建议

建议先完成下面这条最保守、最接近现有基础设施的路径：

1. 先做单视频单请求的 `SP` 版本
2. 再评估是否叠加 `TP`
3. 最后再看是否需要引入 `CFG parallel`

对于首个可交付版本，优先尝试：

- `tp_size=1`
- `sp_degree=num_gpus`
- `ulysses_degree=sp_degree`
- `ring_degree=1`

如果后续确实需要 `ring_degree > 1`，则 attention backend 只能使用：

- `fa`
- `sage_attn`

### 启动与入口建议

`VividVR` 多卡入口建议不要继续依赖当前写死脚本路径的 `launch_distributed()`。

更合理的做法是二选一：

- 为 `run_vividvr_inference.py` 补一个 `torchrun` 友好的启动方式
- 或把现有 `launch_distributed()` 泛化成“可传目标脚本路径”的公共工具，再让 `VividVR` 复用

### 建议重点改动位置

- `python/sglang/multimodal_gen/tools/run_vividvr_inference.py`
- `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py`
- `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py`
- 必要时补充分布式入口封装

阶段验收：

- 多卡版本在 `step=20` 上继续 `pass_compare=true`
- 多卡版本至少连续两次运行稳定，不出现死锁、hang、rank 间 shape 不一致
- report 中完整记录并行配置
- 加速倍数继续以 `model_inference_runtime_seconds` 为主字段计算


## 6. 按阶段细化的实施顺序

`Phase E` 不建议把四条加速线并行混做，而应按“冻结基线 -> 单项接入 -> 单项验收 -> 再进入下一项”的顺序推进。

### 6.1 Phase E0：冻结基线与补可观测性

目标：

- 固定 `130f / 20 step` 日常 benchmark 口径
- 让 report/indicator 能完整记录后续所有加速变量
- 形成后续所有单项加速的共同 `control` 基线

本阶段允许改动：

- profiler / timer
- report / indicator 字段
- CLI 中与记录有关、但不改变推理语义的参数透传

本阶段不允许改动：

- attention backend 的真实执行路径
- compile 生效路径
- fused kernel 接法
- 多卡并行行为

本阶段产物：

- 一份可复现的 `Phase E0` 基线 report
- 一份明确记录默认配置的 `control` 指标文件

### 6.2 Phase E1：attention backend 单项接入与单项验收

目标：

- 冻结并保护当前已经形成正式结果的单卡 backend 基线
- 在需要继续演进 backend 路径时，保留清晰的真实性与稳定性验收口径

推进顺序：

1. 默认冻结 `fa` 作为当前单卡正式 backend 基线
2. 若后续 `E3.2` 继续推进 metadata/backend 原生化，先确认当前 `attention_backend` 参数是否真实影响主 denoise attention
3. 若没有生效，再补局部 adapter
4. `sage_attn / torch_sdpa` 继续保留为候选与回退路径，不在当前文档里重新定义默认结论

本阶段验收规则：

- 每轮只切换一个 backend 变量
- 不同时引入 compile、fusion 或多卡
- 单项收益按“上一稳定配置 vs 当前 backend 配置”计算

本阶段完成条件：

- 当前默认 backend 结论保持清晰且可复现
- report 中能准确写出实际 backend

### 6.3 Phase E2：`torch.compile` 单项接入与单项验收

目标：

- 冻结并保护当前已经成立的 `E2 = FA + torch.compile` 单卡最佳正式结果
- 在后续 `E3` 修改中避免 compile 稳态路径被无意冲掉

推进顺序：

1. 保持 compile 只优先作用于 `transformer / controlnet`
2. 明确区分 cold 与 warm
3. 以后续 `E3` 任一修改都不得破坏 warm steady-state 收益为前提
4. 若后续需要收敛 helper，再评估是否把 compile 相关逻辑进一步向通用 `DenoisingStage` 约定靠拢

本阶段验收规则：

- backend 保持为 `Phase E1` 已冻结结果
- 只允许切换 `enable_torch_compile` 与 compile mode
- 不同时引入新的 fusion 或多卡

本阶段完成条件：

- compile 开启后继续 `pass_compare=true`
- `FA + torch.compile` 仍保持为当前单卡最佳正式结果，或至少不被后续修改无意破坏

### 6.4 Phase E3：后续加速的执行清单

在进入后续 `E3` 前，先固定两个事实：

- 当前单卡最好正式结果仍是 `E2 = FA + torch.compile`。
- 当前正式 `E3` 通过项只有 `modulation / residual fusion`，不能把 `QK norm + RoPE` 或浅层 `QKV fusion` 写成现行默认方案。

因此这里的 `Phase E3` 不再按“先做某个 kernel，再堆下一个 kernel”组织，而是拆成三个有依赖顺序的子阶段。

#### 6.4.1 E3.1：runtime helper 收敛

目标：

- 在不改写 `VividVR` 既有 denoise 主语义的前提下，让 `VividVRDenoisingStage` 尽量复用通用 `DenoisingStage` 已经稳定的 runtime helper。

本阶段允许改动：

- `autocast` 接线
- device placement helper 接线
- profiling / report helper 接线
- attention metadata builder 接口接线

本阶段不允许改动：

- `controlnet -> transformer` 两段式主语义
- tile 内循环与 merge 语义
- scheduler 口径
- 新增任何以“速度”为目标的深层 kernel/fusion

执行清单：

1. 审计 `VividVRDenoisingStage` 当前和 `DenoisingStage` helper 的差异面。
2. 优先补 `torch.autocast(...)` 的标准包裹方式。
3. 评估并接入 `_manage_device_placement(...)` 可直接复用的部分。
4. 把 profiling 与 report 记录尽量对齐到 `StageProfiler / SGLDiffusionProfiler` 约定。
5. 为后续 attention metadata builder 预留统一接口，即使第一版仍允许回退到 `None`。
6. 明确哪些 helper 可直接复用，哪些必须保留 `VividVR` 自定义逻辑。

本阶段验收规则：

- 只允许做 runtime/helper 收敛，不允许混入新 backend、新 fusion 或多卡。
- `step=20` compare 继续通过。
- report 能更完整记录 runtime 配置与阶段耗时。

本阶段完成条件：

- `VividVRDenoisingStage` 已能稳定复用一部分 `DenoisingStage` helper。
- 后续 `E3.2` 不再需要从零搭 metadata / profiling / placement 接口。

#### 6.4.2 E3.2：attention/runtime 原生化与低风险补线

目标：

- 让 `VividVR` 的 attention/runtime 接线更接近 `sglang` 原生通路，并补齐若干低风险但真实存在的漏接项。

本阶段允许改动：

- attention metadata builder 真正落到 `VividVR` 主链
- backend 真实生效性审计与局部 adapter
- decode 侧 `vae_tiling` 等低风险接线补齐
- 与 report/indicator 对应的配置记录补齐

本阶段不允许改动：

- 直接把 `VividVR` 替换成通用 `DenoisingStage.forward()` 实现
- 在同一轮里同时引入新的深层 fused kernel
- 把 `QK norm + RoPE` 重新提升为默认主线

执行清单：

1. 把 `attn_metadata=None` 改造成可审计、可回退的 metadata builder 接口。
2. 验证 transformer 和 controlnet 两条支路是否都真实吃到 backend 选择结果。
3. 确认 `fa` 仍是当前单卡默认后端，并把 backend 生效结果写进 report。
4. 补 decode 侧 `vae.enable_tiling()` 或明确记录为什么继续不接。
5. 清理“config/CLI 名义支持”与“真实执行路径已接通”之间的文档和日志偏差。

本阶段验收规则：

- 一次只允许改变一类 runtime/backend 变量。
- 所有新接线都必须保留明确回退路径。
- `step=20` compare 继续通过。

本阶段完成条件：

- `VividVR` 不再长期停留在 `attn_metadata=None` 的状态。
- attention backend 的真实生效性可被日志、report 或 profile 明确证明。
- 低风险漏接项得到收口，不再干扰后续热点融合判断。

#### 6.4.3 E3.3：深层热点融合

目标：

- 在 `E3.1 / E3.2` 已收口的稳定基线上，继续寻找真正能明显缩短 DiT 主耗时的融合路径。

本阶段允许改动：

- 对已通过的 `modulation / residual fusion` 做更深一层的链式扩展
- `packed-QKV / merged projection`
- 必要时评估 `QKV + MLP in-proj` 这类更大投影融合
- 其他经 profile 证明确实命中主热点的 fused adapter

本阶段不允许改动：

- 把浅层 `diffusers fuse_projections()` 直接等同于“深层投影融合”
- 一次打包多个 fusion 作为同一个验收样本
- 在没有 parity 方案前把 `QK norm + RoPE` 重新升格为默认路径

执行清单：

1. 以 `E3.2` 冻结结果为 `control`，重新 profile DiT 主耗时。
2. 把当前已正式通过的 `modulation / residual fusion` 作为保留参考线，不轻易回退。
3. 优先评估 `packed-QKV / merged projection` 是否能在 `CogVideoX` 结构上局部落地。
4. 若结构允许，再评估 `QKV + MLP in-proj` 这类更深的大投影融合。
5. 若继续扩展 `layernorm / residual / scale_shift / gate` 路径，也必须一次只推进一个局部链路。
6. `QK norm + RoPE` 只保留为候选支线；除非 fast path parity 问题被明确定位并修复，否则不进入默认配置。

本阶段验收规则：

- 每类 fusion 都必须有独立 A/B、独立 report、独立回退开关。
- `incremental_speedup_of_feature` 必须只反映本轮新增 fusion 本身。
- `step=20` compare 继续通过。

本阶段完成条件：

- 每类被保留的 fusion 都有独立、可解释的收益结论。
- 至少形成一条比当前正式 `E3` 更有继续推进价值的深层热点融合路线。

### 6.5 Phase E4：多卡并行的分阶段推进

目标：

- 在已冻结的单卡最优配置上，补齐 `VividVR` 的多卡并行推理

推荐子阶段顺序：

1. `Phase E4.1`：只做 `SP`
2. `Phase E4.2`：在 `SP` 稳定后再评估是否叠加 `TP`
3. `Phase E4.3`：如有必要，再评估 `CFG parallel`
4. `DP` 不作为当前单视频 latency 主线目标

本阶段验收规则：

- `SP`、`TP`、`CFG parallel` 都要分别做单项验收
- 例如评估 `TP` 时，应以“已冻结 `SP` 配置”为 `control`，只切换 `TP`
- 不允许把“多卡 + 新 backend + 新 compile/fusion”一起作为一次结果验收

本阶段完成条件：

- 至少有一条稳定的多卡方案通过 `step=20` compare
- 连续两次运行稳定，无死锁、hang、rank 间 shape 不一致
- report 中完整记录并行拓扑与设备配置

### 6.6 Phase E5：最终组合回归与 release 候选收口

目标：

- 在所有单项加速都完成独立验收后，才运行最终组合配置

本阶段允许做的事：

- 启用已经分别验收通过的 backend、compile、fusion、多卡组合
- 做累计收益统计
- 做 `20 step` 日常回归与必要的 `50 step` 最终回归

本阶段不应替代的事：

- 不能用最终组合结果反推某个单项技术的独立收益
- 不能跳过前面单项阶段，直接用组合结果声称某项加速“有效”

本阶段产物：

- 最终推荐默认配置
- 累计 `cumulative_speedup_vs_original`
- 最终回归报告与 release 候选结论

### 6.7 推荐节奏总结

推荐执行顺序如下：

1. `Phase E0`：观测与基线冻结
2. `Phase E1`：attention backend
3. `Phase E2`：`torch.compile`
4. `Phase E3.1`：runtime helper 收敛
5. `Phase E3.2`：attention/runtime 原生化与低风险补线
6. `Phase E3.3`：深层热点融合
7. `Phase E4`：分阶段多卡
8. `Phase E5`：最终组合回归

这样安排的原因是：

- 先把基线和可观测性固定，后续每一项收益才可归因
- attention backend 和 compile 属于最直接复用 `sglang` 底层能力的低侵入路线
- `VividVR` 当前还没有自然站到 `sglang` 原生 denoise runtime 骨架上，因此 `E3` 不能再只理解为“补几个 kernel”
- 先补 helper 与 runtime 接线，再做深层热点融合，后续收益才更可持续、也更容易复用
- 多卡改动面最大，应该放在单卡路径稳定之后
- 只有所有单项结论成立后，组合结果才有解释力


## 7. 后续代码修改时的建议文件关注面

后续实际进入 `Phase E` 代码阶段时，建议优先关注下面这些文件：

- 推理入口与 report：
  - `python/sglang/multimodal_gen/tools/run_vividvr_inference.py`
- `VividVR` 主 denoise 路径：
  - `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py`
- 通用 denoise helper 与 `Wan` 参考：
  - `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py`
  - `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/videoedit_wan.py`
- pipeline 编排：
  - `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py`
- pipeline config：
  - `python/sglang/multimodal_gen/configs/pipeline_configs/vividvr.py`
- `CogVideoX / VividVR` DiT 模型：
  - `python/sglang/multimodal_gen/runtime/models/dits/cogvideox.py`
  - `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr.py`
  - `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr_controlnet.py`
  - `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_attention_backend.py`
  - `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_operator_fusion.py`
- attention backend 与并行基础设施：
  - `python/sglang/multimodal_gen/runtime/layers/attention/*`
  - `python/sglang/multimodal_gen/runtime/distributed/parallel_state.py`
  - `python/sglang/multimodal_gen/runtime/server_args.py`
- fused kernel 复用参考：
  - `python/sglang/jit_kernel/diffusion/*`
  - `python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py`
  - `python/sglang/multimodal_gen/runtime/models/dits/wanvideo.py`
  - `python/sglang/multimodal_gen/runtime/models/dits/flux_2.py`
  - `python/sglang/multimodal_gen/runtime/models/dits/hunyuanvideo.py`


## 8. 本文档对应的最终判断标准

`Phase E` 最终要达成的不是“做过 compile / 做过多卡 / 做过几个 kernel patch”，而是下面三件事同时成立：

1. 当前 `VividVR` 路径已经尽可能直接复用了 `sglang` 现有底层加速工程。
2. `step=20` 长视频 benchmark 在保持 `Phase D` 质量基线附近的前提下，有明确可复现的 `model_inference_runtime_seconds` 改善。
3. 单卡默认配置和多卡加速配置都形成稳定、可回归、可解释的验收口径。

如果某项“加速”做到了速度提升，但不能复现、不能解释、不能回退，或破坏了 `Phase D` 已验收主语义，则不应视为 `Phase E` 合格实现。
