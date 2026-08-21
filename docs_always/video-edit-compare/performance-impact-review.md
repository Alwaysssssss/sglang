# VideoEdit 算法对齐改动的 Debug 与性能影响审查

## 1. 审查范围

- 审查日期：2026-08-21
- 仓库：SGLang `cos` 分支
- 对比基线：`HEAD`（`5e4e5e915`）与当前工作区未提交改动
- 算法参考：`/mnt/shanhai-ai/liuh/VideoEdit-diffusers`
- 对齐决策：[`alignment-decisions.md`](./alignment-decisions.md)

本文只说明当前对齐改动中的调试代码和性能影响，不代表完整性能 Benchmark。除特别注明的现有 perf JSON 外，影响比例需要在同输入、同模型、同硬件和同采样参数下重新做消融测试。

## 2. 结论

当前新增的纯 Debug 代码很少。`strict_videoedit_math` 已在 VideoEdit 模型及其全部 Transformer block 上明确设置为 `False`；现有 case0008、step_47500 的 crop/full golden 均通过，因此严格数学分支不是当前对齐要求，也不构成当前请求的运行时开销。当前主要性能变化来自原生 Hugging Face/Diffusers 组件，以及偏向数值对齐和低显存的默认执行配置。

影响优先级如下：

1. 原生 HF T5/CLIP、Diffusers VAE 绕过 SGLang 优化组件。
2. 全视频处理仍需资源上限；默认 stream、关闭 crop sidecar 和 CLI 硬件自适应已消除
   一部分不必要的上线开销。
3. 当前仍存在一次已经无消费者的 raw `video_tensor` 构造和 H2D。
4. stream 模式执行 backward pass 时，缓存缺失可能退化为反复从头解码。
5. 保留审查项：`strict_videoedit_math` 分支位于 DiT 热路径，若以后重新开启，会影响每个 block 和每个去噪步骤；但当前固定为 `False`，不应再计入当前性能损耗。

## 3. 当前新增的 Debug/诊断代码

### 3.1 条件式参考图保存

`prepare_global_inputs()` 在传入 `debug_dir` 时，除原有的首帧和 mask 外，新增保存：

```text
global_resized_reference.png
```

位置：[`preprocess.py`](../../python/sglang/multimodal_gen/runtime/videoedit/preprocess.py#L597)

当前主 Pipeline 调用 `prepare_global_inputs()` 时没有传入 `debug_dir`，所以线上默认不执行该图片写盘，对当前性能没有影响。

### 3.2 未新增在线 Tensor dump 或强制同步

当前生产改动中没有新增以下行为：

- `torch.save` / `np.save` 中间 Tensor dump；
- 每步 latent/noise prediction 写盘；
- `torch.cuda.synchronize()`；
- 无条件开启的 CUDA profiler。

仓库 `outputs/videoedit-debug-boundary/` 下虽然存在历史对齐 Tensor 和视频产物，但当前生产路径没有继续生成这些文件的 dump 调用。

### 3.3 原有诊断能力

以下能力在本次改动前已经存在，不应计为本轮新增性能开销：

- `perf_dump_path` 和 Pipeline profiler；
- progress JSON；
- `.videoedit.json` 窗口元数据；
- 每阶段日志。

需要注意，当前原生 CLIP 加载后把 `memory_usages["image_encoder"]` 记录为 `0.0`，因此组件显存/模型体积统计会低估 CLIP：[`wan_videoedit_pipeline.py`](../../python/sglang/multimodal_gen/runtime/pipelines/wan_videoedit_pipeline.py#L200)。

## 4. 会影响性能的改动

### 4.1 已关闭但保留对照：严格 DiT 数学路径

当前 VideoEdit 模型初始化时，为模型和全部 Transformer block 明确设置：

```python
self.strict_videoedit_math = False
block.strict_videoedit_math = False
```

设置位置：[`wan_videoedit.py`](../../python/sglang/multimodal_gen/runtime/models/dits/wan_videoedit.py#L84)

因此 [`wanvideo.py`](../../python/sglang/multimodal_gen/runtime/models/dits/wanvideo.py#L601) 中的 strict 分支当前不执行。非 strict 分支继续使用常规/融合 Norm 与 Residual kernel，并在 CUDA 且 Q/K 形状相同时使用 FlashInfer in-place RoPE。

保留的 strict 分支包含：

- 显式 FP32 Norm、Scale、Shift；
- native RMSNorm；
- eager pairwise RoPE；
- 显式 FP32 residual 和 FFN MulAdd。

实现位置：[`wanvideo.py`](../../python/sglang/multimodal_gen/runtime/models/dits/wanvideo.py#L66)、[`wanvideo.py`](../../python/sglang/multimodal_gen/runtime/models/dits/wanvideo.py#L601)

如果重新开启，其性能机制为：

- 绕过融合 Norm/Residual/ScaleShift kernel；
- eager RoPE 绕过 FlashInfer in-place RoPE；
- 产生更多独立 kernel launch；
- FP32 cast 和临时 Tensor 增加显存带宽与峰值显存；
- 对每个 DiT block、每次 cond/uncond forward、每个去噪步骤重复发生。

仓库历史两份 48 帧单窗口记录显示：

| 指标 | 非 strict | strict | 变化 |
|---|---:|---:|---:|
| Denoising | 59.59 s | 62.11 s | +4.2% |
| Peak allocated | 12,384.56 MB | 12,899.80 MB | +515.24 MB |
| Peak reserved | 27,584 MB | 28,254 MB | +670 MB |

记录：

- [`case0008_sglang_debug_perf.json`](../../outputs/videoedit-debug-boundary/sglang/case0008_sglang_debug_perf.json)
- [`case0008_sglang_strict_math_perf.json`](../../outputs/videoedit-debug-boundary/sglang-strict/case0008_sglang_strict_math_perf.json)

这两份 perf JSON 没有完整记录所有运行参数，因此只能作为“重新开启 strict 可能变慢并增加峰值显存”的方向参考，不能当作正式性能结论。两次总时延还受到 Text/CLIP/VAE 和输出阶段波动影响。

当前 `strict_videoedit_math=false` 的对齐结果如下：

| 对比 | 帧数 | SSIM mean / min | MSE mean | 结论 |
|---|---:|---:|---:|---|
| SGLang crop vs 原算法 crop | 48 | 0.993330 / 0.991836 | 0.986224 | 通过 |
| SGLang full vs 原算法 full | 48 | 0.989090 / 0.988156 | 2.872374 | 通过 |
| 双卡 crop vs 单卡 crop | 48 | 1.000000 / 1.000000 | 0.000000 | 逐像素一致 |

记录：

- [`case0008_step47500_compare_report_crop.strict_false.json`](../../outputs/case0008_step47500_compare_report_crop.strict_false.json)
- [`case0008_step47500_compare_report_full.strict_false.json`](../../outputs/case0008_step47500_compare_report_full.strict_false.json)
- [`case0008_step47500_compare_report_crop.strict_false_dual_vs_single.json`](../../outputs/case0008_step47500_compare_report_crop.strict_false_dual_vs_single.json)

这里的“与原算法对齐”指通过当前视频级 golden 阈值，不代表所有中间 Tensor 都逐值相等。双卡与单卡的 crop 输出则在这次记录中确实为 `MSE=0`、`max_abs_diff=0`。

另外，VideoEdit 专用 cross-attention 当前仍无条件调用 `_videoedit_rms_norm()`，不受该开关控制：[`wan_videoedit.py`](../../python/sglang/multimodal_gen/runtime/models/dits/wan_videoedit.py#L36)。所以把 strict 设为 `False` 只关闭通用 Transformer block 内的 strict 分支，并不等于撤销所有为 VideoEdit 增加的显式数学实现。

### 4.2 高影响：切换为原生 HF/Diffusers 组件

当前 VideoEdit 专门加载：

- `UMT5EncoderModel`；
- `CLIPVisionModel`；
- `AutoencoderKLWan`。

位置：[`wan_videoedit_pipeline.py`](../../python/sglang/multimodal_gen/runtime/pipelines/wan_videoedit_pipeline.py#L72)

这些改动用于匹配参考实现的 conditioning 和 VAE 数值边界，但绕过了 SGLang `PipelineComponentLoader` 的优化组件选择。影响包括：

- T5/CLIP/VAE kernel 和内存管理可能慢于 SGLang 优化实现；
- CLIP 参数保持 FP32，增加模型权重内存和传输量；
- 开启 CPU offload 时，每个窗口都可能发生模型 H2D/D2H；
- Prompt 没有变化，但 Text Encoder 仍在每个窗口重复执行；
- 49 帧窗口增加时，固定编码和 offload 开销会被重复更多次。

当前书面对齐决策写明“当前不修改 VAE”，并把 T5 主路径列为已对齐；如果原生 VAE/T5 是后续 golden 调试确认的必要改动，应更新 `alignment-decisions.md` 并补充数值证据。

### 4.3 已收敛：默认 stream、全视频和关闭额外 crop 输出

当前默认值：

```text
num_frames = None / -1
decode_mode = stream
save_crop_only = false
```

位置：[`videoedit_wan.py`](../../python/sglang/multimodal_gen/configs/sample/videoedit_wan.py#L67)、[`protocol.py`](../../python/sglang/multimodal_gen/runtime/entrypoints/openai/protocol.py#L200)

影响：

- stream 模式不再常驻完整输入帧集合，并可用预取线程重叠后续解码和当前 GPU 推理；
- 最终输出帧仍会集中保存，因此 stream 降低的是输入侧峰值，不是恒定主机内存；
- 默认处理完整视频，使未显式指定 `num_frames` 的 CLI/直调任务工作量仍可能大幅增加；
- 对齐命令显式设置 `save_crop_only=true` 时，仍会在正式输出外额外 resize 并调用一次 FFmpeg，增加 CPU、I/O、磁盘空间和请求尾延迟。

crop sidecar 路径：[`wan_videoedit_pipeline.py`](../../python/sglang/multimodal_gen/runtime/pipelines/wan_videoedit_pipeline.py#L592)

对于短视频或任意参考帧触发的 backward pass，显式 eager 可能避免 stream 的重复解码，
因此仍保留为回退模式。

### 4.4 已收敛：CLI 执行参数恢复硬件自适应

VideoEdit CLI 不再覆盖以下执行参数：

```text
attention_backend = None
dit_layerwise_offload = None
dit_offload_prefetch_size = 0
text_encoder_cpu_offload = None
image_encoder_cpu_offload = None
vae_cpu_offload = None
pin_cpu_memory = None
sp_degree / ulysses_degree / ring_degree = None
```

位置：[`cli.py`](../../python/sglang/multimodal_gen/runtime/videoedit/cli.py#L101)

影响：

- `None` 把 Attention、offload 和并行度选择交回 `ServerArgs` 的硬件与模型自适应逻辑；
- 高显存机器不再被 CLI 强制逐层 offload，设置多卡时也不会被固定为 `sp_degree=1`；
- 数值对齐命令继续显式设置 `torch_sdpa`、逐层 offload 和各编码器 offload；
- `dit_offload_prefetch_size=0` 仅在最终启用逐层 offload 时生效。

生产双服务启动脚本原本就显式开启逐层 offload，因此服务中的全部 offload 时延不能只归因于本轮 CLI 默认值变化：[`start.sh`](../../scripts/videoedit_dual_service/start.sh#L128)。

### 4.5 确定的无效开销：未使用的 raw `video_tensor`

当前已经删除 raw-video 的第二次 VAE encode，只保留 masked-video condition encode。这是正向优化。

但是 `prepare_window_inputs()` 仍执行：

```python
"video_tensor": frames_to_tensor(window_video, normalize=True).to(
    device=device,
    dtype=dtype,
)
```

位置：[`preprocess.py`](../../python/sglang/multimodal_gen/runtime/videoedit/preprocess.py#L825)

该返回值当前没有消费者，却每窗口执行一次：

- PIL/NumPy/Torch 转换；
- 完整窗口 Tensor 分配；
- CPU 到 GPU 传输；
- 瞬时 GPU 显存占用。

这是最明确、风险最低的性能清理候选。

### 4.6 条件性严重问题：stream backward 重复解码

`WindowFrameProvider` 的缓存缺失处理会重新打开视频和 mask 解码器，并从第 0 帧顺序解码到目标帧：[`frame_provider.py`](../../python/sglang/multimodal_gen/runtime/videoedit/frame_provider.py#L217)。

任意参考帧的 backward pass 按递减 global index 访问帧；一旦目标帧被 LRU cache 淘汰，窗口中的多个帧可能各自触发一次从头解码。长视频下 I/O 复杂度可能接近 `O(N²)`。

该问题只影响：

```text
decode_mode=stream
并且 long 或 short pass 的方向为 backward
并且目标帧已离开缓存
```

默认 `ref_frame_idx=0` 的单向 forward 场景通常不会触发，但新支持的任意参考帧会扩大触发范围。

### 4.7 窗口与预处理默认值的混合影响

#### `infer_len: 81 -> 49`

- 正向：降低单窗口空间×时间 token、单次 attention 成本和峰值显存。
- 负向：长视频窗口数量增加，T5/CLIP/VAE、offload、decode 和输出转换重复更多次。
- 总时延方向与视频长度、Attention backend、offload 和 overlap 有关，不能只凭窗口数量判断。

#### `dilate_px/feather_px: 0 -> 8`

- mask dilation 可能扩大 union bbox 和对齐后的 crop，增加 VAE/DiT 空间 token；
- feather 增加 paste-back 阶段的 CPU 图像处理；
- 对已经触发最小 480 短边扩张的 case，dilation 未必继续扩大最终 crop。

#### CPU float32 noise

每窗口使用 CPU float32 generator，并把 noise 复制到 GPU：[`videoedit_wan.py`](../../python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/videoedit_wan.py#L462)。

这会增加一次较小的 H2D，但属于随机数逐值对齐所需语义。

## 5. 正向性能改动

本轮并非全部是性能回退，已有以下正向改动：

1. 删除 raw-video 的第二次 VAE encode，减少每窗口一次 VAE 编码和对应 latent。
2. 删除 weighted overlap 的全视频 float32 accumulator，降低主机内存和 CPU blending。
3. 49 帧窗口通常降低单次 forward 峰值显存。
4. native skip 直接提交非 overlap 帧，减少 weighted blending。
5. stream forward pass 仍保留预取能力，可在适用场景重叠 I/O 与 GPU 推理。

## 6. 建议的模式划分

建议把当前配置明确拆成对齐基线、strict 诊断和性能服务三种用途，避免把诊断分支误当成 golden 的必要条件。

### 6.1 当前 Golden 对齐基线

- 原生 HF/Diffusers 组件；
- `strict_videoedit_math=false`；
- CPU float32 noise；
- `torch_sdpa`；
- TeaCache/Cache-DiT 关闭；
- stream；
- crop sidecar 显式开启；
- 固定输入、seed、checkpoint 和 VAE tiling。

这是当前复跑命令使用的算法 golden 配置。执行参数仍偏向数值稳定性和低显存，
因此通过命令显式设置，不作为通用 CLI 默认值。

### 6.2 Strict 诊断模式

- 仅在定位 Norm、RoPE 或 Residual 数值边界时临时设置 `strict_videoedit_math=true`；
- 使用相同输入、seed、checkpoint、Attention backend 和 offload 配置，与 `False` 基线做单变量消融；
- 同时保存中间 Tensor 边界或至少保存 crop/full 比较报告和完整 perf 参数。

该模式是数值诊断工具，不是当前对齐要求，也不应作为生产默认值。

### 6.3 Performance 模式

- 优先使用 SGLang 优化组件和融合 kernel；
- FlashAttention/高性能 Attention backend；
- 根据显存选择常驻或提高 offload prefetch；
- 长视频默认 stream；
- crop sidecar 默认关闭；
- TeaCache/Cache-DiT 按质量回归结果启用；
- 允许数值误差，但必须通过 crop-only 质量阈值。

首先建议处理两项：

1. 删除未使用的 raw `video_tensor` 构造；
2. 保持 `strict_videoedit_math=false`，并将原生组件和其余对齐专用执行参数收敛到显式模式开关；若要重新启用 strict，必须作为单独诊断实验记录。

## 7. 规范与规格审查

### 规范

未发现违反根 `AGENTS.md` 的硬性问题。

判断项：eager 与 stream 的 pass/window 物化逻辑分散在 `preprocess.py` 和 `frame_provider.py`，存在重复与后续行为漂移风险。

### 规格

当前代码强制使用原生 Diffusers VAE 和 HF T5，而书面对齐决策仍写明“不修改 VAE”、T5 主路径已对齐。这属于未同步到规格文档的范围扩张；需要补充决定、数值边界证据和性能消融。
