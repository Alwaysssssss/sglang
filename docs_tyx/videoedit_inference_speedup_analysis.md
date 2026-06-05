# VideoEdit 推理加速现状与后续方向

日期：2026-06-05

本文基于当前仓库代码和已有 `docs_tyx/`、`output_results/` 性能记录整理。目标是回答两个问题：

1. 当前 SGLang VideoEdit 推理链路已经做了哪些加速。
2. 在不直接牺牲生成质量的前提下，后续还有哪些值得做的推理加速。

相关主文件：

- `python/sglang/multimodal_gen/runtime/pipelines/wan_videoedit_pipeline.py`
- `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/videoedit_wan.py`
- `python/sglang/multimodal_gen/configs/sample/videoedit_wan.py`
- `python/sglang/multimodal_gen/runtime/models/dits/wan_videoedit.py`
- `python/sglang/multimodal_gen/runtime/models/dits/wanvideo.py`
- `python/sglang/multimodal_gen/runtime/cache/cache_dit_integration.py`
- `python/sglang/multimodal_gen/runtime/videoedit/`

## 1. 当前推理链路

VideoEdit 原生 pipeline 是固定 81 帧窗口的 Wan VideoEdit DiT。长视频通过窗口切分跑多次模型，再把窗口结果 commit 回全局输出。

主流程：

1. API/CLI 构造 `WanVideoEditSamplingParams`。
2. `WanVideoEditPipeline.forward()` 准备全局上下文：视频/mask、全局 bbox、窗口规格、输出累积 buffer。
3. 每个窗口依次执行：
   - `VideoEditWindowValidationStage`
   - `VideoEditTextEncodingStage`
   - `VideoEditConditionEncodingStage`
   - `VideoEditLatentPreparationStage`
   - `VideoEditTimestepPreparationStage`
   - `VideoEditLatentInitStage`
   - `VideoEditDenoisingStage`
   - `VideoEditDecodingStage`
   - `VideoEditWindowPostprocessStage`
4. `WanVideoEditPipeline._commit_window_output()` 把窗口输出合并回全局 crop buffer。
5. `_finalize_long_video_output()` 做 paste-back 或 crop-only 输出。

实际瓶颈非常集中在 `VideoEditDenoisingStage`。已有 156 帧双窗口 perf 中，最后一个窗口的 stage 约为：

| 配置 | total | text | condition VAE encode | denoise | decode |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sp1_no_offload_fa_156f_all_gpu0` | 683.67s | 0.66s | 5.55s | 323.37s | 4.81s |
| `sp1_no_offload_fa_156f_all_gpu0_stream` | 698.90s | 0.64s | 5.38s | 323.85s | 4.62s |
| `phase23_eager` | 685.31s | 0.64s | 5.47s | 323.38s | 4.69s |
| `phase23_stream` | 690.50s | 0.64s | 5.38s | 323.21s | 4.78s |

注意：这些 perf JSON 的 `steps` 只保留最后一个窗口的 stage 明细，`total_duration_ms` 是全请求耗时。156 帧默认切成 2 个 81 帧窗口，因此全局 denoise 近似是最后窗口 denoise 的 2 倍。

## 2. 已经做了的加速

### 2.1 全局 bbox 裁剪，只修 mask 区域

代码位置：

- `runtime/videoedit/preprocess.py`
- `WanVideoEditPipeline._prepare_global_videoedit_context()`

当前不会把 1920x1080 全帧直接送进 DiT。流程会先根据 mask 扫全局 bbox，必要时对小区域扩到最小尺寸，再按 16 对齐得到 `runtime_aligned_h/runtime_aligned_w`。后续 VAE encode、DiT denoise、VAE decode 都跑 crop 后的局部区域。

这是最重要的结构性加速之一。DiT token 数和空间分辨率成正比，bbox 裁剪直接减少 VAE 和 DiT 的输入规模。

### 2.2 81 帧窗口化，支持长视频分段推理

代码位置：

- `runtime/videoedit/windowing.py`
- `WanVideoEditPipeline.forward()`

VideoEdit 模型固定 `infer_len=81`。当前通过 `build_videoedit_window_specs()` 把长视频切成多个 81 帧窗口；最后不足 81 帧时用时间维度反射补帧，补帧只参与模型输入，不写入最终输出。

这个能力本身不是单窗口加速，但它避免为了长视频改模型输入形状，也让显存峰值保持在单窗口级别。

### 2.3 默认 stream 解码和窗口预取

代码位置：

- `WanVideoEditSamplingParams.decode_mode = "stream"`
- `runtime/videoedit/frame_provider.py`
- `runtime/videoedit/stream_decoder.py`

当前默认 `decode_mode="stream"`。stream 模式先扫 mask 得到全局几何信息，然后用 `WindowFrameProvider` 逐窗口顺序解码、crop、resize，并启动 `videoedit-frame-prefetch` 线程做预取。

收益：

- 长视频不需要一次性把所有原始帧和 mask 全部留在内存里。
- 下一窗口输入可以在当前窗口推理期间提前解码。
- 对长视频和高分辨率视频更稳。

已有 perf 显示，`stream` 对当前 156 帧样例的总耗时没有明显下降，说明它主要解决内存和 IO 组织，不是 denoise 级别的核心加速。

### 2.4 bf16/autocast

代码位置：

- `VideoEditDenoisingStage.forward()`
- `WanVideoEditPipelineConfig`

默认精度是 bf16：

- `precision = "bf16"`
- `dit_precision = "bf16"`
- `vae_precision = "bf16"`
- `WanVideoEditSamplingParams.dtype = "bf16"`

`VideoEditDenoisingStage` 会根据 DiT dtype 开 autocast。相比 fp32，bf16 能明显降低显存和提升 tensor core 吞吐。

### 2.5 Dynamic CFG 跳过后 5 步 negative pass

代码位置：

- `calc_current_cfg()`
- `VideoEditDenoisingStage.forward()`

默认：

- `num_inference_steps = 20`
- `guidance_scale = 5.0`
- `dynamic_cfg = True`
- `dynamic_cfg_max_step = 15`
- `dynamic_cfg_min = 1.0`

前 15 步执行 conditional + unconditional 两次 DiT forward；后 5 步 `current_cfg` 回到 1.0，不再执行 negative prompt 分支。

因此单窗口 DiT forward 数量约为：

```text
15 * 2 + 5 * 1 = 35
```

如果静态 CFG 全程双分支，则是：

```text
20 * 2 = 40
```

这个默认行为已经省掉约 12.5% 的 DiT forward。

### 2.6 FlashAttention / attention backend 接入

代码位置：

- `runtime/server_args.py`
- `runtime/layers/attention/`
- `runtime/models/dits/wan_videoedit.py`

VideoEdit 的 DiT 继承 Wan attention 实现，支持通过 `--attention-backend` 选择后端。已有运行命令中使用过 `--attention-backend fa`。

需要注意：

- A100 上 `fa` 是优先候选。
- `sage_attn` 可以单独安装后实测。
- `sage_attn_3` 更偏 Blackwell，A100 上如果日志 fallback，就不能把结果记为 SageAttention3 成绩。
- `WanVideoEditCrossAttention` 会过滤 sparse backend，只保留非 sparse 后端，避免 VideoEdit cross-attn 走不兼容稀疏注意力。

### 2.7 多卡 SP/Ulysses 已可用于 VideoEdit

代码位置：

- `runtime/server_args.py`
- `runtime/models/dits/wanvideo.py`

Wan DiT forward 内支持 sequence shard：

- `forward_batch.enable_sequence_shard`
- `self.sp_size > 1`
- 按 token 维度切分 hidden states。
- block 后通过 `sequence_model_parallel_all_gather()` 收回完整序列。

已有记录中，100 帧双卡 SP2 no-offload + FA 对比单卡 offload 明显加速：

| 配置 | 输入帧 | total | denoise |
| --- | ---: | ---: | ---: |
| 单卡 SP1 + offload | 100 | 1113.29s | 1019.70s |
| 双卡 SP2 + no-offload + FA | 100 | 345.79s | 302.34s |

这是当前最明确的性能收益之一。主要原因是：

- SP2 降低单卡 token 负载。
- no-offload 避免 CPU/GPU 迁移。
- FA 降低 attention 开销。

### 2.8 offload / layerwise offload 保证低显存可跑

代码位置：

- `runtime/server_args.py`
- `runtime/utils/layerwise_offload.py`
- `VideoEditDenoisingStage.forward()`

offload 分支不是性能主线，但对显存受限机器很重要。已有 156 帧全帧单卡 offload 成功记录：

| 配置 | 输入帧 | total | 备注 |
| --- | ---: | ---: | --- |
| 单卡 GPU0 + CPU/layerwise offload | 156 | 627.77s | 全帧成功 |

offload 会降低 OOM 风险，但一般会拖慢 denoise。建议把它作为显存兜底方案，而不是最快配置。

### 2.9 VAE tiling 默认开启

代码位置：

- `WanVideoEditPipelineConfig.vae_tiling = True`
- `VideoEditDecodingStage.forward()`

decode 阶段如果 `vae_tiling` 开启，会调用 `self.vae.enable_tiling()`。这主要降低 VAE decode 显存峰值。已有失败记录显示，双卡 no-offload 156 帧曾在 `VideoEditDecodingStage` OOM，说明 VAE decode 显存仍然是全帧/多卡 no-offload 的稳定性瓶颈之一。

### 2.10 torch.compile 已接入

代码位置：

- `DenoisingStage._maybe_enable_torch_compile()`

启动参数 `--enable-torch-compile true` 时，DenoisingStage 初始化会对 transformer 调用 `module.compile()`。

结论：

- compile 已有通用接入。
- CLI 冷启动不适合作为 compile 收益结论。
- serve 常驻模式需要同 shape 多请求，第一轮编译/预热不计入稳定性能。
- 当前 Cache-DiT 在 `forward()` 里启用，而 torch.compile 在 stage 初始化启用；如果同时用 Cache-DiT + compile，需要确认 patch/compile 顺序，否则容易测出混淆结果。

### 2.11 TeaCache 已有代码路径，但默认不开

代码位置：

- `WanVideoEditSamplingParams.teacache_params`
- `runtime/cache/teacache.py`
- `runtime/models/dits/wanvideo.py`
- `VideoEditDenoisingStage.forward()`

当前 VideoEdit sampling params 已有 Wan 1.3B TeaCache 参数：

```text
teacache_thresh = 0.08
use_ret_steps = True
start_skipping = 5
end_skipping = 1.0
```

TeaCache 逻辑也已经能识别 `WanVideoEdit` 前缀，并且 VideoEdit denoising 会在正/负 CFG pass 前设置：

```python
batch.do_classifier_free_guidance
batch.is_cfg_negative
```

Wan forward 中如果 TeaCache 命中，会跳过 transformer blocks，只复用 residual，然后继续跑 output norm/proj/unpatchify。

但它是请求级开关，默认 `enable_teacache=False`。因此当前不能把 TeaCache 写成默认加速收益；必须在请求中显式打开，并做 perf 与逐帧质量对比。

### 2.12 Cache-DiT 已有 VideoEdit adapter，但默认不开

代码位置：

- `DenoisingStage._maybe_enable_cache_dit()`
- `runtime/cache/cache_dit_integration.py`
- `test/unit/test_videoedit_cache_dit.py`

当前 `cache_dit_integration.py` 已经识别 `WanVideoEditTransformer3DModel`，并走显式 Wan adapter：

```text
forward_pattern = Pattern_2
has_separate_cfg = True
```

启用方式是环境变量：

```bash
export SGLANG_CACHE_DIT_ENABLED=true
export SGLANG_CACHE_DIT_FN=1
export SGLANG_CACHE_DIT_BN=1
export SGLANG_CACHE_DIT_WARMUP=4
export SGLANG_CACHE_DIT_RDT=0.12
export SGLANG_CACHE_DIT_MC=2
export SGLANG_CACHE_DIT_SCM_PRESET=medium
```

注意：

- Cache-DiT 默认不开。
- 它和 `dit_layerwise_offload` 有硬冲突，不应同时打开。
- 它和 TeaCache 都会改 DiT block 复用逻辑，首轮不建议同时打开。
- 必须逐帧 compare，不能只看速度。

## 3. 当前主要瓶颈

### 3.1 Denoising 仍占绝对大头

156 帧双窗口样例中，单个窗口 denoise 约 323s；总请求约 683-699s。除 denoise 外，text、VAE encode、decode 单窗口加起来约 10-11s。

因此后续最值得投入的是减少 DiT forward 次数、降低每次 DiT forward 成本、或者把 CFG 双分支并行化。

### 3.2 多窗口重复 text encoding

`WanVideoEditSamplingParams.reset_window_runtime()` 每个窗口会清掉：

```python
runtime_prompt_embeds = None
runtime_negative_prompt_embeds = None
```

所以同一个请求内多个窗口会重复跑同一 prompt 和 negative prompt 的 text encoder。当前单窗口 text 约 0.64s，2 个窗口只浪费约 0.64s；但窗口数变多时会线性增加。

这不是最大瓶颈，但实现简单，属于低风险优化。

### 3.3 每步都重新 cat latent_model_input

`VideoEditDenoisingStage.forward()` 每个 timestep 都执行：

```python
latent_model_input = torch.cat(
    [latents, params.runtime_cond_masks, params.runtime_cond_latents],
    dim=1,
).to(target_dtype)
```

其中 `runtime_cond_masks` 和 `runtime_cond_latents` 在窗口内不变，只有 `latents` 每步变化。当前每个窗口约 35 次 DiT forward，也就有约 35 次大 tensor cat 和 dtype 转换。

相对 transformer blocks，这不是最大计算量，但会造成额外显存分配、内存带宽和 allocator 压力。

### 3.4 Condition VAE encode 每窗口跑两次

`VideoEditConditionEncodingStage` 分别对 masked video 和 raw video 调用 VAE encode：

```text
masked_video -> runtime_cond_latents
raw_video    -> runtime_video_latents
```

单窗口 condition encoding 约 5.4-5.6s。相比 denoise 不大，但它是明确可优化区域。

### 3.5 VAE decode 不是最大耗时，但会触发 OOM

decode 单窗口约 4.6-4.9s，但已有全帧 SP2 no-offload 失败记录显示 OOM 出现在 `VideoEditDecodingStage`。因此 decode 更像稳定性瓶颈：不一定最慢，但会限制最快配置能否跑通。

### 3.6 stream paste-back 会重新解码原视频和 mask

`WindowFrameProvider.paste_back_frames()` 会重新打开 video/mask decoder，并从头顺序读一遍用于 paste-back。当前样例里这不是主耗时，但长视频、远程盘、高码率输入下会变成明显 IO 成本。

## 4. 后续可做加速方向

### 4.1 高优先级：CFG parallel

当前状态：

```python
if server_args.enable_cfg_parallel:
    raise NotImplementedError("VideoEdit MVP does not support CFG parallel yet")
```

当前 dynamic CFG 下，单窗口约 35 次 DiT forward。如果实现 CFG parallel，用两组 rank 分别跑 conditional/unconditional，前 15 步可把串行双 pass 变成并行双 pass，理论 denoise wall time 从 35 个 forward 降到接近 20 个 forward。

粗略理论上限：

```text
35 / 20 = 1.75x denoise speedup
```

实现要点：

- conditional rank 和 unconditional rank 使用相同 latents、timestep、condition latent。
- unconditional rank 只跑 negative prompt embeds。
- 每步结束后需要同步 `noise_pred` 或 CFG 后结果。
- dynamic CFG 后 5 步没有 negative pass，需要处理 rank 空转或切换策略。
- TeaCache/Cache-DiT 需要分别验证 CFG branch state。

这是最值得做的 DiT 级加速之一，但需要多卡资源。

### 4.2 高优先级：让双卡 SP2 no-offload 全帧稳定

已有 100 帧 SP2 no-offload + FA 明显快于单卡 offload，但 156 帧曾在 decode OOM。建议优先解决全帧稳定性，因为这是不改变模型计算逻辑的纯系统加速。

可做项：

- 确认 GPU 残留显存清理后重新跑 156 帧 SP2 no-offload + FA。
- 开启或加强 VAE decode chunk/tiling。
- 接入 `vae_sp` / parallel decode，降低单卡 decode 峰值。
- decode 完立刻释放中间 tensor，避免窗口间残留。
- final output 不再把所有帧转成一个大 tensor 后交给通用保存逻辑。

如果 156 帧 SP2 no-offload 跑通，预期会比单卡 offload/no-offload 都更有价值。

### 4.3 高优先级：TeaCache 单独标定

当前代码路径已经接上，但默认不开。建议独立评估：

```json
{
  "enable_teacache": true
}
```

评估方法：

1. 固定输入、prompt、seed、steps、dynamic CFG。
2. baseline 不开 TeaCache。
3. 只新增 `enable_teacache=true`。
4. 记录 denoise 时间、总耗时、逐帧 SSIM/MSE/MAE/PSNR。
5. 对 VideoEdit 场景重点检查 mask 边缘、身份一致性、纹理闪烁。

建议先测阈值：

```text
0.05 / 0.08 / 0.12
```

`0.08` 是当前 Wan 1.3B 默认，不一定是 VideoEdit 最优。VideoEdit 是编辑任务，比纯文生视频更容易在局部边缘暴露 cache 误差。

### 4.4 高优先级：Cache-DiT 标定

Cache-DiT 已有 VideoEdit adapter。建议在最佳非 cache 配置上单独打开，先从保守参数开始：

```bash
SGLANG_CACHE_DIT_ENABLED=true
SGLANG_CACHE_DIT_FN=1
SGLANG_CACHE_DIT_BN=1
SGLANG_CACHE_DIT_WARMUP=4
SGLANG_CACHE_DIT_RDT=0.12
SGLANG_CACHE_DIT_MC=2
SGLANG_CACHE_DIT_SCM_PRESET=medium
```

调参顺序：

1. `RDT=0.10/0.12/0.18/0.24`
2. `MC=2/3`
3. `WARMUP=3/4`
4. `SCM_PRESET=medium/fast`

注意事项：

- 不要和 `dit_layerwise_offload` 同时开。
- 首轮不要和 TeaCache 同时开。
- 首轮建议先关闭 torch.compile，确认 Cache-DiT patch 正常后再测 compile 组合。
- 每轮必须输出 compare JSON。

### 4.5 中优先级：prompt embedding 跨窗口缓存

当前同一请求内每个窗口重复 text encoding。可以把 prompt embeds 提升到全局 runtime：

- `runtime_global_prompt_embeds`
- `runtime_global_negative_prompt_embeds`

缓存 key 至少包括：

- prompt
- negative prompt
- tokenizer/text encoder 配置
- dtype
- device
- 是否 CFG

预期收益：

- 2 窗口样例只省约 0.6s。
- 长视频窗口数多时线性收益。
- 实现风险低。

这是低成本小收益优化，适合作为代码洁净度优化顺手做。

### 4.6 中优先级：Condition VAE encode 合批

当前 masked video 和 raw video 分两次 VAE encode。可以尝试在显存允许时合成 batch=2 一次 encode：

```text
[masked_video, raw_video] -> VAE encode -> split
```

可能收益：

- 减少一次 VAE encode 调度和部分 kernel overhead。
- 提高 VAE encode batch 利用率。

风险：

- 峰值显存上升。
- Wan VAE 对 batch 维支持需要实测。
- offload/tiling/parallel encode 组合需要单独验证。

如果显存不足，可以保留双 encode 作为 fallback。

### 4.7 中优先级：避免每步重复 cat condition latent

可以考虑两种做法：

1. 在 denoise loop 前预分配 `latent_model_input`，每步只更新前 16 个 latent channels。
2. 改 transformer 输入接口，传入 `latents + condition`，在 patch embedding 前做更低开销组合。

第一种改动更小：

```python
latent_model_input = torch.empty((..., 36, ...), device=device, dtype=target_dtype)
latent_model_input[:, 16:20] = cond_masks
latent_model_input[:, 20:36] = cond_latents
for step:
    latent_model_input[:, :16].copy_(latents.to(target_dtype))
```

需要验证：

- 不破坏 autocast。
- 不引入额外同步。
- scheduler 输出 dtype/device 一致。
- cache/compile 下 shape 稳定。

预期收益不会像 CFG parallel 那么大，但能降低 allocator 压力，对 compile/cudagraph 也更友好。

### 4.8 中优先级：VAE decode chunk/parallel decode

当前 decode 不是主要耗时，但会限制最快配置。建议目标从“更快”改成“让 no-offload/SP2 不 OOM”。

可做：

- 确认 `vae_sp` 到 Wan VAE parallel decode 的链路是否完整。
- 支持时间维 chunk decode，并保证 temporal VAE 边界正确。
- decode 后及时释放 latent 和 decoded tensor。
- 对最终输出走 streaming writer，避免一次性堆所有 PIL frame 和大 tensor。

验收标准：

- 156 帧 SP2 no-offload + FA 不再在 decode OOM。
- decode 画面和 baseline 对齐。

### 4.9 中优先级：output streaming writer

当前 `_finalize_long_video_output()` 返回 list of PIL frames，随后又转成视频 tensor：

```python
batch.output = _pil_frames_to_video_tensor(output_frames)
```

对长视频来说，这会制造一份额外大内存副本。可以做 VideoEdit 专用输出路径：

- paste-back 后直接写 ffmpeg pipe。
- 或窗口 commit 后最终逐帧写出。
- 避免 `list[PIL] -> np.stack -> torch.Tensor -> 通用保存` 的额外复制。

仓库里已有 `runtime/videoedit/ffmpeg_io.py`，可以复用 `save_video_frames_like_reference()` 思路，但更理想是边生成边写，减少峰值内存。

### 4.10 中优先级：stream paste-back 避免二次完整解码

stream 模式下 paste-back 会重新读原视频和 mask。可以考虑：

- 在第一次窗口预取时保留 paste-back 所需的原始帧/mask ring buffer。
- 或把 paste-back 延迟到每帧 commit 后就写出，避免最后二次遍历。
- 对本地短视频收益不大，对长视频/远程盘收益更明显。

需要注意，当前全局 bbox 需要先扫 mask，所以完全单 pass 不容易；但可以避免 scan 后的第三次读取。

### 4.11 中优先级：torch.compile 与 Cache-DiT 顺序修正

当前 compile 在 `DenoisingStage.__init__()`，Cache-DiT 在 `forward()` 首次真实请求前 `_maybe_enable_cache_dit()`。如果要稳定测 `compile + Cache-DiT`，建议改成：

1. 初始化或 warmup 前先挂 Cache-DiT adapter。
2. 再 compile 已 patch 的 transformer。
3. serve 第二次同 shape 请求作为正式成绩。

否则可能出现：

- compile 的是未 patch 版本。
- cache patch 后图失效或退化。
- perf 无法解释。

### 4.12 低优先级：progress 写文件降频

当前 denoise 每步会调用 `write_videoedit_progress()`。单步 denoise 很长时开销可忽略，但如果后续 DiT 加速到很快，或 progress path 在慢盘/网络盘，可以改为：

- 每 N 步写一次。
- 或只有窗口开始/结束和关键比例写。

这是低优先级优化。

## 5. 可选但有质量 tradeoff 的加速

这些能加速，但会改变生成分布或质量，不建议混入系统优化主线：

- 降低 `num_inference_steps`。
- 降低 `dynamic_cfg_max_step`。
- 降低输入 crop 分辨率或加大 bbox 限制。
- 减少/关闭 overlap。
- 调低 guidance。
- 更激进 TeaCache/Cache-DiT 阈值。
- 低比特量化。

如果业务能接受质量变化，可以单独开 quality-speed 分支评估，不要和系统加速结果混在同一张表里。

## 6. 量化方向

建议只先量化 `WanVideoEditTransformer3DModel`，不要一开始量化 VAE 和 text encoder。

候选：

- FP8/W8A8。
- weight-only int8。
- 更低比特需要单独白名单敏感层。

需要重点保护或单独验证：

- cross-attention。
- QK norm。
- time embedding / condition embedder。
- output projection。
- mask 边缘和纹理稳定性。

量化验收必须做逐帧 compare，并检查 runtime 日志确认量化权重真正生效。

## 7. 推荐优先级

### 第一优先级

1. 清理显存环境后重跑 `SP2 no-offload + FA + 156f`，确认最快非 cache 配置能否稳定跑全帧。
2. 实现或评估 CFG parallel，这是最有理论收益的 DiT 级加速。
3. 独立标定 TeaCache，确认 `enable_teacache=true` 在 VideoEdit 上是否有稳定收益。
4. 独立标定 Cache-DiT，先保守参数，必须逐帧 compare。

### 第二优先级

1. prompt embedding 跨窗口缓存。
2. condition VAE encode 合批。
3. denoise loop 里预分配 `latent_model_input`，避免每步重复 cat。
4. VAE decode chunk/parallel decode，优先解决 SP2 全帧 decode OOM。

### 第三优先级

1. output streaming writer，减少最终输出内存复制。
2. stream paste-back 避免二次完整解码。
3. progress 写文件降频。
4. 量化专项分支。

## 8. 建议 benchmark 矩阵

固定输入、prompt、seed、steps、CFG，不改变质量相关参数，按下面顺序逐个加项：

| 阶段 | 新增项 | 目标 |
| --- | --- | --- |
| `sp1_offload` | 低显存基线 | 确认可跑 |
| `sp1_no_offload_fa` | 关闭 offload + FA | 单卡性能基线 |
| `sp2_no_offload_fa` | SP2/Ulysses | 多卡系统加速 |
| `sp2_no_offload_fa_compile` | torch.compile | 常驻 serve 第二请求收益 |
| `sp2_no_offload_fa_teacache` | TeaCache | DiT 跳步收益和质量 |
| `sp2_no_offload_fa_cache_dit` | Cache-DiT | block/cache 复用收益和质量 |
| `sp2_no_offload_fa_cfg_parallel` | CFG parallel | CFG 双分支并行收益 |
| `quant_branch` | DiT 量化 | 显存/吞吐专项 |

每个阶段至少记录：

- 总耗时。
- 每窗口 denoise。
- text / condition encode / decode。
- peak allocated/reserved。
- actual attention backend。
- 输出帧数、分辨率。
- compare JSON。

## 9. 结论

当前 VideoEdit 已经具备一批系统级加速和稳定性能力：bbox 裁剪、窗口化、stream 解码、bf16/autocast、dynamic CFG、FlashAttention、多卡 SP、offload、VAE tiling、torch.compile 接口、TeaCache 接口、Cache-DiT adapter。

但从现有 perf 看，核心瓶颈仍然是 DiT denoising。后续最值得做的是：

1. 多卡 no-offload 全帧稳定跑通。
2. CFG parallel。
3. TeaCache / Cache-DiT 的 VideoEdit 专项标定。
4. 减少 denoise loop 内重复构造输入和窗口间重复 text/VAE 工作。

其中 CFG parallel、TeaCache、Cache-DiT 是最可能带来显著收益的方向；prompt cache、VAE encode 合批、输入预分配、输出 streaming 是更稳妥的小到中等收益优化。
