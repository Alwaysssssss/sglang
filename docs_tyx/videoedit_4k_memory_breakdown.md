# VideoEdit 4K 显存占用模块说明

本文档用于解释 SGLang VideoEdit 跑 4K 视频时，显存主要可能消耗在哪些模块和阶段。这里先讲清楚“哪些地方可能吃显存最多”，不是精确 profiling 结果。精确到每个模块用了多少 GiB，需要再加显存打点或用 PyTorch profiler / memory snapshot 实测。

## 结论先行

当前 VideoEdit pipeline 的核心模块是：

```text
text_encoder / tokenizer / vae / transformer(DiT) / scheduler
```

其中当前 VideoEdit 路径没有单独的 ViT / image_encoder。`referenceImageUrl` 会作为参考帧进入视频预处理和 VAE/DiT 路径，不是走 CLIP ViT 图像编码器。

显存压力通常分两类：

1. **模型权重常驻显存**：主要是 `transformer(DiT)`，其次是 `VAE` 和 `text_encoder`。
2. **运行时峰值显存**：4K 爆显存通常主要来自单窗口视频张量、VAE encode/decode 激活、DiT denoising 的 latent/condition 拼接和 attention/MLP workspace。

对 4K 来说，最危险的不是 text embedding，而是分辨率相关的张量和 DiT/VAE 中间激活。

## 模块占用优先级

| 模块/阶段 | 是否参与当前 VideoEdit | 显存压力 | 说明 |
| --- | --- | --- | --- |
| `transformer` / DiT | 是 | 最高 | 常驻权重最大；denoising 每一步都会跑；4K token 数量上升后，attention/MLP 中间激活和 workspace 会显著放大。 |
| `VAE encode` | 是 | 高 | 每个窗口会对 masked video 和 raw video 各 encode 一次。4K 输入张量和 VAE 激活会产生明显峰值。 |
| `VAE decode` | 是 | 高 | denoising 后把 latent 解码回视频帧。即使 `vae_tiling=True`，4K 解码仍可能有较高峰值。 |
| 单窗口输入张量 | 是 | 高 | `masked_video_tensor`、`video_tensor`、`mask_video_tensor` 会被搬到 GPU。4K 时这些张量本身就是 GiB 级别。 |
| latent / condition 张量 | 是 | 中到高 | latent 空间比原图小 8 倍空间压缩、4 倍时间压缩，但 denoising 会同时保存 latents、cond masks、cond latents、noise 等多个张量。 |
| `text_encoder` / T5 | 是 | 中 | 权重可能占显存，但运行时 prompt embedding 很小；通常不是 4K OOM 的主要原因。 |
| tokenizer | 是 | 极低 | CPU 侧为主，不是显存问题。 |
| scheduler | 是 | 极低 | timesteps/sigmas 等很小，不是显存问题。 |
| ViT / image_encoder | 当前 VideoEdit 不参与 | 0 | 这个 pipeline 的 required modules 里没有 image_encoder。不要把 4K OOM 优先归因到 ViT。 |
| paste back / 保存视频 | 是 | GPU 低，CPU/RAM 高 | 主要是 CPU/PIL/numpy 帧处理，可能吃系统内存，但不是 GPU 显存大头。 |

## 当前 VideoEdit 执行顺序

代码中的 VideoEdit stages 顺序是：

```text
VideoEditWindowValidationStage
VideoEditTextEncodingStage
VideoEditConditionEncodingStage
VideoEditLatentPreparationStage
VideoEditTimestepPreparationStage
VideoEditLatentInitStage
VideoEditDenoisingStage
VideoEditDecodingStage
VideoEditWindowPostprocessStage
```

按显存峰值风险排序，最需要关注的是：

1. `VideoEditDenoisingStage`
2. `VideoEditConditionEncodingStage`
3. `VideoEditDecodingStage`
4. `VideoEditTextEncodingStage`

## 4K 单窗口为什么容易爆

当前 VideoEdit 固定 `infer_len=81`。以 4K 输入 `3840x2160`、单窗口 81 帧、bf16 粗略估算：

### 1. GPU 原始视频张量

`prepare_window_inputs()` 会构造并放到 GPU：

```text
masked_video_tensor: [T, 3, H, W]
video_tensor:        [T, 3, H, W]
mask_video_tensor:   [T, 1, H, W]
```

以 `T=81, H=2160, W=3840, bf16=2 bytes` 估算：

```text
一个 [81, 3, 2160, 3840] bf16 张量约 3.75 GiB
masked_video_tensor + video_tensor 约 7.5 GiB
mask_video_tensor 约 1.25 GiB
```

这还没有算 VAE encode 的中间激活、latent、DiT、CUDA workspace、缓存碎片。也就是说，4K 时仅单窗口输入张量就已经很重。

### 2. VAE latent 张量

VideoEdit latent shape 来自配置：

```text
[batch, 16, (num_frames - 1) // 4 + 1, height // 8, width // 8]
```

对于 81 帧 4K：

```text
latent shape = [1, 16, 21, 270, 480]
```

单个 bf16 latent 约 83 MiB。这个数字看起来不大，但实际同时存在：

```text
runtime_video_latents
runtime_cond_latents
runtime_cond_masks
runtime_noise(float32)
runtime_latents
latent_model_input = cat([latents, cond_masks, cond_latents])
noise_pred / noise_uncond
```

所以 latent 相关总和会叠加，并且 denoising 中还有 transformer 内部激活和 attention workspace。

### 3. DiT denoising 是最大峰值来源

Denoising 每一步会构造：

```python
latent_model_input = torch.cat([
    latents,
    runtime_cond_masks,
    runtime_cond_latents,
], dim=1)
```

当前 VideoEdit 配置里：

```text
DiT in_channels = 36
DiT out_channels = 16
image_dim = 1280
added_kv_proj_dim = 5120
```

4K latent grid 是 `21 x 270 x 480`。即使模型内部会 patchify/优化 attention，4K 的 token/grid 规模仍远高于 720p/1080p。经验上，DiT denoising 的峰值显存通常是 4K OOM 的第一嫌疑。

### 4. VAE encode/decode 也会产生峰值

`VideoEditConditionEncodingStage` 会执行两次 VAE encode：

```text
masked_video -> VAE encode -> cond_latents
raw_video    -> VAE encode -> video_latents
```

`VideoEditDecodingStage` 会把 denoised latents decode 回视频帧。

当前配置默认：

```text
vae_precision = bf16
vae_tiling = True
```

`vae_tiling=True` 会缓解 VAE 阶段显存，但不能解决 DiT denoising 的显存问题。

## 模型权重显存 vs 运行时峰值显存

需要区分两件事：

### 1. 服务启动后就占用的显存

这部分主要是模型权重：

```text
transformer(DiT)  最大
VAE               次之
text_encoder      再次
scheduler/tokenizer 基本可忽略
```

如果你看到服务刚启动、还没跑任务时显存已经很高，这主要是权重常驻造成的。

### 2. 请求运行过程中新增的峰值显存

这部分主要来自：

```text
4K window 输入张量
VAE encode/decode 激活
DiT denoising 激活/workspace
CFG 额外 forward 路径
CUDA allocator 缓存和碎片
```

如果服务启动后显存还够，但一跑 4K 就 OOM，通常是运行时峰值显存的问题。

## 各模块可能的占比判断

没有实测打点前，只能给定性占比：

| 场景 | 主要占比 |
| --- | --- |
| 空载已加载模型 | DiT/transformer 权重通常占最大头，VAE 和 text_encoder 其次。 |
| VAE encode 阶段 | 4K 输入张量 + VAE 激活是峰值重点。 |
| DiT denoising 阶段 | DiT 权重 + latent/condition + attention/MLP workspace 通常是最大峰值。 |
| VAE decode 阶段 | VAE decoder 激活 + decoded video tensor 可能冲高。 |
| 最终 paste back / 保存 | GPU 显存不是主因，CPU/RAM 和磁盘 IO 更相关。 |

对 4K OOM，优先排查顺序建议是：

```text
DiT denoising > VAE encode/decode > 单窗口输入张量 > text_encoder > scheduler/tokenizer
```

## 和请求参数的关系

| 参数 | 对显存的影响 |
| --- | --- |
| `video` 分辨率 | 最关键。H/W 越大，输入张量、VAE 激活、DiT token/grid 都变大。 |
| `infer_len` | 当前 VideoEdit 要求 81。单窗口帧数固定较大。 |
| `num_frames` | 总帧数影响窗口数量和 CPU 累积帧内存；单窗口 GPU 峰值主要看 `infer_len`。 |
| `overlap` | 增加窗口重叠，通常增加总耗时和窗口数，不一定明显增加单窗口峰值。 |
| `num_inference_steps` | 增加耗时；单步峰值变化不大，但 cache/编译/调度可能影响峰值。 |
| `guidance_scale > 1` | 会启用 CFG，denoising 中可能多一次 negative forward，增加峰值或至少增加时间。 |
| `enable_paste_back` | 主要增加 CPU/RAM 后处理，不是 GPU 显存第一来源。 |
| `dtype=bf16` | 比 fp32 省显存。4K 时必须优先使用 bf16/fp16。 |
| `vae_tiling` | 缓解 VAE 阶段，不缓解 DiT 阶段。 |

## 4K 当前最大风险点

对于 `3840x2160`、`infer_len=81`，最大的风险是：

1. 单窗口 raw/masked/mask GPU 张量已经接近或超过 8 GiB 量级。
2. VAE encode 需要处理两个视频输入：masked video 和 raw video。
3. DiT denoising 的 latent grid 来自 4K 空间尺寸，attention/MLP 工作区会随分辨率显著增大。
4. 模型权重已经常驻显存，留给运行时张量的 headroom 不足。

因此 4K 跑不动时，不应该优先怀疑 text embedding 或 ViT。当前 VideoEdit 没有 ViT；text embedding 通常也不是 4K OOM 主因。

## 如何拿到精确每阶段显存

后续可以加显存打点，建议在每个 stage 前后记录：

```python
torch.cuda.reset_peak_memory_stats()
start_alloc = torch.cuda.memory_allocated()
start_reserved = torch.cuda.memory_reserved()

# run stage

end_alloc = torch.cuda.memory_allocated()
end_reserved = torch.cuda.memory_reserved()
peak_alloc = torch.cuda.max_memory_allocated()
peak_reserved = torch.cuda.max_memory_reserved()
```

建议打点位置：

```text
VideoEditTextEncodingStage.forward
VideoEditConditionEncodingStage.forward
VideoEditLatentPreparationStage.forward
VideoEditLatentInitStage.forward
VideoEditDenoisingStage.forward，每 step 或每 N step
VideoEditDecodingStage.forward
VideoEditWindowPostprocessStage.forward
```

输出字段建议：

```text
stage
window_index
height / width / num_frames
allocated_before_gib
allocated_after_gib
peak_allocated_gib
reserved_before_gib
reserved_after_gib
peak_reserved_gib
```

这样才能回答“每个模块实际用了多少 GiB、占比是多少”。

## 降低显存的优先方向

如果目标是让 4K 跑起来，优先级建议如下：

1. **降低进入模型的 crop 分辨率**：如果 mask 区域很小，确保只裁剪修复区域，不要让模型处理整幅 4K。
2. **确认 `decode_mode=stream`**：避免一次性保留过多完整帧。
3. **开启或保持 `vae_tiling=True`**：降低 VAE encode/decode 峰值。
4. **考虑 DiT layerwise offload**：`--dit-layerwise-offload` 可以降低 DiT 权重常驻和部分峰值，但会牺牲速度。
5. **考虑 `--vae-cpu-offload` / `--text-encoder-cpu-offload`**：能释放部分权重显存，但不能解决所有 4K runtime 激活问题。
6. **减少 CFG 压力**：`guidance_scale=1.0` 会关闭 CFG，可能降低 denoising 负担，但会影响生成质量。
7. **降低输出分辨率或分块修复**：如果整幅 4K 都进模型，单卡 L20 这类显存容量很容易不够。

## 一句话总结

当前 VideoEdit 里，4K 爆显存的第一嫌疑是 **DiT denoising 的 4K latent/token 规模和运行时 workspace**，第二嫌疑是 **VAE encode/decode 和单窗口 4K 输入张量**。`text_encoder` 会占一些模型权重显存，但通常不是 4K OOM 的核心；`ViT/image_encoder` 在当前 VideoEdit pipeline 中不参与。
