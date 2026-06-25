# Wan VideoEdit 推理流程对比

本文对比当前两份源码：

- SGLang pipeline：`python/sglang/multimodal_gen/runtime/pipelines/wan_videoedit_pipeline.py`
- SGLang stages：`python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/videoedit_wan.py`
- SGLang VideoEdit runtime：`python/sglang/multimodal_gen/runtime/videoedit/`
- SGLang DiT/VAE：`python/sglang/multimodal_gen/runtime/models/dits/wanvideo.py`、`wan_videoedit.py`、`runtime/models/vaes/wanvae.py`
- VideoEdit-diffusers：`/mnt/shanhai-ai/shanhai-workspace/zhouhao6/VideoEdit-diffusers/infer.py`
- VideoEdit-diffusers pipeline/model：`pipelines/pipeline_wan_edit.py`、`models/transformer_wan.py`、`models/autoencoder_kl_wan.py`、`models/flow_match.py`
- VideoEdit-diffusers utils：`utils/preprocess.py`、`utils/postprocess.py`

本次只做源码对比，不运行测试或推理程序。结论以当前工作区源码为准；旧文档中关于 `add_noise`、`nearest-exact` 等对齐建议与当前 `infer.py`/`utils/preprocess.py` 已不完全一致。

## 1. 总结

SGLang 已经把 VideoEdit-diffusers 的核心单窗口生成路径拆成了可组合 stages：

```text
global preprocess/windowing
  -> WindowValidation
  -> TextEncoding
  -> ImageEncoding
  -> ConditionEncoding(VAE encode masks/masked video)
  -> LatentPreparation(noise)
  -> TimestepPreparation
  -> LatentInit
  -> Denoising(transformer + scheduler)
  -> Decoding(VAE decode)
  -> WindowPostprocess
  -> window commit/final paste-back/save
```

核心模型契约基本一致：

```python
latent_model_input = torch.cat([latents, cond_masks, cond_latents], dim=1)
```

对应通道为 `16 + 4 + 16 = 36`，DiT 输出仍为 16 latent channels。

但 SGLang 不是默认 1:1 的 VideoEdit-diffusers `infer.py` 复刻。最重要的差异是：

- SGLang 默认 `num_inference_steps=40`，VideoEdit-diffusers CLI 默认 20。
- SGLang sampling 默认 `overlap=10`、CLI 默认 9；VideoEdit-diffusers 默认 9。
- SGLang 默认 `overlap_commit_mode="weighted"`，VideoEdit-diffusers 是 native skip：后续窗口只提交非 overlap 区域。
- SGLang 默认 `tail_padding_mode="reflect"`，VideoEdit-diffusers 尾部是 reverse mirror。
- VideoEdit-diffusers 默认 `clip_preprocess="diffuser"`，SGLang 只实现 DiffSynth 风格 CLIP 预处理。
- SGLang 默认 `decode_mode="stream"`，VideoEdit-diffusers 只 eager 加载完整帧列表。
- SGLang 支持 TeaCache、TP/SP/FSDP、sparse attention、offload、streaming decode、metadata/progress 等服务化能力；严格复现时应关闭或固定。

## 2. 顶层入口对比

| 层级 | VideoEdit-diffusers | SGLang | 复现影响 |
| --- | --- | --- | --- |
| CLI 入口 | `infer.py:454` 调用 `infer(build_parser().parse_args())` | `runtime/videoedit/cli.py:216` 构造 `WanVideoEditSamplingParams` 后通过 `DiffGenerator` 发送 `Req` | SGLang CLI 是服务/runtime 包装，不是直接调用 pipeline 函数 |
| 模型加载 | `infer.py:197-233` 直接加载 tokenizer/text_encoder/VAE/transformer/image_encoder/scheduler/pipeline | `WanVideoEditPipeline` 声明必需组件 `text_encoder/tokenizer/vae/transformer/scheduler`，由 SGLang loader 加载；`initialize_pipeline()` 创建 scheduler | SGLang 需要显式保证 transformer checkpoint 路径是 VideoEdit fine-tuned transformer |
| 顶层推理 | `infer.py:236-449` 串行做 preprocess、window loop、保存 | `wan_videoedit_pipeline.py:567-654` 做 global context、window specs、逐窗口执行 stages、commit、finalize | SGLang 顶层被拆为 pipeline orchestration + stage executor |
| 单窗口生成 | `infer.py:338-360` 调用 `WanPipeline.__call__`，`pipeline_wan_edit.py:560-860` 完成编码/denoise/可选 decode | `create_pipeline_stages()` 注册单窗口 stages，`forward()` 每个窗口调用 `executor.execute_with_profiling(self.stages, ...)` | SGLang stage 组合对应 diffusers `WanPipeline.__call__` 的内部步骤 |
| 输出 | `infer.py:410-446` 保存 pasted/crop/color 三种视频，函数返回 `None` | pipeline 最终设置 `batch.output`，SGLang 输出工具保存主视频；可选 crop sidecar 和 `.videoedit.json` metadata | SGLang 没有当前 VideoEdit 的 `save_color` 独立输出路径 |

## 3. 组件构建与加载

### 3.1 VideoEdit-diffusers

`infer.py` 的模型构建顺序：

1. `AutoTokenizer.from_pretrained(args.model_path, subfolder="tokenizer")`
2. `UMT5EncoderModel.from_pretrained(args.model_path, subfolder="text_encoder", torch_dtype=load_dtype)`
3. `AutoencoderKLWan.from_pretrained(args.model_path, subfolder="vae", torch_dtype=load_dtype)`
4. 可选 `vae.enable_tiling()`
5. `WanTransformer3DModel.from_pretrained(args.transformer_path, subfolder="transformer", torch_dtype=load_dtype, low_cpu_mem_usage=True, local_files_only=True)`
6. `use_clip` 时加载 `CLIPVisionModel` 和 `CLIPImageProcessor`
7. `FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)`
8. `WanPipeline(...).to("cuda")`

### 3.2 SGLang

SGLang 通过 registry 与 component loader 构建：

- `registry.py:719-729` 注册 `WanVideoEditSamplingParams` 与 `WanVideoEditPipelineConfig`。
- `wan_videoedit_pipeline.py:105-111` 声明必需模块。
- `wan_videoedit_pipeline.py:113-118` 用 `VideoEditFlowMatchScheduler(shift=flow_shift or 5.0, sigma_min=0.0, extra_one_step=True)` 覆盖 scheduler。
- `wan_videoedit_pipeline.py:121-140` 可从 `component_paths["image_encoder"]` 或 `model_path/image_encoder` 加载 CLIP image encoder。
- `transformer_loader.py:39-60` 从 component config 更新 SGLang DiT config，再按 `_class_name` 解析 model class。
- `transformer_load_utils.py:160-181` 支持 `server_args.transformer_weights_path` 覆盖 transformer 权重。

1:1 复现要求：

- `component_paths["transformer"]` 或 `transformer_weights_path` 必须指向 VideoEdit fine-tuned transformer，而不是 base Wan transformer。
- transformer class 必须解析到 `WanVideoEditTransformer3DModel` 或等价 VideoEdit config；不能落到 base `WanTransformer3DModel` 的 I2V hard-coded token split。
- 关闭或固定 runtime quantization、TP/SP/FSDP/offload/sparse attention，避免数值路径变化。

## 4. 配置和默认值差异

| 参数 | VideoEdit-diffusers 当前默认 | SGLang sampling 默认 | SGLang CLI 默认 | 1:1 建议 |
| --- | --- | --- | --- | --- |
| `infer_len` | 81 | 81 | 81 | 一致 |
| `num_inference_steps` | 20 (`infer.py:134`) | 40 | 40 | 显式设 20 或两边同值 |
| `guidance_scale` | 5.0 | 5.0 | 5.0 | 一致 |
| `seed` | 42 | 由基础 params/请求决定，CLI 为 42 | 42 | 显式固定 |
| `overlap` | 9 | 10 | 9 | 显式设 9 或同值 |
| `overlap_commit_mode` | native skip 语义 | `weighted` | `weighted` | 设 `native_skip`，并注意 SGLang 仍只替换首帧 reference |
| `tail_padding_mode` | reverse mirror | `reflect` | `reflect` | 设 `native_reverse_mirror` |
| `dtype` | bf16 | bf16 | bf16 | 一致 |
| `dynamic_cfg` | True | True | True | 一致 |
| `use_clip` | True | True | True | 是否一致取决于 CLIP preprocess |
| `clip_preprocess` | `diffuser` | 无参数，只有 DiffSynth 风格 | 无参数 | 在 reference 侧设 `--clip_preprocess diffsynth`，或在 SGLang 增加 diffuser path |
| `vae_tiling` | True | pipeline config True | pipeline config True | 注意 SGLang VAE feature cache 路径和 tiling 语义不同 |
| `dilate_px` | 0 | 0 | 0 | 一致 |
| `mask_scale` | 1.0 | 1.0 | 1.0 | 一致 |
| `bbox_expand_scale` | 0.3 | 0.3 | 0.3 | 一致 |
| `feather_px` | 0 | 0 | 0 | 一致 |
| `mask_downsample_mode` | nearest | nearest | nearest | 一致；不要改成 `nearest-exact` 追求当前源码 parity |
| `init_latent_mode` | pure noise (`video_latents=None`) | `noise` | `noise` | 一致 |
| `decode_mode` | eager only | `stream` | `stream` | strict 对齐设 `eager` |
| `enable_teacache` | 无 | True | False | strict 对齐关闭 |

## 5. Stage/dataflow 一比一映射

### 5.1 Global preprocess

VideoEdit-diffusers：

- `infer.py:240-251` 调用 `prepare_global_inputs()`。
- `utils/preprocess.py:440-447` eager 读取视频和 mask 视频帧。
- `utils/preprocess.py:452-463` 如果 `img_path` 存在，prepend reference image 和 zero mask。
- `utils/preprocess.py:469-502` 做 mask dilation/scale、bbox、bbox expand、小 bbox 扩展、crop、align、resize。

SGLang：

- `wan_videoedit_pipeline.py:168-241` 准备全局上下文。
- `decode_mode="stream"` 时走 `scan_global_bbox()` 和 `WindowFrameProvider`，不一次性保留全部 resized frames。
- eager 时调用 `runtime/videoedit/preprocess.py:453-541`，逻辑与 reference 更接近。
- SGLang mask 读入支持视频、numpy、npz、COCO JSON 等，reference 只把 mask 当视频帧读。

差异点：

- strict 对齐建议用 `decode_mode="eager"`，先排除 streaming decode 的 bbox/materialize 差异。
- reference 的 `infer.py:390-392` 删除 reference 首帧逻辑被注释；SGLang `drop_reference_frame=True` 时最终会丢首帧。使用 reference image 时要显式统一。

### 5.2 Windowing 和 overlap

VideoEdit-diffusers：

- `infer.py:256-273` 使用 `stride = infer_len - overlap`，`while next_start + overlap < total_frames` 生成 starts。
- `infer.py:300-310` 从上一窗口取 `prev_window_frames[stride:stride+overlap]`。
- `utils/preprocess.py:589-597` 将当前窗口前 `overlap` 帧全部替换成上一窗口生成结果，并将这些 mask 置黑。
- `infer.py:381-386` 首窗口提交所有 valid frames，后续窗口从 `local_idx=overlap` 开始提交。

SGLang：

- `windowing.py:52-65` 支持 native skip 与 weighted 两套窗口步进；weighted 使用 `infer_len - overlap - 1`。
- `wan_videoedit_pipeline.py:286-308` 当前只把窗口 `frames[0]` 替换为上一窗口某个输出帧。
- `wan_videoedit_pipeline.py:310-326` 根据 spec 将前若干 mask 置黑。
- `wan_videoedit_pipeline.py:399-429` commit 时支持 native skip 或 weighted accumulation。

关键差异：

- SGLang 默认 `weighted` 不是 VideoEdit-diffusers 行为。
- 即使设为 `native_skip`，当前 SGLang 也不是完全等价：reference 替换前 `overlap` 帧，SGLang 只替换首帧并置黑 overlap masks。
- 若要严格复现 reference，多窗口场景需要把 SGLang `_apply_previous_window_reference()` 扩展为替换前 `overlap` 帧，或在文档/参数中明确当前是 SGLang 连续性策略。

### 5.3 Per-window tensor preparation

VideoEdit-diffusers：

- `infer.py:325-331` 将 `masked_video_tensor` 和 raw `video_tensor` 从 `(T,C,H,W)` 变为 `(1,C,T,H,W)` 后 VAE encode。
- `utils/preprocess.py:616-645` 生成 `masked_video_tensor`、`mask_video_tensor`、`cond_masks`、`video_tensor`。

SGLang：

- `VideoEditConditionEncodingStage.forward()` 在 `videoedit_wan.py:377-388` 调用 SGLang `prepare_window_inputs()`。
- `runtime/videoedit/preprocess.py:565-582` 生成同样的 tensor 字段。

对齐点：

- `masked_video_tensor`: `(T,3,H,W)`，值域 `[-1,1]`。
- `mask_video_tensor`: `(T,1,H,W)`，值域 `[0,1]`。
- `cond_masks`: 首 mask repeat 4 次，再拼接后续 mask，`F.interpolate(..., scale_factor=1/8, mode="nearest")`，反转 `(cond_masks < 0.5).float()`，reshape 为 `(1,4,F_lat,H/8,W/8)`。
- `infer_len=81` 时，mask 帧数变为 84，latent frame 数为 21。

### 5.4 Text encoding

VideoEdit-diffusers：

- `pipeline_wan_edit.py:217-256` `_get_t5_prompt_embeds()` 清理 prompt、tokenize、UMT5 encode、按 attention mask 长度截断再 pad。
- helper 默认参数是 226，但实际 `WanPipeline.__call__` 默认 `max_sequence_length=512`，`infer.py` 未覆盖，所以当前实际路径是 512。
- negative prompt 在 CFG 时同样 encode。

SGLang：

- `VideoEditTextEncodingStage` 用通用 `TextEncodingStage`，目标 dtype 来自 transformer。
- `WanVideoEditPipelineConfig` 使用 `T5Config()`，`T5Config.text_len=512`。
- `videoedit_prompt_clean()` 执行 ftfy/html unescape/whitespace clean，语义对应 reference。

对齐点：

- 当前源码实际 text length 都是 512。
- `WanVideoEditCrossAttention.text_context_len=512` 与 VideoEdit-diffusers 的 `context_len - 512` image split 一致。

### 5.5 CLIP image encoding

VideoEdit-diffusers：

- `infer.py:215-227` 仅当 transformer `image_dim` 非 None 时加载 CLIP image encoder/processor。
- `pipeline_wan_edit.py:722-735` 如果传入 image，先 resize 到 `(width,height)`，再根据 `clip_preprocess` 选择路径。
- `clip_preprocess="diffuser"` 默认走 `CLIPImageProcessor`。
- `clip_preprocess="diffsynth"` 走手工 `[-1,1] -> bicubic 224 -> [0,1] -> CLIP mean/std`。

SGLang：

- `VideoEditImageEncodingStage` 只实现手工 DiffSynth 风格预处理。
- image encoder 输出后通过 `pipeline_config.postprocess_image()` 取 `hidden_states[-2]`。
- 若 `use_clip=True` 但 image encoder 未加载，stage 直接报错；VideoEdit-diffusers 在 `image_dim is None` 时会自动禁用 CLIP。

关键差异：

- 当前 VideoEdit-diffusers CLI 默认是 `diffuser`，SGLang 是 DiffSynth 风格。严格复现要么在 reference 侧设 `--clip_preprocess diffsynth`，要么给 SGLang 增加 `CLIPImageProcessor` 路径。

### 5.6 VAE condition/raw latent encoding

VideoEdit-diffusers：

- `infer.py:42-57` `prepare_latents()` 调用 `vae.encode()`，取 latent_dist.mode，执行 `(latents - mean) / std`。
- `infer.py:330-331` 同时 encode masked video 和 raw video。
- 但 `infer.py:349-356` 当前传给 pipeline 的 `video_latents=None`，所以 denoise 初始 latent 是 pure noise，不是 SDEdit/add_noise(raw_video)。

SGLang：

- `VideoEditConditionEncodingStage._encode_video_latents()` 调用 VAE encode，取 mode，执行 `(latents - mean) / std`。
- `VideoEditConditionEncodingStage.forward()` 同样 encode masked video 和 raw video。
- `VideoEditLatentInitStage` 默认 `init_latent_mode="noise"`，只有显式 `add_noise` 才使用 raw video latents。

对齐点：

- 当前两边默认都是 pure noise start。
- 旧文档若写 strict 应设 `add_noise`，已经不符合当前 `infer.py`。

### 5.7 Noise、timesteps、scheduler

VideoEdit-diffusers：

- `infer.py:333-337` 每个窗口重新创建 CPU generator 并用同一个 seed，匹配 DiffSynth noise。
- `pipeline_wan_edit.py:462-503` 如果未传 latents，则在 CPU/指定 generator 上生成 float32 noise，再转目标 device/dtype。
- `pipeline_wan_edit.py:738-742` `scheduler.set_timesteps(num_inference_steps, shift=5)`。
- `models/flow_match.py:40-68` step/add_noise 公式。

SGLang：

- `VideoEditLatentPreparationStage` 按 `params.seed` 创建 generator；`generator_device` 默认来自 pipeline config，当前是 CPU。
- `VideoEditTimestepPreparationStage` 用 `flow_shift or 5.0`。
- `VideoEditFlowMatchScheduler` 复刻 set_timesteps、step、add_noise、get_timesteps，并支持 device 放置。

对齐点：

- scheduler runtime 参数一致：`shift=5`、`sigma_min=0.0`、`extra_one_step=True`。
- 若 `vary_seed_by_window=False`，SGLang 每窗口 seed 不变，和 reference 一致。

### 5.8 Denoising loop

VideoEdit-diffusers：

- `pipeline_wan_edit.py:778-784` 计算 dynamic CFG。
- `pipeline_wan_edit.py:787-801` cond pass：`current_model(hidden_states=cat(...), timestep, encoder_hidden_states=prompt_embeds, encoder_hidden_states_image=image_embeds)`。
- `pipeline_wan_edit.py:803-813` CFG 时 uncond pass。
- `pipeline_wan_edit.py:816` scheduler step。

SGLang：

- `VideoEditDenoisingStage.forward()` 在 `videoedit_wan.py:591-682` 执行相同循环。
- `videoedit_wan.py:605-608` 连接 `[latents, cond_masks, cond_latents]`。
- `videoedit_wan.py:623-633` cond pass。
- `videoedit_wan.py:636-659` uncond pass 和 CFG merge。
- `videoedit_wan.py:662` scheduler step。

差异点：

- SGLang 额外设置 `set_forward_context()`、attention metadata、progress、request timeout。
- SGLang transformer 支持 TeaCache skip；VideoEdit-diffusers 没有该 skip。严格复现关闭 TeaCache。
- SGLang 不支持当前 VideoEdit stage 的 CFG parallel，打开会报 `NotImplementedError`。

### 5.9 Decode、window commit、final output

VideoEdit-diffusers：

- `infer.py:60-73` `post_latents()` 先 `latents * std + mean`，再 VAE decode。
- `infer.py:362-374` postprocess video 到 `[0,1]`，转 uint8 PIL frames。
- `infer.py:381-386` 后续窗口只提交非 overlap frames。
- `infer.py:410-446` 保存 pasted、crop-only、color-corrected 三种输出。

SGLang：

- `VideoEditDecodingStage` 先 denormalize，再 VAE decode，`decoded / 2 + 0.5` 后生成 PIL frames。
- `WanVideoEditPipeline._commit_window_output()` 支持 weighted/native_skip accumulation。
- `_finalize_long_video_output()` 可 paste_back，或只 resize crop frames；可 drop reference first frame。
- `_save_crop_sidecar()` 可保存 crop-only sidecar。
- `_write_metadata()` 写 `.videoedit.json`。

差异点：

- SGLang `runtime/videoedit/postprocess.py` 当前忽略 `adain_boundary_dilate`，没有 `color_correct=True` 对应路径。
- SGLang 主输出保存由通用 runtime 处理；VideoEdit-diffusers 用 moviepy `ImageSequenceClip(..., codec="libx264", bitrate="10M")`。
- 做视觉/数值 compare 时，应先比较解码后的 RGB frames，避免视频编码参数差异污染结论。

## 6. 模型结构和 layer 调用

### 6.1 Transformer 架构

两边核心顺序一致：

```text
hidden_states [B,36,F,H,W]
  -> patch embedding Conv3d/PatchEmbed, patch_size=(1,2,2)
  -> flatten to token sequence
  -> time/text/image embedding
  -> concat image tokens before text tokens
  -> repeated WanTransformerBlock:
       self attention with RoPE
       cross attention to text/image context
       FFN
  -> output norm + projection
  -> unpatchify to [B,16,F,H,W]
```

关键配置：

| 配置 | VideoEdit-diffusers | SGLang VideoEdit |
| --- | --- | --- |
| patch size | `(1,2,2)` | `(1,2,2)` |
| heads | 40 | 40 |
| head dim | 128 | 128 |
| hidden size | 5120 | 5120 |
| in channels | 36 | 36 |
| out channels | 16 | 16 |
| text dim | 4096 | 4096 |
| image dim | checkpoint/config dependent, VideoEdit uses CLIP path | forced to 1280 in `WanVideoEditPipelineConfig` |
| added KV dim | optional, VideoEdit image KV | forced to 5120 in `WanVideoEditPipelineConfig` |
| FFN dim | 13824 | 13824 |
| layers | 40 | 40 |
| qk norm | rms norm across heads | rms norm across heads |
| text context len | 512 actual call path | 512 |

### 6.2 Cross attention token split

VideoEdit-diffusers：

- `models/transformer_wan.py:84-89` 当有 added KV projection 时，`image_context_length = encoder_hidden_states.shape[1] - 512`，前面是 image tokens，后面是 text tokens。

SGLang：

- base `WanI2VCrossAttention` 在 `wanvideo.py:337-338` hard-code `context[:, :257]` 为 image tokens。
- VideoEdit 专用 `WanVideoEditCrossAttention` 在 `wan_videoedit.py:24-30` 改成动态 `context.shape[1] - 512`。
- `WanVideoEditTransformer3DModel.__init__()` 会替换每个 block 的 `attn2`。

1:1 复现要求：

- 必须实例化 VideoEdit 专用 transformer。若错误加载 base Wan I2V transformer，CLIP/text token split 会错。

### 6.3 Attention、RoPE、FFN 差异

| 层 | VideoEdit-diffusers | SGLang | 影响 |
| --- | --- | --- | --- |
| self attention | diffusers `WanAttention` + `dispatch_attention_fn(is_causal=False)` | `USPAttention(causal=False)` 或 sparse/VSA variants | 默认 original 后数学目标一致；sparse/VSA 不是 1:1 |
| QKV projection | 支持 fused QKV/KV processors | tensor-parallel separate `to_q/to_k/to_v` | 未量化 TP=1 时应等价，但 rounding/call granularity 可不同 |
| RoPE | full grid `WanRotaryPosEmbed` | `NDRotaryEmbedding`，支持 sequence shard | strict 关闭 SP，使用 full path |
| cross attention | image attention result + text attention result | VideoEdit adapter 同样 image/text 分开 attention 后相加 | 需要专用 adapter |
| FFN | `FeedForward(... gelu-approximate)` | `MLP(... gelu_pytorch_tanh)` | 目标应对应，但实现细节不同 |
| output projection | dense `nn.Linear` | `ColumnParallelLinear(gather_output=True)` | TP=1 最接近 |

### 6.4 VAE layer 顺序

VAE latent 约定一致：

- `z_dim=16`
- temporal compression 4
- spatial compression 8
- latent mean/std 16 个值一致
- encode 取 posterior mode
- normalize `(latents - mean) / std`
- decode 前 denormalize `latents * std + mean`

Encoder 主链路：

```text
WanCausalConv3d conv_in
  -> down blocks
  -> mid block
  -> RMS norm
  -> SiLU
  -> WanCausalConv3d conv_out
  -> quant_conv
  -> DiagonalGaussianDistribution
```

Decoder 主链路：

```text
post_quant_conv
  -> WanCausalConv3d conv_in
  -> mid block
  -> up blocks
  -> RMS norm
  -> SiLU
  -> WanCausalConv3d conv_out
  -> clamp[-1,1]
```

差异点：

- VideoEdit-diffusers `AutoencoderKLWan` 是 diffusers `ModelMixin/ConfigMixin`，支持 `return_dict`、slicing、tiling 分支。
- SGLang `AutoencoderKLWan` 继承 `ParallelTiledVAE`，默认 `WanVAEConfig.use_feature_cache=True`，encode/decode 走 feature-cache temporal chunks。
- 在 feature-cache 路径下，SGLang `encode()`/`decode()` 不等价于 diffusers `_encode()`/`_decode()` 的 tiling 检查路径。大分辨率 strict parity 要重点验证这一点。

## 7. 1:1 复现检查项

为了让 SGLang 尽量复现当前 VideoEdit-diffusers `infer.py`，建议先使用以下约束：

1. transformer checkpoint 显式指向 VideoEdit fine-tuned transformer。
2. transformer class 确认是 `WanVideoEditTransformer3DModel` 或等价动态 image/text split。
3. `num_inference_steps=20`，或两边显式同值。
4. `overlap=9`，或两边显式同值。
5. `overlap_commit_mode="native_skip"`，并修正/确认多窗口前 `overlap` 帧替换语义。
6. `tail_padding_mode="native_reverse_mirror"`。
7. `decode_mode="eager"`，先排除 streaming materialize 差异。
8. `init_latent_mode="noise"`。
9. `mask_downsample_mode="nearest"`。
10. `vary_seed_by_window=False`，`generator_device="cpu"`。
11. `dynamic_cfg=True`，`dynamic_cfg_max_step=15`，`dynamic_cfg_min=1.0`。
12. `use_clip=True` 时，两边 CLIP preprocessing 必须统一：reference 设 `--clip_preprocess diffsynth`，或 SGLang 实现 diffuser CLIPImageProcessor 路径。
13. 关闭 TeaCache、Cache-DiT、sparse/video sparse attention、TP/SP/FSDP、CPU/GPU offload、quantization、torch compile。
14. `tp_size=1`、`sp_degree=1`。
15. 用解码后 RGB frame 对比，视频编码只作为最终保存验证。

## 8. 当前 SGLang 仍需补齐的严格差异

| 优先级 | 差异 | 当前证据 | 建议 |
| --- | --- | --- | --- |
| P0 | 多窗口 reference overlap 替换不等价 | reference 替换前 `overlap` 帧；SGLang 当前只替换 `frames[0]` | 若目标是逐帧复现，扩展 `_apply_previous_window_reference()` 为多帧替换 |
| P0 | CLIP preprocess 默认不等价 | reference CLI 默认 `diffuser`；SGLang 只有 DiffSynth path | 增加 `clip_preprocess` 参数和 `CLIPImageProcessor` 路径，或强制 reference 用 `diffsynth` |
| P1 | weighted overlap 是 SGLang 扩展默认 | SGLang defaults/CLI 为 `weighted` | strict preset 设 `native_skip` |
| P1 | tail padding 默认不同 | SGLang 默认 `reflect` | strict preset 设 `native_reverse_mirror` |
| P1 | VAE feature cache/tiling 路径不同 | SGLang VAE 默认 feature cache | 大图 strict parity 下验证或提供 diffusers-like VAE path |
| P2 | color-correct output 缺失 | SGLang `adain_boundary_dilate` 被忽略 | 若要复现 `save_color`，移植 `_adain_boundary` 和 `color_correct=True` |
| P2 | 服务化扩展会改变行为 | TeaCache、SP、sparse attention、streaming decode | strict preset 或命令模板中显式关闭 |

## 9. 结论

SGLang 当前已经复现了 VideoEdit-diffusers 的核心模型形状、scheduler 公式、mask packing、VAE latent normalize/denormalize、DiT 输入输出通道和 denoising loop。真正阻碍“整个推理流程一比一”的不是单个 layer 缺失，而是默认参数和窗口/CLIP/VAE runtime 语义：

- 单窗口、关闭扩展能力、统一 CLIP preprocess 时，核心 latent denoise 路径接近 reference。
- 多窗口时，overlap reference 替换语义仍不是 1:1。
- 服务默认值偏向 SGLang 部署和优化，不应直接声称是 VideoEdit-diffusers strict parity。

后续如果要把 SGLang 作为严格 reference 复刻，建议新增 `reference-strict` preset，集中固定上述参数，并在 metadata 中写入所有 effective values，避免 compare 时无法定位差异来源。

