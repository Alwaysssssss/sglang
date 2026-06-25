# VideoEdit-diffusers 接入 SGLang Diffusion 重构方案

本文档以当前本机源码为准：

- SGLang：`/mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang`
- VideoEdit-diffusers：`/mnt/shanhai-ai/shanhai-workspace/zhouhao6/VideoEdit-diffusers`
- wan_eraser serve 参考：`/mnt/shanhai-ai/shanhai-workspace/zhouhao6/wan_eraser/run_parallel_ray_95_erase.py`
- 模型目录：`/mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model`
- 测试视频：`/mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pexel_test_data_0410/videos/15108907_3840_2160_50fps_short.mp4`
- 测试 mask：`/mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pexel_test_data_0410/masks/15108907_3840_2160_50fps_No_bbox_mask.mp4`

目标不是把 `VideoEdit-diffusers` 原仓库整体搬进 SGLang，而是新增一条原生 `VIDEO_EDIT` 任务链路。除注册点和 enum 这类必要入口外，优先新增文件，尽量不改已有 Wan 实现。

## 目录

1. 结论与关键改动
2. Reference 行为拆解
3. 新任务类型 `VIDEO_EDIT`
4. 新增文件与最小修改清单
5. WanTransformer3DModel 结构比较
6. `WanVideoEditPipeline.forward` 重构
7. Stage 设计
8. `WanVideoEditSamplingParams` 设计
9. 长视频窗口策略
10. 服务与 CLI 方案
11. 端到端测试方案
12. 实施顺序

## 1. 结论与关键改动

本次重构采用 skill 中推荐的 Hybrid 思路，但不是把 reference `__call__` 原样复制为一个巨型函数。推荐做法是：

- 新增 `ModelTaskType.VIDEO_EDIT`，不伪装成已有生成任务。
- 新增 `WanVideoEditPipeline`，重写 `forward`，由 `forward` 做长视频窗口编排和 stage 循环。
- 所有 stage 只处理单个 81 帧窗口；多窗口、反射补齐、窗口融合、paste-back 编排放在 stage 外部。
- 所有 stage 的运行态中间变量统一写入 `WanVideoEditSamplingParams`，不再散落到 `batch.extra`、`Req` 顶层或 helper 私有字段。
- 复用现有 Wan VAE / text encoder / tokenizer loader；DiT 复用底层模块和权重映射，但新增 VideoEdit 专用入口文件；同时新增 VideoEdit scheduler adapter、stage、preprocess/postprocess 纯函数。
- 验收只保留 reference baseline、SGLang CLI、SGLang serve 三条端到端测试。

必须对齐的模型契约：

```python
latent_model_input = torch.cat([latents, cond_masks, cond_latents], dim=1)
```

张量形状：

- `latents`: `[B, 16, F_lat, H/8, W/8]`
- `cond_masks`: `[B, 4, F_lat, H/8, W/8]`
- `cond_latents`: `[B, 16, F_lat, H/8, W/8]`
- DiT 输入：`36` 通道
- DiT 输出：`16` 通道

## 2. Reference 行为拆解

核心参考文件：

- `VideoEdit-diffusers/pipelines/pipeline_wan_edit.py`
- `VideoEdit-diffusers/infer.py`
- `VideoEdit-diffusers/utils/preprocess.py`
- `VideoEdit-diffusers/utils/postprocess.py`
- `wan_eraser/run_parallel_ray_95_erase.py`

reference 的关键行为：

- 原始 pipeline 基于 Wan I2V 改来，但 VideoEdit 的实际输入不是首帧 image condition，而是完整视频、mask 视频、masked video latent 和原视频 latent。
- scheduler 是 `FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)`。
- timestep 逻辑是先 `set_timesteps()`，再在 `strength < 1.0` 时 `get_timesteps()` 裁剪有效 steps。
- noisy latent 初始化为：

```python
latents = scheduler.add_noise(video_latents, noise, timesteps[:1])
```

- mask packing 规则：

```python
first_frame_mask = mask_video_tensor[0:1].repeat(4, 1, 1, 1)
expanded_masks = torch.cat([first_frame_mask, mask_video_tensor[1:]], dim=0)
cond_masks = F.interpolate(expanded_masks, scale_factor=1 / 8, mode="nearest-exact")
cond_masks = (cond_masks < 0.5).float()
cond_masks = cond_masks.view(1, num_mask_frames // 4, 4, latent_h, latent_w)
cond_masks = cond_masks.transpose(1, 2).contiguous()
```

`infer_len=81` 时，mask 先变成 84 帧，再 pack 成 `[B, 4, 21, H/8, W/8]`。

wan_eraser 只作为服务行为参考，不复用其协议：

- FastAPI 提交后后台执行。
- 单任务 admission，运行中请求直接拒绝。
- `/health` 返回进程可用性。
- 任务完成后回调 / 状态更新。

它的 `/generate` 绑定 MinIO key、RLE mask JSON、bbox CSV 和 Ray actor，不适合作为 SGLang VideoEdit 的最终 API。

## 3. 新任务类型 `VIDEO_EDIT`

必须新增任务类型：

```python
class ModelTaskType(Enum):
    VIDEO_EDIT = auto()
```

`VIDEO_EDIT` 的语义：

- 输出类型仍是 video。
- 输入不是 image，不应触发 I2V/TI2V 的 `image_path`、`condition_image`、首帧图像校验逻辑。
- 输入由 `WanVideoEditSamplingParams.video_input_path` 和 `mask_input_path` 定义。
- 通用 `InputValidationStage` 不应处理 VideoEdit 业务输入；VideoEdit 自己的 validation stage 负责。

`ModelTaskType` helper 需要最小修改：

```python
def data_type(self) -> DataType:
    if self == ModelTaskType.I2M:
        return DataType.MESH
    if self.is_image_gen():
        return DataType.IMAGE
    return DataType.VIDEO
```

`requires_image_input()` 和 `accepts_image_input()` 不应把 `VIDEO_EDIT` 归为 image input。VideoEdit 的视频输入是独立业务输入，不是 SGLang 现有 image condition。

## 4. 新增文件与最小修改清单

### 4.1 新增文件

```text
python/sglang/multimodal_gen/configs/pipeline_configs/videoedit_wan.py
python/sglang/multimodal_gen/configs/sample/videoedit_wan.py
python/sglang/multimodal_gen/configs/models/dits/wan_videoedit.py
python/sglang/multimodal_gen/runtime/pipelines/wan_videoedit_pipeline.py
python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/videoedit_wan.py
python/sglang/multimodal_gen/runtime/models/dits/wan_videoedit.py
python/sglang/multimodal_gen/runtime/models/schedulers/videoedit_flow_match.py
python/sglang/multimodal_gen/runtime/videoedit/__init__.py
python/sglang/multimodal_gen/runtime/videoedit/contracts.py
python/sglang/multimodal_gen/runtime/videoedit/preprocess.py
python/sglang/multimodal_gen/runtime/videoedit/postprocess.py
python/sglang/multimodal_gen/runtime/videoedit/windowing.py
python/sglang/multimodal_gen/runtime/videoedit/io.py
python/sglang/multimodal_gen/runtime/videoedit/compare.py
python/sglang/multimodal_gen/runtime/videoedit/cli.py
```

### 4.2 必要修改文件

```text
python/sglang/multimodal_gen/configs/pipeline_configs/base.py
python/sglang/multimodal_gen/configs/pipeline_configs/__init__.py
python/sglang/multimodal_gen/configs/sample/__init__.py
python/sglang/multimodal_gen/registry.py
python/sglang/multimodal_gen/runtime/models/registry.py
python/sglang/multimodal_gen/runtime/entrypoints/openai/protocol.py
python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py
```

这些修改只做注册、协议和 endpoint 增量，不改现有 Wan pipeline 的行为。

### 4.3 模型目录

overlay 模型目录保持 diffusers-style：

```text
VideoEdit-diffusers-model/
  model_index.json
  tokenizer/
  text_encoder/
  vae/
  transformer/
  scheduler/
```

`model_index.json`：

```json
{
  "_class_name": "WanVideoEditPipeline",
  "tokenizer": ["transformers", "AutoTokenizer"],
  "text_encoder": ["transformers", "UMT5EncoderModel"],
  "vae": ["diffusers", "AutoencoderKLWan"],
  "transformer": ["sglang", "WanVideoEditTransformer3DModel"],
  "scheduler": ["sglang", "VideoEditFlowMatchScheduler"]
}
```

`--transformer-path` 仍然作为组件覆盖：

```bash
--transformer-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/step-55000-diffusers-lh/transformer
```

业务参数不要用通用 unknown args 传给 `sglang serve`，避免 `--xxx-path` 被 `ServerArgs._extract_component_paths()` 误解析为组件路径。

## 5. WanTransformer3DModel 结构比较

对比对象：

- reference：`VideoEdit-diffusers/models/transformer_wan.py`
- SGLang：`python/sglang/multimodal_gen/runtime/models/dits/wanvideo.py`
- 本地权重配置：`VideoEdit-diffusers-model/transformer/config.json`

### 5.1 配置对比

VideoEdit transformer config：

```json
{
  "_class_name": "WanTransformer3DModel",
  "patch_size": [1, 2, 2],
  "num_attention_heads": 40,
  "attention_head_dim": 128,
  "in_channels": 36,
  "out_channels": 16,
  "text_dim": 4096,
  "freq_dim": 256,
  "ffn_dim": 13824,
  "num_layers": 40,
  "cross_attn_norm": true,
  "qk_norm": "rms_norm_across_heads",
  "eps": 1e-6,
  "image_dim": 1280,
  "added_kv_proj_dim": 5120,
  "rope_max_seq_len": 1024,
  "pos_embed_seq_len": null
}
```

SGLang `WanVideoConfig` 已支持这些字段，但默认 `in_channels=16`。VideoEdit 必须用专属 config 覆盖为：

```python
in_channels = 36
out_channels = 16
image_dim = 1280
added_kv_proj_dim = 5120
num_channels_latents = 16
```

### 5.2 模块结构对比

| 模块 | reference | SGLang 当前 Wan | 结论 |
| --- | --- | --- | --- |
| patch embedding | `nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)` | `PatchEmbed(...).proj` | 参数结构可映射，设置 `in_channels=36` 后可加载 |
| rope | `WanRotaryPosEmbed` | `NDRotaryEmbedding` | 数学等价，SGLang 额外支持 SP shard |
| time embedding | `Timesteps + TimestepEmbedding + time_proj` | `TimestepEmbedder + ModulateProjection` | 参数通过 `param_names_mapping` 映射 |
| text embedder | `PixArtAlphaTextProjection` | `MLP(..., act_type="gelu_pytorch_tanh")` | 参数通过 `param_names_mapping` 映射 |
| image embedder | `WanImageEmbedding(1280 -> 5120)` | `WanImageEmbedding(1280 -> 5120)` | 结构等价 |
| self attention | `attn1.to_q/to_k/to_v/to_out + RMSNorm` | 同名语义，TP/SP 优化实现 | 参数结构可映射 |
| cross attention | `attn2.to_q/to_k/to_v/to_out + add_k_proj/add_v_proj` | 同样有 I2V cross-attn 分支 | 参数结构可映射，但上下文切分语义不同 |
| ffn | diffusers `FeedForward` | SGLang `MLP` | 参数通过 `param_names_mapping` 映射 |
| output projection | `proj_out` | `ColumnParallelLinear proj_out` | 参数结构可映射 |

结论：权重参数结构接近 Wan I2V，且 SGLang 当前 `wanvideo.py` 已经具备大部分可复用低层模块和权重映射能力。但按可执行模型结构判断，VideoEdit 与当前 `WanTransformer3DModel` 的 I2V forward 语义不同，不能直接复用当前类。

### 5.3 关键差异：cross-attention image context 切分

reference 的 `WanAttnProcessor` 在有 `add_k_proj` 时这样切分：

```python
image_context_length = encoder_hidden_states.shape[1] - 512
encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
encoder_hidden_states = encoder_hidden_states[:, image_context_length:]
```

VideoEdit pipeline 只传文本 embedding，长度为 512，因此：

```python
image_context_length = 0
```

也就是说，VideoEdit 权重虽然保留 I2V 的 `add_k_proj/add_v_proj` 结构，但该路径在当前推理输入中没有真实 image tokens。reference 的行为是“允许 0 个 image tokens”。

SGLang 当前 `WanI2VCrossAttention` 固定切分：

```python
context_img = context[:, :257]
context = context[:, 257:]
```

这适合现有 Wan I2V 图像条件链路，但不适合 VideoEdit。若 VideoEdit 只传 512 个 text tokens，当前 SGLang 会错误地把前 257 个 text tokens 当成 image context，导致 cross-attention 语义偏离 reference。

### 5.4 决策：新增 VideoEdit DiT 文件

由于关键 forward 语义不同，首版不要直接复用 `runtime/models/dits/wanvideo.py` 的 `WanTransformer3DModel`。新增：

```text
python/sglang/multimodal_gen/configs/models/dits/wan_videoedit.py
python/sglang/multimodal_gen/runtime/models/dits/wan_videoedit.py
```

新增类：

```python
class WanVideoEditConfig(WanVideoConfig):
    arch_config = WanVideoEditArchConfig()


class WanVideoEditTransformer3DModel(WanTransformer3DModel):
    ...
```

实现策略：

- 复用 SGLang `PatchEmbed`、`WanTimeTextImageEmbedding`、`WanImageEmbedding`、rotary、TP/SP、offload、Teacache 和权重映射。
- 新增 `WanVideoEditCrossAttention`，不要硬编码 257。
- 新增 `WanVideoEditTransformerBlock`，只替换 cross-attention 类型。
- `WanVideoEditTransformer3DModel.forward()` 保持 SGLang 的 TP/SP/缓存能力，但在 cross-attention 前保留 reference 的动态切分语义。

建议 cross-attention 切分：

```python
text_context_len = 512
image_context_length = max(context.shape[1] - text_context_len, 0)
context_img = context[:, :image_context_length]
context_text = context[:, image_context_length:]
```

当 `image_context_length == 0` 时：

- 不调用 `add_k_proj/add_v_proj` attention。
- 只执行 text cross-attention。
- 保留 `add_k_proj/add_v_proj` 参数用于加载 VideoEdit 权重，但 forward 中不让空 image context 改变结果。

### 5.5 何时可以回退为复用

只有同时满足以下条件，才可以不新增 DiT 文件、直接复用现有 `wanvideo.py`：

- SGLang 当前 `WanI2VCrossAttention` 改为支持动态 image token 数。
- 对现有 Wan I2V 的 257 image token 行为保持兼容。
- VideoEdit 只传 512 text tokens 时，与 reference 首步 DiT 输出对齐。
- VideoEdit 权重 `in_channels=36/out_channels=16` 通过同一个 config 路径 fail-fast 校验。

在这些条件未满足前，新增 `wan_videoedit.py` 是更稳妥的落地方案，也符合“优先新增文件，尽量不修改原文件”的约束。

## 6. `WanVideoEditPipeline.forward` 重构

新增 pipeline：

```python
class WanVideoEditPipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "WanVideoEditPipeline"
    pipeline_config_cls = WanVideoEditPipelineConfig
    sampling_params_cls = WanVideoEditSamplingParams

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]
```

关键要求：重写 `forward`，不要只依赖 `ComposedPipelineBase` 默认串行执行。`forward` 需要把多窗口处理放在 stage 外部，所有 stage 每次只看到 81 帧窗口。

推荐结构：

```python
def forward(self, req, server_args):
    params: WanVideoEditSamplingParams = req.sampling_params

    self._prepare_global_videoedit_context(params, server_args)
    window_specs = build_videoedit_window_specs(
        num_frames=params.runtime_num_input_frames,
        infer_len=params.infer_len,
        overlap=params.overlap,
    )
    params.runtime_window_specs = window_specs

    repaired_windows = []
    for window_spec in window_specs:
        params.reset_window_runtime(window_spec)
        self._materialize_window_inputs(params, window_spec)

        for stage in self.videoedit_stages:
            req = stage.forward(req, server_args)

        repaired_windows.append(params.runtime_window_output_frames)
        self._commit_window_output(params, window_spec)

    self._finalize_long_video_output(params)
    return req
```

职责边界：

- `_prepare_global_videoedit_context()`：读取原视频 / mask，计算全局 bbox、fps、尺寸、全局 resized frames。它不是 stage，因为它处理的是整段视频。
- `_materialize_window_inputs()`：根据 `VideoEditWindowSpec` 取 81 帧，构造当前窗口视频、mask 和 window-local 元信息。
- `for stage in self.videoedit_stages`：每个 stage 只处理当前 81 帧窗口。
- `_commit_window_output()`：把当前窗口输出按 `commit_local_to_global` 写入全局累积 buffer。
- `_finalize_long_video_output()`：窗口融合、paste-back、音频拷贝、编码保存和 metadata 输出。

这样满足两个约束：

- 多帧 / 长视频编排不进入 stage。
- stage 不需要知道全局视频长度、窗口数量、overlap 融合或最终输出路径。

## 7. Stage 设计

先定义 stage，再定义 `WanVideoEditSamplingParams`。所有 stage 间中间变量都写入 `params`。

推荐 stage 列表：

```text
VideoEditWindowValidationStage
  -> VideoEditTextEncodingStage
  -> VideoEditConditionEncodingStage
  -> VideoEditLatentPreparationStage
  -> VideoEditTimestepPreparationStage
  -> VideoEditLatentInitStage
  -> VideoEditDenoisingStage
  -> VideoEditDecodingStage
  -> VideoEditWindowPostprocessStage
```

### 7.1 VideoEditWindowValidationStage

输入：`params.runtime_window_frames`、`params.runtime_window_masks`。

输出写入：

- `params.runtime_height`
- `params.runtime_width`
- `params.runtime_num_frames`
- `params.runtime_window_validated`

校验：

- 当前窗口必须正好 81 帧。
- 当前窗口 mask 必须正好 81 帧。
- `num_frames == infer_len == 81`。
- `(infer_len - 1) % 4 == 0`。
- `height`、`width` 可被 16 整除。

### 7.2 VideoEditTextEncodingStage

复用 Wan T5 tokenizer / text encoder，但输出写入 `params`：

- `params.runtime_prompt_embeds`
- `params.runtime_negative_prompt_embeds`
- `params.runtime_do_cfg`

当当前 dynamic CFG 已经降到 `<= 1.0` 时，denoising stage 可以跳过 negative pass，但 text encoding stage 仍可提前准备 negative embeds，避免控制流分散。

### 7.3 VideoEditConditionEncodingStage

职责：

- 使用当前 81 帧窗口构造 masked video。
- window-local frame 0 的 mask 强制全黑。
- 按 reference 规则 pack `cond_masks`。
- VAE encode masked video 得到 `cond_latents`。
- VAE encode raw video 得到 `video_latents`。

输出写入：

- `params.runtime_masked_video_tensor`
- `params.runtime_raw_video_tensor`
- `params.runtime_mask_video_tensor`
- `params.runtime_cond_masks`
- `params.runtime_cond_latents`
- `params.runtime_video_latents`
- `params.runtime_condition_latent`

其中：

```python
params.runtime_condition_latent = torch.cat(
    [params.runtime_cond_masks, params.runtime_cond_latents],
    dim=1,
)
```

不再写 `batch.image_latent` 作为 VideoEdit 的主要上下文来源；如果复用底层 denoising helper 需要该字段，可以在 denoising 前由 stage 从 `params.runtime_condition_latent` 同步一次。

### 7.4 VideoEditLatentPreparationStage

职责：

- 根据 `[B, 16, 21, H/8, W/8]` 生成初始 noise。
- 使用同一 `seed` 或按策略 `seed + window_index`。

输出写入：

- `params.runtime_noise`
- `params.runtime_latents`
- `params.runtime_generator`

此 stage 只准备纯噪声，不做 `add_noise(video_latents)`。

### 7.5 VideoEditTimestepPreparationStage

职责：

- `scheduler.set_timesteps(num_inference_steps, shift=5)`。
- `strength < 1.0` 时调用 `scheduler.get_timesteps()` 裁剪。

输出写入：

- `params.runtime_timesteps`
- `params.runtime_effective_num_inference_steps`
- `params.runtime_num_warmup_steps`

reference 语义是“先生成完整 timesteps，再按 strength 裁剪”，不要把 `strength` 伪装成 `set_timesteps()` 参数。

### 7.6 VideoEditLatentInitStage

职责：

```python
params.runtime_latents = scheduler.add_noise(
    params.runtime_video_latents.to(dtype=torch.float32),
    params.runtime_noise,
    params.runtime_timesteps[:1],
)
```

输出写入：

- `params.runtime_latents`
- `params.runtime_initial_timestep`

### 7.7 VideoEditDenoisingStage

职责：

- 用 VideoEdit 36 通道输入执行 denoising loop。
- 每一步计算 dynamic CFG。
- `current_cfg <= 1.0` 时跳过 negative pass。

核心逻辑：

```python
for i, t in enumerate(params.runtime_timesteps):
    current_cfg, do_cfg = calc_current_cfg(
        max_cfg=params.guidance_scale,
        current_step=i,
        max_step=params.dynamic_cfg_max_step,
        min_cfg=params.dynamic_cfg_min,
        dynamic_cfg=params.dynamic_cfg,
    )

    latent_model_input = torch.cat(
        [
            params.runtime_latents,
            params.runtime_cond_masks,
            params.runtime_cond_latents,
        ],
        dim=1,
    ).to(transformer_dtype)

    noise_pred = transformer(
        hidden_states=latent_model_input,
        timestep=t.expand(params.runtime_latents.shape[0]),
        encoder_hidden_states=params.runtime_prompt_embeds,
        return_dict=False,
    )[0]

    if do_cfg:
        noise_uncond = transformer(
            hidden_states=latent_model_input,
            timestep=t.expand(params.runtime_latents.shape[0]),
            encoder_hidden_states=params.runtime_negative_prompt_embeds,
            return_dict=False,
        )[0]
        noise_pred = noise_uncond + current_cfg * (noise_pred - noise_uncond)

    params.runtime_latents = scheduler.step(
        noise_pred,
        t,
        params.runtime_latents,
    )
```

输出写入：

- `params.runtime_latents`
- `params.runtime_current_step`
- `params.runtime_current_timestep`
- `params.runtime_current_cfg`
- `params.runtime_progress`

MVP 可以先不支持 CFG parallel；如果后续启用，需要保证 `do_cfg=False` 时各 rank 控制流一致。

### 7.8 VideoEditDecodingStage

职责：

- 对 `params.runtime_latents` 做 Wan VAE denormalize + decode。
- 转成窗口 crop-only frames。

输出写入：

- `params.runtime_decoded_video_tensor`
- `params.runtime_window_output_frames`

### 7.9 VideoEditWindowPostprocessStage

职责：

- 只做窗口级轻量整理，不做全局 paste-back。
- 校验窗口输出帧数为 81。
- 可按 `drop_reference_frame` 标记跳过 local 0 的提交，但不在 denoising 或 decode 中隐式丢帧。

输出写入：

- `params.runtime_window_output_frames`
- `params.runtime_window_metadata`

全局 paste-back、overlap 融合、音频拷贝由 `WanVideoEditPipeline.forward()` 的 `_finalize_long_video_output()` 完成。

## 8. `WanVideoEditSamplingParams` 设计

`WanVideoEditSamplingParams` 同时承担两类字段：

- 请求参数：来自 CLI / API，可序列化。
- 运行态字段：stage 之间交换的中间变量，只在单次请求内使用，不写入 API 响应。

示例：

```python
@dataclass
class WanVideoEditSamplingParams(SamplingParams):
    # request fields
    video_input_path: str | None = None
    mask_input_path: str | None = None
    output_path: str | None = None
    output_file_name: str | None = None

    prompt: str | None = None
    negative_prompt: str | None = None
    num_frames: int = 81
    infer_len: int = 81
    overlap: int = 0
    strength: float = 1.0
    num_inference_steps: int = 20
    guidance_scale: float = 5.0
    seed: int = 42
    dtype: str = "bf16"

    dynamic_cfg: bool = True
    dynamic_cfg_max_step: int = 15
    dynamic_cfg_min: float = 1.0

    bbox_padding: int = 0
    dilate_px: int = 15
    mask_scale: float = 1.2
    feather_px: int = 12
    adain_boundary_dilate: int = 15

    enable_paste_back: bool = True
    save_crop_only: bool = False
    drop_reference_frame: bool = True
    keep_intermediate_windows: bool = False
    use_repaired_context: bool = True
    vary_seed_by_window: bool = False

    # global runtime fields
    runtime_original_frames: list[Any] | None = field(default=None, init=False, repr=False)
    runtime_original_masks: list[Any] | None = field(default=None, init=False, repr=False)
    runtime_resized_frames: list[Any] | None = field(default=None, init=False, repr=False)
    runtime_resized_masks: list[Any] | None = field(default=None, init=False, repr=False)
    runtime_dilated_cropped_masks: list[Any] | None = field(default=None, init=False, repr=False)
    runtime_window_specs: list[Any] | None = field(default=None, init=False, repr=False)
    runtime_accum_frames: Any | None = field(default=None, init=False, repr=False)
    runtime_accum_weights: Any | None = field(default=None, init=False, repr=False)
    runtime_bbox: tuple[int, int, int, int] | None = field(default=None, init=False, repr=False)
    runtime_crop_h: int | None = field(default=None, init=False, repr=False)
    runtime_crop_w: int | None = field(default=None, init=False, repr=False)
    runtime_aligned_h: int | None = field(default=None, init=False, repr=False)
    runtime_aligned_w: int | None = field(default=None, init=False, repr=False)
    runtime_fps: float | None = field(default=None, init=False, repr=False)
    runtime_num_input_frames: int | None = field(default=None, init=False, repr=False)

    # per-window runtime fields
    runtime_window_spec: Any | None = field(default=None, init=False, repr=False)
    runtime_window_frames: list[Any] | None = field(default=None, init=False, repr=False)
    runtime_window_masks: list[Any] | None = field(default=None, init=False, repr=False)
    runtime_window_index: int | None = field(default=None, init=False, repr=False)
    runtime_window_validated: bool = field(default=False, init=False, repr=False)

    # stage runtime tensors
    runtime_prompt_embeds: Any | None = field(default=None, init=False, repr=False)
    runtime_negative_prompt_embeds: Any | None = field(default=None, init=False, repr=False)
    runtime_do_cfg: bool = field(default=False, init=False, repr=False)
    runtime_masked_video_tensor: Any | None = field(default=None, init=False, repr=False)
    runtime_raw_video_tensor: Any | None = field(default=None, init=False, repr=False)
    runtime_mask_video_tensor: Any | None = field(default=None, init=False, repr=False)
    runtime_cond_masks: Any | None = field(default=None, init=False, repr=False)
    runtime_cond_latents: Any | None = field(default=None, init=False, repr=False)
    runtime_video_latents: Any | None = field(default=None, init=False, repr=False)
    runtime_condition_latent: Any | None = field(default=None, init=False, repr=False)
    runtime_generator: Any | None = field(default=None, init=False, repr=False)
    runtime_noise: Any | None = field(default=None, init=False, repr=False)
    runtime_latents: Any | None = field(default=None, init=False, repr=False)
    runtime_timesteps: Any | None = field(default=None, init=False, repr=False)
    runtime_effective_num_inference_steps: int | None = field(default=None, init=False, repr=False)
    runtime_num_warmup_steps: int | None = field(default=None, init=False, repr=False)
    runtime_initial_timestep: Any | None = field(default=None, init=False, repr=False)
    runtime_current_step: int | None = field(default=None, init=False, repr=False)
    runtime_current_timestep: Any | None = field(default=None, init=False, repr=False)
    runtime_current_cfg: float | None = field(default=None, init=False, repr=False)
    runtime_decoded_video_tensor: Any | None = field(default=None, init=False, repr=False)
    runtime_window_output_frames: list[Any] | None = field(default=None, init=False, repr=False)
    runtime_window_metadata: dict[str, Any] | None = field(default=None, init=False, repr=False)
    runtime_output_video_path: str | None = field(default=None, init=False, repr=False)
    runtime_crop_video_path: str | None = field(default=None, init=False, repr=False)
    runtime_metadata_path: str | None = field(default=None, init=False, repr=False)

    def reset_window_runtime(self, window_spec: Any) -> None:
        self.runtime_window_spec = window_spec
        self.runtime_window_index = window_spec.window_index
        self.runtime_window_frames = None
        self.runtime_window_masks = None
        self.runtime_window_validated = False
        self.runtime_prompt_embeds = None
        self.runtime_negative_prompt_embeds = None
        self.runtime_masked_video_tensor = None
        self.runtime_raw_video_tensor = None
        self.runtime_mask_video_tensor = None
        self.runtime_cond_masks = None
        self.runtime_cond_latents = None
        self.runtime_video_latents = None
        self.runtime_condition_latent = None
        self.runtime_noise = None
        self.runtime_latents = None
        self.runtime_timesteps = None
        self.runtime_window_output_frames = None
        self.runtime_window_metadata = None
```

校验规则：

- `video_input_path`、`mask_input_path` 必填，除非 API 上传 / URL 下载已落成本地临时路径。
- `num_frames == infer_len == 81` 是单窗口 stage 的硬约束；长视频对外可以任意帧数，但拆给 stage 时必须是 81。
- `(infer_len - 1) % 4 == 0`。
- `0 <= overlap < infer_len`。
- `0 < strength <= 1`。
- `num_outputs_per_prompt == 1`。
- `guidance_scale <= 1.0` 时 negative prompt 可为空。
- `drop_reference_frame` 显式控制是否沿用 reference 保存策略跳过第 0 帧。

## 9. 长视频窗口策略

窗口生成由 `runtime/videoedit/windowing.py` 提供，pipeline forward 调用。stage 不直接接触窗口规划。

固定规则：

- `infer_len = 81`
- `stride = infer_len - overlap`
- 第一个窗口覆盖 `[0, 80]`
- 第二个窗口起点为 `81 - overlap`
- 尾窗口不足 81 帧时用反射序列补齐，不截断到 81 的整数倍

数据结构：

```python
@dataclass
class VideoEditWindowSpec:
    window_index: int
    start_index: int
    end_index: int
    input_indices: list[int]
    commit_local_to_global: dict[int, int]
    reflected_count: int = 0
```

示例：

- `N=81, overlap=0`：一个窗口 `[0..80]`。
- `N=99, overlap=0`：窗口 0 为 `[0..80]`，窗口 1 为 `[81..98,97,96,...]`，只提交真实帧 `[81..98]`。
- `N=99, overlap=8`：窗口 1 从 `73` 开始，重叠 `[73..80]` 用权重融合。
- `N<81`：一个窗口，反射补满 81 帧，提交所有真实帧。

提交规则：

- 每个窗口只提交 `commit_local_to_global` 指定 local frame。
- 反射补齐帧默认不提交，短视频需要覆盖真实帧时例外。
- overlap 区域用线性 ramp 融合。
- `use_repaired_context=True` 时，后一窗口 overlap 输入优先使用已融合结果，因此窗口顺序执行。

全局 bbox：

- MVP 使用全局 mask union bbox，避免每个窗口 crop 不一致导致抖动。
- 若 bbox 面积过小，沿用 reference 的小 bbox 扩张策略。
- 后续可以新增 `bbox_mode="window"`，但不作为首版目标。

## 10. 服务与 CLI 方案

### 10.1 本地 CLI

新增：

```text
python/sglang/multimodal_gen/runtime/videoedit/cli.py
```

命令：

```bash
conda deactivate
source /mnt/shanhai-ai/shanhai-workspace/zhouhao6/env/sglang/bin/activate
cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang

python -m sglang.multimodal_gen.runtime.videoedit.cli repair \
  --model-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model \
  --transformer-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/step-55000-diffusers-lh/transformer \
  --prompt "A close-up of an orange flower with a yellow center, remaining in focus against a blurred green grass background throughout the video." \
  --video-input-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pexel_test_data_0410/videos/15108907_3840_2160_50fps_short.mp4 \
  --mask-input-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pexel_test_data_0410/masks/15108907_3840_2160_50fps_No_bbox_mask.mp4 \
  --output-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs \
  --output-file-name 15108907_3840_2160_50fps.mp4 \
  --num-frames 81 \
  --infer-len 81 \
  --overlap 0 \
  --num-inference-steps 20 \
  --guidance-scale 5.0 \
  --seed 42 \
  --dtype bf16 \
  --enable-paste-back \
  --drop-reference-frame
```

### 10.2 Serve API

新增专用 endpoint：

```text
POST   /v1/videos/repairs
GET    /v1/videos/{video_id}
GET    /v1/videos/{video_id}/progress
GET    /v1/videos/{video_id}/content
DELETE /v1/videos/{video_id}
GET    /health
```

`VideoRepairRequest` 新增在 `protocol.py`：

```python
class VideoRepairRequest(BaseModel):
    task_id: str | None = None
    prompt: str
    negative_prompt: str | None = None

    video_input_path: str | None = None
    mask_input_path: str | None = None
    video_url: str | None = None
    mask_url: str | None = None
    video_bucket: str | None = None
    video_object_key: str | None = None
    mask_bucket: str | None = None
    mask_object_key: str | None = None

    callback_url: str | None = None
    output_storage: str = "local"
    output_path: str | None = None
    output_bucket: str | None = None
    output_object_key: str | None = None

    num_frames: int = 81
    infer_len: int = 81
    overlap: int = 0
    num_inference_steps: int = 20
    guidance_scale: float = 5.0
    seed: int = 42
    dynamic_cfg: bool = True
    dynamic_cfg_max_step: int = 15
    dynamic_cfg_min: float = 1.0
    enable_paste_back: bool = True
    drop_reference_frame: bool = True
```

admission 规则参考 wan_eraser 的 `BoundedSemaphore(1)`：

```python
if active_videoedit_jobs + queued_videoedit_jobs >= queue_capacity:
    raise HTTPException(status_code=429, detail="videoedit_queue_full")
```

服务启动：

```bash
conda deactivate
source /mnt/shanhai-ai/shanhai-workspace/zhouhao6/env/sglang/bin/activate
cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang

VIDEOEDIT_QUEUE_CAPACITY=1 \
sglang serve \
  --model-type diffusion \
  --model-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model \
  --host 0.0.0.0 \
  --port 30000 \
  --tp-size 1 \
  --output-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs \
  --input-save-path /tmp/sglang-videoedit-inputs \
  --transformer-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/step-55000-diffusers-lh/transformer
```

提交任务：

```bash
curl -s -X POST http://127.0.0.1:30000/v1/videos/repairs \
  -H 'Content-Type: application/json' \
  -d '{
    "task_id": "pexel_15108907_first_81",
    "prompt": "A close-up of an orange flower with a yellow center, remaining in focus against a blurred green grass background throughout the video.",
    "video_input_path": "/mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pexel_test_data_0410/videos/15108907_3840_2160_50fps_short.mp4",
    "mask_input_path": "/mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pexel_test_data_0410/masks/15108907_3840_2160_50fps_No_bbox_mask.mp4",
    "output_storage": "local",
    "output_path": "/mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs/15108907_3840_2160_50fps.mp4",
    "num_frames": 81,
    "infer_len": 81,
    "overlap": 0,
    "num_inference_steps": 20,
    "guidance_scale": 5.0,
    "seed": 42,
    "enable_paste_back": true,
    "drop_reference_frame": true
  }'
```

## 11. 端到端测试与逐帧对齐

### 11.1 Reference baseline

```bash
deactivate
conda activate VideoEdit
cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/VideoEdit-diffusers

python infer.py \
  --video_path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pexel_test_data_0410/videos/15108907_3840_2160_50fps_short.mp4 \
  --mask_path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pexel_test_data_0410/masks/15108907_3840_2160_50fps_No_bbox_mask.mp4 \
  --prompt "A close-up of an orange flower with a yellow center, remaining in focus against a blurred green grass background throughout the video." \
  --model_path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model \
  --transformer_path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/wan_converted_step_9500 \
  --output_dir /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs/reference \
  --output_name 15108907_3840_2160_50fps \
  --num_frames 81 \
  --infer_len 81 \
  --num_inference_steps 20 \
  --guidance_scale 5.0 \
  --seed 42 \
  --dtype bf16
```

注意：reference `--transformer_path` 传 overlay 根目录，因为其代码内部再加 `subfolder="transformer"`。

### 11.2 SGLang CLI 端到端

使用第 10.1 节的 `videoedit.cli repair` 命令，输出：

```text
/mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs/15108907_3840_2160_50fps.mp4
```

### 11.3 SGLang Serve 端到端

启动服务、提交 `/v1/videos/repairs`、轮询状态，直到 `completed`。

轮询：

```bash
JOB_ID=video_repair_xxx

while true; do
  resp=$(curl -s "http://127.0.0.1:30000/v1/videos/${JOB_ID}")
  python -c 'import json,sys; d=json.load(sys.stdin); print(d.get("status"), d.get("progress"), d.get("file_path") or d.get("url"))' <<< "$resp"
  status=$(python -c 'import json,sys; print(json.load(sys.stdin).get("status"))' <<< "$resp")
  [ "$status" = "completed" ] && break
  [ "$status" = "failed" ] && exit 1
  sleep 5
done
```

### 11.4 输出检查

```bash
python - <<'PY'
import cv2

path = "/mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs/15108907_3840_2160_50fps.mp4"
cap = cv2.VideoCapture(path)
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()
print({"frames": frames, "width": width, "height": height, "fps": fps})
assert frames in (80, 81), frames
assert width > 0 and height > 0
PY
```

`80` 帧表示显式启用了 `drop_reference_frame`，`81` 帧表示保留完整窗口输出。两种行为都必须由 CLI/API 参数显式控制，不能让 CLI 和 serve 隐式不一致。

### 11.5 Reference 逐帧对齐

集成后的输出必须与原始 `VideoEdit-diffusers/infer.py` 的输出做自动化逐帧对齐。只检查视频能打开、帧数正确还不够；必须验证视觉效果和数值尺度没有大范围偏移。

新增独立模块：

```text
python/sglang/multimodal_gen/runtime/videoedit/compare.py
```

比较口径：

- 逐帧读取 reference mp4 和 SGLang candidate mp4。
- 对每一帧计算 `SSIM`、`MSE`、`MAE`、`PSNR`、`max_abs_diff`。
- 输出全局统计：`ssim_mean`、`ssim_min`、`mse_mean`、`mse_max`、`mae_mean`、`mae_max`、`failed_frames`。
- 任一帧低于阈值即记录到 `failed_frames`；默认允许少量失败帧，用于兼容 H.264/HEVC 编码器以及不同 attention backend / GPU kernel 引入的微小漂移。
- 允许 frame count 有 1 帧差异,用于兼容 reference 默认跳过第 0 帧的保存策略；差异必须通过 `drop_reference_first_frame` 或 `drop_candidate_first_frame` 显式指定。

默认阈值建议（宽松基线，用于发现「质性偏差」而不是 bit-exact 一致性）：

```text
min_ssim = 0.90
max_mse = 150.0
max_mae = 8.0
allow_frame_count_delta = 1
max_failed_frame_ratio = 0.05
```

这些阈值用于发现整帧错误、通道顺序错误、尺度错误、mask packing 错误、窗口提交错位和解码后处理错误，而不要求逐像素一致。视频编码器（H.264/HEVC）的有损压缩、不同 attention backend（FA2/FA3/torch SDPA）、不同 GPU 数值精度、VAE/dtype 差异都可能带来 SSIM 0.93-0.98 量级的小波动，按 0.985 这种严格阈值会大量误报。

如果只关心是否完全跑通，不关心微小漂移，可以使用更宽松的「smoke」阈值：

```text
min_ssim = 0.80
max_mse = 400.0
max_mae = 15.0
max_failed_frame_ratio = 0.10
```

如果是非常稳定的对照（同一台机、同一个 backend、同一个视频编码器），可以收紧成「strict」阈值用于 release gate：

```text
min_ssim = 0.95
max_mse = 60.0
max_mae = 5.0
max_failed_frame_ratio = 0.0
```

无论选择哪一档，都必须保留 `ssim_min` 和 `mse_max` 的上报，便于事后定位回归。

CLI 用法：

```bash
python python/sglang/multimodal_gen/runtime/videoedit/compare.py \
  --reference /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs/reference/15108907_3840_2160_50fps.mp4 \
  --candidate /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs/15108907_3840_2160_50fps.mp4 \
  --report-json /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs/videoedit_compare_report.json \
  --min-ssim 0.90 \
  --max-mse 150.0 \
  --max-mae 8.0 \
  --allow-frame-count-delta 1 \
  --max-failed-frame-ratio 0.05
```

如果 reference 输出是 80 帧、candidate 输出是 81 帧，则必须明确指定丢弃哪一侧的第 0 帧：

```bash
python python/sglang/multimodal_gen/runtime/videoedit/compare.py \
  --reference /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs/reference/15108907_3840_2160_50fps.mp4 \
  --candidate /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs/15108907_3840_2160_50fps.mp4 \
  --drop-candidate-first-frame \
  --report-json /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs/videoedit_compare_report.json
```

退出码语义：

- `0`：逐帧指标全部通过。
- `1`：存在失败帧或全局统计不满足阈值。
- 非 `0/1` 异常：视频打不开、帧数差异超过容忍范围、无帧可比或参数错误。

JSON report 结构：

```json
{
  "summary": {
    "compared_frames": 80,
    "ssim_mean": 0.952,
    "ssim_min": 0.918,
    "mse_mean": 60.4,
    "mse_max": 132.7,
    "mae_mean": 4.6,
    "mae_max": 7.4,
    "psnr_mean": 30.3,
    "max_abs_diff": 64,
    "failed_frames": [],
    "pass_compare": true,
    "thresholds": {
      "min_ssim": 0.90,
      "max_mse": 150.0,
      "max_mae": 8.0,
      "max_failed_frame_ratio": 0.05
    }
  },
  "frames": [
    {
      "index": 0,
      "ssim": 0.948,
      "mse": 65.1,
      "mae": 4.8,
      "psnr": 30.0,
      "max_abs_diff": 58,
      "pass_frame": true
    }
  ]
}
```

CI / nightly 建议：

- PR 阶段：只跑 lightweight unit，确保 comparison 模块能对 synthetic identical / shifted videos 给出正确通过和失败。
- GPU nightly：跑 reference `infer.py`、SGLang CLI、逐帧 compare，保存 JSON report 为 artifact。
- release gate：必须对固定 seed、固定 prompt、固定前 81 帧样例通过逐帧对齐。
- future update：每次更新 DiT、scheduler、VAE、preprocess/postprocess、attention backend 或 windowing，都必须重新生成 compare report。

对齐失败定位顺序：

1. `ssim_min` 很低且 `mse_max` 很高：优先检查帧错位、BGR/RGB 通道、`drop_reference_frame`、视频 resize。
2. 全部帧 MSE 接近常数偏大：优先检查像素归一化、VAE mean/std、decode 后 `[0,1]` 到 `[0,255]` 尺度转换。
3. 只有 mask 区域差异大：优先检查 `cond_masks` preserve/inpaint 语义、mask packing、首帧黑 mask。
4. 越到后面差异越大：优先检查 scheduler timesteps、`add_noise(video_latents)`、dynamic CFG。
5. 每 81 帧边界附近差异大：优先检查 window commit map、overlap 融合和 `use_repaired_context`。

## 12. 实施顺序

1. 新增 `ModelTaskType.VIDEO_EDIT` 和 registry 注册。
2. 新增 `WanVideoEditTransformer3DModel` 和 `WanVideoEditConfig`，保留 SGLang Wan 底层模块，修正 VideoEdit 的 cross-attention context 切分语义。
3. 新增 `WanVideoEditPipelineConfig`，显式设置 `task_type=VIDEO_EDIT`、DiT `in_channels=36`、`out_channels=16`，VAE encoder/decoder 都加载。
4. 新增 `WanVideoEditSamplingParams`，先落请求字段，再落全部 `runtime_*` 中间变量。
5. 新增 `VideoEditFlowMatchScheduler`，对齐 `shift=5`、`sigma_min=0.0`、`extra_one_step=True`、`add_noise()`、`get_timesteps()`。
6. 新增 `runtime/videoedit/preprocess.py`、`postprocess.py`、`windowing.py`、`io.py`、`compare.py`，从 reference 迁移纯函数，不 import 原仓库，并提供逐帧对齐统计。
7. 新增 `videoedit_wan.py` stages，所有中间结果只写 `WanVideoEditSamplingParams`。
8. 新增 `WanVideoEditPipeline.forward`，外层处理窗口，内层循环 stage，每个 stage 固定 81 帧。
9. 新增本地 CLI `repair`，跑通第一条端到端。
10. 新增 serve `/v1/videos/repairs`，实现单任务 admission、后台执行、查询、下载和 `/health`。
11. 跑 reference baseline、SGLang CLI、SGLang serve 三条端到端验收。
12. 运行 `runtime/videoedit/compare.py` 生成逐帧对齐 JSON report，只有 SSIM/MSE/MAE 阈值全部通过才算集成完成。

首版不要做的事：

- 不复用 I2V/TI2V 的 image input 分支。
- 不复制整份通用 Wan Transformer / Wan VAE；DiT 只新增 VideoEdit 必需的差异入口和 cross-attention 适配。
- 不把长视频窗口逻辑放进 stage。
- 不把 stage 中间变量放到 `batch.extra` 或新增一堆 `Req` 顶层字段。
