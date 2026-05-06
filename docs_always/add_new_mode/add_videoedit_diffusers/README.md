# VideoEdit-diffusers 接入 SGLang Diffusion 重构方案

本文档以当前本机源码为准：

- SGLang：`/mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang`
- VideoEdit-diffusers：`/mnt/shanhai-ai/shanhai-workspace/zhouhao6/VideoEdit-diffusers`
- wan_eraser serve 参考：`/mnt/shanhai-ai/shanhai-workspace/zhouhao6/wan_eraser/run_parallel_ray_95_erase.py`
- 模型目录：`/mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model`
- 测试视频：`/mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pexel_test_data_0410/videos/15108907_3840_2160_50fps_short.mp4`
- 测试 mask：`/mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pexel_test_data_0410/masks/15108907_3840_2160_50fps_No_bbox_mask.mp4`

目标不是把 `VideoEdit-diffusers` 原仓库整体搬进 SGLang，而是新增一条原生 `VIDEO_EDIT` 任务链路。除注册点和 enum 这类必要入口外，优先新增文件，尽量不改已有 Wan / T2V / I2V 实现。

## 目录

1. 结论与关键改动
2. Reference 行为拆解
3. 新任务类型 `VIDEO_EDIT`
4. 新增文件与最小修改清单
5. `WanVideoEditPipeline.forward` 重构
6. Stage 设计
7. `WanVideoEditSamplingParams` 设计
8. 长视频窗口策略
9. 服务与 CLI 方案
10. 端到端测试方案
11. 实施顺序

## 1. 结论与关键改动

本次重构采用 skill 中推荐的 Hybrid 思路，但不是把 reference `__call__` 原样复制为一个巨型函数。推荐做法是：

- 新增 `ModelTaskType.VIDEO_EDIT`，不复用 `T2V`，也不伪装成 `I2V/TI2V`。
- 新增 `WanVideoEditPipeline`，重写 `forward`，由 `forward` 做长视频窗口编排和 stage 循环。
- 所有 stage 只处理单个 81 帧窗口；多窗口、反射补齐、窗口融合、paste-back 编排放在 stage 外部。
- 所有 stage 的运行态中间变量统一写入 `WanVideoEditSamplingParams`，不再散落到 `batch.extra`、`Req` 顶层或 helper 私有字段。
- 复用现有 Wan DiT / VAE / text encoder / tokenizer loader；新增 VideoEdit scheduler adapter、stage、preprocess/postprocess 纯函数。
- 删除 dry-run / 冒烟测试方案，验收只保留 reference baseline、SGLang CLI、SGLang serve 三条端到端测试。

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
- `/healthz` 返回进程可用性。
- 任务完成后回调 / 状态更新。

它的 `/generate` 绑定 MinIO key、RLE mask JSON、bbox CSV 和 Ray actor，不适合作为 SGLang VideoEdit 的最终 API。

## 3. 新任务类型 `VIDEO_EDIT`

必须新增任务类型：

```python
class ModelTaskType(Enum):
    I2V = auto()
    T2V = auto()
    TI2V = auto()
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
python/sglang/multimodal_gen/runtime/pipelines/wan_videoedit_pipeline.py
python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/videoedit_wan.py
python/sglang/multimodal_gen/runtime/models/schedulers/videoedit_flow_match.py
python/sglang/multimodal_gen/runtime/videoedit/__init__.py
python/sglang/multimodal_gen/runtime/videoedit/contracts.py
python/sglang/multimodal_gen/runtime/videoedit/preprocess.py
python/sglang/multimodal_gen/runtime/videoedit/postprocess.py
python/sglang/multimodal_gen/runtime/videoedit/windowing.py
python/sglang/multimodal_gen/runtime/videoedit/io.py
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

这些修改只做注册、协议和 endpoint 增量，不改现有 Wan T2V/I2V pipeline 的行为。

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
  "vae": ["sglang", "WanVAE"],
  "transformer": ["sglang", "WanTransformer3DModel"],
  "scheduler": ["sglang", "VideoEditFlowMatchScheduler"]
}
```

`--transformer-path` 仍然作为组件覆盖：

```bash
--transformer-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model/transformer
```

业务参数不要用通用 unknown args 传给 `sglang serve`，避免 `--xxx-path` 被 `ServerArgs._extract_component_paths()` 误解析为组件路径。

## 5. `WanVideoEditPipeline.forward` 重构

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

## 6. Stage 设计

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

### 6.1 VideoEditWindowValidationStage

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

### 6.2 VideoEditTextEncodingStage

复用 Wan T5 tokenizer / text encoder，但输出写入 `params`：

- `params.runtime_prompt_embeds`
- `params.runtime_negative_prompt_embeds`
- `params.runtime_do_cfg`

当当前 dynamic CFG 已经降到 `<= 1.0` 时，denoising stage 可以跳过 negative pass，但 text encoding stage 仍可提前准备 negative embeds，避免控制流分散。

### 6.3 VideoEditConditionEncodingStage

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

### 6.4 VideoEditLatentPreparationStage

职责：

- 根据 `[B, 16, 21, H/8, W/8]` 生成初始 noise。
- 使用同一 `seed` 或按策略 `seed + window_index`。

输出写入：

- `params.runtime_noise`
- `params.runtime_latents`
- `params.runtime_generator`

此 stage 只准备纯噪声，不做 `add_noise(video_latents)`。

### 6.5 VideoEditTimestepPreparationStage

职责：

- `scheduler.set_timesteps(num_inference_steps, shift=5)`。
- `strength < 1.0` 时调用 `scheduler.get_timesteps()` 裁剪。

输出写入：

- `params.runtime_timesteps`
- `params.runtime_effective_num_inference_steps`
- `params.runtime_num_warmup_steps`

reference 语义是“先生成完整 timesteps，再按 strength 裁剪”，不要把 `strength` 伪装成 `set_timesteps()` 参数。

### 6.6 VideoEditLatentInitStage

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

### 6.7 VideoEditDenoisingStage

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

### 6.8 VideoEditDecodingStage

职责：

- 对 `params.runtime_latents` 做 Wan VAE denormalize + decode。
- 转成窗口 crop-only frames。

输出写入：

- `params.runtime_decoded_video_tensor`
- `params.runtime_window_output_frames`

### 6.9 VideoEditWindowPostprocessStage

职责：

- 只做窗口级轻量整理，不做全局 paste-back。
- 校验窗口输出帧数为 81。
- 可按 `drop_reference_frame` 标记跳过 local 0 的提交，但不在 denoising 或 decode 中隐式丢帧。

输出写入：

- `params.runtime_window_output_frames`
- `params.runtime_window_metadata`

全局 paste-back、overlap 融合、音频拷贝由 `WanVideoEditPipeline.forward()` 的 `_finalize_long_video_output()` 完成。

## 7. `WanVideoEditSamplingParams` 设计

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

## 8. 长视频窗口策略

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

## 9. 服务与 CLI 方案

### 9.1 本地 CLI

新增：

```text
python/sglang/multimodal_gen/runtime/videoedit/cli.py
```

命令：

```bash
source /opt/venv/bin/activate
cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang

python -m sglang.multimodal_gen.runtime.videoedit.cli repair \
  --model-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model \
  --transformer-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model/transformer \
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

不提供 `--dry-run` 作为验收路径。参数解析类测试可以有，但文档验收以端到端为准。

### 9.2 Serve API

新增专用 endpoint：

```text
POST   /v1/videos/repairs
GET    /v1/videos/{video_id}
GET    /v1/videos/{video_id}/progress
GET    /v1/videos/{video_id}/content
DELETE /v1/videos/{video_id}
GET    /healthz
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
source /opt/venv/bin/activate
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
  --transformer-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model/transformer
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

## 10. 端到端测试方案

不再写冒烟测试作为验收。端到端测试分三步。

### 10.1 Reference baseline

```bash
source /opt/venv/bin/activate
cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/VideoEdit-diffusers

python infer.py \
  --video_path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pexel_test_data_0410/videos/15108907_3840_2160_50fps_short.mp4 \
  --mask_path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pexel_test_data_0410/masks/15108907_3840_2160_50fps_No_bbox_mask.mp4 \
  --prompt "A close-up of an orange flower with a yellow center, remaining in focus against a blurred green grass background throughout the video." \
  --model_path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model \
  --transformer_path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model \
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

### 10.2 SGLang CLI 端到端

使用第 9.1 节的 `videoedit.cli repair` 命令，输出：

```text
/mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs/15108907_3840_2160_50fps.mp4
```

### 10.3 SGLang Serve 端到端

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

### 10.4 输出检查

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

## 11. 实施顺序

1. 新增 `ModelTaskType.VIDEO_EDIT` 和 registry 注册。
2. 新增 `WanVideoEditPipelineConfig`，显式设置 `task_type=VIDEO_EDIT`、DiT `in_channels=36`、`out_channels=16`，VAE encoder/decoder 都加载。
3. 新增 `WanVideoEditSamplingParams`，先落请求字段，再落全部 `runtime_*` 中间变量。
4. 新增 `VideoEditFlowMatchScheduler`，对齐 `shift=5`、`sigma_min=0.0`、`extra_one_step=True`、`add_noise()`、`get_timesteps()`。
5. 新增 `runtime/videoedit/preprocess.py`、`postprocess.py`、`windowing.py`、`io.py`，从 reference 迁移纯函数，不 import 原仓库。
6. 新增 `videoedit_wan.py` stages，所有中间结果只写 `WanVideoEditSamplingParams`。
7. 新增 `WanVideoEditPipeline.forward`，外层处理窗口，内层循环 stage，每个 stage 固定 81 帧。
8. 新增本地 CLI `repair`，跑通第一条端到端。
9. 新增 serve `/v1/videos/repairs`，实现单任务 admission、后台执行、查询、下载和 `/healthz`。
10. 跑 reference baseline、SGLang CLI、SGLang serve 三条端到端验收。

首版不要做的事：

- 不复用 `T2V` task_type。
- 不复用 I2V/TI2V 的 image input 分支。
- 不复制 Wan Transformer / Wan VAE。
- 不把长视频窗口逻辑放进 stage。
- 不把 stage 中间变量放到 `batch.extra` 或新增一堆 `Req` 顶层字段。
- 不用 dry-run / smoke test 代替端到端验收。
