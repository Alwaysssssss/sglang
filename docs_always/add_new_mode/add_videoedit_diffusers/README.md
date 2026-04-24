# VideoEdit-diffusers 接入 SGLang Diffusion 方案

## 目标

将 `../VideoEdit-diffusers` 中的 Wan2.1-I2V 视频编辑 / inpainting 模型接入 `python/sglang/multimodal_gen`，先做原生 SGLang Diffusion pipeline 方案设计，后续再按方案实现。

参考 skill：`python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-add-model/SKILL.md`。

## 参考实现结论

`../VideoEdit-diffusers` 是一个基于 Wan2.1-I2V 的推理仓库，核心文件如下：

- `pipelines/pipeline_wan_edit.py`：Diffusers 风格 pipeline，核心 `__call__`。
- `infer.py`：端到端 CLI，包括视频 / mask 预处理、滑窗推理、VAE encode/decode、paste-back。
- `models/transformer_wan.py`：Wan DiT，默认 `in_channels=36`、`out_channels=16`。
- `models/autoencoder_kl_wan.py`：Wan 3D causal VAE。
- `models/flow_match.py`：简单 FlowMatch scheduler。
- `utils/preprocess.py` / `utils/postprocess.py`：视频编辑专属预处理和回贴后处理。

核心差异不在 Wan 主干结构，而在 DiT 输入构造：

```python
latent_model_input = torch.cat([latents, cond_masks, cond_latents], dim=1)
```

其中：

- `latents`：当前噪声 latent，形状 `[B, 16, F_lat, H/8, W/8]`。
- `cond_masks`：由 mask video 下采样和 temporal packing 得到，形状 `[B, 4, F_lat, H/8, W/8]`。
- `cond_latents`：masked video 经 Wan VAE encode 后的 latent，形状 `[B, 16, F_lat, H/8, W/8]`。
- 拼接后输入通道数为 `36`，DiT 输出仍为 `16` 通道 noise prediction。

参考实现还包含三点需要对齐：

- scheduler 使用 `FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)`。
- 支持 `video_latents` 初始化：`scheduler.add_noise(video_latents, noise, first_timestep)`。
- 默认启用 dynamic CFG：前若干步 guidance scale 从 `guidance_scale` 逐步衰减到 `1.0`。

## 接入策略

采用 skill 中的“优先复用现有组件，再新增模型专属 Stage”的策略。

不新写 Wan DiT / VAE 主体。SGLang 已有：

- `runtime/models/dits/wanvideo.py`
- `runtime/models/vaes/wanvae.py`
- `configs/models/dits/wanvideo.py`
- `configs/models/vaes/wanvae.py`
- `configs/pipeline_configs/wan.py`
- `configs/sample/wan.py`

VideoEdit 应作为 Wan 家族的一个编辑变体接入：

```text
标准文本编码
  -> VideoEdit 专属视频/mask 条件准备
  -> 标准 latent/timestep 准备
  -> VideoEdit 专属 denoising 或标准 DenoisingStage 扩展
  -> 标准 Wan VAE decoding
```

原因：

- 文本编码、Wan VAE、Wan Transformer、TP/SP、attention backend 都能复用。
- 条件输入、mask packing、滑窗、paste-back 是 VideoEdit 独有，不应污染通用 Wan pipeline。
- 如果只追求最小 MVP，也必须先解决 scheduler 与通用 Stage 的接口兼容问题；之后可把 `batch.image_latent = torch.cat([cond_masks, cond_latents], dim=1)`，并用独立 `VideoEditLatentInitStage` 处理 `video_latents` 加噪，再临时复用标准 `DenoisingStage`。为了和参考实现完全对齐，仍建议新增 `VideoEditDenoisingStage` 处理 dynamic CFG 和 `video_latents` 初始化。

## 模型目录与 pipeline 选择

需要避免直接传入基础 `Wan2.1-I2V-14B-480P-Diffusers` 时被解析到现有 `WanImageToVideoPipeline`。

推荐方案是提供一个 VideoEdit 的 diffusers-style wrapper / overlay 模型目录：

```text
VideoEdit-diffusers-model/
  model_index.json                 # _class_name = "WanVideoEditPipeline"
  tokenizer/                       # 来自基础 Wan2.1-I2V
  text_encoder/                    # 来自基础 Wan2.1-I2V
  vae/                             # 来自基础 Wan2.1-I2V
  transformer/                     # VideoEdit finetuned transformer
  scheduler/                       # 可沿用占位配置，运行时替换成 VideoEditFlowMatchScheduler
```

也可以支持运行时覆盖：

```bash
sglang generate \
  --model-path /path/to/VideoEdit-diffusers-model \
  --transformer-path /path/to/finetuned_transformer \
  --prompt "..." \
  --video-path /path/to/input.mp4 \
  --mask-path /path/to/mask.mp4
```

`--transformer-path` 已能通过 `ServerArgs.component_paths` 被解析为 `component_paths["transformer"]`。

## 拟新增 / 修改文件

### 1. Pipeline

新增：

`python/sglang/multimodal_gen/runtime/pipelines/wan_videoedit_pipeline.py`

职责：

- 定义 `WanVideoEditPipeline(LoRAPipeline, ComposedPipelineBase)`。
- `pipeline_name = "WanVideoEditPipeline"`，必须和 wrapper `model_index.json` 的 `_class_name` 一致。
- `_required_config_modules = ["text_encoder", "tokenizer", "vae", "transformer", "scheduler"]`。
- `initialize_pipeline()` 中使用与通用 Stage 兼容的 VideoEdit scheduler adapter：

```python
from sglang.multimodal_gen.runtime.models.schedulers.videoedit_flow_match import (
    VideoEditFlowMatchScheduler,
)

self.modules["scheduler"] = VideoEditFlowMatchScheduler(
    shift=server_args.pipeline_config.flow_shift,
    sigma_min=0.0,
    extra_one_step=True,
)
```

配套新增：

`python/sglang/multimodal_gen/runtime/models/schedulers/videoedit_flow_match.py`

职责：

- 对齐 `../VideoEdit-diffusers/models/flow_match.py` 的 sigma/timestep 公式。
- 兼容 SGLang 通用 Stage API：接受 `device=`，提供 `set_begin_index()`，并在 `return_dict=False` 时让 `step()` 返回 `(prev_sample,)`。
- 保留 `add_noise(original_samples, noise, timestep)`，用于 `video_latents` 初始化。

- `create_pipeline_stages()` 串接：
  - `InputValidationStage`
  - `TextEncodingStage`
  - `VideoEditConditionStage`
  - `LatentPreparationStage`
  - `TimestepPreparationStage`
  - `VideoEditLatentInitStage` 或 `VideoEditDenoisingStage` 内部处理
  - `VideoEditDenoisingStage`
  - `DecodingStage`

### 2. 模型专属 Stage

新增：

`python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/videoedit_wan.py`

建议拆为两个 Stage：

- `VideoEditConditionStage`
  - 读取 `batch.video_path`、`batch.mask_path`。
  - 复用 / 改写 `VideoEdit-diffusers/utils/preprocess.py` 逻辑。
  - 生成 `masked_video_tensor`、`raw_video_tensor`、`cond_masks`。
  - 用 Wan VAE encode 得到 `cond_latents`。
  - 可选 encode `raw_video_tensor` 得到 `video_latents`。
  - 设置 `batch.image_latent = torch.cat([cond_masks, cond_latents], dim=1)`。
  - 把 paste-back 所需的 `bbox`、`crop_h`、`crop_w`、`fps`、原始帧等元数据放入 `batch.extra["videoedit"]`。

- `VideoEditLatentInitStage`
  - 在 `LatentPreparationStage` 和 `TimestepPreparationStage` 后执行。
  - 如果存在 `batch.extra["videoedit"]["video_latents"]`，按参考实现执行：

```python
batch.latents = scheduler.add_noise(video_latents, batch.latents, batch.timesteps[:1])
```

如果实现 `VideoEditDenoisingStage`，该初始化也可以放进 denoising stage 的 `_before_denoising_loop()`。

### 3. VideoEditDenoisingStage

新增或扩展：

`python/sglang/multimodal_gen/runtime/pipelines_core/stages/videoedit_denoising.py`

推荐继承 `DenoisingStage`，只覆盖必要 hook：

- 复用标准 `_prepare_denoising_loop()`、CFG 并行、SP 处理、scheduler.step；前提是 scheduler adapter 已兼容 `TimestepPreparationStage` 和 `DenoisingStage` 的调用约定。
- 覆盖 `_run_denoising_step()` 或 `_select_and_manage_model()`，在每步计算参考实现的 dynamic CFG：

```python
current_cfg, do_cfg = calc_current_cfg(
    max_cfg=batch.guidance_scale,
    current_step=step_index,
    max_step=batch.dynamic_cfg_max_step,
    min_cfg=batch.dynamic_cfg_min,
    dynamic_cfg=batch.dynamic_cfg,
)
```

- `latent_model_input` 保持标准逻辑：`ctx.latents` 与 `batch.image_latent` 拼接，得到 36 通道输入。

MVP 阶段可以先关闭 dynamic CFG，并在 scheduler adapter 与 `VideoEditLatentInitStage` 完成后使用标准 `DenoisingStage`。但准确性验收必须补齐 dynamic CFG。

### 4. PipelineConfig

新增：

`python/sglang/multimodal_gen/configs/pipeline_configs/videoedit_wan.py`

建议定义：

```python
@dataclass
class WanVideoEditPipelineConfig(WanI2V480PConfig):
    task_type: ModelTaskType = ModelTaskType.TI2V
    flow_shift: float | None = 5.0
    vae_precision: str = "bf16"
```

需要关注：

- `dit_config` 仍使用 `WanVideoConfig`，由 transformer `config.json` 覆盖到 `in_channels=36`。
- `vae_config` 仍使用 `WanVAEConfig`，保持 Wan latent mean/std。
- `postprocess_image_latent()` 不走通用 I2V 第一帧逻辑，VideoEdit 条件 Stage 直接构造 `batch.image_latent`。
- 如需要 postprocess 输出视频并回贴，可在 `post_decoding()` 里根据 `batch.extra["videoedit"]` 做 paste-back；也可以先放在 CLI/helper 层。

### 5. SamplingParams

新增：

`python/sglang/multimodal_gen/configs/sample/videoedit_wan.py`

建议字段：

```python
@dataclass
class WanVideoEditSamplingParams(SamplingParams):
    height: int = 480
    width: int = 832
    num_frames: int = 81
    fps: int = 16
    num_inference_steps: int = 20
    guidance_scale: float = 5.0
    seed: int = 42
    negative_prompt: str | None = "..."

    video_path: str | None = None
    mask_path: str | None = None
    infer_len: int = 81
    strength: float = 1.0
    dynamic_cfg: bool = True
    dynamic_cfg_max_step: int = 15
    dynamic_cfg_min: float = 1.0

    bbox_padding: int = 0
    dilate_px: int = 15
    mask_scale: float = 1.2
    feather_px: int = 12
```

`num_frames` 必须满足 Wan VAE 约束：`(num_frames - 1) % 4 == 0`。

同时需要扩展 CLI 参数注册，把 `--video-path`、`--mask-path`、`--infer-len`、`--dynamic-cfg` 等加入 `SamplingParams.add_cli_args()` 或 VideoEdit 专属 CLI wrapper。否则这些参数会出现在 `parse_known_args()` 的 unknown args 中，而 `ServerArgs._extract_component_paths()` 会把任何 `--<name>-path` 形式误解析成组件权重覆盖，例如 `--video-path` 会变成 `component_paths["video"]`。

如果暂时不改通用 CLI parser，MVP 应提供专用 wrapper CLI 或 Python API 直接构造 `WanVideoEditSamplingParams`。JSON/YAML config 只有在 `generate_cmd` 改为按 `model_info.sampling_param_cls` 提取字段后才可靠；当前逻辑只识别基础 `SamplingParams` 字段。

### 6. 注册

修改：

- `python/sglang/multimodal_gen/configs/pipeline_configs/__init__.py`
- `python/sglang/multimodal_gen/configs/sample/__init__.py` 如该目录维护导出。
- `python/sglang/multimodal_gen/registry.py`

在 `registry.py` 中新增：

```python
register_configs(
    sampling_param_cls=WanVideoEditSamplingParams,
    pipeline_config_cls=WanVideoEditPipelineConfig,
    hf_model_paths=[
        "VideoEdit-diffusers",
        "Wan2.1-VideoEdit-Diffusers",
    ],
    model_detectors=[lambda hf_id: "videoedit" in hf_id.lower()],
)
```

实际 pipeline class 仍由 `model_index.json` 的 `_class_name = "WanVideoEditPipeline"` 触发。

## 预处理与长视频策略

分两阶段实现。

第一阶段只支持单窗口：

- 输入一段长度为 `infer_len` 的视频和 mask。
- 复用参考实现的 crop、mask dilation、mask scale、对齐到 16。
- 直接输出 crop 区域编辑后的视频。

第二阶段补齐长视频：

- 把 `infer.py` 的 global preprocess、sliding window、paste-back 迁移为 CLI/helper。
- 每个窗口独立调用 native pipeline。
- 保存完整回贴结果和 crop-only 结果。

这样可以先验证核心 DiT/VAE/scheduler 对齐，避免把窗口调度和后处理问题混入第一轮准确性调试。

## 准确性验收

必须和 `../VideoEdit-diffusers` 做同 seed、同 prompt、同窗口输入的对齐。

建议逐层比较：

1. 文本编码：`prompt_embeds`、`negative_prompt_embeds` 形状和均值范围。
2. VAE encode：`cond_latents`、`video_latents` 的 shape、mean/std、dtype。
3. mask packing：`cond_masks` 是否为 `[B, 4, F_lat, H/8, W/8]`，且取值语义为 preserve=1、inpaint=0。
4. scheduler：`timesteps` / `sigmas` 与参考实现完全一致。
5. 首步 DiT：同输入下 `noise_pred` 误差在合理范围内。
6. 最终 latent：同 seed 输出不应是噪声。
7. 解码视频：视觉效果与参考实现一致；首帧跳过、paste-back 边界无明显错误。

## 主要风险

- pipeline 选择风险：如果没有 VideoEdit wrapper `model_index.json`，会被解析到已有 `WanImageToVideoPipeline`。
- scheduler 风险：现有 Wan pipeline 用 `FlowUniPCMultistepScheduler`，VideoEdit 必须用参考实现的简单 `FlowMatchScheduler`。
- dynamic CFG 风险：标准 `DenoisingStage` 当前按固定 guidance scale 设计；完全对齐需要新增 VideoEdit denoising hook。
- mask 语义风险：参考实现中 mask 会取反，必须保持 preserve=1、inpaint=0。
- VAE 归一化风险：Wan latent 必须使用 `latents_mean` / `latents_std`，否则输出容易变成噪声。
- 长视频风险：滑窗和 paste-back 属于应用层逻辑，建议晚于单窗口准确性验收实现。

## 风险解决方案

### P0：先解除会直接阻断运行的风险

| 风险 | 解决方案 | 验收方式 |
| --- | --- | --- |
| pipeline 选错 | 必须准备 VideoEdit wrapper 模型目录，并让 `model_index.json` 的 `_class_name` 固定为 `WanVideoEditPipeline`。同时注册 `WanVideoEditPipeline` 的 `EntryClass`、`WanVideoEditPipelineConfig` 和 `WanVideoEditSamplingParams`。本地调试可额外支持 `--pipeline-class-name WanVideoEditPipeline` 和 `--model-id VideoEdit-diffusers` 作为兜底。 | 单测调用 `get_model_info(videoedit_model_path, backend="sglang")`，断言返回的 pipeline 是 `WanVideoEditPipeline`，且传基础 Wan I2V 目录不会误进 VideoEdit。 |
| `FlowMatchScheduler` 和通用 Stage API 不兼容 | 不要直接把当前 `flow_match_pair.FlowMatchScheduler` 塞进标准 `TimestepPreparationStage` / `DenoisingStage` 后期望可用。当前通用 Stage 会传 `device=`、调用 `set_begin_index()`，并假设 `scheduler.step(..., return_dict=False)[0]`；而 `FlowMatchScheduler` 没有这些兼容接口。应新增 `VideoEditFlowMatchScheduler` 适配层，或扩展现有 `FlowMatchScheduler`：`set_timesteps(..., device=None, **kwargs)`、`set_begin_index()` no-op、`step(..., return_dict=False)` 返回 `(prev_sample,)`。 | 单测逐项比较 SGLang scheduler 与 `../VideoEdit-diffusers/models/flow_match.py` 的 `sigmas`、`timesteps`、`add_noise()`、`step()`，误差在 dtype 允许范围内。 |
| `--video-path` / `--mask-path` 被误解析 | 当前 CLI 只注册基础 `SamplingParams`，未知的 `--<name>-path` 会被 `ServerArgs._extract_component_paths()` 当作组件权重覆盖，`--video-path` 会变成 `component_paths["video"]`。MVP 用专用 wrapper CLI 或 Python API 传参；正式方案要让 generate CLI 先根据 `--model-path` 解析模型，再注册模型专属 `SamplingParams.add_cli_args()`，或把 `video_path` / `mask_path` 加入通用 `SamplingParams`。 | 加 CLI 单测：`--video-path a.mp4 --mask-path m.mp4 --transformer-path ckpt` 后，`video_path/mask_path` 进入 sampling params，只有 `transformer_path` 进入 `component_paths["transformer"]`。 |
| transformer 通道数不匹配 | VideoEdit transformer 必须是 `in_channels=36, out_channels=16`。在 pipeline 初始化或 condition stage 之后做 fail-fast 校验，禁止使用基础 Wan I2V transformer 静默跑。wrapper 目录中 `transformer/config.json` 应来自 finetuned 权重；运行时覆盖时只允许 `--transformer-path` 指向 VideoEdit finetuned transformer。 | 加启动前校验：`server_args.pipeline_config.dit_config.in_channels == 36`、`out_channels == 16`；错误时提示使用 VideoEdit finetuned transformer。 |

### P1：保证单窗口结果正确

| 风险 | 解决方案 | 验收方式 |
| --- | --- | --- |
| 条件构造错误 | 新增 `VideoEditConditionStage`，从 `batch.video_path`、`batch.mask_path` 读取输入，先只支持单窗口；移植 `prepare_window_inputs()` 中的 tensor 构造逻辑，生成 `cond_masks`、`cond_latents`、`video_latents`。设置 `batch.image_latent = torch.cat([cond_masks, cond_latents], dim=1)`，并把 `video_latents` 放入 `batch.extra["videoedit"]`。 | 断言 `latents=[B,16,F_lat,H/8,W/8]`、`cond_masks=[B,4,F_lat,H/8,W/8]`、`cond_latents=[B,16,F_lat,H/8,W/8]`、拼接后 20 通道；denoising 时与当前噪声 latent 拼接后正好 36 通道。 |
| mask 语义反了 | 严格复用参考实现：window-local 第 0 帧 mask 强制全黑；`first_frame_mask = mask_video_tensor[0:1].repeat(4, ...)`；下采样后执行 `(cond_masks < 0.5).float()`，保证 preserve=1、inpaint=0。 | 用一段 synthetic mask 做单测，检查白色编辑区域进入模型前为 0，黑色保留区域为 1，第 0 帧全 preserve。 |
| VAE encode/decode 归一化错误 | condition 和 raw video VAE encode 使用 `argmax/mode()`，然后按 Wan `latents_mean`、`latents_std` 做 `(latent - mean) / std`；decode 继续复用 `DecodingStage.scale_and_shift()`，避免重复手写反归一化。`WanVideoEditPipelineConfig.__post_init__()` 必须设置 `vae_config.load_encoder = True` 且 `load_decoder = True`。 | 对同一窗口比较参考实现与 SGLang 的 `cond_latents`、`video_latents` shape、dtype、mean/std 和最大误差。 |
| `video_latents` 加噪时机错误 | 在 `LatentPreparationStage` 和 `TimestepPreparationStage` 之后、SP 分片之前执行初始化：`batch.latents = scheduler.add_noise(video_latents, noise, batch.timesteps[:1])`。如果复用 `DenoisingStage`，放入 `VideoEditDenoisingStage._before_denoising_loop()`；如果独立 Stage，则放在 denoising 前。 | 同 seed 比较参考实现首步输入 latent；开启 SP 时确认 `video_latents` 未被重复分片或漏分片。 |
| dynamic CFG 不一致 | 新增 `VideoEditDenoisingStage`，继承 `DenoisingStage`，只覆盖必要 hook。每步按参考实现 `calc_current_cfg()` 计算 `current_guidance_scale` 和 `do_cfg`；当 `do_cfg=False` 时跳过 negative pass。不要在通用 `DenoisingStage` 里硬编码 VideoEdit 逻辑。 | 单测检查 `guidance_scale=5, dynamic_cfg_max_step=15, dynamic_cfg_min=1` 时每步 CFG 序列与参考实现一致；再做首步 DiT `noise_pred` 对齐。 |

### P2：降低集成和维护风险

| 风险 | 解决方案 | 验收方式 |
| --- | --- | --- |
| 长视频逻辑污染 native pipeline | native pipeline 只负责单窗口 VideoEdit。滑窗、paste-back、跳过首帧保存等放到 CLI/helper 层，第二阶段再迁移 `prepare_global_inputs()`、`paste_back()`。 | 单窗口测试先通过；长视频 helper 对齐参考 `infer.py`，输出 crop-only 和 paste-back 两个结果。 |
| 输入参数和 frame 约束漂移 | `WanVideoEditSamplingParams` 中显式校验 `video_path`、`mask_path` 必填，`num_frames == infer_len`，且 `(num_frames - 1) % 4 == 0`。如果输入视频不足一个窗口，直接报错或由 wrapper 明确补帧策略。 | 参数单测覆盖缺 video/mask、非法 `infer_len`、视频和 mask 帧数不一致。 |
| SP / CFG parallel 引入隐性差异 | MVP 建议先固定 `num_gpus=1`、关闭 CFG parallel，完成准确性对齐；随后再开启 SP，依赖通用 `shard_latents_for_sp()` 同步切分 `latents` 和 `batch.image_latent`。dynamic CFG 支持 CFG parallel 前，要验证 `do_cfg=False` 的 rank 行为。 | 先单卡逐层对齐；再多卡只比较 shape、无报错、输出非噪声；最后做多卡一致性阈值测试。 |
| 回归定位困难 | 增加 side-by-side 脚本，固定 seed、prompt、窗口输入，逐层 dump `prompt_embeds`、`cond_masks`、`cond_latents`、`video_latents`、`timesteps/sigmas`、首步 `noise_pred`、最终 latent。 | CI 中至少跑轻量 shape/参数单测；人工准确性验收用 side-by-side 脚本，不只看最终视频。 |

## 实施顺序

1. 准备 VideoEdit wrapper 模型目录，确认 `_class_name = "WanVideoEditPipeline"`，transformer 指向 finetuned 权重。
2. 新增 `WanVideoEditSamplingParams` 和 `WanVideoEditPipelineConfig`，补齐 `video_path`、`mask_path`、`infer_len` 等校验。
3. 新增 `VideoEditFlowMatchScheduler` adapter，先通过 scheduler 对齐单测。
4. 新增 `WanVideoEditPipeline`，串标准 text / latent / timestep / decode，并使用 VideoEdit scheduler adapter。
5. 新增 `VideoEditConditionStage`，完成 video/mask 到 `batch.image_latent`、`video_latents` 的构造。
6. 新增 `VideoEditLatentInitStage`，先关闭 dynamic CFG，用标准 `DenoisingStage` 跑通单窗口，确认非噪声输出。
7. 新增 `VideoEditDenoisingStage`，补齐 dynamic CFG 和 `do_cfg=False` 时跳过 negative pass。
8. 注册模型并补 CLI 方案；MVP 用专用 wrapper，正式方案再改通用 generate CLI。
9. 加单元测试和 Diffusers side-by-side 准确性脚本。
10. 再迁移滑窗、paste-back 和保存完整视频的应用层逻辑。
