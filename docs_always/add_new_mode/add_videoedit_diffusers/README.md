# VideoEdit-diffusers 接入 SGLang Diffusion 方案

> 运行环境：source /opt/venv/bin/activate
> 算法实现参考库：/root/zhouhao6/VideoEdit-diffusers
> serve实现：/root/zhouhao6/wan_eraser/run_parallel_ray_95_erase.py
> 模型路径：/root/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model
> 输入视频：/root/zhouhao6/video_diffusers/pexel_test_data_0410/videos/15108907_3840_2160_50fps_short.mp4
> mask视频：/root/zhouhao6/video_diffusers/pexel_test_data_0410/masks/15108907_3840_2160_50fps_No_bbox_mask.mp4
> 提示词："A close-up of an orange flower with a yellow center, remaining in focus against a blurred green grass background throughout the video."

## 1. 背景与目标

目标是把 `../VideoEdit-diffusers` 中基于 Wan2.1 的视频编辑 / inpainting 模型接入 `python/sglang/multimodal_gen`，但集成后的 SGLang 实现必须满足以下约束：

1. 运行时不依赖原 `VideoEdit-diffusers` 仓库的目录、工具函数、私有数据结构和脚本调用。
2. 优先复用 SGLang 已有的 Wan VAE、Wan DiT、通用 pipeline/stage、分布式和解码能力，只补 VideoEdit 专属的数据组装、scheduler 适配和 denoising hook。
3. 预处理、模型推理、后处理三层解耦，后续无论是 SGLang 升级还是 VideoEdit upstream 更新，都能局部同步而不是整体重写。
4. 文档中明确接口层级、数据流、模块边界和 upstream 对齐方式，便于维护和自动化回归。

参考 skill：
`python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-add-model/SKILL.md`

本方案先定义原生 SGLang pipeline 的集成设计，后续实现按该设计推进。

### 1.1 本次方案审查结论

对照当前 `python/sglang/multimodal_gen` 代码和 `../VideoEdit-diffusers` reference 后，原方案的主方向是正确的：不能复制 Wan 主干，应该把 VideoEdit 收敛为 Wan family 的编辑变体。但需要补齐以下落地细节，否则实现时会出现接口不对齐或边界行为偏差：

- `strength < 1.0` 不是单纯 scheduler 参数问题。reference 先 `set_timesteps()`，再用 `get_timesteps()` 裁剪 timestep 序列，同时更新有效 `num_inference_steps`。因此需要新增一个很薄的 `VideoEditTimestepPreparationStage`，不能只复用通用 `TimestepPreparationStage`。
- `WanVideoEditPipelineConfig` 必须显式把 DiT 输入通道配置为 `36`，输出 latent 通道保持 `16`。否则 `LatentPreparationStage` 和 transformer 权重加载可能会按默认 Wan T2V 的 `16` 通道契约工作。
- pipeline 文件需要声明 `pipeline_config_cls` / `sampling_params_cls`，并暴露 `EntryClass = WanVideoEditPipeline`，让自动发现和 safetensors / local overlay 路径都能稳定解析。
- `VideoEditDenoisingStage` 不应重写整段 denoising loop。更稳妥的落点是覆盖 `_predict_noise_with_cfg()`，在其中计算 dynamic CFG，并在 `current_cfg <= 1.0` 时跳过 negative pass。
- 长视频窗口应按固定 81 帧窗口和 `overlap` 参数规划：默认 `[0,80]`、`[81,161]`，启用 overlap 后第二个窗口为 `[81-overlap,161-overlap]`。不足 81 帧或尾窗口越界时用反射序列逆向补齐，不再截断到 81 的整数倍。
- mask packing 的首帧规则需要更精确：reference 中先把第 0 帧 mask repeat 4 次，再拼接第 1..80 帧，形成 `84` 个 mask frame，最后 reshape 为 `[B, 4, 21, H/8, W/8]`。

## 2. 参考实现拆解

`../VideoEdit-diffusers` 的核心文件如下：

- `pipelines/pipeline_wan_edit.py`
- `infer.py`
- `models/transformer_wan.py`
- `models/autoencoder_kl_wan.py`
- `models/flow_match.py`
- `utils/preprocess.py`
- `utils/postprocess.py`

结论很明确：VideoEdit 的差异不在 Wan 主干，而在输入条件构造和调度逻辑。

核心模型输入为：

```python
latent_model_input = torch.cat([latents, cond_masks, cond_latents], dim=1)
```

各项语义：

- `latents`: 当前噪声 latent，`[B, 16, F_lat, H/8, W/8]`
- `cond_masks`: 由 mask video 下采样和时域 packing 得到，`[B, 4, F_lat, H/8, W/8]`
- `cond_latents`: masked video 经 Wan VAE 编码后的 latent，`[B, 16, F_lat, H/8, W/8]`
- 拼接后 DiT 输入通道数为 `36`，输出仍为 `16`

此外还必须对齐三类行为：

- scheduler 使用 `FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)`
- 支持 `video_latents` 初始化：`scheduler.add_noise(video_latents, noise, first_timestep)`
- dynamic CFG：前若干步 guidance scale 从 `guidance_scale` 衰减到 `1.0`

因此，SGLang 的接入不应复制一份 Wan pipeline，而应把 VideoEdit 视为“Wan family 的编辑变体”。

## 3. 采用的接入风格

按照 skill 的原则，本模型不适合重写一条 monolithic pipeline，也不适合硬塞进现有 I2V/TI2V 图像输入链路。推荐采用：

- 以 SGLang 现有 Wan pipeline 为骨架
- 以标准 `TextEncodingStage` / `LatentPreparationStage` / `DecodingStage` 为主体
- 仅新增 4 个 VideoEdit 专属扩展点：
  - `VideoEditConditionStage`
  - `VideoEditTimestepPreparationStage`
  - `VideoEditLatentInitStage`
  - `VideoEditDenoisingStage`

推荐 stage 链路：

```text
InputValidationStage
  -> TextEncodingStage
  -> VideoEditConditionStage
  -> LatentPreparationStage
  -> VideoEditTimestepPreparationStage
  -> VideoEditLatentInitStage
  -> VideoEditDenoisingStage
  -> DecodingStage
  -> optional postprocess/helper
```

这条链路满足两件事：

- Wan 通用能力保持完全复用
- VideoEdit 专属逻辑被压缩在条件准备、timestep 裁剪、首步 latent 初始化和 CFG 计算这四个局部模块中

## 4. 为什么不能直接复用 I2V/TI2V 语义

这是本次方案里最重要的边界设计。

VideoEdit 的业务输入是：

- `video_input_path`
- `mask_input_path`
- 可选长视频滑窗参数

它不是现有 SGLang 里的：

- `image_path`
- `condition_image`
- `TI2V` 第一帧条件图

如果把 VideoEdit 伪装成 `I2V` 或 `TI2V`，会引入三个问题：

1. `SamplingParams._validate_with_pipeline_config()` 会对 `image_path` 施加错误约束。
2. `InputValidationStage` 会走通用 condition image resize / crop 分支，语义不对。
3. `DenoisingStage` 对 `TI2V` 有专门分支，且默认假设 `batch.image_latent is None`，与 VideoEdit 的 20 通道条件 latent 设计冲突。

因此建议：

- `WanVideoEditPipelineConfig.task_type` 不复用 `I2V/TI2V`
- 直接沿用 `T2V` 作为“输出类型是视频”的基础任务类型
- VideoEdit 的输入约束全部由 `WanVideoEditSamplingParams` 和 `VideoEditConditionStage` 负责

换言之，`task_type` 只表达“输出是什么”，不再错误地表达“输入长什么样”。

这是后续保持低耦合的关键。

## 5. 总体架构

### 5.1 分层原则

整体分为 4 层：

1. 模型层
   - Wan VAE
   - Wan DiT
   - tokenizer / text encoder
   - scheduler adapter

2. SGLang pipeline 适配层
   - pipeline class
   - pipeline config
   - sampling params
   - model-specific stages
   - 纯函数预处理/后处理 adapter

3. 应用编排层
   - 单窗口推理 API
   - 长视频滑窗编排
   - paste-back 输出保存

4. 验证与回归层
   - scheduler 对齐测试
   - preprocess/postprocess 对齐测试
   - side-by-side latent 对齐脚本

### 5.2 依赖方向

必须保证依赖单向流动：

```text
SGLang Pipeline
  -> VideoEdit Adapter Utilities
  -> SGLang Core Wan Components

Application Helpers
  -> SGLang Pipeline

Tests / Alignment Scripts
  -> SGLang Pipeline
  -> Reference outputs or frozen fixtures
```

禁止出现：

- SGLang runtime `import ../VideoEdit-diffusers/...`
- 运行时读取原 repo 私有目录结构
- 用 `infer.py` 作为子进程或 helper
- 让 pipeline 依赖原 repo 的 `prepare_*` / `paste_back` 函数

## 6. 模块边界设计

### 6.1 可直接复用的现有模块

这些模块不应复制：

- `runtime/models/dits/wanvideo.py`
- `runtime/models/vaes/wanvae.py`
- `configs/models/dits/wanvideo.py`
- `configs/models/vaes/wanvae.py`
- `TextEncodingStage`
- `LatentPreparationStage`
- `TimestepPreparationStage`
- `DecodingStage`
- 通用 TP/SP、attention backend、offload、LoRA 机制

### 6.2 必须新增的薄适配层

新增的内容只限于 VideoEdit 差异：

1. `WanVideoEditPipeline`
   - 只负责组装 stages 和替换 scheduler

2. `VideoEditFlowMatchScheduler`
   - 只负责把 VideoEdit 的 scheduler 行为适配到 SGLang stage API

3. `VideoEditConditionStage`
   - 只负责从 `video_input_path` / `mask_input_path` 生产 `cond_masks`、`cond_latents`、`video_latents`

4. `VideoEditTimestepPreparationStage`
   - 只负责对齐 reference 的 timestep 生成、`strength` 裁剪和有效 step 数更新

5. `VideoEditLatentInitStage`
   - 只负责在 denoising 前调用 `scheduler.add_noise(video_latents, noise, first_timestep)`

6. `VideoEditDenoisingStage`
   - 只负责 dynamic CFG 和少量 VideoEdit 特殊 hook

7. VideoEdit 纯函数工具模块
   - 预处理
   - mask packing
   - paste-back
   - 元数据结构定义

### 6.3 不建议新增的内容

不建议做以下事情：

- 不新写一份 Wan Transformer
- 不新写一份 Wan VAE
- 不复制整份 `pipeline_wan_edit.py`
- 不让 `VideoEditConditionStage` 承担滑窗调度
- 不把 paste-back 硬编码进通用 `DecodingStage`
- 不为了 VideoEdit 修改通用 `DenoisingStage` 主流程分支

## 7. 推荐文件布局

### 7.1 Pipeline 与配置

- `python/sglang/multimodal_gen/runtime/pipelines/wan_videoedit_pipeline.py`
- `python/sglang/multimodal_gen/configs/pipeline_configs/videoedit_wan.py`
- `python/sglang/multimodal_gen/configs/sample/videoedit_wan.py`

### 7.2 Scheduler 与 model-specific stage

- `python/sglang/multimodal_gen/runtime/models/schedulers/videoedit_flow_match.py`
- `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/videoedit_wan.py`

### 7.3 纯函数工具与数据契约

建议新增一个独立 adapter 目录，而不是把所有逻辑塞进 stage 文件：

- `python/sglang/multimodal_gen/runtime/videoedit/contracts.py`
- `python/sglang/multimodal_gen/runtime/videoedit/preprocess.py`
- `python/sglang/multimodal_gen/runtime/videoedit/postprocess.py`

职责：

- `contracts.py`
  - 定义 `VideoEditWindowInput`
  - 定义 `VideoEditConditionBundle`
  - 定义 `VideoEditPostprocessMeta`

- `preprocess.py`
  - 视频读取
  - mask dilation / scale
  - bbox 计算
  - 裁剪与 resize
  - tensor 化
  - cond mask packing

- `postprocess.py`
  - crop-only 输出适配
  - paste-back
  - feather blend

这些文件必须是纯函数模块，不依赖 `Req`、`PipelineStage` 或 `ServerArgs`。stage 只负责把这些纯函数拼起来。

## 8. 模型目录与 overlay 方案

### 8.1 必须使用 overlay 模型目录

为了让 SGLang 自动解析到新的 pipeline，同时摆脱原 repo 目录耦合，建议提供一个独立的 diffusers-style overlay 模型目录：

```text
VideoEdit-diffusers-model/
  model_index.json
  tokenizer/
  text_encoder/
  vae/
  transformer/
  scheduler/
```

当前本地 overlay 模型目录为：

```text
/root/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model
```

其中 VideoEdit finetuned transformer 默认位于：

```text
/root/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model/transformer
```

约束：

- `_class_name = "WanVideoEditPipeline"`
- `tokenizer` / `text_encoder` / `vae` 可以来自基础 Wan2.1 模型
- `transformer` 必须来自 VideoEdit finetuned 权重
- `scheduler/` 只保留占位配置，真正运行时由 SGLang 替换成 `VideoEditFlowMatchScheduler`

### 8.2 组件覆盖

保留现有 `ServerArgs.component_paths` 机制，只允许覆盖标准模块，例如：

```bash
--transformer-path /path/to/videoedit_transformer
```

当前本地路径对应为：

```bash
--transformer-path /root/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model/transformer
```

但不要让业务输入依赖 `ServerArgs` 的 unknown args 解析。当前 `ServerArgs._extract_component_paths()` 会把任意未知的 `--<name>-path` 识别成组件路径覆盖，所以 `--video-path`、`--mask-path`，甚至未注册的 `--video-input-path` 都会被误收进 `component_paths`。VideoEdit 专用 CLI 必须先消费这些业务参数，或通过已注册的模型专属 SamplingParams 动态 CLI 解析。

## 9. SamplingParams 与 CLI 设计

### 9.1 建议新增 SamplingParams

新增：

`python/sglang/multimodal_gen/configs/sample/videoedit_wan.py`

建议字段：

```python
@dataclass
class WanVideoEditSamplingParams(SamplingParams):
    _default_height = 480
    _default_width = 832

    video_input_path: str | None = None
    mask_input_path: str | None = None

    infer_len: int = 81
    strength: float = 1.0
    dynamic_cfg: bool = True
    dynamic_cfg_max_step: int = 15
    dynamic_cfg_min: float = 1.0

    bbox_padding: int = 0
    dilate_px: int = 15
    mask_scale: float = 1.2
    feather_px: int = 12
    adain_boundary_dilate: int = 15

    enable_paste_back: bool = False
    save_crop_only: bool = True
```

注意这里建议改名为：

- `video_input_path`
- `mask_input_path`

而不是：

- `video_path`
- `mask_path`

原因是 Python 字段名需要和模型业务语义绑定，并避免与组件覆盖的 `video_path` / `mask_path` 这种短名混淆。但仅靠字段改名不能解决当前通用 CLI 的 unknown args 问题：`--video-input-path` 仍然以 `-path` 结尾。只有在 VideoEdit wrapper CLI 或未来模型专属动态 CLI 中，这些参数被 argparse 注册为已知参数后，才不会进入 `ServerArgs._extract_component_paths()`。

### 9.2 参数校验

`WanVideoEditSamplingParams.__post_init__()` 中应显式校验：

- `video_input_path` 必填
- `mask_input_path` 必填
- 单窗口 native pipeline 中 `num_frames == infer_len`；长视频 helper 对外可以接收任意长度，但拆到单窗口后必须满足这个关系
- `(infer_len - 1) % 4 == 0`
- `strength` 范围合法
- 输入视频和 mask 帧数一致
- 当前 native pipeline 仅支持单窗口时，禁止传入长视频滑窗参数组合
- `num_outputs_per_prompt == 1`，除非后续明确实现多输出时的条件 tensor repeat 规则
- MVP 阶段如果没有完成 SP alignment，先限制 VideoEdit 单窗口请求在单 SP 下运行；允许 SP 前必须验证 `batch.latents` 与 `batch.image_latent` 同步 shard

### 9.3 CLI 方案

现状问题：

- `generate.py` 只基于基类 `SamplingParams` 静态注册 CLI 参数
- config 文件提取 sampling fields 时也只看 `SamplingParams`
- `unknown_args` 中的 `--xxx-path` 会被 `ServerArgs._extract_component_paths()` 抢走

所以建议分两阶段：

1. MVP
   - 提供专用 wrapper CLI 或 Python API
   - 直接构造 `WanVideoEditSamplingParams`

2. 正式接入
   - `generate_cmd()` 先解析 `model_path` 得到 `model_info.sampling_param_cls`
   - 再基于模型专属 SamplingParams 注册和提取 CLI/config 字段

在通用 CLI 动态注册改完之前，不建议把 VideoEdit 直接暴露给通用 `sglang generate`。

## 10. PipelineConfig 设计

新增：

`python/sglang/multimodal_gen/configs/pipeline_configs/videoedit_wan.py`

建议不要继承 `WanI2V480PConfig`，而应继承 `WanT2V480PConfig`，再补充 VideoEdit 需要的 VAE encoder 和 frame 约束：

```python
@dataclass
class WanVideoEditPipelineConfig(WanT2V480PConfig):
    task_type: ModelTaskType = ModelTaskType.T2V
    flow_shift: float | None = 5.0
    vae_precision: str = "bf16"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True
        self.dit_config.arch_config.in_channels = 36
        self.dit_config.arch_config.out_channels = 16
        self.dit_config.arch_config.num_channels_latents = 16
```

这样做的好处：

- 不会误走通用 I2V/TI2V 图像输入分支
- 仍保持视频输出任务语义
- 复用 Wan 的 latent 形状和解码流程
- fail-fast 地表达 VideoEdit transformer 的 36 输入通道契约

如果后续需要更明确的任务类型，可以再引入新的 `ModelTaskType.VIDEO_EDIT`。但在第一阶段，为了最小侵入和低风险，建议先不修改全局 task enum。

### 10.1 配置校验

`WanVideoEditPipelineConfig.__post_init__()` 或 pipeline 初始化阶段需要增加以下校验：

- `dit_config.arch_config.in_channels == 36`
- `dit_config.arch_config.out_channels == 16`
- `vae_config.load_encoder is True`
- `vae_config.load_decoder is True`
- `vae_config.arch_config.scale_factor_temporal == 4`
- `vae_config.arch_config.scale_factor_spatial == 8`

如果 transformer config 从权重目录加载后覆盖了这些字段，应以实际权重 config 为准再校验；不要静默降级到默认 Wan T2V 的 16 通道输入。

## 11. Pipeline 设计

新增：

`python/sglang/multimodal_gen/runtime/pipelines/wan_videoedit_pipeline.py`

职责：

- 定义 `WanVideoEditPipeline(LoRAPipeline, ComposedPipelineBase)`
- `pipeline_name = "WanVideoEditPipeline"`
- `_required_config_modules = ["text_encoder", "tokenizer", "vae", "transformer", "scheduler"]`
- `initialize_pipeline()` 中把 model_index 的 scheduler 实例替换成 `VideoEditFlowMatchScheduler`
- `create_pipeline_stages()` 中组装标准 stage 和 VideoEdit 专属 stage

伪代码：

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

    def initialize_pipeline(self, server_args):
        self.modules["scheduler"] = VideoEditFlowMatchScheduler(
            shift=server_args.pipeline_config.flow_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )

    def create_pipeline_stages(self, server_args):
        self.add_stage(InputValidationStage())
        self.add_standard_text_encoding_stage()
        self.add_stage(VideoEditConditionStage(vae=self.get_module("vae")))
        self.add_standard_latent_preparation_stage()
        self.add_stage(VideoEditTimestepPreparationStage(scheduler=self.get_module("scheduler")))
        self.add_stage(VideoEditLatentInitStage(scheduler=self.get_module("scheduler")))
        self.add_stage(
            VideoEditDenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
                vae=self.get_module("vae"),
                pipeline=self,
            )
        )
        self.add_standard_decoding_stage()
```

文件末尾必须提供：

```python
EntryClass = WanVideoEditPipeline
```

原因是当前 registry 会扫描 `runtime/pipelines/*` 下带 `EntryClass` 的模块，并用 `pipeline_name` 匹配 overlay `model_index.json` 中的 `_class_name`。

## 12. Scheduler 适配层

新增：

`python/sglang/multimodal_gen/runtime/models/schedulers/videoedit_flow_match.py`

职责不是重新发明 scheduler，而是适配 SGLang 的 stage 协议。

必须对齐的接口：

- `set_timesteps(..., device=None, **kwargs)`
- `set_begin_index()`，即使只是 no-op
- `scale_model_input(sample, timestep)`，对 VideoEdit 可实现为 no-op
- `step(..., return_dict=False)` 返回 `(prev_sample,)`
- `add_noise(original_samples, noise, timestep)`
- `order = 1`
- `num_train_timesteps = 1000`

必须对齐的行为：

- sigma 公式
- timestep 序列
- `extra_one_step=True`
- `shift=5`
- `strength < 1.0` 时的 `get_timesteps()`

适配层边界：

- 不修改通用 `TimestepPreparationStage`，但为 VideoEdit 增加专属 `VideoEditTimestepPreparationStage`
- 不修改通用 `DenoisingStage` 的 scheduler 约定
- 让 scheduler 自己满足通用 stage 的调用要求

### 12.1 VideoEditTimestepPreparationStage

新增在 `videoedit_wan.py` 中即可，不需要单独文件。

职责：

- 调用 `scheduler.set_timesteps(batch.num_inference_steps, device=device, shift=5)`。
- 当 `batch.strength < 1.0` 时调用 `scheduler.get_timesteps(batch.num_inference_steps, scheduler.timesteps, batch.strength)`。
- 将裁剪后的 timesteps 写回 `batch.timesteps`。
- 将返回的有效 step 数写回 `batch.num_inference_steps`，保证 progress bar、warmup step 和 denoising loop 与 reference 一致。

伪代码：

```python
class VideoEditTimestepPreparationStage(TimestepPreparationStage):
    def forward(self, batch, server_args):
        device = get_local_torch_device()
        self.scheduler.set_timesteps(
            batch.num_inference_steps,
            device=device,
            shift=server_args.pipeline_config.flow_shift,
        )
        timesteps = self.scheduler.timesteps
        if batch.strength < 1.0:
            timesteps, effective_steps = self.scheduler.get_timesteps(
                batch.num_inference_steps,
                timesteps,
                batch.strength,
            )
            batch.num_inference_steps = effective_steps
        batch.timesteps = timesteps.to(device)
        return batch
```

这比把 `strength` 塞进通用 `TimestepPreparationStage.prepare_extra_set_timesteps_kwargs` 更清晰，因为 reference 的语义是“先生成完整 schedule，再裁剪 schedule”，不是改变 `set_timesteps()` 的 sigma 起点。

## 13. VideoEditConditionStage 设计

新增：

`python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/videoedit_wan.py`

建议 stage 内只做 orchestration，真正的图像/视频处理放到 `runtime/videoedit/preprocess.py`。

### 13.1 输入

从 `batch.sampling_params` 读取：

- `video_input_path`
- `mask_input_path`
- `infer_len`
- `bbox_padding`
- `dilate_px`
- `mask_scale`

### 13.2 输出契约

stage 结束时必须写入：

- `batch.image_latent = torch.cat([cond_masks, cond_latents], dim=1)`
- `batch.extra["videoedit"]["video_latents"] = video_latents`
- `batch.extra["videoedit"]["post_meta"] = VideoEditPostprocessMeta(...)`

其中：

- `batch.image_latent` 的 shape 必须是 `[B, 20, F_lat, H/8, W/8]`
- 后续 `VideoEditDenoisingStage` 会把 `latents` 和 `batch.image_latent` 拼接成 `36` 通道

### 13.3 纯函数输出对象建议

建议 `preprocess.py` 输出：

```python
@dataclass
class VideoEditConditionBundle:
    masked_video_tensor: torch.Tensor
    raw_video_tensor: torch.Tensor
    cond_masks: torch.Tensor
    cond_latents: torch.Tensor
    video_latents: torch.Tensor
    post_meta: VideoEditPostprocessMeta
```

这样 stage 不感知具体图像处理步骤，只消费 bundle。

### 13.4 关键对齐点

- 首帧 mask 必须强制全黑
- `cond_masks` 必须保持 preserve=1、inpaint=0 语义
- mask packing 必须按 reference 顺序执行：`mask[0:1].repeat(4)` + `mask[1:]` 得到 `infer_len + 3` 帧；当 `infer_len=81` 时为 `84`，再 reshape / transpose 成 `[B, 4, 21, H/8, W/8]`
- VAE encode 使用 Wan 的 mean/std 归一化
- `cond_latents` / `video_latents` 与 reference shape、dtype、统计量一致
- `batch.image_latent` 和 `batch.latents` 必须在 SP shard 前保持相同时间维；如果启用 sequence parallel，二者必须通过同一个 `pipeline_config.shard_latents_for_sp()` 路径切分

## 14. VideoEditLatentInitStage 设计

职责很单一：

- 在 `LatentPreparationStage` 和 `VideoEditTimestepPreparationStage` 之后
- 在 `DenoisingStage` 之前
- 使用 `video_latents` 替换默认纯噪声初始化

逻辑：

```python
batch.latents = scheduler.add_noise(
    video_latents,
    batch.latents,
    batch.timesteps[:1],
)
```

为什么单独拆 stage：

- 便于复用标准 `LatentPreparationStage`
- 便于测试“首步 latent 是否和 reference 一致”
- 避免把 VideoEdit 初始化埋在通用 denoising 流程内部

如果后续确认更适合放进 `VideoEditDenoisingStage._before_denoising_loop()`，也可以迁移，但接口职责保持不变。

## 15. VideoEditDenoisingStage 设计

`VideoEditDenoisingStage` 应继承 `DenoisingStage`，只覆盖最少的 hook。当前 SGLang 通用 `DenoisingStage.forward()` 已经在每步把 `batch.image_latent` 与 `latents` 沿 channel 维拼接，因此 VideoEdit 不需要重写完整 forward。

建议保留的通用能力：

- CFG parallel / SP / TP
- scheduler.step
- offload
- profile
- trajectory latents

只覆盖以下逻辑：

1. 在 `_predict_noise_with_cfg()` 内做每步 guidance 计算

```python
current_cfg, do_cfg = calc_current_cfg(
    max_cfg=batch.guidance_scale,
    current_step=step_index,
    max_step=batch.dynamic_cfg_max_step,
    min_cfg=batch.dynamic_cfg_min,
    dynamic_cfg=batch.dynamic_cfg,
)
```

2. positive pass 永远执行。
3. `do_cfg=False` 时直接返回 positive `noise_pred`，跳过 negative pass。
4. `do_cfg=True` 时执行 negative pass，并按 reference 公式合成：

```python
noise_pred = noise_uncond + current_cfg * (noise_pred_cond - noise_uncond)
```

这样做的原因：

- SGLang 通用 `DenoisingStage` 已支持 `batch.image_latent` 拼接
- VideoEdit 的特殊点主要是 dynamic CFG，而不是完整 denoising 流程重写
- CFG parallel 暂时不作为 MVP 默认能力；如果要支持，需要确认 `do_cfg=False` 时 cfg rank 的同步和 all-reduce 行为，不应只在单 rank 上跳过 negative 分支

## 16. 预处理 / 后处理解耦

### 16.1 预处理放在哪里

不要把 `utils/preprocess.py` 原样搬进 stage。

正确做法是：

- 把纯算法逻辑迁移到 `runtime/videoedit/preprocess.py`
- 把 `Req` 读写和 device/dtype 管理留在 `VideoEditConditionStage`

建议预处理层只暴露纯函数，例如：

- `load_and_validate_video_pair()`
- `compute_edit_bbox()`
- `build_window_inputs()`
- `pack_cond_masks()`
- `encode_video_conditions()`

### 16.2 后处理放在哪里

native pipeline 第一阶段建议只输出 crop-only 视频。

paste-back 不应一开始就耦合进 `DecodingStage`，原因：

- 它是应用层功能，不是核心 diffusion 推理功能
- 它依赖 bbox、原始帧、mask、保存策略
- 这些都不属于模型推理 contract

建议阶段化处理：

1. 阶段一
   - native pipeline 只返回 crop-only 结果

2. 阶段二
   - 在 helper/API 层接入 paste-back
   - 如确有必要，再通过 `PipelineConfig.post_decoding()` 接入

`postprocess.py` 与 stage 的关系：

- `postprocess.py` 是纯函数
- helper/API 调用它
- `Req.extra["videoedit"]["post_meta"]` 只负责传元数据

## 17. 长视频滑动窗口修复方案

长视频滑窗不应放进 native pipeline。native pipeline 只接受一个固定长度窗口，长视频由 helper / serve 应用层编排。

窗口规划采用固定 `81` 帧窗口和可配置 overlap：

- 第 0 个窗口覆盖 `[0, 80]`。
- 第 1 个窗口覆盖 `[81 - overlap, 161 - overlap]`。
- 第 k 个窗口起点为 `k * (81 - overlap)`，终点为 `start + 80`。
- `overlap` 可以为 `0`，此时窗口为 `[0,80]`、`[81,161]`、`[162,242]`。
- 当窗口尾部超过真实视频长度时，不重复尾帧 padding，而是按反射序列逆向往回计数补满 81 帧。

这个策略和 upstream 当前 “按 `infer_len` 截断到整数倍” 的脚本不同。SGLang helper 应覆盖非 81 倍数、短视频和尾窗口不足 81 帧的场景，并在 metadata 中记录每个窗口的真实帧映射，避免后处理阶段猜测。

### 17.1 Native pipeline 的职责

- 只处理单窗口。
- 输入长度固定为 `infer_len=81`。
- 窗口输入一定是长度为 81 的帧序列，序列里的帧可以来自真实连续区间，也可以来自反射补齐。
- 窗口内 local frame 0 的 mask 仍按 VideoEdit 预处理约定强制全黑。
- 输出该窗口的 crop-only 编辑结果，长度仍为 `81`。
- 不感知全局视频总帧数、窗口编号、跨窗口融合、paste-back 或最终编码。

### 17.2 应用层 helper 职责

建议新增：

- `python/sglang/multimodal_gen/runtime/videoedit/long_video.py`

核心对象：

```python
@dataclass
class VideoEditLongVideoParams:
    infer_len: int = 81
    overlap: int = 0
    use_repaired_context: bool = True
    keep_intermediate_windows: bool = False
    enable_paste_back: bool = True
    save_crop_only: bool = False


@dataclass
class VideoEditWindowSpec:
    window_index: int
    start_index: int
    end_index: int
    input_indices: list[int]
    commit_local_to_global: dict[int, int]
    reflected_count: int = 0


@dataclass
class VideoEditLongVideoResult:
    output_video_path: str
    crop_video_path: str | None
    metadata_path: str | None
    num_input_frames: int
    num_output_frames: int
    num_windows: int
    fps: int
```

helper 负责：

- 读取输入视频和 mask 视频。
- 校验两者帧数、fps、尺寸和有效 mask。
- 计算全局 bbox / crop / resize 元数据。
- 生成 `VideoEditWindowSpec`。
- 为每个窗口写出临时 window video / window mask。
- 逐个构造 `WanVideoEditSamplingParams` 并调用 scheduler。
- 读取单窗口输出，按全局帧号拼接或融合。
- 统一做 crop-only 保存、paste-back、音频拷贝和 metadata 输出。

### 17.3 输入 / 输出方式

长视频 helper 的输入：

- `prompt: str`
- `negative_prompt: str | None`
- `video_input_path: str`
- `mask_input_path: str`
- `output_path: str`
- `width: int`
- `height: int`
- `fps: int | None`
- `infer_len: int = 81`
- `overlap: int = 0`
- `seed: int`
- `guidance_scale: float`
- `num_inference_steps: int`
- VideoEdit 预处理参数：`bbox_padding`、`dilate_px`、`mask_scale`、`feather_px` 等

长视频 helper 的输出：

- 最终 repaired video：默认 `mp4`。
- 可选 crop-only video：用于 debug 和对齐。
- 可选 metadata json：记录 bbox、窗口列表、输入输出帧映射、seed、尺寸、fps、每个窗口输出路径。

metadata 建议形态：

```json
{
  "num_input_frames": 243,
  "num_output_frames": 243,
  "fps": 24,
  "infer_len": 81,
  "overlap": 0,
  "stride": 81,
  "bbox": [0, 0, 832, 480],
  "windows": [
    {
      "window_index": 0,
      "start_index": 0,
      "end_index": 80,
      "input_indices": [0, 1, 2, "...", 80],
      "commit_local_to_global": {"0": 0, "1": 1, "80": 80},
      "reflected_count": 0
    }
  ]
}
```

### 17.4 窗口生成逻辑

固定：

```python
window_size = infer_len              # 81
stride = infer_len - overlap         # overlap=0 时 stride=81
```

约束：

- `infer_len == 81`，或至少满足 `(infer_len - 1) % 4 == 0`。
- `0 <= overlap < infer_len`，因此 `overlap=80` 是最大重叠，`stride=1`。
- `stride > 0`。
- 默认 `overlap=0`，即完全非重叠窗口。

窗口生成伪代码：

```python
def reflected_indices(num_frames: int, start: int, length: int) -> list[int]:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if num_frames == 1:
        return [0] * length

    indices = []
    idx = min(max(start, 0), num_frames - 1)
    direction = 1

    while len(indices) < length:
        indices.append(idx)
        next_idx = idx + direction
        if next_idx >= num_frames:
            direction = -1
            next_idx = num_frames - 2
        elif next_idx < 0:
            direction = 1
            next_idx = 1
        idx = next_idx

    return indices


def choose_commit_map(
    input_indices: list[int],
    commit_start: int,
    commit_end: int,
) -> dict[int, int]:
    """Return local_idx -> global_idx for frames owned by this window.

    Reflected filler frames are not committed unless they are needed to cover a
    real frame in a short clip. If a global frame appears multiple times, prefer
    the first non-local-0 occurrence so frame 0 can still be repaired in short
    clips where local 0 mask is forced black.
    """
    candidates: dict[int, list[int]] = {}
    for local_idx, global_idx in enumerate(input_indices):
        if commit_start <= global_idx <= commit_end:
            candidates.setdefault(global_idx, []).append(local_idx)

    commit_map = {}
    for global_idx in range(commit_start, commit_end + 1):
        local_candidates = candidates[global_idx]
        preferred = next(
            (local_idx for local_idx in local_candidates if local_idx != 0),
            local_candidates[0],
        )
        commit_map[preferred] = global_idx
    return commit_map


def build_window_specs(
    num_frames: int,
    infer_len: int,
    overlap: int,
):
    stride = infer_len - overlap
    specs = []

    start = 0
    while start < num_frames:
        nominal_end = start + infer_len - 1

        input_start = start
        if start == 0 and num_frames <= infer_len // 2 and num_frames > 1:
            input_start = 1
        input_indices = reflected_indices(num_frames, input_start, infer_len)

        commit_start = start
        commit_end = min(nominal_end, num_frames - 1)

        # 极短视频从 frame 1 起步，让 frame 0 避开 local 0。
        # 例如 N=20 时 input_indices 为 [1,2,...,19,18,17,...,0,1,2,...]。
        # commit_start/commit_end 仍覆盖 [0, N-1]。
        if num_frames < infer_len:
            commit_start = 0
            commit_end = num_frames - 1

        commit_map = choose_commit_map(input_indices, commit_start, commit_end)

        specs.append(
            VideoEditWindowSpec(
                window_index=len(specs),
                start_index=start,
                end_index=nominal_end,
                input_indices=input_indices,
                commit_local_to_global=commit_map,
                reflected_count=sum(i > num_frames - 1 for i in range(start, start + infer_len)),
            )
        )

        if num_frames <= infer_len:
            break
        start += stride

    return specs
```

举例：

- `N=79`，`overlap=0`：1 个窗口，输入 `[0,1,2,...,78,77,76]`，提交全局 `[0,78]`。
- `N=20`：1 个窗口，输入 `[1,2,...,19,18,17,...,0,1,2,...]`，提交全局 `[0,19]`；提交时优先选择每个全局帧的非 local 0 位置。
- `N=81`，`overlap=0`：1 个窗口，输入 `[0,1,2,...,80]`。
- `N=99`，`overlap=0`：第一个窗口 `[0,1,...,80]`；第二个窗口 `[81,82,...,98,97,96,...]`，只提交全局 `[81,98]`，反射补齐部分不提交。
- `N=99`，`overlap=8`：第一个窗口 `[0,1,...,80]`；第二个窗口从 `81 - 8 = 73` 开始，即 `[73,74,...,98,97,96,...]`，重叠帧 `[73,80]` 参与融合。

为了保证“所有帧完整修复”，应用层提交规则必须是：

- 每个窗口只提交 `commit_local_to_global` 指定的 local 输出帧。
- 反射补齐帧默认只是为了满足 81 帧模型输入，不提交到最终结果；极短视频需要用反射位置覆盖真实全局帧时例外。
- 如果同一个全局帧在一个窗口内出现多次，优先提交非 local 0 的位置。
- 如果启用 overlap，同一全局帧可被多个窗口提交，最终按权重融合。

### 17.5 窗口输入构造

每个窗口需要构造两个临时文件：

- `window_{i:04d}.mp4`
- `window_{i:04d}_mask.mp4` 或 mask frame directory

构造规则：

- local 0 使用 `input_indices[0]` 对应帧。
- local 0 的 mask 强制为全黑。
- local 1..80 按 `input_indices[1:]` 取帧和 mask。
- 反射补齐位置仍使用对应全局帧的真实 mask；是否提交由 `commit_local_to_global` 决定。
- 如果 `use_repaired_context=True`，overlap 区域内已经完成融合的帧可以优先作为后续窗口输入；否则所有窗口都使用原始输入帧。

`use_repaired_context=True` 时窗口必须顺序执行，因为后一窗口可能依赖前一窗口已经融合的 overlap 结果。后续如果要并行执行，可以提供 `use_repaired_context=False`，但这会牺牲跨窗口一致性。

### 17.6 单窗口调用方式

每个窗口都构造一个单窗口 `WanVideoEditSamplingParams`：

```python
window_sampling_params = WanVideoEditSamplingParams(
    request_id=f"{job_id}_window_{window_index:04d}",
    prompt=prompt,
    negative_prompt=negative_prompt,
    video_input_path=window_video_path,
    mask_input_path=window_mask_path,
    output_path=window_output_dir,
    output_file_name=f"{job_id}_window_{window_index:04d}.mp4",
    num_frames=81,
    infer_len=81,
    fps=fps,
    width=width,
    height=height,
    seed=seed + window_index if vary_seed_by_window else seed,
    guidance_scale=guidance_scale,
    num_inference_steps=num_inference_steps,
    enable_paste_back=False,
    save_crop_only=True,
)
```

默认建议所有窗口使用相同 `seed`，这样跨窗口噪声分布更稳定；如果后续实测发现重复纹理明显，再提供 `vary_seed_by_window=True` 作为可选项。

### 17.7 输出拼接与 overlap 融合

维护两个全局 buffer：

```python
accum_frames = np.zeros([N, H, W, C], dtype=np.float32)
accum_weights = np.zeros([N, 1, 1, 1], dtype=np.float32)
```

每个窗口输出解码为 `window_frames[0:81]` 后：

```python
for local_idx, global_idx in spec.commit_local_to_global.items():
    weight = temporal_blend_weight(global_idx, spec, overlap)
    accum_frames[global_idx] += window_frames[local_idx] * weight
    accum_weights[global_idx] += weight
```

最后：

```python
if np.any(accum_weights == 0):
    raise RuntimeError("Some frames were not repaired by any window")
final_crop_frames = accum_frames / accum_weights
```

默认 `overlap=0` 时权重恒为 `1`，等价于直接写入。

启用 overlap 时建议使用简单线性 ramp：

- 窗口前 overlap 区域权重从 0 到 1。
- 窗口后 overlap 区域权重从 1 到 0。
- 中间区域权重为 1。

如果 `overlap=0`，不做 ramp。对短视频反射补齐产生的重复帧，融合权重只作用在 `commit_local_to_global` 选中的 local 帧上。

### 17.8 全局 bbox 与 paste-back

MVP 推荐使用全局 bbox：

1. 读取完整 mask。
2. 对每帧 mask 做 dilation / scale。
3. 对所有非空 mask 的 bbox 求 union。
4. 加 `bbox_padding`。
5. 裁剪所有窗口时使用同一个 bbox。

这样可以避免每个窗口 crop 区域不同导致的画面抖动，也方便最终按全局坐标 paste-back。

如果 mask 在长视频里跨度极大，global bbox 可能接近全画幅，性能会下降。后续可以支持 `bbox_mode="window"`，但那会要求每个窗口独立 paste-back，然后再做全帧融合，MVP 不建议先做。

最终输出有两种模式：

- `save_crop_only=True`：直接把 `final_crop_frames` 编码为视频，适合算法对齐。
- `enable_paste_back=True`：把 `final_crop_frames` resize 回 bbox 区域，再用全局 mask feather blend 回原始帧，保存完整分辨率视频。

音频处理属于应用层后处理：

- 默认从原视频拷贝音轨到最终 mp4。
- 如果 ffmpeg 不存在或音频拷贝失败，只保留无声视频并记录 warning。

### 17.9 边界条件

必须显式处理：

- 输入视频 0 帧：返回 400 / `ValueError`。
- mask 视频 0 帧：返回 400 / `ValueError`。
- video / mask 帧数不一致：默认报错；如果 `allow_mask_loop=True` 才允许短 mask 循环或末帧 repeat。
- video / mask 分辨率不一致：mask resize 到 video 尺寸，使用 nearest 插值。
- fps 不一致：输出 fps 继承原视频；mask fps 只用于读帧，不参与输出。
- `N < 81`：构造 1 个窗口，用反射序列补满 81 帧，例如 `N=79` 为 `[0..78,77,76]`，不重复尾帧。
- `N == 81`：构造 1 个窗口 `[0..80]`。
- `N > 81` 且不是 81 的倍数：最后一个窗口从正常 stride 起点开始，先正向取到末帧，再反射逆向补齐，例如 `N=99` 为 `[81-overlap, ..., 98, 97, ...]`。
- 某些帧 mask 为空：仍送入窗口，模型输出会接近原帧；paste-back 时该帧保持原始内容。
- 全局 mask 全空：可直接返回原视频副本，也可按 `allow_empty_mask=False` 报错。serve 默认建议报错，离线 helper 可配置为复制原视频。
- 首帧需要修复且 `N <= infer_len // 2` 时，短视频首帧避让规则会让 frame 0 出现在非 local 0 的反射位置。
- 末帧需要修复：尾窗口的正向段一定包含末帧；反射补齐只用于补满 81 帧。
- 请求中断 / 服务异常：删除临时窗口文件；已写 job metadata 中记录最后成功窗口，后续可扩展 resume。

## 18. Serve 接口设计

当前 SGLang multimodal_gen 已有 `/v1/videos` 异步生成接口，返回 `VideoResponse`，后台通过 `async_scheduler_client` 调度任务。VideoEdit 修复建议复用这个异步任务模型，但不要把修复请求塞进普通 `VideoGenerationsRequest`，否则 `input_reference`、`reference_url` 的 I2V 语义会混乱。

参考 `../wan_eraser/run_parallel_ray_95_erase.py` 的服务形态，VideoEdit serve 需要保留三个行为：

- 提交接口只做参数校验和任务登记，立即返回 `job_id`，真实计算在后台执行。
- 计算资源只允许一个 VideoEdit 任务占用；由于单任务计算量很大，服务侧队列容量设为 `1`，超过容量直接返回 `429`，不在计算侧堆积请求。
- 回调不能传 Python 函数对象。HTTP API 中应传 `callback_url`，服务在任务完成或失败时向该 URL `POST` 状态；SDK / 调用方可以把这个 URL 封装成业务里的“回调函数”。

### 18.1 路由设计

建议新增专用接口：

```text
POST   /v1/videos/repairs                 # 运行接口，提交一个视频编辑任务
GET    /v1/videos/{video_id}/progress      # 进度接口，只返回轻量进度信息
GET    /v1/videos/{video_id}               # 查询接口，返回完整任务记录
GET    /v1/videos/{video_id}/content       # 下载接口，本地结果文件下载
DELETE /v1/videos/{video_id}               # 删除接口，取消/删除任务和临时文件
GET    /healthz                            # 健康接口，进程存活和模型可用性检查
```

说明：

- `POST /v1/videos/repairs` 必须在 `/{video_id}` 动态路由之前注册。
- 查询、下载、删除继续复用现有 video store 语义。
- job record 中增加 `operation="video_repair"`，但响应仍可复用 `VideoResponse`。
- `GET /v1/videos/{video_id}/progress` 是 `GET /v1/videos/{video_id}` 的轻量版本，便于高频轮询。
- `DELETE` 对 `queued` 任务直接标记为 `deleted` 并清理输入文件；对 `in_progress` 任务先标记 `cancel_requested=True`，后台任务在窗口边界检查后退出。
- `GET /healthz` 返回服务存活、模型加载状态和当前任务槽状态，不承诺 GPU 空闲。

建议健康响应：

```json
{
  "status": "ok",
  "model": "WanVideoEditPipeline",
  "model_loaded": true,
  "queue_capacity": 1,
  "active_jobs": 0,
  "queued_jobs": 0
}
```

### 18.2 请求协议

建议在 `python/sglang/multimodal_gen/runtime/entrypoints/openai/protocol.py` 新增：

```python
class VideoRepairRequest(BaseModel):
    task_id: Optional[str] = None
    prompt: str
    video_input_path: Optional[str] = None
    mask_input_path: Optional[str] = None
    video_url: Optional[str] = None
    mask_url: Optional[str] = None
    video_bucket: Optional[str] = None
    video_object_key: Optional[str] = None
    mask_bucket: Optional[str] = None
    mask_object_key: Optional[str] = None
    model: Optional[str] = None

    callback_url: Optional[str] = None
    callback_events: Optional[list[str]] = None  # default: ["completed", "failed"]
    callback_timeout: Optional[float] = 10.0

    size: Optional[str] = ""
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[int] = None
    seed: Optional[int] = 1024
    generator_device: Optional[str] = "cuda"

    infer_len: Optional[int] = 81
    overlap: Optional[int] = 0
    use_repaired_context: Optional[bool] = True
    keep_intermediate_windows: Optional[bool] = False

    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    negative_prompt: Optional[str] = None
    dynamic_cfg: Optional[bool] = True
    dynamic_cfg_max_step: Optional[int] = 15
    dynamic_cfg_min: Optional[float] = 1.0

    bbox_padding: Optional[int] = 0
    dilate_px: Optional[int] = 15
    mask_scale: Optional[float] = 1.2
    feather_px: Optional[int] = 12
    adain_boundary_dilate: Optional[int] = 15

    enable_paste_back: Optional[bool] = True
    save_crop_only: Optional[bool] = False
    output_storage: Optional[str] = "object_storage"
    output_bucket: Optional[str] = None
    output_object_key: Optional[str] = None
    output_path: Optional[str] = None
    output_quality: Optional[str] = "default"
    output_compression: Optional[int] = None
    perf_dump_path: Optional[str] = None
```

支持两类请求体：

1. `multipart/form-data`
   - `prompt`
   - `video_file`
   - `mask_file`
   - 可选 `extra_body` JSON 字符串承载其他参数

2. `application/json`
   - `video_input_path` / `mask_input_path`
   - 或 `video_url` / `mask_url`
   - 或 `video_bucket` + `video_object_key`、`mask_bucket` + `mask_object_key`
   - 其他字段直接放 top-level

本地路径适合内网 / 离线服务。公网服务建议只开放上传和 URL 下载，并增加 allowlist，避免任意读取服务器本地文件。

### 18.2.1 输入视频和 mask 的来源

推荐明确支持三种输入来源，并把责任边界写清楚：

1. 对象存储 / 数据桶，服务端处理，推荐生产默认
   - 请求传 `video_bucket/video_object_key` 和 `mask_bucket/mask_object_key`，或使用服务配置里的默认 bucket 只传 object key。
   - VideoEdit 服务负责生成临时下载链接、下载到本地工作目录、校验帧数/分辨率、任务结束后清理临时文件。
   - 适合内部存储服务、MinIO、S3、COS、OSS 等。

2. HTTP(S) URL，服务端处理
   - 请求传 `video_url` 和 `mask_url`。
   - 服务端负责下载，但必须配置域名 allowlist、最大文件大小、下载超时和 MIME / 后缀校验。
   - 适合跨服务调用，但不建议开放任意公网 URL。

3. 本地路径，调用方处理
   - 请求传 `video_input_path` 和 `mask_input_path`。
   - 调用方必须保证路径在容器内可见，例如通过 Docker volume 挂载到 `/data/inputs`。
   - 只建议用于离线、同机或内网可信部署；公网服务不要允许任意本地路径读取。

MVP 建议生产模式先采用对象存储输入。这样调用方只需要把原始视频和 mask 视频放到数据桶，服务端统一下载、清理和记录 metadata。

### 18.2.2 输出视频的去向

`output_storage` 建议只允许两个值：

- `object_storage`：默认生产模式。服务端把编辑后视频上传到对象存储，任务记录返回 `url` 或 `bucket/object_key`。本地中间文件在 callback 成功或任务结束后清理。
- `local`：调试 / 离线模式。服务端把最终视频保留在 `output_path` 或服务启动参数 `--output-path` 下，查询接口返回 `file_path`，下载接口 `/content` 直接读这个文件。

生产环境推荐 `object_storage`，因为生成文件通常很大，HTTP 服务本地磁盘不应作为长期结果仓库。若调用方要求自己处理上传，则使用 `local`，并通过共享 volume 读取 `file_path`。

### 18.3 Endpoint 主要流程

建议在 `video_api.py` 中新增：

```python
@router.post("/repairs", response_model=VideoResponse)
async def create_video_repair(...):
    request_id = generate_request_id()
    server_args = get_global_server_args()

    # 1. 解析 VideoRepairRequest。
    # 2. 检查 VideoEdit admission：active + queued 必须小于 queue_capacity=1。
    # 3. 保存上传视频 / mask，或下载 URL / 对象存储文件，或校验本地路径。
    # 4. 校验当前模型是否是 WanVideoEditPipeline。
    # 5. 解析 output_path / output_storage；未配置时创建临时输出目录。
    # 6. 创建 job，写入 VIDEO_STORE，status=queued。
    # 7. asyncio.create_task(_dispatch_video_repair_job_async(...))。
    # 8. 立即返回 VideoResponse。
```

队列容量为 1 时，建议 admission 规则为：

```python
if await VIDEO_STORE.count_active(operation="video_repair") >= 1:
    raise HTTPException(
        status_code=429,
        detail={
            "code": "videoedit_queue_full",
            "message": "A VideoEdit task is already queued or running.",
        },
    )
```

这与参考脚本中 `threading.BoundedSemaphore(1)` 的语义一致：计算侧不排长队，业务方需要在上层重试或调度。

后台任务：

```python
async def _dispatch_video_repair_job_async(job_id, repair_req, temp_dirs, output_persistent):
    try:
        await VIDEO_STORE.update_fields(job_id, {"status": "in_progress", "progress": 1})

        plan = build_long_video_plan(repair_req)
        await VIDEO_STORE.update_fields(job_id, {"progress": 5})

        result = await run_videoedit_long_video(
            scheduler_client=async_scheduler_client,
            server_args=get_global_server_args(),
            request_id=job_id,
            request=repair_req,
            plan=plan,
            progress_cb=lambda p: VIDEO_STORE.update_fields(job_id, {"progress": p}),
        )

        cloud_url = None
        output_object = None
        if repair_req.output_storage == "object_storage":
            output_object = await object_storage.upload_and_cleanup(
                result.output_video_path,
                bucket=repair_req.output_bucket,
                object_key=repair_req.output_object_key,
            )
            cloud_url = output_object.url

        await VIDEO_STORE.update_fields(
            job_id,
            {
                "status": "completed",
                "progress": 100,
                "completed_at": int(time.time()),
                "url": cloud_url,
                "output_object": output_object,
                "file_path": result.output_video_path if not cloud_url and output_persistent else None,
                "metadata_path": result.metadata_path,
            },
        )
    except Exception as e:
        await VIDEO_STORE.update_fields(job_id, {"status": "failed", "error": {"message": str(e)}})
    finally:
        await maybe_post_videoedit_callback(job_id, repair_req.callback_url)
        cleanup_temp_dirs(temp_dirs)
```

完成回调建议发送完整任务摘要：

```json
{
  "task_id": "caller_task_123",
  "id": "video_repair_xxx",
  "operation": "video_repair",
  "status": "completed",
  "progress": 100,
  "url": "https://storage.example.com/bucket/result.mp4",
  "output_object": {
    "bucket": "video-results",
    "object_key": "videoedit/2026/04/27/video_repair_xxx.mp4"
  },
  "file_path": null,
  "error": null
}
```

失败回调：

```json
{
  "task_id": "caller_task_123",
  "id": "video_repair_xxx",
  "operation": "video_repair",
  "status": "failed",
  "progress": 100,
  "url": null,
  "output_object": null,
  "file_path": null,
  "error": {
    "message": "mask video has 0 frames"
  }
}
```

callback 只作为通知机制，不应作为唯一状态来源。调用方收到 callback 后仍可以用查询接口确认最终状态。

进度建议：

- 1%：请求入队。
- 5%：输入保存、解码校验、窗口规划完成。
- 5%..90%：每完成一个窗口更新一次。
- 90%..98%：拼接、paste-back、编码、音频拷贝。
- 100%：上传或持久化完成。

### 18.4 Helper 与 scheduler 的调用边界

`run_videoedit_long_video()` 不直接调用 pipeline 类，也不绕过 scheduler。它应复用现有 serve 调度链路：

```python
batch = prepare_request(
    server_args=server_args,
    sampling_params=window_sampling_params,
)
save_file_path_list, result = await process_generation_batch(
    async_scheduler_client,
    batch,
)
```

这样可以继续复用：

- SGLang scheduler 队列。
- 分布式 worker。
- 现有输出保存逻辑。
- metrics / peak memory / inference time。
- 后续 LoRA、offload、cache 等服务级能力。

长视频 helper 只负责“多次构造单窗口请求并拼接结果”，不直接接触 DiT/VAE 实例。

### 18.5 响应与下载

创建任务响应示例：

```json
{
  "id": "video_repair_xxx",
  "object": "video",
  "model": "WanVideoEditPipeline",
  "status": "queued",
  "progress": 0,
  "size": "832x480",
  "seconds": "10",
  "quality": "standard",
  "file_path": "/outputs/video_repair_xxx.mp4"
}
```

下载继续使用：

```text
GET /v1/videos/{video_id}/content
```

如果最终结果上传到云存储，则 `url` 非空，`content` endpoint 返回提示让调用方使用 `url`。

### 18.6 serve 参数校验

endpoint 层做 HTTP 语义校验：

- `prompt` 必填。
- 必须提供一组视频输入：`video_file` / `video_input_path` / `video_url` / `video_bucket + video_object_key` 四选一。
- 必须提供一组 mask 输入：`mask_file` / `mask_input_path` / `mask_url` / `mask_bucket + mask_object_key` 四选一。
- 不允许同时提供多个视频来源，避免优先级歧义。
- `infer_len` 默认 81，且 `(infer_len - 1) % 4 == 0`。
- `overlap` 在 `[0, infer_len - 1]` 范围内。
- `width` / `height` 未给时使用 VideoEdit sampling 默认值。
- 当前服务模型不是 `WanVideoEditPipeline` 时返回 400，错误信息明确说明该 endpoint 只支持 VideoEdit 修复模型。

sampling params 层继续做模型语义校验：

- `video_input_path` 必填。
- `mask_input_path` 必填。
- 单窗口请求中 `num_frames == infer_len == 81`。
- dynamic CFG 参数范围合法。
- paste-back 参数范围合法。

### 18.7 文件与模块修改清单

在已有新增文件之外，serve 需要额外修改：

- `python/sglang/multimodal_gen/runtime/entrypoints/openai/protocol.py`
  - 新增 `VideoRepairRequest`

- `python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py`
  - 新增 `/v1/videos/repairs`
  - 新增 `_dispatch_video_repair_job_async`
  - 新增视频 / mask 上传保存 helper，或泛化现有 `save_image_to_path`

- `python/sglang/multimodal_gen/runtime/videoedit/long_video.py`
  - 新增窗口规划、窗口临时文件构造、scheduler 调用、结果拼接

- `python/sglang/multimodal_gen/runtime/videoedit/io.py`
  - 可选新增视频读写、URL/base64/upload 保存工具，避免把视频 IO 全塞进 `video_api.py`

### 18.8 API 示例

JSON 请求，对象存储输入和对象存储输出：

```bash
curl -X POST http://127.0.0.1:30000/v1/videos/repairs \
  -H 'Content-Type: application/json' \
  -d '{
    "task_id": "caller_task_123",
    "prompt": "repair the damaged region naturally",
    "video_bucket": "video-inputs",
    "video_object_key": "jobs/123/input.mp4",
    "mask_bucket": "video-inputs",
    "mask_object_key": "jobs/123/mask.mp4",
    "output_storage": "object_storage",
    "output_bucket": "video-results",
    "output_object_key": "jobs/123/repaired.mp4",
    "callback_url": "https://caller.example.com/videoedit/callback",
    "width": 832,
    "height": 480,
    "infer_len": 81,
    "overlap": 0,
    "enable_paste_back": true,
    "guidance_scale": 7.5,
    "num_inference_steps": 50
  }'
```

JSON 请求，本地路径输入和本地输出：

```bash
curl -X POST http://127.0.0.1:30000/v1/videos/repairs \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "repair the damaged region naturally",
    "video_input_path": "/data/input.mp4",
    "mask_input_path": "/data/mask.mp4",
    "width": 832,
    "height": 480,
    "infer_len": 81,
    "overlap": 0,
    "enable_paste_back": true,
    "output_storage": "local",
    "output_path": "/data/outputs/repaired.mp4",
    "guidance_scale": 7.5,
    "num_inference_steps": 50
  }'
```

multipart 请求：

```bash
curl -X POST http://127.0.0.1:30000/v1/videos/repairs \
  -F prompt='repair the damaged region naturally' \
  -F video_file=@/data/input.mp4 \
  -F mask_file=@/data/mask.mp4 \
  -F extra_body='{"width":832,"height":480,"overlap":0,"enable_paste_back":true}'
```

查询和下载：

```bash
curl http://127.0.0.1:30000/healthz
curl http://127.0.0.1:30000/v1/videos/{video_id}
curl http://127.0.0.1:30000/v1/videos/{video_id}/progress
curl -L http://127.0.0.1:30000/v1/videos/{video_id}/content -o repaired.mp4
curl -X DELETE http://127.0.0.1:30000/v1/videos/{video_id}
```

### 18.9 API 进度查询

`POST /v1/videos/repairs` 返回后，调用方拿到的是异步任务 `id`，不要阻塞等待最终视频。进展查询复用现有 video job 查询接口：

```text
GET /v1/videos/{video_id}
```

高频轮询建议使用轻量进度接口：

```text
GET /v1/videos/{video_id}/progress
```

建议 job 状态字段：

- `status`
  - `queued`：任务已入队，尚未开始处理。
  - `in_progress`：正在读帧、跑窗口或做后处理。
  - `completed`：最终视频已生成。
  - `failed`：任务失败，查看 `error.message`。
  - `deleted`：任务已被删除。
- `progress`
  - 整数百分比，范围 `0..100`。
- `url`
  - 云存储上传成功后的下载地址。
- `file_path`
  - 本地持久化输出路径，仅在未上传云存储且服务允许保留本地文件时返回。
- `metadata_path`
  - 可选，长视频窗口规划和输出映射 metadata。
- `error`
  - 失败原因。

查询示例：

```bash
JOB_ID=video_repair_xxx
curl http://127.0.0.1:30000/v1/videos/${JOB_ID}
```

响应示例：

```json
{
  "id": "video_repair_xxx",
  "object": "video",
  "model": "WanVideoEditPipeline",
  "status": "in_progress",
  "progress": 43,
  "size": "832x480",
  "seconds": "10",
  "file_path": null,
  "error": null
}
```

轻量进度响应示例：

```json
{
  "id": "video_repair_xxx",
  "status": "in_progress",
  "progress": 43,
  "updated_at": 1777286400
}
```

轮询示例：

```bash
JOB_ID=video_repair_xxx

while true; do
  resp=$(curl -s http://127.0.0.1:30000/v1/videos/${JOB_ID})
  status=$(echo "$resp" | jq -r '.status')
  progress=$(echo "$resp" | jq -r '.progress')
  echo "status=${status} progress=${progress}%"

  if [ "$status" = "completed" ]; then
    curl -L http://127.0.0.1:30000/v1/videos/${JOB_ID}/content -o repaired.mp4
    break
  fi

  if [ "$status" = "failed" ]; then
    echo "$resp" | jq -r '.error.message'
    exit 1
  fi

  sleep 5
done
```

如果不依赖 `jq`，可以用 Python 解析：

```bash
echo "$resp" | python -c '
import json
import sys

data = json.load(sys.stdin)
print(data["status"], data["progress"])
'
```

进度更新由后台任务写入 `VIDEO_STORE`：

- 窗口规划完成后写 `progress=5`。
- 第 `i` 个窗口完成后写：

```python
progress = 5 + int((i + 1) / num_windows * 85)
```

- paste-back / 编码 / 音频拷贝阶段写 `progress=90..98`。
- 最终文件可下载后写 `status=completed, progress=100`。

### 18.10 Docker 与 CLI 调用设计

服务端推荐用 Docker 固化运行环境，CLI 分两类：服务端启动 CLI 和任务提交 CLI。

#### 18.10.1 Docker 启动服务

示例：

```bash
docker run --gpus all --rm \
  --name sglang-videoedit \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 30000:30000 \
  -v /root/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model:/models/VideoEdit-diffusers-model:ro \
  -v /data/videoedit/inputs:/data/inputs \
  -v /data/videoedit/outputs:/data/outputs \
  -v /data/videoedit/cache:/root/.cache \
  -e VIDEOEDIT_QUEUE_CAPACITY=1 \
  -e VIDEOEDIT_INPUT_ROOT=/data/inputs \
  -e VIDEOEDIT_OUTPUT_ROOT=/data/outputs \
  -e VIDEOEDIT_STORAGE_BACKEND=s3 \
  -e VIDEOEDIT_STORAGE_ENDPOINT=http://minio:9000 \
  -e VIDEOEDIT_STORAGE_BUCKET=video-results \
  sglang-videoedit:latest \
  sglang serve \
    --model-path /models/VideoEdit-diffusers-model \
    --host 0.0.0.0 \
    --port 30000 \
    --tp-size 1 \
    --output-path /data/outputs \
    --input-save-path /data/inputs \
    --transformer-path /models/VideoEdit-diffusers-model/transformer
```

约定：

- `/data/inputs` 是服务端临时输入目录，上传文件、URL 下载和对象存储下载都落到这里。
- `/data/outputs` 是本地输出目录，仅在 `output_storage=local` 或调试时长期保留结果。
- `VIDEOEDIT_QUEUE_CAPACITY=1` 是 VideoEdit 业务 admission 队列，不替代 SGLang 底层 scheduler 参数。
- 对象存储密钥不要写进文档命令，生产环境通过 secret / env 注入。

#### 18.10.2 启动服务

复用当前 multimodal_gen 已有 `serve` 子命令：

```bash
sglang serve \
  --model-path /root/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model \
  --host 0.0.0.0 \
  --port 30000 \
  --tp-size 1 \
  --output-path /data/outputs \
  --input-save-path /data/inputs \
  --transformer-path /root/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model/transformer
```

其中：

- `--model-path` 指向 overlay 模型目录，`model_index.json` 中 `_class_name` 必须是 `WanVideoEditPipeline`。
- `--transformer-path` 是可选组件覆盖，用于加载 VideoEdit finetuned transformer。
- 不要使用 `--video-path` / `--mask-path` 这种名字传业务输入，避免被 `ServerArgs._extract_component_paths()` 当作组件路径。
- 服务启动时应注册 VideoEdit admission 配置，默认 `queue_capacity=1`。

#### 18.10.3 远程提交任务 CLI

建议新增轻量客户端：

- `python/sglang/multimodal_gen/runtime/videoedit/cli.py`

命令形态：

```bash
python -m sglang.multimodal_gen.runtime.videoedit.cli repair-remote \
  --base-url http://127.0.0.1:30000 \
  --prompt "repair the damaged region naturally" \
  --video-input-path /data/input.mp4 \
  --mask-input-path /data/mask.mp4 \
  --width 832 \
  --height 480 \
  --infer-len 81 \
  --overlap 0 \
  --guidance-scale 7.5 \
  --num-inference-steps 50 \
  --callback-url https://caller.example.com/videoedit/callback \
  --enable-paste-back \
  --wait \
  --poll-interval 5 \
  --output repaired.mp4
```

`repair-remote` 的逻辑：

1. 将 CLI 参数组装成 `VideoRepairRequest` JSON。
2. `POST {base_url}/v1/videos/repairs`。
3. 打印返回的 `job_id`。
4. 如果传 `--wait`，循环调用 `GET /v1/videos/{job_id}`。
5. `completed` 后调用 `/content` 下载到 `--output`。
6. `failed` 时打印 `error.message` 并返回非零退出码。

如果使用上传文件而不是服务端本地路径：

```bash
python -m sglang.multimodal_gen.runtime.videoedit.cli repair-remote \
  --base-url http://127.0.0.1:30000 \
  --prompt "repair the damaged region naturally" \
  --video-file ./input.mp4 \
  --mask-file ./mask.mp4 \
  --width 832 \
  --height 480 \
  --wait \
  --output repaired.mp4
```

#### 18.10.4 离线本地 CLI

MVP 阶段还建议提供本地 wrapper CLI，不依赖 HTTP：

```bash
python -m sglang.multimodal_gen.runtime.videoedit.cli repair \
  --model-path /root/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model \
  --transformer-path /root/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model/transformer \
  --prompt "repair the damaged region naturally" \
  --video-input-path /data/input.mp4 \
  --mask-input-path /data/mask.mp4 \
  --output-path /data/outputs \
  --output-file-name repaired.mp4 \
  --width 832 \
  --height 480 \
  --infer-len 81 \
  --overlap 0 \
  --guidance-scale 7.5 \
  --num-inference-steps 50 \
  --enable-paste-back
```

`repair` 的逻辑：

1. 构造 `ServerArgs`。
2. 启动本地 `DiffGenerator.from_pretrained(..., local_mode=True)`。
3. 调用 `run_videoedit_long_video()`。
4. 将最终文件写入 `--output-path/--output-file-name`。
5. 输出 metadata 路径和窗口统计。

#### 18.10.5 与通用 `sglang generate` 的关系

当前通用 `sglang generate` 的 CLI 参数注册只基于基类 `SamplingParams`，无法稳定暴露 `video_input_path`、`mask_input_path`、`overlap` 等 VideoEdit 专属参数。因此：

- MVP 使用 `videoedit.cli repair` 或 `repair-remote`。
- 等 `generate_cmd()` 支持根据 `model_path` 动态加载模型专属 `sampling_param_cls` 后，再允许：

```bash
sglang generate \
  --model-path /root/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model \
  --prompt "repair the damaged region naturally" \
  --video-input-path /data/input.mp4 \
  --mask-input-path /data/mask.mp4 \
  --infer-len 81 \
  --overlap 0 \
  --enable-paste-back \
  --output-path /data/outputs
```

在这个动态 CLI 改造完成前，不建议把长视频修复直接挂到通用 `sglang generate`。

## 19. 数据流与接口契约

### 19.1 请求入口

API / CLI 传入：

- `prompt`
- `negative_prompt`
- `video_input_path`
- `mask_input_path`
- `infer_len`
- `guidance_scale`
- `dynamic_cfg`
- `dynamic_cfg_max_step`
- `dynamic_cfg_min`
- 预处理参数

### 19.2 各 stage 的关键字段

`TextEncodingStage` 后：

- `batch.prompt_embeds`
- `batch.negative_prompt_embeds`

`VideoEditConditionStage` 后：

- `batch.image_latent`
- `batch.extra["videoedit"]["video_latents"]`
- `batch.extra["videoedit"]["post_meta"]`

`LatentPreparationStage` 后：

- `batch.latents`

`VideoEditTimestepPreparationStage` 后：

- `batch.timesteps`
- `batch.num_inference_steps` 已按 `strength` 更新为有效 step 数

`VideoEditLatentInitStage` 后：

- `batch.latents` 已从纯噪声替换为基于 `video_latents` 的 noisy latent

`VideoEditDenoisingStage` 后：

- `batch.latents` 为最终去噪 latent

`DecodingStage` 后：

- `batch.output`

### 19.3 Contract 原则

每层之间都只通过标准张量字段和 `batch.extra["videoedit"]` 交换信息：

- 通用字段放标准字段
- VideoEdit 专属元数据只放 `batch.extra["videoedit"]`
- 不新增一堆散落在 `Req` 顶层的临时字段

这能把模型私有上下文限制在一个命名空间内，避免 future merge 时污染全局 request schema。

## 20. Upstream 同步策略

这是长期维护里最重要的一部分。

### 20.1 同步来源拆分

后续同步应分三类来源：

1. Wan 通用 upstream
   - VAE
   - DiT
   - 通用 pipeline/stage
   - 分布式优化

2. VideoEdit upstream
   - 条件组装公式
   - scheduler 逻辑
   - preprocess/postprocess 算法
   - dynamic CFG 策略

3. SGLang 内部框架演进
   - `Req` / stage API
   - CLI / config 注册
   - loader / offload / executor

### 20.2 如何保持可合并

建议遵守以下规则：

- Wan 主干代码不改或只做通用能力修复
- VideoEdit 差异全部落在 adapter 层
- preprocess/postprocess 写成纯函数，便于用 reference fixture 回归
- scheduler 单独一层 adapter，避免未来 stage API 变化时牵连业务逻辑
- 增加 reference alignment tests，而不是只看最终视频

### 20.3 推荐的同步路径

未来若 `VideoEdit-diffusers` 更新：

1. 先对比 `pipeline_wan_edit.py` 的 `__call__`
2. 如果变化只在条件组装，更新 `runtime/videoedit/preprocess.py` 或 `VideoEditDenoisingStage`
3. 如果变化只在 scheduler，更新 `videoedit_flow_match.py`
4. 如果变化只在后处理，更新 `runtime/videoedit/postprocess.py`
5. Wan 主干无改动则不动 VAE/DiT

未来若 SGLang Wan 升级：

1. 优先合入通用 Wan VAE/DiT/pipeline 优化
2. 检查 `batch.image_latent` contract 是否仍成立
3. 检查 `DenoisingStage` hook 和 scheduler API 是否变化
4. 只在 adapter 层做兼容修复

### 20.4 当前 SGLang 代码约束

实现时需要特别留意当前框架的这些具体约束：

- `ServerArgs._extract_component_paths()` 会把未知的 `--<name>-path` 全部解析成组件路径。`--video-input-path` / `--mask-input-path` 只有在 wrapper CLI 或动态模型专属 CLI 中被注册为已知参数时才安全；在当前通用 `sglang generate` 中作为 unknown args 传入仍会被误判为组件路径。
- `generate.py` 当前只用基类 `SamplingParams` 注册 CLI 字段，所以 `sglang generate` 在动态 CLI 改造前不能稳定接收 VideoEdit 专属字段。
- `ComposedPipelineBase.load_modules()` 会按 `model_index.json` 的 key 加载 required modules。overlay 中 `scheduler` key 不能省略；如果 runtime 要替换 scheduler，可以让该 key 指向轻量占位目录或 `null`，但需要确认 loader 对 `null` 的 required module 处理符合预期。
- `DenoisingStage._preprocess_sp_latents()` 会同时 shard `batch.latents` 和 `batch.image_latent`。VideoEdit 的条件 latent 必须在这个阶段之前已经写入 `batch.image_latent`。
- `DecodingStage` 只做 latent decode 和 `PipelineConfig.post_decoding()`。paste-back、音频拷贝和长视频拼接仍然留在 helper/API 层。

## 21. 需要新增 / 修改的文件

### 21.1 新增

- `python/sglang/multimodal_gen/runtime/pipelines/wan_videoedit_pipeline.py`
- `python/sglang/multimodal_gen/runtime/models/schedulers/videoedit_flow_match.py`
- `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/videoedit_wan.py`
- `python/sglang/multimodal_gen/configs/pipeline_configs/videoedit_wan.py`
- `python/sglang/multimodal_gen/configs/sample/videoedit_wan.py`
- `python/sglang/multimodal_gen/runtime/videoedit/contracts.py`
- `python/sglang/multimodal_gen/runtime/videoedit/preprocess.py`
- `python/sglang/multimodal_gen/runtime/videoedit/postprocess.py`
- `python/sglang/multimodal_gen/runtime/videoedit/long_video.py`
- `python/sglang/multimodal_gen/runtime/videoedit/io.py`
- `python/sglang/multimodal_gen/runtime/videoedit/cli.py`
- `python/sglang/multimodal_gen/test/videoedit/test_videoedit_scheduler.py`
- `python/sglang/multimodal_gen/test/videoedit/test_videoedit_preprocess.py`
- `python/sglang/multimodal_gen/test/videoedit/test_videoedit_windowing.py`
- `python/sglang/multimodal_gen/test/videoedit/test_videoedit_cli.py`
- `python/sglang/multimodal_gen/test/videoedit/test_videoedit_serve.py`
- `python/sglang/multimodal_gen/test/videoedit/test_videoedit_api.py`

### 21.2 修改

- `python/sglang/multimodal_gen/configs/pipeline_configs/__init__.py`
- `python/sglang/multimodal_gen/configs/sample/__init__.py`
- `python/sglang/multimodal_gen/registry.py`
- `python/sglang/multimodal_gen/runtime/entrypoints/openai/protocol.py`
- `python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py`
- `python/sglang/multimodal_gen/runtime/pipelines/__init__.py` 如当前项目导出策略要求显式导入新 pipeline
- 视正式 CLI 接入时机决定是否修改：
  - `python/sglang/multimodal_gen/runtime/entrypoints/cli/generate.py`
  - `python/sglang/multimodal_gen/runtime/server_args.py`

## 22. 注册策略

在 `registry.py` 中新增：

```python
register_configs(
    sampling_param_cls=WanVideoEditSamplingParams,
    pipeline_config_cls=WanVideoEditPipelineConfig,
    hf_model_paths=[
        "VideoEdit-diffusers",
        "Wan2.1-VideoEdit-Diffusers",
    ],
    model_detectors=[
        lambda s: "videoedit" in s.lower(),
    ],
)
```

真正决定 pipeline class 的仍然是 overlay 模型目录里的：

```json
{
  "_class_name": "WanVideoEditPipeline"
}
```

同时，`wan_videoedit_pipeline.py` 中应声明：

```python
pipeline_config_cls = WanVideoEditPipelineConfig
sampling_params_cls = WanVideoEditSamplingParams
EntryClass = WanVideoEditPipeline
```

这样可以覆盖两条路径：

- overlay diffusers 目录通过 `_class_name` 找到 native pipeline。
- safetensors / 本地非标准模型路径可以通过 pipeline class 上的 config class 声明兜底。

## 23. 代码功能性测试方案

本接入不能只做逐层数值对齐。第一批验收目标必须是“代码真的能从用户入口跑起来”，否则后面的 scheduler、preprocess、latent 对齐测试容易变成孤立单测，无法证明 CLI / serve 调度链路可用。

测试分三层推进：

1. P0 smoke tests：不要求真实生成质量，先证明 CLI、serve 和 API 路由能启动、能解析 VideoEdit 参数、能把请求送到正确 pipeline / helper。
2. P1 functional tests：使用小型 synthetic 视频 / mask，证明单窗口请求、长视频窗口规划、请求状态机和输出文件 contract 正确。
3. P2 alignment tests：在有 reference 权重和 fixture 的环境中，与 `../VideoEdit-diffusers` 做逐层数值对齐。

### 23.1 P0：CLI / serve 先跑通

这一级是第一阶段必须先补的功能性测试。要求测试不依赖真实大模型权重、不下载模型、不要求 GPU 推理完成；可以通过 monkeypatch / mock 隔离重模型加载，但必须覆盖真实入口函数和 argparse / FastAPI 路由。

新增：

- `python/sglang/multimodal_gen/test/videoedit/test_videoedit_cli.py`
- `python/sglang/multimodal_gen/test/videoedit/test_videoedit_serve.py`

`test_videoedit_cli.py` 覆盖：

- `python -m sglang.multimodal_gen.runtime.videoedit.cli repair --help` 返回 0，并展示 `--video-input-path`、`--mask-input-path`、`--overlap`、`--enable-paste-back`。
- `repair` 能解析本地离线参数，并构造 `ServerArgs` 与 `WanVideoEditSamplingParams`；mock `DiffGenerator.from_pretrained()` 和 `run_videoedit_long_video()`，断言二者被调用一次。
- `repair-remote` 能解析远程提交参数；mock HTTP client，断言请求发送到 `/v1/videos/repairs`，请求体包含 `video_input_path`、`mask_input_path`、`infer_len`、`overlap`。
- `--video-input-path` / `--mask-input-path` 不会进入 `ServerArgs.component_paths`。这是必须显式断言的回归点，因为当前通用 unknown args 会把 `--<name>-path` 误解析成组件覆盖。
- `repair --dry-run` 或等价调试模式只做参数解析、模型解析和窗口规划，不启动模型；这个模式用于 CI 快速验证 CLI 契约。

`test_videoedit_serve.py` 覆盖：

- `sglang serve --model-path <overlay>` 的参数能被 `ServeSubcommand` / `ServerArgs.from_cli_args()` 解析；mock `launch_server()`，断言 `model_path`、`host`、`port`、`output_path`、`input_save_path`、`component_paths["transformer"]` 正确。
- overlay `model_index.json` 中 `_class_name = "WanVideoEditPipeline"` 时，registry 能解析到 `WanVideoEditPipeline`，且 pipeline class 暴露 `pipeline_config_cls` 和 `sampling_params_cls`。
- 启动 serve 时如果模型不是 `WanVideoEditPipeline`，`/v1/videos/repairs` 返回 400；如果是 VideoEdit 模型，route 存在且能完成 admission 前的参数校验。
- `/healthz` 在 mock model loaded 状态下返回 200，响应包含 `model_loaded`、`queue_capacity`、`active_jobs`、`queued_jobs`。
- admission 队列容量为 1 时，已有 queued / in_progress VideoEdit job 会让第二个请求返回 429。

建议命令：

```bash
pytest -q python/sglang/multimodal_gen/test/videoedit/test_videoedit_cli.py
pytest -q python/sglang/multimodal_gen/test/videoedit/test_videoedit_serve.py
```

通过标准：

- 两个测试文件在 CPU-only 环境可运行。
- 不访问 Hugging Face / 对象存储 / 原 `../VideoEdit-diffusers` 仓库。
- 不启动真实 DiT / VAE 权重加载。
- 失败信息能定位到 CLI 参数、serve 参数、route 注册、admission 或 registry 中的具体一层。

### 23.2 P1：单窗口与 API 功能测试

新增：

- `python/sglang/multimodal_gen/test/videoedit/test_videoedit_api.py`

使用临时目录生成 synthetic 81 帧小视频和同帧数 mask，mock heavy generation，但保留真实请求解析、任务登记、状态更新和输出 contract。

覆盖：

- `POST /v1/videos/repairs` JSON 本地路径请求返回 `VideoResponse(status="queued")`，job record 中有 `operation="video_repair"`。
- multipart 上传请求会保存 `video_file` 和 `mask_file` 到 `input_save_path` 或临时目录。
- 只提供视频、不提供 mask 时返回 400。
- 同时提供多个视频来源时返回 400，避免优先级歧义。
- `infer_len=80`、`overlap>=infer_len`、`strength` 越界等参数返回 400。
- mock 后台任务完成后，`GET /v1/videos/{id}` 返回 `completed`，`file_path` 或 `url` 至少一个非空。
- `GET /v1/videos/{id}/progress` 返回轻量进度结构。
- `DELETE /v1/videos/{id}` 能删除 queued job；in_progress job 标记 `cancel_requested=True`。

建议命令：

```bash
pytest -q python/sglang/multimodal_gen/test/videoedit/test_videoedit_api.py
```

### 23.3 P1：本地 helper 功能测试

已有计划中的：

- `test_videoedit_windowing.py`
- `test_videoedit_preprocess.py`

需要补充功能性断言，而不是只做算法单点：

- `build_window_specs()` 覆盖 `N=20`、`N=79`、`N=81`、`N=99`、`overlap=8`。
- 每个真实全局帧必须至少被一个 `commit_local_to_global` 覆盖。
- 反射补齐帧默认不提交；短视频需要提交反射位置时，优先选择非 local 0。
- helper 生成的 window video / mask 帧数都等于 `infer_len`。
- window 0 的 mask 首帧为全黑，window local 1..80 保留真实 mask。

### 23.4 P2：真实模型端到端测试

真实模型测试放到 manual / nightly，不能阻塞 CPU-only 单测。

建议新增 registered 或 manual 测试：

- 启动 `sglang serve --model-path /root/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model --transformer-path /root/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model/transformer`。
- 轮询 `/healthz` 到 `model_loaded=true`。
- 提交 81 帧本地路径修复请求。
- 轮询 `/v1/videos/{id}` 到 `completed`。
- 下载 `/content`，验证输出是合法 mp4、帧数为 81、分辨率匹配。
- 再提交一个短视频请求，验证长视频 helper 的反射补齐和最终输出帧数等于原始帧数。

建议命令：

```bash
SGLANG_VIDEOEDIT_MODEL=/root/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model \
SGLANG_VIDEOEDIT_TRANSFORMER=/root/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model/transformer \
pytest -q test/manual/videoedit/test_videoedit_e2e.py
```

### 23.5 测试分层与 CI 策略

| 层级 | 文件 | 是否需要 GPU | 是否需要真实权重 | CI 默认 |
| --- | --- | --- | --- | --- |
| P0 CLI smoke | `test_videoedit_cli.py` | 否 | 否 | 是 |
| P0 serve smoke | `test_videoedit_serve.py` | 否 | 否 | 是 |
| P1 API contract | `test_videoedit_api.py` | 否 | 否 | 是 |
| P1 window/preprocess | `test_videoedit_windowing.py` / `test_videoedit_preprocess.py` | 否 | 否 | 是 |
| P1 scheduler | `test_videoedit_scheduler.py` | 否 | 否 | 是 |
| P2 e2e | `test/manual/videoedit/test_videoedit_e2e.py` | 是 | 是 | 手动 / nightly |
| P2 reference alignment | side-by-side scripts | 是 | 是 | 手动 / nightly |

### 23.6 首次合入的最低验收线

第一批代码合入前，最低验收线是：

1. `videoedit.cli repair --dry-run` 能跑通参数解析、overlay 解析、窗口规划。
2. `videoedit.cli repair-remote --dry-run` 能生成正确 `/v1/videos/repairs` 请求体。
3. `sglang serve` 能解析 overlay 模型目录和 `--transformer-path`，mock `launch_server()` 下不报错。
4. FastAPI app 中 `/v1/videos/repairs` 和 `/healthz` route 可见。
5. P0 两个 smoke test 在 CPU-only 环境通过。

这条线通过后，再进入 scheduler、condition stage、latent init、denoising 和 reference alignment 的实现。不要等真实模型完整生成后才补 CLI / serve 测试。

## 24. 风险与解决方案

### 24.1 P0 风险

| 风险 | 解决方案 | 验收方式 |
| --- | --- | --- |
| pipeline 选错 | 必须提供 overlay `model_index.json`，并固定 `_class_name = "WanVideoEditPipeline"` | `get_model_info()` 返回 `WanVideoEditPipeline` |
| scheduler 与通用 stage 不兼容 | 用 `VideoEditFlowMatchScheduler` 适配，不直接把 reference scheduler 塞入 runtime | 单测比较 `timesteps` / `sigmas` / `step()` / `add_noise()` |
| 输入参数与 component path 冲突 | 业务参数改名为 `video_input_path` / `mask_input_path`，并优先走专用 wrapper CLI | CLI 单测确认不进入 `component_paths` |
| transformer 通道数错误 | 启动时 fail-fast 校验 `in_channels=36`, `out_channels=16` | 错误权重加载时直接报错 |
| `strength < 1.0` 步数错误 | `VideoEditTimestepPreparationStage` 同时裁剪 `timesteps` 并更新 `batch.num_inference_steps` | 与 reference 比较裁剪后的 timestep 序列和有效 step 数 |
| CLI / serve 入口不可用 | 先补 P0 smoke tests，mock 重模型加载但覆盖真实 parser、registry、route | `test_videoedit_cli.py`、`test_videoedit_serve.py` CPU-only 通过 |

### 24.2 P1 风险

| 风险 | 解决方案 | 验收方式 |
| --- | --- | --- |
| mask packing 语义错误 | 固定 preserve=1、inpaint=0，并对首帧做黑帧约束 | synthetic mask 单测 |
| VAE 归一化错误 | 统一走 Wan latent mean/std 归一化 | `cond_latents` / `video_latents` 与 reference 对齐 |
| `video_latents` 加噪时机错误 | 独立 `VideoEditLatentInitStage` | 比较首步 denoising 输入 |
| dynamic CFG 不一致 | `VideoEditDenoisingStage` 单独实现 CFG hook | 比较每步 CFG 序列和首步 `noise_pred` |

### 24.3 P2 风险

| 风险 | 解决方案 | 验收方式 |
| --- | --- | --- |
| 长视频逻辑污染 runtime core | 滑窗和 paste-back 放 helper 层 | 单窗口 native pipeline 保持稳定 |
| future merge 难 | 差异都下沉到 adapter 层 | 未来升级只改局部文件 |
| 回归定位困难 | 增加 side-by-side dump | 逐层对齐而不是只看最终视频 |

## 25. 准确性验收

必须和 `../VideoEdit-diffusers` 做同 seed、同 prompt、同窗口输入的逐层对齐。

建议验收顺序：

1. 文本编码
   - `prompt_embeds`
   - `negative_prompt_embeds`

2. 预处理
   - bbox
   - resize 后尺寸
   - `cond_masks`

3. VAE 编码
   - `cond_latents`
   - `video_latents`

4. scheduler
   - `timesteps`
   - `sigmas`
   - `strength < 1.0` 后的有效 `num_inference_steps`
   - `add_noise()` 输出

5. 首步 DiT
   - `latent_model_input`
   - `noise_pred`
   - `dynamic_cfg` 每步 `current_cfg` / `do_cfg` 序列

6. 最终 latent
   - `latents`

7. 解码与后处理
   - crop-only 结果
   - paste-back 结果

## 26. 实施顺序

1. 准备 overlay 模型目录，固定 `_class_name = "WanVideoEditPipeline"`。
2. 新增最小 `WanVideoEditPipelineConfig` / `WanVideoEditSamplingParams` / `WanVideoEditPipeline` 空壳注册，先让 registry 能识别模型。
3. 新增 `videoedit.cli repair --dry-run` / `repair-remote --dry-run`，只做参数解析、overlay 解析和请求体构造。
4. 新增 `/healthz` 和 `/v1/videos/repairs` 的最小 route，先只做模型类型检查、admission 和参数校验。
5. 补齐 P0 smoke tests：`test_videoedit_cli.py`、`test_videoedit_serve.py`，确保 CLI / serve 在 CPU-only、mock 重模型加载条件下通过。
6. 实现 `VideoEditFlowMatchScheduler` adapter 和 `VideoEditTimestepPreparationStage`，并完成 scheduler / strength 对齐测试。
7. 落地 `runtime/videoedit/contracts.py`、`preprocess.py`、`postprocess.py`，把纯函数从原 repo 解耦出来。
8. 实现 `runtime/videoedit/long_video.py`，先用本地 helper 的 dry-run / mocked generation 跑通任意长度窗口规划，验证每帧都有输出覆盖。
9. 补齐 P1 API / helper 功能测试：JSON、multipart、本地路径、短视频、81 帧、尾窗口不足、video/mask 帧数不一致。
10. 实现 `VideoEditConditionStage`，完成 `batch.image_latent` 和 `video_latents` 生产。
11. 实现 `VideoEditLatentInitStage`，先在关闭 dynamic CFG 条件下跑通单窗口。
12. 实现 `VideoEditDenoisingStage`，补齐 dynamic CFG 和 negative pass 控制。
13. 接入 paste-back、最终视频编码和音频拷贝。
14. 完成真实模型 manual / nightly e2e：serve 启动、提交任务、轮询完成、下载 mp4、检查帧数和分辨率。
15. 增加 side-by-side 对齐测试和回归脚本。

## 27. 最终结论

VideoEdit 接入 SGLang 的正确方式，不是把原仓库整套搬进来，而是把它拆成：

- 可复用的 Wan 通用主干
- 最小化的 VideoEdit adapter 层
- 独立的应用编排与后处理层

最终形态应满足：

- runtime 不依赖 `../VideoEdit-diffusers`
- Wan 主干和 SGLang 通用 stage 最大化复用
- VideoEdit 差异被限制在 scheduler、condition stage、denoising hook、纯函数预处理/后处理
- 后续无论同步 SGLang 还是同步 VideoEdit upstream，都能局部更新、低冲突合并

这就是本方案的核心目标：模块边界清晰、可配置、松耦合，并且对 future merge 友好。
