# VideoEdit-diffusers 接入 SGLang Diffusion 方案

## 1. 背景与目标

目标是把 `../VideoEdit-diffusers` 中基于 Wan2.1 的视频编辑 / inpainting 模型接入 `python/sglang/multimodal_gen`，但集成后的 SGLang 实现必须满足以下约束：

1. 运行时不依赖原 `VideoEdit-diffusers` 仓库的目录、工具函数、私有数据结构和脚本调用。
2. 优先复用 SGLang 已有的 Wan VAE、Wan DiT、通用 pipeline/stage、分布式和解码能力，只补 VideoEdit 专属的数据组装、scheduler 适配和 denoising hook。
3. 预处理、模型推理、后处理三层解耦，后续无论是 SGLang 升级还是 VideoEdit upstream 更新，都能局部同步而不是整体重写。
4. 文档中明确接口层级、数据流、模块边界和 upstream 对齐方式，便于维护和自动化回归。

参考 skill：
`python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-add-model/SKILL.md`

本方案先定义原生 SGLang pipeline 的集成设计，后续实现按该设计推进。

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
- 以标准 `TextEncodingStage` / `LatentPreparationStage` / `TimestepPreparationStage` / `DecodingStage` 为主体
- 仅新增 3 个 VideoEdit 专属扩展点：
  - `VideoEditConditionStage`
  - `VideoEditLatentInitStage`
  - `VideoEditDenoisingStage`

推荐 stage 链路：

```text
InputValidationStage
  -> TextEncodingStage
  -> VideoEditConditionStage
  -> LatentPreparationStage
  -> TimestepPreparationStage
  -> VideoEditLatentInitStage
  -> VideoEditDenoisingStage
  -> DecodingStage
  -> optional postprocess/helper
```

这条链路满足两件事：

- Wan 通用能力保持完全复用
- VideoEdit 专属逻辑被压缩在条件准备、首步 latent 初始化和 CFG 计算这三个局部模块中

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

4. `VideoEditLatentInitStage`
   - 只负责在 denoising 前调用 `scheduler.add_noise(video_latents, noise, first_timestep)`

5. `VideoEditDenoisingStage`
   - 只负责 dynamic CFG 和少量 VideoEdit 特殊 hook

6. VideoEdit 纯函数工具模块
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

但不要把业务输入命名成 `--video-path` / `--mask-path` 后再依赖 unknown args 解析。因为当前 `ServerArgs._extract_component_paths()` 会把任意 `--<name>-path` 识别成组件路径覆盖。

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

原因是这样可以彻底避开当前 CLI 中 `--<name>-path` 被误识别为 component path 的问题。

### 9.2 参数校验

`WanVideoEditSamplingParams.__post_init__()` 中应显式校验：

- `video_input_path` 必填
- `mask_input_path` 必填
- `num_frames == infer_len` 或明确规定二者的关系
- `(infer_len - 1) % 4 == 0`
- `strength` 范围合法
- 输入视频和 mask 帧数一致
- 当前 native pipeline 仅支持单窗口时，禁止传入长视频滑窗参数组合

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
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True
```

这样做的好处：

- 不会误走通用 I2V/TI2V 图像输入分支
- 仍保持视频输出任务语义
- 复用 Wan 的 latent 形状和解码流程

如果后续需要更明确的任务类型，可以再引入新的 `ModelTaskType.VIDEO_EDIT`。但在第一阶段，为了最小侵入和低风险，建议先不修改全局 task enum。

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
        self.add_standard_timestep_preparation_stage()
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

## 12. Scheduler 适配层

新增：

`python/sglang/multimodal_gen/runtime/models/schedulers/videoedit_flow_match.py`

职责不是重新发明 scheduler，而是适配 SGLang 的 stage 协议。

必须对齐的接口：

- `set_timesteps(..., device=None, **kwargs)`
- `set_begin_index()`，即使只是 no-op
- `step(..., return_dict=False)` 返回 `(prev_sample,)`
- `add_noise(original_samples, noise, timestep)`

必须对齐的行为：

- sigma 公式
- timestep 序列
- `extra_one_step=True`
- `shift=5`
- `strength < 1.0` 时的 `get_timesteps()`

适配层边界：

- 不修改通用 `TimestepPreparationStage`
- 不修改通用 `DenoisingStage` 的 scheduler 约定
- 让 scheduler 自己满足通用 stage 的调用要求

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
- VAE encode 使用 Wan 的 mean/std 归一化
- `cond_latents` / `video_latents` 与 reference shape、dtype、统计量一致

## 14. VideoEditLatentInitStage 设计

职责很单一：

- 在 `LatentPreparationStage` 和 `TimestepPreparationStage` 之后
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

`VideoEditDenoisingStage` 应继承 `DenoisingStage`，只覆盖最少的 hook。

建议保留的通用能力：

- CFG parallel / SP / TP
- scheduler.step
- offload
- profile
- trajectory latents

只覆盖以下逻辑：

1. 每步 guidance 计算

```python
current_cfg, do_cfg = calc_current_cfg(
    max_cfg=batch.guidance_scale,
    current_step=step_index,
    max_step=batch.dynamic_cfg_max_step,
    min_cfg=batch.dynamic_cfg_min,
    dynamic_cfg=batch.dynamic_cfg,
)
```

2. 构造 DiT 输入

```python
latent_model_input = torch.cat(
    [ctx.latents, batch.image_latent],
    dim=1,
)
```

3. `do_cfg=False` 时跳过 negative pass

这样做的原因：

- SGLang 通用 `DenoisingStage` 已支持 `batch.image_latent` 拼接
- VideoEdit 的特殊点主要是 dynamic CFG，而不是完整 denoising 流程重写

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

## 17. 长视频策略

长视频滑窗不应放进 native pipeline。

建议明确拆分：

### 17.1 Native pipeline 的职责

- 只处理单窗口
- 输入长度固定为 `infer_len`
- 输出该窗口的 crop-only 编辑结果

### 17.2 Helper / 应用层职责

- 全局预处理
- 统一 bbox
- 按窗口切分
- 多窗口逐次调用 native pipeline
- 结果拼接与 paste-back

这样可以保证：

- pipeline 本身稳定、可测试、与推理内核强相关
- 长视频编排独立演进，不污染 runtime core

## 18. 数据流与接口契约

### 18.1 请求入口

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

### 18.2 各 stage 的关键字段

`TextEncodingStage` 后：

- `batch.prompt_embeds`
- `batch.negative_prompt_embeds`

`VideoEditConditionStage` 后：

- `batch.image_latent`
- `batch.extra["videoedit"]["video_latents"]`
- `batch.extra["videoedit"]["post_meta"]`

`LatentPreparationStage` 后：

- `batch.latents`

`TimestepPreparationStage` 后：

- `batch.timesteps`

`VideoEditLatentInitStage` 后：

- `batch.latents` 已从纯噪声替换为基于 `video_latents` 的 noisy latent

`VideoEditDenoisingStage` 后：

- `batch.latents` 为最终去噪 latent

`DecodingStage` 后：

- `batch.output`

### 18.3 Contract 原则

每层之间都只通过标准张量字段和 `batch.extra["videoedit"]` 交换信息：

- 通用字段放标准字段
- VideoEdit 专属元数据只放 `batch.extra["videoedit"]`
- 不新增一堆散落在 `Req` 顶层的临时字段

这能把模型私有上下文限制在一个命名空间内，避免 future merge 时污染全局 request schema。

## 19. Upstream 同步策略

这是长期维护里最重要的一部分。

### 19.1 同步来源拆分

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

### 19.2 如何保持可合并

建议遵守以下规则：

- Wan 主干代码不改或只做通用能力修复
- VideoEdit 差异全部落在 adapter 层
- preprocess/postprocess 写成纯函数，便于用 reference fixture 回归
- scheduler 单独一层 adapter，避免未来 stage API 变化时牵连业务逻辑
- 增加 reference alignment tests，而不是只看最终视频

### 19.3 推荐的同步路径

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

## 20. 需要新增 / 修改的文件

### 20.1 新增

- `python/sglang/multimodal_gen/runtime/pipelines/wan_videoedit_pipeline.py`
- `python/sglang/multimodal_gen/runtime/models/schedulers/videoedit_flow_match.py`
- `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/videoedit_wan.py`
- `python/sglang/multimodal_gen/configs/pipeline_configs/videoedit_wan.py`
- `python/sglang/multimodal_gen/configs/sample/videoedit_wan.py`
- `python/sglang/multimodal_gen/runtime/videoedit/contracts.py`
- `python/sglang/multimodal_gen/runtime/videoedit/preprocess.py`
- `python/sglang/multimodal_gen/runtime/videoedit/postprocess.py`

### 20.2 修改

- `python/sglang/multimodal_gen/configs/pipeline_configs/__init__.py`
- `python/sglang/multimodal_gen/configs/sample/__init__.py`
- `python/sglang/multimodal_gen/registry.py`
- 视正式 CLI 接入时机决定是否修改：
  - `python/sglang/multimodal_gen/runtime/entrypoints/cli/generate.py`
  - `python/sglang/multimodal_gen/runtime/server_args.py`

## 21. 注册策略

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

## 22. 风险与解决方案

### 22.1 P0 风险

| 风险 | 解决方案 | 验收方式 |
| --- | --- | --- |
| pipeline 选错 | 必须提供 overlay `model_index.json`，并固定 `_class_name = "WanVideoEditPipeline"` | `get_model_info()` 返回 `WanVideoEditPipeline` |
| scheduler 与通用 stage 不兼容 | 用 `VideoEditFlowMatchScheduler` 适配，不直接把 reference scheduler 塞入 runtime | 单测比较 `timesteps` / `sigmas` / `step()` / `add_noise()` |
| 输入参数与 component path 冲突 | 业务参数改名为 `video_input_path` / `mask_input_path`，并优先走专用 wrapper CLI | CLI 单测确认不进入 `component_paths` |
| transformer 通道数错误 | 启动时 fail-fast 校验 `in_channels=36`, `out_channels=16` | 错误权重加载时直接报错 |

### 22.2 P1 风险

| 风险 | 解决方案 | 验收方式 |
| --- | --- | --- |
| mask packing 语义错误 | 固定 preserve=1、inpaint=0，并对首帧做黑帧约束 | synthetic mask 单测 |
| VAE 归一化错误 | 统一走 Wan latent mean/std 归一化 | `cond_latents` / `video_latents` 与 reference 对齐 |
| `video_latents` 加噪时机错误 | 独立 `VideoEditLatentInitStage` | 比较首步 denoising 输入 |
| dynamic CFG 不一致 | `VideoEditDenoisingStage` 单独实现 CFG hook | 比较每步 CFG 序列和首步 `noise_pred` |

### 22.3 P2 风险

| 风险 | 解决方案 | 验收方式 |
| --- | --- | --- |
| 长视频逻辑污染 runtime core | 滑窗和 paste-back 放 helper 层 | 单窗口 native pipeline 保持稳定 |
| future merge 难 | 差异都下沉到 adapter 层 | 未来升级只改局部文件 |
| 回归定位困难 | 增加 side-by-side dump | 逐层对齐而不是只看最终视频 |

## 23. 准确性验收

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
   - `add_noise()` 输出

5. 首步 DiT
   - `latent_model_input`
   - `noise_pred`

6. 最终 latent
   - `latents`

7. 解码与后处理
   - crop-only 结果
   - paste-back 结果

## 24. 实施顺序

1. 准备 overlay 模型目录，固定 `_class_name = "WanVideoEditPipeline"`。
2. 新增 `WanVideoEditSamplingParams`，避免 `video_path`/`mask_path` 命名冲突。
3. 新增 `WanVideoEditPipelineConfig`，基于 `WanT2V480PConfig` 打开 VAE encoder。
4. 实现 `VideoEditFlowMatchScheduler` adapter，并先完成 scheduler 对齐测试。
5. 落地 `runtime/videoedit/contracts.py`、`preprocess.py`、`postprocess.py`，把纯函数从原 repo 解耦出来。
6. 实现 `VideoEditConditionStage`，完成 `batch.image_latent` 和 `video_latents` 生产。
7. 实现 `VideoEditLatentInitStage`，先在关闭 dynamic CFG 条件下跑通单窗口。
8. 实现 `VideoEditDenoisingStage`，补齐 dynamic CFG 和 negative pass 控制。
9. 注册 pipeline/config/sampling params。
10. 增加 side-by-side 对齐测试和回归脚本。
11. 最后再补长视频滑窗 helper 和 paste-back。

## 25. 最终结论

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
