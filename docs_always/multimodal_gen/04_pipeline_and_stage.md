# Pipeline 与 Stage 机制

## 1. `ComposedPipelineBase` 的定位

位置：`runtime/pipelines_core/composed_pipeline_base.py`

它是整个框架最重要的抽象之一。一个具体 Pipeline 主要做三件事：

1. 声明需要哪些模块：`_required_config_modules`
2. 在 `initialize_pipeline()` 中做少量初始化/替换
3. 在 `create_pipeline_stages()` 中组织 stage

换句话说，Pipeline 本身更像“编排器”，不直接承担所有前向细节。

## 2. Pipeline 初始化过程

构造 `pipeline_cls(model_path, server_args)` 时，`ComposedPipelineBase.__init__()` 会依次完成：

1. 记录 `server_args`、`model_path`
2. 创建 executor
3. 根据 disagg role 过滤需要加载的模块
4. `load_modules()`
5. `__post_init__()`
6. `initialize_pipeline()`
7. `create_pipeline_stages()`

因此，Pipeline 在构造完成时就已经是“可执行状态”。

## 3. `load_modules()` 做了什么

`load_modules()` 的职责远不只是“读权重”：

- 下载并验证 `model_index.json`
- 处理 `boundary_ratio` 等模型级特殊字段
- 在 disagg 模式下为未加载组件补 config
- 根据 `required_config_modules` 过滤模块
- 结合 `_extra_config_module_map` 做模块名映射
- 调 `PipelineComponentLoader.load_component(...)`
- 记录各组件显存占用到 `memory_usages`

这一步把“diffusers 风格模型目录”变成“运行时可用的组件字典”。

## 4. 为什么 `_required_config_modules` 很重要

以 `FluxPipeline` 为例：

```python
_required_config_modules = [
    "text_encoder",
    "text_encoder_2",
    "tokenizer",
    "tokenizer_2",
    "vae",
    "transformer",
    "scheduler",
]
```

这相当于声明：

- 我需要哪些 diffusers/transformers 组件；
- `load_modules()` 应该去哪些子目录加载；
- 后续 stage 能依赖哪些模块。

这让不同模型只需要声明“组件图”，而不用自己重复写很多加载样板代码。

## 5. Stage 的统一接口

位置：`runtime/pipelines_core/stages/base.py`

每个 stage 都是 `PipelineStage` 子类，统一约定：

- `forward(batch, server_args) -> Req`
- `verify_input()`
- `verify_output()`
- `role_affinity`
- `parallelism_type`

`PipelineStage.__call__()` 会统一包裹：

- 输入校验
- 阶段级 profiling
- 输出校验

这保证了所有 stage 的运行语义一致。

## 6. `Req` 作为 stage 间状态载体

典型的 stage 写法就是：

- 从 `batch` 取输入；
- 计算；
- 把结果写回 `batch`；
- 返回同一个 `batch`。

例如：

- `TextEncodingStage` 写入 `prompt_embeds`
- `LatentPreparationStage` 写入 `latents`
- `TimestepPreparationStage` 写入 `timesteps`
- `DenoisingStage` 更新 `latents`
- `DecodingStage` 产出 `OutputBatch`

所以 stage 不是函数式不可变对象，而是围绕一个共享 `Req` 做增量更新。

## 7. 通用 stage 链

最典型的 T2I/T2V 链路是：

1. `InputValidationStage`
2. `TextEncodingStage`
3. `LatentPreparationStage`
4. `TimestepPreparationStage`
5. `DenoisingStage`
6. `DecodingStage`

`ComposedPipelineBase` 提供了大量 helper：

- `add_standard_text_encoding_stage()`
- `add_standard_timestep_preparation_stage()`
- `add_standard_latent_preparation_stage()`
- `add_standard_denoising_stage()`
- `add_standard_decoding_stage()`
- `add_standard_t2i_stages()`
- `add_standard_ti2i_stages()`
- `add_standard_ti2v_stages()`

这使得很多模型 Pipeline 只需几行代码就能完成定义。

## 8. 核心 stage 的职责

## 8.1 `InputValidationStage`

位置：`stages/input_validation.py`

职责：

- 生成 seed 和 `torch.Generator`
- 处理输入图像/视频
- 调整输出尺寸
- 针对不同模型做图像预处理

这一步承担了大量“模型相关输入整理”工作，例如：

- TI2I 条件图处理
- TI2V 图像 resize/crop
- Wan I2V 专用尺寸策略
- MOVA 专用预处理

它本质上是“把用户输入整理成模型可消费形式”。

## 8.2 `TextEncodingStage`

位置：`stages/text_encoding.py`

职责：

- 调 tokenizer
- 调一个或多个 text encoder
- 生成正/负 prompt embeddings
- 生成 pooled embeds 与 attention mask

它支持多编码器组合，这对 FLUX、SD3 等多 encoder 模型很重要。

## 8.3 `ImageEncodingStage`

位置：`stages/image_encoding.py`

职责：

- 把条件图编码成 image embeddings
- 或在 Qwen-Image-Edit 等场景中把图像走进带视觉输入的 text encoder

这说明框架并不假设“图像输入一定走 image encoder”，而是允许模型自定义图像编码路径。

## 8.4 `LatentPreparationStage`

位置：`stages/latent_preparation.py`

职责：

- 根据 `PipelineConfig` 计算 latent shape
- 生成初始噪声
- 调整视频 latent 帧数
- 准备 latent ids / packed latents

这一层主要服务“扩散起点”的准备。

## 8.5 `TimestepPreparationStage`

位置：`stages/timestep_preparation.py`

职责：

- 处理 `num_inference_steps`
- 处理自定义 `timesteps` / `sigmas`
- 调 scheduler 的 `set_timesteps`
- 支持额外 kwargs，比如 FLUX/Qwen 的 `mu`

它把模型族特有的 scheduler 入参差异抽象进了可注入的 `prepare_extra_set_timesteps_kwargs`。

## 8.6 `DenoisingStage`

位置：`stages/denoising.py`

这是最复杂、最重的 stage。

职责包括：

- 去噪主循环
- CFG / true-CFG
- 多 transformer 支持
- cache-dit 集成
- attention backend 选择
- torch.compile
- rollout/post-training 支持
- sequence parallel / cfg parallel 相关逻辑
- 部分模型特化逻辑（如 Wan TI2V）

它可以看作“真正的推理热路径”。

## 8.7 `DecodingStage`

位置：`stages/decoding.py`

职责：

- 把 latent 反解码为图像/视频
- 处理 decode scale/shift
- 执行 VAE tiling/slicing
- 支持 trajectory latents 批量解码
- 按需做 VAE offload

这里输出的是 `OutputBatch`，因此它通常是原生 Pipeline 的最后一站。

## 9. Stage 并行语义

stage 可以声明自己的 `parallelism_type`：

- `REPLICATED`
- `MAIN_RANK_ONLY`
- `CFG_PARALLEL`

这让 executor 可以按 stage 粒度决定：

- 全 rank 执行
- 仅主 rank 执行
- 在 CFG 分组内广播后执行

也就是说，分布式并行不是整条 pipeline 一刀切，而是按阶段细粒度控制。

## 10. 典型 Pipeline 对比

## 10.1 `FluxPipeline`

位置：`runtime/pipelines/flux.py`

特点：

- 双 text encoder
- 标准 T2I stages
- 通过 `prepare_mu()` 向 scheduler 注入 FLUX 特有的 `mu`

## 10.2 `WanPipeline`

位置：`runtime/pipelines/wan_pipeline.py`

特点：

- 用官方 `FlowUniPCMultistepScheduler` 替换通用 scheduler
- 其它部分基本复用标准 stage 链

这说明 Pipeline 很适合做“局部替换”，不需要完全重写。

## 10.3 `QwenImageEditPipeline`

位置：`runtime/pipelines/qwen_image.py`

特点：

- 不是纯 text encoding，而是 `prompt_encoding="image_encoding"`
- 结合 `processor + text_encoder + vae`
- 走 `add_standard_ti2i_stages(...)`

这体现了通用 helper 的可组合性。

## 10.4 `QwenImageLayeredPipeline`

特点：

- 在标准链路前插入 `QwenImageLayeredBeforeDenoisingStage`
- 然后再走 timestep / denoising / decoding

这展示了框架留给模型专用逻辑的注入点。

## 10.5 `DiffusersPipeline`

位置：`runtime/pipelines/diffusers_pipeline.py`

它是兜底 Pipeline。其核心是用一个 `DiffusersExecutionStage` 包住原生 diffusers pipeline 调用，再把输出规范化成框架内部张量格式。

这让未原生适配的模型仍然能接入统一的 CLI/API/服务体系。

## 11. LoRA 为什么是 mixin

`LoRAPipeline` 继承自 `ComposedPipelineBase`，但实际使用时通常是：

- `class FluxPipeline(LoRAPipeline, ComposedPipelineBase)`
- `class WanPipeline(LoRAPipeline, ComposedPipelineBase)`

它负责：

- 把线性层替换成支持 LoRA 的层；
- 加载 adapter；
- 支持多 LoRA / 多 target；
- 支持 merge / unmerge；
- 处理 layerwise offload 与 LoRA 的交互。

把 LoRA 做成 mixin 的好处是：所有 Pipeline 共享一套 LoRA 能力，而不污染主干 Pipeline 逻辑。

## 12. 总结：Pipeline 机制的本质

这套设计的本质是：

- 组件加载归 `ComposedPipelineBase`
- 执行步骤归 `PipelineStage`
- 并行方式归 `Executor`
- 模型差异归具体 Pipeline 和 model-specific stages

因此它具备较强的可扩展性。新增模型通常只需要在以下几处动手：

1. 注册模型配置；
2. 声明一个 Pipeline；
3. 必要时新增少量专用 stage。

