# STAR 在 SGLang 中的集成与推理流程分析

## 1. 文档目的

这份文档用于系统说明：

1. `STAR` 是怎么被接入到 `sglang` 的
2. `sglang` 是怎么把底层加速能力接到 `STAR` 上的
3. 从一次请求进入，到最终视频写盘，完整的输入输出和 `forward` 串联过程是什么

本文只分析当前仓库中的 `STAR CogVideoX-SR` 路线，不讨论量化分支。

---

## 2. 总览

当前 `STAR` 在 `sglang` 中不是以“直接调用原版 `sample_sr.py`”的方式运行，而是被重构为一条 `sglang` 的 **modular composed pipeline**。

最关键的几个入口是：

1. 用户调用生成入口：
   [diffusion_generator.py](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/entrypoints/diffusion_generator.py:1)
2. `GPUWorker` 构建 pipeline 并执行：
   [gpu_worker.py](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/managers/gpu_worker.py:1)
3. STAR 专用 pipeline：
   [star_cogvideox_sr_pipeline.py](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/pipelines/star_cogvideox_sr_pipeline.py:1)
4. 最终 pipeline 组成的 stage 链：
   `InputValidation -> ConditionVideoLoading -> TextEncoding -> ConditionVideoVAEEncoding -> STARLatentPreparation -> TimestepPreparation -> Denoising -> STARDecoding`

一句话概括：

1. `sglang` 负责把请求包装成 `Req`
2. `Scheduler/GPUWorker` 负责生命周期、分发和执行
3. STAR 的语义通过专用 `pipeline config`、专用 `stage`、专用 `transformer / scheduler / VAE` 落地
4. 底层加速通过 `torch.compile`、FlashAttention、本地 fused 路径和自定义 loader 接入

---

## 3. 模型目录与注册关系

### 3.1 `model_index.json` 决定 pipeline 组件

当前转换后的本地 STAR 模型目录在：

- [model_index.json](/sgl-workspace/sglang/model_artifacts/sglang_star_cogvideox_sr/model_index.json:1)

这里定义了这个模型在 `sglang/diffusers` 视角下的组件关系：

1. pipeline class: `StarCogVideoXSRPipeline`
2. scheduler: `StarVPSDEDPMPP2MScheduler`
3. text encoder: `T5EncoderModel`
4. tokenizer: `T5Tokenizer`
5. transformer: `StarCogVideoXSRTransformer3DModel`
6. vae: `StarCogVideoXSRVAE`

这意味着 `build_pipeline()` 会按这个清单把组件加载出来。

### 3.2 `star_integration_config.json` 注入 STAR 特有语义

模型目录里还有一份：

- [star_integration_config.json](/sgl-workspace/sglang/model_artifacts/sglang_star_cogvideox_sr/star_integration_config.json:1)

它用于把原版 STAR 的一些关键参数注入到 `sglang` pipeline config 中，例如：

1. `latent_scale_factor = 0.7`
2. `default_sampling_num_frames = 7`
3. `dynamic_cfg_exp = 5`
4. `latent_channels = 16`
5. transformer 的 `latent_height / latent_width / patch_size / time_compressed_rate`

这个注入发生在：

- [star_cogvideox_sr_pipeline.py](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/pipelines/star_cogvideox_sr_pipeline.py:47)
- [star_cogvideox_sr.py apply_integration_config](/sgl-workspace/sglang/python/sglang/multimodal_gen/configs/pipeline_configs/star_cogvideox_sr.py:103)

---

## 4. 请求从哪里进入

### 4.1 用户入口：`DiffGenerator.from_pretrained()`

用户或手工脚本通常通过：

- [DiffGenerator.from_pretrained](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/entrypoints/diffusion_generator.py:77)

创建生成器。

这个过程会：

1. 把用户传入的参数组装成 `ServerArgs`
2. 在 `local_mode=True` 时启动本地 scheduler / worker
3. 生成时把 prompt、video path、seed、steps 等包装成 `SamplingParams`

### 4.2 请求被包装成 `Req`

`generate()` 会做三件事：

1. 解析 prompt / prompt_path
2. 构造 `SamplingParams`
3. 用 `prepare_request()` 包装成 `Req`

对应代码：

- [generate](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/entrypoints/diffusion_generator.py:164)
- [prepare_request](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/entrypoints/utils.py:242)
- [Req 定义](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/schedule_batch.py:35)

`Req` 是整个 pipeline 的共享状态对象，后续所有 stage 都围绕它读写字段，例如：

1. `prompt`
2. `negative_prompt`
3. `condition_video_path`
4. `prompt_embeds`
5. `image_latent`
6. `latents`
7. `timesteps`
8. `metrics`
9. `output`

---

## 5. Scheduler 与 Worker 怎么接上 STAR

### 5.1 `Scheduler` 收到 `Req` 后交给 `GPUWorker`

当前本地生成模式下：

1. `DiffGenerator` 把请求送给 scheduler client
2. rank0 `Scheduler` 收到 `Req`
3. `Scheduler._handle_generation()` 调用 `GPUWorker.execute_forward()`

对应代码：

- [Scheduler._handle_generation](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/managers/scheduler.py:188)
- [GPUWorker.execute_forward](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/managers/gpu_worker.py:205)

### 5.2 `GPUWorker` 启动时就会构建 STAR pipeline

`GPUWorker.__init__()` 会调用：

- [init_device_and_model](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/managers/gpu_worker.py:87)

这里的关键步骤是：

1. 初始化分布式环境
2. 调用 `build_pipeline(server_args)`
3. 拿到 `StarCogVideoXSRPipeline`

对应：

- [build_pipeline](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/__init__.py:26)

---

## 6. STAR Pipeline 是怎么组装出来的

### 6.1 `build_pipeline()` 通过 registry 选 STAR pipeline

如果 `server_args.pipeline_class_name = StarCogVideoXSRPipeline`，则直接从 registry 取这个类：

- [build_pipeline](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/__init__.py:37)

然后实例化：

- [pipeline_cls(model_path, server_args)](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/__init__.py:74)

### 6.2 `ComposedPipelineBase` 负责两件事

`StarCogVideoXSRPipeline` 继承：

1. `LoRAPipeline`
2. `ComposedPipelineBase`

其中 `ComposedPipelineBase.__init__()` 负责：

1. `load_modules()`
2. `initialize_pipeline()`
3. `create_pipeline_stages()`

对应代码：

- [ComposedPipelineBase.__init__](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/composed_pipeline_base.py:69)
- [load_modules](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/composed_pipeline_base.py:260)
- [__post_init__](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/composed_pipeline_base.py:126)

### 6.3 STAR pipeline 定义了具体 stage 链

STAR 的 stage 顺序定义在：

- [create_pipeline_stages](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/pipelines/star_cogvideox_sr_pipeline.py:72)

顺序是：

1. `InputValidationStage`
2. `STARConditionVideoLoadingStage`
3. `TextEncodingStage`
4. `STARConditionVideoVAEEncodingStage`
5. `STARLatentPreparationStage`
6. `TimestepPreparationStage`
7. `DenoisingStage`
8. `STARCogVideoXSRDecodingStage`

这就是后面整个 `forward` 串联的骨架。

---

## 7. 组件是怎么被加载成 STAR 专用实现的

### 7.1 通用 loader 分发

`ComposedPipelineBase.load_modules()` 会读取 `model_index.json`，然后对每个组件调用：

- [PipelineComponentLoader.load_component](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/loader/component_loaders/component_loader.py:64)

它的策略是：

1. 先尝试加载 `sglang` 自己的 customized 版本
2. 如果失败，再 fallback 到 native diffusers / transformers 版本

### 7.2 Transformer loader

STAR 的 transformer 由：

- [TransformerLoader](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/loader/component_loaders/transformer_loader.py:49)

负责加载。

关键过程：

1. 读取 diffusers config
2. 更新 `server_args.pipeline_config.dit_config`
3. 通过 `ModelRegistry` 找到 `StarCogVideoXSRTransformer3DModel`
4. 解析 quant / safetensors / fsdp 规格
5. 用 `maybe_load_fsdp_model()` 真正装配模型

这就是 STAR 的 DiT 被接到 `sglang` runtime 的位置。

### 7.3 VAE loader

STAR 的 VAE 由：

- [VAELoader](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/loader/component_loaders/vae_loader.py:58)

负责加载。

关键过程：

1. 解析 `_class_name = StarCogVideoXSRVAE`
2. 更新 `vae_config`
3. 通过 `ModelRegistry` 找到 `StarCogVideoXSRVAE`
4. 读取 safetensors 权重
5. 调用 `current_platform.optimize_vae()`

STAR 的 VAE 运行时实现本体在：

- [star_cogvideox_vae.py](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/models/vaes/star_cogvideox_vae.py:1)

它内部 vendor 了原版 STAR SAT 所需 VAE 代码，而不是再动态依赖 `STAR_mg` 仓库路径。

### 7.4 Scheduler loader

STAR 的 scheduler 由：

- [SchedulerLoader](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/loader/component_loaders/scheduler_loader.py:12)

负责加载。

它会：

1. 从组件目录读 scheduler config
2. 通过 `ModelRegistry` 找到 `StarVPSDEDPMPP2MScheduler`
3. 用 config 初始化 scheduler

实现本体在：

- [star_vpsde_dpmpp2m.py](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/models/schedulers/star_vpsde_dpmpp2m.py:1)

这个 scheduler 是 STAR 原始 `VPSDE DPM++ 2M` 采样语义在 `sglang` 中的适配版本。

---

## 8. 端到端 stage / forward 串联

下面按执行顺序说明每个 `forward`。

### 8.1 `InputValidationStage.forward`

代码：

- [InputValidationStage.forward](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/input_validation.py:292)

STAR 路径里最重要的事情有：

1. 规范输入参数
2. 生成 `batch.seeds`
3. 创建 `batch.generator`
4. 额外创建 `batch.extra["star_initial_noise_generator"]`

其中 STAR 特有逻辑在：

- [_generate_seeds](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/input_validation.py:69)

这里有个很关键的 STAR 定制：

1. 初始 latent 噪声使用单独的 CPU generator
2. VAE / scheduler 相关随机过程继续使用默认 generator 设备

这是为了贴近原版 STAR 的噪声路径。

### 8.2 `STARConditionVideoLoadingStage.forward`

代码：

- [STARConditionVideoLoadingStage.forward](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/video_condition_loading.py:247)

它做的事情是：

1. 从 `condition_video_path` 读取整段低清视频
2. 根据 `condition_video_num_frames` 选帧
3. 做 paired-dataset 风格的 resize/crop
4. 把像素归一化到 `[-1, 1]`
5. 输出 `batch.condition_video`

这一步对应原版：

- [PairedCaptionDataset.__getitem__](/sgl-workspace/STAR_mg/cogvideox-based/sat/data_video.py:431)

也就是说，`sglang` 这里是在复刻原版 STAR 的 `lq` 读取与预处理语义。

### 8.3 `TextEncodingStage.forward`

代码：

- [TextEncodingStage.forward](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/text_encoding.py:69)

它做的事情是：

1. 把 text encoder 迁到正确设备
2. 编码正向 prompt
3. 如果开启 CFG，编码 negative prompt
4. 如果 pipeline config 要求 zero unconditional text embedding，则把负向 embedding 置零

STAR 的 zero unconditional 语义来自：

- [should_force_zero_unconditional_text_embeddings](/sgl-workspace/sglang/python/sglang/multimodal_gen/configs/pipeline_configs/star_cogvideox_sr.py:297)

### 8.4 `STARConditionVideoVAEEncodingStage.forward`

代码：

- [STARConditionVideoVAEEncodingStage.forward](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/video_condition_vae_encoding.py:203)

它做的事情是：

1. 必要时先释放 text encoder / transformer 显存
2. 把 `condition_video` 从 `[B, T, C, H, W]` 转为 `[B, C, T, H, W]`
3. 调用 `vae.encode()`
4. 从 posterior 中取 latent
5. 乘上 STAR 的 `scale_factor = 0.7`
6. 写入 `batch.image_latent`

STAR 特有的关键点：

1. 取样模式来自：
   [encode_sample_mode](/sgl-workspace/sglang/python/sglang/multimodal_gen/configs/models/vaes/star_cogvideox_vae.py:33)
2. 反映原版 `encode_first_stage()` 语义：
   [diffusion_video.py](/sgl-workspace/STAR_mg/cogvideox-based/sat/diffusion_video.py:188)

这一步是把 `lq` 变成 STAR SR 模型的条件 latent。

### 8.5 `STARLatentPreparationStage.forward`

代码：

- [STARLatentPreparationStage.forward](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/star_latent_preparation.py:36)

它做的事情是：

1. 按 STAR 的 latent timeline 直接准备采样 latent
2. 如果 `latents` 为空，则采样初始噪声
3. 初始噪声先以 `[B, T, C, H, W]` 形状采样，再转成内部 `[B, C, T, H, W]`
4. 乘上 scheduler 的 `init_noise_sigma`

关键点：

1. 这里会优先使用 `batch.extra["star_initial_noise_generator"]`
2. 这是在贴近原版 `sample_sr()` 的初始噪声采样方式

对应原版：

- [sample_sr randn](/sgl-workspace/STAR_mg/cogvideox-based/sat/diffusion_video.py:245)

### 8.6 `TimestepPreparationStage.forward`

代码：

- [TimestepPreparationStage.forward](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/timestep_preparation.py:48)

它做的事情是：

1. 根据 `num_inference_steps`
2. 调用 scheduler 的 `set_timesteps()`
3. 把结果写回 `batch.timesteps`

STAR 的 scheduler 会在这里生成：

1. `timesteps`
2. `alphas_cumprod_sqrt`

### 8.7 `DenoisingStage.forward`

代码入口：

- [DenoisingStage.forward](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py:1088)

它是整条链最核心的计算阶段。

它会先准备一个 `DenoisingContext`，关键字段包括：

1. `latents`
2. `timesteps`
3. `image_kwargs`
4. `pos_cond_kwargs`
5. `neg_cond_kwargs`
6. `guidance`
7. `extra_step_kwargs`

其中 `extra_step_kwargs` 会把 `generator=batch.generator` 传给 scheduler：

- [prepare_extra_func_kwargs 调用点](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py:535)

然后进入逐步 denoise 循环：

- [for step_index, t_host in enumerate(...)](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py:1138)

每一步 `_run_denoising_step()` 内部顺序是：

1. `latent_model_input = ctx.latents`
2. 如果有 `batch.image_latent`，则和 denoise latents 在 channel 维拼接
3. `scheduler.scale_model_input(...)`
4. `_predict_noise_with_cfg(...)`
5. `scheduler.step(...)`
6. 更新 `ctx.latents`

关键代码：

- [_run_denoising_step](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py:760)

#### `_predict_noise_with_cfg()` 怎么串

代码：

- [_predict_noise_with_cfg](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py:1784)

逻辑是：

1. 先准备正向和负向条件
2. 如果允许 batched CFG，则优先尝试一次 forward 同时跑 cond/uncond
3. 否则分别跑正向和负向
4. 调用 `pipeline_config.get_classifier_free_guidance_scale_for_step()`
5. 组合得到 `noise_pred`

STAR 的动态 CFG 语义来自：

- [get_classifier_free_guidance_scale_for_step](/sgl-workspace/sglang/python/sglang/multimodal_gen/configs/pipeline_configs/star_cogvideox_sr.py:300)

#### 真正的 Transformer `forward` 怎么接上

`_predict_noise()` 最后会调用 STAR 的 transformer：

- [StarCogVideoXSRTransformer3DModel.forward](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/models/dits/star_cogvideox_sr.py:1308)

它的输入是：

1. `hidden_states`，已经是 `denoise_latent || condition_latent` 拼好的 `[B, C, T, H, W]`
2. `encoder_hidden_states`，来自 text embeddings
3. `timestep`

然后内部会做：

1. timestep embedding
2. patch embedding
3. rotary position embedding cache
4. 42 层 transformer block
5. 最终输出噪声预测

### 8.8 `STARCogVideoXSRDecodingStage.forward`

代码：

- [STARCogVideoXSRDecodingStage.forward](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/star_cogvideox_sr_decoding.py:166)

它做的事情是：

1. 取最终 `batch.latents`
2. 做 STAR 特有的 decode 前 scale / shift 处理
3. 按 STAR 的时间窗口 decode
4. 拼接各窗口输出
5. 转成 `[0, 1]` 区间
6. 返回 `OutputBatch.output`

STAR 特有时间窗口逻辑在：

- [build_decode_windows](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/star_cogvideox_sr_decoding.py:34)

对应原版：

- [sample_sr decode loop](/sgl-workspace/STAR_mg/cogvideox-based/sat/sample_sr.py:170)

---

## 9. 每个 `forward` 是怎么串起来的

把整个推理过程压成一条链，就是：

1. `DiffGenerator.generate()`
2. `prepare_request() -> Req`
3. `Scheduler._handle_generation()`
4. `GPUWorker.execute_forward()`
5. `pipeline.forward(req, server_args)`
6. `InputValidationStage.forward()`
7. `STARConditionVideoLoadingStage.forward()`
8. `TextEncodingStage.forward()`
9. `STARConditionVideoVAEEncodingStage.forward()`
10. `STARLatentPreparationStage.forward()`
11. `TimestepPreparationStage.forward()`
12. `DenoisingStage.forward()`
13. `STARCogVideoXSRDecodingStage.forward()`
14. `OutputBatch`
15. `save_outputs()`
16. `candidate.mp4`

更具体地说，数据在各阶段中的主要形态变化是：

1. 输入：`prompt` + `condition_video_path`
2. `condition_video_path -> batch.condition_video`：像素视频，`[B, T, C, H, W]`
3. `prompt -> prompt_embeds`
4. `condition_video -> image_latent`：条件 latent
5. `seed -> latents`：初始噪声 latent
6. `timesteps`
7. `latents + image_latent + prompt_embeds -> noise_pred`
8. 50 步迭代后得到最终 `latents`
9. `latents -> frames`
10. `frames -> mp4`

---

## 10. STAR 是怎么接上底层加速的

这一部分是“为什么 STAR 能跑在 `sglang` 的加速后端上”。

### 10.1 `torch.compile`

`DenoisingStage.__init__()` 会对 transformer 调用：

- [_maybe_enable_torch_compile](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py:174)

只要 `server_args.enable_torch_compile = true`，STAR 的 transformer 就会在 worker 初始化阶段被 compile。

这也是当前单卡非量化最快主线 `single_fa_compile_fusedln_v2` 的核心前提之一。

### 10.2 FlashAttention / 本地 attention backend

STAR transformer 内部使用的是 `sglang` runtime 的 attention 抽象，而不是直接沿用原版 SAT attention：

- [LocalAttention / USPAttention 引入](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/models/dits/star_cogvideox_sr.py:23)

实际 backend 的选择由：

- [get_attn_backend](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py:28)
- `server_args.attention_backend`

决定。

当前 STAR 单卡主线实际用的是：

1. `attention_backend = fa`
2. `sglang` 自己的 FlashAttention 路径

### 10.3 fused layernorm / modulation 热路径

STAR transformer 里已经把部分原版 norm/modulation 路径换成了 `sglang` 兼容的 fused 实现，例如：

- [_StarLayerNorm.forward_scale_shift](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/models/dits/star_cogvideox_sr.py:182)
- [_StarLayerNorm.forward_residual_scale_shift](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/models/dits/star_cogvideox_sr.py:210)

这些不会改变 pipeline 结构，但会把底层算子执行切到 `sglang` 的加速 kernel。

### 10.4 customized VAE runtime

STAR 的 VAE 没有再依赖运行时去 import 外部 `STAR_mg` 仓库，而是：

1. 在 `sglang` 仓库内 vendor 了 SAT 所需子树
2. 通过 `StarCogVideoXSRVAE` 封装成 `sglang` 可加载组件

代码：

- [star_cogvideox_vae.py](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/models/vaes/star_cogvideox_vae.py:1)

这是 STAR 脱离原仓库路径耦合的关键点之一。

### 10.5 customized STAR scheduler

STAR 的采样器不是直接跑原版 Python sampler，而是通过：

- [StarVPSDEDPMPP2MScheduler](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/models/schedulers/star_vpsde_dpmpp2m.py:1)

把原版 `VPSDE DPM++ 2M` 语义适配进 `sglang` 的 scheduler 接口。

这样 `DenoisingStage` 仍然用统一的 scheduler.step 模式推进，但语义保持 STAR 采样逻辑。

### 10.6 FSDP / offload / resident 策略

`sglang` 还把一些原版没有统一抽象的运行时策略纳入了标准入口，例如：

1. `dit_cpu_offload`
2. `text_encoder_cpu_offload`
3. `vae_cpu_offload`
4. `condition_video_vae_peak_memory_mode`

这些主要体现在：

- [GPUWorker](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/managers/gpu_worker.py:87)
- [STARConditionVideoVAEEncodingStage](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/video_condition_vae_encoding.py:27)

它们的作用是让 STAR 在 `sglang` 里可控地利用显存和执行路径，而不是改变 STAR 的核心采样任务定义。

---

## 11. 最终输出是怎么写成视频的

`GPUWorker.execute_forward()` 得到 `OutputBatch` 后，`DiffGenerator.generate()` 会调用：

- [save_outputs](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/entrypoints/utils.py:322)

而视频写盘真正发生在：

- [imageio.mimsave(... codec=\"libx264\", quality=...)](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/entrypoints/utils.py:493)

这里也解释了为什么：

1. `raw frame` 和 `mp4` 要区分
2. 因为最终 `mp4` 会混入编码器质量参数

最后 `DiffGenerator.generate()` 会把：

1. `frames`
2. `output_file_path`
3. `metrics`
4. `trajectory_latents`

组装成：

- [GenerationResult](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/entrypoints/utils.py:94)

返回给调用方。

---

## 12. 一句话总结

当前 STAR 在 `sglang` 中的集成方式，不是“套壳调用原版”，而是：

1. 把原版 STAR 的 **请求语义、条件视频语义、scheduler 语义、decode 语义**
2. 映射到 `sglang` 的 **composed pipeline + runtime loader + scheduler + attention + compile** 体系里
3. 从而在保持 STAR 采样流程语义的同时，接入 `sglang` 的底层加速能力

所以从运行时角度看，STAR 现在已经是 `sglang` 的一个原生视频模型 pipeline，而不是外部脚本的旁路调用。
