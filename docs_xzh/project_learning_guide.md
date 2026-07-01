# Vivid-VR 接入 SGLang 加速与服务学习指南

本文只讨论工程接入，不讨论算法原理。目标是回答这几个问题：

1. Vivid-VR 在 `sglang.multimodal_gen` 里是怎样成为一个 native pipeline 的。
2. Flash Attention / SDPA、算子融合、`torch.compile`、SP / Ulysses 这些能力分别在哪里接入。
3. 服务层是怎样把外部请求送进 Vivid-VR pipeline，并让它复用同一套 runtime 加速能力的。

全文中的代码引用统一精确到 `path:line`。如果你要现场对照代码，建议一边看本文，一边直接打开这些行号附近的实现。

## 1. 项目整体做什么

从工程视角看，这个项目做的不是“在 `sglang` 外面包一层 Vivid-VR 脚本”，而是把 Vivid-VR 变成 `sglang.multimodal_gen` 体系里的正式 pipeline，再让它直接复用 SGLang 已有的 runtime、加速和服务设施。

最关键的判断标准是：请求最终不是落回原版 `/home/zhiheng/Vivid-VR` 的推理运行时代码，而是落进 `sglang` 自己的注册表、pipeline config、sampling params、runtime pipeline 和 HTTP 服务链路。

这一点可以从四组锚点直接看出来：

- `python/sglang/multimodal_gen/registry.py:731-732`
  这里把 `VividVRSamplingParams` 和 `VividVRPipelineConfig` 接进统一注册表。它说明 Vivid-VR 在 `multimodal_gen` 里是“被系统认识的正式模型管线”，不是旁路逻辑。
- `python/sglang/multimodal_gen/configs/pipeline_configs/vividvr.py:41`
  这里定义 `class VividVRPipelineConfig`。这说明 Vivid-VR 有自己的 pipeline 级默认语义。
- `python/sglang/multimodal_gen/configs/sample/vividvr.py:14`
  这里定义 `class VividVRSamplingParams`。这说明服务请求最终会被规整成 Vivid-VR 自己的采样参数对象。
- `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py:806`
  这里定义 `initialize_pipeline()`。这说明真正的模型装配、backend 应用、融合开关和 compile 都是在 `sglang` 运行时里完成的。

如果只记一句话，可以记成：

`Vivid-VR = 注册进 multimodal_gen 的 native pipeline + 复用 SGLang runtime 能力的运行时装配 + 复用 SGLang HTTP 服务的对外接口。`

## 2. 核心模块有哪些

如果你只关心“Vivid-VR 怎样吃到 SGLang 的加速和服务能力”，整套实现可以按责任分成五层。

### 2.1 注册与配置层

这一层回答“Vivid-VR 是怎样进入 `multimodal_gen` 的统一模型体系”的问题。

- `python/sglang/multimodal_gen/configs/pipeline_configs/vividvr.py:41`
  `VividVRPipelineConfig` 定义 Vivid-VR 的 pipeline 默认配置。
- `python/sglang/multimodal_gen/configs/pipeline_configs/vividvr.py:56`
  `vae_tiling: bool = True` 这类字段说明 Vivid-VR 的稳定默认运行口径被固化在 config 里，而不是散落在脚本参数里。
- `python/sglang/multimodal_gen/configs/pipeline_configs/vividvr.py:68-79`
  `caption_source`、`default_prompt_file_path`、`reference_video_path`、`allow_live_cogvlm2_caption` 这些字段定义了和项目稳定基线直接相关的默认语义。
- `python/sglang/multimodal_gen/configs/sample/vividvr.py:14`
  `VividVRSamplingParams` 定义运行时最终消费的请求对象。
- `python/sglang/multimodal_gen/configs/sample/vividvr.py:206`
  `from_user_kwargs()` 是把 HTTP 请求 / 脚本参数收束成最终执行参数的关键入口。
- `python/sglang/multimodal_gen/registry.py:731-732`
  这两行把上面的 config 和 sampling params 注册起来。

这一层的重要性在于：只有先被注册成正式 pipeline，后面的 runtime 加速逻辑才有明确挂点。

### 2.2 参数入口层

这一层回答“加速选项和服务选项从哪里进入系统”的问题。

- `python/sglang/multimodal_gen/runtime/entrypoints/cli/serve.py:28`
  `ServerArgs.add_cli_args` 把命令行参数注册到 `serve`。
- `python/sglang/multimodal_gen/runtime/entrypoints/cli/serve.py:33`
  `ServerArgs.from_cli_args` 把 CLI 值装配成 `ServerArgs`。
- `python/sglang/multimodal_gen/runtime/server_args.py:97`
  `class ServerArgs` 是所有 runtime 能力的总入口。
- `python/sglang/multimodal_gen/runtime/server_args.py:108`
  `attention_backend` 的定义位置。
- `python/sglang/multimodal_gen/runtime/server_args.py:124-126`
  `sp_degree`、`ulysses_degree` 的定义位置。
- `python/sglang/multimodal_gen/runtime/server_args.py:176-183`
  `enable_torch_compile` 和各类 `enable_cogvideox_*_fusion` 的定义位置。

### 2.3 Pipeline 装配层

这一层回答“这些参数最后怎样真正作用到模型组件”的问题。

- `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py:259`
  `_maybe_initialize_model_parallel_runtime()` 初始化并行运行时。
- `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py:527`
  `_apply_attention_backend()` 把 backend 真正应用到模型。
- `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py:600`
  `_apply_qkv_fusion()`
- `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py:650`
  `_apply_qk_norm_fusion()`
- `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py:699`
  `_apply_qk_norm_rope_fusion()`
- `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py:748`
  `_apply_modulation_fusion()`
- `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py:787`
  `_apply_torch_compile()`
- `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py:806`
  `initialize_pipeline()` 是所有装配动作的主入口。

### 2.4 模型加速适配层

这一层回答“backend、融合和 SP 语义在底层模型实现里怎么落地”的问题。

- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_attention_backend.py:45`
  `normalize_cogvideox_attention_backend()`
- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_attention_backend.py:76`
  `resolve_cogvideox_attention_runtime_choice()`
- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_attention_backend.py:658`
  `set_cogvideox_attention_backend()`
- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_attention_backend.py:848`
  `inspect_cogvideox_attention_backend()`
- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_operator_fusion.py:13`
  `_COGVIDEOX_MODULATION_FUSION_IMPL = "sglang_modulation_fused_ops"`
- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr_common.py:25-28`
  connector SP 上下文模式相关常量和环境变量。
- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr_controlnet.py:164`
  controlnet 侧的 `set_attention_backend()`。

### 2.5 服务入口层

这一层回答“HTTP 请求怎样进入 Vivid-VR pipeline”的问题。

- `python/sglang/multimodal_gen/runtime/launch_server.py:61`
  `launch_server()`
- `python/sglang/multimodal_gen/runtime/launch_server.py:188-191`
  `launch_http_server_only()` 和 `create_app(server_args)` 的衔接位置。
- `python/sglang/multimodal_gen/runtime/entrypoints/http_server.py:288`
  `create_app()`
- `python/sglang/multimodal_gen/runtime/entrypoints/http_server.py:302`
  `app.include_router(vividvr_flowcut_api.router)` 把 FlowCut 路由挂进服务。
- `python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_api.py:899`
  `VividVRSamplingParams.from_user_kwargs(...)` 在服务入口里被真正调用。

## 3. Vivid-VR 如何注册成 native pipeline

这一章只看“模型身份建立”这条链路。

第一步是定义 pipeline 级默认语义。

- `python/sglang/multimodal_gen/configs/pipeline_configs/vividvr.py:41`
  `class VividVRPipelineConfig` 是 Vivid-VR 的原生配置对象。
- `python/sglang/multimodal_gen/configs/pipeline_configs/vividvr.py:56`
  `vae_tiling` 默认打开，说明这类运行时默认值已经被 pipeline config 固化。
- `python/sglang/multimodal_gen/configs/pipeline_configs/vividvr.py:68-71`
  `caption_source`、`default_prompt_file_path`、`reference_video_path`、`allow_live_cogvlm2_caption` 是当前稳定基线的重要边界。
- `python/sglang/multimodal_gen/configs/pipeline_configs/vividvr.py:78-79`
  prompt 相关默认值继续固定在 config，而不是运行时临时拼接。

第二步是定义请求怎样进入 Vivid-VR 的执行参数对象。

- `python/sglang/multimodal_gen/configs/sample/vividvr.py:14`
  `class VividVRSamplingParams`。
- `python/sglang/multimodal_gen/configs/sample/vividvr.py:206`
  `from_user_kwargs()` 是最关键的方法。它说明服务层不会直接把原始 JSON 丢给 pipeline，而是先经过一个 Vivid-VR 专属的参数收束过程。

第三步是把这两个对象注册到统一注册表。

- `python/sglang/multimodal_gen/registry.py:731-732`
  `sampling_param_cls=VividVRSamplingParams`
  `pipeline_config_cls=VividVRPipelineConfig`

这两行的意义很大。它们不是普通引用，而是在告诉 `multimodal_gen`：

1. 当用户选择 Vivid-VR 这条 pipeline 时，运行时要使用哪一个 pipeline config。
2. 服务层或脚本层生成请求时，最后要实例化哪一个 sampling params 类。

因此，native 集成的最小闭环其实是：

`VividVRPipelineConfig` 定义默认边界  
`VividVRSamplingParams` 定义请求对象  
`registry.py` 把两者注册进系统

如果没有这三步，后面所有 runtime 优化都只会变成“孤立工具函数”，而不是一个正式可选的 pipeline。

## 4. 加速参数和服务参数如何从 `serve` 进入 runtime

这一章要看清楚“外部配置进入系统”的主路径。

### 4.1 `serve` 入口本身很薄

- `python/sglang/multimodal_gen/runtime/entrypoints/cli/serve.py:28`
  `ServerArgs.add_cli_args(parser)` 负责声明 CLI 参数。
- `python/sglang/multimodal_gen/runtime/entrypoints/cli/serve.py:33`
  `ServerArgs.from_cli_args(args)` 把命令行参数转成 `ServerArgs`。

这说明 `serve.py` 自己几乎不承载业务语义。真正的语义都在 `ServerArgs`。

### 4.2 `ServerArgs` 是 runtime 的总配线板

- `python/sglang/multimodal_gen/runtime/server_args.py:97`
  `class ServerArgs`
- `python/sglang/multimodal_gen/runtime/server_args.py:108`
  `attention_backend`
- `python/sglang/multimodal_gen/runtime/server_args.py:124`
  `sp_degree`
- `python/sglang/multimodal_gen/runtime/server_args.py:126`
  `ulysses_degree`
- `python/sglang/multimodal_gen/runtime/server_args.py:176`
  `enable_torch_compile`
- `python/sglang/multimodal_gen/runtime/server_args.py:177-183`
  四类 CogVideoX/Vivid-VR 相关 fusion 开关

这几个字段之所以重要，是因为它们正好覆盖了你关心的四类能力：

1. attention backend
2. SP / Ulysses 并行
3. 算子融合
4. `torch.compile`

### 4.3 参数不是“读完就用”，而是先被调整和校验

- `python/sglang/multimodal_gen/runtime/server_args.py:274`
  `self._adjust_parallelism()`
- `python/sglang/multimodal_gen/runtime/server_args.py:275`
  `self._adjust_attention_backend()`
- `python/sglang/multimodal_gen/runtime/server_args.py:284`
  `self._validate_parallelism()`
- `python/sglang/multimodal_gen/runtime/server_args.py:286`
  `self._validate_vividvr_caption_bridge()`

这四个调用说明 `ServerArgs` 不只是“参数容器”，它还负责把用户输入变成一个合法、可执行、符合当前 Vivid-VR 口径的 runtime 配置。

重点继续往下看：

- `python/sglang/multimodal_gen/runtime/server_args.py:372`
  `_adjust_attention_backend()` 的定义位置。这里负责把外部请求 backend 规范化成运行时接受的口径。
- `python/sglang/multimodal_gen/runtime/server_args.py:433`
  `_adjust_parallelism()` 的定义位置。这里负责把 `sp_degree`、`ulysses_degree` 等并行参数收束成一致组合。
- `python/sglang/multimodal_gen/runtime/server_args.py:1239`
  `_validate_parallelism()` 的定义位置。
- `python/sglang/multimodal_gen/runtime/server_args.py:1281`
  `sp_degree == ring_degree * ulysses_degree` 的约束位置。
- `python/sglang/multimodal_gen/runtime/server_args.py:288`
  `_validate_vividvr_caption_bridge()` 的定义位置。它约束 caption bridge 这种服务功能必须配齐 sidecar 依赖。

### 4.4 这一章最重要的结论

外部命令里的：

- `--attention-backend fa`
- `--sp-degree 2`
- `--ulysses-degree 2`
- `--enable-torch-compile`
- `--enable-cogvideox-*fusion`
- `--vividvr-caption-bridge`

不会直接作用在模型上。它们会先进入 `ServerArgs`，被调整、规范化、校验，然后再交给 `VividVRPipeline.initialize_pipeline()` 去真正应用。

## 5. `VividVRPipeline` 在哪里把加速能力真正挂上去

如果你要找“加速接入点”，主文件一定是 `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py`。

### 5.1 先初始化并行运行时

- `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py:259`
  `_maybe_initialize_model_parallel_runtime(server_args)`

这一步的职责不是直接改 attention kernel，而是先把多卡 / SP 所需的分布式上下文和运行时骨架准备好。没有这一步，后面的 `fa_sp` / `sdpa_sp` 之类分布式 backend 就没有运行条件。

### 5.2 再准备 compile 包装能力

- `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py:282`
  `_maybe_torch_compile_module(...)`

注意这里的设计：`torch.compile` 不是在程序某个全局入口“一键开掉”，而是先封装成“对单个模块进行 compile 的帮助函数”，后面由 `_apply_torch_compile()` 显式调用。这说明当前工程是按模块粒度接入 compile 的。

### 5.3 再构建 runtime acceleration debug 视图

- `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py:382`
  `_build_runtime_acceleration_debug(...)`

这个函数很值得读，因为它把“请求值”和“实际落地值”做了统一汇总。你能从这里读出：

1. 用户请求了什么 backend。
2. 运行时实际解析成了什么 choice。
3. 哪些 fusion 开了。
4. compile 是否开了。

这相当于给整套加速接入提供了一个自解释层。

### 5.4 真正的加速 hook 列表

- `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py:527`
  `_apply_attention_backend(server_args)`
- `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py:600`
  `_apply_qkv_fusion(server_args)`
- `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py:650`
  `_apply_qk_norm_fusion(server_args)`
- `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py:699`
  `_apply_qk_norm_rope_fusion(server_args)`
- `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py:748`
  `_apply_modulation_fusion(server_args)`
- `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py:787`
  `_apply_torch_compile(server_args)`

这些函数的命名非常直接，已经把“能力类型”和“作用时机”暴露出来了。它们的共同特征是：都发生在 pipeline 初始化期间，而不是 forward 期间临时做判断。这说明当前集成策略是“初始化时完成能力装配”，不是“推理时动态走很多分支”。

### 5.5 `initialize_pipeline()` 里的实际调用顺序

- `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py:806`
  `initialize_pipeline()`
- `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py:907-912`
  实际调用顺序如下：
  `907 _apply_attention_backend`
  `908 _apply_qk_norm_fusion`
  `909 _apply_qk_norm_rope_fusion`
  `910 _apply_modulation_fusion`
  `911 _apply_qkv_fusion`
  `912 _apply_torch_compile`

这几行很关键，因为它们把当前工程的“接入优先级”写死了：

1. 先把 backend 选对。
2. 再在正确 backend 语义下决定能不能做融合。
3. 最后再做 compile。

这是一个非常工程化的顺序。因为 compile 通常应该作用在已经完成结构替换、backend 也已经固定好的模块上，而不是先 compile 再去改模块结构。

## 6. Flash Attention / SDPA backend 具体在哪里接

这一章只看 backend，不看算法。

### 6.1 backend 先被规范化

- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_attention_backend.py:45`
  `normalize_cogvideox_attention_backend()`

这个函数的职责是把用户层传进来的 backend 名称统一收束成运行时支持的口径。你可以把它理解为“语义别名清洗层”。它的存在说明对外接口和底层运行时实现之间不是硬绑定字符串，而是允许有历史兼容和别名收敛。

### 6.2 backend 再被解析成 runtime choice

- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_attention_backend.py:76`
  `resolve_cogvideox_attention_runtime_choice()`
- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_attention_backend.py:81`
  这个函数内部会先调用 normalize 逻辑。

这是整个 backend 接入里最值得吃透的函数。因为它回答的不是“用户想要什么 backend”，而是“在当前是否启用 SP 的上下文里，系统最后该跑哪一条 backend 语义”。

当前工程的关键事实是：

1. 用户侧只需要请求 `fa` 或 `sdpa`。
2. 如果没有 SP，运行时就落在单机语义的 `fa` 或 `sdpa`。
3. 如果启用了 SP/Ulysses，运行时会进一步解析成有效 backend `fa_sp` 或 `sdpa_sp`。

这正对应当前仓库口径里“双卡时统一进入 Ulysses distributed joint-attention 语义”的要求。

### 6.3 backend 最终如何被真正写进模块

- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_attention_backend.py:658`
  `set_cogvideox_attention_backend()`
- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_attention_backend.py:848`
  `inspect_cogvideox_attention_backend()`

前者负责设置，后者负责回读。也就是说，backend 在这里不是一个只存在于外部参数层的概念，而是会被显式写进模型对象，并且后续可以被检查出来。

### 6.4 `VividVRPipeline` 怎样调用这套 backend 逻辑

- `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py:527`
  `_apply_attention_backend(server_args)`

这一层的职责是把 `ServerArgs.attention_backend` 和 `sp_degree` 等运行时信息，转成 transformer / controlnet 组件上的实际 backend 设置。也就是说：

`server_args.py` 决定用户请求什么  
`cogvideox_attention_backend.py` 决定这个请求在当前并行语境里应该变成什么 runtime choice  
`vividvr_pipeline.py` 决定把这个 choice 应用到哪些模型组件

### 6.5 controlnet 也走同一套 backend 语义

- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr_controlnet.py:164`
  `set_attention_backend(self, backend: str)`
- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr_controlnet.py:168-169`
  `attention_backend` 属性通过 inspect 逻辑回读

这说明 backend 不是只给主 transformer 用的。Vivid-VR 的 controlnet 分支也会被统一切到相同的 backend 语义上。

### 6.6 看到底层 kernel 痕迹的位置

- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_attention_backend.py:484`
  `flash_attn_func`
- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_attention_backend.py:613`
  创建并缓存 `USPAttention`
- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_attention_backend.py:631`
  返回 `USPAttention`

这些位置说明两件事：

1. `fa` 不是停留在概念层，它最后会落到真实的 Flash Attention kernel 调用。
2. SP 模式下不是简单地在单卡实现外面套通信，而是显式走 `USPAttention` 这条 Ulysses 语义路径。

## 7. 算子融合和 `torch.compile` 是怎么接进去的

这一章分三层看：开关层、应用层、实现层。

### 7.1 开关层：哪些优化可以从 `ServerArgs` 打开

- `python/sglang/multimodal_gen/runtime/server_args.py:176`
  `enable_torch_compile`
- `python/sglang/multimodal_gen/runtime/server_args.py:177`
  `enable_cogvideox_modulation_fusion`
- `python/sglang/multimodal_gen/runtime/server_args.py:179`
  `enable_cogvideox_qkv_fusion`
- `python/sglang/multimodal_gen/runtime/server_args.py:181`
  `enable_cogvideox_qk_norm_fusion`
- `python/sglang/multimodal_gen/runtime/server_args.py:183`
  `enable_cogvideox_qk_norm_rope_fusion`

这说明服务命令层面对这些优化的表达方式非常统一：全都先进入 `ServerArgs`，后续再由 pipeline 初始化阶段决定是否应用。

### 7.2 应用层：`VividVRPipeline` 里哪些 hook 真正做了结构改造

- `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py:600`
  `_apply_qkv_fusion()`
- `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py:650`
  `_apply_qk_norm_fusion()`
- `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py:699`
  `_apply_qk_norm_rope_fusion()`
- `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py:748`
  `_apply_modulation_fusion()`
- `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py:787`
  `_apply_torch_compile()`
- `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py:282`
  `_maybe_torch_compile_module(...)`

这里要理解一个关键工程点：`torch.compile` 在当前实现里不是单独脱离加速链路的。它和 backend、fusion 一样，也是 pipeline 初始化阶段的一个明确 hook。也就是说，compile 是“被接入的能力”，不是环境默认行为。

### 7.3 实现层：融合后的模块长什么样

- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_operator_fusion.py:7`
  导入 `MulAdd`
- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_operator_fusion.py:9-10`
  导入 `LayerNormScaleShift`、`ScaleResidualLayerNormScaleShift`
- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_operator_fusion.py:13`
  `_COGVIDEOX_MODULATION_FUSION_IMPL = "sglang_modulation_fused_ops"`
- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_operator_fusion.py:28-30`
  创建 `LayerNormScaleShift`
- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_operator_fusion.py:59-61`
  创建 `ScaleResidualLayerNormScaleShift`
- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_operator_fusion.py:129`
  `module: MulAdd`
- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_operator_fusion.py:169`
  `self.ff_residual = MulAdd()`
- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_operator_fusion.py:297`
  返回 fusion impl 名称

这几个位置说明，当前项目不是只在 `pipeline` 里打几个布尔开关，然后“期待底层自动优化”。相反，底层已经显式准备好了 fused module 实现，而 pipeline hook 的职责就是在合适时机把原有模块替换成这些 fused 版本。

### 7.4 为什么 compile 放在最后

再看一次 `initialize_pipeline()` 的调用顺序：

- `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py:907-912`

compile 放在最后的原因很直接：

1. 先定 backend，避免 compile 绑定在错误的 kernel 语义上。
2. 再做结构性融合，避免 compile 后再替换模块导致缓存失效或图结构变化。
3. 最后 compile 已经稳定的模块结构。

这正是当前接入顺序体现出来的工程意图。

## 8. SP / Ulysses 并行接入是怎样串起来的

这一章最容易读散，所以按“参数层 -> backend 层 -> Vivid-VR 特有适配层”三步看。

### 8.1 参数层：并行拓扑先在 `ServerArgs` 里收束

- `python/sglang/multimodal_gen/runtime/server_args.py:124`
  `sp_degree`
- `python/sglang/multimodal_gen/runtime/server_args.py:126`
  `ulysses_degree`
- `python/sglang/multimodal_gen/runtime/server_args.py:433`
  `_adjust_parallelism()`
- `python/sglang/multimodal_gen/runtime/server_args.py:1239`
  `_validate_parallelism()`
- `python/sglang/multimodal_gen/runtime/server_args.py:1281`
  `sp_degree == ring_degree * ulysses_degree`

这里的重点是：SP 不是单纯的一个开关，而是一组必须自洽的并行参数。当前工程先在 `ServerArgs` 里把这件事处理干净，然后才让 pipeline 去应用。

### 8.2 backend 层：SP 会改变有效 backend 语义

- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_attention_backend.py:76`
  `resolve_cogvideox_attention_runtime_choice()`
- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_attention_backend.py:613`
  SP 场景创建缓存 `USPAttention`
- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_attention_backend.py:631`
  返回 `USPAttention`

也就是说，当 `sp_degree > 1` 时，变化的不只是通信策略，backend 本身的有效执行语义也会变。用户请求 `fa`，最终可能落成 `fa_sp`；用户请求 `sdpa`，最终可能落成 `sdpa_sp`。

### 8.3 Vivid-VR 特有层：connector/control 的 SP 适配

- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr_common.py:25`
  `_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE_ENV`
- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr_common.py:27`
  `eager_global`
- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr_common.py:28`
  `distributed_local`

这三行非常关键，因为它们表明 Vivid-VR 的 connector 在 SP 下不是完全复用通用逻辑，而是保留了自己的一层上下文策略控制。

继续看 token shard / gather：

- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr_common.py:270`
  `shard_vividvr_video_tokens(...)`
- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr_common.py:334`
  `gather_vividvr_video_tokens(...)`

这两个函数说明 Vivid-VR 在 SP 下对视频 token 做了显式的切分和聚合，不是依赖某个外部黑盒自动完成。

再看 connector 里的 SP attention 路径：

- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr_common.py:220`
  `flash_attn_func`
- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr_common.py:259-260`
  返回 `USPAttention`

这说明 connector 侧的注意力路径也会根据 SP 进入 Ulysses 语义。

最后看 controlnet 侧如何参与：

- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr_controlnet.py:248`
  `shard_vividvr_video_tokens(...)`

这说明 Vivid-VR 的 control 分支不是游离在 SP 之外，而是显式接入了同一套 token shard 语义。

### 8.4 这一章的核心结论

SP 接入不是“只在启动命令里加一个 `--sp-degree 2`”。完整链路是：

1. `ServerArgs` 收束并校验并行参数。
2. attention backend 根据 SP 解析成 `fa_sp` / `sdpa_sp` 这类有效 runtime choice。
3. Vivid-VR connector/control 再通过 shard/gather 和 context mode 衔接自己的局部语义。

## 9. 服务接入：FlowCut API 怎样进入 Vivid-VR pipeline

这一章只看服务链路。

### 9.1 服务从哪里启动

- `python/sglang/multimodal_gen/runtime/launch_server.py:61`
  `launch_server()`
- `python/sglang/multimodal_gen/runtime/launch_server.py:188`
  `launch_http_server_only()`
- `python/sglang/multimodal_gen/runtime/launch_server.py:191`
  `app = create_app(server_args)`

这说明 server 启动分两部分：

1. runtime worker / pipeline 运行时
2. HTTP app

两者之间靠同一个 `server_args` 对象衔接。

### 9.2 Vivid-VR 路由是在哪里挂进去的

- `python/sglang/multimodal_gen/runtime/entrypoints/http_server.py:288`
  `create_app()`
- `python/sglang/multimodal_gen/runtime/entrypoints/http_server.py:302`
  `app.include_router(vividvr_flowcut_api.router)`

这两行说明 FlowCut 不是独立的旁路服务，而是被挂在 `multimodal_gen` 的统一 HTTP server 里。

### 9.3 FlowCut API 自己做了哪些事情

- `python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_api.py:46`
  引入 `read_vividvr_runtime_progress`
- `python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_api.py:304`
  `file_progress = read_vividvr_runtime_progress(progress_path)`

这说明服务层会回读 runtime 写出的进度文件，而不是自己发明一套完全独立的进度体系。

继续看请求进入执行的主线：

- `python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_api.py:816`
  `is_vividvr_video_repair_pipeline(server_args)`
- `python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_api.py:871`
  `caption_file_path = await ensure_vividvr_caption_file(...)`
- `python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_api.py:891`
  `vividvr_kwargs = build_vividvr_repair_kwargs(...)`
- `python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_api.py:899`
  `sampling_params = VividVRSamplingParams.from_user_kwargs(...)`

这四步基本就是服务侧的最小主链：

1. 先确认当前 server 确实在跑 Vivid-VR repair pipeline。
2. 如果请求需要 caption bridge，就先拿到 caption 文件。
3. 再把请求整理成 Vivid-VR 能理解的 kwargs。
4. 最后实例化 `VividVRSamplingParams`，把请求真正送进 native pipeline。

### 9.4 caption bridge 和共享服务逻辑在哪里

- `python/sglang/multimodal_gen/runtime/entrypoints/openai/video_repair_shared.py:46`
  `is_vividvr_video_repair_pipeline`
- `python/sglang/multimodal_gen/runtime/entrypoints/openai/video_repair_shared.py:84`
  `build_vividvr_repair_kwargs`
- `python/sglang/multimodal_gen/runtime/entrypoints/openai/video_repair_shared.py:153`
  `ensure_vividvr_caption_file`

这几个函数说明 caption bridge、请求整理和 pipeline 模式识别并没有散在很多 handler 里，而是被抽成了共享服务工具层。

### 9.5 这一章最重要的结论

服务层没有绕开 native pipeline。它做的事情是：

1. 接收 HTTP 请求。
2. 调用共享逻辑做输入清洗、caption bridge、repair kwargs 组装。
3. 调用 `VividVRSamplingParams.from_user_kwargs(...)`。
4. 回到和本地脚本一致的 pipeline 执行链路。

所以服务接入本质上不是另一套运行时，只是 Vivid-VR native pipeline 的一个对外入口。

## 10. 推荐学习顺序、重点文件与阅读方法

如果你已经知道自己不关心算法，只关心“接入”，最省时间的读法如下。

### 10.1 第一轮：先抓主链

1. `python/sglang/multimodal_gen/runtime/server_args.py:97-183`
   先看有哪些 runtime 参数。
2. `python/sglang/multimodal_gen/runtime/server_args.py:274-288`
   再看参数会被怎样调整和校验。
3. `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py:806-912`
   再看这些参数怎样在 pipeline 初始化时被应用。
4. `python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_api.py:816-899`
   最后看服务请求怎样落成 `VividVRSamplingParams`。

这一轮读完，你就已经能回答“加速能力和服务能力是怎样接进去的”。

### 10.2 第二轮：再补专题

1. backend 专题
   `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_attention_backend.py:45-81, 658, 848`
2. fusion / compile 专题
   `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py:600-787`
   `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_operator_fusion.py:7-169, 297`
3. SP / Ulysses 专题
   `python/sglang/multimodal_gen/runtime/server_args.py:433, 1239, 1281`
   `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr_common.py:25-28, 220, 259-260, 270, 334`
   `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr_controlnet.py:164, 248`

### 10.3 第三轮：回头补“模型身份”

1. `python/sglang/multimodal_gen/configs/pipeline_configs/vividvr.py:41-79`
2. `python/sglang/multimodal_gen/configs/sample/vividvr.py:14, 206`
3. `python/sglang/multimodal_gen/registry.py:731-732`

这一轮的目的，是在已经理解主链后，再回头看“它为什么能作为一个正式 pipeline 存在”，而不是一开始就陷进 config 细节里。

### 10.4 阅读时建议固定问四个问题

1. 这个能力的外部入口参数是什么。
2. 这个参数在 `ServerArgs` 里怎样被调整和校验。
3. 这个参数在 `VividVRPipeline` 里怎样被应用到具体模块。
4. 如果这是服务相关能力，它最后怎样通过 `VividVRSamplingParams.from_user_kwargs()` 回到 native pipeline。

### 10.5 必读文件清单

| 文件 | 必读行号 | 为什么必须读 |
| --- | --- | --- |
| `python/sglang/multimodal_gen/runtime/server_args.py` | `97-183`, `274-288`, `372`, `433`, `1239`, `1281` | 所有加速与服务参数的统一入口、调整点和校验点 |
| `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py` | `259`, `282`, `382`, `527`, `600`, `650`, `699`, `748`, `787`, `806`, `907-912` | runtime 加速能力真正挂接的位置 |
| `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_attention_backend.py` | `45`, `76`, `484`, `613`, `631`, `658`, `848` | backend 规范化、SP runtime choice 和底层 attention 实现落点 |
| `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_operator_fusion.py` | `7-13`, `28-30`, `59-61`, `129`, `169`, `297` | fused operator 的实现层 |
| `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr_common.py` | `25-28`, `220`, `259-260`, `270`, `334` | connector 在 SP 下的上下文策略和 shard/gather 语义 |
| `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr_controlnet.py` | `164`, `168-169`, `248` | controlnet 如何接入同一套 backend / SP 语义 |
| `python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_api.py` | `304`, `816`, `871`, `891`, `899` | 服务请求如何被整理并送入 Vivid-VR sampling params |
| `python/sglang/multimodal_gen/runtime/entrypoints/openai/video_repair_shared.py` | `46`, `84`, `153` | pipeline 识别、caption bridge 和 repair kwargs 组装 |
| `python/sglang/multimodal_gen/configs/pipeline_configs/vividvr.py` | `41`, `56`, `68-79` | Vivid-VR 的 pipeline 默认边界 |
| `python/sglang/multimodal_gen/configs/sample/vividvr.py` | `14`, `206` | 请求对象定义与最终参数收束入口 |
| `python/sglang/multimodal_gen/registry.py` | `731-732` | native pipeline 注册落点 |

如果你只想用最短时间建立全局认识，就按下面这个最小阅读闭环走：

1. `python/sglang/multimodal_gen/runtime/server_args.py:97-183, 274-288`
2. `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py:806-912`
3. `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_attention_backend.py:45-81, 658`
4. `python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_api.py:816-899`

这四段代码足够让你先看清：

`参数怎么进来 -> backend / fusion / compile / SP 怎么挂上去 -> 服务请求怎么落回 native pipeline`
