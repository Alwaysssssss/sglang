# 模型注册与配置系统

## 1. `registry.py` 的职责

`python/sglang/multimodal_gen/registry.py` 是整个系统的“模型识别中枢”。它做两类事：

1. 动态发现有哪些 Pipeline 类可用；
2. 把模型路径映射到 `PipelineConfig` / `SamplingParams` / `Pipeline`。

## 2. Pipeline 发现机制

### 2.1 自动扫描 `runtime.pipelines`

`_discover_and_register_pipelines()` 会扫描 `sglang.multimodal_gen.runtime.pipelines` 下所有模块，寻找 `EntryClass`：

- `EntryClass = FluxPipeline`
- 或 `EntryClass = [QwenImagePipeline, QwenImageEditPipeline, ...]`

如果类是 `ComposedPipelineBase` 的子类，就以其 `pipeline_name` 注册到 `_PIPELINE_REGISTRY`。

### 2.2 ComfyUI 的特殊处理

如果某个 Pipeline 类还定义了：

- `pipeline_config_cls`
- `sampling_params_cls`

那么它还会被登记进 `_PIPELINE_CONFIG_REGISTRY`。这主要是给 ComfyUI safetensors 单文件加载场景用的，因为这种场景可能没有完整的 `model_index.json`。

## 3. 模型配置注册机制

`_register_configs()` 把不同模型家族注册到 `_CONFIG_REGISTRY`：

- `sampling_param_cls`
- `pipeline_config_cls`
- `hf_model_paths`
- `model_detectors`

例如：

- Wan 系列 -> `Wan*SamplingParams` + `Wan*PipelineConfig`
- Flux 系列 -> `Flux*SamplingParams` + `Flux*PipelineConfig`
- Qwen-Image 系列 -> `QwenImage*SamplingParams` + `QwenImage*PipelineConfig`

这样做的好处是：模型识别与 Pipeline 具体实现解耦了。你可以新增一个模型家族，而不用改所有入口。

## 4. `get_model_info()` 的解析流程

`get_model_info(model_path, backend, model_id)` 是最关键的解析函数。

它大致按下面的优先级工作：

1. 规范化 backend：`auto` / `sglang` / `diffusers`
2. 如果显式指定 `diffusers`，直接走 `_get_diffusers_model_info()`
3. 自动发现所有 Pipeline
4. 若检测到量化格式但原生后端不支持，则回退 diffusers
5. 尝试确定 `pipeline_class_name`
   - 先看是不是已知 non-diffusers 模型
   - 否则下载/读取 `model_index.json` 的 `_class_name`
6. 用 `_PIPELINE_REGISTRY` 找到原生 Pipeline 类
7. 用 `_get_config_info()` 找到 `PipelineConfig` / `SamplingParams`
8. 若任一环失败且 backend 是 `auto`，则回退 diffusers

最终返回 `ModelInfo`：

- `pipeline_cls`
- `sampling_param_cls`
- `pipeline_config_cls`

## 5. `_get_config_info()` 的匹配策略

这个函数比表面上复杂，实际上做了多级匹配：

1. `model_id` 显式覆盖；
2. `model_path` 精确匹配；
3. 短名/子串匹配；
4. HuggingFace cache 路径匹配；
5. 基于 detector 的启发式匹配；
6. 读 `model_index.json` 的 `_class_name` 做辅助判断。

这也是为什么即使你传入的是本地 cache 路径，系统通常也能识别出模型族。

## 6. `PipelineConfig`、`SamplingParams`、`ServerArgs` 的边界

这三个类容易混淆，但职责很清晰。

## 6.1 `PipelineConfig`

位置：`configs/pipeline_configs/base.py`

它描述“模型和 pipeline 的静态行为”，例如：

- `task_type`
- `dit_config`
- `vae_config`
- `text_encoder_configs`
- prompt/image 的预处理与后处理函数
- latent shape、decode scale/shift、scheduler 参数处理逻辑

它更像“这个模型应该怎样被执行”的定义。

## 6.2 `SamplingParams`

位置：`configs/sample/sampling_params.py`

它描述“这一次请求想怎么生成”，例如：

- `prompt` / `negative_prompt`
- `image_path`
- `height` / `width` / `num_frames`
- `num_inference_steps`
- `guidance_scale`
- `save_output`
- `return_trajectory_latents`
- 插帧、超分、性能 dump 等

它更像“每次请求的用户参数”。

## 6.3 `ServerArgs`

位置：`runtime/server_args.py`

它描述“服务/运行时如何启动”，例如：

- `model_path`
- `backend`
- `num_gpus` / `tp_size` / `sp_degree`
- `dit_cpu_offload` / `vae_cpu_offload`
- `attention_backend`
- `host` / `port` / `scheduler_port`
- `component_paths`
- `pipeline_class_name`

它更像“服务实例级别的控制面配置”。

## 7. `PipelineConfig.from_kwargs()` 的意义

`PipelineConfig.from_kwargs()` 会在 `ServerArgs` 初始化时被调用。它做了三件非常重要的事：

1. 根据 `model_path` 先从 registry 解析出正确的 `pipeline_config_cls`；
2. 如果用户给了 JSON / dict / 已实例化的 PipelineConfig，则在默认配置上覆盖；
3. 再把 CLI / kwargs 参数更新到 config 上。

因此，它不是简单构造 dataclass，而是“基于模型族做配置归一化”。

## 8. 参数优先级

从 `DiffGenerator.from_pretrained()` 以及相关注释看，优先级基本是：

`模型默认配置 < 用户传入 PipelineConfig < 用户 kwargs`

这使得调用者既可以：

- 什么都不管，吃默认；
- 也可以精准覆盖某些行为；
- 还可以通过 `component_paths` 替换某个组件权重。

## 9. `Req`：配置到执行态的桥梁

位置：`runtime/pipelines_core/schedule_batch.py`

`Req` 是贯穿 stage 的状态对象，内部持有 `sampling_params`，并通过 `__getattr__` / `__setattr__` 将很多字段透明代理到 `SamplingParams`。

这意味着：

- Stage 可以像访问普通字段一样访问 `req.prompt`、`req.height`；
- 但这些字段实际可能来自 `SamplingParams`；
- 同时 `Req` 还保存执行态中间结果，如 `prompt_embeds`、`latents`、`timesteps`、`image_latent`、`trajectory_latents`。

这是该框架最关键的状态组织方式之一。

## 10. 自动调参逻辑在 `ServerArgs`

`ServerArgs.__post_init__()` 会触发一系列自动调整：

- offload 默认值推断；
- warmup 行为；
- 端口选择；
- TP/SP/CFG parallel 自动推断；
- 平台特定调整；
- autocast 开关；
- `PipelineConfig` 二次调整。

几个值得注意的点：

- 图像模型与视频模型的 offload 默认策略不同；
- 某些模型会自动启用 `dit_layerwise_offload`；
- 未显式指定时，会根据模型默认是否使用 CFG 决定是否自动启用 CFG parallel；
- LTX-2.3 有专门的两阶段 device mode 逻辑。

## 11. diffusers fallback 的处理很细

`_get_diffusers_model_info()` 并不是简单地返回一个通用 config。它会：

- 使用 `DiffusersPipeline`；
- 同时尽量继承原生注册模型的 `task_type`。

这样做的价值是：即使退回 diffusers，系统仍然知道这是 T2I、I2V 还是 TI2I，因此输入校验、图片输入接受能力等逻辑仍然能工作。

## 12. 配置系统的整体评价

这套配置系统的核心优点是：

- 模型识别和运行时参数清晰分层；
- 兼容 CLI、Python API、HTTP 三种入口；
- 对“本地路径、HF repo、本地 cache、safetensors 单文件”都有兼容逻辑；
- 能在不破坏统一框架的前提下，为模型家族做强特化。

代价是：

- 入口参数和自动推断逻辑很多；
- 初读时容易分不清“静态配置”和“请求参数”。

理解这个模块的关键，不是记住所有字段，而是抓住三层边界：

- `ServerArgs` 管服务；
- `PipelineConfig` 管模型；
- `SamplingParams` 管请求。

