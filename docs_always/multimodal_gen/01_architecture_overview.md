# 架构总览

## 1. 设计目标

`multimodal_gen` 的核心目标不是“包装 diffusers”，而是提供一套可扩展、可服务化、可多平台运行的多模态生成引擎。它试图统一处理以下问题：

- 多模型家族共存：Wan、Flux、Qwen-Image、Z-Image、GLM-Image、Hunyuan、LTX-2 等；
- 多任务形态共存：T2I、I2I、TI2I、T2V、I2V、TI2V、I2M；
- 多运行模式共存：离线单次生成、本地内嵌服务、HTTP 服务、OpenAI API、ComfyUI、WebUI；
- 多后端共存：SGLang 原生 Pipeline 与 diffusers fallback；
- 多硬件/并行策略共存：单卡、多卡 TP/SP/CFG parallel、CPU offload、FSDP、分段式 disaggregation。

## 2. 分层结构

可以把整个子系统看成 6 层：

### 第 1 层：用户入口层

- Python API：`DiffGenerator`
- CLI：`sglang generate`、`sglang serve`
- HTTP/FastAPI：`runtime/entrypoints/http_server.py`
- OpenAI 风格接口：`runtime/entrypoints/openai/*.py`
- 应用层：`apps/webui`、`apps/ComfyUI_SGLDiffusion`

### 第 2 层：模型识别与配置层

- `registry.py`：识别模型属于哪个 Pipeline/配置族
- `ServerArgs`：服务级参数、并行/平台/offload/端口等
- `PipelineConfig`：模型结构和 Pipeline 行为
- `SamplingParams`：单次请求采样参数

### 第 3 层：运行时调度层

- `launch_server.py`：拉起 worker/scheduler/http server
- `Scheduler`：接收请求、转给 worker、返回结果
- `GPUWorker` / `CPUWorker`：持有真实 Pipeline，并执行 forward

### 第 4 层：Pipeline 编排层

- `build_pipeline`：选择 Pipeline 类
- `ComposedPipelineBase`：加载模块、注册 stage、执行 stage
- `PipelineExecutor`：按并行语义执行 stage

### 第 5 层：Stage 与模型执行层

- `InputValidationStage`
- `TextEncodingStage`
- `ImageEncodingStage`
- `LatentPreparationStage`
- `TimestepPreparationStage`
- `DenoisingStage`
- `DecodingStage`
- 以及若干模型专用 stage

### 第 6 层：组件加载与底层实现层

- `ComponentLoader` / `TransformerLoader` / `VAELoader` / `TextEncoderLoader`
- `runtime/models/*`：自定义 DiT、VAE、Encoder、Scheduler
- `runtime/layers/*`：注意力、量化、LoRA、rotary、linear 等
- `runtime/platforms/*`：平台适配
- `csrc/*`：底层算子和渲染扩展

## 3. 两条核心主线

## 3.1 控制流主线

控制流关心“请求怎么走”：

1. 用户调用 `DiffGenerator.generate()` 或发起 HTTP 请求。
2. 请求被整理为 `SamplingParams`，再封装成 `Req`。
3. `Req` 经 scheduler client 发往 `Scheduler`。
4. `Scheduler` 调 `GPUWorker.execute_forward()`。
5. `GPUWorker` 调 `pipeline.forward(req, server_args)`。
6. `ComposedPipelineBase` 让 executor 按 stage 顺序执行。
7. 最终 `OutputBatch` 返回到上层，再做文件保存、base64/URL 包装等。

## 3.2 数据流主线

数据流关心“状态怎么传”：

- `SamplingParams` 是用户显式传入的采样参数；
- `Req` 是贯穿整个 pipeline 的状态容器；
- 每个 stage 在 `Req` 上增量写入自己的产物；
- 最终 `DecodingStage` 或 diffusers stage 把结果写入 `OutputBatch`。

这种做法的关键收益是：stage 之间不需要长参数列表，而是围绕同一个 `Req` 协作。

## 4. 典型请求链路

以离线单次生成为例：

1. `DiffGenerator.from_pretrained(...)`
2. `ServerArgs.from_kwargs(...)`
3. `launch_server(..., launch_http_server=False)`
4. `GPUWorker.build_pipeline(...)`
5. `DiffGenerator.generate(...)`
6. `prepare_request(...) -> Req`
7. `sync_scheduler_client.forward([req])`
8. `Scheduler._handle_generation(...)`
9. `GPUWorker.execute_forward(...)`
10. `pipeline.forward(...)`
11. `save_outputs(...)`

以 HTTP 服务为例，多了一层：

`FastAPI route -> build_sampling_params -> prepare_request -> async_scheduler_client.forward`

## 5. 为什么它能统一很多模型

原因不在于所有模型逻辑都一样，而在于它把“变化点”收束到了几个可替换抽象：

- `registry.py` 决定模型属于哪一类；
- `PipelineConfig` 决定该类模型的结构和行为差异；
- `ComponentLoader` 决定不同组件怎么加载；
- `Pipeline` 决定 stage 组合方式；
- `model_specific_stages` 处理少量无法抽象进通用 stage 的特殊逻辑。

也就是说，这套系统本质上是“通用骨架 + 模型特化点”的设计。

## 6. 架构上的关键判断

### 6.1 Pipeline 是“组合式”，不是“大一统 if/else”

每个模型 Pipeline 只负责两件事：

- 声明需要哪些模块；
- 决定 stage 怎么排。

这样比把所有模型逻辑堆进一个 `forward()` 更可维护。

### 6.2 配置与执行解耦

- `PipelineConfig` 偏“静态结构/行为配置”
- `SamplingParams` 偏“单次请求参数”
- `Req` 偏“执行时状态”

三者分离后，框架同时保住了灵活性和可读性。

### 6.3 原生实现优先，diffusers 作为兜底

`registry.get_model_info()` 会优先尝试原生 SGLang Pipeline；只有在找不到原生实现、模型不匹配或显式指定 `backend=diffusers` 时，才退回 `DiffusersPipeline`。

这保证了：

- 已适配模型可以获得自定义优化；
- 未适配模型仍然可以通过 diffusers 跑起来。

## 7. 阅读源码时最值得优先理解的类

- `DiffGenerator`
- `ServerArgs`
- `PipelineConfig`
- `SamplingParams`
- `Req`
- `Scheduler`
- `GPUWorker`
- `ComposedPipelineBase`
- `PipelineStage`
- `DenoisingStage`
- `ComponentLoader`



