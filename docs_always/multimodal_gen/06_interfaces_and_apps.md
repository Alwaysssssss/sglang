# 接口层、服务层与应用集成

## 1. 接口层的目标

`multimodal_gen` 并没有把“推理核心”和“接口层”混在一起，而是单独抽出了多种入口：

- Python API
- CLI
- FastAPI/HTTP
- OpenAI 风格 API
- Vertex 风格接口
- WebUI
- ComfyUI

这样做的好处是：同一套运行时内核可以被多种产品形态复用。

## 2. Python API：`DiffGenerator`

位置：`runtime/entrypoints/diffusion_generator.py`

这是最直接的编程接口，适合：

- 本地脚本
- notebook
- SDK 式调用

它提供：

- `from_pretrained()`
- `generate()`
- `generate_with_lora()`
- `set_lora()` / `merge_lora_weights()` / `list_loras()`
- `shutdown()`

本质上它是“面向用户的客户端包装器”。

## 3. CLI

位置：

- `runtime/entrypoints/cli/main.py`
- `runtime/entrypoints/cli/generate.py`
- `runtime/entrypoints/cli/serve.py`

### 3.1 `sglang generate`

特点：

- 构造 `ServerArgs` + `SamplingParams`
- 本地模式拉起 generator
- 调一次 `generate()`
- 可选输出 performance report

它适合单次离线生成和 benchmark。

### 3.2 `sglang serve`

特点：

- 解析 `ServerArgs`
- 调 `dispatch_launch(server_args)`
- 根据 `disagg_role` 决定启动 monolithic server、disagg server 或 role worker

CLI 层非常薄，说明真正的逻辑已经被下沉到了 runtime。

## 4. FastAPI HTTP Server

位置：`runtime/entrypoints/http_server.py`

`create_app(server_args)` 会注册多类 router：

- health router
- vertex router
- OpenAI common router
- `image_api`
- `video_api`
- `mesh_api`
- post-training 的 `weights_api` / `rollout_api`

### 4.1 生命周期

在 FastAPI `lifespan` 中：

1. 初始化 `async_scheduler_client`
2. 后台启动 ZMQ broker
3. 退出时关闭 broker 与 client

这说明 HTTP 层自己不做推理，只是异步前端。

### 4.2 健康与模型发现接口

HTTP server 提供：

- `/health`
- `/server_info`
- `/model_info`
- `/stats`

这些接口不仅给用户看，也显然是为了模型网关/服务发现做兼容。

## 5. OpenAI 风格 API

位置：`runtime/entrypoints/openai/*.py`

这一层做的事情不是“重新实现推理”，而是：

- 定义协议模型；
- 处理 multipart/base64/cloud storage；
- 构造 `SamplingParams`；
- 把请求发给 scheduler；
- 再包装成 OpenAI 风格响应。

### 5.1 `image_api.py`

它支持：

- `/v1/images/generations`
- `/v1/images/edits`

关键逻辑包括：

- 处理上传图片和 URL；
- 保存输入图片；
- 调 `build_sampling_params()`；
- `prepare_request()`；
- `process_generation_batch(...)`；
- 根据 `response_format` 返回 `b64_json` 或 `url`。

### 5.2 为什么这一层值得注意

它不仅是协议包装层，还做了：

- 输出格式选择
- 图片格式选择
- 本地文件与云存储协同
- 响应元数据拼装

因此它已经是一个完整的产品化 API 层，而不只是 demo。

## 6. WebUI

位置：`apps/webui/main.py`

这是一个 Gradio 应用，主要用途是快速交互验证。

特点：

- 通过 `model_info(...).pipeline_tag` 推断任务类型；
- 用 `sync_scheduler_client` 直连 scheduler；
- 把表单输入转成 `SamplingParams`；
- 调 `prepare_request()` 和 `forward()`；
- 最终展示图片或视频。

它本质上是“直接坐在 scheduler client 上的前端”。

## 7. ComfyUI 集成

目录：`apps/ComfyUI_SGLDiffusion/`

这一部分是 `multimodal_gen` 非常有代表性的扩展能力。

### 7.1 `SGLDiffusionGenerator`

位置：`apps/ComfyUI_SGLDiffusion/core/generator.py`

它负责：

- 从 ComfyUI 提供的 checkpoint 推断模型类型；
- 选择对应的 `pipeline_class_name`
- 用 `DiffGenerator.from_pretrained(..., comfyui_mode=True)` 初始化生成器；
- 绑定对应 executor 到 ComfyUI model。

### 7.2 ComfyUI Pipeline 的特点

例如 `runtime/pipelines/comfyui_qwen_image_pipeline.py`：

- 只加载必须组件（通常只保留 transformer + scheduler）
- 依赖 ComfyUI 已经准备好的 prompt embeds / latents
- 用 pass-through scheduler
- 允许直接从 safetensors 单文件加载 transformer

这说明 ComfyUI 集成不是“套一层 API”，而是把 SGLang runtime 嵌进 ComfyUI 的图执行体系。

## 8. `comfyui_mode` 的影响

从多个地方可以看到，`comfyui_mode` 会影响：

- 日志行为
- 输入假设
- stage 设计
- scheduler 使用方式

也就是说，ComfyUI 不是普通客户端，而是被当作另一种宿主运行时。

## 9. 接口层与核心层的边界

这套设计做得比较清楚：

- 接口层负责参数协议与文件管理；
- 核心层负责 `Req -> Pipeline -> OutputBatch`；
- 两者之间用 `SamplingParams` 和 `prepare_request()` 接口粘合。

因此你要新增一个新入口，通常不需要改 pipeline 或 loader，只需要：

1. 解析用户输入；
2. 生成 `SamplingParams`；
3. 调 `prepare_request()`；
4. 发给 scheduler。

## 10. 这层最重要的工程价值

它让 `multimodal_gen` 不只是一个“模型库”，而是一个能直接落地为：

- 本地工具
- 服务 API
- 前端应用
- 工作流节点

的多模态生成平台。

