# multimodal_gen 接口层（Entrypoints）源码详解

> 学习目标：从 0 理解 sglang 多模态生成子系统的**对外服务层**是如何设计的。

---

## 一、目录全景

```
runtime/entrypoints/
├── __init__.py                   # 包初始化（日志抑制）
├── diffusion_generator.py        # 核心 Python API：DiffGenerator
├── http_server.py                # FastAPI HTTP Server + Vertex AI 接口
├── utils.py                      # 请求构造 / 结果持久化 / 后处理
│
├── cli/                          # 命令行工具
│   ├── __init__.py
│   ├── main.py                   # CLI 入口 main()
│   ├── generate.py               # sglang generate 子命令
│   ├── serve.py                  # sglang serve 子命令
│   ├── cli_types.py              # CLI 子命令基类
│   └── utils.py                  # 分布式启动工具
│
├── openai/                       # OpenAI 兼容 API
│   ├── protocol.py               # Pydantic 请求/响应模型
│   ├── common_api.py             # /v1/models、LoRA 管理、model_info
│   ├── image_api.py              # /v1/images/generations + /edits
│   ├── video_api.py              # /v1/videos CRUD + 异步任务
│   ├── mesh_api.py               # /v1/meshes 3D 网格生成
│   ├── utils.py                  # build_sampling_params、图像下载/上传
│   ├── storage.py                # S3 云存储
│   └── stores.py                 # 内存 KV 存储（任务状态）
│
└── post_training/                # 训练后权重管理
    ├── __init__.py
    ├── weights_api.py            # 权重热更新 + checksum
    └── io_struct.py              # 请求数据结构
```

---

## 二、核心设计思想：接口层与核心层的解耦

在深入每个文件之前，理解**接口层在整个系统中的位置**是关键。

```
┌──────────────────────────────────────────────────────────┐
│                     用户 / 客户端                          │
│         CLI  │  Python SDK  │  HTTP  │  OpenAI API       │
└────────────────────┬─────────────────────────────────────┘
                     │
     ┌───────────────┴────────────────┐
     │   接 口 层  (entrypoints/)      │
     │   - 参数解析 & 协议适配          │
     │   - 文件上传/下载/格式转换        │
     │   - SamplingParams 构造          │
     │   - 响应拼装（b64_json/url等）    │
     └───────────────┬────────────────┘
                     │  prepare_request() → Req
                     │  scheduler_client.forward(Req)
                     ▼
     ┌────────────────────────────────┐
     │   核 心 层  (runtime/)          │
     │   Scheduler → GPUWorker →      │
     │   ComposedPipeline → Output    │
     └────────────────────────────────┘
```

**接口层只做三件事**：
1. 把用户输入变成 `SamplingParams`
2. 调 `prepare_request()` 生成 `Req`
3. 把推理结果包装成用户期望的格式返回

**接口层不做的**：模型加载、调度、前向推理、GPU 管理 —— 这些全部在核心层。

---

## 三、核心粘合层：`utils.py` — 请求构造与结果处理

文件：`runtime/entrypoints/utils.py`

这是接口层最重要的基础设施文件，所有入口（CLI、HTTP、OpenAI API）都依赖它。

### 3.1 `prepare_request()` — 从参数到请求

```python
def prepare_request(
    server_args: ServerArgs,
    sampling_params: SamplingParams,
) -> Req:
```

**作用**：将 `SamplingParams`（用户意图）转换成核心层能理解的 `Req` 对象。

**核心流程**：
1. 创建 `Req(sampling_params=sampling_params, VSA_sparsity=...)`
2. 提取 `diffusers_kwargs` 放入 `req.extra`
3. 调 `req.adjust_size(server_args)` 自适应宽高
4. 校验 `prompt` 类型和宽高的合法性

**关键设计**：这是接口层和核心层之间的**唯一桥梁**。所有入口都通过这个函数把用户参数翻译成 `Req`。

### 3.2 `GenerationResult` — 统一的结果容器

```python
@dataclass
class GenerationResult:
    samples: Any = None          # 生成的图像/视频数据（tensor/numpy）
    frames: Any = None           # 帧列表
    audio: Any = None            # 音频数据
    prompt: str | None = None
    size: tuple | None = None    # (height, width, num_frames)
    generation_time: float = 0.0
    peak_memory_mb: float = 0.0
    metrics: dict = ...
    output_file_path: str | None = None
```

每种接口都可以从这个统一容器中提取自己需要的字段。

### 3.3 `save_outputs()` — 输出持久化

```python
def save_outputs(
    outputs, data_type, fps, save_output, build_output_path, *,
    audio, audio_sample_rate, samples_out, audios_out, frames_out,
    output_compression, enable_frame_interpolation, ...
) -> list[str]:
```

**作用**：将推理结果的 tensor/numpy 数据保存为文件（图片/视频），同时支持：
- 视频帧插值（frame interpolation）
- 超分辨率（upscaling）
- 音频合成到视频（ffmpeg mux）
- 压缩质量控制

### 3.4 辅助请求类型

```python
@dataclass
class SetLoraReq:        # 设置 LoRA 适配器
    lora_nickname, lora_path, target, strength

@dataclass
class MergeLoraWeightsReq:   # 合并 LoRA 权重

@dataclass
class UnmergeLoraWeightsReq: # 取消合并

@dataclass
class ListLorasReq:          # 列出所有 LoRA

@dataclass
class ShutdownReq:           # 关闭服务
```

这些是跨模块共享的**控制类请求类型**，不包含推理参数，直接发给 scheduler。

---

## 四、Python API：`DiffGenerator`

文件：`runtime/entrypoints/diffusion_generator.py`

### 4.1 设计定位

`DiffGenerator` 是面向开发者的**最直接编程入口**，适合本地脚本、Jupyter Notebook、SDK 调用。

### 4.2 关键方法

| 方法 | 作用 |
|---|---|
| `from_pretrained(**kwargs)` | 类工厂方法，从模型路径创建实例 |
| `from_server_args(server_args, local_mode)` | 从 ServerArgs 创建，控制本地/远程模式 |
| `generate(sampling_params_kwargs)` | **核心方法**：参数→推理→结果 |
| `generate_with_lora(prompt, lora_path, ...)` | 带 LoRA 的生成 |
| `set_lora(...)` / `merge_lora_weights(...)` | LoRA 管理 |
| `shutdown()` | 优雅关闭 |
| `__enter__` / `__exit__` | 上下文管理器支持 |

### 4.3 `generate()` 的完整执行流程

```
generate(sampling_params_kwargs)
  │
  ├─ 1. _resolve_prompts()
  │    从 prompt / prompt_path / prompt_file_path 解析提示词列表
  │
  ├─ 2. SamplingParams.from_user_sampling_params_args()
  │    将 kwargs 转换为类型安全的 SamplingParams
  │
  ├─ 3. 对每个 prompt 循环：
  │     ├─ prepare_request(server_args, sampling_params) → Req
  │     └─ _send_to_scheduler_and_wait_for_response([req])
  │           └─ sync_scheduler_client.forward(batch) → OutputBatch
  │
  ├─ 4. save_outputs()  将 tensor 存为图像/视频文件
  │
  ├─ 5. 组装 GenerationResult 列表
  │
  └─ 6. _log_summary()  输出耗时和显存统计
```

### 4.4 本地模式 vs 远程模式

```python
if local_mode:
    # 启动本地 scheduler 进程
    instance.local_scheduler_process = instance._start_local_server_if_needed()
else:
    # 连接远程 scheduler，验证可达性
    sync_scheduler_client.initialize(server_args)
    instance._check_remote_scheduler()
```

- **本地模式**：`DiffGenerator` 自己启动 scheduler 进程（适合单机脚本）
- **远程模式**：连接已运行的 scheduler 服务（适合客户端-服务器架构）

### 4.5 生命周期管理

`DiffGenerator` 实现了上下文管理器协议，支持 `with` 语句：

```python
with DiffGenerator.from_pretrained("model_path") as gen:
    result = gen.generate(...)
# 自动调用 shutdown()
```

也注册了 `__del__` 作为兜底，防止垃圾回收时资源泄漏。

---

## 五、CLI 命令行工具

目录：`runtime/entrypoints/cli/`

### 5.1 架构

```
main.py
  ├─ main()                         # CLI 入口
  │   ├─ GenerateSubcommand         # sglang generate
  │   └─ ServeSubcommand            # sglang serve
  │
  ├─ cli_types.py
  │   └─ CLISubcommand (ABC)        # 子命令基类
  │       ├─ name: str
  │       ├─ cmd(args)              # 执行逻辑
  │       ├─ validate(args)         # 参数校验
  │       └─ subparser_init(...)    # 注册 argparse 子解析器
```

`CLISubcommand` 是一个抽象基类，定义了 CLI 子命令的三段式接口：
1. **subparser_init** — 在 argparse 中注册子命令和参数
2. **validate** — 参数合法性校验
3. **cmd** — 执行命令逻辑

### 5.2 `sglang generate` — 单次生成

```python
# generate.py
def generate_cmd(args, unknown_args):
    server_args = ServerArgs.from_cli_args(args, unknown_args)
    sampling_params_kwargs = SamplingParams.get_cli_args(args)

    generator = DiffGenerator.from_pretrained(
        model_path=server_args.model_path,
        server_args=server_args,
        local_mode=True,
    )
    results = generator.generate(sampling_params_kwargs=sampling_params_kwargs)
    maybe_dump_performance(args, server_args, prompt, results)  # 可选性能报告
```

**流程极简**：解析参数 → 创建 `DiffGenerator` → 调 `generate()` → 可选 dump 性能数据。

支持从 JSON/YAML 配置文件加载参数（`--config`），配置文件中的 `SamplingParams` 字段会自动映射。

### 5.3 `sglang serve` — 启动服务

```python
# serve.py
def execute_serve_cmd(args, unknown_args):
    server_args = ServerArgs.from_cli_args(args, unknown_args)
    launch_server(server_args)           # 启动 scheduler + worker 进程

    if server_args.webui:
        run_sgl_diffusion_webui(server_args)  # 可选启动 WebUI
```

更简单：解析参数 → 调 `launch_server()` → 可选 WebUI。

### 5.4 `utils.py` — 分布式启动

```python
def launch_distributed(num_gpus, args, master_port=None):
    # 使用 torch.distributed.run 启动多 GPU 推理
    cmd = [python, "-m", "torch.distributed.run", f"--nproc_per_node={num_gpus}", ...]
    subprocess.Popen(cmd, ...)
```

用于多 GPU 分布式推理的子进程启动。

---

## 六、FastAPI HTTP Server

文件：`runtime/entrypoints/http_server.py`

### 6.1 应用工厂函数

```python
def create_app(server_args: ServerArgs) -> FastAPI:
    app = FastAPI(lifespan=lifespan)

    app.include_router(health_router)        # /health, /models, /server_info 等
    app.include_router(vertex_router)        # Vertex AI 格式接口
    app.include_router(common_api.router)    # /v1/models, LoRA 管理
    app.include_router(image_api.router)     # /v1/images/*
    app.include_router(video_api.router)     # /v1/videos/*
    app.include_router(mesh_api.router)      # /v1/meshes/*
    app.include_router(weights_api.router)   # 权重更新

    app.state.server_args = server_args
    return app
```

所有 API 通过 FastAPI 的 Router 机制模块化注册。

### 6.2 生命周期管理

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时
    async_scheduler_client.initialize(server_args)     # 初始化异步 scheduler 客户端
    broker_task = asyncio.create_task(run_zeromq_broker(server_args))  # 后台 ZMQ broker

    yield  # 服务运行中...

    # 关闭时
    broker_task.cancel()
    async_scheduler_client.close()
```

**核心设计**：HTTP 层自己不做推理，只是异步前端。它通过 `async_scheduler_client` 把请求转发给 scheduler。

### 6.3 健康检查与模型发现接口

| 端点 | 作用 |
|---|---|
| `/health` | 存活检查，返回 `{"status": "ok"}` |
| `/health_generate` | 生成健康检查（TODO） |
| `/models` (已废弃) | 模型信息，请用 `/v1/models` |
| `/server_info` | 服务信息（model_path、tp_size 等），兼容模型网关 |
| `/model_info` | 模型能力信息（task_type, is_image_gen 等），供网关做服务发现 |

这些接口显然是为生产环境的**模型网关/服务发现**设计的。

### 6.4 Vertex AI 接口

```python
@vertex_router.post(VERTEX_ROUTE)  # 默认 /vertex_generate
async def vertex_generate(vertex_req: VertexGenerateReqInput):
    # 对每个 instance 构造 SamplingParams → prepare_request → forward
    # 并发执行所有请求
    results = await asyncio.gather(*futures)
    return {"predictions": results}
```

兼容 Google Vertex AI 的预测请求格式，支持批量并发推理。

---

## 七、OpenAI 兼容 API

目录：`runtime/entrypoints/openai/`

### 7.1 协议层：`protocol.py`

定义了所有 API 的 Pydantic 请求/响应模型：

| 模型 | 用途 |
|---|---|
| `ImageGenerationsRequest` | 图片生成请求（兼容 OpenAI） |
| `ImageResponse` / `ImageResponseData` | 图片生成响应（b64_json / url） |
| `VideoGenerationsRequest` | 视频生成请求 |
| `VideoResponse` | 视频生成响应（异步任务模型） |
| `VideoRepairRequest` | 视频修复请求 |
| `MeshGenerationsRequest` | 3D 网格生成请求 |
| `MeshResponse` | 3D 网格响应 |
| `VertexGenerateReqInput` | Vertex AI 请求格式 |

这些模型不仅做类型校验，还定义了 SGLang 的扩展参数（如 `enable_teacache`、`enable_upscaling`、`diffusers_kwargs` 等）。

### 7.2 工具函数：`openai/utils.py`

#### `build_sampling_params()` — 参数转换核心

```python
def build_sampling_params(request_id: str, **kwargs) -> SamplingParams:
```

**职责**：
1. 解析 `"WxH"` 格式的 size 字符串
2. 处理 `output_quality` → `output_compression` 映射
3. 过滤 `None` 值，让 `SamplingParams` 的默认值生效
4. 调 `SamplingParams.from_user_sampling_params_args()` 构造最终参数

#### 其他关键函数

| 函数 | 作用 |
|---|---|
| `temp_dir_if_disabled(path)` | 上下文管理器：有持久化路径就用，否则创建临时目录 |
| `save_image_to_path(image, path)` | 保存上传文件/URL/base64 图片到本地 |
| `process_generation_batch(client, batch)` | 发给 scheduler → 保存输出 → 记录性能 |
| `merge_image_input_list(*inputs)` | 合并多个图片输入源（支持单个和列表） |
| `add_common_data_to_response(resp, ...)` | 向响应注入 peak_memory_mb 和 inference_time_s |

### 7.3 图片 API：`image_api.py`

**端点**：

| 方法 | 路径 | 功能 |
|---|---|---|
| POST | `/v1/images/generations` | 文生图 |
| POST | `/v1/images/edits` | 图生图（编辑） |
| GET | `/v1/images/{image_id}/content` | 下载已生成的图片 |

**`/v1/images/generations` 的完整流程**：

```
1. 解析 ImageGenerationsRequest
       │
2. choose_output_image_ext()  根据 output_format 和 background 确定扩展名
       │
3. build_sampling_params()    转换为 SamplingParams
       │
4. prepare_request()          生成 Req
       │
5. process_generation_batch() 发给 scheduler → 等待结果 → 保存文件
       │
6. _read_b64_for_paths()      （如需 b64_json）预读文件为 base64
       │
7. cloud_storage.upload_and_cleanup()  上传到 S3（如配置）
       │
8. 构造 ImageResponse  根据 response_format 返回 b64_json 或 url
```

**`/v1/images/edits`** 额外支持：
- `image` / `image[]` 上传多张输入图
- `url` / `url[]` 通过 URL 引用图片
- `mask` 上传遮罩

### 7.4 视频 API：`video_api.py`

**端点**：

| 方法 | 路径 | 功能 |
|---|---|---|
| POST | `/v1/videos` | 创建视频生成任务（异步） |
| POST | `/v1/videos/repairs` | 创建视频修复任务（异步） |
| GET | `/v1/videos` | 列出所有视频任务（支持分页） |
| GET | `/v1/videos/{video_id}` | 查询单个任务状态 |
| GET | `/v1/videos/{video_id}/progress` | 查询任务进度（含 callback 状态） |
| GET | `/v1/videos/{video_id}/content` | 下载已生成的视频 |
| DELETE | `/v1/videos/{video_id}` | 删除任务 |

**关键设计：异步任务模型**

视频生成耗时较长，因此采用**异步任务模型**：

```
POST /v1/videos
  │
  ├─ 1. 保存输入图片（如有）
  ├─ 2. 构造 SamplingParams + Req
  ├─ 3. 创建 job 记录到 VIDEO_STORE（status="queued"）
  ├─ 4. asyncio.create_task(_dispatch_job_async(...))   ← 后台执行
  └─ 5. 立即返回 VideoResponse（status="queued"）

用户轮询 GET /v1/videos/{id} 查看进度
  status: "queued" → "running" → "completed" | "failed"
```

**视频修复（VideoEdit）**额外支持：
- `video_input_path` / `video_url`：输入视频
- `mask_input_path` / `mask_url`：遮罩
- 信号量限流（`VIDEOEDIT_QUEUE_CAPACITY`，默认 1）
- 回调通知（`callback_url`），支持 3 次重试，指数退避

### 7.5 网格 API：`mesh_api.py`

**端点**：与视频 API 结构类似，支持 3D 网格（`.glb` / `.obj`）的创建、查询、下载、删除。

流程与视频 API 相同：`POST 创建任务 → 异步执行 → 轮询状态 → 下载结果`。

### 7.6 通用 API：`common_api.py`

**端点**：

| 方法 | 路径 | 功能 |
|---|---|---|
| GET | `/v1/models` | 列出模型（OpenAI 兼容 + 扩散模型扩展字段） |
| GET | `/v1/models/{model}` | 获取单个模型详情 |
| GET | `/v1/model_info` | 模型基本信息 |
| POST | `/v1/set_lora` | 设置 LoRA 适配器（支持批量） |
| POST | `/v1/merge_lora_weights` | 合并 LoRA 权重 |
| POST | `/v1/unmerge_lora_weights` | 取消合并 LoRA |
| GET | `/v1/list_loras` | 列出已加载的 LoRA |

`/v1/models` 返回的 `DiffusionModelCard` 包含扩散模型特有的字段：`num_gpus`、`task_type`、`dit_precision`、`vae_precision`、`pipeline_name` 等。

### 7.7 云存储：`storage.py`

```python
class CloudStorage:
    def __init__(self):
        self.enabled = (SGLANG_CLOUD_STORAGE_TYPE == "s3")

    async def upload_file(local_path, destination_key):
        # 使用 boto3 上传到 S3

    async def upload_and_cleanup(file_path):
        # 上传成功后删除本地文件
```

通过环境变量 `SGLANG_CLOUD_STORAGE_TYPE=s3` 启用，支持任意 S3 兼容存储（AWS S3、MinIO 等）。

### 7.8 内存存储：`stores.py`

```python
class AsyncDictStore:
    """async-safe 的内存 KV 存储"""
    async def upsert(key, value)
    async def update_fields(key, updates)
    async def get(key) / pop(key) / list_values()

# 全局实例
VIDEO_STORE = AsyncDictStore()
IMAGE_STORE = AsyncDictStore()
MESH_STORE = AsyncDictStore()
```

用于存储异步任务的状态（queued / running / completed / failed），支持并发安全的读写。

---

## 八、Post-Training 权重 API

目录：`runtime/entrypoints/post_training/`

### 8.1 `weights_api.py`

| 方法 | 路径 | 功能 |
|---|---|---|
| POST | `/update_weights_from_disk` | 从磁盘热更新模型权重，无需重启 |
| POST | `/get_weights_checksum` | 获取指定模块权重的 SHA-256 校验和 |

```python
# 权重热更新
@router.post("/update_weights_from_disk")
async def update_weights_from_disk(request: Request):
    body = await request.json()
    req = UpdateWeightFromDiskReqInput(
        model_path=body.get("model_path"),
        flush_cache=body.get("flush_cache", True),
        target_modules=body.get("target_modules"),
    )
    response = await async_scheduler_client.forward(req)
    # ...
```

这为在线模型更新（如 LoRA 训练后的权重回写）提供了 API 支持。

---

## 九、完整请求链路图

以 `/v1/images/generations` 为例，追踪一次完整的请求：

```
HTTP POST /v1/images/generations
  │  {"prompt": "a cat", "n": 1, "size": "1024x1024", "response_format": "b64_json"}
  ▼
image_api.generations()                          # openai/image_api.py
  │
  ├─ generate_request_id()                       # 生成唯一请求 ID
  ├─ choose_output_image_ext("png", "auto")       # → "png"
  │
  ├─ build_sampling_params(                       # openai/utils.py
  │     request_id, prompt="a cat",
  │     size="1024x1024", → width=1024, height=1024
  │     num_outputs_per_prompt=1,
  │     output_file_name=f"{request_id}.png",
  │     seed=1024, ...
  │  ) → SamplingParams
  │
  ├─ prepare_request(server_args, sampling_params) # entrypoints/utils.py
  │     → Req(sampling_params=..., VSA_sparsity=..., ...)
  │
  ├─ process_generation_batch(client, batch)       # openai/utils.py
  │     ├─ async_scheduler_client.forward([batch]) # → ZMQ → Scheduler → GPUWorker
  │     └─ save_outputs(...)                       # 保存为 PNG 文件
  │
  ├─ _read_b64_for_paths(save_file_path_list)      # 读文件 → base64
  ├─ cloud_storage.upload_and_cleanup(...)         # 如有 S3 配置则上传
  │
  └─ 返回 ImageResponse(
        id=request_id,
        data=[ImageResponseData(b64_json="...", revised_prompt="a cat")]
     )
```

---

## 十、如何新增一个入口

基于这套架构，新增一个入口（如 gRPC、WebSocket 等）只需四步：

```
1. 解析用户输入  →  拿到 prompt, size, seed 等参数
2. build_sampling_params()  →  构造 SamplingParams
3. prepare_request()  →  生成 Req
4. scheduler_client.forward(Req)  →  获取 OutputBatch → 包装响应
```

**不需要**修改 pipeline、loader、scheduler 等任何核心层代码。

---

## 十一、关键文件速查表

| 文件 | 核心内容 | 代码行数 |
|---|---|---|
| `diffusion_generator.py` | `DiffGenerator` 类：`from_pretrained()`, `generate()`, LoRA 管理, 生命周期 | ~567 |
| `http_server.py` | FastAPI 应用工厂, lifespan, health/model_info/vertex 路由 | ~288 |
| `utils.py` | `prepare_request()`, `GenerationResult`, `save_outputs()`, `post_process_sample()` | ~518 |
| `cli/main.py` | CLI 入口, 子命令注册与分发 | ~44 |
| `cli/generate.py` | `sglang generate` 实现 | ~200 |
| `cli/serve.py` | `sglang serve` 实现 | ~72 |
| `openai/protocol.py` | 所有 Pydantic 请求/响应模型 | ~237 |
| `openai/utils.py` | `build_sampling_params()`, 图片下载/上传/格式转换, `process_generation_batch()` | ~349 |
| `openai/image_api.py` | `/v1/images/generations`, `/v1/images/edits`, `/v1/images/{id}/content` | ~357 |
| `openai/video_api.py` | 视频 CRUD + 异步任务 + 视频修复 + callback 通知 | ~733 |
| `openai/mesh_api.py` | 3D 网格 CRUD + 异步任务 | ~297 |
| `openai/common_api.py` | `/v1/models`, LoRA 管理 API | ~249 |
| `openai/storage.py` | S3 云存储上传 | ~109 |
| `openai/stores.py` | `AsyncDictStore` 内存 KV 存储 | ~49 |
| `post_training/weights_api.py` | 权重热更新 + checksum | ~63 |
