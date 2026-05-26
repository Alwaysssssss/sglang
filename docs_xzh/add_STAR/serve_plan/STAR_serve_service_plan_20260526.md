# STAR 在 SGLang 中增加 Serve 服务实施文档

## 1. 目标

当前 `STAR` 已经作为一条本地 `sglang` diffusion pipeline 跑通。  
新的目标是在**不破坏现有 STAR 推理与加速主线**的前提下，进一步接入 `sglang` 的 HTTP serve 能力，使用户可以通过 `curl` 发请求并完成 STAR 视频超分推理。

这份文档只用于指导后续代码修改，重点回答：

1. `sglang` 现有服务是怎么拉起来的
2. 当前 STAR 离 serve 化还缺什么
3. 推荐以什么方式把 STAR 接入 HTTP API
4. 后续代码应该改哪些文件
5. 如何验收

---

## 2. 当前服务链路概览

`sglang` 现有的 diffusion serve 能力已经具备完整的“HTTP -> scheduler -> worker -> pipeline -> 输出文件”链路。

主链路如下：

1. 服务入口：
   [launch_server.py](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/launch_server.py:87)
2. HTTP app 创建：
   [http_server.py create_app](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/entrypoints/http_server.py:298)
3. 视频 API：
   [video_api.py](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py:1)
4. 请求转成 `SamplingParams`：
   [openai/utils.py build_sampling_params](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/entrypoints/openai/utils.py:82)
5. 请求包装成 `Req`：
   [entrypoints/utils.py prepare_request](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/entrypoints/utils.py:242)
6. 调度到 scheduler：
   [video_api.py _dispatch_job_async](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py:118)
7. rank0 scheduler 分发：
   [scheduler.py _handle_generation](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/managers/scheduler.py:188)
8. worker 执行 pipeline：
   [gpu_worker.py execute_forward](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/managers/gpu_worker.py:205)
9. 结果写盘：
   [entrypoints/utils.py save_outputs](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/entrypoints/utils.py:322)

换句话说，**服务框架已有，当前问题不是“怎么起 HTTP 服务”，而是“怎么让 HTTP 请求正确表达 STAR 的输入语义”**。

---

## 3. 当前服务拉起逻辑分析

## 3.1 服务启动入口

标准启动逻辑在：

- [launch_server.py dispatch_launch](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/launch_server.py:684)

如果是普通 monolithic 模式，会走：

- [launch_server](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/launch_server.py:87)

它做的事情是：

1. 创建多进程 worker
2. 启动 rank0 scheduler
3. 等 worker ready
4. 如果 `launch_http_server=True`，再启动 FastAPI

HTTP server 实际是通过：

- [launch_http_server_only](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/launch_server.py:465)

调用：

- `app = create_app(server_args)`
- `uvicorn.run(...)`

这意味着 STAR serve 化本质上不需要新建一套 server 框架，只要复用这条标准路径。

## 3.2 FastAPI app 结构

HTTP app 定义在：

- [http_server.py](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/entrypoints/http_server.py:1)

`create_app()` 当前会挂这些 router：

1. `health_router`
2. `vertex_router`
3. `common_api.router`
4. `image_api.router`
5. `video_api.router`
6. `mesh_api.router`
7. `weights_api.router`
8. `rollout_api.router`

对应代码：

- [http_server.py create_app](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/entrypoints/http_server.py:298)

当前 STAR 最直接相关的是：

- [video_api.router prefix=/v1/videos](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py:40)

## 3.3 HTTP 请求如何进入调度器

视频请求进入 `create_video()` 后，会：

1. 解析 JSON 或 multipart/form-data
2. 构造 `VideoGenerationsRequest`
3. 调用 `_build_video_sampling_params()`
4. 再经由 `prepare_request()` 转成 `Req`
5. 用 `async_scheduler_client.forward([batch])` 发给 scheduler

对应代码：

- [video_api.py create_video](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py:155)
- [video_api.py _build_video_sampling_params](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py:44)
- [openai/utils.py process_generation_batch](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/entrypoints/openai/utils.py:284)

---

## 4. 当前 STAR 与现有视频服务的主要不匹配点

这是后续实施中最重要的分析。

## 4.1 当前 `/v1/videos` 的输入语义偏向 T2V / I2V

现有视频协议在：

- [protocol.py VideoGenerationsRequest](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/entrypoints/openai/protocol.py:79)

主要字段是：

1. `prompt`
2. `input_reference`
3. `reference_url`
4. `seconds / fps / num_frames`
5. `guidance_scale`
6. `num_inference_steps`

它更像通用：

1. 文生视频
2. 图生视频

但 STAR 需要的核心输入其实是：

1. `condition_video_path`
2. 或一个上传的低清 `mp4`
3. 可选 `condition_video_start_frame`
4. 可选 `condition_video_num_frames`
5. 可选 `condition_video_sample_fps`
6. 可选 `condition_video_frame_stride`

而这些字段虽然已经存在于 STAR 的 sampling params 中：

- [star_cogvideox_sr.py](/sgl-workspace/sglang/python/sglang/multimodal_gen/configs/sample/star_cogvideox_sr.py:1)

但**还没有被现有视频 HTTP 协议原生暴露出来**。

## 4.2 现有 multipart 路径默认按“图片输入”处理

`video_api.create_video()` 在 multipart 模式下会把输入走：

- [_save_first_input_image](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py:96)
- [save_image_to_path](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/entrypoints/openai/utils.py:129)

这里逻辑默认是：

1. 接收图片上传
2. 保存图片
3. 填入 `input_reference`

这对 STAR 是不够的，因为 STAR 需要的是**视频上传**，不是图片上传。

## 4.3 当前 `task_type` 也不完全表达 STAR 的“视频超分”形态

当前 STAR pipeline config：

- [StarCogVideoXSRPipelineConfig.task_type](/sgl-workspace/sglang/python/sglang/multimodal_gen/configs/pipeline_configs/star_cogvideox_sr.py:43)

仍然设为：

- `ModelTaskType.T2V`

这意味着通用 video API 不会自动要求 `condition_video_path` 这类 STAR SR 必需输入。

所以如果直接把 STAR 暴露到当前 `/v1/videos`，会有两个风险：

1. 请求协议不完整
2. 缺少 STAR 专属输入校验

---

## 5. 推荐实施方向

## 5.1 推荐新增 STAR 专用 serve 路由，而不是直接硬改通用 `/v1/videos`

推荐方案：

1. 保留通用 `/v1/videos`
2. 新增 STAR 专用路由，例如：
   - `POST /v1/star/videos`
   - 或 `POST /v1/videos/star-sr`

原因：

1. STAR 是“带 condition video 的视频超分”，输入语义明显不同于通用 T2V/I2V
2. 直接污染通用协议，会把所有视频模型都带上 STAR 特有字段
3. 单独路由更容易：
   - 做请求校验
   - 做 curl 示例
   - 做后续扩展

## 5.2 复用现有异步 job / scheduler / 存储框架

虽然建议新增 STAR 专用 router，但下面这些层都应该复用现有实现：

1. `FastAPI` app 挂载机制
2. `async_scheduler_client`
3. `prepare_request()`
4. `Req`
5. `VIDEO_STORE`
6. `_dispatch_job_async()`
7. `process_generation_batch()`
8. `VideoResponse / VideoListResponse`

也就是说：

1. **新的是 STAR 的输入协议和请求构造**
2. **复用的是整个服务与调度基础设施**

这能最大程度降低实现风险。

---

## 6. 推荐的目标接口设计

建议定义一套 STAR 专用请求模型。

### 6.1 JSON 请求模型

建议最少支持：

```json
{
  "prompt": "A serene scene ...",
  "condition_video_path": "/abs/path/to/lq.mp4",
  "seed": 1234,
  "width": 720,
  "height": 480,
  "fps": 8,
  "num_frames": 7,
  "condition_video_num_frames": 25,
  "num_inference_steps": 50,
  "guidance_scale": 6.0,
  "negative_prompt": "",
  "output_quality": "maximum"
}
```

### 6.2 multipart/form-data 请求模型

建议同时支持：

1. `prompt`
2. `condition_video` 上传文件
3. 其余数字参数用 form 字段传入

对应 `curl` 形态应该类似：

```bash
curl -X POST http://127.0.0.1:30000/v1/star/videos \
  -F 'prompt=A serene scene ...' \
  -F 'condition_video=@/path/to/023_klingai_reedit.mp4' \
  -F 'seed=1234' \
  -F 'width=720' \
  -F 'height=480' \
  -F 'fps=8' \
  -F 'num_frames=7' \
  -F 'condition_video_num_frames=25' \
  -F 'num_inference_steps=50' \
  -F 'guidance_scale=6.0' \
  -F 'output_quality=maximum'
```

### 6.3 为什么必须保留 `output_quality=maximum`

当前 STAR 验收已明确依赖统一视频编码质量。

如果不显式传：

```text
output_quality = maximum
```

则视频会按默认较低质量写盘，`mp4` 级别的 `SSIM` 会下降。

所以 STAR serve 默认值建议直接设为：

1. `output_quality = "maximum"`
2. `negative_prompt = ""`
3. `fps = 8`
4. `num_frames = 7`
5. `condition_video_num_frames = 25`

---

## 7. 推荐修改点

下面按代码文件给出建议。

## 7.1 协议层

建议在：

- [protocol.py](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/entrypoints/openai/protocol.py:1)

新增 STAR 专用请求模型，例如：

1. `StarVideoSRRequest`

建议字段：

1. `prompt`
2. `condition_video_path`
3. `seed`
4. `width`
5. `height`
6. `fps`
7. `num_frames`
8. `condition_video_start_frame`
9. `condition_video_num_frames`
10. `condition_video_sample_fps`
11. `condition_video_frame_stride`
12. `num_inference_steps`
13. `guidance_scale`
14. `negative_prompt`
15. `output_quality`
16. `output_compression`
17. `output_path`

如果想减少协议扩散，也可以不新增 Pydantic class，而是在 STAR 路由文件里本地定义。但从后续维护看，放进 `protocol.py` 更统一。

## 7.2 新增 STAR 专用 router

建议新增文件：

- `python/sglang/multimodal_gen/runtime/entrypoints/openai/star_video_api.py`

职责建议：

1. 定义 STAR 专用 `APIRouter`
2. 提供 `POST /v1/star/videos`
3. 同时支持 JSON 和 multipart/form-data
4. 负责：
   - 保存上传的视频文件
   - 构造 STAR 专用 `SamplingParams`
   - 生成 `Req`
   - 创建异步任务
   - 返回 `VideoResponse`

这里强烈建议直接复用这些现有函数：

1. `_video_job_from_sampling`
2. `_dispatch_job_async`
3. `process_generation_batch`
4. `prepare_request`
5. `VideoResponse`
6. `VIDEO_STORE`

## 7.3 新增视频保存辅助函数

当前 `openai/utils.py` 中只有图片保存辅助：

- [save_image_to_path](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/entrypoints/openai/utils.py:129)

建议新增一个并列辅助：

1. `save_video_to_path(...)`

支持：

1. `UploadFile`
2. 本地绝对路径
3. `http(s)` URL

这一步要避免把“图片辅助函数”硬改成既存图片又存视频，否则通用路径容易混乱。

## 7.4 SamplingParams 构造

建议在新 STAR router 内新增一个类似：

1. `_build_star_sampling_params()`

它应当复用：

- [build_sampling_params](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/entrypoints/openai/utils.py:82)

但要显式补充 STAR 所需字段：

1. `condition_video_path`
2. `condition_video_start_frame`
3. `condition_video_num_frames`
4. `condition_video_sample_fps`
5. `condition_video_frame_stride`
6. `output_quality = maximum` 默认值

重要点：

当前 `StarCogVideoXSRSamplingParams` 已经有这些字段：

- [star_cogvideox_sr.py](/sgl-workspace/sglang/python/sglang/multimodal_gen/configs/sample/star_cogvideox_sr.py:1)

所以后端 pipeline 本身并不缺字段，缺的是 HTTP 层把这些字段正确构造进来。

## 7.5 HTTP app 挂载

在：

- [http_server.py create_app](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/entrypoints/http_server.py:298)

里把新 router include 进去，例如：

1. `app.include_router(star_video_api.router)`

建议放在 `video_api.router` 之后或相邻位置，便于维护。

## 7.6 文档与启动命令

后续代码改完后，还需要补一份面向使用者的文档，至少包括：

1. serve 启动命令
2. JSON 请求示例
3. multipart `curl` 示例
4. 输出查询示例

推荐同时更新：

1. `infer_STAR.md`
2. 新增 `serve_STAR.md`

---

## 8. 推荐启动方式

由于服务逻辑已经是标准 `launch_server -> FastAPI`，STAR 不需要新写启动器。

推荐直接用现有 server 启动方式，只是明确 STAR pipeline 参数。

推荐形态：

```bash
python -m sglang.multimodal_gen.runtime.launch_server \
  --model-path /sgl-workspace/sglang/model_artifacts/sglang_star_cogvideox_sr \
  --pipeline-class-name StarCogVideoXSRPipeline \
  --attention-backend fa \
  --num-gpus 1 \
  --enable-torch-compile \
  --dit-cpu-offload \
  --text-encoder-cpu-offload \
  --host 0.0.0.0 \
  --port 30000
```

这个命令的含义是：

1. 启动 rank0 scheduler + worker
2. 启动 FastAPI
3. 对外暴露 HTTP 接口

后续 STAR serve 化主要改的是“HTTP 请求如何变成 STAR 的 `SamplingParams`”，不是启动器本身。

---

## 9. 验收建议

后续代码修改完成后，建议按三层验收。

### 9.1 服务可用性验收

1. `GET /health` 返回正常
2. `GET /model_info` 返回 diffusion worker 信息
3. `POST /v1/star/videos` 能返回 `VideoResponse`
4. `GET /v1/star/videos` 或复用 `GET /v1/videos/{id}` 能查询任务状态

### 9.2 结果正确性验收

使用当前固定 STAR case：

1. prompt: `023_klingai_reedit.txt`
2. condition video: `023_klingai_reedit.mp4`
3. seed: `1234`
4. `fps = 8`
5. `num_frames = 7`
6. `condition_video_num_frames = 25`

要求：

1. serve 请求产物与离线命令产物的 `raw / mp4` parity 不应新增额外退化
2. 至少保持当前 `strict 0.95` release gate

### 9.3 服务路径与离线路径一致性验收

需要确认：

1. serve 路径最终进入的 pipeline 与离线路径一致
2. serve 路径使用的 sampling params 与离线路径一致
3. `output_quality` 默认值一致
4. `condition_video_path` 经过保存/下载后，实际进入 STAR 的仍是同一文件内容

---

## 10. 实施顺序建议

推荐按以下顺序落地：

1. 新增 `save_video_to_path()` 辅助
2. 新增 `StarVideoSRRequest`
3. 新增 `star_video_api.py`
4. 在 `create_app()` 中挂载 router
5. 写 `curl` 示例并本地打通
6. 用固定 case 做 parity 验收

不建议一开始就去改：

1. STAR pipeline 本体
2. scheduler / worker
3. 现有通用 `/v1/videos` 行为

优先做“新增 STAR 专用 serve 接口”，风险最低，也最容易回滚。

---

## 11. 一句话结论

当前 `sglang` 的 serve 基础设施已经具备，STAR serve 化的关键工作不是“重写服务框架”，而是：

**为 STAR 增加一层能正确表达 `condition_video_path` / 视频上传 / STAR 默认参数 的 HTTP 协议与 router，然后复用现有 scheduler 与 pipeline 执行链。**
