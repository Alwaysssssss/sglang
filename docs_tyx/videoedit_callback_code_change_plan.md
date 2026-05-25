# VideoEdit 回调函数修改方案

本文给出为 `/v1/videos/repairs` 增加 HTTP 回调的代码修改计划。目标是在 VideoEdit 任务完成或失败后，SGLang 主动向请求方提供的 `callback_url` 发送一次结果通知，避免调用方只能轮询 `/v1/videos/{id}/progress`。

## 1. 当前状态

当前协议里已经有字段：

```python
class VideoRepairRequest(BaseModel):
    ...
    callback_url: Optional[str] = None
```

位置：

```text
python/sglang/multimodal_gen/runtime/entrypoints/openai/protocol.py
```

但当前代码没有任何地方使用 `callback_url`。全局搜索只有协议定义，没有实际 HTTP POST。

当前任务状态更新位置在：

```text
python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py
```

关键流程：

```python
async def _dispatch_job_async(...):
    try:
        save_file_path_list, result = await process_generation_batch(...)
        ...
        await VIDEO_STORE.update_fields(job_id, update_fields)
    except Exception as e:
        await VIDEO_STORE.update_fields(
            job_id, {"status": "failed", "error": {"message": str(e)}}
        )
```

现在调用方只能轮询：

```bash
curl http://127.0.0.1:30000/v1/videos/<task_id>/progress
```

## 2. 目标行为

请求方提交任务时带：

```json
{
  "task_id": "cloud_sp1_offload_81f",
  "callback_url": "http://callback-server.local/videoedit/callback",
  "video_url": "...",
  "mask_url": "...",
  "num_frames": 81
}
```

任务完成后，SGLang 自动 POST：

```http
POST http://callback-server.local/videoedit/callback
Content-Type: application/json
```

payload 示例：

```json
{
  "id": "cloud_sp1_offload_81f",
  "object": "video",
  "status": "completed",
  "progress": 100,
  "created_at": 1779080000,
  "completed_at": 1779080415,
  "file_path": null,
  "url": "http://127.0.0.1:19000/flowcut/cloud_sp1_offload_81f.mp4",
  "error": null
}
```

任务失败后，SGLang 自动 POST：

```json
{
  "id": "cloud_sp1_offload_81f",
  "object": "video",
  "status": "failed",
  "progress": 1,
  "created_at": 1779080000,
  "completed_at": null,
  "file_path": null,
  "url": null,
  "error": {
    "message": "Upload failed ..."
  }
}
```

推荐语义：

- 回调在任务最终状态后触发：`completed` 或 `failed`。
- 回调失败不应该把已经完成的推理任务改成失败；只记录日志和可选的 callback 状态。
- 回调应该有 timeout，不能长期阻塞清理逻辑。
- 第一版可以做固定次数重试，默认 3 次。

## 3. 推荐设计

### 3.1 保存 callback_url

创建 job 时把 `callback_url` 存进 `VIDEO_STORE`，便于后续回调和查询：

```python
def _video_repair_job_from_sampling(
    request_id: str, req: VideoRepairRequest, sampling: SamplingParams
) -> Dict[str, Any]:
    return {
        "id": request_id,
        "object": "video",
        "model": req.model or "videoedit",
        "status": "queued",
        "progress": 0,
        "created_at": int(time.time()),
        "size": "",
        "seconds": "",
        "quality": "standard",
        "file_path": os.path.abspath(sampling.output_file_path()),
        "callback_url": req.callback_url,
        "callback_status": None,
        "callback_error": None,
    }
```

如果不希望 `/progress` 暴露 `callback_url`，可以只存内部字段；但当前 `VIDEO_STORE` 是内存 store，简单实现可以先保存。

### 3.2 增加回调发送 helper

在 `video_api.py` 中增加：

```python
async def _post_video_callback(
    job_id: str,
    callback_url: str | None,
    payload: dict[str, Any],
    *,
    timeout: float = 10.0,
    max_retries: int = 3,
) -> None:
    if not callback_url:
        return

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
                response = await client.post(callback_url, json=payload)
                response.raise_for_status()
            await VIDEO_STORE.update_fields(
                job_id,
                {
                    "callback_status": "succeeded",
                    "callback_error": None,
                    "callback_attempts": attempt,
                    "callback_completed_at": int(time.time()),
                },
            )
            return
        except Exception as e:
            last_error = str(e)
            logger.warning(
                "Video callback failed for job=%s attempt=%s/%s url=%s: %s",
                job_id,
                attempt,
                max_retries,
                callback_url,
                last_error,
            )
            if attempt < max_retries:
                await asyncio.sleep(min(2 ** (attempt - 1), 5))

    await VIDEO_STORE.update_fields(
        job_id,
        {
            "callback_status": "failed",
            "callback_error": last_error,
            "callback_attempts": max_retries,
            "callback_completed_at": int(time.time()),
        },
    )
```

注意：`video_api.py` 已经 import 了 `asyncio`、`time`、`httpx`，所以不需要新增这些依赖。

### 3.3 构造回调 payload

增加 helper，尽量复用 `/progress` 的输出字段，同时保留完整 job 信息：

```python
def _build_video_callback_payload(video_id: str, job: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": video_id,
        "object": job.get("object", "video"),
        "model": job.get("model"),
        "status": job.get("status"),
        "progress": job.get("progress", 0),
        "created_at": job.get("created_at"),
        "completed_at": job.get("completed_at"),
        "file_path": job.get("file_path"),
        "url": job.get("url"),
        "error": job.get("error"),
    }
```

如果后续需要带性能信息，可以把 `add_common_data_to_response()` 加进去的字段也透传：

```python
for key in ("timings", "metrics", "usage"):
    if key in job:
        payload[key] = job[key]
```

具体字段以当前 `add_common_data_to_response()` 实际输出为准。

### 3.4 在成功路径触发回调

修改 `_dispatch_job_async()`，增加参数：

```python
async def _dispatch_job_async(
    job_id: str,
    batch: Req,
    *,
    temp_dirs: list[str] | None = None,
    output_persistent: bool = True,
    callback_url: str | None = None,
) -> None:
```

成功更新 store 后调度回调。这里用 `asyncio.create_task()`，避免 callback server 慢或不可达时阻塞 VideoEdit 队列释放和临时目录清理：

```python
await VIDEO_STORE.update_fields(job_id, update_fields)

job = await VIDEO_STORE.get(job_id)
if job and callback_url:
    asyncio.create_task(
        _post_video_callback(
            job_id,
            callback_url,
            _build_video_callback_payload(job_id, job),
        )
    )
```

### 3.5 在失败路径触发回调

失败时先更新 store，再回调：

```python
except Exception as e:
    logger.error(f"{e}")
    failed_fields = {
        "status": "failed",
        "error": {"message": str(e)},
    }
    await VIDEO_STORE.update_fields(job_id, failed_fields)

    job = await VIDEO_STORE.get(job_id)
    if job and callback_url:
        asyncio.create_task(
            _post_video_callback(
                job_id,
                callback_url,
                _build_video_callback_payload(job_id, job),
            )
        )
```

这里不要让 `_post_video_callback()` 的异常继续往外抛。上面的 helper 已经内部 catch 并记录状态。

### 3.6 从 repair dispatch 透传 callback_url

修改 `_dispatch_video_repair_job_async()`：

```python
async def _dispatch_video_repair_job_async(
    job_id: str,
    batch: Req,
    *,
    temp_dirs: list[str] | None = None,
    output_persistent: bool = True,
    callback_url: str | None = None,
) -> None:
    try:
        await VIDEO_STORE.update_fields(job_id, {"status": "running", "progress": 1})
        await _dispatch_job_async(
            job_id,
            batch,
            temp_dirs=None,
            output_persistent=output_persistent,
            callback_url=callback_url,
        )
    finally:
        _VIDEOEDIT_SEMAPHORE.release()
        for td in temp_dirs or []:
            shutil.rmtree(td, ignore_errors=True)
```

在 `create_video_repair()` 创建后台任务时传入：

```python
asyncio.create_task(
    _dispatch_video_repair_job_async(
        request_id,
        batch,
        temp_dirs=temp_dirs or None,
        output_persistent=output_persistent,
        callback_url=req.callback_url,
    )
)
```

## 4. 是否要支持普通 `/v1/videos` 生成接口

当前 `callback_url` 只存在于 `VideoRepairRequest`，不在 `VideoGenerationsRequest`。

第一阶段建议只支持：

```text
POST /v1/videos/repairs
```

如果后续普通视频生成也需要回调，再给 `VideoGenerationsRequest` 增加：

```python
callback_url: Optional[str] = None
```

并在 `create_video()` 的 `asyncio.create_task(_dispatch_job_async(...))` 中透传。

## 5. 使用示例

请求方启动一个 callback server，例如：

```python
from fastapi import FastAPI, Request
import uvicorn

app = FastAPI()

@app.post("/videoedit/callback")
async def videoedit_callback(request: Request):
    payload = await request.json()
    print("callback payload:", payload)
    return {"ok": True}

uvicorn.run(app, host="0.0.0.0", port=18080)
```

提交 VideoEdit repair 请求：

```python
import json
import os
import urllib.request

payload = {
    "task_id": "cloud_sp1_offload_81f",
    "prompt": "A close-up of an orange flower with a yellow center, remaining in focus against a blurred green grass background throughout the video.",
    "video_url": "http://127.0.0.1:19000/flowcut/test/video/15108907_3840_2160_50fps_short.mp4",
    "mask_url": "http://127.0.0.1:19000/flowcut/test/mask/15108907_3840_2160_50fps_No_bbox_mask.mp4",
    "callback_url": "http://127.0.0.1:18080/videoedit/callback",
    "num_frames": 81,
    "infer_len": 81,
    "overlap": 0,
    "num_inference_steps": 20,
    "guidance_scale": 5.0,
    "dynamic_cfg": True,
    "dynamic_cfg_max_step": 15,
    "seed": 42,
    "dtype": "bf16",
    "enable_paste_back": True,
    "drop_reference_frame": True,
}

req = urllib.request.Request(
    "http://127.0.0.1:30000/v1/videos/repairs",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=60) as resp:
    print(resp.status)
    print(resp.read().decode("utf-8"))
```

任务完成后，callback server 应收到类似：

```json
{
  "id": "cloud_sp1_offload_81f",
  "object": "video",
  "model": "videoedit",
  "status": "completed",
  "progress": 100,
  "created_at": 1779080000,
  "completed_at": 1779080415,
  "file_path": null,
  "url": "http://127.0.0.1:19000/flowcut/cloud_sp1_offload_81f.mp4",
  "error": null
}
```

如果没有收到回调，仍可轮询：

```bash
curl --noproxy '*' -s http://127.0.0.1:30000/v1/videos/cloud_sp1_offload_81f/progress
```

## 6. 回调失败时的语义

推荐第一版语义：

- 任务推理成功、上传成功，但回调失败：任务仍为 `completed`。
- 回调失败信息写入 `VIDEO_STORE`：

```json
{
  "callback_status": "failed",
  "callback_error": "...",
  "callback_attempts": 3
}
```

- 不建议因为 callback server 临时不可用而把推理任务标成 `failed`，否则会混淆“视频生成失败”和“通知失败”。

如果业务要求“回调必须成功才算任务成功”，可以增加请求字段：

```json
{
  "callback_required": true
}
```

但当前协议没有这个字段，第一版不建议加。

## 7. 安全和稳定性注意事项

回调 URL 是外部输入，存在 SSRF 风险。内部测试阶段可以先接受任意 HTTP URL；如果要进生产，建议至少加以下限制：

- 只允许 `http` / `https`。
- 可选：禁止访问 link-local、metadata IP、内网保留地址，除非明确配置 allowlist。
- 增加环境变量 allowlist，例如：

```bash
SGLANG_VIDEO_CALLBACK_ALLOWED_HOSTS=callback.example.com,10.0.0.5
```

- 设置 timeout，例如 10 秒。
- 设置最大重试次数，例如 3 次。
- 不在回调 payload 里放敏感凭据。

## 8. 测试计划

### 8.1 单元测试

建议新增或扩展测试：

```text
python/sglang/multimodal_gen/test/unit/test_video_callback.py
```

测试点：

- `_build_video_callback_payload()` 对 completed job 输出正确字段。
- `_post_video_callback()` 成功时写入 `callback_status=succeeded`。
- `_post_video_callback()` 多次失败后写入 `callback_status=failed`。
- `_dispatch_job_async()` 成功路径会调用 `_post_video_callback()`。
- `_dispatch_job_async()` 失败路径也会调用 `_post_video_callback()`。

可以用 `pytest` + `monkeypatch` mock `httpx.AsyncClient.post`，不需要真实启动 HTTP server。

### 8.2 手工测试

启动 callback server：

```bash
python /tmp/callback_server.py
```

提交带 `callback_url` 的 81 帧请求。

验证：

```bash
curl --noproxy '*' -s http://127.0.0.1:30000/v1/videos/cloud_sp1_offload_81f/progress
```

预期：

- `/progress` 是 `completed`。
- callback server 打印一次 completed payload。
- 如果启用了 S3 上传，payload 里的 `url` 是云端对象 URL。
- 如果故意把 `callback_url` 写错，任务仍然 `completed`，但日志里有 callback failed，store 里有 `callback_status=failed`。

## 9. 和 S3 指定输出路径的关系

回调功能和 S3 指定输出路径是两个独立能力：

- S3 指定输出路径解决“输出视频上传到哪个 bucket/key”。
- callback 解决“任务完成后如何通知调用方”。

两者组合后的理想 payload 是：

```json
{
  "id": "cloud_sp1_offload_81f",
  "status": "completed",
  "url": "http://127.0.0.1:19000/flowcut/test/output/cloud_sp1_offload_81f.mp4",
  "file_path": null,
  "error": null
}
```

调用方收到 callback 后，直接读取 `url` 或用 `mc cp` / S3 SDK 下载该对象即可。
