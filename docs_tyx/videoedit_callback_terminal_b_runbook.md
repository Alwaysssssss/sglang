# VideoEdit 回调监听使用流程

本文说明启用 VideoEdit 回调后，如何用“终端 B”监听任务完成/失败结果。

核心结论：

- 终端 A 用 `curl` 或 Python 提交 `/v1/videos/repairs` 请求后，请求会很快返回 `queued`，随后命令结束。
- 任务真正完成或失败时，不会自动回到终端 A 打印。
- 需要提前在终端 B 启动一个 HTTP callback server。
- 请求体里传 `callback_url`，SGLang 在任务完成或失败后会 POST 到这个 URL。
- 终端 B 会打印回调 payload。

## 1. 终端 B 启动 callback server

在终端 B 执行：

```bash
cd /home/tyx/workspace/zhouhao6/sglang
source .venv/bin/activate

cat >/tmp/videoedit_callback_server.py <<'PY'
from fastapi import FastAPI, Request
import json
import time
import uvicorn

app = FastAPI()


@app.post("/videoedit/callback")
async def videoedit_callback(request: Request):
    payload = await request.json()
    print("\n=== VideoEdit callback received ===", flush=True)
    print("time:", time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    return {"ok": True}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=18080)
PY

python /tmp/videoedit_callback_server.py
```

看到类似输出说明监听已启动：

```text
Uvicorn running on http://0.0.0.0:18080
```

这个终端要一直开着，直到任务完成。它不是普通空终端，而是在运行一个 HTTP 服务。

如果 `fastapi` 或 `uvicorn` 不存在，先安装：

```bash
pip install fastapi uvicorn
```

## 2. callback_url 怎么填

如果 callback server 和 SGLang serve 在同一台机器、同一网络命名空间内运行，可以用：

```text
http://127.0.0.1:18080/videoedit/callback
```

如果 callback server 在另一台机器上，例如 Mac，需要使用 SGLang serve 机器能访问到的 IP：

```text
http://MAC_LAN_IP:18080/videoedit/callback
```

判断原则：`callback_url` 必须从运行 `sglang serve` 的进程所在环境能访问。

提交长任务前，可以在 serve 所在机器上先测：

```bash
curl --noproxy '*' -s -X POST http://127.0.0.1:18080/videoedit/callback \
  -H 'Content-Type: application/json' \
  -d '{"id":"probe","status":"ok"}'
```

终端 B 应该打印这条 probe。

## 3. 终端 A 提交 VideoEdit 请求

终端 A 执行：

```bash
cd /home/tyx/workspace/zhouhao6/sglang
source .venv/bin/activate

export VIDEO_URL='http://127.0.0.1:19000/flowcut/test/video/15108907_3840_2160_50fps_short.mp4'
export MASK_URL='http://127.0.0.1:19000/flowcut/test/mask/15108907_3840_2160_50fps_No_bbox_mask.mp4'
export CALLBACK_URL='http://127.0.0.1:18080/videoedit/callback'

python - <<'PY'
import json
import os
import urllib.request

task_id = "cloud_sp1_offload_81f_callback"
payload = {
    "task_id": task_id,
    "prompt": "A close-up of an orange flower with a yellow center, remaining in focus against a blurred green grass background throughout the video.",
    "video_url": os.environ["VIDEO_URL"],
    "mask_url": os.environ["MASK_URL"],
    "callback_url": os.environ["CALLBACK_URL"],
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
    "perf_dump_path": "/tmp/videoedit_perf_api_cloud_sp1_offload_81f_callback.json",
}

body = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(
    "http://127.0.0.1:30000/v1/videos/repairs",
    data=body,
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=60) as resp:
    print(resp.status)
    print(resp.read().decode("utf-8"))
PY
```

终端 A 会很快返回类似：

```json
{
  "id": "cloud_sp1_offload_81f_callback",
  "status": "queued",
  "file_path": "/tmp/sglang_videoedit_output_xxx/cloud_sp1_offload_81f_callback.mp4"
}
```

之后终端 A 的请求已经结束。你可以在终端 A 干别的事情。

## 4. 任务完成后终端 B 会看到什么

成功时，终端 B 会打印类似：

```json
{
  "id": "cloud_sp1_offload_81f_callback",
  "object": "video",
  "model": "videoedit",
  "status": "completed",
  "progress": 100,
  "created_at": 1779080000,
  "completed_at": 1779080415,
  "file_path": null,
  "url": "http://127.0.0.1:19000/flowcut/cloud_sp1_offload_81f_callback.mp4",
  "error": null,
  "peak_memory_mb": 12345.0,
  "inference_time_s": 415.27
}
```

失败时，终端 B 会打印类似：

```json
{
  "id": "cloud_sp1_offload_81f_callback",
  "object": "video",
  "model": "videoedit",
  "status": "failed",
  "progress": 1,
  "created_at": 1779080000,
  "completed_at": null,
  "file_path": null,
  "url": null,
  "error": {
    "message": "..."
  }
}
```

## 5. 仍然可以轮询进度

回调不是替代轮询，而是让服务端主动通知你。你仍然可以随时查：

```bash
curl --noproxy '*' -s \
  http://127.0.0.1:30000/v1/videos/cloud_sp1_offload_81f_callback/progress
```

回调成功后会看到：

```json
{
  "id": "cloud_sp1_offload_81f_callback",
  "status": "completed",
  "progress": 100,
  "file_path": null,
  "url": "http://127.0.0.1:19000/flowcut/cloud_sp1_offload_81f_callback.mp4",
  "error": null,
  "callback_status": "succeeded",
  "callback_error": null,
  "callback_attempts": 1
}
```

如果 callback server 不可达，任务本身仍可能是 `completed`，但回调状态会是：

```json
{
  "callback_status": "failed",
  "callback_error": "..."
}
```

这是故意设计的：回调失败不等于推理失败。

## 6. 常见问题

### 为什么终端 A 不会提示

终端 A 的请求是异步提交任务。它拿到 `queued` 后命令就结束了，和 SGLang 不再保持连接。因此任务完成后无法自动回到终端 A 打印。

如果希望同一个终端最终打印 completed/failed，不应该用 callback，而应该写一个“提交后自动轮询”的脚本。

### 为什么终端 B 没有输出

按顺序检查：

- 终端 B 是否仍在运行 callback server。
- 请求体里是否传了 `callback_url`。
- `callback_url` 是否是 SGLang serve 进程能访问的地址。
- serve 日志里是否有 `Video callback failed`。
- `/progress` 里的 `callback_status` 是否是 `failed`。

### callback_url 用 127.0.0.1 是否正确

只有当 callback server 和 SGLang serve 在同一台机器、同一网络命名空间中运行时才正确。

如果 callback server 在 Mac，而 SGLang serve 在 GPU 机器上，`127.0.0.1` 会指向 GPU 机器自己，不是 Mac。此时要用 Mac 的局域网 IP 或可达域名。

### 回调会发几次

当前实现：最终状态触发一次回调。若回调请求失败，会最多重试 3 次。成功后不会继续发送。

### 支持普通 `/v1/videos` 吗

当前实现只接到 `/v1/videos/repairs`，因为 `callback_url` 字段目前只存在于 `VideoRepairRequest`。普通 `/v1/videos` 生成接口如果也需要回调，需要额外给 `VideoGenerationsRequest` 增加字段并透传。
