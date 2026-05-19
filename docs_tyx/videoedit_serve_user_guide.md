# VideoEdit Serve 服务启动与请求操作文档

本文面向实际使用者，说明如何启动 VideoEdit 的 `sglang serve` 服务、如何启动回调监听、如何发送本地视频请求和云端视频 URL 请求，以及如何验证 serve 端口、callback 端口、本地输出、云端视频连通性和云端输出上传结果。

本文中所有本地路径都使用 `/path/to/...` 占位。实际执行时，请把这些路径替换成你机器上的真实目录或文件路径。

本文案例沿用一直使用的花朵视频样例：

- 输入视频文件：`15108907_3840_2160_50fps_short.mp4`
- 输入 mask 文件：`15108907_3840_2160_50fps_No_bbox_mask.mp4`
- prompt：`A close-up of an orange flower with a yellow center, remaining in focus against a blurred green grass background throughout the video.`
- 推理帧数：`num_frames=81`
- 推理窗口：`infer_len=81`
- 采样步数：`num_inference_steps=20`
- 随机种子：`seed=42`

本文区分两种输出方式：

- 本地视频推理：输入来自本地 `video_input_path` / `mask_input_path`，输出视频保存在本地，不上传云端。
- 云端视频 URL 推理：输入来自 `video_url` / `mask_url`，输出视频上传到 S3-compatible 云端存储，例如 MinIO。

## 1. 终端分工

建议打开 3 个终端：

| 终端 | 用途 | 是否需要保持运行 |
| --- | --- | --- |
| 终端 A | 启动 `sglang serve` | 是 |
| 终端 B | 启动 callback 监听服务 | 是，如果请求里传 `callback_url` |
| 终端 C | 发送请求、查询进度、执行验证命令 | 否 |

`/v1/videos/repairs` 是异步接口。发送请求后会很快返回 `queued`，这只表示任务已经入队，不表示推理完成。任务完成后有两种确认方式：

- 轮询 `/v1/videos/{task_id}/progress`。
- 请求体传 `callback_url`，任务完成或失败时由 serve 主动 POST 回调。

## 2. 公共变量

在每个需要执行 SGLang 命令的终端中，先进入仓库并激活环境：

```bash
cd /path/to/sglang
source .venv/bin/activate
```

设置本次样例使用的路径。请把 `/path/to/...` 替换为你的真实路径：

```bash
export SGLANG_REPO=/path/to/sglang
export MODEL_PATH=/path/to/VideoEdit-diffusers-model
export TRANSFORMER_PATH=/path/to/VideoEdit-diffusers-model/transformer

export INPUT_VIDEO=/path/to/pexel_test_data_0410/videos/15108907_3840_2160_50fps_short.mp4
export INPUT_MASK=/path/to/pexel_test_data_0410/masks/15108907_3840_2160_50fps_No_bbox_mask.mp4

export INPUT_SAVE_DIR=/path/to/sglang-videoedit-inputs
export PERF_DIR=/path/to/videoedit-perf
export CACHE_DIR=/path/to/sglang-cache
export CALLBACK_SCRIPT=/path/to/scripts/videoedit_callback_server.py

export PROMPT="A close-up of an orange flower with a yellow center, remaining in focus against a blurred green grass background throughout the video."

mkdir -p "$INPUT_SAVE_DIR" "$PERF_DIR" "$CACHE_DIR" "$(dirname "$CALLBACK_SCRIPT")"
```

确认样例文件存在：

```bash
ls -lh "$INPUT_VIDEO" "$INPUT_MASK"
```

通过标准：

- 两个文件都存在。
- 视频文件和 mask 文件大小大于 `0`。

确认 GPU 和 PyTorch 可用：

```bash
nvidia-smi
python - <<'PY'
import torch
print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
PY
```

通过标准：

- `cuda_available: True`
- `device_count` 大于等于 `1`
- `nvidia-smi` 能看到 GPU

## 3. 云端存储认证说明

云端视频 URL 推理的输出会上传到 S3-compatible 云端存储。MinIO 暴露到公网时通常会有 access key 和 secret key，它们用于两类操作：

| 场景 | 是否需要账号密码 | serve 当前如何使用 |
| --- | --- | --- |
| 上传云端任务的输出视频到 MinIO/S3 | 需要 | 通过 `SGLANG_S3_ACCESS_KEY_ID` 和 `SGLANG_S3_SECRET_ACCESS_KEY` |
| 下载公开输入 URL | 不需要 | URL 已公开，serve 直接 HTTP GET |
| 下载 presigned 输入 URL | 不需要额外传账号密码 | 权限在 URL 签名里，serve 直接 HTTP GET |
| 下载私有对象普通 URL | 需要，但当前 repair 请求体没有用户名/密码字段 | 不推荐，改用公开 URL 或 presigned URL |

重点：

- `video_url` / `mask_url` 是给 serve 下载输入文件用的。当前实现是直接 HTTP GET URL，不会读取 MinIO 用户名和密码字段。
- 如果对象不是公开下载，建议用 MinIO/S3 凭证生成 presigned URL，再把 presigned URL 填到 `video_url` / `mask_url`。
- 云端输出上传使用 S3 环境变量认证，和 `video_url` / `mask_url` 的下载权限是两条链路。
- 本地视频推理不需要设置 S3 环境变量；如果当前终端已经设置过这些变量，启动本地输出模式前需要显式 `unset`，否则服务端会尝试上传输出。

## 4. 启动 Callback 监听

### 4.1 启动 callback server

在终端 B 执行：

```bash
cd /path/to/sglang
source .venv/bin/activate

export CALLBACK_SCRIPT=/path/to/scripts/videoedit_callback_server.py
mkdir -p "$(dirname "$CALLBACK_SCRIPT")"

cat >"$CALLBACK_SCRIPT" <<'PY'
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

python "$CALLBACK_SCRIPT"
```

看到下面日志表示 callback listener 已经启动：

```text
Uvicorn running on http://0.0.0.0:18080
```

如果缺少依赖：

```bash
pip install fastapi uvicorn
```

### 4.2 验证 callback 端口

另开终端 C 执行：

```bash
ss -ltnp | grep ':18080'
```

通过标准：能看到 `LISTEN`，并且端口是 `18080`。

发送一条 probe：

```bash
export CALLBACK_URL='http://127.0.0.1:18080/videoedit/callback'

curl --noproxy '*' -sS -X POST "$CALLBACK_URL" \
  -H 'Content-Type: application/json' \
  -d '{"id":"probe","status":"ok"}'
```

通过标准：

- curl 返回 `{"ok":true}`。
- 终端 B 打印 `VideoEdit callback received`。
- payload 中能看到 `id` 为 `probe`。

如果 callback server 不在 serve 同一台机器上，`callback_url` 不能写 `127.0.0.1`，必须写 serve 进程能访问到的地址，例如：

```text
http://CALLBACK_MACHINE_IP:18080/videoedit/callback
```

## 5. 启动 Serve 服务

### 5.1 启动前检查端口

本文默认端口：

- Serve：`30000`
- Callback：`18080`

检查端口是否被占用：

```bash
ss -ltnp | grep -E ':(30000|18080) '
```

如果 `30000` 被占用，可以停止旧 serve，或者把本文所有 `30000` 改成另一个空闲端口。

### 5.2 选择启动模式

本地视频推理和云端视频推理建议分开启动 serve：

- 跑第 6 节本地视频请求时，使用“本地输出模式”。不要设置 S3 环境变量，输出保存在本地。
- 跑第 7 节云端视频 URL 请求时，使用“云端输出模式”。启动前设置 S3 环境变量，输出上传到 MinIO/S3。

两种模式不能同时占用同一个 `30000` 端口。如果需要切换模式，先在终端 A 按 `Ctrl-C` 停止当前 serve，再按另一种模式重新启动。

### 5.3 启动 serve：本地输出模式

本模式用于第 6 节。本地输入视频推理完成后，输出视频保存在 `LOCAL_OUT_DIR`，不会上传云端。

在终端 A 执行：

```bash
cd /path/to/sglang
source .venv/bin/activate

export MODEL_PATH=/path/to/VideoEdit-diffusers-model
export TRANSFORMER_PATH=/path/to/VideoEdit-diffusers-model/transformer
export INPUT_SAVE_DIR=/path/to/sglang-videoedit-inputs
export LOCAL_OUT_DIR=/path/to/videoedit-local-outputs
export CACHE_DIR=/path/to/sglang-cache

unset SGLANG_CLOUD_STORAGE_TYPE
unset SGLANG_S3_BUCKET_NAME
unset SGLANG_S3_ENDPOINT_URL
unset SGLANG_S3_REGION_NAME
unset SGLANG_S3_ACCESS_KEY_ID
unset SGLANG_S3_SECRET_ACCESS_KEY

export FLASHINFER_WORKSPACE_BASE="$CACHE_DIR/flashinfer"
export XDG_CACHE_HOME="$CACHE_DIR/xdg"
export VIDEOEDIT_QUEUE_CAPACITY=1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$INPUT_SAVE_DIR" "$LOCAL_OUT_DIR" "$FLASHINFER_WORKSPACE_BASE" "$XDG_CACHE_HOME"

sglang serve \
  --model-type diffusion \
  --model-path "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port 30000 \
  --num-gpus 1 \
  --sp-degree 1 \
  --ulysses-degree 1 \
  --ring-degree 1 \
  --dit-cpu-offload true \
  --dit-layerwise-offload true \
  --text-encoder-cpu-offload true \
  --image-encoder-cpu-offload true \
  --vae-cpu-offload true \
  --warmup true \
  --warmup-steps 1 \
  --output-path "$LOCAL_OUT_DIR" \
  --input-save-path "$INPUT_SAVE_DIR" \
  --transformer-path "$TRANSFORMER_PATH"
```

这里使用 `--output-path "$LOCAL_OUT_DIR"`，并且显式 `unset` S3 环境变量。任务完成后，`/progress` 中应看到 `file_path` 指向本地 mp4，`url` 为 `null`。

### 5.4 启动 serve：云端输出模式

本模式用于第 7 节。输入可以来自云端 URL，输出视频上传到 MinIO/S3。S3 环境变量必须在启动 serve 前设置，修改后必须重启 serve 才会生效。

先设置云端输出环境变量：

```bash
export SGLANG_CLOUD_STORAGE_TYPE=s3
export SGLANG_S3_BUCKET_NAME=your-output-bucket
export SGLANG_S3_ENDPOINT_URL='https://your-minio-or-s3-endpoint'
export SGLANG_S3_REGION_NAME=us-east-1
export SGLANG_S3_ACCESS_KEY_ID='your-access-key'
export SGLANG_S3_SECRET_ACCESS_KEY='your-secret-key'
```

如果 MinIO endpoint 是内网地址或端口转发地址，也可以写成：

```bash
export SGLANG_S3_ENDPOINT_URL='http://MINIO_HOST_OR_IP:9000'
```

验证 `boto3`：

```bash
python - <<'PY'
import boto3
print("boto3 ok:", boto3.__version__)
PY
```

如果缺少依赖：

```bash
pip install boto3
```

验证 S3/MinIO 上传权限：

```bash
python - <<'PY'
import os
import tempfile
import boto3

endpoint = os.environ["SGLANG_S3_ENDPOINT_URL"]
bucket = os.environ["SGLANG_S3_BUCKET_NAME"]
key = "sglang_s3_upload_probe.txt"

client = boto3.client(
    "s3",
    endpoint_url=endpoint,
    region_name=os.environ.get("SGLANG_S3_REGION_NAME") or "us-east-1",
    aws_access_key_id=os.environ["SGLANG_S3_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["SGLANG_S3_SECRET_ACCESS_KEY"],
)

path = tempfile.NamedTemporaryFile(delete=False, suffix=".txt").name
with open(path, "w") as f:
    f.write("sglang s3 upload probe\n")

client.upload_file(path, bucket, key, ExtraArgs={"ContentType": "text/plain"})
print("upload probe ok:", f"{endpoint.rstrip('/')}/{bucket}/{key}")
PY
```

通过标准：打印 `upload probe ok`，并且 MinIO/S3 中能看到 `sglang_s3_upload_probe.txt`。

再启动 serve：

```bash
cd /path/to/sglang
source .venv/bin/activate

export MODEL_PATH=/path/to/VideoEdit-diffusers-model
export TRANSFORMER_PATH=/path/to/VideoEdit-diffusers-model/transformer
export INPUT_SAVE_DIR=/path/to/sglang-videoedit-inputs
export CACHE_DIR=/path/to/sglang-cache

export FLASHINFER_WORKSPACE_BASE="$CACHE_DIR/flashinfer"
export XDG_CACHE_HOME="$CACHE_DIR/xdg"
export VIDEOEDIT_QUEUE_CAPACITY=1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$INPUT_SAVE_DIR" "$FLASHINFER_WORKSPACE_BASE" "$XDG_CACHE_HOME"

sglang serve \
  --model-type diffusion \
  --model-path "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port 30000 \
  --num-gpus 1 \
  --sp-degree 1 \
  --ulysses-degree 1 \
  --ring-degree 1 \
  --dit-cpu-offload true \
  --dit-layerwise-offload true \
  --text-encoder-cpu-offload true \
  --image-encoder-cpu-offload true \
  --vae-cpu-offload true \
  --warmup true \
  --warmup-steps 1 \
  --output-path "" \
  --input-save-path "$INPUT_SAVE_DIR" \
  --transformer-path "$TRANSFORMER_PATH"
```

这里使用 `--output-path ""`，表示本地只使用临时目录中转输出视频。云端上传成功后，本地临时视频会被清理，最终结果以 `/progress` 里的 `url` 为准。

启动成功后，终端 A 日志中应看到类似：

```text
Using pipeline from model_index.json: WanVideoEditPipeline
Worker 0: Scheduler loop started.
Uvicorn running on http://0.0.0.0:30000
```

### 5.5 验证 serve 服务和端口

在终端 C 执行：

```bash
curl --noproxy '*' -sS -w '\nHTTP %{http_code}\n' http://127.0.0.1:30000/health
```

通过标准：

```text
{"status":"ok"}
HTTP 200
```

查看模型信息：

```bash
curl --noproxy '*' -sS http://127.0.0.1:30000/model_info
```

通过标准：

- 返回 JSON。
- JSON 中 `model_type` 为 `diffusion`。
- `task_type` 和当前模型配置匹配。

确认端口监听：

```bash
ss -ltnp | grep ':30000'
```

通过标准：能看到 `LISTEN`，端口是 `30000`。

如果从另一台机器访问 serve：

```bash
curl --noproxy '*' -sS http://SERVE_MACHINE_IP:30000/health
```

如果本机能访问、远端不能访问，检查防火墙、安全组、容器端口映射，以及 `--host` 是否为 `0.0.0.0`。

## 6. 发送本地视频请求

本节要求 serve 使用第 5.3 节的“本地输出模式”启动。本地视频推理结果只保存在本地，不上传云端。

本地视频请求使用：

- `video_input_path`
- `mask_input_path`

HTTP URL 不能填到这两个字段里。云端 URL 输入请使用第 7 节的 `video_url` 和 `mask_url`。

### 6.1 提交本地视频任务

在终端 C 执行：

```bash
cd /path/to/sglang
source .venv/bin/activate

export INPUT_VIDEO=/path/to/pexel_test_data_0410/videos/15108907_3840_2160_50fps_short.mp4
export INPUT_MASK=/path/to/pexel_test_data_0410/masks/15108907_3840_2160_50fps_No_bbox_mask.mp4
export LOCAL_OUT_DIR=/path/to/videoedit-local-outputs
export CALLBACK_URL='http://127.0.0.1:18080/videoedit/callback'
export PERF_DIR=/path/to/videoedit-perf

mkdir -p "$LOCAL_OUT_DIR" "$PERF_DIR"

python - <<'PY'
import json
import os
import urllib.request

task_id = "local_sp1_offload_81f"
payload = {
    "task_id": task_id,
    "prompt": "A close-up of an orange flower with a yellow center, remaining in focus against a blurred green grass background throughout the video.",
    "video_input_path": os.environ["INPUT_VIDEO"],
    "mask_input_path": os.environ["INPUT_MASK"],
    "callback_url": os.environ["CALLBACK_URL"],
    "output_storage": "local",
    "output_path": f"{os.environ['LOCAL_OUT_DIR']}/15108907_3840_2160_50fps_local_sp1_offload_81f.mp4",
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
    "perf_dump_path": f"{os.environ['PERF_DIR']}/videoedit_perf_local_sp1_offload_81f.json",
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

成功提交后，返回类似：

```json
{
  "id": "local_sp1_offload_81f",
  "object": "video",
  "model": "videoedit",
  "status": "queued",
  "progress": 0,
  "file_path": "/path/to/videoedit-local-outputs/15108907_3840_2160_50fps_local_sp1_offload_81f.mp4"
}
```

返回中的 `file_path` 是本地输出路径。由于本节使用本地输出模式，任务完成后也应继续以 `file_path` 为准，`url` 应为 `null`。

### 6.2 查询本地视频任务进度

```bash
curl --noproxy '*' -sS \
  http://127.0.0.1:30000/v1/videos/local_sp1_offload_81f/progress
```

运行中通常会看到：

```json
{
  "id": "local_sp1_offload_81f",
  "status": "running",
  "progress": 1,
  "file_path": "/path/to/videoedit-local-outputs/15108907_3840_2160_50fps_local_sp1_offload_81f.mp4",
  "url": null,
  "error": null,
  "callback_status": null,
  "callback_error": null,
  "callback_attempts": null
}
```

完成后应看到：

```json
{
  "id": "local_sp1_offload_81f",
  "status": "completed",
  "progress": 100,
  "file_path": "/path/to/videoedit-local-outputs/15108907_3840_2160_50fps_local_sp1_offload_81f.mp4",
  "url": null,
  "error": null,
  "callback_status": "succeeded",
  "callback_error": null,
  "callback_attempts": 1
}
```

### 6.3 验证本地输出

检查输出文件和性能 JSON：

```bash
ls -lh \
  /path/to/videoedit-local-outputs/15108907_3840_2160_50fps_local_sp1_offload_81f.mp4 \
  /path/to/videoedit-perf/videoedit_perf_local_sp1_offload_81f.json
```

检查视频 metadata：

```bash
python - <<'PY'
import cv2
from pathlib import Path

path = Path("/path/to/videoedit-local-outputs/15108907_3840_2160_50fps_local_sp1_offload_81f.mp4")
cap = cv2.VideoCapture(str(path))
info = {
    "exists": path.exists(),
    "size_mb": round(path.stat().st_size / 1024 / 1024, 2) if path.exists() else 0,
    "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    "fps": cap.get(cv2.CAP_PROP_FPS),
}
cap.release()
print(info)
PY
```

通过标准：

- `exists` 为 `True`。
- `size_mb` 大于 `0`。
- 本样例 `num_frames=81` 且 `drop_reference_frame=true`，输出帧数通常是 `80`。
- `/progress` 中 `url` 为 `null`。
- `/progress` 中 `file_path` 指向本地输出 mp4。

检查 perf JSON：

```bash
python - <<'PY'
import json
path = "/path/to/videoedit-perf/videoedit_perf_local_sp1_offload_81f.json"
data = json.load(open(path))
print(json.dumps(data, indent=2, ensure_ascii=False))
PY
```

## 7. 发送云端视频 URL 请求并上传输出到云端

本节要求 serve 使用第 5.4 节的“云端输出模式”启动：输入视频和 mask 来自云端 URL，输出视频上传到云端。

云端输入请求使用：

- `video_url`
- `mask_url`

服务端收到请求后，会先把远程视频和 mask 下载到 `--input-save-path`，再进入 VideoEdit 推理流程。

### 7.1 准备 MinIO 对象和下载 URL

如果用 MinIO 模拟云端对象存储，在 MinIO 所在机器执行。下面路径按实际机器替换：

```bash
export MINIO_ROOT_USER=your-minio-user
export MINIO_ROOT_PASSWORD=your-minio-password
export MINIO_DATA_DIR=/path/to/minio-data

minio server --address ":9000" --console-address ":9001" "$MINIO_DATA_DIR"
```

上传输入视频和 mask：

```bash
export MINIO_ENDPOINT=http://MINIO_HOST_OR_IP:9000
export INPUT_BUCKET=your-input-bucket

mc alias set macminio "$MINIO_ENDPOINT" "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD"
mc mb -p "macminio/$INPUT_BUCKET"

mc cp /path/to/15108907_3840_2160_50fps_short.mp4 \
  "macminio/$INPUT_BUCKET/test/video/15108907_3840_2160_50fps_short.mp4"

mc cp /path/to/15108907_3840_2160_50fps_No_bbox_mask.mp4 \
  "macminio/$INPUT_BUCKET/test/mask/15108907_3840_2160_50fps_No_bbox_mask.mp4"

mc stat "macminio/$INPUT_BUCKET/test/video/15108907_3840_2160_50fps_short.mp4"
mc stat "macminio/$INPUT_BUCKET/test/mask/15108907_3840_2160_50fps_No_bbox_mask.mp4"
```

有两种推荐下载方式。

方式一：公开下载 URL。执行：

```bash
mc anonymous set download "macminio/$INPUT_BUCKET"
```

然后输入 URL 形如：

```text
http://MINIO_HOST_OR_IP:9000/your-input-bucket/test/video/15108907_3840_2160_50fps_short.mp4
http://MINIO_HOST_OR_IP:9000/your-input-bucket/test/mask/15108907_3840_2160_50fps_No_bbox_mask.mp4
```

公开下载 URL 不需要在 URL 中携带用户名和密码。用户名和密码只是在执行 `mc anonymous set download` 时用于管理 bucket 权限。

方式二：presigned URL。对象仍是私有，但 URL 自带临时签名：

```bash
mc share download --expire 24h \
  "macminio/$INPUT_BUCKET/test/video/15108907_3840_2160_50fps_short.mp4"

mc share download --expire 24h \
  "macminio/$INPUT_BUCKET/test/mask/15108907_3840_2160_50fps_No_bbox_mask.mp4"
```

把输出中的 `Share:` URL 分别填到 `VIDEO_URL` 和 `MASK_URL`。presigned URL 通常包含 `?` 和 `&`，在 shell 中必须使用单引号：

```bash
export VIDEO_URL='https://example.com/your-input-bucket/test/video/file.mp4?X-Amz-Algorithm=...&X-Amz-Signature=...'
export MASK_URL='https://example.com/your-input-bucket/test/mask/file.mp4?X-Amz-Algorithm=...&X-Amz-Signature=...'
```

不要把 MinIO console 登录地址当成对象下载地址。对象 API 端口通常是 `9000`，console 端口通常是 `9001`。

### 7.2 验证 serve 能否连接到云端视频

在运行 `sglang serve` 的机器上执行。该检查只读取前 1KB，不会下载完整视频：

```bash
python - <<'PY'
import os
import urllib.request

for name in ("VIDEO_URL", "MASK_URL"):
    url = os.environ[name]
    req = urllib.request.Request(url, headers={"Range": "bytes=0-1023"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        print(name)
        print("  status:", resp.status)
        print("  content-type:", resp.headers.get("Content-Type"))
        print("  content-length:", resp.headers.get("Content-Length"))
        print("  first-bytes:", resp.read(16))
PY
```

通过标准：

- `VIDEO_URL` 和 `MASK_URL` 都返回 `200` 或 `206`。
- `Content-Type` 是 `video/mp4`，或者至少不是 `text/html` 错误页。
- 能读到二进制内容。

如果这里失败，不要提交推理任务。先修复 URL、网络、防火墙、MinIO 权限或 presigned URL 过期问题。

### 7.3 输出视频命名规则

本节只说明云端输出模式的命名规则。本地视频推理见第 6 节，输出保存在本地，不上传云端。

当前云端上传逻辑是：上传到 `SGLANG_S3_BUCKET_NAME` 指定的 bucket，object key 使用输出文件 basename。

推荐做法：

- 启动 serve 时使用 `--output-path ""`。
- 请求体不传 `output_path`。
- 每次请求显式传 `task_id`。

在这种推荐做法下：

| 请求字段 | 输出文件名 | 云端 object key | 云端 URL |
| --- | --- | --- | --- |
| `task_id=cloud_sp1_offload_81f` | `cloud_sp1_offload_81f.mp4` | `cloud_sp1_offload_81f.mp4` | `$SGLANG_S3_ENDPOINT_URL/$SGLANG_S3_BUCKET_NAME/cloud_sp1_offload_81f.mp4` |

如果请求体传了：

```json
{
  "output_path": "/path/to/outputs/custom_name.mp4"
}
```

则云端 object key 会是：

```text
custom_name.mp4
```

不是完整本地路径。也就是说，上传 key 只取 basename。

如果请求体不传 `task_id`，服务端会自动生成随机 id，输出对象也会使用随机 id 命名。面向用户或联调时建议总是显式传 `task_id`，这样输出 URL 可预测。

当前实现中，`output_bucket` 和 `output_object_key` 字段不作为默认推荐路径；不要依赖它们控制上传位置。上传 bucket 使用 `SGLANG_S3_BUCKET_NAME`，上传 key 使用输出文件 basename。

### 7.4 提交云端视频任务

确认 serve 已按第 5.4 节的云端输出模式启动后，在终端 C 执行。URL 较长时推荐用 Python 发送请求，避免 shell 转义问题：

```bash
cd /path/to/sglang
source .venv/bin/activate

export VIDEO_URL='https://your-video-download-url/15108907_3840_2160_50fps_short.mp4'
export MASK_URL='https://your-mask-download-url/15108907_3840_2160_50fps_No_bbox_mask.mp4'
export CALLBACK_URL='http://127.0.0.1:18080/videoedit/callback'
export PERF_DIR=/path/to/videoedit-perf

python - <<'PY'
import json
import os
import urllib.request

task_id = "cloud_sp1_offload_81f"
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
    "perf_dump_path": f"{os.environ['PERF_DIR']}/videoedit_perf_cloud_sp1_offload_81f.json",
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

成功提交后，返回类似：

```json
{
  "id": "cloud_sp1_offload_81f",
  "object": "video",
  "model": "videoedit",
  "status": "queued",
  "progress": 0,
  "file_path": "/path/to/temporary/cloud_sp1_offload_81f.mp4"
}
```

### 7.5 查询云端视频任务进度

```bash
curl --noproxy '*' -sS \
  http://127.0.0.1:30000/v1/videos/cloud_sp1_offload_81f/progress
```

完成且上传成功后应看到：

```json
{
  "id": "cloud_sp1_offload_81f",
  "status": "completed",
  "progress": 100,
  "file_path": null,
  "url": "https://your-minio-or-s3-endpoint/your-output-bucket/cloud_sp1_offload_81f.mp4",
  "error": null,
  "callback_status": "succeeded",
  "callback_error": null,
  "callback_attempts": 1
}
```

如果 callback server 正常，终端 B 会打印一次最终 payload：

```json
{
  "id": "cloud_sp1_offload_81f",
  "object": "video",
  "model": "videoedit",
  "status": "completed",
  "progress": 100,
  "file_path": null,
  "url": "https://your-minio-or-s3-endpoint/your-output-bucket/cloud_sp1_offload_81f.mp4",
  "error": null,
  "peak_memory_mb": 12345.0,
  "inference_time_s": 415.27
}
```

### 7.6 验证远程输入已下载

远程 `video_url` 和 `mask_url` 会被下载到 serve 启动时指定的 `--input-save-path`。

```bash
ls -lh /path/to/sglang-videoedit-inputs | grep cloud_sp1_offload_81f
```

通过标准：能看到类似文件：

```text
cloud_sp1_offload_81f_video.mp4
cloud_sp1_offload_81f_mask.mp4
```

如果看不到，检查：

- 请求体是否使用了 `video_url` / `mask_url`。
- serve 启动时 `--input-save-path` 是否是同一个目录。
- 任务是否已经开始执行。

### 7.7 验证云端输出对象

用 `/progress` 返回的 `url` 验证：

```bash
export OUTPUT_URL='https://your-minio-or-s3-endpoint/your-output-bucket/cloud_sp1_offload_81f.mp4'

python - <<'PY'
import os
import urllib.request

req = urllib.request.Request(os.environ["OUTPUT_URL"], headers={"Range": "bytes=0-1023"})
with urllib.request.urlopen(req, timeout=30) as resp:
    print("status:", resp.status)
    print("content-type:", resp.headers.get("Content-Type"))
    print("content-length:", resp.headers.get("Content-Length"))
    print("first-bytes:", resp.read(16))
PY
```

通过标准：

- 状态码是 `200` 或 `206`。
- `Content-Type` 是 `video/mp4`，或者至少不是 HTML 错误页。
- 能读到二进制内容。

如果使用 `mc`，也可以检查对象：

```bash
mc stat your-minio-alias/your-output-bucket/cloud_sp1_offload_81f.mp4
```

检查 perf JSON：

```bash
python - <<'PY'
import json
path = "/path/to/videoedit-perf/videoedit_perf_cloud_sp1_offload_81f.json"
data = json.load(open(path))
print(json.dumps(data, indent=2, ensure_ascii=False))
PY
```

## 8. 完整验证清单

### 8.1 Serve 服务验证

```bash
curl --noproxy '*' -sS -w '\nHTTP %{http_code}\n' http://127.0.0.1:30000/health
ss -ltnp | grep ':30000'
```

通过标准：

- `/health` 返回 `{"status":"ok"}` 和 `HTTP 200`。
- `30000` 处于 `LISTEN`。
- 终端 A 没有退出，没有持续报错。

### 8.2 Callback 监听验证

```bash
curl --noproxy '*' -sS -X POST http://127.0.0.1:18080/videoedit/callback \
  -H 'Content-Type: application/json' \
  -d '{"id":"probe","status":"ok"}'
ss -ltnp | grep ':18080'
```

通过标准：

- curl 返回 `{"ok":true}`。
- 终端 B 打印 probe payload。
- `18080` 处于 `LISTEN`。

### 8.3 云端输入连通性验证

```bash
python - <<'PY'
import os
import urllib.request

for name in ("VIDEO_URL", "MASK_URL"):
    req = urllib.request.Request(os.environ[name], headers={"Range": "bytes=0-1023"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        print(name, resp.status, resp.headers.get("Content-Type"), resp.read(16))
PY
```

通过标准：

- `VIDEO_URL` 和 `MASK_URL` 都能读到数据。
- 状态码为 `200` 或 `206`。
- 返回内容不是 HTML 错误页。

### 8.4 本地输出验证

本地视频任务使用第 5.3 节本地输出模式时，验证标准如下：

```bash
curl --noproxy '*' -sS http://127.0.0.1:30000/v1/videos/local_sp1_offload_81f/progress
ls -lh /path/to/videoedit-local-outputs/15108907_3840_2160_50fps_local_sp1_offload_81f.mp4
```

通过标准：

- `status` 最终为 `completed`。
- `progress` 为 `100`。
- `error` 为 `null`。
- `file_path` 指向本地输出 mp4。
- `url` 为 `null`。
- 本地 mp4 文件存在且大小大于 `0`。
- 如果传了 `callback_url`，`callback_status` 为 `succeeded`。

### 8.5 云端输出上传验证

```bash
curl --noproxy '*' -sS http://127.0.0.1:30000/v1/videos/TASK_ID/progress
```

通过标准：

- `status` 最终为 `completed`。
- `progress` 为 `100`。
- `error` 为 `null`。
- `url` 不为 `null`。
- `file_path` 为 `null`，表示本地临时视频不是最终交付物。
- 使用 `url` 能读回视频对象。
- 如果传了 `callback_url`，`callback_status` 为 `succeeded`。

## 9. 常见问题

### 9.1 公开 URL 为什么不需要用户名和密码

MinIO 的 access key 和 secret key 用于管理 bucket、上传对象、生成 presigned URL。执行 `mc anonymous set download` 后，对象下载权限已经公开，普通 HTTP GET 不需要用户名和密码。

如果不能公开对象，就不要使用公开 URL。请使用 `mc share download --expire ...` 生成 presigned URL。presigned URL 自带临时签名，serve 直接 HTTP GET 即可。

当前 `/v1/videos/repairs` 的 `video_url` / `mask_url` 没有单独的用户名、密码、Authorization header 字段，因此私有普通 URL 不适合作为输入。

### 9.2 `/health` 访问失败

检查：

- 终端 A 的 serve 是否还在运行。
- 请求端口是否和 serve 启动端口一致。
- `ss -ltnp | grep ':30000'` 是否能看到监听。
- 远程访问时防火墙、安全组或容器端口映射是否放行。

### 9.3 `videoedit_queue_full`

当前文档使用 `VIDEOEDIT_QUEUE_CAPACITY=1`，同一时间只允许一个 VideoEdit 任务。等当前任务完成后再提交，或者重启 serve 后重新提交。

### 9.4 callback 没有输出

按顺序检查：

- 终端 B 是否还在运行 callback server。
- 请求体里是否传了 `callback_url`。
- `callback_url` 是否从 serve 进程所在机器可访问。
- 先用 probe 请求测试 callback URL。
- 查询 `/progress`，看 `callback_status`、`callback_error` 和 `callback_attempts`。

注意：callback 失败不等于推理失败。任务可能已经 `completed`，只是回调通知没有送达。

### 9.5 云端输入 URL 返回 403、404 或 HTML

常见原因：

- bucket 没有公开下载权限。
- presigned URL 过期。
- URL 复制不完整，漏掉了 `?` 后面的签名参数。
- 使用了 MinIO console 端口，而不是对象 API 端口。
- URL 写成 `127.0.0.1`，但 serve 运行在另一台机器上。

先用第 7.2 节的 Range 读取命令验证，确认云端视频能从 serve 机器访问。

### 9.6 `video_input_path or video_url is required`

说明请求体没有被服务端识别到输入视频。检查字段名：

- 本地输入必须使用 `video_input_path` 和 `mask_input_path`。
- 云端输入必须使用 `video_url` 和 `mask_url`。
- 不要把 HTTP URL 填到 `video_input_path`。

### 9.7 `/progress` 中 `url` 是 `null`

如果是第 6 节本地视频任务，这是正常结果：本地任务不上传云端，最终结果看 `file_path`。

如果是第 7 节云端视频任务，`status=completed` 但 `url=null`，说明云端上传没有生效或上传失败。检查：

- `SGLANG_CLOUD_STORAGE_TYPE=s3` 是否在启动 serve 前设置。
- `SGLANG_S3_BUCKET_NAME`、`SGLANG_S3_ENDPOINT_URL`、access key 和 secret key 是否正确。
- `boto3` 是否安装在当前环境。
- 启动 serve 前的上传 probe 是否成功。
- serve 日志里是否有 `Upload failed`。
- `SGLANG_S3_ENDPOINT_URL` 是否能从 serve 进程访问。

### 9.8 本地视频任务被上传到云端

说明 serve 进程启动时仍然启用了 S3 cloud storage。处理方式：

- 停止当前 serve。
- 按第 5.3 节重新启动本地输出模式。
- 确认启动前已经执行 `unset SGLANG_CLOUD_STORAGE_TYPE` 和相关 `SGLANG_S3_*` 变量。
- 请求体中传 `output_storage: "local"` 和本地 `output_path`。

### 9.9 输出对象名不符合预期

检查：

- 请求是否显式传了 `task_id`。
- 请求是否传了 `output_path`。如果传了，云端 object key 会取 `output_path` 的 basename。
- 云端输出 bucket 来自 `SGLANG_S3_BUCKET_NAME`，不是请求体字段。

推荐保持：

```json
{
  "task_id": "cloud_sp1_offload_81f"
}
```

并且不传 `output_path`，这样输出对象名就是：

```text
cloud_sp1_offload_81f.mp4
```

### 9.10 GPU 不可见或显存不足

如果 `nvidia-smi` 或 PyTorch 看不到 GPU，需要在能访问宿主 GPU 的终端里运行。单卡 offload 配置主要用于降低显存压力；如果仍然 OOM，可以先确认没有其他进程占用 GPU，或者保持 `VIDEOEDIT_QUEUE_CAPACITY=1`。

## 10. 停止服务

停止 serve：

- 回到终端 A，按 `Ctrl-C`。

停止 callback listener：

- 回到终端 B，按 `Ctrl-C`。

如果终端已经关闭但进程还在，可以先查端口：

```bash
ss -ltnp | grep -E ':(30000|18080) '
```

找到 PID 后优先使用：

```bash
kill -TERM PID
```

确认端口已释放：

```bash
ss -ltnp | grep -E ':(30000|18080) '
```

没有输出表示端口已不再监听。
