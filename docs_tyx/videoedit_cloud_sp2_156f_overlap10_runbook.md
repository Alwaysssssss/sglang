# VideoEdit 云端输入输出双卡 156 帧 overlap=10 完整测试流程

本文只覆盖云端输入和云端输出，不包含本地视频输入测试。

目标配置：

- 模型：`/mnt/nas/models/DifusserEdit/pretrain_models/VideoEdit-diffusers-model`
- 输入视频对象：`s3://flowcut/test-edit-tyx/video/15108907_3840_2160_50fps_short.mp4`
- 输入 mask 对象：`s3://flowcut/test-edit-tyx/mask/15108907_3840_2160_50fps_No_bbox_mask.mp4`
- 输出：上传到 S3-compatible 云端存储
- GPU：双卡，容器内 `CUDA_VISIBLE_DEVICES=0,1`
- 并行：`num_gpus=2`、`sp_degree=2`、`ulysses_degree=2`、`ring_degree=1`
- 帧数：`num_frames=156`
- 窗口：`infer_len=81`
- overlap：`10`
- step：`num_inference_steps=20`
- callback：启用
- offload：默认开启

注意：当前 `video_url` / `mask_url` 下载逻辑支持 `http://` 和 `https://`，不直接支持把 `s3://...` 填进请求体。本文先用 S3 凭证生成 presigned HTTP URL，再把 URL 传给 `video_url` / `mask_url`。

另一个重要点：当前窗口切分逻辑使用 `stride = infer_len - overlap`。所以 `num_frames=156`、`infer_len=81`、`overlap=10` 时，stride 是 `71`，窗口起点是 `0, 71, 142`，会跑 3 个窗口：

```text
window 0: [0, 81)
window 1: [71, 152)
window 2: [142, 156) + 反射补帧到 81 帧
```

如果日志里 `VideoEditDenoisingStage` 出现 3 次，这是符合当前实现的。

## 1. 终端分工

先创建容器。这里要映射 serve 端口 `30000`，并通过 `host.docker.internal` 让容器内的 serve 可以访问宿主机上的 callback listener；不要映射 callback 端口 `18080`，因为本文的 callback listener 运行在宿主机上。

```bash
docker run -itd \
  --name sglang-tyx-dev \
  --gpus '"device=4,5"' \
  --shm-size 32g \
  --ipc=host \
  -p 30000:30000 \
  --add-host=host.docker.internal:host-gateway \
  -v /mnt/nas/tyx/sglang:/sgl-workspace/sglang \
  -v /mnt/nas/models:/mnt/nas/models \
  -w /sgl-workspace/sglang \
  sglang-mgtv:1.0 \
  /bin/zsh
```

三个终端用途：

| 终端 | 用途 |
| --- | --- |
| 终端 A | 进入 Docker 容器，启动 `sglang serve` |
| 终端 B | 不进入容器，在宿主机启动 callback listener |
| 终端 C | 不进入容器，在宿主机生成 presigned URL、提交请求、轮询和检查输出 |

终端 A 进入容器：

```bash
docker exec -it sglang-tyx-dev bash39.108.238.21
```

## 2. 终端 C：配置 S3 环境变量并验证上传权限

下面变量在终端 A 和终端 C 都需要设置。`SGLANG_CLOUD_STORAGE_TYPE` 等变量必须在启动 serve 前设置好。

```bash
cd /sgl-workspace/sglang

export SGLANG_CLOUD_STORAGE_TYPE=s3
export SGLANG_S3_BUCKET_NAME=flowcut
export SGLANG_S3_ENDPOINT_URL='http://s3-legacy.mediacloud.imgo.tv'
export SGLANG_S3_REGION_NAME=cn-changsha-2
export SGLANG_S3_ACCESS_KEY_ID='YOUR_ACCESS_KEY'
export SGLANG_S3_SECRET_ACCESS_KEY='YOUR_SECRET_KEY'
export NO_PROXY=127.0.0.1,localhost,s3-legacy.mediacloud.imgo.tv
export no_proxy=127.0.0.1,localhost,s3-legacy.mediacloud.imgo.tv
```

确认 `boto3` 可用：

```bash
python3 - <<'PY'
import boto3
print("boto3 ok:", boto3.__version__)
PY
```

做一次云端上传 probe：

```bash
python3 - <<'PY'
import os
import tempfile
import boto3

client = boto3.client(
    "s3",
    endpoint_url=os.environ.get("SGLANG_S3_ENDPOINT_URL") or None,
    region_name=os.environ.get("SGLANG_S3_REGION_NAME") or "us-east-1",
    aws_access_key_id=os.environ["SGLANG_S3_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["SGLANG_S3_SECRET_ACCESS_KEY"],
)

path = tempfile.NamedTemporaryFile(delete=False, suffix=".txt").name
with open(path, "w") as f:
    f.write("sglang cloud upload probe\n")

key = "test-edit-tyx/sglang_upload_probe.txt"
client.upload_file(path, os.environ["SGLANG_S3_BUCKET_NAME"], key, ExtraArgs={"ContentType": "text/plain"})
print("upload probe ok:", f"s3://{os.environ['SGLANG_S3_BUCKET_NAME']}/{key}")
PY
```

通过标准：打印 `upload probe ok`，且没有异常。

## 3. 终端 C：生成并验证云端输入 presigned URL

生成 `VIDEO_URL` 和 `MASK_URL`：

```bash
python3 - <<'PY'
import os
import boto3

client = boto3.client(
    "s3",
    endpoint_url=os.environ.get("SGLANG_S3_ENDPOINT_URL") or None,
    region_name=os.environ.get("SGLANG_S3_REGION_NAME") or "us-east-1",
    aws_access_key_id=os.environ["SGLANG_S3_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["SGLANG_S3_SECRET_ACCESS_KEY"],
)

objects = {
    "VIDEO_URL": "test-edit-tyx/video/15108907_3840_2160_50fps_short.mp4",
    "MASK_URL": "test-edit-tyx/mask/15108907_3840_2160_50fps_No_bbox_mask.mp4",
}

for env_name, key in objects.items():
    url = client.generate_presigned_url(
        "get_object",
        Params={"Bucket": os.environ["SGLANG_S3_BUCKET_NAME"], "Key": key},
        ExpiresIn=24 * 3600,
    )
    print(f"export {env_name}='{url}'")
PY
```

把输出的两行 `export VIDEO_URL=...` 和 `export MASK_URL=...` 复制并执行。

验证容器内能读取这两个 URL：

```bash
python3 - <<'PY'
import os
import urllib.request

for name in ("VIDEO_URL", "MASK_URL"):
    req = urllib.request.Request(os.environ[name], headers={"Range": "bytes=0-1023"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        print(name, resp.status, resp.headers.get("Content-Type"), resp.headers.get("Content-Length"), resp.read(16))
PY
```

通过标准：

- `VIDEO_URL` 和 `MASK_URL` 都返回 `200` 或 `206`。
- 返回内容不是 HTML 错误页。
- mp4 通常能看到类似 `b'\x00\x00\x00...ftyp'` 的开头。

## 4. 终端 B：启动 callback listener

```bash
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

python3 /tmp/videoedit_callback_server.py
```

看到下面日志表示 callback listener 已启动：

```text
Uvicorn running on http://0.0.0.0:18080
```

终端 B 需要一直保持运行。

## 5. 终端 A：启动双卡云端输出 serve

在终端 A 里也设置同样的 S3 环境变量：

```bash
cd /sgl-workspace/sglang

export SGLANG_CLOUD_STORAGE_TYPE=s3
export SGLANG_S3_BUCKET_NAME=flowcut
export SGLANG_S3_ENDPOINT_URL='http://s3-legacy.mediacloud.imgo.tv'
export SGLANG_S3_REGION_NAME=cn-changsha-2
export SGLANG_S3_ACCESS_KEY_ID='YOUR_ACCESS_KEY'
export SGLANG_S3_SECRET_ACCESS_KEY='YOUR_SECRET_KEY'
export NO_PROXY=127.0.0.1,localhost,s3-legacy.mediacloud.imgo.tv
export no_proxy=127.0.0.1,localhost,s3-legacy.mediacloud.imgo.tv
```

启动 serve：

```bash
export MODEL_PATH=/mnt/nas/models/DifusserEdit/pretrain_models/VideoEdit-diffusers-model
export TRANSFORMER_PATH=$MODEL_PATH/transformer
export INPUT_SAVE_DIR=/tmp/sglang-videoedit-cloud-inputs
export CACHE_DIR=/tmp/sglang-cache

export CUDA_VISIBLE_DEVICES=0,1
export VIDEOEDIT_QUEUE_CAPACITY=1
export FLASHINFER_WORKSPACE_BASE=$CACHE_DIR/flashinfer
export XDG_CACHE_HOME=$CACHE_DIR/xdg
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$INPUT_SAVE_DIR" "$FLASHINFER_WORKSPACE_BASE" "$XDG_CACHE_HOME" /tmp/videoedit-cloud-perf

ls -lh "$MODEL_PATH/model_index.json" "$TRANSFORMER_PATH/config.json"

sglang serve \
  --model-type diffusion \
  --model-path "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port 30000 \
  --num-gpus 2 \
  --sp-degree 2 \
  --ulysses-degree 2 \
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

这里 `--output-path ""` 表示本地只用临时目录中转输出视频。任务完成后会上传到云端；上传成功后，本地临时 mp4 会被清理。

服务启动成功后，日志应出现：

```text
Uvicorn running on http://0.0.0.0:30000
```

## 6. 终端 C：验证 serve 和 callback

```bash
curl --noproxy '*' -sS -w '\nHTTP %{http_code}\n' http://127.0.0.1:30000/health
```

通过标准：

```text
{"status":"ok"}
HTTP 200
```

验证 callback listener：

```bash
curl --noproxy '*' -sS -X POST http://127.0.0.1:18080/videoedit/callback \
  -H 'Content-Type: application/json' \
  -d '{"id":"probe","status":"ok"}'
```

通过标准：

- curl 返回 `{"ok":true}`。
- 终端 B 打印 `VideoEdit callback received`。

## 7. 终端 C：提交 156 帧、overlap=10、20 step 云端任务

确认终端 C 已经有 `VIDEO_URL` 和 `MASK_URL`：

```bash
test -n "$VIDEO_URL" && test -n "$MASK_URL" && echo "input urls ready"
```

提交请求：

```bash
curl --noproxy '*' -sS -X POST http://127.0.0.1:30000/v1/videos/repairs \
  -H 'Content-Type: application/json' \
  -d "{
    \"task_id\": \"cloud_sp2_overlap10_156f_20step\",
    \"prompt\": \"A close-up of an orange flower with a yellow center, remaining in focus against a blurred green grass background throughout the video.\",
    \"video_url\": \"$VIDEO_URL\",
    \"mask_url\": \"$MASK_URL\",
    \"callback_url\": \"http://host.docker.internal:18080/videoedit/callback\",
    \"num_frames\": -1,
    \"infer_len\": 81,
    \"overlap\": 5,
    \"num_inference_steps\": 20,
    \"guidance_scale\": 5.0,
    \"dynamic_cfg\": true,
    \"dynamic_cfg_max_step\": 15,
    \"seed\": 42,
    \"dtype\": \"bf16\",
    \"enable_paste_back\": true,
    \"drop_reference_frame\": false,
    \"perf_dump_path\": \"/tmp/videoedit-cloud-perf/cloud_sp2_overlap10_156f_20step.perf.json\"
  }"
```

输出视频的名字为task_id，成功提交后应返回 `queued`，类似：

```json
{
  "id": "cloud_sp2_overlap10_156f_20step",
  "status": "queued",
  "progress": 0
}
```

## 8. 终端 C：轮询进度

```bash
curl --noproxy '*' -sS \
  http://127.0.0.1:30000/v1/videos/cloud_sp2_overlap10_156f_20step/progress
```

运行中会看到：

```json
{
  "id": "cloud_sp2_overlap10_156f_20step",
  "status": "running",
  "progress": 1,
  "file_path": "...",
  "url": null,
  "error": null
}
```

完成后应看到：

```json
{
  "id": "cloud_sp2_overlap10_156f_20step",
  "status": "completed",
  "progress": 100,
  "file_path": null,
  "url": "http://s3-legacy.mediacloud.imgo.tv/flowcut/cloud_sp2_overlap10_156f_20step.mp4",
  "error": null,
  "callback_status": "succeeded",
  "callback_error": null,
  "callback_attempts": 1
}
```

当前代码上传云端输出时，object key 使用输出文件 basename。因此本任务的输出对象默认是：

```text
s3://flowcut/cloud_sp2_overlap10_156f_20step.mp4
```

请求体里的 `output_object_key` 当前不会改变这个 key。

## 9. 终端 C：检查云端输出对象

用 `head_object` 检查对象：

```bash
python3 - <<'PY'
import os
import boto3

bucket = os.environ["SGLANG_S3_BUCKET_NAME"]
key = "cloud_sp2_overlap10_156f_20step.mp4"

client = boto3.client(
    "s3",
    endpoint_url=os.environ.get("SGLANG_S3_ENDPOINT_URL") or None,
    region_name=os.environ.get("SGLANG_S3_REGION_NAME") or "us-east-1",
    aws_access_key_id=os.environ["SGLANG_S3_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["SGLANG_S3_SECRET_ACCESS_KEY"],
)

resp = client.head_object(Bucket=bucket, Key=key)
print("exists:", f"s3://{bucket}/{key}")
print("size:", resp["ContentLength"])
print("content_type:", resp.get("ContentType"))
PY
```

也可以生成一个输出视频的临时下载 URL：

```bash
python3 - <<'PY'
import os
import boto3

bucket = os.environ["SGLANG_S3_BUCKET_NAME"]
key = "cloud_sp2_overlap10_156f_20step.mp4"

client = boto3.client(
    "s3",
    endpoint_url=os.environ.get("SGLANG_S3_ENDPOINT_URL") or None,
    region_name=os.environ.get("SGLANG_S3_REGION_NAME") or "us-east-1",
    aws_access_key_id=os.environ["SGLANG_S3_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["SGLANG_S3_SECRET_ACCESS_KEY"],
)

url = client.generate_presigned_url(
    "get_object",
    Params={"Bucket": bucket, "Key": key},
    ExpiresIn=24 * 3600,
)
print(url)
PY
```

检查返回 URL 的前 1KB：

```bash
export OUTPUT_URL='<上一步打印的 URL>'

python3 - <<'PY'
import os
import urllib.request

req = urllib.request.Request(os.environ["OUTPUT_URL"], headers={"Range": "bytes=0-1023"})
with urllib.request.urlopen(req, timeout=30) as resp:
    print(resp.status, resp.headers.get("Content-Type"), resp.headers.get("Content-Length"), resp.read(16))
PY
```
