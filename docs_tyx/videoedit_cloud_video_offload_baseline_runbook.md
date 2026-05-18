# VideoEdit 云端输入输出单卡 Offload 基线测试流程

本文记录如何验证 `sglang serve` 能否使用云端视频 URL 作为 VideoEdit 输入完成一次推理，并把生成的视频自动上传回云端。这里的“云端”由本机 Mac 上的 MinIO 模拟，GPU Linux 机器运行 SGLang serve。

测试目标：

- Mac MinIO 中的视频和 mask 能被 GPU 机器通过 HTTP(S) URL 访问。
- `/v1/videos/repairs` 使用 `video_url` / `mask_url` 提交任务，而不是本地 `video_input_path` / `mask_input_path`。
- 单卡 `sp1` + CPU/layerwise offload 基线跑完一次，输出视频通过 S3-compatible 接口上传回 Mac MinIO。
- 关闭本地持久化输出视频；serve 只使用临时目录中转生成文件，上传成功后删除临时视频。

当前已知对象侧检查结果中，视频对象已经能连接到 `macminio`：

```text
Added `macminio` successfully.
Name      : 15108907_3840_2160_50fps_short.mp4
Date      : 2026-05-18 03:38:07 UTC
Size      : 7.3 MiB
ETag      : 2c8b637bdba6096a79ba34a2455067a0
Type      : file
Metadata  :
  Content-Type: video/mp4
```

如果第二个 `Size: 358 KiB` 对应 mask，也按同样方式确认 mask 对象存在并能下载。

## 1. 关键结论

当前 VideoEdit repair API 对远程输入的支持路径是：

```json
{
  "video_url": "http://...",
  "mask_url": "http://..."
}
```

服务端收到请求后会先把远程文件下载到 `--input-save-path`，再把下载后的本地路径传给 VideoEdit。不要把 HTTP URL 或 `s3://...` 填进 `video_input_path` / `mask_input_path`，这两个字段按本地文件路径处理。

MinIO 是 S3-compatible 对象存储，但本次测试建议先使用 MinIO 的公开 HTTP URL 或 presigned URL。这样测试覆盖的是“serve 能否从云端 URL 拉取视频并推理”，链路最直接。

输出视频上传使用当前代码已有的 S3-compatible cloud storage 逻辑，不需要改代码。开启方式是在启动 serve 之前设置：

```bash
export SGLANG_CLOUD_STORAGE_TYPE=s3
export SGLANG_S3_BUCKET_NAME=flowcut
export SGLANG_S3_ENDPOINT_URL='http://127.0.0.1:19000'
export SGLANG_S3_REGION_NAME=us-east-1
export SGLANG_S3_ACCESS_KEY_ID='你的 MinIO access key'
export SGLANG_S3_SECRET_ACCESS_KEY='你的 MinIO secret key'
```

当前上传 key 固定为输出文件名的 basename。本文请求不传 `output_path`，因此云端输出文件名会是 `cloud_sp1_offload_81f.mp4`，完整 URL 预期为：

```text
http://127.0.0.1:19000/flowcut/cloud_sp1_offload_81f.mp4
```

## 2. Mac 端准备 MinIO 对象

在 Mac 上启动 MinIO。`MINIO_ROOT_USER` 和 `MINIO_ROOT_PASSWORD` 按你的实际配置替换。

```bash
export MINIO_ROOT_USER=minioadmin
export MINIO_ROOT_PASSWORD=minioadmin
minio server --address ":9000" --console-address ":9001" "$HOME/minio-data"
```

查看 Mac 局域网 IP。不要把给 GPU 机器访问的 URL 写成 `127.0.0.1`，因为在 GPU 机器上 `127.0.0.1` 指向 GPU 机器自己，不是 Mac。

```bash
ipconfig getifaddr en0
```

假设 Mac IP 是 `MAC_LAN_IP`，设置 alias 并上传视频和 mask：

```bash
export MAC_MINIO_ENDPOINT=http://MAC_LAN_IP:9000
export BUCKET=sglang-videoedit

mc alias set macminio "$MAC_MINIO_ENDPOINT" "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD"
mc mb -p "macminio/$BUCKET"

mc cp /path/to/15108907_3840_2160_50fps_short.mp4 \
  "macminio/$BUCKET/videos/15108907_3840_2160_50fps_short.mp4"

mc cp /path/to/15108907_3840_2160_50fps_No_bbox_mask.mp4 \
  "macminio/$BUCKET/masks/15108907_3840_2160_50fps_No_bbox_mask.mp4"

mc stat "macminio/$BUCKET/videos/15108907_3840_2160_50fps_short.mp4"
mc stat "macminio/$BUCKET/masks/15108907_3840_2160_50fps_No_bbox_mask.mp4"
```

生成 GPU 机器可访问的 URL。推荐二选一。

公开 bucket 下载：

```bash
mc anonymous set download "macminio/$BUCKET"

export VIDEO_URL="$MAC_MINIO_ENDPOINT/$BUCKET/videos/15108907_3840_2160_50fps_short.mp4"
export MASK_URL="$MAC_MINIO_ENDPOINT/$BUCKET/masks/15108907_3840_2160_50fps_No_bbox_mask.mp4"
```

或者使用临时 presigned URL：

```bash
mc share download --expire 24h \
  "macminio/$BUCKET/videos/15108907_3840_2160_50fps_short.mp4"

mc share download --expire 24h \
  "macminio/$BUCKET/masks/15108907_3840_2160_50fps_No_bbox_mask.mp4"
```

把输出里的 `Share:` URL 分别复制到 GPU 机器的 `VIDEO_URL` 和 `MASK_URL`。presigned URL 通常包含 `?` 和 `&`，在 shell 中必须用单引号包起来。

## 3. GPU 机器检查云端连通性

在 GPU 机器上进入仓库环境：

```bash
cd /home/tyx/workspace/zhouhao6/sglang
source .venv/bin/activate
```

设置从 Mac MinIO 得到的 URL。这里使用当前测试中的 `127.0.0.1:19000`；如果 serve 不在同一个网络命名空间内运行，需要把 `127.0.0.1` 换成 serve 机器可访问的 Mac IP 或代理地址。

```bash
export VIDEO_URL='http://127.0.0.1:19000/flowcut/test/video/15108907_3840_2160_50fps_short.mp4'
export MASK_URL='http://127.0.0.1:19000/flowcut/test/mask/15108907_3840_2160_50fps_No_bbox_mask.mp4'
```

先在 GPU 机器上确认能访问这两个 URL：

```bash
python - <<'PY'
import os
import urllib.request

for name in ("VIDEO_URL", "MASK_URL"):
    url = os.environ[name]
    req = urllib.request.Request(url, headers={"Range": "bytes=0-1023"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        print(name, resp.status, resp.headers.get("Content-Type"), resp.headers.get("Content-Length"))
        data = resp.read(32)
        print(name, "first_bytes", data[:16])
PY
```

通过标准：

- 两个 URL 都返回 `200` 或 `206`。
- `VIDEO_URL` 的 `Content-Type` 是 `video/mp4` 或至少不是 HTML 错误页。
- 如果这里连不上，先修 Mac 防火墙、MinIO 监听地址、局域网 IP、代理或 presigned URL 过期问题，不要先启动推理。

## 4. 启动单卡 Offload Serve，并开启 S3 输出上传

这是保守的单卡基线：`num_gpus=1`、`sp_degree=1`、启用 CPU/layerwise offload。它的目标是先稳定跑通，不追求最快速度。S3/MinIO 相关环境变量必须在启动 serve 前设置，因为 cloud storage 是服务启动时初始化的。

```bash
cd /home/tyx/workspace/zhouhao6/sglang
source .venv/bin/activate

export MODEL_PATH=/home/tyx/workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model
export TRANSFORMER_PATH=/home/tyx/workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model/transformer
export INPUT_SAVE_DIR=/tmp/sglang-videoedit-cloud-inputs
export FLASHINFER_WORKSPACE_BASE=/tmp/sglang-flashinfer
export XDG_CACHE_HOME=/tmp/sglang-cache
export VIDEOEDIT_QUEUE_CAPACITY=1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export SGLANG_CLOUD_STORAGE_TYPE=s3
export SGLANG_S3_BUCKET_NAME=flowcut
export SGLANG_S3_ENDPOINT_URL='http://127.0.0.1:19000'
export SGLANG_S3_REGION_NAME=us-east-1
export SGLANG_S3_ACCESS_KEY_ID='你的 MinIO access key'
export SGLANG_S3_SECRET_ACCESS_KEY='你的 MinIO secret key'
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost

mkdir -p "$INPUT_SAVE_DIR" /tmp/sglang-flashinfer /tmp/sglang-cache

python - <<'PY'
import boto3
print("boto3 ok", boto3.__version__)
PY

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

服务启动成功后，确认日志中出现：

```text
Uvicorn running on http://0.0.0.0:30000
LayerwiseOffloadManager initialized
```

如果启动日志里有 `boto3 is not installed`，先安装 `boto3` 后重启 serve：

```bash
pip install boto3
```

另开终端检查 health：

```bash
curl --noproxy '*' -s http://127.0.0.1:30000/health
```

## 5. 提交云端视频基线请求

另开请求终端，保持同样的 `VIDEO_URL` 和 `MASK_URL` 环境变量。这里跑一次 81 帧基线，和之前 `sp1_offload` 结果可对齐；如果要跑完整 156 帧，把 `num_frames` 改成 `156` 或 `-1`。

请求体不要传 `output_path`，这样 serve 会用临时目录生成 `cloud_sp1_offload_81f.mp4`，随后上传到 `flowcut` bucket 根目录并删除本地临时视频。上传开关由 serve 启动前的 `SGLANG_CLOUD_STORAGE_TYPE=s3` 等环境变量控制，不依赖请求体里的 `output_storage` 字段。

`perf_dump_path` 是本地性能 JSON，可保留；如果连 perf JSON 也不想落本地，可以删掉该字段。

```bash
cd /home/tyx/workspace/zhouhao6/sglang
source .venv/bin/activate

export VIDEO_URL='http://127.0.0.1:19000/flowcut/test/video/15108907_3840_2160_50fps_short.mp4'
export MASK_URL='http://127.0.0.1:19000/flowcut/test/mask/15108907_3840_2160_50fps_No_bbox_mask.mp4'

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
    "perf_dump_path": "/tmp/videoedit_perf_api_cloud_sp1_offload_81f.json",
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

成功提交后应返回 `queued`，例如：

```json
{
  "id": "cloud_sp1_offload_81f",
  "status": "queued",
  "file_path": "/tmp/sglang_videoedit_output_xxx/cloud_sp1_offload_81f.mp4"
}
```

这里的 `file_path` 是服务端临时文件路径。任务完成并上传成功后，最终结果以 `url` 为准。

## 6. 查询进度和检查下载缓存

查询任务进度：

```bash
curl --noproxy '*' -s http://127.0.0.1:30000/v1/videos/cloud_sp1_offload_81f/progress
```

运行中会看到 `running`，完成后应看到：

```json
{
  "id": "cloud_sp1_offload_81f",
  "status": "completed",
  "progress": 100,
  "file_path": null,
  "url": "http://127.0.0.1:19000/flowcut/cloud_sp1_offload_81f.mp4",
  "error": null
}
```

检查远程输入是否已经被 serve 下载到本地：

```bash
ls -lh /tmp/sglang-videoedit-cloud-inputs | grep cloud_sp1_offload_81f
```

预期至少看到类似文件：

```text
cloud_sp1_offload_81f_video.mp4
cloud_sp1_offload_81f_mask.mp4
```

## 7. 检查云端输出和记录基线

在 Mac 或配置了 `macminio` alias 的机器上检查输出对象：

```bash
mc ls macminio/flowcut/
mc stat macminio/flowcut/cloud_sp1_offload_81f.mp4
```

也可以从 GPU 机器用 HTTP 读回云端输出，确认对象已经上传成功：

```bash
python - <<'PY'
import urllib.request

url = "http://127.0.0.1:19000/flowcut/cloud_sp1_offload_81f.mp4"
req = urllib.request.Request(url, headers={"Range": "bytes=0-1023"})
with urllib.request.urlopen(req, timeout=30) as resp:
    print(resp.status, resp.headers.get("Content-Type"), resp.headers.get("Content-Length"))
    print(resp.read(16))
PY
```

如果保留了 `perf_dump_path`，查看本地 perf JSON：

```bash
python - <<'PY'
import json

path = "/tmp/videoedit_perf_api_cloud_sp1_offload_81f.json"
data = json.load(open(path))
print(json.dumps(data, indent=2, ensure_ascii=False))
PY
```

建议记录：

```text
task_id:
video_url 类型: public / presigned
video object size / etag:
mask object size / etag:
serve 启动时间:
warmup 耗时:
total_duration_ms:
VideoEditDenoisingStage:
输出视频帧数 / 分辨率 / fps:
GPU 峰值显存:
```

通过标准：

- `/progress` 最终为 `completed`。
- `error` 为 `null`。
- `/progress` 里的 `url` 指向 `http://127.0.0.1:19000/flowcut/cloud_sp1_offload_81f.mp4`。
- `/progress` 里的 `file_path` 为 `null`，表示本地输出视频没有持久保存。
- `mc stat macminio/flowcut/cloud_sp1_offload_81f.mp4` 能查到对象。
- 如果保留 perf JSON，perf JSON 存在并包含各 stage 耗时。
- `/tmp/sglang-videoedit-cloud-inputs` 中能看到从 Mac MinIO 下载下来的 video/mask 文件。

## 8. 常见问题

### GPU 机器访问不到 Mac MinIO

先在 GPU 机器上执行第 3 节的 URL 检查。常见原因：

- URL 使用了 `127.0.0.1`，导致 GPU 机器访问的是自己。
- MinIO 只监听了 Mac localhost，没有监听局域网地址；启动时使用 `--address ":9000"`。
- Mac 防火墙阻止了 `9000` 端口。
- GPU 机器和 Mac 不在同一网络，或 VPN/代理改写了请求。

### `403 AccessDenied` 或 presigned URL 失效

重新生成 `mc share download --expire 24h ...`。复制 URL 时要完整保留 `?` 后面的所有 query 参数，在 shell 中用单引号包住。

### 返回的是 HTML，不是 mp4

说明拿到的是错误页或控制台页面，不是对象下载地址。检查：

- MinIO API 端口一般是 `9000`，console 端口一般是 `9001`。
- public bucket URL 形如 `http://MAC_IP:9000/bucket/path/file.mp4`。
- presigned URL 要使用 `mc share download` 输出的 `Share:` 地址。

### `video_input_path or video_url is required`

确认请求 JSON 中字段名是 `video_url` 和 `mask_url`。如果把 URL 填进 `video_input_path`，服务端会按本地路径处理。

### `/progress` 里 `url` 是 `null`

说明自动上传没有生效或上传失败。按顺序检查：

- `SGLANG_CLOUD_STORAGE_TYPE=s3`、`SGLANG_S3_BUCKET_NAME`、`SGLANG_S3_ENDPOINT_URL`、access key 和 secret key 是否在启动 serve 前已经 export。
- 修改环境变量后必须重启 serve。
- `boto3` 是否安装在当前 `.venv` 里。
- serve 日志里是否有 `Upload failed`。
- GPU 机器是否能访问 `SGLANG_S3_ENDPOINT_URL`，例如 `http://127.0.0.1:19000`。
- 如果报 `Could not connect to the endpoint URL`，优先把 `SGLANG_S3_ENDPOINT_URL` 从 `127.0.0.1` 改成 serve 进程真正能访问的 MinIO API 地址，例如 `http://MAC_LAN_IP:19000` 或实际端口 `http://MAC_LAN_IP:9000`；如果使用本地端口转发，确认转发进程在整次推理期间一直存活。

### `videoedit_queue_full`

当前基线设置 `VIDEOEDIT_QUEUE_CAPACITY=1`，同一时间只跑一个 VideoEdit 任务。等当前任务完成，或者重启 serve 后再提交。

### 端口被占用

如果 `30000` 已被旧服务占用，先停止旧服务，或者把 serve 和请求里的端口一起改成新的端口。

### offload 很慢

这是预期现象。单卡 offload/layerwise offload 主要用于降低显存压力、先保证云端输入链路和推理能跑通；性能优化应在基线成功后再单独做。
