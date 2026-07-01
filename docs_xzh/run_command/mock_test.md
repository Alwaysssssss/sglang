# Vivid-VR FlowCut 本地手动验收命令

本文档收口当前 `Vivid-VR FlowCut` 服务的本地手动验收方法，目标是让你自己完成下面几类检查：

- 手动启动本地 S3/MinIO 模拟服务
- 手动启动 callback receiver
- 手动启动 Vivid-VR FlowCut 服务
- 提交 caption bridge 请求并验证真实链路
- 手动取消任务并验证 `online_videoedit` 对齐后的取消语义
- 检查回调、S3 上传、输出扩展名继承、输入缓存清理

当前文档以当前分支的真实实现为准：

- 提交接口：`POST /v1/videos/repairs/flowcut`
- 查询接口：`GET /v1/videos/repairs/flowcut/{taskId}`
- 进度接口：`GET /v1/videos/repairs/flowcut/{taskId}/progress`
- 取消接口：`DELETE /v1/videos/repairs/flowcut/{taskId}`
- `callbackUrl` 当前必填
- `upscale` 是原版 Vivid-VR 的输入预缩放语义，不是后处理超分
- `minioConfig.endpoint` 当前应传不带 scheme 的 host:port，例如 `127.0.0.1:4566`

## 1. 统一变量

```bash
cd /home/zhiheng/sglang

export PYTHONPATH=python
export NO_PROXY=127.0.0.1,localhost
export LOG_DIR=/home/zhiheng/sglang/Vivid_Acceptance/logs
export OUTPUT_DIR=/home/zhiheng/sglang/Vivid_Acceptance/result_videos/service_benchmark
export INDICATOR_DIR=/home/zhiheng/sglang/Vivid_Acceptance/indicator/service_benchmark
export CAPTION_SIDECAR_DIR=/home/zhiheng/sglang/Vivid_Acceptance/captions/service_sidecars

export INPUT_VIDEO_130F=/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.mp4
export CAPTION_FILE=/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.txt
export PROMPT_FILE=/home/zhiheng/Vivid-VR/input/720p/prompt.txt

export MOTO_S3_ENDPOINT=127.0.0.1:4566
export MOTO_S3_BUCKET=flowcut
export MOTO_S3_ACCESS_KEY=test
export MOTO_S3_SECRET_KEY=test

export BRIDGE_BASE_URL=http://127.0.0.1:31221
export CALLBACK_BASE_URL=http://127.0.0.1:39090

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}" "${INDICATOR_DIR}" "${CAPTION_SIDECAR_DIR}"
```

## 2. 准备 caption sidecar 环境

这一步通常只需要做一次：

```bash
cd /home/zhiheng/sglang
bash python/sglang/multimodal_gen/tools/setup_vividvr_caption_env.sh
```

创建成功后，caption sidecar 独立环境应位于：

```bash
/home/zhiheng/sglang/.venv-vividvr-caption
```

## 3. 启动本地 S3 / MinIO 模拟服务

```bash
tmux new-session -d -s vividvr_moto_s3 \
  'cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/logs && /home/zhiheng/sglang/.venv/bin/moto_server -H 127.0.0.1 -p 4566 2>&1 | tee Vivid_Acceptance/logs/vividvr_moto_s3_$(date -u +%Y%m%dT%H%M%SZ).log'
```

查看服务：

```bash
tmux attach -r -t vividvr_moto_s3
```

创建 bucket：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python - <<'PY'
import boto3

s3 = boto3.client(
    "s3",
    endpoint_url="http://127.0.0.1:4566",
    aws_access_key_id="test",
    aws_secret_access_key="test",
    region_name="us-east-1",
)
try:
    s3.create_bucket(Bucket="flowcut")
except s3.exceptions.BucketAlreadyOwnedByYou:
    pass
print([b["Name"] for b in s3.list_buckets()["Buckets"]])
PY
```

预期结果：

- 输出中包含 `flowcut`

## 4. 启动 callback receiver

```bash
tmux new-session -d -s vividvr_flowcut_callback_receiver \
  'cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/logs && export CALLBACK_LOG=Vivid_Acceptance/logs/mock_callback_$(date -u +%Y%m%dT%H%M%SZ).jsonl && /home/zhiheng/sglang/.venv/bin/python - <<'"'"'PY'"'"'
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

log_path = os.environ["CALLBACK_LOG"]

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception as exc:
            payload = {"invalid_json": str(exc), "raw": body.decode("utf-8", "replace")}
        with open(log_path, "a", encoding="utf-8") as fout:
            fout.write(json.dumps(payload, ensure_ascii=False))
            fout.write("\n")
        response = b"{\"code\":0}"
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format, *args):
        return

server = ThreadingHTTPServer(("127.0.0.1", 39090), Handler)
print(json.dumps({"callback_url": "http://127.0.0.1:39090/tasks/mock/callback", "log_path": log_path}, ensure_ascii=False), flush=True)
server.serve_forever()
PY'
```

查看 receiver：

```bash
tmux attach -r -t vividvr_flowcut_callback_receiver
```

## 5. 启动服务

### 5.1 完整真实链路服务

这条链路显式启用 caption bridge。请求会先去 sidecar 生成 caption，再继续主推理。

先起 caption sidecar：

```bash
tmux new-session -d -s vividvr_caption_sidecar_mock \
  'cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/logs && CUDA_VISIBLE_DEVICES=0,1 /home/zhiheng/sglang/.venv-vividvr-caption/bin/python python/sglang/multimodal_gen/tools/run_vividvr_caption_sidecar.py --host 127.0.0.1 --port 31200 --parallel-workers 2 --worker-devices cuda:0,cuda:1 2>&1 | tee Vivid_Acceptance/logs/vividvr_caption_sidecar_mock_$(date -u +%Y%m%dT%H%M%SZ).log'
```

健康检查：

```bash
curl --noproxy '*' --silent --show-error --fail http://127.0.0.1:31200/health
```

再起 bridge 主服务：

```bash
tmux new-session -d -s vividvr_flowcut_bridge_mock_service \
  'cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/logs Vivid_Acceptance/captions/service_sidecars && \
   export CUDA_VISIBLE_DEVICES=1 && \
   export PYTHONUNBUFFERED=1 && \
   export PYTHONPATH=python && \
   export NO_PROXY=127.0.0.1,localhost && \
   export AWS_EC2_METADATA_DISABLED=true && \
   export SGLANG_FLOWCUT_PROGRESS_INTERVAL_SECONDS=5 && \
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
   /home/zhiheng/sglang/.venv/bin/sglang serve \
     --model-path /home/zhiheng/Vivid-VR/ckpts/CogVideoX1.5-5B \
     --model-id VividVR \
     --pipeline-class-name CogVideoXVividVRControlNetPipeline \
     --component-paths.vividvr /home/zhiheng/Vivid-VR/ckpts/Vivid-VR \
     --num-gpus 1 \
     --attention-backend fa \
     --host 127.0.0.1 \
     --port 31221 \
     --master-port 30221 \
     --scheduler-port 56221 \
     --strict-ports \
     --input-save-path "" \
     --output-path /home/zhiheng/sglang/Vivid_Acceptance/result_videos/service_benchmark \
     --prompt-file-path /home/zhiheng/Vivid-VR/input/720p/prompt.txt \
     --vividvr-caption-bridge \
     --vividvr-caption-sidecar-url http://127.0.0.1:31200 \
     --vividvr-caption-work-dir /home/zhiheng/sglang/Vivid_Acceptance/captions/service_sidecars \
     --vividvr-caption-sidecar-timeout 1800 \
     2>&1 | tee Vivid_Acceptance/logs/vividvr_flowcut_bridge_mock_service_$(date -u +%Y%m%dT%H%M%SZ).log'
```

健康检查：

```bash
curl --noproxy '*' --silent --show-error --fail "${BRIDGE_BASE_URL}/health"
```

## 6. caption bridge 成功链路

### 6.1 提交请求

这条请求由 bridge 主服务自动补 caption。`outputObjectKey` 故意写成不带扩展名，借此一起验证“输出扩展名继承输入格式”。

```bash
export TASK_ID=vividvr-bridge-$(date -u +%Y%m%dT%H%M%SZ)

curl --noproxy '*' -sS -X POST "${BRIDGE_BASE_URL}/v1/videos/repairs/flowcut" \
  -H 'Content-Type: application/json' \
  --data-binary @- <<JSON | tee "${LOG_DIR}/${TASK_ID}.submit.log"
{
  "taskId": "${TASK_ID}",
  "timeout": -1,
  "callbackUrl": "${CALLBACK_BASE_URL}/tasks/${TASK_ID}/callback",
  "video_input_path": "${INPUT_VIDEO_130F}",
  "num_inference_steps": 20,
  "seed": 42,
  "num_temporal_process_frames": 121,
  "upscale": 1.0,
  "output_path": "${OUTPUT_DIR}/${TASK_ID}.mp4",
  "outputObjectKey": "bridge-semantic-check/${TASK_ID}",
  "perf_dump_path": "${INDICATOR_DIR}/${TASK_ID}_perf.json",
  "minioConfig": {
    "endpoint": "${MOTO_S3_ENDPOINT}",
    "bucket_name": "${MOTO_S3_BUCKET}",
    "access_key": "${MOTO_S3_ACCESS_KEY}",
    "secret_key": "${MOTO_S3_SECRET_KEY}",
    "secure": false,
    "region": "us-east-1"
  }
}
JSON
```

预期结果：

- 提交响应仍然是 `{"code":0,"message":"ok"}`
- 因为服务要先补 caption，提交到出现首个 `caption_ready` 回调之间的间隔会更长

### 6.2 轮询与预期

```bash
curl --noproxy '*' --silent "${BRIDGE_BASE_URL}/v1/videos/repairs/flowcut/${TASK_ID}"
echo
curl --noproxy '*' --silent "${BRIDGE_BASE_URL}/v1/videos/repairs/flowcut/${TASK_ID}/progress"
echo
```

持续轮询：

```bash
while true; do
  curl --noproxy '*' --silent "${BRIDGE_BASE_URL}/v1/videos/repairs/flowcut/${TASK_ID}/progress"
  echo
  sleep 10
done
```

成功终态预期：

- `GET /repairs/flowcut/{taskId}` 返回：
  - `status = "completed"`
  - `url` 为 `http://127.0.0.1:4566/...`
  - `file_path = null`
  - `reason = null`
- `GET /repairs/flowcut/{taskId}/progress` 返回：
  - `status = "completed"`
  - `progress = 100`
  - `callback_status = "succeeded"`
- callback 中间阶段预期会依次出现：
  - `reason = "accepted"`
  - `reason = "input_ready"`
  - `reason = "caption_ready"`
  - `reason = "denoising"`
  - `reason = "uploading_result"`
  - 最后一条 `status = "succeeded"`

### 6.3 查看 callback

```bash
tail -f "${LOG_DIR}"/mock_callback_*.jsonl
```

预期最后一条 callback：

- `status = "succeeded"`
- `progress = 100`
- `reason = ""`
- `output` 是 JSON string，至少包含 `result_url`

### 6.4 检查 S3 对象

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python - <<'PY'
import boto3
import os

task_id = os.environ["TASK_ID"]
key = f"bridge-semantic-check/{task_id}.mp4"

s3 = boto3.client(
    "s3",
    endpoint_url="http://127.0.0.1:4566",
    aws_access_key_id="test",
    aws_secret_access_key="test",
    region_name="us-east-1",
)
head = s3.head_object(Bucket="flowcut", Key=key)
print({"key": key, "content_length": head["ContentLength"]})
PY
```

预期结果：

- `head_object` 成功
- key 形态是 `bridge-semantic-check/<TASK_ID>.mp4`
- 因为输入样例是 `.mp4`，服务会把未带后缀的 `outputObjectKey` 自动补成 `.mp4`

如果你换成 `.mov` 输入视频，预期 key 和最终 `result_url` 后缀都应变成 `.mov`。

### 6.5 下载结果文件

先从 bridge 详情接口取回 `result_url`：

```bash
export RESULT_URL=$(curl --noproxy '*' --silent "${BRIDGE_BASE_URL}/v1/videos/repairs/flowcut/${TASK_ID}" | /home/zhiheng/sglang/.venv/bin/python -c 'import json,sys; print(json.load(sys.stdin)["url"])')
echo "${RESULT_URL}"
```

预期结果：

- 输出为 `http://127.0.0.1:4566/flowcut/bridge-semantic-check/<TASK_ID>.mp4`

直接通过 `result_url` 下载：

```bash
curl --noproxy '*' --silent --show-error --fail -L \
  -o "${OUTPUT_DIR}/${TASK_ID}.bridge-downloaded.mp4" \
  "${RESULT_URL}"
ls -lh "${OUTPUT_DIR}/${TASK_ID}.bridge-downloaded.mp4"
```

如果你想绕过 `result_url`，也可以直接从 mock S3 key 下载：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python - <<'PY'
import boto3
import os

task_id = os.environ["TASK_ID"]
target = os.path.join(os.environ["OUTPUT_DIR"], f"{task_id}.bridge-downloaded-from-s3.mp4")
key = f"bridge-semantic-check/{task_id}.mp4"

s3 = boto3.client(
    "s3",
    endpoint_url="http://127.0.0.1:4566",
    aws_access_key_id="test",
    aws_secret_access_key="test",
    region_name="us-east-1",
)
s3.download_file("flowcut", key, target)
print({"downloaded_to": target, "key": key})
PY
```

预期结果：

- 本地出现 `${OUTPUT_DIR}/${TASK_ID}.bridge-downloaded.mp4`
- 或 `${OUTPUT_DIR}/${TASK_ID}.bridge-downloaded-from-s3.mp4`
- 文件大小大于 `0`

### 6.6 检查 bridge caption 与 request workdir

按上面的启动命令，bridge 服务显式配置了：

- `--input-save-path ""`
- `--vividvr-caption-work-dir /home/zhiheng/sglang/Vivid_Acceptance/captions/service_sidecars`

因此当前默认预期是：

- request workdir 会被清理
- bridge 自动生成的 caption sidecar 会保留在 `CAPTION_SIDECAR_DIR`

任务完成后检查：

```bash
find /home/zhiheng/sglang/Vivid_Acceptance/captions/service_sidecars -maxdepth 1 -type f -name "${TASK_ID}*" 2>/dev/null
find /tmp -maxdepth 2 -type d -name "${TASK_ID}" 2>/dev/null
```

预期结果：

- `CAPTION_SIDECAR_DIR` 下能看到 `${TASK_ID}.txt` 和对应 manifest
- request workdir 不应残留 `${TASK_ID}` 目录

如果你想验证“bridge 自动生成 caption 也随任务一起清理”，请重新启动 bridge 服务并去掉 `--vividvr-caption-work-dir ...`。那样 caption 会回落到 request workdir 下的 `outputs/caption_sidecars`，在临时 workdir 模式下会随任务一起删除。

## 7. 取消链路

### 7.1 提交一个待取消任务

```bash
export TASK_ID=vividvr-cancel-$(date -u +%Y%m%dT%H%M%SZ)

curl --noproxy '*' -sS -X POST "${BRIDGE_BASE_URL}/v1/videos/repairs/flowcut" \
  -H 'Content-Type: application/json' \
  --data-binary @- <<JSON | tee "${LOG_DIR}/${TASK_ID}.submit.log"
{
  "taskId": "${TASK_ID}",
  "timeout": -1,
  "callbackUrl": "${CALLBACK_BASE_URL}/tasks/${TASK_ID}/callback",
  "video_input_path": "${INPUT_VIDEO_130F}",
  "num_inference_steps": 50,
  "seed": 42,
  "num_temporal_process_frames": 121,
  "upscale": 1.0,
  "output_path": "${OUTPUT_DIR}/${TASK_ID}.mp4",
  "outputObjectKey": "cancel-semantic-check/${TASK_ID}",
  "perf_dump_path": "${INDICATOR_DIR}/${TASK_ID}_perf.json",
  "minioConfig": {
    "endpoint": "${MOTO_S3_ENDPOINT}",
    "bucket_name": "${MOTO_S3_BUCKET}",
    "access_key": "${MOTO_S3_ACCESS_KEY}",
    "secret_key": "${MOTO_S3_SECRET_KEY}",
    "secure": false,
    "region": "us-east-1"
  }
}
JSON
```

先轮询到任务进入运行中：

```bash
while true; do
  curl --noproxy '*' --silent "${BRIDGE_BASE_URL}/v1/videos/repairs/flowcut/${TASK_ID}/progress"
  echo
  sleep 5
done
```

当你看到：

- `status = "running"`
- `progress > 0`

就可以发取消请求。

### 7.2 取消任务

```bash
curl --noproxy '*' --silent -X DELETE "${BRIDGE_BASE_URL}/v1/videos/repairs/flowcut/${TASK_ID}"
echo
```

取消返回预期：

- `status = "failed"`
- `reason = "Request timed out."`
- `error.message = "Request timed out."`

再查详情和进度：

```bash
curl --noproxy '*' --silent "${BRIDGE_BASE_URL}/v1/videos/repairs/flowcut/${TASK_ID}"
echo
curl --noproxy '*' --silent "${BRIDGE_BASE_URL}/v1/videos/repairs/flowcut/${TASK_ID}/progress"
echo
```

终态预期：

- `status = "failed"`
- `reason = "Request timed out."`
- `callback_status = "succeeded"`
- 不应再继续上传成功结果

### 7.3 检查 callback 和 S3

查看 callback：

```bash
tail -f "${LOG_DIR}"/mock_callback_*.jsonl
```

最后一条预期：

- `status = "failed"`
- `reason = "Request timed out."`
- `output = ""`

确认取消后没有结果对象：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python - <<'PY'
import boto3
import botocore
import os

task_id = os.environ["TASK_ID"]
key = f"cancel-semantic-check/{task_id}.mp4"

s3 = boto3.client(
    "s3",
    endpoint_url="http://127.0.0.1:4566",
    aws_access_key_id="test",
    aws_secret_access_key="test",
    region_name="us-east-1",
)
try:
    s3.head_object(Bucket="flowcut", Key=key)
    print({"unexpected_object": key})
except botocore.exceptions.ClientError as exc:
    print({"missing_as_expected": key, "error": exc.response["Error"]["Code"]})
PY
```

预期结果：

- 对象不存在

如果 bridge 服务是按本文命令启动的，再检查：

```bash
find "${CAPTION_SIDECAR_DIR}" -maxdepth 1 -type f -name "${TASK_ID}*" 2>/dev/null
find /tmp -maxdepth 2 -type d -name "${TASK_ID}" 2>/dev/null
```

预期结果：

- request workdir 应被清理
- `CAPTION_SIDECAR_DIR` 下的 `${TASK_ID}.txt` 和 manifest 仍会保留，因为当前 bridge caption 输出目录是显式持久目录

## 8. 最低验收标准

至少完成下面 2 组检查：

1. bridge 成功链路
- 中间阶段能看到 `caption_ready`
- 终态 `completed`
- 按本文默认命令启动时，自动生成的 caption 会保留在 `CAPTION_SIDECAR_DIR`
- S3 对象存在
- request workdir 已清理

2. 取消链路
- `DELETE` 之后终态是 `failed`
- `reason = Request timed out.`
- callback 最后一条是 `failed`
- S3 不存在结果对象

## 9. 停止服务

```bash
tmux kill-session -t vividvr_flowcut_bridge_mock_service
tmux kill-session -t vividvr_caption_sidecar_mock
tmux kill-session -t vividvr_flowcut_callback_receiver
tmux kill-session -t vividvr_moto_s3
```
