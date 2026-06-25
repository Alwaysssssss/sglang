# Vivid-VR FlowCut 本地模拟测试命令

本文档收口一条完整的本地模拟测试链路，目标是让你可以自己完成：

- 启动本地 S3/MinIO 模拟服务
- 启动 caption sidecar 服务
- 启动 Vivid-VR FlowCut 服务
- 提交带 `upscale` 的模拟请求
- 查看 callback、轮询 progress、检查上传结果

当前文档默认使用：

- Python 环境：`/home/zhiheng/sglang/.venv/bin/python`
- S3 模拟：`moto_server`
- FlowCut 服务：单卡 `fa eager`
- FlowCut 路由：`POST /v1/videos/repairs/flowcut`
- `upscale` 语义：官方原版 `/home/zhiheng/Viviv-VR-origin` 的输入预缩放，不是后处理超分

文档同时覆盖两条测试链路：

- 显式 replay：请求显式传 `caption_file_path + reference_video_path`
- 完整真实链路：先起 `caption sidecar`，再向 bridge 主服务发“不带 `caption_file_path` / `reference_video_path`”的真实请求

`upscale` 的请求语义：

- `0.0`：短边缩放到 `1024`
- `1.0`：输入分辨率保持不变
- 其他正数：按倍率做推理前输入 resize

## 1. 预设变量

先准备统一变量，后面的命令默认依赖这些环境变量：

```bash
cd /home/zhiheng/sglang

export PYTHONPATH=python
export LOG_DIR=/home/zhiheng/sglang/Vivid_Acceptance/logs
export OUTPUT_DIR=/home/zhiheng/sglang/Vivid_Acceptance/result_videos/service_benchmark
export INDICATOR_DIR=/home/zhiheng/sglang/Vivid_Acceptance/indicator

export INPUT_VIDEO_130F=/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.mp4
export CAPTION_FILE=/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.txt
export PROMPT_FILE=/home/zhiheng/Vivid-VR/input/720p/prompt.txt
export REFERENCE_VIDEO=/home/zhiheng/Vivid-VR/result/720p_long_up1_result_vivid_ori_20step/videos/test_video_long_960x720_130f.mp4

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}" "${INDICATOR_DIR}"
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

启动 `moto_server`：

```bash
tmux new-session -d -s vividvr_moto_s3 \
  'cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/logs && /home/zhiheng/sglang/.venv/bin/moto_server -H 127.0.0.1 -p 4566 2>&1 | tee Vivid_Acceptance/logs/vividvr_moto_s3_$(date -u +%Y%m%dT%H%M%SZ).log'
```

查看服务：

```bash
tmux attach -r -t vividvr_moto_s3
```

准备 S3 变量并创建 bucket：

```bash
export MOTO_S3_ENDPOINT=127.0.0.1:4566
export MOTO_S3_BUCKET=flowcut
export MOTO_S3_ACCESS_KEY=test
export MOTO_S3_SECRET_KEY=test

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

## 4. 启动 caption sidecar 服务

如果你要模拟“真实完整请求”，需要先启动 caption sidecar。

```bash
tmux new-session -d -s vividvr_caption_sidecar_mock \
  'cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/logs && CUDA_VISIBLE_DEVICES=0,1 /home/zhiheng/sglang/.venv-vividvr-caption/bin/python python/sglang/multimodal_gen/tools/run_vividvr_caption_sidecar.py --host 127.0.0.1 --port 31200 --parallel-workers 2 --worker-devices cuda:0,cuda:1 2>&1 | tee Vivid_Acceptance/logs/vividvr_caption_sidecar_mock_$(date -u +%Y%m%dT%H%M%SZ).log'
```

查看 caption sidecar：

```bash
tmux attach -r -t vividvr_caption_sidecar_mock
```

健康检查：

```bash
curl --noproxy '*' --silent --show-error --fail http://127.0.0.1:31200/health
```

## 5. 启动 Vivid-VR FlowCut 服务

### 5.1 显式 replay 服务

当前这条模拟链路使用单卡 `fa eager`，不启用 caption bridge，直接复用已有 `caption_file_path + reference_video_path`。

```bash
tmux new-session -d -s vividvr_flowcut_mock_service \
  'cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/logs && \
   export CUDA_VISIBLE_DEVICES=1 && \
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
     --port 31220 \
     --master-port 30220 \
     --scheduler-port 56220 \
     --strict-ports \
     --output-path /home/zhiheng/sglang/Vivid_Acceptance/result_videos/service_benchmark \
     --prompt-file-path /home/zhiheng/Vivid-VR/input/720p/prompt.txt \
     2>&1 | tee Vivid_Acceptance/logs/vividvr_flowcut_mock_service_$(date -u +%Y%m%dT%H%M%SZ).log'
```

查看服务：

```bash
tmux attach -r -t vividvr_flowcut_mock_service
```

健康检查：

```bash
curl --noproxy '*' --silent --show-error --fail http://127.0.0.1:31220/health
```

### 5.2 完整真实链路服务

这条命令会显式打开 caption bridge，服务在收到不带 `caption_file_path` 的请求后，会先调用 `caption sidecar` 生成 sidecar，再继续主推理。

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
     --output-path /home/zhiheng/sglang/Vivid_Acceptance/result_videos/service_benchmark \
     --prompt-file-path /home/zhiheng/Vivid-VR/input/720p/prompt.txt \
     --vividvr-caption-bridge \
     --vividvr-caption-sidecar-url http://127.0.0.1:31200 \
     --vividvr-caption-work-dir /home/zhiheng/sglang/Vivid_Acceptance/captions/service_sidecars \
     --vividvr-caption-sidecar-timeout 1800 \
     2>&1 | tee Vivid_Acceptance/logs/vividvr_flowcut_bridge_mock_service_$(date -u +%Y%m%dT%H%M%SZ).log'
```

查看 bridge 主服务：

```bash
tmux attach -r -t vividvr_flowcut_bridge_mock_service
```

健康检查：

```bash
curl --noproxy '*' --silent --show-error --fail http://127.0.0.1:31221/health
```

## 6. 启动本地 callback receiver

如果你想手动用 `curl` 提交请求，先起一个最小 callback receiver，把每条回调直接写到日志文件。

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

查看 callback receiver：

```bash
tmux attach -r -t vividvr_flowcut_callback_receiver
```

统一设置服务地址：

```bash
export MOTO_BASE_URL=http://127.0.0.1:31220
export BRIDGE_BASE_URL=http://127.0.0.1:31221
export CALLBACK_BASE_URL=http://127.0.0.1:39090
```

## 7. 手动提交 FlowCut 模拟请求

### 7.1 显式 replay 请求

下面这条命令会直接向服务发 `curl` 请求，带上：

- `caption_file_path`
- `reference_video_path`
- `upscale`
- `minioConfig`

```bash
export TASK_ID=vividvr-mock-$(date -u +%Y%m%dT%H%M%SZ)

NO_PROXY=* curl -sS -X POST "${MOTO_BASE_URL}/v1/videos/repairs/flowcut" \
  -H 'Content-Type: application/json' \
  --data-binary @- <<JSON | tee "${LOG_DIR}/${TASK_ID}.submit.log"
{
  "taskId": "${TASK_ID}",
  "timeout": -1,
  "callbackUrl": "${CALLBACK_BASE_URL}/tasks/${TASK_ID}/callback",
  "video_input_path": "${INPUT_VIDEO_130F}",
  "caption_file_path": "${CAPTION_FILE}",
  "reference_video_path": "${REFERENCE_VIDEO}",
  "num_inference_steps": 20,
  "seed": 42,
  "num_temporal_process_frames": 121,
  "upscale": 1.0,
  "output_path": "${OUTPUT_DIR}/${TASK_ID}.mp4",
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

如果你要测试官方原版 `upscale` 语义，可以只改这个字段：

- `1.0`：保持当前已验收基线
- `0.0`：短边缩到 `1024`
- `2.0`：输入先放大两倍再进推理

当前代码已经对齐官方原版的 `gen_height / gen_width` 规划语义，所以 `960x720_130f` 这条验收视频用 `upscale=0.0` 时，期望行为是：

- 请求被正常 accept
- progress 最终进入 `completed`
- callback 最后一条为 `status=succeeded`
- `output` 里返回 `result_url`

### 7.2 完整真实请求

这条请求用于模拟“客户端只给视频，服务端自动补 caption sidecar”的真实链路：

- 不传 `caption_file_path`
- 不传 `reference_video_path`
- 服务端先同步等待 caption bridge 生成 sidecar
- 然后再进入主推理

```bash
export TASK_ID=vividvr-bridge-mock-$(date -u +%Y%m%dT%H%M%SZ)

NO_PROXY=* curl -sS -X POST "${BRIDGE_BASE_URL}/v1/videos/repairs/flowcut" \
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

这条请求的关键点是：

- 它更接近外部真实客户端
- `submit` 阶段会更长，因为服务端要先等 caption sidecar
- bridge 生成的 sidecar 会写到：
  - `/home/zhiheng/sglang/Vivid_Acceptance/captions/service_sidecars/${TASK_ID}.txt`
- 如果把 `upscale` 改成 `0.0`，当前期望也是成功完成，而不是在模型内部报尺寸错误

## 8. 轮询 progress 和查看 callback

轮询任务进度：

```bash
# 显式 replay 链路
curl --noproxy '*' -X GET "${MOTO_BASE_URL}/v1/videos/${TASK_ID}/progress"

# 完整真实链路
curl --noproxy '*' -X GET "${BRIDGE_BASE_URL}/v1/videos/${TASK_ID}/progress"
```

持续轮询直到任务完成：

```bash
# 显式 replay 链路
while true; do
  curl --noproxy '*' --silent "${MOTO_BASE_URL}/v1/videos/${TASK_ID}/progress"
  echo
  sleep 10
done

# 完整真实链路
while true; do
  curl --noproxy '*' --silent "${BRIDGE_BASE_URL}/v1/videos/${TASK_ID}/progress"
  echo
  sleep 10
done
```

查看 callback 日志：

```bash
tail -f "${LOG_DIR}"/mock_callback_*.jsonl
```

成功时你应看到：

- progress 终态 `status=completed`
- progress 返回的 `url` 为 S3 模拟地址
- callback 最后一条 `status=succeeded`
- callback 的 `output` 只包含 `result_url` 和可选 `duration`

如果你走的是完整真实链路，也建议同时检查：

```bash
ls -l /home/zhiheng/sglang/Vivid_Acceptance/captions/service_sidecars/
```

## 9. 检查 S3 / MinIO 上传结果

检查模拟 S3 中是否已有上传对象：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python - <<'PY'
import boto3
import os

task_id = os.environ["TASK_ID"]
s3 = boto3.client(
    "s3",
    endpoint_url="http://127.0.0.1:4566",
    aws_access_key_id="test",
    aws_secret_access_key="test",
    region_name="us-east-1",
)
head = s3.head_object(Bucket="flowcut", Key=f"outputs/{task_id}.mp4")
print({"content_length": head["ContentLength"]})
PY
```

成功时返回的 `result_url` 形态应为：

```text
http://127.0.0.1:4566/flowcut/outputs/<TASK_ID>.mp4
```

## 10. 一键验收命令

如果你不想自己单独起 callback receiver 和手动轮询，可以直接使用仓库内的 acceptance runner。它会：

- 本地起 callback server
- 自动 submit
- 自动轮询 progress
- 自动等待最终 callback
- 校验成功 callback 的 `output` 契约

命令如下：

```bash
export TASK_ID=vividvr-mock-runner-$(date -u +%Y%m%dT%H%M%SZ)

NO_PROXY=* PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python \
  python/sglang/multimodal_gen/tools/run_flowcut_vividvr_service_acceptance.py \
  --base-url "${MOTO_BASE_URL}" \
  --task-id "${TASK_ID}" \
  --callback-log "${LOG_DIR}/${TASK_ID}.callback.jsonl" \
  --video-input-path "${INPUT_VIDEO_130F}" \
  --caption-file-path "${CAPTION_FILE}" \
  --reference-video-path "${REFERENCE_VIDEO}" \
  --output-path "${OUTPUT_DIR}/${TASK_ID}.mp4" \
  --perf-dump-path "${INDICATOR_DIR}/${TASK_ID}_perf.json" \
  --num-inference-steps 20 \
  --num-temporal-process-frames 121 \
  --upscale 1.0 \
  --seed 42
```

注意：

- 这个 runner 当前不会帮你传 `minioConfig`
- 所以它适合做服务协议、callback、progress、`upscale` 请求透传验收
- 如果你要验证 MinIO 上传，仍然要使用第 `7` 节那条手动 `curl` 请求

完整真实链路的 runner 验收命令如下。它会走 caption bridge，不传 `caption_file_path` 和 `reference_video_path`：

```bash
export TASK_ID=vividvr-bridge-runner-$(date -u +%Y%m%dT%H%M%SZ)

NO_PROXY=* PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python \
  python/sglang/multimodal_gen/tools/run_flowcut_vividvr_service_acceptance.py \
  --base-url "${BRIDGE_BASE_URL}" \
  --task-id "${TASK_ID}" \
  --callback-log "${LOG_DIR}/${TASK_ID}.callback.jsonl" \
  --video-input-path "${INPUT_VIDEO_130F}" \
  --output-path "${OUTPUT_DIR}/${TASK_ID}.mp4" \
  --perf-dump-path "${INDICATOR_DIR}/${TASK_ID}_perf.json" \
  --num-inference-steps 20 \
  --num-temporal-process-frames 121 \
  --upscale 1.0 \
  --seed 42 \
  --submit-timeout-s 2400 \
  --poll-timeout-s 2400
```

## 11. 最低验收标准

完成一次本地模拟测试，至少检查这几项：

- 服务 submit 返回 `{"code":0,"message":"ok"}`
- `GET /v1/videos/<task_id>/progress` 最终是：
  - `status=completed`
  - `progress=100`
  - `file_path=null`
  - `url=http://127.0.0.1:4566/flowcut/outputs/<TASK_ID>.mp4`
- callback 最后一条是：
  - `status=succeeded`
  - `progress=100`
  - `reason=succeeded`
  - `output` 中只包含 `result_url` 和可选 `duration`
- S3 模拟中存在 `outputs/<TASK_ID>.mp4`
- 本地 perf 文件已生成：
  - `${INDICATOR_DIR}/${TASK_ID}_perf.json`

如果走的是完整真实链路，还要额外检查：

- 主服务没有显式传 `caption_file_path`
- sidecar 目录下生成了 `${TASK_ID}.txt`
- caption sidecar 日志中没有失败记录

推荐额外检查：

```bash
tail -n 200 "${LOG_DIR}"/vividvr_caption_sidecar_mock_*.log
tail -n 200 "${LOG_DIR}"/vividvr_flowcut_bridge_mock_service_*.log
```

## 12. 常用停止命令

```bash
tmux kill-session -t vividvr_flowcut_mock_service
tmux kill-session -t vividvr_flowcut_bridge_mock_service
tmux kill-session -t vividvr_moto_s3
tmux kill-session -t vividvr_caption_sidecar_mock
tmux kill-session -t vividvr_flowcut_callback_receiver
```
