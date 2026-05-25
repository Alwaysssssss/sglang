# VideoEdit Stage 0 SP1 Offload Serve 复现流程

本文记录在当前机器上复现 VideoEdit-diffusers Stage 0 `sp1_offload` serve 跑通的完整流程。

实际已跑通的产物：

- 输出视频：`/home/tyx/workspace/zhouhao6/sglang/outputs/15108907_3840_2160_50fps_api_sp1_offload.mp4`
- perf JSON：`/home/tyx/workspace/zhouhao6/sglang/outputs/videoedit_perf_api_sp1_offload.json`
- 任务状态：`completed`

## 1. 进入仓库和环境

```bash
cd /home/tyx/workspace/zhouhao6/sglang
source .venv/bin/activate
```

确认 GPU 可见：

```bash
nvidia-smi
python - <<'PY'
import torch
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
PY
```

本次实际环境中，GPU 是 A100-SXM4-80GB。若在 Codex 沙箱内直接运行看不到 GPU，需要在能访问宿主 GPU 的终端里跑，或者允许命令在沙箱外执行。

## 2. 设置公共路径

```bash
export MODEL_PATH=/home/tyx/workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model
export TRANSFORMER_PATH=/home/tyx/workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model/transformer
export INPUT_VIDEO=/home/tyx/workspace/zhouhao6/video_diffusers/pexel_test_data_0410/videos/15108907_3840_2160_50fps_short.mp4
export INPUT_MASK=/home/tyx/workspace/zhouhao6/video_diffusers/pexel_test_data_0410/masks/15108907_3840_2160_50fps_No_bbox_mask.mp4
export OUT_DIR=/home/tyx/workspace/zhouhao6/sglang/outputs
export PROMPT="A close-up of an orange flower with a yellow center, remaining in focus against a blurred green grass background throughout the video."

mkdir -p "$OUT_DIR" /tmp/sglang-videoedit-inputs /tmp/sglang-flashinfer /tmp/sglang-cache
```

`FLASHINFER_WORKSPACE_BASE` 和 `XDG_CACHE_HOME` 指到 `/tmp` 是为了避免某些环境下 `flashinfer` 写 `/home/tyx/.cache` 报只读文件系统。

## 3. 启动 Stage 0 Serve

如果 `30000` 端口已有旧服务，先停止旧服务或换一个端口。

```bash
export FLASHINFER_WORKSPACE_BASE=/tmp/sglang-flashinfer
export XDG_CACHE_HOME=/tmp/sglang-cache
export VIDEOEDIT_QUEUE_CAPACITY=1

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
  --output-path "$OUT_DIR" \
  --input-save-path /tmp/sglang-videoedit-inputs \
  --transformer-path "$TRANSFORMER_PATH"
```

服务启动成功后，日志里应看到类似内容：

```text
Using pipeline from model_index.json: WanVideoEditPipeline
Worker 0: Scheduler loop started.
Uvicorn running on http://0.0.0.0:30000
```

本次加载模型时，serve 会先加载 `text_encoder`、`tokenizer`、`vae`、`transformer`、`scheduler`，然后启动 FastAPI。

## 4. 提交修复请求

另开一个终端，仍然进入同一个环境：

```bash
cd /home/tyx/workspace/zhouhao6/sglang
source .venv/bin/activate
```

发送请求：

```bash
curl --noproxy '*' -s -X POST http://127.0.0.1:30000/v1/videos/repairs \
  -H 'Content-Type: application/json' \
  -d '{
    "task_id": "sp1_offload",
    "prompt": "A close-up of an orange flower with a yellow center, remaining in focus against a blurred green grass background throughout the video.",
    "video_input_path": "/home/tyx/workspace/zhouhao6/video_diffusers/pexel_test_data_0410/videos/15108907_3840_2160_50fps_short.mp4",
    "mask_input_path": "/home/tyx/workspace/zhouhao6/video_diffusers/pexel_test_data_0410/masks/15108907_3840_2160_50fps_No_bbox_mask.mp4",
    "output_storage": "local",
    "output_path": "/home/tyx/workspace/zhouhao6/sglang/outputs/15108907_3840_2160_50fps_api_sp1_offload.mp4",
    "num_frames": 81,
    "infer_len": 81,
    "overlap": 0,
    "num_inference_steps": 20,
    "guidance_scale": 5.0,
    "dynamic_cfg": true,
    "dynamic_cfg_max_step": 15,
    "seed": 42,
    "dtype": "bf16",
    "enable_paste_back": true,
    "drop_reference_frame": true,
    "perf_dump_path": "/home/tyx/workspace/zhouhao6/sglang/outputs/videoedit_perf_api_sp1_offload.json"
  }'
```

成功提交后会返回 `queued`，例如：

```json
{
  "id": "sp1_offload",
  "status": "queued",
  "file_path": "/home/tyx/workspace/zhouhao6/sglang/outputs/15108907_3840_2160_50fps_api_sp1_offload.mp4"
}
```

如果要跑完整输入帧，把请求里的 `"num_frames": 81` 改成 `"num_frames": -1`。`-1` 会在 VideoEdit 入口解析成 `min(video_frames, mask_frames)`；当前这组 video 和 mask 都是 `156` 帧，因此实际会按 `156` 帧切窗口运行。

## 5. 查询进度

```bash
curl --noproxy '*' -s http://127.0.0.1:30000/v1/videos/sp1_offload/progress
```

运行中会看到：

```json
{
  "id": "sp1_offload",
  "status": "running",
  "progress": 1,
  "file_path": "/home/tyx/workspace/zhouhao6/sglang/outputs/15108907_3840_2160_50fps_api_sp1_offload.mp4",
  "url": null,
  "error": null
}
```

完成后会看到：

```json
{
  "id": "sp1_offload",
  "status": "completed",
  "progress": 100,
  "file_path": "/home/tyx/workspace/zhouhao6/sglang/outputs/15108907_3840_2160_50fps_api_sp1_offload.mp4",
  "url": null,
  "error": null
}
```

## 6. 本次实际耗时

服务端日志中，本次请求的关键耗时：

- warmup request：约 `93.14s`
- `VideoEditTextEncodingStage`：约 `1.06s`
- `VideoEditConditionEncodingStage`：约 `9.78s`
- `VideoEditDenoisingStage`：约 `282.58s`
- `VideoEditDecodingStage`：约 `8.03s`
- perf JSON 里的 `total_duration_ms`：约 `310315ms`
- 日志里的整批处理时间：约 `405.17s`

`sp1_offload` 开启了 CPU offload 和 layerwise offload，显存压力低但速度会慢。本次推理中 GPU0 大约使用二十多 GB 显存，denoising 前半段约 `16s/step`。

## 7. 检查输出

```bash
ls -lh \
  /home/tyx/workspace/zhouhao6/sglang/outputs/15108907_3840_2160_50fps_api_sp1_offload.mp4 \
  /home/tyx/workspace/zhouhao6/sglang/outputs/videoedit_perf_api_sp1_offload.json
```

检查视频 metadata：

```bash
python - <<'PY'
import cv2
from pathlib import Path

path = Path("/home/tyx/workspace/zhouhao6/sglang/outputs/15108907_3840_2160_50fps_api_sp1_offload.mp4")
cap = cv2.VideoCapture(str(path))
info = {
    "exists": path.exists(),
    "size_mb": round(path.stat().st_size / 1024 / 1024, 2),
    "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    "fps": cap.get(cv2.CAP_PROP_FPS),
}
cap.release()
print(info)
PY
```

本次实际输出：

```text
{'exists': True, 'size_mb': 0.45, 'frames': 80, 'width': 1920, 'height': 1088, 'fps': 25.0}
```

查看 perf JSON：

```bash
python - <<'PY'
import json

path = "/home/tyx/workspace/zhouhao6/sglang/outputs/videoedit_perf_api_sp1_offload.json"
data = json.load(open(path))
print(json.dumps(data, indent=2, ensure_ascii=False))
PY
```

## 8. 常见问题

### `nvidia-smi` 或 PyTorch 看不到 GPU

如果报：

```text
CUDA initialization: Unexpected error from cudaGetDeviceCount()
Can't initialize NVML
```

说明当前 shell 或沙箱看不到 NVIDIA driver。需要在能访问 GPU 的终端运行，或者允许命令在沙箱外执行。

### `flashinfer` 写 cache 报只读

如果报：

```text
OSError: [Errno 30] Read-only file system: '/home/tyx/.cache/flashinfer/...'
```

启动 serve 前设置：

```bash
export FLASHINFER_WORKSPACE_BASE=/tmp/sglang-flashinfer
export XDG_CACHE_HOME=/tmp/sglang-cache
mkdir -p /tmp/sglang-flashinfer /tmp/sglang-cache
```

### 端口被占用

默认 serve 使用 `30000` 端口。若端口被旧进程占用，可以停止旧服务，或者把 serve 命令和 curl URL 中的 `30000` 换成其他端口。

### 停止服务

在启动 serve 的终端里按 `Ctrl-C`。停止后如需重跑，重新执行第 3 节启动命令即可。
