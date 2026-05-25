# VideoEdit 跑超过 81 帧的说明和命令

本文说明刚刚 `sp1_offload` 为什么只输出 80 帧，以及如何用同一个 serve 跑 100 帧或完整输入视频帧数。

## 1. 刚刚为什么只有 80 帧

本次输入视频和 mask 的实际帧数都是 `156`：

```bash
cd /home/tyx/workspace/zhouhao6/sglang
source .venv/bin/activate

python - <<'PY'
import cv2

for path in [
    "/home/tyx/workspace/zhouhao6/video_diffusers/pexel_test_data_0410/videos/15108907_3840_2160_50fps_short.mp4",
    "/home/tyx/workspace/zhouhao6/video_diffusers/pexel_test_data_0410/masks/15108907_3840_2160_50fps_No_bbox_mask.mp4",
]:
    cap = cv2.VideoCapture(path)
    print(path)
    print({
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
    })
    cap.release()
PY
```

刚刚的请求里显式写了：

```json
{
  "num_frames": 81,
  "infer_len": 81,
  "overlap": 0,
  "drop_reference_frame": true
}
```

含义是：只从输入里取前 `81` 帧做一次窗口推理；保存结果时因为 `drop_reference_frame=true`，会丢掉第 0 帧参考帧，所以最后输出 `80` 帧。

刚刚输出的视频 metadata：

```text
frames=80, width=1920, height=1088, fps=25.0
```

## 2. 是否可以超过 81 帧

可以，但不要把 `infer_len` 改成 100 或 156。

当前 SGLang VideoEdit 代码里的约束是：

- 单个 VideoEdit stage 固定要求 `infer_len=81`。
- 对外的 `num_frames` 可以大于 81。
- pipeline 会把长视频拆成多个 81 帧窗口。
- 尾窗口不足 81 帧时，会用反射帧补齐到 81 帧，但只提交真实输入帧。
- 如果 `drop_reference_frame=true`，最终输出帧数通常是 `num_frames - 1`。
- 如果希望输出帧数等于 `num_frames`，把 `drop_reference_frame` 改成 `false`。

也就是说：

```text
错误思路：infer_len=100
正确思路：num_frames=100, infer_len=81
跑全帧：num_frames=-1, infer_len=81
```

`num_frames=-1` 表示读取 video 和 mask 的全部可用帧，实际使用 `min(video_frames, mask_frames)`，避免 video/mask 长度不一致时越界。当前输入视频和 mask 都是 `156` 帧，所以 `num_frames=-1` 会解析为 `156`。

## 3. 窗口切分方式

窗口长度固定：

```text
infer_len = 81
stride = infer_len - overlap
```

示例：

```text
num_frames=81, overlap=0
  window 0: 0..80
  输出：drop_reference_frame=true 时为 80 帧

num_frames=100, overlap=0
  window 0: 0..80
  window 1: 81..99 + 反射补齐到 81 帧
  输出：drop_reference_frame=true 时为 99 帧

num_frames=156, overlap=0
  window 0: 0..80
  window 1: 81..155 + 反射补齐 6 帧
  输出：drop_reference_frame=true 时为 155 帧

num_frames=156, overlap=8
  window 0: 0..80
  window 1: 73..153
  window 2: 146..155 + 反射补齐到 81 帧
  输出：drop_reference_frame=true 时为 155 帧
```

`overlap=8` 会让窗口交界更平滑，但会增加窗口数量和耗时。当前这条 156 帧视频，`overlap=0` 是 2 个窗口，`overlap=8` 是 3 个窗口。

## 4. Serve 启动命令

如果刚刚的 serve 还在运行，可以不重启，直接跳到第 5 节提交新请求。

如果需要重新启动：

```bash
cd /home/tyx/workspace/zhouhao6/sglang
source .venv/bin/activate

export MODEL_PATH=/home/tyx/workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model
export TRANSFORMER_PATH=/home/tyx/workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model/transformer
export OUT_DIR=/home/tyx/workspace/zhouhao6/sglang/outputs
export FLASHINFER_WORKSPACE_BASE=/tmp/sglang-flashinfer
export XDG_CACHE_HOME=/tmp/sglang-cache
export VIDEOEDIT_QUEUE_CAPACITY=1

mkdir -p "$OUT_DIR" /tmp/sglang-videoedit-inputs /tmp/sglang-flashinfer /tmp/sglang-cache

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

## 5. 请求命令：跑 100 帧

这个请求会读取输入视频前 `100` 帧。由于 `drop_reference_frame=true`，预计输出 `99` 帧。

```bash
curl --noproxy '*' -s -X POST http://127.0.0.1:30000/v1/videos/repairs \
  -H 'Content-Type: application/json' \
  -d '{
    "task_id": "sp1_offload_100f",
    "prompt": "A close-up of an orange flower with a yellow center, remaining in focus against a blurred green grass background throughout the video.",
    "video_input_path": "/home/tyx/workspace/zhouhao6/video_diffusers/pexel_test_data_0410/videos/15108907_3840_2160_50fps_short.mp4",
    "mask_input_path": "/home/tyx/workspace/zhouhao6/video_diffusers/pexel_test_data_0410/masks/15108907_3840_2160_50fps_No_bbox_mask.mp4",
    "output_storage": "local",
    "output_path": "/home/tyx/workspace/zhouhao6/sglang/outputs/15108907_3840_2160_50fps_api_sp1_offload_100f.mp4",
    "num_frames": 100,
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
    "perf_dump_path": "/home/tyx/workspace/zhouhao6/sglang/outputs/videoedit_perf_api_sp1_offload_100f.json"
  }'
```

查询进度：

```bash
curl --noproxy '*' -s http://127.0.0.1:30000/v1/videos/sp1_offload_100f/progress
```

## 6. 请求命令：跑完整输入帧

这个请求会读取当前输入视频和 mask 的全部可用帧；当前两者都是 `156` 帧，所以实际等价于 `num_frames=156`。由于 `drop_reference_frame=true`，预计输出 `155` 帧。

```bash
curl --noproxy '*' -s -X POST http://127.0.0.1:30000/v1/videos/repairs \
  -H 'Content-Type: application/json' \
  -d '{
    "task_id": "sp1_offload_all_frames",
    "prompt": "A close-up of an orange flower with a yellow center, remaining in focus against a blurred green grass background throughout the video.",
    "video_input_path": "/home/tyx/workspace/zhouhao6/video_diffusers/pexel_test_data_0410/videos/15108907_3840_2160_50fps_short.mp4",
    "mask_input_path": "/home/tyx/workspace/zhouhao6/video_diffusers/pexel_test_data_0410/masks/15108907_3840_2160_50fps_No_bbox_mask.mp4",
    "output_storage": "local",
    "output_path": "/home/tyx/workspace/zhouhao6/sglang/outputs/15108907_3840_2160_50fps_api_sp1_offload_all_frames.mp4",
    "num_frames": -1,
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
    "perf_dump_path": "/home/tyx/workspace/zhouhao6/sglang/outputs/videoedit_perf_api_sp1_offload_all_frames.json"
  }'
```

查询进度：

```bash
curl --noproxy '*' -s http://127.0.0.1:30000/v1/videos/sp1_offload_all_frames/progress
```

## 7. 请求命令：完整 156 帧，带 overlap

如果想让窗口交界更平滑，可以用 `overlap=8`。代价是这条 156 帧输入会从 2 个窗口变成 3 个窗口，耗时明显增加。

```bash
curl --noproxy '*' -s -X POST http://127.0.0.1:30000/v1/videos/repairs \
  -H 'Content-Type: application/json' \
  -d '{
    "task_id": "sp1_offload_156f_overlap8",
    "prompt": "A close-up of an orange flower with a yellow center, remaining in focus against a blurred green grass background throughout the video.",
    "video_input_path": "/home/tyx/workspace/zhouhao6/video_diffusers/pexel_test_data_0410/videos/15108907_3840_2160_50fps_short.mp4",
    "mask_input_path": "/home/tyx/workspace/zhouhao6/video_diffusers/pexel_test_data_0410/masks/15108907_3840_2160_50fps_No_bbox_mask.mp4",
    "output_storage": "local",
    "output_path": "/home/tyx/workspace/zhouhao6/sglang/outputs/15108907_3840_2160_50fps_api_sp1_offload_156f_overlap8.mp4",
    "num_frames": 156,
    "infer_len": 81,
    "overlap": 8,
    "num_inference_steps": 20,
    "guidance_scale": 5.0,
    "dynamic_cfg": true,
    "dynamic_cfg_max_step": 15,
    "seed": 42,
    "dtype": "bf16",
    "enable_paste_back": true,
    "drop_reference_frame": true,
    "perf_dump_path": "/home/tyx/workspace/zhouhao6/sglang/outputs/videoedit_perf_api_sp1_offload_156f_overlap8.json"
  }'
```

查询进度：

```bash
curl --noproxy '*' -s http://127.0.0.1:30000/v1/videos/sp1_offload_156f_overlap8/progress
```

## 8. 如果希望输出不丢第 0 帧

把请求里的：

```json
"drop_reference_frame": true
```

改成：

```json
"drop_reference_frame": false
```

这样：

- `num_frames=100` 预计输出 `100` 帧。
- `num_frames=156` 预计输出 `156` 帧。

建议同时换新的 `task_id`、`output_path`、`perf_dump_path`，避免覆盖已有文件。

## 9. 输出检查命令

以 100 帧输出为例：

```bash
python - <<'PY'
import cv2
from pathlib import Path

path = Path("/home/tyx/workspace/zhouhao6/sglang/outputs/15108907_3840_2160_50fps_api_sp1_offload_100f.mp4")
cap = cv2.VideoCapture(str(path))
info = {
    "exists": path.exists(),
    "size_mb": round(path.stat().st_size / 1024 / 1024, 2) if path.exists() else None,
    "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    "fps": cap.get(cv2.CAP_PROP_FPS),
}
cap.release()
print(info)
PY
```

也可以检查 metadata 里的窗口：

```bash
cat /home/tyx/workspace/zhouhao6/sglang/outputs/15108907_3840_2160_50fps_api_sp1_offload_100f.videoedit.json
```

重点看：

```json
{
  "num_input_frames": 100,
  "window_specs": [
    {"window_index": 0, "start_index": 0, "end_index": 81, "reflected_count": 0},
    {"window_index": 1, "start_index": 81, "end_index": 100, "reflected_count": 62}
  ]
}
```

## 10. 耗时预期

刚刚单窗口 `81` 帧、`sp1_offload` 的实际关键耗时：

```text
warmup: 约 93s
denoising: 约 282.6s
decoding: 约 8.0s
整批日志时间: 约 405.2s
```

超过 81 帧时，每多一个窗口，denoising 基本就会再跑一次 81 帧窗口。因此：

- `num_frames=100, overlap=0`：2 个窗口，预计比 81 帧接近翻倍。
- `num_frames=156, overlap=0`：2 个窗口，预计也接近翻倍。
- `num_frames=156, overlap=8`：3 个窗口，预计明显更慢。

如果只是确认长视频链路，建议先跑 `num_frames=100, overlap=0`。如果要覆盖当前输入视频的完整长度，直接用 `num_frames=-1, overlap=0`。只有在窗口边界质量明显不连续时，再尝试 `overlap=8`。
