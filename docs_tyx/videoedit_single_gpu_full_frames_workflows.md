# VideoEdit 单卡全帧运行流程

本文只写单卡运行当前这条 VideoEdit 样例视频的流程，包括：

1. 正常单卡 no-offload 全帧运行；
2. 单卡开启 offload 的保守全帧运行；
3. 在单卡请求里开启 TeaCache；
4. 单卡开启 `torch.compile`。

当前输入视频和 mask 都是 `156` 帧。本文所有全帧请求都显式写：

```json
"num_frames": 156
```

当前代码不支持 `"num_frames": -1` 表示全帧，所以不要在请求里写 `-1`。

输出目录统一使用：

```text
/home/tyx/workspace/zhouhao6/sglang/output_tyx
```

## 0. 先看结论

如果只是想先稳定把全帧结果跑出来，优先用第 3 节的单卡 offload 流程。这个配置已经在 GPU0 上跑通过：

```text
task_id: sp1_offload_156f_all_gpu0
输出帧数: 156
正式生成耗时: 约 723.29s
```

正常 no-offload 和 `torch.compile` 都需要更大的 GPU0 空闲显存。当前 GPU0 有约 26GB 驱动残留显存时，全帧 no-offload 很容易在 decode 阶段 OOM。

TeaCache 是请求级开关，不是 serve 启动参数。当前 VideoEdit 接口能接收 `"enable_teacache": true`，但 VideoEdit 采样参数里没有模型专用 `teacache_params`，所以它可能不会真正加速。跑 TeaCache 时必须用 perf 对比和日志确认，不能只看请求有没有成功。

## 1. 公共准备

进入仓库并激活环境：

```bash
cd /home/tyx/workspace/zhouhao6/sglang
source .venv/bin/activate
```

设置公共变量：

```bash
export MODEL_PATH=/home/tyx/workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model
export TRANSFORMER_PATH=/home/tyx/workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model/transformer
export INPUT_VIDEO=/home/tyx/workspace/zhouhao6/video_diffusers/pexel_test_data_0410/videos/15108907_3840_2160_50fps_short.mp4
export INPUT_MASK=/home/tyx/workspace/zhouhao6/video_diffusers/pexel_test_data_0410/masks/15108907_3840_2160_50fps_No_bbox_mask.mp4
export OUT_DIR=/home/tyx/workspace/zhouhao6/sglang/output_tyx
export FLASHINFER_WORKSPACE_BASE=/tmp/sglang-flashinfer
export XDG_CACHE_HOME=/tmp/sglang-cache
export VIDEOEDIT_QUEUE_CAPACITY=1

mkdir -p "$OUT_DIR" /tmp/sglang-videoedit-inputs /tmp/sglang-flashinfer /tmp/sglang-cache
```

确认输入视频和 mask 都是 `156` 帧：

```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=nb_frames,width,height,r_frame_rate,duration \
  -of json "$INPUT_VIDEO"

ffprobe -v error -select_streams v:0 \
  -show_entries stream=nb_frames,width,height,r_frame_rate,duration \
  -of json "$INPUT_MASK"
```

预期结果：

```text
nb_frames = 156
width = 1920
height = 1080
r_frame_rate = 25/1
duration = 6.240000
```

检查 GPU：

```bash
nvidia-smi
```

如果只想用 GPU0，后面的 serve 命令都写：

```bash
export CUDA_VISIBLE_DEVICES=0
```

如果 GPU0 有明显残留显存，例如已经占了 `26000MiB` 以上，正常 no-offload 和 compile 配置更容易 OOM。保守 offload 配置仍然更可能跑通。

## 2. 终端分工

### serve 终端

运行 `sglang serve ...` 的终端会一直被占用，这是正常的。不要在这个终端继续输入 `curl`。

### 查询终端

健康检查、提交请求、查进度都另开一个终端。

所有本机 `curl` 都加：

```bash
--noproxy '*'
```

否则当前机器可能走代理，访问 `127.0.0.1:30000` 时卡住。

### 停止旧服务

如果旧 serve 终端还在，直接按：

```text
Ctrl-C
```

检查 `30000` 端口：

```bash
netstat -ltnp 2>/dev/null | grep ':30000' || true
```

没有输出才说明 `30000` 没有旧服务。

## 3. 单卡正常跑：no-offload + FA，全帧 156

这个配置速度相对快，但显存压力最大。只有 GPU0 接近空卡时才建议先跑它。

### 3.1 启动 serve

在 serve 终端执行：

```bash
cd /home/tyx/workspace/zhouhao6/sglang
source .venv/bin/activate

export MODEL_PATH=/home/tyx/workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model
export TRANSFORMER_PATH=/home/tyx/workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model/transformer
export OUT_DIR=/home/tyx/workspace/zhouhao6/sglang/output_tyx
export FLASHINFER_WORKSPACE_BASE=/tmp/sglang-flashinfer
export XDG_CACHE_HOME=/tmp/sglang-cache
export VIDEOEDIT_QUEUE_CAPACITY=1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
  --dit-cpu-offload false \
  --dit-layerwise-offload false \
  --text-encoder-cpu-offload false \
  --image-encoder-cpu-offload false \
  --vae-cpu-offload false \
  --attention-backend fa \
  --warmup true \
  --warmup-steps 1 \
  --output-path "$OUT_DIR" \
  --input-save-path /tmp/sglang-videoedit-inputs \
  --transformer-path "$TRANSFORMER_PATH"
```

serve 终端看到下面这行，说明服务启动成功：

```text
Uvicorn running on http://0.0.0.0:30000
```

### 3.2 健康检查

另开查询终端：

```bash
cd /home/tyx/workspace/zhouhao6/sglang
source .venv/bin/activate

curl --noproxy '*' -s http://127.0.0.1:30000/health
```

正常返回：

```json
{"status":"ok"}
```

### 3.3 提交全帧请求

```bash
curl --noproxy '*' -s -X POST http://127.0.0.1:30000/v1/videos/repairs \
  -H 'Content-Type: application/json' \
  -d '{
    "task_id": "sp1_no_offload_fa_156f_all_gpu0",
    "prompt": "A close-up of an orange flower with a yellow center, remaining in focus against a blurred green grass background throughout the video.",
    "video_input_path": "/home/tyx/workspace/zhouhao6/video_diffusers/pexel_test_data_0410/videos/15108907_3840_2160_50fps_short.mp4",
    "mask_input_path": "/home/tyx/workspace/zhouhao6/video_diffusers/pexel_test_data_0410/masks/15108907_3840_2160_50fps_No_bbox_mask.mp4",
    "output_storage": "local",
    "output_path": "/home/tyx/workspace/zhouhao6/sglang/output_tyx/15108907_3840_2160_50fps_api_sp1_no_offload_fa_156f_all_gpu0.mp4",
    "num_frames": 156,
    "infer_len": 81,
    "overlap": 0,
    "num_inference_steps": 20,
    "guidance_scale": 5.0,
    "dynamic_cfg": true,
    "dynamic_cfg_max_step": 15,
    "seed": 42,
    "dtype": "bf16",
    "enable_paste_back": true,
    "drop_reference_frame": false,
    "perf_dump_path": "/home/tyx/workspace/zhouhao6/sglang/output_tyx/videoedit_perf_api_sp1_no_offload_fa_156f_all_gpu0.json"
  }'
```

查进度：

```bash
curl --noproxy '*' -s http://127.0.0.1:30000/v1/videos/sp1_no_offload_fa_156f_all_gpu0/progress
```

如果这里 OOM，先不要继续 TeaCache 或 compile，直接切到第 4 节 offload。

## 4. 单卡保守跑：offload + layerwise offload，全帧 156

这是当前最稳的单卡全帧方案。它已经在 GPU0 上跑通。

### 4.1 启动 serve

在 serve 终端执行：

```bash
cd /home/tyx/workspace/zhouhao6/sglang
source .venv/bin/activate

export MODEL_PATH=/home/tyx/workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model
export TRANSFORMER_PATH=/home/tyx/workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model/transformer
export OUT_DIR=/home/tyx/workspace/zhouhao6/sglang/output_tyx
export FLASHINFER_WORKSPACE_BASE=/tmp/sglang-flashinfer
export XDG_CACHE_HOME=/tmp/sglang-cache
export VIDEOEDIT_QUEUE_CAPACITY=1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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

确认日志里有：

```text
"num_gpus": 1
"dit_cpu_offload": true
"dit_layerwise_offload": true
"text_encoder_cpu_offload": true
"vae_cpu_offload": true
```

以及：

```text
LayerwiseOffloadManager initialized
```

### 4.2 提交全帧请求

查询终端执行：

```bash
curl --noproxy '*' -s -X POST http://127.0.0.1:30000/v1/videos/repairs \
  -H 'Content-Type: application/json' \
  -d '{
    "task_id": "sp1_offload_156f_all_gpu0",
    "prompt": "A close-up of an orange flower with a yellow center, remaining in focus against a blurred green grass background throughout the video.",
    "video_input_path": "/home/tyx/workspace/zhouhao6/video_diffusers/pexel_test_data_0410/videos/15108907_3840_2160_50fps_short.mp4",
    "mask_input_path": "/home/tyx/workspace/zhouhao6/video_diffusers/pexel_test_data_0410/masks/15108907_3840_2160_50fps_No_bbox_mask.mp4",
    "output_storage": "local",
    "output_path": "/home/tyx/workspace/zhouhao6/sglang/output_tyx/15108907_3840_2160_50fps_api_sp1_offload_156f_all_gpu0.mp4",
    "num_frames": 156,
    "infer_len": 81,
    "overlap": 0,
    "num_inference_steps": 20,
    "guidance_scale": 5.0,
    "dynamic_cfg": true,
    "dynamic_cfg_max_step": 15,
    "seed": 42,
    "dtype": "bf16",
    "enable_paste_back": true,
    "drop_reference_frame": false,
    "perf_dump_path": "/home/tyx/workspace/zhouhao6/sglang/output_tyx/videoedit_perf_api_sp1_offload_156f_all_gpu0.json"
  }'
```

查进度：

```bash
curl --noproxy '*' -s http://127.0.0.1:30000/v1/videos/sp1_offload_156f_all_gpu0/progress
```

当前实测结果：

```text
warmup: 约 92.23s
正式生成: 约 723.29s
输出帧数: 156
两个 denoising window: 约 284.15s + 284.05s
```

## 5. 单卡 TeaCache 流程

TeaCache 是请求级开关，不需要重新启动一个特殊 serve。你可以在第 3 节 no-offload serve 或第 4 节 offload serve 上提交带 TeaCache 的请求。

重要限制：

- 当前 VideoEdit 接口接受 `"enable_teacache": true`。
- 但当前 VideoEdit 采样参数没有模型专用 `teacache_params`，代码里 TeaCache 需要 `teacache_params` 才会真正工作。
- 所以这个流程用于验证当前分支是否已接入 TeaCache；如果 perf 没有变化、日志没有 TeaCache skip 信息，就不能认为 TeaCache 生效。

### 5.1 推荐验证方式

先跑一个不带 TeaCache 的同配置请求，再跑一个只多加 `"enable_teacache": true` 的请求。其他参数必须完全相同。

对比：

```text
total_duration_ms
VideoEditDenoisingStage
输出质量
serve 日志里是否有 TeaCache 相关信息
```

### 5.2 在 offload serve 上提交 TeaCache 请求

先按第 4 节启动 offload serve，然后查询终端执行：

```bash
curl --noproxy '*' -s -X POST http://127.0.0.1:30000/v1/videos/repairs \
  -H 'Content-Type: application/json' \
  -d '{
    "task_id": "sp1_offload_teacache_156f_all_gpu0",
    "prompt": "A close-up of an orange flower with a yellow center, remaining in focus against a blurred green grass background throughout the video.",
    "video_input_path": "/home/tyx/workspace/zhouhao6/video_diffusers/pexel_test_data_0410/videos/15108907_3840_2160_50fps_short.mp4",
    "mask_input_path": "/home/tyx/workspace/zhouhao6/video_diffusers/pexel_test_data_0410/masks/15108907_3840_2160_50fps_No_bbox_mask.mp4",
    "output_storage": "local",
    "output_path": "/home/tyx/workspace/zhouhao6/sglang/output_tyx/15108907_3840_2160_50fps_api_sp1_offload_teacache_156f_all_gpu0.mp4",
    "num_frames": 156,
    "infer_len": 81,
    "overlap": 0,
    "num_inference_steps": 20,
    "guidance_scale": 5.0,
    "dynamic_cfg": true,
    "dynamic_cfg_max_step": 15,
    "seed": 42,
    "dtype": "bf16",
    "enable_paste_back": true,
    "drop_reference_frame": false,
    "enable_teacache": true,
    "perf_dump_path": "/home/tyx/workspace/zhouhao6/sglang/output_tyx/videoedit_perf_api_sp1_offload_teacache_156f_all_gpu0.json"
  }'
```

查进度：

```bash
curl --noproxy '*' -s http://127.0.0.1:30000/v1/videos/sp1_offload_teacache_156f_all_gpu0/progress
```

### 5.3 判断 TeaCache 是否有效

跑完后对比 perf：

```bash
python - <<'PY'
import json
from pathlib import Path

base = Path("output_tyx/videoedit_perf_api_sp1_offload_156f_all_gpu0.json")
tea = Path("output_tyx/videoedit_perf_api_sp1_offload_teacache_156f_all_gpu0.json")

for p in [base, tea]:
    if not p.exists():
        print(f"missing: {p}")
        continue
    data = json.load(open(p))
    denoise = [s for s in data["steps"] if s["name"] == "VideoEditDenoisingStage"]
    print({
        "file": str(p),
        "total_duration_ms": data.get("total_duration_ms"),
        "denoising_ms": denoise[0]["duration_ms"] if denoise else None,
    })
PY
```

如果 TeaCache 版本的 denoising 时间没有明显下降，或者 serve 日志没有 TeaCache skip/decision 相关信息，就把本轮记录为：

```text
enable_teacache accepted by API, but no confirmed speedup for VideoEdit in this branch.
```

## 6. 单卡 torch.compile 流程

`torch.compile` 是 serve 启动级配置，不是请求级开关。第一次请求会包含编译成本，所以必须发两次同 shape 请求：

1. 第一次：compile 预热；
2. 第二次：正式记录性能。

注意：

- `torch.compile` 可能增加显存压力。
- 当前 GPU0 有 26GB 残留显存时，不建议直接跑全帧 compile。
- 推荐只在 no-offload 全帧已经能稳定跑通后，再试 compile。
- 不要和 TeaCache 同时评估；一次只改一个变量。

### 6.1 启动 compile serve

在 serve 终端执行：

```bash
cd /home/tyx/workspace/zhouhao6/sglang
source .venv/bin/activate

export MODEL_PATH=/home/tyx/workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model
export TRANSFORMER_PATH=/home/tyx/workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model/transformer
export OUT_DIR=/home/tyx/workspace/zhouhao6/sglang/output_tyx
export FLASHINFER_WORKSPACE_BASE=/tmp/sglang-flashinfer
export XDG_CACHE_HOME=/tmp/sglang-cache
export VIDEOEDIT_QUEUE_CAPACITY=1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SGLANG_TORCH_COMPILE_MODE=max-autotune-no-cudagraphs

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
  --dit-cpu-offload false \
  --dit-layerwise-offload false \
  --text-encoder-cpu-offload false \
  --image-encoder-cpu-offload false \
  --vae-cpu-offload false \
  --attention-backend fa \
  --enable-torch-compile true \
  --warmup true \
  --warmup-steps 1 \
  --output-path "$OUT_DIR" \
  --input-save-path /tmp/sglang-videoedit-inputs \
  --transformer-path "$TRANSFORMER_PATH"
```

确认日志里有：

```text
"enable_torch_compile": true
"attention_backend": "fa"
"dit_cpu_offload": false
"dit_layerwise_offload": false
```

### 6.2 第一次请求：compile 预热

```bash
curl --noproxy '*' -s -X POST http://127.0.0.1:30000/v1/videos/repairs \
  -H 'Content-Type: application/json' \
  -d '{
    "task_id": "sp1_no_offload_compile_fa_156f_all_gpu0_warmup",
    "prompt": "A close-up of an orange flower with a yellow center, remaining in focus against a blurred green grass background throughout the video.",
    "video_input_path": "/home/tyx/workspace/zhouhao6/video_diffusers/pexel_test_data_0410/videos/15108907_3840_2160_50fps_short.mp4",
    "mask_input_path": "/home/tyx/workspace/zhouhao6/video_diffusers/pexel_test_data_0410/masks/15108907_3840_2160_50fps_No_bbox_mask.mp4",
    "output_storage": "local",
    "output_path": "/home/tyx/workspace/zhouhao6/sglang/output_tyx/15108907_3840_2160_50fps_api_sp1_no_offload_compile_fa_156f_all_gpu0_warmup.mp4",
    "num_frames": 156,
    "infer_len": 81,
    "overlap": 0,
    "num_inference_steps": 20,
    "guidance_scale": 5.0,
    "dynamic_cfg": true,
    "dynamic_cfg_max_step": 15,
    "seed": 42,
    "dtype": "bf16",
    "enable_paste_back": true,
    "drop_reference_frame": false,
    "perf_dump_path": "/home/tyx/workspace/zhouhao6/sglang/output_tyx/videoedit_perf_api_sp1_no_offload_compile_fa_156f_all_gpu0_warmup.json"
  }'
```

查进度：

```bash
curl --noproxy '*' -s http://127.0.0.1:30000/v1/videos/sp1_no_offload_compile_fa_156f_all_gpu0_warmup/progress
```

### 6.3 第二次请求：正式记录

第一次完成后，提交第二次：

```bash
curl --noproxy '*' -s -X POST http://127.0.0.1:30000/v1/videos/repairs \
  -H 'Content-Type: application/json' \
  -d '{
    "task_id": "sp1_no_offload_compile_fa_156f_all_gpu0",
    "prompt": "A close-up of an orange flower with a yellow center, remaining in focus against a blurred green grass background throughout the video.",
    "video_input_path": "/home/tyx/workspace/zhouhao6/video_diffusers/pexel_test_data_0410/videos/15108907_3840_2160_50fps_short.mp4",
    "mask_input_path": "/home/tyx/workspace/zhouhao6/video_diffusers/pexel_test_data_0410/masks/15108907_3840_2160_50fps_No_bbox_mask.mp4",
    "output_storage": "local",
    "output_path": "/home/tyx/workspace/zhouhao6/sglang/output_tyx/15108907_3840_2160_50fps_api_sp1_no_offload_compile_fa_156f_all_gpu0.mp4",
    "num_frames": 156,
    "infer_len": 81,
    "overlap": 0,
    "num_inference_steps": 20,
    "guidance_scale": 5.0,
    "dynamic_cfg": true,
    "dynamic_cfg_max_step": 15,
    "seed": 42,
    "dtype": "bf16",
    "enable_paste_back": true,
    "drop_reference_frame": false,
    "perf_dump_path": "/home/tyx/workspace/zhouhao6/sglang/output_tyx/videoedit_perf_api_sp1_no_offload_compile_fa_156f_all_gpu0.json"
  }'
```

查进度：

```bash
curl --noproxy '*' -s http://127.0.0.1:30000/v1/videos/sp1_no_offload_compile_fa_156f_all_gpu0/progress
```

## 7. 输出检查

检查输出文件：

```bash
ls -lh output_tyx/*156f_all_gpu0*
```

检查帧数：

```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=nb_frames,width,height,r_frame_rate,duration \
  -of json output_tyx/15108907_3840_2160_50fps_api_sp1_offload_156f_all_gpu0.mp4
```

`drop_reference_frame=false` 时，预期：

```text
nb_frames = 156
duration = 6.240000
```

如果把 `drop_reference_frame` 改成 `true`，最终输出会变成 `155` 帧。

## 8. 反射补帧检查

全帧 `156`、`infer_len=81`、`overlap=0` 会分两个窗口：

```json
[
  {
    "window_index": 0,
    "start_index": 0,
    "end_index": 81,
    "reflected_count": 0
  },
  {
    "window_index": 1,
    "start_index": 81,
    "end_index": 156,
    "reflected_count": 6
  }
]
```

第二个窗口只有真实帧 `81..155`，一共 `75` 帧，不够 `81`，所以会从尾部按时间反射补 `6` 帧。这个“反射”不是左右翻转画面，而是时间维度补帧，只用于模型输入，不会额外写进输出视频。

检查 metadata：

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("output_tyx/15108907_3840_2160_50fps_api_sp1_offload_156f_all_gpu0.videoedit.json")
meta = json.load(open(path))
print(json.dumps({
    "num_input_frames": meta.get("num_input_frames"),
    "drop_reference_frame": meta.get("drop_reference_frame"),
    "window_specs": meta.get("window_specs"),
}, indent=2, ensure_ascii=False))
PY
```

如果第二个窗口是 `"reflected_count": 6`，说明用了尾部反射补帧。

## 9. perf 对比

查看单个 perf：

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("output_tyx/videoedit_perf_api_sp1_offload_156f_all_gpu0.json")
data = json.load(open(path))
print("total_duration_ms:", data.get("total_duration_ms"))
for step in data.get("steps", []):
    print(step["name"], step["duration_ms"])
print("memory:", json.dumps(data.get("memory_checkpoints"), indent=2))
PY
```

对比多个配置：

```bash
python - <<'PY'
import json
from pathlib import Path

paths = [
    Path("output_tyx/videoedit_perf_api_sp1_no_offload_fa_156f_all_gpu0.json"),
    Path("output_tyx/videoedit_perf_api_sp1_offload_156f_all_gpu0.json"),
    Path("output_tyx/videoedit_perf_api_sp1_offload_teacache_156f_all_gpu0.json"),
    Path("output_tyx/videoedit_perf_api_sp1_no_offload_compile_fa_156f_all_gpu0.json"),
]

for path in paths:
    if not path.exists():
        continue
    data = json.load(open(path))
    denoise = next((s for s in data["steps"] if s["name"] == "VideoEditDenoisingStage"), None)
    decode = next((s for s in data["steps"] if s["name"] == "VideoEditDecodingStage"), None)
    print({
        "file": path.name,
        "total_ms": round(data.get("total_duration_ms", 0), 2),
        "denoise_ms": round(denoise["duration_ms"], 2) if denoise else None,
        "decode_ms": round(decode["duration_ms"], 2) if decode else None,
    })
PY
```

## 10. 推荐执行顺序

当前机器建议按这个顺序：

```text
1. sp1_offload_156f_all_gpu0
2. sp1_offload_teacache_156f_all_gpu0
3. 等 GPU0 残留显存清掉后，再试 sp1_no_offload_fa_156f_all_gpu0
4. no-offload 跑通后，再试 sp1_no_offload_compile_fa_156f_all_gpu0_warmup
5. compile warmup 跑通后，再试 sp1_no_offload_compile_fa_156f_all_gpu0
```

不要同时改多个变量。比如不要在第一次评估时同时打开 TeaCache 和 `torch.compile`，否则无法判断速度变化来自哪里。

## 11. 常见报错判断

### decode 阶段 OOM

典型日志：

```text
VideoEditDecodingStage ... CUDA out of memory
```

处理：

```text
优先切到第 4 节 offload 配置。
```

### no-offload 全帧 OOM

如果 GPU0 已经有 20GB 以上残留显存，no-offload 全帧非常容易 OOM。先用 offload 跑出结果，等 GPU reset 或节点重启后再测 no-offload。

### TeaCache 没有加速

如果 `"enable_teacache": true` 但 denoising 时间没有下降，记录为：

```text
TeaCache request accepted, but no confirmed VideoEdit speedup in current branch.
```

### curl 卡住

用：

```bash
curl --noproxy '*' -s http://127.0.0.1:30000/health
```

不要漏掉 `--noproxy '*'`。
