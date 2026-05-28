# VideoEdit 全帧双卡 Serve 傻瓜操作文档

本文记录当前机器上运行 VideoEdit 的推荐流程。当前输入视频和 mask 都是 `156` 帧，所以本文的全帧命令统一使用：

```json
"num_frames": 156
```

输出目录统一放到：

```text
/home/tyx/workspace/zhouhao6/sglang/output_tyx
```

注意：当前代码不支持 `"num_frames": -1` 表示全部帧。要跑全部帧，需要显式传入当前视频帧数。本文针对当前这个视频写死为 `156`。

## 0. 先看结论

推荐优先跑双卡 SP2：

```text
--num-gpus 2
--sp-degree 2
--ulysses-degree 2
--ring-degree 1
```

原因是：`100` 帧 no-offload 单卡已经出现 OOM，`156` 帧全帧单卡更容易 OOM。双卡 SP2 已经实测可以跑过 `100` 帧 no-offload + FA。

本文主命令使用：

```json
"drop_reference_frame": false
```

这样输出视频也保留 `156` 帧。如果改成 `true`，会处理 `156` 个输入帧，但最终输出会丢掉第 1 帧，变成 `155` 帧，这和之前 `81 -> 80`、`100 -> 99` 的现象一致。

## 1. 公共准备

在任意终端先进入仓库：

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

确认输入视频和 mask 帧数：

```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=nb_frames,width,height,r_frame_rate,duration \
  -of json "$INPUT_VIDEO"

ffprobe -v error -select_streams v:0 \
  -show_entries stream=nb_frames,width,height,r_frame_rate,duration \
  -of json "$INPUT_MASK"
```

当前应看到两者都是：

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

如果 `30000` 已经有旧服务，先停止旧服务。最推荐在旧 serve 终端按：

```text
Ctrl-C
```

如果找不到旧终端，可以检查端口：

```bash
netstat -ltnp 2>/dev/null | grep ':30000' || true
```

## 2. 终端怎么分工

### serve 终端

启动 `sglang serve ...` 的终端会一直被占用，这是正常的。不要在这个终端里继续输入 `curl`。

### 查询终端

健康检查、提交请求、查进度，都另开一个新终端执行。

### 为什么 curl 要加 `--noproxy '*'`

当前机器里的 `curl` 可能会走 `http_proxy=127.0.0.1:18080`，导致访问本机 `30000` 卡住。

所以所有本地请求都写成：

```bash
curl --noproxy '*' ...
```

## 3. 推荐：双卡 SP2 + no-offload + FA，全帧 156

这是当前最建议先跑的版本：不引入 Cache-DiT 的质量 tradeoff，只用双卡 SP2 和 FlashAttention 加速。

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
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p "$OUT_DIR" /tmp/sglang-videoedit-inputs /tmp/sglang-flashinfer /tmp/sglang-cache

sglang serve \
  --model-type diffusion \
  --model-path "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port 30000 \
  --num-gpus 2 \
  --sp-degree 2 \
  --ulysses-degree 2 \
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

如果 `PYTORCH_ALLOC_CONF` 在你的 PyTorch 版本里不生效，再改用旧变量：

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

不要两个 serve 同时占用同一个端口。切换配置前，先在旧 serve 终端按 `Ctrl-C`。

### 3.2 判断 serve 已启动

serve 终端里看到下面这行，说明 HTTP 服务已经可用：

```text
Uvicorn running on http://0.0.0.0:30000
```

还要看启动日志里的 `server_args`，确认：

```text
"num_gpus": 2
"sp_degree": 2
"ulysses_degree": 2
"ring_degree": 1
"dit_cpu_offload": false
"dit_layerwise_offload": false
"attention_backend": "fa"
```

日志里也应出现类似：

```text
Using FlashAttention ...
```

### 3.3 健康检查

另开一个查询终端：

```bash
cd /home/tyx/workspace/zhouhao6/sglang
source .venv/bin/activate

curl --noproxy '*' -s http://127.0.0.1:30000/health
```

正常返回：

```json
{"status":"ok"}
```

### 3.4 提交全帧 156 请求

仍然在查询终端执行：

```bash
curl --noproxy '*' -s -X POST http://127.0.0.1:30000/v1/videos/repairs \
  -H 'Content-Type: application/json' \
  -d '{
    "task_id": "sp2_no_offload_fa_156f_all",
    "prompt": "A close-up of an orange flower with a yellow center, remaining in focus against a blurred green grass background throughout the video.",
    "video_input_path": "/home/tyx/workspace/zhouhao6/video_diffusers/pexel_test_data_0410/videos/15108907_3840_2160_50fps_short.mp4",
    "mask_input_path": "/home/tyx/workspace/zhouhao6/video_diffusers/pexel_test_data_0410/masks/15108907_3840_2160_50fps_No_bbox_mask.mp4",
    "output_storage": "local",
    "output_path": "/home/tyx/workspace/zhouhao6/sglang/output_tyx/15108907_3840_2160_50fps_api_sp2_no_offload_fa_156f_all.mp4",
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
    "perf_dump_path": "/home/tyx/workspace/zhouhao6/sglang/output_tyx/videoedit_perf_api_sp2_no_offload_fa_156f_all.json"
  }'
```

正常会很快返回：

```json
{"status":"queued", ...}
```

### 3.5 查看进度

查询终端执行：

```bash
curl --noproxy '*' -s http://127.0.0.1:30000/v1/videos/sp2_no_offload_fa_156f_all/progress
```

常见状态：

```json
{"status":"running","progress":1,...}
```

完成后：

```json
{"status":"completed","progress":100,...}
```

注意：API 的 `progress` 现在比较粗，只能看 queued、running、completed。真实 step 进度要看 serve 终端里的进度条，例如：

```text
35%|███▌      | 7/20
```

### 3.6 看 GPU

查询终端可以随时执行：

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader
```

如果 GPU0 还有别的残留占用，双卡也可能 OOM。跑之前最好确认 GPU0/GPU1 都有足够空闲显存。

### 3.7 检查输出

完成后：

```bash
ls -lh \
  output_tyx/15108907_3840_2160_50fps_api_sp2_no_offload_fa_156f_all.mp4 \
  output_tyx/videoedit_perf_api_sp2_no_offload_fa_156f_all.json \
  output_tyx/15108907_3840_2160_50fps_api_sp2_no_offload_fa_156f_all.videoedit.json
```

检查帧数：

```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=nb_frames,width,height,r_frame_rate,duration \
  -of json output_tyx/15108907_3840_2160_50fps_api_sp2_no_offload_fa_156f_all.mp4
```

因为本文主命令使用 `"drop_reference_frame": false`，预期输出是：

```text
nb_frames = 156
```

如果你把 `"drop_reference_frame"` 改成 `true`，预期输出是：

```text
nb_frames = 155
```

查看 perf：

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("output_tyx/videoedit_perf_api_sp2_no_offload_fa_156f_all.json")
data = json.load(open(path))
print(json.dumps(data, indent=2, ensure_ascii=False))
PY
```

## 4. 它有没有用“翻转”或“反射补帧”

有，但这里的“翻转”不是画面左右翻转，也不是上下翻转。

当前代码使用的是时间维度的反射补帧，目的是让每个模型窗口都凑够固定的 `infer_len=81` 帧。

全帧 156、`infer_len=81`、`overlap=0` 时，窗口会是：

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

第二个窗口只有真实帧 `81..155`，一共 `75` 帧，不够 `81`，所以代码会从视频尾部按时间反射补 `6` 帧：

```text
156 -> 154
157 -> 153
158 -> 152
159 -> 151
160 -> 150
161 -> 149
```

这些反射出来的帧只用于喂给模型，让第二个窗口长度达到 `81`。它们不会作为新的全局输出帧提交出去。真正写回输出的还是原始全局帧 `0..155`。

跑完后可以直接看 metadata 验证：

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("output_tyx/15108907_3840_2160_50fps_api_sp2_no_offload_fa_156f_all.videoedit.json")
meta = json.load(open(path))
print(json.dumps({
    "num_input_frames": meta.get("num_input_frames"),
    "drop_reference_frame": meta.get("drop_reference_frame"),
    "window_specs": meta.get("window_specs"),
}, indent=2, ensure_ascii=False))
PY
```

如果看到第二个窗口 `"reflected_count": 6`，就是用了尾部反射补帧。

## 5. 可选：双卡 SP2 + Cache-DiT fast，全帧 156

这一步可能更快，但会引入质量变化风险。建议先跑完第 3 节的非 cache 版本，再跑 Cache-DiT 对比。

Cache-DiT 不能和 `dit_layerwise_offload` 一起用，所以必须保持：

```text
--dit-layerwise-offload false
```

原生 VideoEdit 第一阶段只通过 `SGLANG_CACHE_DIT_*` 环境变量启用 Cache-DiT；不要使用 `--cache-dit-config`，该参数当前保留给 Diffusers backend。首轮排障建议同时保持 `--enable-torch-compile false`，确认 Cache-DiT 正确后再单独验证 compile。

### 5.1 启动 serve

先停止第 3 节的 serve，再在 serve 终端执行：

```bash
cd /home/tyx/workspace/zhouhao6/sglang
source .venv/bin/activate

export MODEL_PATH=/home/tyx/workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model
export TRANSFORMER_PATH=/home/tyx/workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model/transformer
export OUT_DIR=/home/tyx/workspace/zhouhao6/sglang/output_tyx
export FLASHINFER_WORKSPACE_BASE=/tmp/sglang-flashinfer
export XDG_CACHE_HOME=/tmp/sglang-cache
export VIDEOEDIT_QUEUE_CAPACITY=1
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_ALLOC_CONF=expandable_segments:True

export SGLANG_CACHE_DIT_ENABLED=true
export SGLANG_CACHE_DIT_FN=1
export SGLANG_CACHE_DIT_BN=0
export SGLANG_CACHE_DIT_WARMUP=2
export SGLANG_CACHE_DIT_RDT=0.24
export SGLANG_CACHE_DIT_MC=3
export SGLANG_CACHE_DIT_SCM_PRESET=fast

mkdir -p "$OUT_DIR" /tmp/sglang-videoedit-inputs /tmp/sglang-flashinfer /tmp/sglang-cache

sglang serve \
  --model-type diffusion \
  --model-path "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port 30000 \
  --num-gpus 2 \
  --sp-degree 2 \
  --ulysses-degree 2 \
  --ring-degree 1 \
  --dit-cpu-offload false \
  --dit-layerwise-offload false \
  --enable-torch-compile false \
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

### 5.2 判断 Cache-DiT 是否启用

serve 终端先确认：

```text
Uvicorn running on http://0.0.0.0:30000
```

再看日志中是否出现类似：

```text
cache-dit enabled on transformer
VideoEdit cache-dit uses Wan adapter: blocks=..., pattern=Pattern_2, separate_cfg=True
```

或包含：

```text
Fn=1
Bn=0
rdt=0.240
```

如果日志出现 cache policy 无效、fallback、或者没有任何 cache-dit 启用信息，本轮不能算有效 Cache-DiT 测试。

### 5.3 提交全帧 156 请求

查询终端执行：

```bash
curl --noproxy '*' -s -X POST http://127.0.0.1:30000/v1/videos/repairs \
  -H 'Content-Type: application/json' \
  -d '{
    "task_id": "sp2_no_offload_fa_cache_fast_156f_all",
    "prompt": "A close-up of an orange flower with a yellow center, remaining in focus against a blurred green grass background throughout the video.",
    "video_input_path": "/home/tyx/workspace/zhouhao6/video_diffusers/pexel_test_data_0410/videos/15108907_3840_2160_50fps_short.mp4",
    "mask_input_path": "/home/tyx/workspace/zhouhao6/video_diffusers/pexel_test_data_0410/masks/15108907_3840_2160_50fps_No_bbox_mask.mp4",
    "output_storage": "local",
    "output_path": "/home/tyx/workspace/zhouhao6/sglang/output_tyx/15108907_3840_2160_50fps_api_sp2_no_offload_fa_cache_fast_156f_all.mp4",
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
    "perf_dump_path": "/home/tyx/workspace/zhouhao6/sglang/output_tyx/videoedit_perf_api_sp2_no_offload_fa_cache_fast_156f_all.json"
  }'
```

查进度：

```bash
curl --noproxy '*' -s http://127.0.0.1:30000/v1/videos/sp2_no_offload_fa_cache_fast_156f_all/progress
```

## 6. 可选：保守 Cache-DiT

如果 fast 质量不好，改用保守参数。只替换第 5 节里的 Cache-DiT 环境变量：

```bash
export SGLANG_CACHE_DIT_ENABLED=true
export SGLANG_CACHE_DIT_FN=1
export SGLANG_CACHE_DIT_BN=1
export SGLANG_CACHE_DIT_WARMUP=4
export SGLANG_CACHE_DIT_RDT=0.12
export SGLANG_CACHE_DIT_MC=2
export SGLANG_CACHE_DIT_SCM_PRESET=medium
```

对应 task_id 和输出文件建议改成：

```text
sp2_no_offload_fa_cache_medium_156f_all
15108907_3840_2160_50fps_api_sp2_no_offload_fa_cache_medium_156f_all.mp4
videoedit_perf_api_sp2_no_offload_fa_cache_medium_156f_all.json
```

## 7. 可选：torch.compile

`torch.compile` 第一次请求会包含编译成本，不要用第一次请求判断稳定速度。并且它可能进一步增加显存压力，所以建议先跑通第 3 节，再考虑它。

如果要试，在第 3 节 serve 命令基础上增加：

```bash
export SGLANG_TORCH_COMPILE_MODE=max-autotune-no-cudagraphs
```

并在 `sglang serve` 参数里增加：

```text
--enable-torch-compile true
```

同一个 serve 里连续发两次完全相同 shape 的请求，第一次作为 compile 预热，第二次才作为正式性能记录。task_id 可以分别写成：

```text
sp2_no_offload_compile_fa_156f_all_warmup
sp2_no_offload_compile_fa_156f_all
```

## 8. 常用排查命令

### 服务是否可用

```bash
curl --noproxy '*' -s http://127.0.0.1:30000/health
```

### 当前任务列表

```bash
curl --noproxy '*' -s http://127.0.0.1:30000/v1/videos
```

### 查某个任务进度

```bash
curl --noproxy '*' -s http://127.0.0.1:30000/v1/videos/<task_id>/progress
```

例如：

```bash
curl --noproxy '*' -s http://127.0.0.1:30000/v1/videos/sp2_no_offload_fa_156f_all/progress
```

### 看 GPU

```bash
nvidia-smi
```

或者只看显存和利用率：

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader
```

### 输出文件检查

```bash
ls -lh output_tyx/*156f_all*
```

### 检查输出帧数

```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=nb_frames,width,height,r_frame_rate,duration \
  -of json output_tyx/15108907_3840_2160_50fps_api_sp2_no_offload_fa_156f_all.mp4
```

### 检查是否用了反射补帧

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("output_tyx/15108907_3840_2160_50fps_api_sp2_no_offload_fa_156f_all.videoedit.json")
meta = json.load(open(path))
for spec in meta["window_specs"]:
    print(spec)
PY
```

## 9. 结果记录建议

每跑完一组，记录：

```text
task_id:
输出视频:
perf JSON:
metadata JSON:
输出帧数:
warmup 时间:
total_duration_ms:
DenoisingStage:
DecodingStage:
peak allocated/reserved:
window_specs:
主观质量:
```

对 Cache-DiT 结果，至少人工看一遍：

- mask 边缘是否闪烁；
- 花朵纹理是否漂移；
- 窗口边界附近是否突然变化；
- 和 no-offload + FA 输出是否肉眼差异明显。

## 10. 推荐执行顺序

严格按这个顺序：

```text
1. sp2_no_offload_fa_156f_all
2. sp2_no_offload_fa_cache_fast_156f_all
3. 如果 fast 质量不好，再测 sp2_no_offload_fa_cache_medium_156f_all
4. 如果还有时间，再单独试 sp2_no_offload_compile_fa_156f_all_warmup 和 sp2_no_offload_compile_fa_156f_all
```

不要在同一次对比里同时改变太多变量。先确认双卡非 cache 版本能稳定跑完，再单独比较 Cache-DiT 或 compile。
