# VideoEdit 50-step 优化方案

本文档记录 VideoEdit-diffusers 接入 SGLang 后的 50-step 性能优化路线。参考文档：

- 接入方案：[`../add_videoedit_diffusers/README.md`](../add_videoedit_diffusers/README.md)
- CLI / serve 命令：[`../add_videoedit_diffusers/cli.md`](../add_videoedit_diffusers/cli.md)
- 历史 20-step benchmark：[`../add_videoedit_diffusers/benchmark_results.md`](../add_videoedit_diffusers/benchmark_results.md)

本文只安排无代码调参和验证，不修改源码。若某项优化需要改代码，先记录到
[后续代码级优化](#19-后续代码级优化)，不要混入本轮 50-step 结论。

## 1. 硬约束

本轮优化只比较系统实现、算子、并行、cache、offload 和量化，不通过降低采样质量换速度。

- 固定同一视频、mask、prompt、seed、`num_frames`、`infer_len`、`overlap`、`num_inference_steps`、`guidance_scale`。
- 固定质量对比 reference：`$OUT_DIR/reference/15108907_3840_2160_50fps.mp4`。
- Reference 必须确认是 50-step 输出；如果该文件来自 20-step 历史实验，需要先重建 reference，否则本轮 compare 无效。
- `num_inference_steps` 固定为 `50`，不允许降低 steps。
- `dynamic_cfg_max_step` 固定为默认 `15`，不允许改成 `10/12` 等较低值。
- 除 Cache-DiT / TeaCache 和量化外，不允许引入其他可能降低质量的方法。
- 默认非 cache 阶段必须显式关闭 TeaCache：CLI 用 `--no-enable-teacache`，serve 请求 JSON 用 `"enable_teacache": false`。
- Cache-DiT 和 TeaCache 分开测试，不在同一阶段同时打开。
- 每个优化阶段同时给 CLI 和 serve 命令。
- 优化项逐渐累加，每一步只新增一个优化维度；记录时同时和上一阶段、主基线对比。
- 每个命令都启用 `--warmup --warmup-steps 1` 或 serve 等价参数，并写 `perf_dump_path`。
- 每个输出都必须保留 perf JSON、视频 metadata、CUDA memory、GPU utilization 和逐窗口日志。
- 每个候选输出都必须和固定 50-step reference 做输出检查；cache 和量化必须额外做逐帧 compare 和人工抽检。

不纳入推荐优化的历史项：

- 低步数：例如 `num_inference_steps=16/18/20`。20-step benchmark 只能用于排序参考，不能作为 50-step 结论。
- 缩短 dynamic CFG：例如 `dynamic_cfg_max_step=10/12`。
- CLI 冷启动下的 `torch.compile` wall time：可以记录，但不作为 compile 收益结论。

## 2. 公共环境

```bash
source /mnt/shanhai-ai/shanhai-workspace/zhouhao6/env/sglang/bin/activate
cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang

export MODEL_PATH=/mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model
export TRANSFORMER_PATH=/mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/step-55000-diffusers-lh/transformer
export INPUT_VIDEO=/mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pexel_test_data_0410/videos/15108907_3840_2160_50fps_short.mp4
export INPUT_MASK=/mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pexel_test_data_0410/masks/15108907_3840_2160_50fps_No_bbox_mask.mp4
export OUT_DIR=/mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs
export PROMPT="A close-up of an orange flower with a yellow center, remaining in focus against a blurred green grass background throughout the video."
export VIDEO_BASENAME=15108907_3840_2160_50fps

mkdir -p "$OUT_DIR" "$OUT_DIR/reference"
set -o pipefail
```

固定请求参数：

```text
num_frames = 81
infer_len = 81
overlap = 0
num_inference_steps = 50
guidance_scale = 5.0
dynamic_cfg = true
dynamic_cfg_max_step = 15
seed = 42
dtype = bf16
enable_paste_back = true
drop_reference_frame = true
warmup = true
warmup_steps = 1
enable_teacache = false unless the stage is TeaCache
```

## 3. Reference Gate

优化前先确认 reference 视频存在且对应 50-step 质量基准。

```bash
python - <<'PY'
import cv2
import os

out_dir = os.environ["OUT_DIR"]
path = os.path.join(out_dir, "reference", "15108907_3840_2160_50fps.mp4")
if not os.path.exists(path):
    raise SystemExit(f"missing reference: {path}")

cap = cv2.VideoCapture(path)
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()
print({"path": path, "frames": frames, "width": width, "height": height, "fps": fps})
assert frames in (80, 81), frames
assert width > 0 and height > 0
PY
```

如果有 reference sidecar，检查采样参数：

```bash
python -m json.tool "$OUT_DIR/reference/15108907_3840_2160_50fps.videoedit.json"
```

验收口径：

- sidecar 中 `num_inference_steps` 必须是 `50`。
- 如果没有 sidecar，需要从生成 reference 的日志或命令确认 50-step；证据缺失时不要继续跑优化结论。
- Reference 不作为最快配置，只作为固定质量锚点。

## 4. 记录口径

每个阶段至少记录：

- 单窗口耗时：从逐窗口日志或 stage 日志提取。
- 端到端耗时：CLI wall time 或 serve 请求 `inference_time_s`。
- 分段耗时：`VideoEditTextEncodingStage`、`VideoEditConditionEncodingStage`、`VideoEditDenoisingStage`、`VideoEditDecodingStage`、postprocess。
- 显存：`perf_dump_path` 中的 peak allocated/reserved，外部补充 `nvidia-smi` 峰值。
- GPU utilization：运行期间用 `nvidia-smi dmon` 或等价监控记录。
- Attention backend：同时记录命令行期望值和日志解析出的实际值。
- 输出质量：视频帧数、分辨率、metadata、逐帧 compare JSON。

推荐命名：

```text
输出视频：$OUT_DIR/15108907_3840_2160_50fps_<stage>.mp4
API 输出：$OUT_DIR/15108907_3840_2160_50fps_api_<stage>.mp4
perf JSON：$OUT_DIR/videoedit_perf_<stage>.json
API perf JSON：$OUT_DIR/videoedit_perf_api_<stage>.json
compare JSON：$OUT_DIR/videoedit_compare_<stage>.json
bench 日志：$OUT_DIR/videoedit_bench_<stage>.log
serve 日志：$OUT_DIR/videoedit_serve_<stage>.log
dmon 日志：$OUT_DIR/videoedit_dmon_<stage>.csv
```

监控命令：

```bash
STAGE=sp2_no_offload_fa
nvidia-smi dmon -s pucm -d 1 -o DT > "$OUT_DIR/videoedit_dmon_${STAGE}.csv"
```

结束阶段后解析实际 backend：

```bash
STAGE=sp2_no_offload_fa
rg -n "attention backend|Using .*Attention|Sage|fallback|No module named|Selected attention" \
  "$OUT_DIR/videoedit_bench_${STAGE}.log" \
  "$OUT_DIR/videoedit_serve_${STAGE}.log"
```

## 5. 公共命令模板

先定义三个 shell helper。后续每个阶段只改并行、offload、compile、attention、cache 等增量参数。

### 5.1 CLI helper

```bash
run_videoedit_cli() {
  local stage="$1"
  shift
  local transformer_path="${CLI_TRANSFORMER_PATH:-$TRANSFORMER_PATH}"

  local teacache_args=(--no-enable-teacache)
  if [[ "${ENABLE_TEACACHE:-false}" == "true" ]]; then
    teacache_args=(
      --enable-teacache
      --teacache-thresh "${TEACACHE_THRESH:-0.3}"
      --teacache-start-skipping "${TEACACHE_START_SKIPPING:-5}"
      --teacache-end-skipping "${TEACACHE_END_SKIPPING:-1.0}"
    )
  fi

  python -m sglang.multimodal_gen.runtime.videoedit.cli repair \
    --model-path "$MODEL_PATH" \
    --transformer-path "$transformer_path" \
    --prompt "$PROMPT" \
    --video-input-path "$INPUT_VIDEO" \
    --mask-input-path "$INPUT_MASK" \
    --output-path "$OUT_DIR" \
    --output-file-name "${VIDEO_BASENAME}_${stage}.mp4" \
    --num-frames 81 \
    --infer-len 81 \
    --overlap 0 \
    --num-inference-steps 50 \
    --guidance-scale 5.0 \
    --dynamic-cfg \
    --dynamic-cfg-max-step 15 \
    --seed 42 \
    --dtype bf16 \
    --enable-paste-back \
    --drop-reference-frame \
    "${teacache_args[@]}" \
    --warmup \
    --warmup-steps 1 \
    --perf-dump-path "$OUT_DIR/videoedit_perf_${stage}.json" \
    "$@"
}
```

### 5.2 Serve helper

Serve 每个配置在单独终端启动。换阶段前停止旧 serve，避免端口和分布式进程残留影响结果。

```bash
start_videoedit_serve() {
  local transformer_path="${SERVE_TRANSFORMER_PATH:-$TRANSFORMER_PATH}"
  VIDEOEDIT_QUEUE_CAPACITY=1 \
  sglang serve \
    --model-type diffusion \
    --model-path "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port 30000 \
    --warmup true \
    --warmup-steps 1 \
    --output-path "$OUT_DIR" \
    --input-save-path /tmp/sglang-videoedit-inputs \
    --transformer-path "$transformer_path" \
    "$@"
}
```

提交请求：

```bash
submit_videoedit_request() {
  : "${STAGE:?set STAGE}"
  python - <<'PY'
import json
import os
import urllib.request


def as_bool(name: str, default: str = "false") -> bool:
    return os.environ.get(name, default).lower() in {"1", "true", "yes", "on"}


def as_num(name: str, default: str):
    value = os.environ.get(name, default)
    return float(value) if any(ch in value for ch in ".eE") else int(value)


stage = os.environ["STAGE"]
out_dir = os.environ["OUT_DIR"]
base = os.environ.get("VIDEO_BASENAME", "15108907_3840_2160_50fps")
payload = {
    "task_id": stage,
    "prompt": os.environ["PROMPT"],
    "video_input_path": os.environ["INPUT_VIDEO"],
    "mask_input_path": os.environ["INPUT_MASK"],
    "output_storage": "local",
    "output_path": f"{out_dir}/{base}_api_{stage}.mp4",
    "num_frames": 81,
    "infer_len": 81,
    "overlap": 0,
    "num_inference_steps": 50,
    "guidance_scale": 5.0,
    "dynamic_cfg": True,
    "dynamic_cfg_max_step": 15,
    "seed": 42,
    "dtype": "bf16",
    "enable_paste_back": True,
    "drop_reference_frame": True,
    "enable_teacache": as_bool("ENABLE_TEACACHE"),
    "teacache_thresh": float(os.environ.get("TEACACHE_THRESH", "0.3")),
    "teacache_start_skipping": as_num("TEACACHE_START_SKIPPING", "5"),
    "teacache_end_skipping": as_num("TEACACHE_END_SKIPPING", "1.0"),
    "perf_dump_path": f"{out_dir}/videoedit_perf_api_{stage}.json",
}
data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
req = urllib.request.Request(
    "http://127.0.0.1:30000/v1/videos/repairs",
    data=data,
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=30) as resp:
    print(resp.read().decode())
PY
}
```

轮询任务：

```bash
poll_videoedit_job() {
  local job_id="$1"
  python - "$job_id" <<'PY'
import json
import sys
import time
import urllib.request

job_id = sys.argv[1]
url = f"http://127.0.0.1:30000/v1/videos/{job_id}"
while True:
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.loads(resp.read().decode())
    print(data.get("status"), data.get("progress"), data.get("file_path") or data.get("url"), data.get("inference_time_s"))
    if data.get("status") == "completed":
        break
    if data.get("status") == "failed":
        raise SystemExit(json.dumps(data.get("error"), ensure_ascii=False))
    time.sleep(5)
PY
}
```

### 5.3 Compare helper

```bash
compare_videoedit_candidate() {
  local stage="$1"
  local candidate="${2:-$OUT_DIR/${VIDEO_BASENAME}_${stage}.mp4}"
  python python/sglang/multimodal_gen/runtime/videoedit/compare.py \
    --reference "$OUT_DIR/reference/15108907_3840_2160_50fps.mp4" \
    --candidate "$candidate" \
    --report-json "$OUT_DIR/videoedit_compare_${stage}.json" \
    --min-ssim 0.90 \
    --max-mse 150.0 \
    --max-mae 8.0 \
    --allow-frame-count-delta 1 \
    --max-failed-frame-ratio 0.05
}
```

如果 candidate 比 reference 多 1 帧且多出的帧是第 0 帧：

```bash
compare_videoedit_candidate_drop_first() {
  local stage="$1"
  local candidate="${2:-$OUT_DIR/${VIDEO_BASENAME}_${stage}.mp4}"
  python python/sglang/multimodal_gen/runtime/videoedit/compare.py \
    --reference "$OUT_DIR/reference/15108907_3840_2160_50fps.mp4" \
    --candidate "$candidate" \
    --drop-candidate-first-frame \
    --report-json "$OUT_DIR/videoedit_compare_${stage}.json" \
    --min-ssim 0.90 \
    --max-mse 150.0 \
    --max-mae 8.0 \
    --allow-frame-count-delta 1 \
    --max-failed-frame-ratio 0.05
}
```

## 6. 优化阶梯

主线从 `sp1_offload` 开始，逐步累加：

1. `sp1_offload`：基线，优先保证低显存和可跑通。
2. `sp1_no_offload`：关闭 offload，观察同一单卡配置下的 latency / 显存交换。
3. `sp1_no_offload_compile`：在 no-offload 基础上加 `torch.compile`。
4. `sp1_no_offload_compile_<backend>`：在 compile 基础上加 attention backend，先确认后端真实生效。
5. `sp2_no_offload_<backend>`：扩展到双卡 SP，并分别对比 `torch_sdpa`、`fa`、`sage_attn`、`sage_attn_3`。
6. `sp2_ring_no_offload_fa` / `tp2_no_offload_fa`：只改变并行策略，对比 Ulysses SP、Ring SP、TP。
7. `sp2_no_offload_compile_<backend>`：只新增 `torch.compile`，serve 模式至少两次请求，以第二次为准。
8. `sp2_no_offload_compile_fa_teacache`：在最佳非 cache 配置上单独加 TeaCache。
9. `sp2_no_offload_compile_fa_cache_<policy>`：在最佳非 cache 配置上单独加 Cache-DiT。
10. `quant_branch`：在最佳非量化配置上替换量化 DiT 权重或启用已有量化配置。
11. `offload_branch`：显存受限分支，不并入性能主线。

历史 20-step benchmark 显示 A100 上 `fa` 通常优于 `torch_sdpa` 和 `sage_attn`，`sage_attn_3`
容易 fallback 或超时。但 50-step 仍必须重跑，不能继承绝对耗时。

## 7. Stage 0：SP1 Offload 基线

目标：固定最低显存、最稳可跑的性能起点。该阶段不追求最快，质量统一和 50-step reference 对比。

### CLI

```bash
STAGE=sp1_offload
SGLANG_CACHE_DIT_ENABLED=false ENABLE_TEACACHE=false \
run_videoedit_cli "$STAGE" \
  --num-gpus 1 \
  --sp-degree 1 \
  --ulysses-degree 1 \
  --ring-degree 1 \
  --dit-cpu-offload \
  --dit-layerwise-offload \
  --text-encoder-cpu-offload \
  --image-encoder-cpu-offload \
  --vae-cpu-offload \
  2>&1 | tee "$OUT_DIR/videoedit_bench_${STAGE}.log"

compare_videoedit_candidate "$STAGE"
```

### Serve

```bash
STAGE=sp1_offload
SGLANG_CACHE_DIT_ENABLED=false \
start_videoedit_serve \
  --num-gpus 1 \
  --sp-degree 1 \
  --ulysses-degree 1 \
  --ring-degree 1 \
  --dit-cpu-offload true \
  --dit-layerwise-offload true \
  --text-encoder-cpu-offload true \
  --image-encoder-cpu-offload true \
  --vae-cpu-offload true \
  2>&1 | tee "$OUT_DIR/videoedit_serve_${STAGE}.log"
```

另一个终端提交请求：

```bash
STAGE=sp1_offload ENABLE_TEACACHE=false submit_videoedit_request
poll_videoedit_job sp1_offload
compare_videoedit_candidate api_sp1_offload "$OUT_DIR/${VIDEO_BASENAME}_api_sp1_offload.mp4"
```

## 8. Stage 1：SP1 No Offload

目标：只关闭 offload，观察单卡速度收益和峰值显存变化。

### CLI

```bash
STAGE=sp1_no_offload
SGLANG_CACHE_DIT_ENABLED=false ENABLE_TEACACHE=false \
run_videoedit_cli "$STAGE" \
  --num-gpus 1 \
  --sp-degree 1 \
  --ulysses-degree 1 \
  --ring-degree 1 \
  --no-dit-cpu-offload \
  --no-dit-layerwise-offload \
  --no-text-encoder-cpu-offload \
  --no-image-encoder-cpu-offload \
  --no-vae-cpu-offload \
  2>&1 | tee "$OUT_DIR/videoedit_bench_${STAGE}.log"

compare_videoedit_candidate "$STAGE"
```

### Serve

```bash
STAGE=sp1_no_offload
SGLANG_CACHE_DIT_ENABLED=false \
start_videoedit_serve \
  --num-gpus 1 \
  --sp-degree 1 \
  --ulysses-degree 1 \
  --ring-degree 1 \
  --dit-cpu-offload false \
  --dit-layerwise-offload false \
  --text-encoder-cpu-offload false \
  --image-encoder-cpu-offload false \
  --vae-cpu-offload false \
  2>&1 | tee "$OUT_DIR/videoedit_serve_${STAGE}.log"
```

```bash
STAGE=sp1_no_offload ENABLE_TEACACHE=false submit_videoedit_request
poll_videoedit_job sp1_no_offload
compare_videoedit_candidate api_sp1_no_offload "$OUT_DIR/${VIDEO_BASENAME}_api_sp1_no_offload.mp4"
```

## 9. Stage 2：SP1 No Offload + torch.compile

目标：在 `sp1_no_offload` 基础上只新增 `torch.compile`。CLI 可用于确认能跑通；正式收益看 serve 常驻模式。

### CLI

```bash
STAGE=sp1_no_offload_compile
SGLANG_CACHE_DIT_ENABLED=false \
SGLANG_TORCH_COMPILE_MODE=max-autotune-no-cudagraphs \
ENABLE_TEACACHE=false \
run_videoedit_cli "$STAGE" \
  --num-gpus 1 \
  --sp-degree 1 \
  --ulysses-degree 1 \
  --ring-degree 1 \
  --no-dit-cpu-offload \
  --no-dit-layerwise-offload \
  --no-text-encoder-cpu-offload \
  --no-image-encoder-cpu-offload \
  --no-vae-cpu-offload \
  --enable-torch-compile \
  2>&1 | tee "$OUT_DIR/videoedit_bench_${STAGE}.log"

compare_videoedit_candidate "$STAGE"
```

### Serve

```bash
STAGE=sp1_no_offload_compile
SGLANG_CACHE_DIT_ENABLED=false \
SGLANG_TORCH_COMPILE_MODE=max-autotune-no-cudagraphs \
start_videoedit_serve \
  --num-gpus 1 \
  --sp-degree 1 \
  --ulysses-degree 1 \
  --ring-degree 1 \
  --dit-cpu-offload false \
  --dit-layerwise-offload false \
  --text-encoder-cpu-offload false \
  --image-encoder-cpu-offload false \
  --vae-cpu-offload false \
  --enable-torch-compile true \
  2>&1 | tee "$OUT_DIR/videoedit_serve_${STAGE}.log"
```

同 shape 至少提交两次请求。第一次只作为服务内 compile/warmup，不计入稳定收益。

```bash
STAGE=sp1_no_offload_compile_warmup_request ENABLE_TEACACHE=false submit_videoedit_request
poll_videoedit_job sp1_no_offload_compile_warmup_request

STAGE=sp1_no_offload_compile ENABLE_TEACACHE=false submit_videoedit_request
poll_videoedit_job sp1_no_offload_compile
compare_videoedit_candidate api_sp1_no_offload_compile "$OUT_DIR/${VIDEO_BASENAME}_api_sp1_no_offload_compile.mp4"
```

## 10. Stage 3：Attention Backend 对比

目标：在 compile 配置基础上只改变 attention backend。记录表中必须同时写 `expected_backend`
和日志解析出的 `actual_backend`。

A100 优先比较：

- `torch_sdpa`：PyTorch 原生 SDPA，保守基线。
- `fa`：FlashAttention，历史 20-step 中是 A100 主力高性能后端。
- `sage_attn`：SageAttention 2.x，可尝试安装后评估。
- `sage_attn_3`：Blackwell 专项；A100 上预期 fallback，不应把 fallback 结果记成 SageAttention3 成绩。

SageAttention 2.x 安装检查：

```bash
source /mnt/shanhai-ai/shanhai-workspace/zhouhao6/env/sglang/bin/activate
pip install ninja packaging
pip install sageattention==2.2.0 --no-build-isolation

python - <<'PY'
from sageattention import sageattn
print("sageattention ok", sageattn)
PY
```

### CLI

每个 backend 单独跑一次：

```bash
for BACKEND in torch_sdpa fa sage_attn sage_attn_3; do
  STAGE="sp1_no_offload_compile_${BACKEND}"
  SGLANG_CACHE_DIT_ENABLED=false \
  SGLANG_TORCH_COMPILE_MODE=max-autotune-no-cudagraphs \
  ENABLE_TEACACHE=false \
  run_videoedit_cli "$STAGE" \
    --num-gpus 1 \
    --sp-degree 1 \
    --ulysses-degree 1 \
    --ring-degree 1 \
    --no-dit-cpu-offload \
    --no-dit-layerwise-offload \
    --no-text-encoder-cpu-offload \
    --no-image-encoder-cpu-offload \
    --no-vae-cpu-offload \
    --enable-torch-compile \
    --attention-backend "$BACKEND" \
    2>&1 | tee "$OUT_DIR/videoedit_bench_${STAGE}.log"
  compare_videoedit_candidate "$STAGE"
done
```

### Serve

每次只启动一个 backend：

```bash
BACKEND=fa
STAGE="sp1_no_offload_compile_${BACKEND}"
SGLANG_CACHE_DIT_ENABLED=false \
SGLANG_TORCH_COMPILE_MODE=max-autotune-no-cudagraphs \
start_videoedit_serve \
  --num-gpus 1 \
  --sp-degree 1 \
  --ulysses-degree 1 \
  --ring-degree 1 \
  --dit-cpu-offload false \
  --dit-layerwise-offload false \
  --text-encoder-cpu-offload false \
  --image-encoder-cpu-offload false \
  --vae-cpu-offload false \
  --enable-torch-compile true \
  --attention-backend "$BACKEND" \
  2>&1 | tee "$OUT_DIR/videoedit_serve_${STAGE}.log"
```

```bash
STAGE=sp1_no_offload_compile_fa_warmup_request ENABLE_TEACACHE=false submit_videoedit_request
poll_videoedit_job sp1_no_offload_compile_fa_warmup_request

STAGE=sp1_no_offload_compile_fa ENABLE_TEACACHE=false submit_videoedit_request
poll_videoedit_job sp1_no_offload_compile_fa
compare_videoedit_candidate api_sp1_no_offload_compile_fa "$OUT_DIR/${VIDEO_BASENAME}_api_sp1_no_offload_compile_fa.mp4"
```

## 11. Stage 4：双卡 SP / TP 并行对比

目标：在无 cache、无 TeaCache、无 offload 的配置上扩展到双卡。先测 SP2 Ulysses，再测 Ring SP 和 TP2。

不要在同一次对比中同时改变 attention backend、compile、offload 或 cache。

### CLI：SP2 Ulysses backend 矩阵

```bash
for BACKEND in torch_sdpa fa sage_attn sage_attn_3; do
  STAGE="sp2_no_offload_${BACKEND}"
  SGLANG_CACHE_DIT_ENABLED=false ENABLE_TEACACHE=false \
  run_videoedit_cli "$STAGE" \
    --num-gpus 2 \
    --sp-degree 2 \
    --ulysses-degree 2 \
    --ring-degree 1 \
    --no-dit-cpu-offload \
    --no-dit-layerwise-offload \
    --no-text-encoder-cpu-offload \
    --no-image-encoder-cpu-offload \
    --no-vae-cpu-offload \
    --attention-backend "$BACKEND" \
    2>&1 | tee "$OUT_DIR/videoedit_bench_${STAGE}.log"
  compare_videoedit_candidate "$STAGE"
done
```

### Serve：SP2 Ulysses backend 矩阵

每次只启动一个 backend：

```bash
BACKEND=fa
STAGE="sp2_no_offload_${BACKEND}"
SGLANG_CACHE_DIT_ENABLED=false \
start_videoedit_serve \
  --num-gpus 2 \
  --sp-degree 2 \
  --ulysses-degree 2 \
  --ring-degree 1 \
  --dit-cpu-offload false \
  --dit-layerwise-offload false \
  --text-encoder-cpu-offload false \
  --image-encoder-cpu-offload false \
  --vae-cpu-offload false \
  --attention-backend "$BACKEND" \
  2>&1 | tee "$OUT_DIR/videoedit_serve_${STAGE}.log"
```

```bash
STAGE=sp2_no_offload_fa_warmup_request ENABLE_TEACACHE=false submit_videoedit_request
poll_videoedit_job sp2_no_offload_fa_warmup_request

STAGE=sp2_no_offload_fa ENABLE_TEACACHE=false submit_videoedit_request
poll_videoedit_job sp2_no_offload_fa
compare_videoedit_candidate api_sp2_no_offload_fa "$OUT_DIR/${VIDEO_BASENAME}_api_sp2_no_offload_fa.mp4"
```

### CLI：Ring SP 和 TP2

```bash
STAGE=sp2_ring_no_offload_fa
SGLANG_CACHE_DIT_ENABLED=false ENABLE_TEACACHE=false \
run_videoedit_cli "$STAGE" \
  --num-gpus 2 \
  --sp-degree 2 \
  --ulysses-degree 1 \
  --ring-degree 2 \
  --no-dit-cpu-offload \
  --no-dit-layerwise-offload \
  --no-text-encoder-cpu-offload \
  --no-image-encoder-cpu-offload \
  --no-vae-cpu-offload \
  --attention-backend fa \
  2>&1 | tee "$OUT_DIR/videoedit_bench_${STAGE}.log"
compare_videoedit_candidate "$STAGE"

STAGE=tp2_no_offload_fa
SGLANG_CACHE_DIT_ENABLED=false ENABLE_TEACACHE=false \
run_videoedit_cli "$STAGE" \
  --num-gpus 2 \
  --tp-size 2 \
  --sp-degree 1 \
  --ulysses-degree 1 \
  --ring-degree 1 \
  --no-dit-cpu-offload \
  --no-dit-layerwise-offload \
  --no-text-encoder-cpu-offload \
  --no-image-encoder-cpu-offload \
  --no-vae-cpu-offload \
  --attention-backend fa \
  2>&1 | tee "$OUT_DIR/videoedit_bench_${STAGE}.log"
compare_videoedit_candidate "$STAGE"
```

### Serve：Ring SP 和 TP2

Ring SP：

```bash
STAGE=sp2_ring_no_offload_fa
SGLANG_CACHE_DIT_ENABLED=false \
start_videoedit_serve \
  --num-gpus 2 \
  --sp-degree 2 \
  --ulysses-degree 1 \
  --ring-degree 2 \
  --dit-cpu-offload false \
  --dit-layerwise-offload false \
  --text-encoder-cpu-offload false \
  --image-encoder-cpu-offload false \
  --vae-cpu-offload false \
  --attention-backend fa \
  2>&1 | tee "$OUT_DIR/videoedit_serve_${STAGE}.log"
```

TP2：

```bash
STAGE=tp2_no_offload_fa
SGLANG_CACHE_DIT_ENABLED=false \
start_videoedit_serve \
  --num-gpus 2 \
  --tp-size 2 \
  --sp-degree 1 \
  --ulysses-degree 1 \
  --ring-degree 1 \
  --dit-cpu-offload false \
  --dit-layerwise-offload false \
  --text-encoder-cpu-offload false \
  --image-encoder-cpu-offload false \
  --vae-cpu-offload false \
  --attention-backend fa \
  2>&1 | tee "$OUT_DIR/videoedit_serve_${STAGE}.log"
```

提交时把 `STAGE` 换成对应名字：

```bash
STAGE=tp2_no_offload_fa ENABLE_TEACACHE=false submit_videoedit_request
poll_videoedit_job tp2_no_offload_fa
compare_videoedit_candidate api_tp2_no_offload_fa "$OUT_DIR/${VIDEO_BASENAME}_api_tp2_no_offload_fa.mp4"
```

## 12. Stage 5：SP2 + torch.compile

目标：在最佳非 cache 并行配置上只新增 `torch.compile`。历史 20-step 中 `fa` 最优，因此先以
`sp2_no_offload_fa` 为默认候选；如果 50-step backend 矩阵显示其它 backend 更快且质量通过，替换 backend。

### CLI

```bash
STAGE=sp2_no_offload_compile_fa
SGLANG_CACHE_DIT_ENABLED=false \
SGLANG_TORCH_COMPILE_MODE=max-autotune-no-cudagraphs \
ENABLE_TEACACHE=false \
run_videoedit_cli "$STAGE" \
  --num-gpus 2 \
  --sp-degree 2 \
  --ulysses-degree 2 \
  --ring-degree 1 \
  --no-dit-cpu-offload \
  --no-dit-layerwise-offload \
  --no-text-encoder-cpu-offload \
  --no-image-encoder-cpu-offload \
  --no-vae-cpu-offload \
  --enable-torch-compile \
  --attention-backend fa \
  2>&1 | tee "$OUT_DIR/videoedit_bench_${STAGE}.log"

compare_videoedit_candidate "$STAGE"
```

### Serve

```bash
STAGE=sp2_no_offload_compile_fa
SGLANG_CACHE_DIT_ENABLED=false \
SGLANG_TORCH_COMPILE_MODE=max-autotune-no-cudagraphs \
start_videoedit_serve \
  --num-gpus 2 \
  --sp-degree 2 \
  --ulysses-degree 2 \
  --ring-degree 1 \
  --dit-cpu-offload false \
  --dit-layerwise-offload false \
  --text-encoder-cpu-offload false \
  --image-encoder-cpu-offload false \
  --vae-cpu-offload false \
  --enable-torch-compile true \
  --attention-backend fa \
  2>&1 | tee "$OUT_DIR/videoedit_serve_${STAGE}.log"
```

```bash
STAGE=sp2_no_offload_compile_fa_warmup_request ENABLE_TEACACHE=false submit_videoedit_request
poll_videoedit_job sp2_no_offload_compile_fa_warmup_request

STAGE=sp2_no_offload_compile_fa ENABLE_TEACACHE=false submit_videoedit_request
poll_videoedit_job sp2_no_offload_compile_fa
compare_videoedit_candidate api_sp2_no_offload_compile_fa "$OUT_DIR/${VIDEO_BASENAME}_api_sp2_no_offload_compile_fa.mp4"
```

## 13. Stage 6：TeaCache 分支

目标：在最佳非 cache 配置上只新增 TeaCache。TeaCache 是请求级参数，和 Cache-DiT 不是同一个开关。

默认先测保守阈值：

```text
enable_teacache = true
teacache_thresh = 0.3
teacache_start_skipping = 5
teacache_end_skipping = 1.0
```

### CLI

```bash
STAGE=sp2_no_offload_compile_fa_teacache
SGLANG_CACHE_DIT_ENABLED=false \
SGLANG_TORCH_COMPILE_MODE=max-autotune-no-cudagraphs \
ENABLE_TEACACHE=true \
TEACACHE_THRESH=0.3 \
TEACACHE_START_SKIPPING=5 \
TEACACHE_END_SKIPPING=1.0 \
run_videoedit_cli "$STAGE" \
  --num-gpus 2 \
  --sp-degree 2 \
  --ulysses-degree 2 \
  --ring-degree 1 \
  --no-dit-cpu-offload \
  --no-dit-layerwise-offload \
  --no-text-encoder-cpu-offload \
  --no-image-encoder-cpu-offload \
  --no-vae-cpu-offload \
  --enable-torch-compile \
  --attention-backend fa \
  2>&1 | tee "$OUT_DIR/videoedit_bench_${STAGE}.log"

compare_videoedit_candidate "$STAGE"
```

### Serve

```bash
STAGE=sp2_no_offload_compile_fa_teacache
SGLANG_CACHE_DIT_ENABLED=false \
SGLANG_TORCH_COMPILE_MODE=max-autotune-no-cudagraphs \
start_videoedit_serve \
  --num-gpus 2 \
  --sp-degree 2 \
  --ulysses-degree 2 \
  --ring-degree 1 \
  --dit-cpu-offload false \
  --dit-layerwise-offload false \
  --text-encoder-cpu-offload false \
  --image-encoder-cpu-offload false \
  --vae-cpu-offload false \
  --enable-torch-compile true \
  --attention-backend fa \
  2>&1 | tee "$OUT_DIR/videoedit_serve_${STAGE}.log"
```

```bash
STAGE=sp2_no_offload_compile_fa_teacache_warmup_request \
ENABLE_TEACACHE=true TEACACHE_THRESH=0.3 TEACACHE_START_SKIPPING=5 TEACACHE_END_SKIPPING=1.0 \
submit_videoedit_request
poll_videoedit_job sp2_no_offload_compile_fa_teacache_warmup_request

STAGE=sp2_no_offload_compile_fa_teacache \
ENABLE_TEACACHE=true TEACACHE_THRESH=0.3 TEACACHE_START_SKIPPING=5 TEACACHE_END_SKIPPING=1.0 \
submit_videoedit_request
poll_videoedit_job sp2_no_offload_compile_fa_teacache
compare_videoedit_candidate api_sp2_no_offload_compile_fa_teacache "$OUT_DIR/${VIDEO_BASENAME}_api_sp2_no_offload_compile_fa_teacache.mp4"
```

TeaCache 通过条件：

- 和 50-step reference compare 通过。
- 和上一阶段无 TeaCache 输出相比没有明显 mask 边缘闪烁、纹理漂移。
- 如果加速小于 3%，默认不作为推荐项。

## 14. Stage 7：Cache-DiT 分支

目标：在最佳非 cache 配置上只新增 Cache-DiT。Cache-DiT 允许轻微质量 tradeoff，但必须和固定
50-step reference、上一阶段非 cache 输出都做 compare。

`dit_layerwise_offload` 与 Cache-DiT 互斥，因此 cache 阶段必须保持 `--no-dit-layerwise-offload`。

### 14.1 Cache-DiT 参数矩阵

每次只改变一个策略：

| 阶段 | FN | BN | WARMUP | RDT | MC | SCM_PRESET | 用途 |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `sp2_no_offload_compile_fa_cache_rdt010` | 1 | 1 | 4 | 0.10 | 2 | medium | 保守 |
| `sp2_no_offload_compile_fa_cache_rdt012` | 1 | 1 | 4 | 0.12 | 2 | medium | 默认候选 |
| `sp2_no_offload_compile_fa_cache_rdt018` | 1 | 1 | 4 | 0.18 | 2 | medium | 较激进 |
| `sp2_no_offload_compile_fa_cache_fast` | 1 | 0 | 2 | 0.24 | 3 | fast | 吞吐优先 |

日志里如果出现 cache policy 初始化失败、fallback 或 `mask_policy None is not valid`，该轮不能作为有效成绩。

### CLI

保守 / 默认候选：

```bash
STAGE=sp2_no_offload_compile_fa_cache_rdt012
(
  export SGLANG_CACHE_DIT_ENABLED=true
  export SGLANG_CACHE_DIT_FN=1
  export SGLANG_CACHE_DIT_BN=1
  export SGLANG_CACHE_DIT_WARMUP=4
  export SGLANG_CACHE_DIT_RDT=0.12
  export SGLANG_CACHE_DIT_MC=2
  export SGLANG_CACHE_DIT_SCM_PRESET=medium
  export SGLANG_TORCH_COMPILE_MODE=max-autotune-no-cudagraphs
  ENABLE_TEACACHE=false run_videoedit_cli "$STAGE" \
    --num-gpus 2 \
    --sp-degree 2 \
    --ulysses-degree 2 \
    --ring-degree 1 \
    --no-dit-cpu-offload \
    --no-dit-layerwise-offload \
    --no-text-encoder-cpu-offload \
    --no-image-encoder-cpu-offload \
    --no-vae-cpu-offload \
    --enable-torch-compile \
    --attention-backend fa
) 2>&1 | tee "$OUT_DIR/videoedit_bench_${STAGE}.log"

compare_videoedit_candidate "$STAGE"
```

吞吐优先：

```bash
STAGE=sp2_no_offload_compile_fa_cache_fast
(
  export SGLANG_CACHE_DIT_ENABLED=true
  export SGLANG_CACHE_DIT_FN=1
  export SGLANG_CACHE_DIT_BN=0
  export SGLANG_CACHE_DIT_WARMUP=2
  export SGLANG_CACHE_DIT_RDT=0.24
  export SGLANG_CACHE_DIT_MC=3
  export SGLANG_CACHE_DIT_SCM_PRESET=fast
  export SGLANG_TORCH_COMPILE_MODE=max-autotune-no-cudagraphs
  ENABLE_TEACACHE=false run_videoedit_cli "$STAGE" \
    --num-gpus 2 \
    --sp-degree 2 \
    --ulysses-degree 2 \
    --ring-degree 1 \
    --no-dit-cpu-offload \
    --no-dit-layerwise-offload \
    --no-text-encoder-cpu-offload \
    --no-image-encoder-cpu-offload \
    --no-vae-cpu-offload \
    --enable-torch-compile \
    --attention-backend fa
) 2>&1 | tee "$OUT_DIR/videoedit_bench_${STAGE}.log"

compare_videoedit_candidate "$STAGE"
```

### Serve

```bash
STAGE=sp2_no_offload_compile_fa_cache_rdt012
(
  export SGLANG_CACHE_DIT_ENABLED=true
  export SGLANG_CACHE_DIT_FN=1
  export SGLANG_CACHE_DIT_BN=1
  export SGLANG_CACHE_DIT_WARMUP=4
  export SGLANG_CACHE_DIT_RDT=0.12
  export SGLANG_CACHE_DIT_MC=2
  export SGLANG_CACHE_DIT_SCM_PRESET=medium
  export SGLANG_TORCH_COMPILE_MODE=max-autotune-no-cudagraphs
  start_videoedit_serve \
    --num-gpus 2 \
    --sp-degree 2 \
    --ulysses-degree 2 \
    --ring-degree 1 \
    --dit-cpu-offload false \
    --dit-layerwise-offload false \
    --text-encoder-cpu-offload false \
    --image-encoder-cpu-offload false \
    --vae-cpu-offload false \
    --enable-torch-compile true \
    --attention-backend fa
) 2>&1 | tee "$OUT_DIR/videoedit_serve_${STAGE}.log"
```

```bash
STAGE=sp2_no_offload_compile_fa_cache_rdt012_warmup_request ENABLE_TEACACHE=false submit_videoedit_request
poll_videoedit_job sp2_no_offload_compile_fa_cache_rdt012_warmup_request

STAGE=sp2_no_offload_compile_fa_cache_rdt012 ENABLE_TEACACHE=false submit_videoedit_request
poll_videoedit_job sp2_no_offload_compile_fa_cache_rdt012
compare_videoedit_candidate api_sp2_no_offload_compile_fa_cache_rdt012 "$OUT_DIR/${VIDEO_BASENAME}_api_sp2_no_offload_compile_fa_cache_rdt012.mp4"
```

Cache-DiT 通过条件：

- 逐帧 compare 通过。
- SSIM / MSE / MAE 不只看平均值，必须检查最差帧。
- 人工抽检 mask 边缘、快速运动区域、背景纹理和颜色稳定性。
- 若 fast 档通过自动指标但人工抽检有明显抖动，推荐降级到 `rdt012` 或 `rdt010`。

## 15. Quant Branch：量化专项

目标：降低 DiT 显存并评估矩阵乘吞吐。量化只作为独立分支，不覆盖 bf16 主线结论。

建议优先只量化 `WanVideoEditTransformer3DModel`，不量化 VAE 和 text encoder。先测已有 `fp8_dynamic`
或权重量化 checkpoint，再评估更低比特；VideoEdit 对 mask 边缘、身份一致性和纹理稳定性敏感，不建议一开始做全模型 4bit。

### CLI

```bash
export QUANT_TRANSFORMER_PATH=/path/to/quantized/videoedit/transformer

STAGE=quant_branch_fp8_dynamic
SGLANG_CACHE_DIT_ENABLED=false ENABLE_TEACACHE=false CLI_TRANSFORMER_PATH="$QUANT_TRANSFORMER_PATH" \
run_videoedit_cli "$STAGE" \
  --transformer-quantization fp8_dynamic \
  --num-gpus 2 \
  --sp-degree 2 \
  --ulysses-degree 2 \
  --ring-degree 1 \
  --no-dit-cpu-offload \
  --no-dit-layerwise-offload \
  --no-text-encoder-cpu-offload \
  --no-image-encoder-cpu-offload \
  --no-vae-cpu-offload \
  --attention-backend fa \
  2>&1 | tee "$OUT_DIR/videoedit_bench_${STAGE}.log"

compare_videoedit_candidate "$STAGE"
```

如果量化 checkpoint 不需要 `--transformer-quantization fp8_dynamic`，删除该参数并在结果表中写明实际加载方式。

### Serve

```bash
export QUANT_TRANSFORMER_PATH=/path/to/quantized/videoedit/transformer

STAGE=quant_branch_fp8_dynamic
SGLANG_CACHE_DIT_ENABLED=false \
SERVE_TRANSFORMER_PATH="$QUANT_TRANSFORMER_PATH" \
start_videoedit_serve \
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
  --transformer-quantization fp8_dynamic \
  2>&1 | tee "$OUT_DIR/videoedit_serve_${STAGE}.log"
```

量化验收：

- 保留原 bf16 transformer 输出作为质量对照。
- 单窗口固定 seed 回归，记录逐帧 SSIM/MSE/MAE/PSNR。
- 检查 cross-attention、QK norm、time/text embedding 等敏感模块是否需要量化白名单。
- 检查日志中 runtime quant config 是否真的生效。
- 量化和 Cache-DiT、SP、attention backend 的组合必须逐项重新测试。

## 16. Offload Branch：显存受限分支

该分支从 `sp1_offload` 出发，目标是让任务能跑起来，不作为性能主线。质量仍和固定 reference 视频对比。
Offload 和 Cache-DiT 不要组合，尤其 `dit_layerwise_offload` 与 Cache-DiT 存在硬冲突。

推荐分级：

1. 保持 VAE 默认显存策略。
2. 开启 `text_encoder_cpu_offload`。
3. 开启 `vae_cpu_offload`。
4. 仍然 OOM 时开启 `dit_layerwise_offload`，并设置较小 prefetch。
5. 最后才考虑 `dit_cpu_offload`。

### CLI

```bash
STAGE=offload_branch
SGLANG_CACHE_DIT_ENABLED=false ENABLE_TEACACHE=false \
run_videoedit_cli "$STAGE" \
  --num-gpus 1 \
  --sp-degree 1 \
  --ulysses-degree 1 \
  --ring-degree 1 \
  --text-encoder-cpu-offload \
  --vae-cpu-offload \
  --dit-layerwise-offload \
  --dit-offload-prefetch-size 1 \
  2>&1 | tee "$OUT_DIR/videoedit_bench_${STAGE}.log"

compare_videoedit_candidate "$STAGE"
```

### Serve

```bash
STAGE=offload_branch
SGLANG_CACHE_DIT_ENABLED=false \
start_videoedit_serve \
  --num-gpus 1 \
  --sp-degree 1 \
  --ulysses-degree 1 \
  --ring-degree 1 \
  --text-encoder-cpu-offload true \
  --vae-cpu-offload true \
  --dit-layerwise-offload true \
  --dit-offload-prefetch-size 1 \
  2>&1 | tee "$OUT_DIR/videoedit_serve_${STAGE}.log"
```

```bash
STAGE=offload_branch ENABLE_TEACACHE=false submit_videoedit_request
poll_videoedit_job offload_branch
compare_videoedit_candidate api_offload_branch "$OUT_DIR/${VIDEO_BASENAME}_api_offload_branch.mp4"
```

## 17. 结果记录模板

每个阶段填写一行。Serve 结果以第二次同配置请求为准。

| 阶段 | 新增优化 | CLI/serve | expected backend | actual backend | forward(s) | denoise(s) | text(s) | VAE encode(s) | decode(s) | wall/inference(s) | peak MB | compare | 结论 |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `sp1_offload` | baseline | CLI + serve | default |  |  |  |  |  |  |  |  | reference |  |
| `sp1_no_offload` | no offload | CLI + serve | default |  |  |  |  |  |  |  |  |  |  |
| `sp1_no_offload_compile` | torch.compile | CLI + serve | default |  |  |  |  |  |  |  |  |  |  |
| `sp1_no_offload_compile_torch_sdpa` | attention backend | CLI + serve | torch_sdpa |  |  |  |  |  |  |  |  |  |  |
| `sp1_no_offload_compile_fa` | attention backend | CLI + serve | fa |  |  |  |  |  |  |  |  |  |  |
| `sp1_no_offload_compile_sage_attn` | attention backend | CLI + serve | sage_attn |  |  |  |  |  |  |  |  |  |  |
| `sp1_no_offload_compile_sage_attn_3` | attention backend | CLI + serve | sage_attn_3 |  |  |  |  |  |  |  |  |  |  |
| `sp2_no_offload_torch_sdpa` | SP2 + backend | CLI + serve | torch_sdpa |  |  |  |  |  |  |  |  |  |  |
| `sp2_no_offload_fa` | SP2 + backend | CLI + serve | fa |  |  |  |  |  |  |  |  |  |  |
| `sp2_no_offload_sage_attn` | SP2 + backend | CLI + serve | sage_attn |  |  |  |  |  |  |  |  |  |  |
| `sp2_no_offload_sage_attn_3` | SP2 + backend | CLI + serve | sage_attn_3 |  |  |  |  |  |  |  |  |  |  |
| `sp2_ring_no_offload_fa` | Ring SP | CLI + serve | fa |  |  |  |  |  |  |  |  |  |  |
| `tp2_no_offload_fa` | TP2 | CLI + serve | fa |  |  |  |  |  |  |  |  |  |  |
| `sp2_no_offload_compile_fa` | torch.compile | CLI + serve | fa |  |  |  |  |  |  |  |  |  |  |
| `sp2_no_offload_compile_fa_teacache` | TeaCache | CLI + serve | fa |  |  |  |  |  |  |  |  |  |  |
| `sp2_no_offload_compile_fa_cache_rdt010` | Cache-DiT RDT 0.10 | CLI + serve | fa |  |  |  |  |  |  |  |  |  |  |
| `sp2_no_offload_compile_fa_cache_rdt012` | Cache-DiT RDT 0.12 | CLI + serve | fa |  |  |  |  |  |  |  |  |  |  |
| `sp2_no_offload_compile_fa_cache_rdt018` | Cache-DiT RDT 0.18 | CLI + serve | fa |  |  |  |  |  |  |  |  |  |  |
| `sp2_no_offload_compile_fa_cache_fast` | Cache-DiT fast | CLI + serve | fa |  |  |  |  |  |  |  |  |  |  |
| `quant_branch_fp8_dynamic` | quant | CLI + serve | fa |  |  |  |  |  |  |  |  |  |  |
| `offload_branch` | memory branch | CLI + serve | default |  |  |  |  |  |  |  |  |  |  |

结论只从这个表里产生，不混用历史低步数、改 CFG 或不同输入的数据。

## 18. 结果判定规则

推荐默认配置时按以下优先级判定：

1. 必须完成 CLI 和 serve，且 serve 第二次同配置请求成功。
2. 必须通过固定 50-step reference 的 compare。
3. 必须确认 `actual_backend`，fallback 结果按 fallback 后端归类。
4. 非 cache 主线优先稳定和质量；cache 分支只在质量通过且加速明显时作为吞吐优先配置。
5. 显存受限配置只和 offload branch 内部比较，不和性能主线混排。
6. 量化分支必须单独标注质量风险，不能替代 bf16 主线，除非多素材质量验证通过。

最低质量阈值：

```text
min_ssim >= 0.90
max_mse <= 150.0
max_mae <= 8.0
max_failed_frame_ratio <= 0.05
allow_frame_count_delta <= 1
```

人工抽检重点：

- mask 边缘闪烁。
- 花瓣和背景草地纹理漂移。
- 颜色稳定性和过曝。
- 窗口边界和首帧/尾帧是否异常。
- Cache-DiT fast 和量化输出需要逐帧拖动检查，不只看自动指标。

## 19. 后续代码级优化

这些需要源码改动，不能和无代码调参混在同一轮评估：

- Prompt embedding 跨窗口缓存：同一 prompt、negative prompt、dtype 和 tokenizer 配置下复用 text encoder 输出。
- VAE encode/decode 并行接入：确认 `vae_sp` 到 `WanVAEConfig.use_parallel_encode/use_parallel_decode` 的链路，并验证跨 rank gather。
- CFG parallel：当前 VideoEdit 对 `enable_cfg_parallel` 未实现；若实现，需要单独验证 conditional/unconditional rank 的同步和 latents 广播。
- 条件 latent 缓存：只先缓存完全相同窗口；overlap 区域 latent 复用需要额外验证 temporal boundary。
- Serve Ring SP 启动失败排查：如果 50-step 仍出现 health check 或 scheduler timeout，需要独立分析端口、进程清理和分布式初始化日志。
- SageAttention3 on Blackwell：只在 Blackwell + CUDA 12.8+ + PyTorch 2.8+ 环境重新评估；A100 结果不能外推。

代码级优化必须重新生成 perf JSON 和 compare JSON，不能沿用本文件里的无代码优化结论。
