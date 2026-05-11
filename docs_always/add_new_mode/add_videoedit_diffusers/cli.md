# VideoEdit CLI 与 Serve 测试命令

本文档记录当前机器双卡 A100 80GB 上 VideoEdit 的端到端测试命令。命令以
`python/sglang/multimodal_gen/runtime/videoedit/cli.py` 和
`python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py` 为准。

> 约定：下面的命令不修改代码，只运行 CLI、启动 serve、提交请求或做输出检查。

## 0. 公共路径与环境

```bash
conda deactivate
source /mnt/shanhai-ai/shanhai-workspace/zhouhao6/env/sglang/bin/activate
cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang

export MODEL_PATH=/mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model
export TRANSFORMER_PATH=/mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model/transformer
export INPUT_VIDEO=/mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pexel_test_data_0410/videos/15108907_3840_2160_50fps_short.mp4
export INPUT_MASK=/mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pexel_test_data_0410/masks/15108907_3840_2160_50fps_No_bbox_mask.mp4
export OUT_DIR=/mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs
export PROMPT="A close-up of an orange flower with a yellow center, remaining in focus against a blurred green grass background throughout the video."
```

当前测试素材的实际处理区域可从输出 metadata 看到：原输入 81 帧、crop 约
`672x624` 对齐尺寸，因此双卡 A100 80GB 应优先测试不 offload 的配置。

## 1. 单卡基线：默认行为

用于确认链路可用。默认会在 Wan/VideoEdit 上自动打开 `dit_layerwise_offload`，
显存低但速度不是最高。

```bash
python -m sglang.multimodal_gen.runtime.videoedit.cli repair \
  --model-path "$MODEL_PATH" \
  --transformer-path "$TRANSFORMER_PATH" \
  --prompt "$PROMPT" \
  --video-input-path "$INPUT_VIDEO" \
  --mask-input-path "$INPUT_MASK" \
  --output-path "$OUT_DIR" \
  --output-file-name 15108907_3840_2160_50fps_1gpu_default.mp4 \
  --num-frames 81 \
  --infer-len 81 \
  --overlap 0 \
  --num-inference-steps 20 \
  --guidance-scale 5.0 \
  --seed 42 \
  --dtype bf16 \
  --enable-paste-back \
  --drop-reference-frame \
  --warmup \
  --warmup-steps 1 \
  --perf-dump-path "$OUT_DIR/videoedit_perf_1gpu_default.json"
  
```

## 2. 双卡 A100 主测：SP=2，无 offload

这是当前机器优先测试的最高性能基线。`sp_degree=2` 会自动使用序列维度切分；
显式关闭 CPU/offload，避免 H2D 拷贝拖慢 denoising。

```bash
python -m sglang.multimodal_gen.runtime.videoedit.cli repair \
  --model-path "$MODEL_PATH" \
  --transformer-path "$TRANSFORMER_PATH" \
  --prompt "$PROMPT" \
  --video-input-path "$INPUT_VIDEO" \
  --mask-input-path "$INPUT_MASK" \
  --output-path "$OUT_DIR" \
  --output-file-name 15108907_3840_2160_50fps_sp2_no_offload.mp4 \
  --num-gpus 2 \
  --sp-degree 2 \
  --ulysses-degree 2 \
  --ring-degree 1 \
  --no-dit-cpu-offload \
  --no-dit-layerwise-offload \
  --no-text-encoder-cpu-offload \
  --no-image-encoder-cpu-offload \
  --no-vae-cpu-offload \
  --num-frames 81 \
  --infer-len 81 \
  --overlap 0 \
  --num-inference-steps 20 \
  --guidance-scale 5.0 \
  --seed 42 \
  --dtype bf16 \
  --enable-paste-back \
  --drop-reference-frame \
  --perf-dump-path "$OUT_DIR/videoedit_perf_sp2_no_offload.json"

python -m sglang.multimodal_gen.runtime.videoedit.cli repair \
  --model-path "$MODEL_PATH" \
  --transformer-path "$TRANSFORMER_PATH" \
  --prompt "$PROMPT" \
  --video-input-path "$INPUT_VIDEO" \
  --mask-input-path "$INPUT_MASK" \
  --output-path "$OUT_DIR" \
  --output-file-name 15108907_3840_2160_50fps_sp2_no_offload.mp4 \
  --num-gpus 2 \
  --sp-degree 2 \
  --ulysses-degree 1 \
  --ring-degree 2 \
  --no-dit-cpu-offload \
  --no-dit-layerwise-offload \
  --no-text-encoder-cpu-offload \
  --no-image-encoder-cpu-offload \
  --no-vae-cpu-offload \
  --num-frames 81 \
  --infer-len 81 \
  --overlap 0 \
  --num-inference-steps 20 \
  --guidance-scale 5.0 \
  --seed 42 \
  --dtype bf16 \
  --enable-paste-back \
  --drop-reference-frame \
  --perf-dump-path "$OUT_DIR/videoedit_perf_sp2r_no_offload.json"
```

## 3. 双卡 TP=2 对照

用于和 SP=2 对比。不要和 SP 同时打开，避免混合并行影响排查。

```bash
python -m sglang.multimodal_gen.runtime.videoedit.cli repair \
  --model-path "$MODEL_PATH" \
  --transformer-path "$TRANSFORMER_PATH" \
  --prompt "$PROMPT" \
  --video-input-path "$INPUT_VIDEO" \
  --mask-input-path "$INPUT_MASK" \
  --output-path "$OUT_DIR" \
  --output-file-name 15108907_3840_2160_50fps_tp2_no_offload.mp4 \
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
  --num-frames 81 \
  --infer-len 81 \
  --overlap 0 \
  --num-inference-steps 20 \
  --guidance-scale 5.0 \
  --seed 42 \
  --dtype bf16 \
  --enable-paste-back \
  --drop-reference-frame \
  --perf-dump-path "$OUT_DIR/videoedit_perf_tp2_no_offload.json"
```

## 4. torch.compile 测试

`torch.compile` 首次运行会包含编译开销。性能评估时至少跑两次：
第一次预热/编译，第二次看稳定耗时。CLI 每次会重新拉起本地 server，因此编译缓存收益
可能不如常驻 serve 明显；更推荐在 serve 模式下测。

```bash
python -m sglang.multimodal_gen.runtime.videoedit.cli repair \
  --model-path "$MODEL_PATH" \
  --transformer-path "$TRANSFORMER_PATH" \
  --prompt "$PROMPT" \
  --video-input-path "$INPUT_VIDEO" \
  --mask-input-path "$INPUT_MASK" \
  --output-path "$OUT_DIR" \
  --output-file-name 15108907_3840_2160_50fps_sp2_compile_warmup.mp4 \
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
  --warmup \
  --warmup-steps 1 \
  --num-frames 81 \
  --infer-len 81 \
  --overlap 0 \
  --num-inference-steps 20 \
  --guidance-scale 5.0 \
  --seed 42 \
  --dtype bf16 \
  --enable-paste-back \
  --drop-reference-frame \
  --perf-dump-path "$OUT_DIR/videoedit_perf_sp2_compile_warmup.json"
```

第二次稳定测试：

```bash
python -m sglang.multimodal_gen.runtime.videoedit.cli repair \
  --model-path "$MODEL_PATH" \
  --transformer-path "$TRANSFORMER_PATH" \
  --prompt "$PROMPT" \
  --video-input-path "$INPUT_VIDEO" \
  --mask-input-path "$INPUT_MASK" \
  --output-path "$OUT_DIR" \
  --output-file-name 15108907_3840_2160_50fps_sp2_compile.mp4 \
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
  --num-frames 81 \
  --infer-len 81 \
  --overlap 0 \
  --num-inference-steps 20 \
  --guidance-scale 5.0 \
  --seed 42 \
  --dtype bf16 \
  --enable-paste-back \
  --drop-reference-frame \
  --perf-dump-path "$OUT_DIR/videoedit_perf_sp2_compile.json"
```

## 5. Cache-DiT 测试

Cache-DiT 通过环境变量启用，和 `dit_layerwise_offload` 互斥，必须显式
`--no-dit-layerwise-offload`。Cache-DiT 可能改变数值，需要和无 cache 输出做逐帧比较。

### 5.1 保守 Cache-DiT

```bash
SGLANG_CACHE_DIT_ENABLED=true \
SGLANG_CACHE_DIT_FN=1 \
SGLANG_CACHE_DIT_BN=1 \
SGLANG_CACHE_DIT_WARMUP=4 \
SGLANG_CACHE_DIT_RDT=0.12 \
SGLANG_CACHE_DIT_MC=2 \
python -m sglang.multimodal_gen.runtime.videoedit.cli repair \
  --model-path "$MODEL_PATH" \
  --transformer-path "$TRANSFORMER_PATH" \
  --prompt "$PROMPT" \
  --video-input-path "$INPUT_VIDEO" \
  --mask-input-path "$INPUT_MASK" \
  --output-path "$OUT_DIR" \
  --output-file-name 15108907_3840_2160_50fps_sp2_cache_conservative.mp4 \
  --num-gpus 2 \
  --sp-degree 2 \
  --ulysses-degree 2 \
  --ring-degree 1 \
  --no-dit-cpu-offload \
  --no-dit-layerwise-offload \
  --no-text-encoder-cpu-offload \
  --no-image-encoder-cpu-offload \
  --no-vae-cpu-offload \
  --num-frames 81 \
  --infer-len 81 \
  --overlap 0 \
  --num-inference-steps 20 \
  --guidance-scale 5.0 \
  --seed 42 \
  --dtype bf16 \
  --enable-paste-back \
  --drop-reference-frame \
  --perf-dump-path "$OUT_DIR/videoedit_perf_sp2_cache_conservative.json"
```

### 5.2 激进 Cache-DiT

```bash
SGLANG_CACHE_DIT_ENABLED=true \
SGLANG_CACHE_DIT_FN=1 \
SGLANG_CACHE_DIT_BN=0 \
SGLANG_CACHE_DIT_WARMUP=2 \
SGLANG_CACHE_DIT_RDT=0.24 \
SGLANG_CACHE_DIT_MC=3 \
SGLANG_CACHE_DIT_SCM_PRESET=fast \
python -m sglang.multimodal_gen.runtime.videoedit.cli repair \
  --model-path "$MODEL_PATH" \
  --transformer-path "$TRANSFORMER_PATH" \
  --prompt "$PROMPT" \
  --video-input-path "$INPUT_VIDEO" \
  --mask-input-path "$INPUT_MASK" \
  --output-path "$OUT_DIR" \
  --output-file-name 15108907_3840_2160_50fps_sp2_cache_fast.mp4 \
  --num-gpus 2 \
  --sp-degree 2 \
  --ulysses-degree 2 \
  --ring-degree 1 \
  --no-dit-cpu-offload \
  --no-dit-layerwise-offload \
  --no-text-encoder-cpu-offload \
  --no-image-encoder-cpu-offload \
  --no-vae-cpu-offload \
  --num-frames 81 \
  --infer-len 81 \
  --overlap 0 \
  --num-inference-steps 20 \
  --guidance-scale 5.0 \
  --seed 42 \
  --dtype bf16 \
  --enable-paste-back \
  --drop-reference-frame \
  --perf-dump-path "$OUT_DIR/videoedit_perf_sp2_cache_fast.json"
```

## 6. TeaCache 测试

TeaCache 是请求级参数 `--enable-teacache` / `"enable_teacache": true`。
它和 Cache-DiT 不是同一个开关，建议单独测，先不要和 Cache-DiT 混在同一组。

```bash
python -m sglang.multimodal_gen.runtime.videoedit.cli repair \
  --model-path "$MODEL_PATH" \
  --transformer-path "$TRANSFORMER_PATH" \
  --prompt "$PROMPT" \
  --video-input-path "$INPUT_VIDEO" \
  --mask-input-path "$INPUT_MASK" \
  --output-path "$OUT_DIR" \
  --output-file-name 15108907_3840_2160_50fps_sp2_teacache.mp4 \
  --num-gpus 2 \
  --sp-degree 2 \
  --ulysses-degree 2 \
  --ring-degree 1 \
  --no-dit-cpu-offload \
  --no-dit-layerwise-offload \
  --no-text-encoder-cpu-offload \
  --no-image-encoder-cpu-offload \
  --no-vae-cpu-offload \
  --enable-teacache \
  --num-frames 81 \
  --infer-len 81 \
  --overlap 0 \
  --num-inference-steps 20 \
  --guidance-scale 5.0 \
  --seed 42 \
  --dtype bf16 \
  --enable-paste-back \
  --drop-reference-frame \
  --perf-dump-path "$OUT_DIR/videoedit_perf_sp2_teacache.json"
```

## 7. 低步数 smoke/perf 快测

用于快速验证命令组合是否能跑通，不用于最终质量判断。

```bash
python -m sglang.multimodal_gen.runtime.videoedit.cli repair \
  --model-path "$MODEL_PATH" \
  --transformer-path "$TRANSFORMER_PATH" \
  --prompt "$PROMPT" \
  --video-input-path "$INPUT_VIDEO" \
  --mask-input-path "$INPUT_MASK" \
  --output-path "$OUT_DIR" \
  --output-file-name 15108907_3840_2160_50fps_sp2_2steps_smoke.mp4 \
  --num-gpus 2 \
  --sp-degree 2 \
  --ulysses-degree 2 \
  --ring-degree 1 \
  --no-dit-cpu-offload \
  --no-dit-layerwise-offload \
  --no-text-encoder-cpu-offload \
  --no-image-encoder-cpu-offload \
  --no-vae-cpu-offload \
  --num-frames 81 \
  --infer-len 81 \
  --overlap 0 \
  --num-inference-steps 2 \
  --guidance-scale 5.0 \
  --seed 42 \
  --dtype bf16 \
  --enable-paste-back \
  --drop-reference-frame \
  --perf-dump-path "$OUT_DIR/videoedit_perf_sp2_2steps_smoke.json"
```

## 8. Serve：双卡 A100 常驻服务

Serve 模式更适合测试 `torch.compile`、warmup 和多次请求，因为模型常驻进程内。

```bash
VIDEOEDIT_QUEUE_CAPACITY=1 \
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
  --output-path "$OUT_DIR" \
  --input-save-path /tmp/sglang-videoedit-inputs \
  --transformer-path "$TRANSFORMER_PATH"
```

提交任务：

```bash
curl -s -X POST http://127.0.0.1:30000/v1/videos/repairs \
  -H 'Content-Type: application/json' \
  -d '{
    "task_id": "pexel_15108907_sp2_no_offload",
    "prompt": "A close-up of an orange flower with a yellow center, remaining in focus against a blurred green grass background throughout the video.",
    "video_input_path": "/mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pexel_test_data_0410/videos/15108907_3840_2160_50fps_short.mp4",
    "mask_input_path": "/mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pexel_test_data_0410/masks/15108907_3840_2160_50fps_No_bbox_mask.mp4",
    "output_storage": "local",
    "output_path": "/mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs/15108907_3840_2160_50fps_api_sp2_no_offload.mp4",
    "num_frames": 81,
    "infer_len": 81,
    "overlap": 0,
    "num_inference_steps": 20,
    "guidance_scale": 5.0,
    "seed": 42,
    "dtype": "bf16",
    "enable_paste_back": true,
    "drop_reference_frame": true,
    "perf_dump_path": "/mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs/videoedit_perf_api_sp2_no_offload.json"
  }'
```

轮询：

```bash
JOB_ID=pexel_15108907_sp2_no_offload
while true; do
  resp=$(curl -s "http://127.0.0.1:30000/v1/videos/${JOB_ID}")
  python -c 'import json,sys; d=json.load(sys.stdin); print(d.get("status"), d.get("progress"), d.get("file_path") or d.get("url"), d.get("inference_time_s"))' <<< "$resp"
  status=$(python -c 'import json,sys; print(json.load(sys.stdin).get("status"))' <<< "$resp")
  [ "$status" = "completed" ] && break
  [ "$status" = "failed" ] && exit 1
  sleep 5
done
```

## 9. Serve：torch.compile 常驻服务

首次请求主要用于编译预热，第二个同配置请求用于统计稳定性能。

```bash
VIDEOEDIT_QUEUE_CAPACITY=1 \
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
  --enable-torch-compile true \
  --warmup true \
  --warmup-steps 1 \
  --output-path "$OUT_DIR" \
  --input-save-path /tmp/sglang-videoedit-inputs \
  --transformer-path "$TRANSFORMER_PATH"
```

## 10. Serve：Cache-DiT 常驻服务

不要同时打开 `--dit-layerwise-offload true`。

```bash
SGLANG_CACHE_DIT_ENABLED=true \
SGLANG_CACHE_DIT_FN=1 \
SGLANG_CACHE_DIT_BN=1 \
SGLANG_CACHE_DIT_WARMUP=4 \
SGLANG_CACHE_DIT_RDT=0.12 \
SGLANG_CACHE_DIT_MC=2 \
VIDEOEDIT_QUEUE_CAPACITY=1 \
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
  --output-path "$OUT_DIR" \
  --input-save-path /tmp/sglang-videoedit-inputs \
  --transformer-path "$TRANSFORMER_PATH"
```

提交请求时把 `task_id`、`output_path`、`perf_dump_path` 改成 cache 对应名字即可。

## 11. 输出检查

检查输出视频基础信息：

```bash
python - <<'PY'
import cv2

path = "/mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs/15108907_3840_2160_50fps_sp2_no_offload.mp4"
cap = cv2.VideoCapture(path)
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()
print({"frames": frames, "width": width, "height": height, "fps": fps})
assert frames in (80, 81), frames
assert width > 0 and height > 0
PY
```

检查 metadata：

```bash
python -m json.tool "$OUT_DIR/15108907_3840_2160_50fps_sp2_no_offload.videoedit.json"
```

## 12. 输出对比

性能优化组必须和主测基线做逐帧比较。Cache-DiT、TeaCache、torch.compile 都建议保留
JSON report。

```bash
python python/sglang/multimodal_gen/runtime/videoedit/compare.py \
  --reference "$OUT_DIR/15108907_3840_2160_50fps_sp2_no_offload.mp4" \
  --candidate "$OUT_DIR/15108907_3840_2160_50fps_sp2_cache_conservative.mp4" \
  --report-json "$OUT_DIR/videoedit_compare_sp2_cache_conservative.json" \
  --min-ssim 0.90 \
  --max-mse 150.0 \
  --max-mae 8.0 \
  --allow-frame-count-delta 1 \
  --max-failed-frame-ratio 0.05
```

如果两侧帧数相差 1 且 candidate 多出参考帧：

```bash
python python/sglang/multimodal_gen/runtime/videoedit/compare.py \
  --reference "$OUT_DIR/15108907_3840_2160_50fps_sp2_no_offload.mp4" \
  --candidate "$OUT_DIR/15108907_3840_2160_50fps_sp2_cache_conservative.mp4" \
  --drop-candidate-first-frame \
  --report-json "$OUT_DIR/videoedit_compare_sp2_cache_conservative.json"
```

## 13. 性能记录口径

优先记录这些值：

- CLI 日志中的 `VideoEditDenoisingStage finished in ... seconds`
- CLI JSON 里的 `metrics.total_duration_ms`
- `perf_dump_path` 生成的 JSON
- 输出 metadata 中的 `aligned_h/aligned_w`、`num_input_frames` 和窗口数量
- `nvidia-smi` 中两张 A100 的峰值显存与 GPU 利用率

当前已知单卡默认 offload 跑通记录：20 steps denoising 约 `288-335s`，
`total_duration_ms` 约 `318-366s`。双卡 A100 主测目标是优先验证
`sp2_no_offload` 是否显著低于该耗时。
