# VideoEdit 优化方案

本文档记录 VideoEdit-diffusers 接入 SGLang 后的性能优化路线。命令参考
[`cli.md`](./cli.md) 和 [`README.md`](./README.md)，优化时必须先固定基线，再按顺序逐步累加优化项。

## 1. 硬约束

本轮优化只比较系统实现、算子、并行、cache、offload 和量化，不通过降低采样质量换速度：

- 固定同一视频、mask、prompt、seed、`num_frames`、`infer_len`、`overlap`、`num_inference_steps`、`guidance_scale`。
- 所有 CLI 和 serve 输出的质量对比基准视频固定为 `$OUT_DIR/reference/15108907_3840_2160_50fps.mp4`。
- `num_inference_steps` 固定为 `20`，不允许降低 steps。
- `dynamic_cfg_max_step` 固定为默认 `15`，不允许改成 `10/12` 等较低值。
- 除 Cache-DiT / TeaCache 和量化外，不允许引入其他可能降低质量的方法。
- 每个优化阶段同时给 CLI 和 serve 命令。
- 优化项逐渐累加，每一步只新增一个优化维度；记录时和上一阶段、主基线都做对比。
- 每个命令都启用 `--warmup --warmup-steps 1` 和 `--perf-dump-path`。
- 每个输出都必须保留 perf JSON、视频 metadata、CUDA memory、GPU utilization 和逐窗口日志。
- 每个候选输出都必须和固定基线做输出检查；cache 和量化必须做逐帧 compare。

不纳入推荐优化的历史项：

- 低步数：例如 `num_inference_steps=16/18`。
- 缩短 dynamic CFG：例如 `dynamic_cfg_max_step=10/12`。
- CLI 冷启动下的 `torch.compile` wall time：可以记录，但不作为 compile 收益结论。

## 2. 公共环境

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

固定请求参数：

```text
num_frames = 81
infer_len = 81
overlap = 0
num_inference_steps = 20
guidance_scale = 5.0
dynamic_cfg = true
dynamic_cfg_max_step = 15
seed = 42
dtype = bf16
enable_paste_back = true
drop_reference_frame = true
warmup = true
warmup_steps = 1
```

## 3. 记录口径

每个阶段至少记录：

- 单窗口耗时：从每窗口日志或 stage 日志提取。
- 端到端耗时：CLI wall time 或 serve 请求 `inference_time_s`。
- 分段耗时：`VideoEditTextEncodingStage`、`VideoEditConditionEncodingStage`、`VideoEditDenoisingStage`、`VideoEditDecodingStage`、postprocess。
- 显存：`perf_dump_path` 中的 peak allocated/reserved，外部补充 `nvidia-smi` 峰值。
- GPU utilization：运行期间用 `nvidia-smi dmon` 或等价监控记录。
- 输出质量：视频帧数、分辨率、metadata、逐帧 compare JSON。

推荐命名：

```text
输出视频：$OUT_DIR/15108907_3840_2160_50fps_<stage>.mp4
perf JSON：$OUT_DIR/videoedit_perf_<stage>.json
compare JSON：$OUT_DIR/videoedit_compare_<stage>.json
bench 日志：$OUT_DIR/videoedit_bench_<stage>.log
```

## 4. 优化阶梯

主线从 `sp1_offload` 开始，逐步累加：

1. `sp1_offload`：基线，优先保证低显存和可跑通。
2. `sp1_no_offload`：关闭 offload，观察同一单卡配置下的 latency / 显存交换。
3. `sp1_no_offload_compile`：在 no-offload 基础上加 `torch.compile`。
4. `sp1_no_offload_compile_attn`：在 compile 基础上加 attention backend，先确认后端真实生效。
5. `sp2_no_offload_<backend>`：扩展到双卡 SP，并分别对比 `torch_sdpa`、`fa`、`sage_attn`、`sage_attn_3`。
6. `sp2_ring_no_offload_fa` / `tp2_no_offload_fa`：只改变并行策略，对比 Ulysses SP、Ring SP、TP。
7. `sp2_no_offload_compile_<backend>`：只新增 `torch.compile`，serve 模式至少两次请求，以第二次为准。
8. `sp2_no_offload_compile_fa_teacache`：在最佳非 cache 配置上单独加 TeaCache。
9. `sp2_no_offload_compile_fa_cache_<policy>`：在最佳非 cache 配置上单独加 Cache-DiT。
10. `quant_branch`：在最佳非量化配置上替换量化 DiT 权重。
11. `offload_branch`：显存受限分支，不并入性能主线。

`torch.compile` 可以用，但最终收益优先看 serve 常驻模式。CLI 每次重新拉起进程，compile 缓存复用差，只用于验证能否跑通和记录冷启动成本。

## 5. Stage 0：SP1 Offload 基线

目标：固定最低显存、最稳可跑的性能起点。该阶段不追求最快；质量对比不以本阶段输出为 reference，而统一以 `$OUT_DIR/reference/15108907_3840_2160_50fps.mp4` 为 reference。

### CLI

```bash
python -m sglang.multimodal_gen.runtime.videoedit.cli repair \
  --model-path "$MODEL_PATH" \
  --transformer-path "$TRANSFORMER_PATH" \
  --prompt "$PROMPT" \
  --video-input-path "$INPUT_VIDEO" \
  --mask-input-path "$INPUT_MASK" \
  --output-path "$OUT_DIR" \
  --output-file-name 15108907_3840_2160_50fps_sp1_offload.mp4 \
  --num-gpus 1 \
  --sp-degree 1 \
  --ulysses-degree 1 \
  --ring-degree 1 \
  --dit-cpu-offload \
  --dit-layerwise-offload \
  --text-encoder-cpu-offload \
  --image-encoder-cpu-offload \
  --vae-cpu-offload \
  --num-frames 81 \
  --infer-len 81 \
  --overlap 0 \
  --num-inference-steps 20 \
  --guidance-scale 5.0 \
  --dynamic-cfg \
  --dynamic-cfg-max-step 15 \
  --seed 42 \
  --dtype bf16 \
  --enable-paste-back \
  --drop-reference-frame \
  --warmup \
  --warmup-steps 1 \
  --perf-dump-path "$OUT_DIR/videoedit_perf_sp1_offload.json"
```

### Serve

```bash
VIDEOEDIT_QUEUE_CAPACITY=1 \
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

```bash
curl -s -X POST http://127.0.0.1:30000/v1/videos/repairs \
  -H 'Content-Type: application/json' \
  -d '{
    "task_id": "sp1_offload",
    "prompt": "A close-up of an orange flower with a yellow center, remaining in focus against a blurred green grass background throughout the video.",
    "video_input_path": "/mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pexel_test_data_0410/videos/15108907_3840_2160_50fps_short.mp4",
    "mask_input_path": "/mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pexel_test_data_0410/masks/15108907_3840_2160_50fps_No_bbox_mask.mp4",
    "output_storage": "local",
    "output_path": "/mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs/15108907_3840_2160_50fps_api_sp1_offload.mp4",
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
    "perf_dump_path": "/mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs/videoedit_perf_api_sp1_offload.json"
  }'
```

## 6. Stage 1：SP1 No Offload

目标：只关闭 offload，观察同一单卡配置下的速度收益和峰值显存变化。

### CLI

```bash
python -m sglang.multimodal_gen.runtime.videoedit.cli repair \
  --model-path "$MODEL_PATH" \
  --transformer-path "$TRANSFORMER_PATH" \
  --prompt "$PROMPT" \
  --video-input-path "$INPUT_VIDEO" \
  --mask-input-path "$INPUT_MASK" \
  --output-path "$OUT_DIR" \
  --output-file-name 15108907_3840_2160_50fps_sp1_no_offload.mp4 \
  --num-gpus 1 \
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
  --dynamic-cfg \
  --dynamic-cfg-max-step 15 \
  --seed 42 \
  --dtype bf16 \
  --enable-paste-back \
  --drop-reference-frame \
  --warmup \
  --warmup-steps 1 \
  --perf-dump-path "$OUT_DIR/videoedit_perf_sp1_no_offload.json"
```

### Serve

```bash
VIDEOEDIT_QUEUE_CAPACITY=1 \
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
  --warmup true \
  --warmup-steps 1 \
  --output-path "$OUT_DIR" \
  --input-save-path /tmp/sglang-videoedit-inputs \
  --transformer-path "$TRANSFORMER_PATH"
```

提交请求时使用与 Stage 0 相同 JSON，仅替换：

```json
{
  "task_id": "sp1_no_offload",
  "output_path": "/mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs/15108907_3840_2160_50fps_api_sp1_no_offload.mp4",
  "perf_dump_path": "/mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs/videoedit_perf_api_sp1_no_offload.json"
}
```

## 7. Stage 2：加 torch.compile

目标：在 `sp1_no_offload` 基础上只新增 `torch.compile`。CLI 可跑，但正式结论看 serve 同 shape 多请求。

### CLI

```bash
SGLANG_TORCH_COMPILE_MODE=max-autotune-no-cudagraphs \
python -m sglang.multimodal_gen.runtime.videoedit.cli repair \
  --model-path "$MODEL_PATH" \
  --transformer-path "$TRANSFORMER_PATH" \
  --prompt "$PROMPT" \
  --video-input-path "$INPUT_VIDEO" \
  --mask-input-path "$INPUT_MASK" \
  --output-path "$OUT_DIR" \
  --output-file-name 15108907_3840_2160_50fps_sp1_no_offload_compile.mp4 \
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
  --num-frames 81 \
  --infer-len 81 \
  --overlap 0 \
  --num-inference-steps 20 \
  --guidance-scale 5.0 \
  --dynamic-cfg \
  --dynamic-cfg-max-step 15 \
  --seed 42 \
  --dtype bf16 \
  --enable-paste-back \
  --drop-reference-frame \
  --warmup \
  --warmup-steps 1 \
  --perf-dump-path "$OUT_DIR/videoedit_perf_sp1_no_offload_compile.json"
```

### Serve

```bash
SGLANG_TORCH_COMPILE_MODE=max-autotune-no-cudagraphs \
VIDEOEDIT_QUEUE_CAPACITY=1 \
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
  --enable-torch-compile true \
  --warmup true \
  --warmup-steps 1 \
  --output-path "$OUT_DIR" \
  --input-save-path /tmp/sglang-videoedit-inputs \
  --transformer-path "$TRANSFORMER_PATH"
```

同 shape 至少提交两次请求：

- `sp1_no_offload_compile_warmup_request`：编译预热，不计入稳定收益。
- `sp1_no_offload_compile`：第二次或后续请求计入稳定收益。

## 8. Stage 3：加 attention backend

目标：在 compile 配置基础上只新增 attention backend。每次只测一个 backend，并以日志里的实际后端为准，不能只看命令行参数。

当前双卡 A100 80GB 机器不是 Blackwell，`sage_attn_3` 日志中出现 `No module named 'sageattn3'` 后会回退到 Torch SDPA；即使安装 SageAttention3，也只适合 Blackwell 系列，不适合 A100。A100 上优先比较：

- `--attention-backend torch_sdpa`：PyTorch 原生 SDPA，作为保守基线。
- `--attention-backend fa`：FlashAttention，当前 A100 默认可用时通常是主力高性能后端。
- `--attention-backend sage_attn`：SageAttention 2.x，A100 可尝试安装后评估。
- `--attention-backend sage_attn_3`：只作为 Blackwell 专项；A100 上预期回退，不应把它记成 SageAttention3 成绩。

### SageAttention 安装

A100 建议安装 SageAttention 2.x：

```bash
source /mnt/shanhai-ai/shanhai-workspace/zhouhao6/env/sglang/bin/activate
pip install ninja packaging
pip install sageattention==2.2.0 --no-build-isolation

python - <<'PY'
from sageattention import sageattn
print("sageattention ok", sageattn)
PY
```

如果 pip wheel / build 不匹配当前 CUDA，可从源码编译：

```bash
source /mnt/shanhai-ai/shanhai-workspace/zhouhao6/env/sglang/bin/activate
git clone https://github.com/thu-ml/SageAttention /tmp/SageAttention
cd /tmp/SageAttention
pip install -v . --no-build-isolation
```

SageAttention3 只建议在 Blackwell + CUDA 12.8+ + PyTorch 2.8+ 环境尝试：

```bash
source /mnt/shanhai-ai/shanhai-workspace/zhouhao6/env/sglang/bin/activate
git clone https://github.com/thu-ml/SageAttention /tmp/SageAttention
cd /tmp/SageAttention/sageattention3_blackwell
python setup.py install

python - <<'PY'
import sageattn3
print("sageattn3 ok", sageattn3)
PY
```

安装后必须重新跑 `torch_sdpa/fa/sage_attn/sage_attn_3` 四组，检查日志中是否出现 `Using Sage Attention backend` 或 `Using Sage Attention 3 backend`。如果日志出现 fallback，则该结果应记为 fallback 后端。

### CLI

```bash
SGLANG_TORCH_COMPILE_MODE=max-autotune-no-cudagraphs \
python -m sglang.multimodal_gen.runtime.videoedit.cli repair \
  --model-path "$MODEL_PATH" \
  --transformer-path "$TRANSFORMER_PATH" \
  --prompt "$PROMPT" \
  --video-input-path "$INPUT_VIDEO" \
  --mask-input-path "$INPUT_MASK" \
  --output-path "$OUT_DIR" \
  --output-file-name 15108907_3840_2160_50fps_sp1_no_offload_compile_sage3.mp4 \
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
  --attention-backend sage_attn_3 \
  --num-frames 81 \
  --infer-len 81 \
  --overlap 0 \
  --num-inference-steps 20 \
  --guidance-scale 5.0 \
  --dynamic-cfg \
  --dynamic-cfg-max-step 15 \
  --seed 42 \
  --dtype bf16 \
  --enable-paste-back \
  --drop-reference-frame \
  --warmup \
  --warmup-steps 1 \
  --perf-dump-path "$OUT_DIR/videoedit_perf_sp1_no_offload_compile_sage3.json"
```

### Serve

```bash
SGLANG_TORCH_COMPILE_MODE=max-autotune-no-cudagraphs \
VIDEOEDIT_QUEUE_CAPACITY=1 \
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
  --enable-torch-compile true \
  --attention-backend sage_attn_3 \
  --warmup true \
  --warmup-steps 1 \
  --output-path "$OUT_DIR" \
  --input-save-path /tmp/sglang-videoedit-inputs \
  --transformer-path "$TRANSFORMER_PATH"
```

算子对比命令只替换下面一项，其他参数保持不变：

```bash
--attention-backend torch_sdpa
--attention-backend fa
--attention-backend sage_attn
--attention-backend sage_attn_3
```

记录表中需要同时记录 `expected_backend` 和日志解析出的 `actual_backend`。

## 9. Stage 4：加多卡 SP

目标：在前一阶段稳定配置基础上扩展到双卡 SP。先测 Ulysses SP，再测 Ring SP 和 TP2；只有胜出的并行配置继续进入 TeaCache / Cache-DiT 阶段。

### CLI：SP2 Ulysses

```bash
SGLANG_TORCH_COMPILE_MODE=max-autotune-no-cudagraphs \
python -m sglang.multimodal_gen.runtime.videoedit.cli repair \
  --model-path "$MODEL_PATH" \
  --transformer-path "$TRANSFORMER_PATH" \
  --prompt "$PROMPT" \
  --video-input-path "$INPUT_VIDEO" \
  --mask-input-path "$INPUT_MASK" \
  --output-path "$OUT_DIR" \
  --output-file-name 15108907_3840_2160_50fps_sp2_no_offload_compile_sage3.mp4 \
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
  --attention-backend sage_attn_3 \
  --num-frames 81 \
  --infer-len 81 \
  --overlap 0 \
  --num-inference-steps 20 \
  --guidance-scale 5.0 \
  --dynamic-cfg \
  --dynamic-cfg-max-step 15 \
  --seed 42 \
  --dtype bf16 \
  --enable-paste-back \
  --drop-reference-frame \
  --warmup \
  --warmup-steps 1 \
  --perf-dump-path "$OUT_DIR/videoedit_perf_sp2_no_offload_compile_sage3.json"
```

### Serve：SP2 Ulysses

```bash
SGLANG_TORCH_COMPILE_MODE=max-autotune-no-cudagraphs \
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
  --attention-backend sage_attn_3 \
  --warmup true \
  --warmup-steps 1 \
  --output-path "$OUT_DIR" \
  --input-save-path /tmp/sglang-videoedit-inputs \
  --transformer-path "$TRANSFORMER_PATH"
```

并行对照只改并行参数：

```bash
# Ring SP
--num-gpus 2 --sp-degree 2 --ulysses-degree 1 --ring-degree 2

# TP2
--num-gpus 2 --tp-size 2 --sp-degree 1 --ulysses-degree 1 --ring-degree 1
```

不要在同一次对比中同时改变 attention backend、compile、offload 或 cache。

serve 模式每个阶段至少提交两次相同请求：第一次用于服务内 warmup / compile / cache 初始化，第二次作为最终性能记录。

## 10. Stage 5：加 Cache-DiT

目标：在最佳非 cache 配置上新增 Cache-DiT。Cache-DiT 允许轻微质量 tradeoff，但必须和固定 reference 视频、上一阶段非 cache 输出都做 compare。

`dit_layerwise_offload` 与 Cache-DiT 互斥，因此 cache 阶段必须保持 `--no-dit-layerwise-offload`。

### CLI

```bash
SGLANG_CACHE_DIT_ENABLED=true \
SGLANG_CACHE_DIT_FN=1 \
SGLANG_CACHE_DIT_BN=1 \
SGLANG_CACHE_DIT_WARMUP=4 \
SGLANG_CACHE_DIT_RDT=0.12 \
SGLANG_CACHE_DIT_MC=2 \
SGLANG_CACHE_DIT_SCM_PRESET=medium \
SGLANG_TORCH_COMPILE_MODE=max-autotune-no-cudagraphs \
python -m sglang.multimodal_gen.runtime.videoedit.cli repair \
  --model-path "$MODEL_PATH" \
  --transformer-path "$TRANSFORMER_PATH" \
  --prompt "$PROMPT" \
  --video-input-path "$INPUT_VIDEO" \
  --mask-input-path "$INPUT_MASK" \
  --output-path "$OUT_DIR" \
  --output-file-name 15108907_3840_2160_50fps_sp2_no_offload_compile_sage3_cache_rdt012.mp4 \
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
  --attention-backend sage_attn_3 \
  --num-frames 81 \
  --infer-len 81 \
  --overlap 0 \
  --num-inference-steps 20 \
  --guidance-scale 5.0 \
  --dynamic-cfg \
  --dynamic-cfg-max-step 15 \
  --seed 42 \
  --dtype bf16 \
  --enable-paste-back \
  --drop-reference-frame \
  --warmup \
  --warmup-steps 1 \
  --perf-dump-path "$OUT_DIR/videoedit_perf_sp2_no_offload_compile_sage3_cache_rdt012.json"
```

### Serve

```bash
SGLANG_CACHE_DIT_ENABLED=true \
SGLANG_CACHE_DIT_FN=1 \
SGLANG_CACHE_DIT_BN=1 \
SGLANG_CACHE_DIT_WARMUP=4 \
SGLANG_CACHE_DIT_RDT=0.12 \
SGLANG_CACHE_DIT_MC=2 \
SGLANG_CACHE_DIT_SCM_PRESET=medium \
SGLANG_TORCH_COMPILE_MODE=max-autotune-no-cudagraphs \
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
  --attention-backend sage_attn_3 \
  --warmup true \
  --warmup-steps 1 \
  --output-path "$OUT_DIR" \
  --input-save-path /tmp/sglang-videoedit-inputs \
  --transformer-path "$TRANSFORMER_PATH"
```

Cache-DiT 调参只允许一次改变一个环境变量：

- `SGLANG_CACHE_DIT_RDT=0.10/0.12/0.18/0.24`：越大跳过越多，质量风险越高。
- `SGLANG_CACHE_DIT_MC=2/3`：连续 cache 步上限，视频编辑不建议一开始过大。
- `SGLANG_CACHE_DIT_WARMUP=3/4`：保留前几步完整计算，默认先用 4。
- `SGLANG_CACHE_DIT_SCM_PRESET=medium/fast`：serve 连续请求时建议显式设置；日志里出现 `mask_policy None is not valid` 说明 cache policy 没有被正确初始化，不能把该轮作为有效成绩。

`--enable-teacache` 是请求级开关，和 Cache-DiT 不是同一个机制。若要评估 TeaCache，应在同一最佳非 cache 配置上单独新增 `--enable-teacache`，不要和 Cache-DiT 同时启用。

## 11. Quant Branch：量化专项

目标：降低 DiT 显存和提升矩阵乘吞吐。量化只作为独立分支，不覆盖 bf16 主线结论。

建议优先只量化 `WanVideoEditTransformer3DModel`，不量化 VAE 和 text encoder。先测 W8A8 或 weight-only，再评估更低比特；VideoEdit 对 mask 边缘、身份一致性和纹理稳定性敏感，不建议一开始做全模型 4bit。

### CLI 模板

```bash
export QUANT_TRANSFORMER_PATH=/path/to/quantized/videoedit/transformer

python -m sglang.multimodal_gen.runtime.videoedit.cli repair \
  --model-path "$MODEL_PATH" \
  --transformer-path "$QUANT_TRANSFORMER_PATH" \
  --prompt "$PROMPT" \
  --video-input-path "$INPUT_VIDEO" \
  --mask-input-path "$INPUT_MASK" \
  --output-path "$OUT_DIR" \
  --output-file-name 15108907_3840_2160_50fps_quant_branch.mp4 \
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
  --dynamic-cfg \
  --dynamic-cfg-max-step 15 \
  --seed 42 \
  --dtype bf16 \
  --enable-paste-back \
  --drop-reference-frame \
  --warmup \
  --warmup-steps 1 \
  --perf-dump-path "$OUT_DIR/videoedit_perf_quant_branch.json"
```

### Serve 模板

```bash
export QUANT_TRANSFORMER_PATH=/path/to/quantized/videoedit/transformer

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
  --warmup true \
  --warmup-steps 1 \
  --output-path "$OUT_DIR" \
  --input-save-path /tmp/sglang-videoedit-inputs \
  --transformer-path "$QUANT_TRANSFORMER_PATH"
```

量化验收：

- 保留原 bf16 transformer 输出作为 reference。
- 单窗口固定 seed 回归，记录逐帧 SSIM/MSE/MAE/PSNR。
- 检查 cross-attention、QK norm、time/text embedding 等敏感模块是否需要量化白名单。
- 检查日志中 runtime quant config 是否真的生效。
- 量化和 Cache-DiT、SP、attention backend 的组合必须逐项重新测试。

## 12. Offload Branch：显存受限分支

该分支从 `sp1_offload` 出发，目标是让任务能跑起来，不作为性能主线。质量仍和固定 reference 视频对比。Offload 和 Cache-DiT 不要组合，尤其 `dit_layerwise_offload` 与 Cache-DiT 已有硬冲突。

推荐分级：

1. 保持 `vae_tiling=True`。
2. 开启 `text_encoder_cpu_offload`。
3. 开启 `vae_cpu_offload`。
4. 仍然 OOM 时开启 `dit_layerwise_offload`，并设置较小 prefetch。
5. 最后才考虑 `dit_cpu_offload`。

### CLI

```bash
python -m sglang.multimodal_gen.runtime.videoedit.cli repair \
  --model-path "$MODEL_PATH" \
  --transformer-path "$TRANSFORMER_PATH" \
  --prompt "$PROMPT" \
  --video-input-path "$INPUT_VIDEO" \
  --mask-input-path "$INPUT_MASK" \
  --output-path "$OUT_DIR" \
  --output-file-name 15108907_3840_2160_50fps_offload_branch.mp4 \
  --num-gpus 1 \
  --sp-degree 1 \
  --ulysses-degree 1 \
  --ring-degree 1 \
  --text-encoder-cpu-offload \
  --vae-cpu-offload \
  --dit-layerwise-offload \
  --dit-offload-prefetch-size 1 \
  --num-frames 81 \
  --infer-len 81 \
  --overlap 0 \
  --num-inference-steps 20 \
  --guidance-scale 5.0 \
  --dynamic-cfg \
  --dynamic-cfg-max-step 15 \
  --seed 42 \
  --dtype bf16 \
  --enable-paste-back \
  --drop-reference-frame \
  --warmup \
  --warmup-steps 1 \
  --perf-dump-path "$OUT_DIR/videoedit_perf_offload_branch.json"
```

### Serve

```bash
VIDEOEDIT_QUEUE_CAPACITY=1 \
sglang serve \
  --model-type diffusion \
  --model-path "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port 30000 \
  --num-gpus 1 \
  --sp-degree 1 \
  --ulysses-degree 1 \
  --ring-degree 1 \
  --text-encoder-cpu-offload true \
  --vae-cpu-offload true \
  --dit-layerwise-offload true \
  --dit-offload-prefetch-size 1 \
  --warmup true \
  --warmup-steps 1 \
  --output-path "$OUT_DIR" \
  --input-save-path /tmp/sglang-videoedit-inputs \
  --transformer-path "$TRANSFORMER_PATH"
```

## 13. 输出检查和逐帧对比

基础视频检查：

```bash
python - <<'PY'
import cv2

path = "/mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs/reference/15108907_3840_2160_50fps.mp4"
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

查看候选输出 metadata。Reference 视频可能来自原始 `VideoEdit-diffusers/infer.py`，不一定有 `.videoedit.json` sidecar：

```bash
python -m json.tool "$OUT_DIR/15108907_3840_2160_50fps_sp2_no_offload_compile_fa_cache_rdt012.videoedit.json"
```

逐帧 compare：

```bash
python python/sglang/multimodal_gen/runtime/videoedit/compare.py \
  --reference "$OUT_DIR/reference/15108907_3840_2160_50fps.mp4" \
  --candidate "$OUT_DIR/15108907_3840_2160_50fps_sp2_no_offload_compile_fa_cache_rdt012.mp4" \
  --report-json "$OUT_DIR/videoedit_compare_sp2_no_offload_compile_fa_cache_rdt012.json" \
  --min-ssim 0.90 \
  --max-mse 150.0 \
  --max-mae 8.0 \
  --allow-frame-count-delta 1 \
  --max-failed-frame-ratio 0.05
```

如果 candidate 比 reference 多 1 帧且多出的是第 0 帧：

```bash
python python/sglang/multimodal_gen/runtime/videoedit/compare.py \
  --reference "$OUT_DIR/reference/15108907_3840_2160_50fps.mp4" \
  --candidate "$OUT_DIR/15108907_3840_2160_50fps_sp2_no_offload_compile_fa_cache_rdt012.mp4" \
  --drop-candidate-first-frame \
  --report-json "$OUT_DIR/videoedit_compare_sp2_no_offload_compile_fa_cache_rdt012.json" \
  --min-ssim 0.90 \
  --max-mse 150.0 \
  --max-mae 8.0 \
  --allow-frame-count-delta 1 \
  --max-failed-frame-ratio 0.05
```

## 14. 结果记录模板

每个阶段填写一行：

| 阶段 | 新增优化 | CLI/serve | expected backend | actual backend | forward(s) | denoise(s) | text(s) | VAE encode(s) | decode(s) | wall/inference(s) | peak MB | compare |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `sp1_offload` | baseline | CLI + serve | default |  |  |  |  |  |  |  |  | reference |
| `sp1_no_offload` | no offload | CLI + serve | default |  |  |  |  |  |  |  |  |  |
| `sp2_no_offload_torch_sdpa` | SP2 + torch SDPA | CLI + serve | torch_sdpa |  |  |  |  |  |  |  |  |  |
| `sp2_no_offload_fa` | SP2 + FlashAttention | CLI + serve | fa |  |  |  |  |  |  |  |  |  |
| `sp2_no_offload_sage_attn` | SP2 + SageAttention | CLI + serve | sage_attn |  |  |  |  |  |  |  |  |  |
| `sp2_no_offload_sage3` | SP2 + SageAttention3 | CLI + serve | sage_attn_3 |  |  |  |  |  |  |  |  |  |
| `sp2_ring_no_offload_fa` | Ring SP | CLI + serve | fa |  |  |  |  |  |  |  |  |  |
| `tp2_no_offload_fa` | TP2 | CLI + serve | fa |  |  |  |  |  |  |  |  |  |
| `sp2_no_offload_compile_fa` | torch.compile | CLI + serve | fa |  |  |  |  |  |  |  |  |  |
| `sp2_no_offload_compile_fa_teacache` | TeaCache | CLI + serve | fa |  |  |  |  |  |  |  |  |  |
| `sp2_no_offload_compile_fa_cache_rdt010` | Cache-DiT RDT 0.10 | CLI + serve | fa |  |  |  |  |  |  |  |  |  |
| `sp2_no_offload_compile_fa_cache_rdt012` | Cache-DiT RDT 0.12 | CLI + serve | fa |  |  |  |  |  |  |  |  |  |
| `sp2_no_offload_compile_fa_cache_rdt018` | Cache-DiT RDT 0.18 | CLI + serve | fa |  |  |  |  |  |  |  |  |  |
| `sp2_no_offload_compile_fa_cache_fast` | Cache-DiT fast | CLI + serve | fa |  |  |  |  |  |  |  |  |  |
| `offload_branch` | memory branch | CLI + serve | default |  |  |  |  |  |  |  |  |  |

结论只从这个表里产生，不混用历史低步数、改 CFG 或不同输入的数据。

## 15. 后续代码级优化

这些需要源码改动，不能和无代码调参混在同一轮评估：

- Prompt embedding 跨窗口缓存：同一 prompt、negative prompt、dtype 和 tokenizer 配置下复用 text encoder 输出。
- VAE encode/decode 并行接入：确认 `vae_sp` 到 `WanVAEConfig.use_parallel_encode/use_parallel_decode` 的链路，并验证跨 rank gather。
- CFG parallel：当前 VideoEdit 对 `enable_cfg_parallel` 未实现；若实现，需要单独验证 conditional/unconditional rank 的同步和 latents 广播。
- 条件 latent 缓存：只先缓存完全相同窗口；overlap 区域 latent 复用需要额外验证 temporal boundary。

代码级优化必须重新生成 perf JSON 和 compare JSON，不能沿用本文件里的无代码优化结论。
