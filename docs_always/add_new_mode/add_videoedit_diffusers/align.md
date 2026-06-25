# VideoEdit-diffusers 与 SGLang VIDEO_EDIT 输出细微不对齐排查方案

本文档用于完善当前 SGLang `VIDEO_EDIT` 部署链路与原始
`/mnt/shanhai-ai/shanhai-workspace/zhouhao6/VideoEdit-diffusers` 的输出对齐方案。

目标不是重做 VideoEdit 接入，也不是继续做性能优化；目标是围绕“当前结果视觉上基本正确，但与 reference 仍有细微差异”建立一套可复现的排查、修改和验收流程，最终把差异定位到具体默认参数、stage 语义或数值路径。

## 1. 背景与当前判断

当前 SGLang 已经具备原生 `VIDEO_EDIT` 任务链路，核心文档和实现入口包括：

- 集成方案：`docs_always/add_new_mode/add_videoedit_diffusers/README.md`
- CLI/Serve 命令：`docs_always/add_new_mode/add_videoedit_diffusers/cli.md`
- 性能优化结果：`docs_always/add_new_mode/add_videoedit_diffusers/benchmark_results.md`
- SGLang pipeline：`python/sglang/multimodal_gen/runtime/pipelines/wan_videoedit_pipeline.py`
- SGLang stage：`python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/videoedit_wan.py`
- SGLang preprocess/postprocess：`python/sglang/multimodal_gen/runtime/videoedit/preprocess.py`、`postprocess.py`
- Reference：`VideoEdit-diffusers/infer.py`、`pipelines/pipeline_wan_edit.py`、`utils/preprocess.py`、`utils/postprocess.py`

从源码对照看，当前最可疑的不是 DiT 权重结构或 scheduler 大框架，而是 SGLang 的默认部署参数已经偏向“可服务化/可优化”的本地语义，和 reference `infer.py` 的严格默认语义并不完全相同。因此排查应先建立 `reference-strict` 配置，再逐个打开当前部署默认项。

## 2. 对齐目标

### 核心目标

- 用同一模型、同一 transformer、同一输入视频、同一 mask、同一 prompt、同一 seed 生成 reference baseline 和 SGLang candidate。
- 将 SGLang 的 `VIDEO_EDIT` 提供一个可显式选择的 `reference-strict` 对齐配置，先证明 strict 配置下输出接近 reference。
- 对当前部署默认配置逐项回放差异来源，确认哪些差异是刻意的产品/性能取舍，哪些是应该修正的默认语义偏差。

### 非目标

- 不在本轮引入 Cache-DiT、TeaCache、量化、低步数、改 CFG 上限等性能优化变量。
- 不依赖原 `VideoEdit-diffusers` 仓库的运行时代码来执行 SGLang 推理；原仓库只作为 reference baseline 生成与源码语义对照。
- 不改通用 Wan T2V/I2V 路径来迁就 VideoEdit；VideoEdit 差异应留在专用 config、sampling params、stage 或 CLI/API 层。

## 3. 当前高概率差异点

下表是基于当前源码的排查优先级。优先级越高，越可能解释“整体效果对，但细节或边缘不完全一致”。

| 优先级 | 差异点 | Reference 行为 | SGLang 当前行为 | 修改建议 |
| --- | --- | --- | --- | --- |
| P0 | latent 初始化 | `pipeline_wan_edit.py` 在传入 `video_latents` 时执行 `scheduler.add_noise(video_latents, noise, timesteps[:1])` | `WanVideoEditSamplingParams.init_latent_mode` 默认是 `"noise"` | strict 对齐默认改为 `"add_noise"`；当前部署若保留 `"noise"`，必须作为显式 fast/deploy preset |
| P0 | CLIP/image context | reference `infer.py` 调用 pipeline 时只传 `encoder_hidden_states=prompt_embeds`，不传 `encoder_hidden_states_image` | SGLang 默认 `use_clip=True`，会尝试加载 image encoder 并传首帧 image context | strict 对齐默认应为 `use_clip=False`；若实测 use_clip 更好，应记录为 SGLang 扩展语义 |
| P0 | prompt token 长度 | 当前 reference `pipeline_wan_edit.py::__call__` 默认 `max_sequence_length=512`，`infer.py` 未覆盖该参数 | SGLang T5Config 和 `t5_postprocess_text` 也是 512 | 文档中不要再把 VideoEdit strict 对齐写成 226；226 是早期理解或其他模型语义，不适用于当前 VideoEdit reference |
| P1 | mask 膨胀/缩放 | `dilate_px=15`、`mask_scale=1.2` | 默认 `dilate_px=0`、`mask_scale=1.0` | strict 对齐改为 15/1.2；部署默认若需要原 mask，应显式说明 |
| P1 | bbox 扩展 | reference 只做 `bbox_padding=0` 和小 bbox 扩到短边 480 | SGLang 默认还有 `bbox_expand_scale=0.3` | strict 对齐应设 `bbox_expand_scale=0.0` |
| P1 | paste-back feather | reference 默认 `feather_px=12` | SGLang 默认 `feather_px=0` | strict 对齐应设 `feather_px=12` |
| P1 | mask downsample | reference 使用 `F.interpolate(..., mode="nearest-exact")` | SGLang 默认 `"nearest"`，虽支持 `"nearest-exact"` | strict 对齐应设 `"nearest-exact"` |
| P2 | generator device | reference 使用 `torch.Generator(device="cuda")` | SGLang `WanVideoEditPipelineConfig.generator_device="cpu"` | strict 对齐优先设 `generator_device=cuda`；若多卡/SP 下不能稳定复现，再记录为数值差异 |
| P2 | decode mode | reference eager 读取完整视频帧列表 | SGLang 默认 `decode_mode="stream"` | strict 首轮用 `decode_mode=eager`，确认 stream/eager bbox、mask、fps 完全一致后再切回 stream |
| P2 | overlap/window | reference 对 81 帧样例 `overlap=0`、单窗口 | SGLang params 默认 `overlap=9`，CLI 文档主测显式 `overlap=0` | strict 对齐固定 `overlap=0` |
| P3 | 输出编码 | reference 用 moviepy `ImageSequenceClip(..., codec="libx264", bitrate="10M")` | SGLang 使用内部 `save_video_frames` | 先比较解码后 RGB 帧指标；必要时新增编码参数对齐，避免把 H.264 压缩差异误判为模型差异 |

## 4. 需要修改与不能修改的范围

### 建议修改文件

- `python/sglang/multimodal_gen/configs/sample/videoedit_wan.py`
  - 新增 `alignment_preset` 或 `reference_strict` 字段，或调整 strict 模式下的默认参数。
  - 建议保留当前部署默认的显式入口，避免已有性能 benchmark 失效。
- `python/sglang/multimodal_gen/runtime/videoedit/cli.py`
  - 增加 `--alignment-preset reference-strict|deploy`，或增加 `--reference-strict` 快捷开关。
  - strict 开关应一次性落下 `init_latent_mode/use_clip/dilate_px/mask_scale/bbox_expand_scale/feather_px/mask_downsample_mode/generator_device/decode_mode/overlap`。
- `python/sglang/multimodal_gen/runtime/entrypoints/openai/protocol.py`
  - 给 serve 请求加相同 preset 字段，避免 CLI 与 API 参数语义分叉。
- `python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py`
  - 在 `/v1/videos/repairs` 中应用 preset，并把最终 effective 参数写入 metadata。
- `python/sglang/multimodal_gen/runtime/pipelines/wan_videoedit_pipeline.py`
  - metadata 中补齐 `init_latent_mode`、`use_clip`、`dilate_px`、`mask_scale`、`bbox_expand_scale`、`feather_px`、`mask_downsample_mode`、`decode_mode`、`generator_device`、`overlap_commit_mode`、`tail_padding_mode`。
- `python/sglang/multimodal_gen/test/unit/test_videoedit_decode_mode_params.py`
  - 更新默认参数测试，新增 preset 展开测试。
- `docs_always/add_new_mode/add_videoedit_diffusers/cli.md`
  - 增加 strict 对齐命令，与性能 benchmark 命令区分。
- `docs_always/add_new_mode/add_videoedit_diffusers/benchmark_results.md`
  - 若默认参数被调整，必须注明历史 benchmark 属于旧 deploy preset，不能混用。

### 不建议修改文件

- 不修改 `python/sglang/multimodal_gen/configs/sample/sampling_params.py` 的内部实现。
- 不修改 `python/sglang/multimodal_gen/runtime/loader/component_loaders/component_loader.py` 的内部实现。
- 不修改通用 Wan `python/sglang/multimodal_gen/runtime/models/dits/wanvideo.py` 来处理 VideoEdit 特例。
- 不修改 reference 仓库源码来“适配”SGLang；reference 只用于生成 baseline。

## 5. 设计方案

### 5.1 增加对齐 preset

推荐增加枚举式 preset，而不是散落十几个默认值判断：

```text
alignment_preset = "deploy" | "reference-strict"
```

`reference-strict` 展开为：

```text
overlap = 0
init_latent_mode = "add_noise"
use_clip = false
dilate_px = 15
mask_scale = 1.2
bbox_expand_scale = 0.0
feather_px = 12
adain_boundary_dilate = 15
mask_downsample_mode = "nearest-exact"
decode_mode = "eager"
generator_device = "cuda"
enable_paste_back = true
drop_reference_frame = true
dynamic_cfg = true
dynamic_cfg_max_step = 15
dynamic_cfg_min = 1.0
num_inference_steps = 20
guidance_scale = 5.0
dtype = "bf16"
```

`deploy` 保持当前服务默认或性能 benchmark 默认，但必须在 metadata 里写清楚，不再声称它是 reference-strict。

### 5.2 保持 512 token 文本语义

当前源码证据显示 VideoEdit reference 的实际 `infer.py` 路径会使用 `pipeline_wan_edit.py::__call__(max_sequence_length=512)` 默认值。SGLang 现有 `T5Config.text_len=512` 和 `WanVideoEditCrossAttention.text_context_len=512` 与此一致。

因此本轮不建议把 VideoEdit 的 prompt embed 改成 226。若后续有人提出 226，需要先跑一个只改 `max_sequence_length` 的 A/B compare，并保存 prompt embed shape、首步 DiT 输入和输出统计。

### 5.3 metadata 必须记录 effective 参数

当前 `.videoedit.json` 已记录 bbox、尺寸、窗口等信息，但不足以解释细微差异。建议补齐：

```json
{
  "alignment_preset": "reference-strict",
  "init_latent_mode": "add_noise",
  "use_clip": false,
  "dilate_px": 15,
  "mask_scale": 1.2,
  "bbox_expand_scale": 0.0,
  "feather_px": 12,
  "mask_downsample_mode": "nearest-exact",
  "decode_mode": "eager",
  "generator_device": "cuda",
  "num_inference_steps": 20,
  "guidance_scale": 5.0,
  "dynamic_cfg": true,
  "dynamic_cfg_max_step": 15,
  "seed": 42
}
```

没有这些字段时，compare JSON 只能证明“两个视频接近或不接近”，不能证明本轮差异来自哪里。

## 6. 实施步骤

### Step 1: 固化 reference baseline

在原仓库用原环境生成 baseline。推理属于长任务，必须在 tmux 中运行：

注意：reference `infer.py` 会以 `subfolder="transformer"` 加载 DiT，因此这里的
`--transformer_path` 应传模型根目录，而不是 `.../transformer` 子目录。

```bash
tmux new-session -d -s videoedit_ref_align \
  'cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/VideoEdit-diffusers && \
   mkdir -p /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs/reference && \
   source /mnt/shanhai-ai/shanhai-workspace/zhouhao6/env/sglang/bin/activate && \
   python infer.py \
     --model_path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model \
     --transformer_path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model \
     --video_path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pexel_test_data_0410/videos/15108907_3840_2160_50fps_short.mp4 \
     --mask_path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pexel_test_data_0410/masks/15108907_3840_2160_50fps_No_bbox_mask.mp4 \
     --prompt "A close-up of an orange flower with a yellow center, remaining in focus against a blurred green grass background throughout the video." \
     --output_dir /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs/reference \
     --output_name 15108907_3840_2160_50fps_ref \
     --num_frames 81 \
     --infer_len 81 \
     --num_inference_steps 20 \
     --guidance_scale 5.0 \
     --seed 42 \
     --dtype bf16 2>&1 | tee /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs/videoedit_ref_align.log'
```

Attach：

```bash
tmux attach -t videoedit_ref_align
```

### Step 2: 跑 SGLang reference-strict candidate

新增 preset 后用同样口径运行。推理同样必须放入 tmux：

```bash
tmux new-session -d -s videoedit_sglang_strict_align \
  'cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang && \
   mkdir -p /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs && \
   source /mnt/shanhai-ai/shanhai-workspace/zhouhao6/env/sglang/bin/activate && \
   PYTHONPATH=python python -m sglang.multimodal_gen.runtime.videoedit.cli repair \
     --model-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model \
     --transformer-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/step-55000-diffusers-lh/transformer \
     --prompt "A close-up of an orange flower with a yellow center, remaining in focus against a blurred green grass background throughout the video." \
     --video-input-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pexel_test_data_0410/videos/15108907_3840_2160_50fps_short.mp4 \
     --mask-input-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pexel_test_data_0410/masks/15108907_3840_2160_50fps_No_bbox_mask.mp4 \
     --output-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs \
     --output-file-name 15108907_3840_2160_50fps_sglang_reference_strict.mp4 \
     --alignment-preset reference-strict \
     --perf-dump-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs/videoedit_perf_reference_strict.json \
     2>&1 | tee /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs/videoedit_sglang_reference_strict.log'
```

如果暂未实现 `--alignment-preset`，临时等价命令为：

```bash
PYTHONPATH=python python -m sglang.multimodal_gen.runtime.videoedit.cli repair \
  --model-path "$MODEL_PATH" \
  --transformer-path "$TRANSFORMER_PATH" \
  --prompt "$PROMPT" \
  --video-input-path "$INPUT_VIDEO" \
  --mask-input-path "$INPUT_MASK" \
  --output-path "$OUT_DIR" \
  --output-file-name 15108907_3840_2160_50fps_sglang_reference_strict.mp4 \
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
  --init-latent-mode add_noise \
  --no-use-clip \
  --dilate-px 15 \
  --mask-scale 1.2 \
  --bbox-expand-scale 0.0 \
  --feather-px 12 \
  --adain-boundary-dilate 15 \
  --mask-downsample-mode nearest-exact \
  --decode-mode eager \
  --generator-device cuda
```

### Step 3: 逐帧 compare

```bash
cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang
source /mnt/shanhai-ai/shanhai-workspace/zhouhao6/env/sglang/bin/activate

PYTHONPATH=python python python/sglang/multimodal_gen/runtime/videoedit/compare.py \
  --reference outputs/reference/15108907_3840_2160_50fps_ref.mp4 \
  --candidate outputs/15108907_3840_2160_50fps_sglang_reference_strict.mp4 \
  --report-json outputs/videoedit_compare_reference_strict.json \
  --min-ssim 0.90 \
  --max-mse 150.0 \
  --max-mae 8.0 \
  --allow-frame-count-delta 1 \
  --max-failed-frame-ratio 0.05
```

若 strict 仍不过，再按下面顺序单变量回放：

1. `init_latent_mode=noise` vs `add_noise`
2. `use_clip=true` vs `false`
3. `dilate_px/mask_scale/bbox_expand_scale/feather_px`
4. `mask_downsample_mode=nearest` vs `nearest-exact`
5. `generator_device=cpu` vs `cuda`
6. `decode_mode=stream` vs `eager`
7. `attention_backend` / SP / TP / compile / offload

每次只改一个变量，并保存 candidate、metadata、perf JSON、compare JSON。

## 7. 验收标准

### 功能验收

- Reference baseline 能生成视频，帧数为 80 或 81，分辨率非 0。
- SGLang reference-strict candidate 能生成视频，metadata 完整记录 effective 参数。
- compare JSON 生成成功，并记录 `ssim_mean`、`ssim_min`、`mse_mean`、`mse_max`、`mae_mean`、`mae_max`、`failed_frames`。

### 对齐阈值

先使用现有宽松集成阈值作为故障检测：

```text
min_ssim = 0.90
max_mse = 150.0
max_mae = 8.0
max_failed_frame_ratio = 0.05
```

如果 strict 配置稳定通过，再增加一档 release 对齐阈值，建议先用当前机器实测结果定线，不要凭空写死：

```text
ssim_mean >= 0.985
ssim_min >= 0.980
mse_mean <= 8.0
mae_mean <= 2.0
```

如果编码器差异导致 strict 阈值误报，必须补充“编码前 RGB 帧 compare”或统一编码参数后再定线。

### 回归验收

- 单元测试覆盖 preset 展开和默认值：
  - `WanVideoEditSamplingParams(alignment_preset="reference-strict")`
  - CLI 默认值和 API 默认值一致
  - invalid preset fail-fast
- 端到端 smoke 测试覆盖 81 帧固定样例。
- 现有性能 benchmark 文档需要标明使用的是 `deploy` 还是 `reference-strict`，不能混用结论。

## 8. 风险与应对

- 如果 `generator_device=cuda` 在多卡 SP/TP 下无法完全复现 reference，应将 strict 对齐先限定为单卡，并把多卡数值差异作为性能配置的已知偏差。
- 如果 `use_clip=false` 显著降低当前部署主观质量，说明 SGLang 当前行为已经变成 reference 之外的增强语义；这时不能把它称为 reference 对齐，需要在 API 和文档里明确命名。
- 如果把 `dilate_px=15/mask_scale=1.2/feather_px=12` 改成默认导致历史输出变化，先通过 preset 保持兼容，再决定是否切换默认。
- 如果 compare 指标显示全局轻微偏差但 mask 区域正常，优先排查视频编码和后处理 rounding；如果只在 mask 边缘失败，优先排查 mask 膨胀、bbox、feather、paste-back。

## 9. 结论

当前最值得先改的不是模型结构，而是把 `reference-strict` 语义显式化。首轮建议优先验证三项：

1. `init_latent_mode="add_noise"`
2. `use_clip=false`
3. `dilate_px=15, mask_scale=1.2, bbox_expand_scale=0.0, feather_px=12, mask_downsample_mode="nearest-exact"`

如果这三项收敛后 compare 明显提升，再把它们固化为 strict preset，并让 CLI/Serve/metadata/文档统一暴露。只有 strict preset 仍失败时，才继续深入 DiT forward、VAE decode、attention backend 或保存编码器的数值差异。
