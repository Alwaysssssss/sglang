# Wan VideoEdit 推理结果对比方案

> 目标：完善 `VideoEdit-diffusers` 原始脚本与 SGLang `WanVideoEditPipeline` 的结果对比流程，先复现已完成的原始 `result.mp4`，再用 SGLang 生成同一输入/同一参数下的候选视频，最后用帧级指标和关键人工检查定位差异。
>
> 方法：采用多 subagent 分片阅读源码和文档后汇总。本方案只规划对比与参数对齐，不要求重新运行原始仓库推理；原始结果已经存在于 `outputs/compare/origin_video_edit_diffusers/`。

## 1. 已确认的源码依据

### 1.1 原始 VideoEdit-diffusers

- 仓库：`/mnt/shanhai-ai/liuh/VideoEdit-diffusers`
- CLI 默认参数：`/mnt/shanhai-ai/liuh/VideoEdit-diffusers/infer.py:96` 的 `build_parser()`
- 主流程：`/mnt/shanhai-ai/liuh/VideoEdit-diffusers/infer.py:192` 的 `infer(args)`
- chunks 选择：`/mnt/shanhai-ai/liuh/VideoEdit-diffusers/infer.py:86` 的 `parse_chunks()`
- 全局预处理：`/mnt/shanhai-ai/liuh/VideoEdit-diffusers/utils/preprocess.py:481` 的 `prepare_global_inputs()`
- 窗口预处理：`/mnt/shanhai-ai/liuh/VideoEdit-diffusers/utils/preprocess.py:568` 的 `prepare_window_inputs()`
- DiffSynth CLIP 分支：`/mnt/shanhai-ai/liuh/VideoEdit-diffusers/pipelines/pipeline_wan_edit.py:339` 的 `WanPipeline.encode_image()`
- Diffusers CLIP 分支：`/mnt/shanhai-ai/liuh/VideoEdit-diffusers/pipelines/pipeline_wan_edit.py:373` 的 `WanPipeline.encode_image_diffuser()`
- Pipeline 入口：`/mnt/shanhai-ai/liuh/VideoEdit-diffusers/pipelines/pipeline_wan_edit.py:560` 的 `WanPipeline.__call__()`
- 输出贴回：`/mnt/shanhai-ai/liuh/VideoEdit-diffusers/utils/postprocess.py:85` 的 `paste_back()`

### 1.2 SGLang VideoEdit

- CLI 参数入口：`python/sglang/multimodal_gen/runtime/videoedit/cli.py:32` 的 `_add_common_repair_args()`
- CLI 到 runtime 参数：`python/sglang/multimodal_gen/runtime/videoedit/cli.py:218` 的 `repair_cmd()`
- Sampling dataclass：`python/sglang/multimodal_gen/configs/sample/videoedit_wan.py:47` 的 `WanVideoEditSamplingParams`
- 参数校验：`python/sglang/multimodal_gen/configs/sample/videoedit_wan.py:178` 的 `_validate_videoedit()`
- Stage 创建：`python/sglang/multimodal_gen/runtime/pipelines/wan_videoedit_pipeline.py:160` 的 `create_pipeline_stages()`
- 全局上下文：`python/sglang/multimodal_gen/runtime/pipelines/wan_videoedit_pipeline.py:191` 的 `_prepare_global_videoedit_context()`
- 窗口物化：`python/sglang/multimodal_gen/runtime/pipelines/wan_videoedit_pipeline.py:366` 的 `_materialize_window_inputs()`
- 窗口提交：`python/sglang/multimodal_gen/runtime/pipelines/wan_videoedit_pipeline.py:438` 的 `_commit_window_output()`
- 最终输出：`python/sglang/multimodal_gen/runtime/pipelines/wan_videoedit_pipeline.py:570` 的 `_finalize_long_video_output()`
- Pipeline 主循环：`python/sglang/multimodal_gen/runtime/pipelines/wan_videoedit_pipeline.py:610` 的 `forward()`
- 窗口规划：`python/sglang/multimodal_gen/runtime/videoedit/windowing.py:23` 的 `build_videoedit_window_specs()`
- 帧数解析：`python/sglang/multimodal_gen/runtime/videoedit/preprocess.py:88` 的 `resolve_videoedit_num_frames()`
- 全局预处理：`python/sglang/multimodal_gen/runtime/videoedit/preprocess.py:490` 的 `prepare_global_inputs()`
- 窗口预处理：`python/sglang/multimodal_gen/runtime/videoedit/preprocess.py:581` 的 `prepare_window_inputs()`
- 帧级比较工具：`python/sglang/multimodal_gen/runtime/videoedit/compare.py:61` 的 `compare_videos()`

### 1.3 已有对齐文档

- 总览：`docs_always/add_new_mode/compare/README.md`
- 全流程对比：`docs_always/add_new_mode/compare/wan_videoedit_pipeline_vs_videoedit_diffusers.md`
- 前后处理专项：`docs_always/add_new_mode/compare/videoedit_prepost_alignment.md`

## 2. 原始结果基线

原始命令已经在其他机器完成运行，本环境只使用产物，不再运行原始推理：

```bash
cd /mnt/shanhai-ai/liuh/VideoEdit-diffusers
python3 infer.py \
  --output_dir /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs/compare/origin_video_edit_diffusers \
  --output_name result \
  --chunks 1 \
  --clip_preprocess diffsynth
```

已存在产物：

- 主视频：`/mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs/compare/origin_video_edit_diffusers/result.mp4`
- crop-only：`/mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs/compare/origin_video_edit_diffusers/result_crop_only.mp4`
- color-corrected：`/mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs/compare/origin_video_edit_diffusers/result_color.mp4`

`ffprobe` 已确认当前文件：

- 输入视频 `1080.mp4`：105 帧、1920x1080、25 fps
- 输入 mask `mask_1080_merged.mp4`：105 帧、1920x1080、25 fps
- 原始 `result.mp4`：81 帧、1920x1080、25 fps
- 原始 `result_crop_only.mp4`：81 帧、1920x1080、25 fps
- 原始 `result_color.mp4`：81 帧、1920x1080、25 fps

## 3. 原始命令的真实语义

当前基线命令只显式传了 `--chunks 1 --clip_preprocess diffsynth`，其余采用原始仓默认值：

| 类别 | 参数 | 原始值 | 说明 |
| --- | --- | --- | --- |
| 输入 | `video_path` | `test_videos/lh/1080.mp4` | 默认输入视频 |
| 输入 | `mask_path` | `test_videos/lh/mask_1080_merged.mp4` | 默认 mask 视频 |
| 输入 | `img_path` | `test_videos/lh/reg.png` | 默认参考图，预处理时 prepend 到第 0 帧 |
| 生成 | `num_inference_steps` | `10` | 与 SGLang CLI 默认 `40` 不同 |
| 生成 | `guidance_scale` | `5.0` | CFG scale |
| 生成 | `seed` | `42` | 每个窗口重新创建 CPU generator |
| 生成 | `dtype` | `bf16` | autocast/load dtype |
| 生成 | `infer_len` | `81` | 单窗口帧数 |
| 生成 | `overlap` | `9` | stride = 72 |
| 生成 | `dynamic_cfg` | `True` | 默认开启 |
| 生成 | `use_clip` | `True` | 默认使用窗口首帧做 CLIP conditioning |
| 预处理 | `num_frames` | `None` | 读取完整输入；但 `chunks=1` 只跑首窗口 |
| 预处理 | `bbox_padding` | `0` | bbox padding |
| 预处理 | `bbox_expand_scale` | `2.5` | 小区域 bbox 扩展 |
| 预处理 | `dilate_px` | `0` | 与 SGLang CLI 默认 `15` 不同 |
| 预处理 | `mask_scale` | `1` | 与 SGLang CLI 默认 `1.2` 不同 |
| 后处理 | `feather_px` | `0` | 与 SGLang CLI 默认 `15` 不同 |
| 后处理 | `adain_boundary_dilate` | `0` | AdaIN boundary |
| 输出 | `save_paste` | `True` | 生成 `result.mp4` |
| 输出 | `save_crop` | `True` | 生成 `result_crop_only.mp4` |
| 输出 | `save_color` | `True` | 生成 `result_color.mp4` |

关键语义：

1. `img_path` 默认非空，`prepare_global_inputs()` 会把 `reg.png` resize 后 prepend 到视频第 0 帧，并在 mask 第 0 帧 prepend 全黑帧。
2. `--chunks 1` 经过 `parse_chunks()` 后只选择窗口 `[0]`，所以虽然输入视频有 105 帧，加参考图后全局帧数为 106，但实际只生成首个 81 帧窗口。
3. 首窗口 `take_start=0`，因此原始 `result.mp4` 保留 prepended reference frame 对应的第 0 帧；`infer.py` 中删除首帧的逻辑处于注释状态。
4. `clip_preprocess=diffsynth` 走 `WanPipeline.encode_image()`：PIL RGB 到 `[0,1]`，转 `[-1,1]`，bicubic resize 到 224，再回 `[0,1]` 并按 CLIP mean/std normalize。
5. pipeline 输出 latent 后由 VAE decode，再 paste 回原视频；主比较对象应优先使用 paste-back 的 `result.mp4`。

## 4. SGLang 当前默认值与差异

SGLang `repair` CLI 可直接生成 VideoEdit 结果，但默认值不是原始命令的 1:1 复刻：

| 参数 | SGLang 默认 | 原始基线 | 对比时动作 |
| --- | --- | --- | --- |
| `num_inference_steps` | `40` | `10` | 必须显式传 `--num-inference-steps 10` |
| `num_frames` | `81` | 全局读取完整，但 `chunks=1` 只输出 81 | 首窗口对齐时传 `--num-frames 80`，加参考图后成为 81 |
| `dilate_px` | `15` | `0` | 必须传 `--dilate-px 0` |
| `mask_scale` | `1.2` | `1` | 必须传 `--mask-scale 1` |
| `feather_px` | `15` | `0` | 必须传 `--feather-px 0` |
| `drop_reference_frame` | `True` | 原始保留第 0 帧 | 必须传 `--no-drop-reference-frame` |
| `clip_preprocess` | `diffuser` | `diffsynth` | 必须传 `--clip-preprocess diffsynth` |
| `overlap_commit_mode` | `weighted` | native skip | 单窗口不影响提交，仍建议传 `--overlap-commit-mode native_skip` |
| `tail_padding_mode` | `reflect` | reverse mirror | 首窗口不足/多窗口时传 `--tail-padding-mode native_reverse_mirror` |
| `decode_mode` | `stream` | eager 风格完整帧列表 | 首窗口对齐建议传 `--decode-mode eager` |
| `enable_teacache` | `True` | 无 | 必须传 `--no-enable-teacache` |
| `generator_device` | 未显式 | 原始 CPU generator | 必须传 `--generator-device cpu` |
| `save_crop_only` | `False` | 原始保存 crop-only | 主对比不要求；如要 sidecar，再单独补齐 |

`--num-frames` 的注意点：SGLang 的 `num_frames` 表示从输入视频/mask 读取多少原始帧，`reference_image_path` 非空时会再 prepend 参考图。原始基线是全局读完整 105 帧后只跑首窗口，输出 81 帧。当前 SGLang 没有 `--chunks` 等价参数，因此为了得到同样 81 帧首窗口，应传 `--num-frames 80 --infer-len 81 --no-drop-reference-frame`，即读取 80 个真实视频帧 + 1 个参考首帧。

## 5. 推荐 SGLang 首窗口对齐命令

主输出单独放到 `outputs/compare/sglang_video_edit_first_window/`，避免和原始输出混放：

```bash
source /mnt/shanhai-ai/shanhai-workspace/zhouhao6/env/sglang/bin/activate
cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang
mkdir -p /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs/compare/sglang_video_edit_first_window

python -m sglang.multimodal_gen.runtime.videoedit.cli repair \
  --model-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model \
  --transformer-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model/transformer \
  --prompt "一个男人在舞台演讲，背后有两行文字，背景保持不变。" \
  --video-input-path /mnt/shanhai-ai/liuh/VideoEdit-diffusers/test_videos/lh/1080.mp4 \
  --mask-input-path /mnt/shanhai-ai/liuh/VideoEdit-diffusers/test_videos/lh/mask_1080_merged.mp4 \
  --reference-image-path /mnt/shanhai-ai/liuh/VideoEdit-diffusers/test_videos/lh/reg.png \
  --output-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs/compare/sglang_video_edit_first_window \
  --output-file-name result \
  --num-gpus 4 \
  --sp-degree 4 \
  --num-frames 80 \
  --infer-len 81 \
  --overlap 9 \
  --num-inference-steps 10 \
  --guidance-scale 5.0 \
  --seed 42 \
  --generator-device cpu \
  --dtype bf16 \
  --dynamic-cfg \
  --bbox-padding 0 \
  --bbox-expand-scale 2.5 \
  --dilate-px 0 \
  --mask-scale 1 \
  --feather-px 0 \
  --adain-boundary-dilate 0 \
  --enable-paste-back \
  --no-drop-reference-frame \
  --use-clip \
  --clip-preprocess diffsynth \
  --no-use-repaired-context \
  --no-vary-seed-by-window \
  --init-latent-mode noise \
  --mask-downsample-mode nearest \
  --overlap-commit-mode native_skip \
  --tail-padding-mode native_reverse_mirror \
  --decode-mode eager \
  --no-enable-teacache \
  --no-enable-frame-interpolation \
  --no-enable-upscaling \
  --warmup \
  --warmup-steps 1 \
  --perf-dump-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs/compare/sglang_video_edit_first_window/perf.json
```

运行后预期主视频路径：

```text
/mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs/compare/sglang_video_edit_first_window/result.mp4
```

若 CLI 实际保存扩展名或目录层级不同，先用以下命令确认：

```bash
find /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs/compare/sglang_video_edit_first_window -maxdepth 2 -type f | sort
```

## 6. 备选：全视频语义对齐命令

若目标从“复现原始 `--chunks 1` 产物”改为“对齐原始 `num_frames=None` 的全视频读取语义”，则使用 `--num-frames -1`。但由于 SGLang 当前没有 `--chunks`，该命令会跑全部窗口，输出帧数不应拿来直接对比已有 81 帧 `result.mp4`：

```bash
python -m sglang.multimodal_gen.runtime.videoedit.cli repair \
  --model-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model \
  --transformer-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model/transformer \
  --prompt "一个男人在舞台演讲，背后有两行文字，背景保持不变。" \
  --video-input-path /mnt/shanhai-ai/liuh/VideoEdit-diffusers/test_videos/lh/1080.mp4 \
  --mask-input-path /mnt/shanhai-ai/liuh/VideoEdit-diffusers/test_videos/lh/mask_1080_merged.mp4 \
  --reference-image-path /mnt/shanhai-ai/liuh/VideoEdit-diffusers/test_videos/lh/reg.png \
  --output-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs/compare/sglang_video_edit_full_video \
  --output-file-name result \
  --num-gpus 4 \
  --sp-degree 4 \
  --num-frames -1 \
  --infer-len 81 \
  --overlap 9 \
  --num-inference-steps 10 \
  --guidance-scale 5.0 \
  --seed 42 \
  --generator-device cpu \
  --dtype bf16 \
  --dynamic-cfg \
  --bbox-padding 0 \
  --bbox-expand-scale 2.5 \
  --dilate-px 0 \
  --mask-scale 1 \
  --feather-px 0 \
  --adain-boundary-dilate 0 \
  --enable-paste-back \
  --no-drop-reference-frame \
  --use-clip \
  --clip-preprocess diffsynth \
  --no-use-repaired-context \
  --no-vary-seed-by-window \
  --init-latent-mode noise \
  --mask-downsample-mode nearest \
  --overlap-commit-mode native_skip \
  --tail-padding-mode native_reverse_mirror \
  --decode-mode eager \
  --no-enable-teacache \
  --no-enable-frame-interpolation \
  --no-enable-upscaling
```

## 7. 对比命令

### 7.1 基础文件检查

```bash
cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang

for f in \
  outputs/compare/origin_video_edit_diffusers/result.mp4 \
  outputs/compare/sglang_video_edit_first_window/result.mp4; do
  echo "--- $f"
  ffprobe -v error -select_streams v:0 \
    -show_entries stream=width,height,nb_frames,r_frame_rate,avg_frame_rate,duration \
    -of default=noprint_wrappers=1 "$f"
done
```

预期：两边都应为 81 帧、1920x1080、25 fps。若 SGLang 为 80 帧，优先检查是否漏传 `--no-drop-reference-frame`；若 SGLang 大于 81 帧，优先检查是否误用 `--num-frames -1` 或未来实现了全窗口路径。

### 7.2 帧级指标比较

```bash
cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang
mkdir -p outputs/compare/reports

python -m sglang.multimodal_gen.runtime.videoedit.compare \
  --reference outputs/compare/origin_video_edit_diffusers/result.mp4 \
  --candidate outputs/compare/sglang_video_edit_first_window/result.mp4 \
  --report-json outputs/compare/reports/origin_vs_sglang_first_window.json \
  --min-ssim 0.90 \
  --max-mse 150 \
  --max-mae 8 \
  --allow-frame-count-delta 0 \
  --max-failed-frame-ratio 0.05
```

若 SGLang 因默认或历史命令丢掉了 reference 首帧，可临时比较真实视频帧部分：

```bash
python -m sglang.multimodal_gen.runtime.videoedit.compare \
  --reference outputs/compare/origin_video_edit_diffusers/result.mp4 \
  --candidate outputs/compare/sglang_video_edit_first_window/result.mp4 \
  --report-json outputs/compare/reports/origin_vs_sglang_drop_reference_first_frame.json \
  --drop-reference-first-frame \
  --allow-frame-count-delta 0
```

但正式对齐建议仍是修正 SGLang 命令，保留首帧后再比较。

### 7.3 人工检查点

除 JSON 指标外，至少人工检查：

- 第 0 帧：是否为 `reg.png` reference/paste-back 后对应帧。
- mask 区域：演讲者背后文字修复区域是否位置一致。
- 非 mask 区域：背景和人物边缘是否被不必要修改。
- 边界：`feather_px=0` 时边界应更接近原始硬贴回；若边缘明显不同，检查 SGLang 是否仍使用默认 `feather_px=15`。
- 色彩：主视频 `result.mp4` 不启用 color correction；不要拿原始 `result_color.mp4` 和 SGLang 主输出直接比较。

## 8. 预期差异与排查顺序

如果指标不达标，按以下顺序排查：

1. **帧数/首帧**：确认 SGLang 使用 `--num-frames 80 --no-drop-reference-frame`，输出 81 帧。原始 `infer.py` 没有删除 prepended reference frame。
2. **预处理参数**：确认 `--dilate-px 0 --mask-scale 1 --feather-px 0 --bbox-expand-scale 2.5`，这些默认值与 SGLang 当前 CLI 默认不同。
3. **采样步数**：确认 `--num-inference-steps 10`，否则 SGLang 默认 40 步会显著偏离。
4. **随机数设备**：确认 `--generator-device cpu --seed 42 --no-vary-seed-by-window`，原始每个窗口重新创建 CPU generator。
5. **CLIP 预处理**：确认 `--clip-preprocess diffsynth --use-clip`，原始命令显式使用 DiffSynth 分支。
6. **窗口提交/尾部 padding**：单窗口首 81 帧通常不触发后续提交差异，但为了未来多窗口对齐仍固定 `native_skip` 与 `native_reverse_mirror`。
7. **服务化优化**：确认关闭 TeaCache、frame interpolation、upscaling；SP 并行本身用于性能，但若追求 bit-level parity，可追加单卡 `--num-gpus 1` 复测。
8. **输出编码差异**：原始使用 moviepy/libx264/10M，SGLang 使用自身 ffmpeg 保存逻辑；即使帧内容相近，编码带来的小幅 MSE/SSIM 差异是可能的。

## 9. 暂不覆盖的范围

- 不重新运行 `/mnt/shanhai-ai/liuh/VideoEdit-diffusers/infer.py`。
- 不把原始 `result_color.mp4` 作为主指标基线，因为 SGLang 当前没有完全等价的 `_color.mp4` sidecar 输出。
- 不比较多窗口长视频结果，除非先为 SGLang 增加 `--chunks` 或明确改用全视频语义。
- 不要求 bit-level 完全一致；当前计划以帧数一致、首帧语义一致、SSIM/MSE/MAE 达阈值和人工检查通过为准。

## 10. 后续源码改进建议

如需要把对比流程固化到 SGLang 中，建议后续单独做以下改动：

1. 给 `python/sglang/multimodal_gen/runtime/videoedit/cli.py` 增加 `--chunks`，在 `WanVideoEditSamplingParams` 与 `build_videoedit_window_specs()` 后过滤窗口，复刻原始 `parse_chunks()` 的 `<=0 = all`、正数取前 N 个窗口语义。
2. 增加 `--save-color` 或 color-correct sidecar，复刻 `/mnt/shanhai-ai/liuh/VideoEdit-diffusers/utils/postprocess.py:85` 中 `color_correct=True` 路径，方便比较 `result_color.mp4`。
3. 增加 compare preset，例如 `--preset videoedit-diffusers-first-window`，自动设置 `num_frames=80`、`drop_reference_frame=False`、`num_inference_steps=10`、`dilate_px=0`、`mask_scale=1`、`feather_px=0`、`clip_preprocess=diffsynth`、`teacache=False`。
4. 在 metadata 中记录 resolved frame count、是否 prepend reference、是否 drop reference、window starts、commit mode、tail padding mode，降低之后排查首帧和窗口差异的成本。
