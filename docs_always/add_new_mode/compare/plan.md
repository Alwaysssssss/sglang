# Wan VideoEdit 推理结果高指标对比方案

> 目标：把 `/mnt/shanhai-ai/liuh/VideoEdit-diffusers` 原始 `infer.py --chunks 1 --clip_preprocess diffsynth` 产物，与 SGLang `WanVideoEditPipeline` 在同输入、同关键参数、同首窗口语义下做高指标对齐。若最后指标不合格，本方案要求先确认帧数/参数/窗口/保存链路，再分析 SGLang 的 VideoEdit 运行日志和 metadata，给出“不合格”和最可能未对齐原因。
>
> 结论先行：当前工作区只存在原始基线视频 `outputs/compare/origin_video_edit_diffusers/*.mp4`，没有发现 `outputs/compare/sglang_video_edit_first_window/result.mp4`。因此本计划不能声称指标已通过；下一步必须先运行第 5 节命令生成 SGLang 候选，再执行第 7 节指标比较。

## 1. 当前状态与完成标准

### 1.1 当前已确认状态

- 原始仓库：`/mnt/shanhai-ai/liuh/VideoEdit-diffusers`
- SGLang 仓库：`/mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang`
- 原始基线目录：`outputs/compare/origin_video_edit_diffusers/`
- 已存在原始产物：
  - `outputs/compare/origin_video_edit_diffusers/result.mp4`
  - `outputs/compare/origin_video_edit_diffusers/result_crop_only.mp4`
  - `outputs/compare/origin_video_edit_diffusers/result_color.mp4`
- 当前未发现 SGLang 首窗口候选：`outputs/compare/sglang_video_edit_first_window/result.mp4`
- 当前未发现本次首窗口对比报告：`outputs/compare/reports/origin_vs_sglang_first_window.json`

### 1.2 完成标准

完成本次对齐时，必须同时满足：

1. SGLang 候选视频存在于 `outputs/compare/sglang_video_edit_first_window/result.mp4`。
2. `ffprobe` 显示原始 `result.mp4` 与 SGLang 候选均为 `81` 帧、`1920x1080`、`25 fps`。
3. `outputs/compare/sglang_video_edit_first_window/result.videoedit.json` 存在，并显示：
   - `num_input_frames = 81`
   - `num_output_frames = 81`
   - `drop_reference_frame = false`
   - 首个窗口 `start_index = 0`、`valid_len = 81`
4. SGLang 运行命令启用低内存模式：显式打开模型/编码器/VAE CPU offload、VAE tiling、VAE slicing，并保留 `--pin-cpu-memory`。
5. 第 7.2 节“高指标门槛”比较命令返回 `0`，并写出 `outputs/compare/reports/origin_vs_sglang_first_window.json`。
6. 人工检查第 0 帧、mask 区域、非 mask 区域、边界、色彩后没有明显语义错位。
7. 若第 5 项不通过，不能改称完成；必须执行第 8 节日志诊断，并在报告中明确“当前指标不合格”和对应证据。

## 2. 源码依据

### 2.1 原始 VideoEdit-diffusers

- CLI 默认参数：`/mnt/shanhai-ai/liuh/VideoEdit-diffusers/infer.py:96` 的 `build_parser()`
- 保存视频：`/mnt/shanhai-ai/liuh/VideoEdit-diffusers/infer.py:80` 的 `save_video()`，使用 `moviepy`、`libx264`、`bitrate=10M`
- chunks 选择：`/mnt/shanhai-ai/liuh/VideoEdit-diffusers/infer.py:86` 的 `parse_chunks()`，`<=0` 表示全部窗口，正数表示取前 N 个窗口
- 主流程：`/mnt/shanhai-ai/liuh/VideoEdit-diffusers/infer.py:192` 的 `infer(args)`
- 全局预处理：`/mnt/shanhai-ai/liuh/VideoEdit-diffusers/utils/preprocess.py:481` 的 `prepare_global_inputs()`
- 窗口预处理：`/mnt/shanhai-ai/liuh/VideoEdit-diffusers/utils/preprocess.py:568` 的 `prepare_window_inputs()`
- DiffSynth CLIP 分支：`/mnt/shanhai-ai/liuh/VideoEdit-diffusers/pipelines/pipeline_wan_edit.py:339` 的 `WanPipeline.encode_image()`
- Diffusers CLIP 分支：`/mnt/shanhai-ai/liuh/VideoEdit-diffusers/pipelines/pipeline_wan_edit.py:373` 的 `WanPipeline.encode_image_diffuser()`
- Pipeline 入口：`/mnt/shanhai-ai/liuh/VideoEdit-diffusers/pipelines/pipeline_wan_edit.py:560` 的 `WanPipeline.__call__()`
- 输出贴回：`/mnt/shanhai-ai/liuh/VideoEdit-diffusers/utils/postprocess.py:85` 的 `paste_back()`

### 2.2 SGLang VideoEdit

- CLI 参数入口：`python/sglang/multimodal_gen/runtime/videoedit/cli.py:32` 的 `_add_common_repair_args()`
- CLI 到 runtime 参数：`python/sglang/multimodal_gen/runtime/videoedit/cli.py:218` 的 `repair_cmd()`
- Sampling dataclass：`python/sglang/multimodal_gen/configs/sample/videoedit_wan.py:47` 的 `WanVideoEditSamplingParams`
- 参数校验：`python/sglang/multimodal_gen/configs/sample/videoedit_wan.py:178` 的 `_validate_videoedit()`
- Stage 创建：`python/sglang/multimodal_gen/runtime/pipelines/wan_videoedit_pipeline.py:160` 的 `create_pipeline_stages()`
- 全局上下文：`python/sglang/multimodal_gen/runtime/pipelines/wan_videoedit_pipeline.py:191` 的 `_prepare_global_videoedit_context()`
- 窗口物化：`python/sglang/multimodal_gen/runtime/pipelines/wan_videoedit_pipeline.py:366` 的 `_materialize_window_inputs()`
- 窗口提交：`python/sglang/multimodal_gen/runtime/pipelines/wan_videoedit_pipeline.py:438` 的 `_commit_window_output()`
- Metadata 写出：`python/sglang/multimodal_gen/runtime/pipelines/wan_videoedit_pipeline.py:483` 的 `_write_metadata()`
- crop sidecar：`python/sglang/multimodal_gen/runtime/pipelines/wan_videoedit_pipeline.py:542` 的 `_save_crop_sidecar()`
- 最终输出：`python/sglang/multimodal_gen/runtime/pipelines/wan_videoedit_pipeline.py:570` 的 `_finalize_long_video_output()`
- Pipeline 主循环：`python/sglang/multimodal_gen/runtime/pipelines/wan_videoedit_pipeline.py:610` 的 `forward()`
- 窗口规划：`python/sglang/multimodal_gen/runtime/videoedit/windowing.py:23` 的 `build_videoedit_window_specs()`
- 帧数解析：`python/sglang/multimodal_gen/runtime/videoedit/preprocess.py:88` 的 `resolve_videoedit_num_frames()`
- 全局预处理：`python/sglang/multimodal_gen/runtime/videoedit/preprocess.py:490` 的 `prepare_global_inputs()`
- 窗口预处理：`python/sglang/multimodal_gen/runtime/videoedit/preprocess.py:581` 的 `prepare_window_inputs()`
- 视频保存：`python/sglang/multimodal_gen/runtime/entrypoints/utils.py:334` 的 `save_outputs()` 和 `python/sglang/multimodal_gen/runtime/entrypoints/utils.py:397` 的 `post_process_sample()`
- 帧级比较工具：`python/sglang/multimodal_gen/runtime/videoedit/compare.py:61` 的 `compare_videos()`
- 低内存 CLI 参数：`python/sglang/multimodal_gen/runtime/videoedit/cli.py:109` 到 `python/sglang/multimodal_gen/runtime/videoedit/cli.py:118` 的 `--dit-cpu-offload`、`--dit-layerwise-offload`、`--text-encoder-cpu-offload`、`--image-encoder-cpu-offload`、`--vae-cpu-offload`、`--vae-tiling`、`--vae-slicing`、`--pin-cpu-memory`
- VAE 分片参数：`python/sglang/multimodal_gen/configs/pipeline_configs/base.py:493` 的 `--vae-tiling` 与 `python/sglang/multimodal_gen/configs/pipeline_configs/base.py:500` 的 `--vae-slicing`
- VideoEdit VAE 分片生效点：`python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/videoedit_wan.py:419` 的 condition encoding tiling、`python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/videoedit_wan.py:423` 的 condition encoding slicing、`python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/videoedit_wan.py:735` 的 decoding tiling 与 `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/videoedit_wan.py:737` 的 decoding slicing

### 2.3 已有对齐文档

- 总览：`docs_always/add_new_mode/compare/README.md`
- 全流程对比：`docs_always/add_new_mode/compare/wan_videoedit_pipeline_vs_videoedit_diffusers.md`
- 前后处理专项：`docs_always/add_new_mode/compare/videoedit_prepost_alignment.md`

## 3. 原始结果基线

原始命令已经在其他机器完成运行，本环境只使用产物，不再运行原始推理：

```bash
cd /mnt/shanhai-ai/liuh/VideoEdit-diffusers
python3 infer.py \
  --output_dir /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs/compare/origin_video_edit_diffusers \
  --output_name result \
  --chunks 1 \
  --clip_preprocess diffsynth
```

当前 `ffprobe` 证据：

| 文件 | 帧数 | 分辨率 | fps | 说明 |
| --- | ---: | --- | --- | --- |
| `/mnt/shanhai-ai/liuh/VideoEdit-diffusers/test_videos/lh/1080.mp4` | 105 | 1920x1080 | 25 | 原始输入视频 |
| `/mnt/shanhai-ai/liuh/VideoEdit-diffusers/test_videos/lh/mask_1080_merged.mp4` | 105 | 1920x1080 | 25 | 原始 mask 视频 |
| `outputs/compare/origin_video_edit_diffusers/result.mp4` | 81 | 1920x1080 | 25 | 主基线，paste-back，无 color-correct sidecar 语义 |
| `outputs/compare/origin_video_edit_diffusers/result_crop_only.mp4` | 81 | 1920x1080 | 25 | crop-only sidecar |
| `outputs/compare/origin_video_edit_diffusers/result_color.mp4` | 81 | 1920x1080 | 25 | color-correct sidecar，不能直接作为主基线 |

## 4. 必须对齐的真实语义

### 4.1 原始命令关键参数

| 类别 | 原始参数 | 值 | 对齐要求 |
| --- | --- | --- | --- |
| 输入 | `video_path` | `test_videos/lh/1080.mp4` | SGLang `--video-input-path` 使用同一绝对路径 |
| 输入 | `mask_path` | `test_videos/lh/mask_1080_merged.mp4` | SGLang `--mask-input-path` 使用同一绝对路径 |
| 输入 | `img_path` | `test_videos/lh/reg.png` | SGLang `--reference-image-path` 使用同一绝对路径 |
| 文本 | `prompt` | `一个男人在舞台演讲，背后有两行文字，背景保持不变。` | 必须完全一致 |
| 文本 | `negative_prompt` | 原始默认长中文负面词 | 使用 SGLang 默认 `DEFAULT_VIDEOEDIT_NEGATIVE_PROMPT`，内容应与原始默认一致 |
| 生成 | `num_inference_steps` | `10` | 必须显式传 `--num-inference-steps 10` |
| 生成 | `guidance_scale` | `5.0` | 显式传 `--guidance-scale 5.0` |
| 生成 | `seed` | `42` | 显式传 `--seed 42` |
| 生成 | `dtype` | `bf16` | 显式传 `--dtype bf16` |
| 生成 | `infer_len` | `81` | 显式传 `--infer-len 81` |
| 生成 | `overlap` | `9` | 显式传 `--overlap 9` |
| 生成 | `dynamic_cfg` | `True` | 显式传 `--dynamic-cfg` |
| 生成 | `use_clip` | `True` | 显式传 `--use-clip` |
| 预处理 | `num_frames` | `None`，但 `chunks=1` 只跑首窗口 | SGLang 首窗口对齐传 `--num-frames 80`，加 reference 后成为 81 |
| 预处理 | `bbox_padding` | `0` | 显式传 `--bbox-padding 0` |
| 预处理 | `bbox_expand_scale` | `2.5` | 显式传 `--bbox-expand-scale 2.5` |
| 预处理 | `dilate_px` | `0` | 必须显式传 `--dilate-px 0` |
| 预处理 | `mask_scale` | `1` | 必须显式传 `--mask-scale 1` |
| 后处理 | `feather_px` | `0` | 必须显式传 `--feather-px 0` |
| 后处理 | `adain_boundary_dilate` | `0` | 显式传 `--adain-boundary-dilate 0` |
| 输出 | `save_paste` | `True` | SGLang 保持 `--enable-paste-back` |
| 输出 | `save_crop` | `True` | 主指标不要求；可传 `--save-crop-only` 生成 sidecar |
| 输出 | `save_color` | `True` | SGLang 当前没有完全等价主流程，不作为主指标 |

### 4.2 原始 `--chunks 1` 的帧语义

1. 原始 `img_path` 默认非空，`prepare_global_inputs()` 会把 `reg.png` resize 后 prepend 到视频第 0 帧，并在 mask 第 0 帧 prepend 全黑帧。
2. 原始全局读取 `105` 个输入帧，加参考图后全局帧数为 `106`。
3. 原始 `--chunks 1` 通过 `parse_chunks()` 只选择窗口 `[0]`。
4. 首窗口范围为 `0..80`，共 `81` 帧，因此原始 `result.mp4` 只有首窗口 `81` 帧。
5. 原始 `infer.py` 中删除 prepended reference frame 的逻辑处于注释状态，所以 `result.mp4` 保留 reference 对应第 0 帧。
6. SGLang 当前没有 `--chunks` 参数；为了得到等价首窗口，应读取 `80` 个真实输入帧，加 reference 后得到 `81` 帧，再传 `--no-drop-reference-frame`。

### 4.3 SGLang 默认值中必须覆盖的差异

| 参数 | SGLang 默认 | 原始基线 | 对比动作 |
| --- | --- | --- | --- |
| `num_inference_steps` | `40` | `10` | 必须传 `--num-inference-steps 10` |
| `num_frames` | `81` | 首窗口 80 个真实帧 + reference | 必须传 `--num-frames 80` |
| `dilate_px` | `15` | `0` | 必须传 `--dilate-px 0` |
| `mask_scale` | `1.2` | `1` | 必须传 `--mask-scale 1` |
| `feather_px` | `15` | `0` | 必须传 `--feather-px 0` |
| `drop_reference_frame` | `True` | 保留第 0 帧 | 必须传 `--no-drop-reference-frame` |
| `clip_preprocess` | `diffuser` | `diffsynth` | 必须传 `--clip-preprocess diffsynth` |
| `overlap_commit_mode` | `weighted` | 原始多窗口 skip 语义 | 建议传 `--overlap-commit-mode native_skip` |
| `tail_padding_mode` | `reflect` | reverse mirror | 建议传 `--tail-padding-mode native_reverse_mirror` |
| `decode_mode` | `stream` | 完整帧列表 decode | 首窗口对齐建议传 `--decode-mode eager` |
| `enable_teacache` | `True` | 原始无 TeaCache | 必须传 `--no-enable-teacache` |
| `generator_device` | 未显式 | CPU generator | 必须传 `--generator-device cpu` |

### 4.4 低内存运行要求

为避免 1080p、81 帧 VideoEdit 在 VAE encode/decode 或组件常驻 GPU 时 OOM，本计划默认用低内存方式运行。低内存开关可能略降吞吐，但不应改变帧数、窗口语义、mask 语义或 seed 语义；若指标不合格，仍按第 8 节先排查参数与日志，不能仅因开启 offload 就跳过对齐诊断。

| 类别 | 参数 | 必须值 | 作用 |
| --- | --- | --- | --- |
| 模型权重 | `--dit-cpu-offload` | 开启 | DiT/transformer 按运行阶段放回 CPU，降低 GPU 常驻显存 |
| 模型权重 | `--text-encoder-cpu-offload` | 开启 | 文本编码后释放文本编码器 GPU 占用 |
| 模型权重 | `--image-encoder-cpu-offload` | 开启 | CLIP 图像编码后释放 image encoder GPU 占用 |
| VAE 权重 | `--vae-cpu-offload` | 开启 | VAE encode/decode 前后在 CPU/GPU 间迁移，降低常驻显存 |
| CPU 传输 | `--pin-cpu-memory` | 开启 | 为 offload 传输使用 pinned memory，减少 CPU/GPU 拷贝抖动 |
| VAE 分片 | `--vae-tiling` | 开启 | 启用 VAE tile encode/decode，降低单次空间显存峰值 |
| VAE 分片 | `--vae-slicing` | 开启 | 启用 VAE slicing，降低 batch/channel 方向显存峰值 |
| 编译 | `--no-enable-torch-compile` | 关闭 compile | 避免首次编译额外显存/缓存干扰对比运行 |

注意：`PipelineConfig.vae_tiling` 默认已经是 `True`，但命令仍显式传 `--vae-tiling`，确保日志和复现实验可读；`vae_slicing` 默认是 `False`，必须显式传 `--vae-slicing`。

## 5. 推荐 SGLang 首窗口对齐命令

主输出单独放到 `outputs/compare/sglang_video_edit_first_window/`，避免和原始输出混放。命令同时写出日志，便于第 8 节排查。该命令默认开启第 4.4 节低内存模式，包括 CPU offload 与 VAE tiling/slicing。

```bash
source /mnt/shanhai-ai/shanhai-workspace/zhouhao6/env/sglang/bin/activate
cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang
mkdir -p outputs/compare/sglang_video_edit_first_window outputs/compare/logs

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
  --save-crop-only \
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
  --dit-cpu-offload \
  --text-encoder-cpu-offload \
  --image-encoder-cpu-offload \
  --vae-cpu-offload \
  --pin-cpu-memory \
  --vae-tiling \
  --vae-slicing \
  --no-enable-torch-compile \
  --no-enable-teacache \
  --no-enable-frame-interpolation \
  --no-enable-upscaling \
  --warmup \
  --warmup-steps 1 \
  --perf-dump-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs/compare/sglang_video_edit_first_window/perf.json \
  2>&1 | tee outputs/compare/logs/sglang_video_edit_first_window.log
```

运行后预期主视频路径：

```text
/mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs/compare/sglang_video_edit_first_window/result.mp4
```

若 CLI 实际保存扩展名或目录层级不同，先确认：

```bash
find /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs/compare/sglang_video_edit_first_window -maxdepth 2 -type f | sort
```

## 6. 备选命令：全视频语义，不用于当前主指标

如果目标从“复现原始 `--chunks 1` 产物”改为“对齐原始 `num_frames=None` 的全视频读取语义”，才使用 `--num-frames -1`。该命令会读取全部 105 个真实输入帧，加 reference 后为 106 帧，并按 SGLang 窗口逻辑跑全部窗口；输出帧数不应拿来直接对比已有 81 帧 `result.mp4`。

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
  --save-crop-only \
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
  --dit-cpu-offload \
  --text-encoder-cpu-offload \
  --image-encoder-cpu-offload \
  --vae-cpu-offload \
  --pin-cpu-memory \
  --vae-tiling \
  --vae-slicing \
  --no-enable-torch-compile \
  --no-enable-teacache \
  --no-enable-frame-interpolation \
  --no-enable-upscaling
```

## 7. 对比与验收命令

### 7.1 文件和 metadata 检查

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

python - <<'PY'
import json
from pathlib import Path
p = Path('outputs/compare/sglang_video_edit_first_window/result.videoedit.json')
meta = json.loads(p.read_text())
print(json.dumps({
    'num_input_frames': meta.get('num_input_frames'),
    'num_output_frames': meta.get('num_output_frames'),
    'drop_reference_frame': meta.get('drop_reference_frame'),
    'peak_memory_mb': meta.get('peak_memory_mb'),
    'window_specs': meta.get('window_specs'),
    'window_materialize': meta.get('window_materialize'),
}, ensure_ascii=False, indent=2))
PY

rg -n -- '--dit-cpu-offload|--text-encoder-cpu-offload|--image-encoder-cpu-offload|--vae-cpu-offload|--pin-cpu-memory|--vae-tiling|--vae-slicing' \
  outputs/compare/logs/sglang_video_edit_first_window.log
```

预期：

- 两个视频均为 `81` 帧、`1920x1080`、`25 fps`。
- metadata 中 `num_input_frames` 为 `81`。
- metadata 中 `num_output_frames` 为 `81`。
- metadata 中 `drop_reference_frame` 为 `false`。
- metadata 中首个窗口 `start_index=0`、`valid_len=81`。
- 日志或本次执行命令中包含 `--dit-cpu-offload`、`--text-encoder-cpu-offload`、`--image-encoder-cpu-offload`、`--vae-cpu-offload`、`--pin-cpu-memory`、`--vae-tiling`、`--vae-slicing`。

若 SGLang 为 `80` 帧，优先检查是否漏传 `--no-drop-reference-frame`。若 SGLang 大于 `81` 帧，优先检查是否误用 `--num-frames -1` 或跑了全窗口语义。
若低内存日志检查失败，先确认日志是否记录完整命令；如果日志没有 echo 命令行，则以第 5 节实际执行命令为准，并在报告中手动记录这些低内存开关。

### 7.2 高指标门槛比较

采用较高门槛作为主验收。该门槛不要求 bit-level 完全一致，但会严格暴露采样、预处理、首帧、mask、编码等差异。

```bash
cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang
mkdir -p outputs/compare/reports

python -m sglang.multimodal_gen.runtime.videoedit.compare \
  --reference outputs/compare/origin_video_edit_diffusers/result.mp4 \
  --candidate outputs/compare/sglang_video_edit_first_window/result.mp4 \
  --report-json outputs/compare/reports/origin_vs_sglang_first_window.json \
  --min-ssim 0.95 \
  --max-mse 80 \
  --max-mae 5 \
  --allow-frame-count-delta 0 \
  --max-failed-frame-ratio 0.02
```

通过标准：命令退出码为 `0`，JSON 中：

- `summary.pass_compare = true`
- `summary.compared_frames = 81`
- `summary.reference_frame_count = 81`
- `summary.candidate_frame_count = 81`
- `summary.frame_count_delta = 0`

### 7.3 诊断门槛比较

如果高指标门槛不通过，先不要调整主阈值。额外跑诊断门槛，判断是“大面积未对齐”还是“接近但未达高指标”：

```bash
python -m sglang.multimodal_gen.runtime.videoedit.compare \
  --reference outputs/compare/origin_video_edit_diffusers/result.mp4 \
  --candidate outputs/compare/sglang_video_edit_first_window/result.mp4 \
  --report-json outputs/compare/reports/origin_vs_sglang_first_window_diagnostic.json \
  --min-ssim 0.90 \
  --max-mse 150 \
  --max-mae 8 \
  --allow-frame-count-delta 0 \
  --max-failed-frame-ratio 0.05
```

解释规则：

- 高指标通过：可报告“高指标对齐通过”。
- 高指标失败、诊断门槛通过：报告“当前高指标不合格，但内容接近；优先分析编码、SP 并行、VAE decode、CLIP 预处理细节”。
- 诊断门槛也失败：报告“当前指标不合格，存在明显未对齐；优先分析帧数/首帧/预处理/采样/模型路径”。

### 7.4 临时首帧错位诊断

若怀疑候选丢掉了 reference 首帧，只用于诊断，不作为正式通过依据：

```bash
python -m sglang.multimodal_gen.runtime.videoedit.compare \
  --reference outputs/compare/origin_video_edit_diffusers/result.mp4 \
  --candidate outputs/compare/sglang_video_edit_first_window/result.mp4 \
  --report-json outputs/compare/reports/origin_vs_sglang_drop_candidate_first_frame_probe.json \
  --drop-candidate-first-frame \
  --allow-frame-count-delta 1
```

如果该诊断明显改善，正式修复动作是重新运行 SGLang 命令并保留 `--no-drop-reference-frame`，而不是用 drop-first-frame 报告通过。

### 7.5 人工检查点

除 JSON 指标外，至少人工检查：

- 第 `0` 帧：是否为 `reg.png` reference/paste-back 后对应帧。
- mask 区域：演讲者背后文字修复区域是否位置一致。
- 非 mask 区域：背景和人物边缘是否被不必要修改。
- 边界：`feather_px=0` 时边界应更接近原始硬贴回；若边缘明显不同，检查 SGLang 是否仍使用默认 `feather_px=15`。
- 色彩：主视频 `result.mp4` 不启用 color-correction sidecar 语义；不要拿原始 `result_color.mp4` 和 SGLang 主输出直接比较。

## 8. 指标不合格时的日志与源码排查

如果第 7.2 节高指标不通过，最终报告必须明确写：

```text
当前高指标不合格。
高指标报告：outputs/compare/reports/origin_vs_sglang_first_window.json
诊断报告：outputs/compare/reports/origin_vs_sglang_first_window_diagnostic.json
SGLang 日志：outputs/compare/logs/sglang_video_edit_first_window.log
SGLang metadata：outputs/compare/sglang_video_edit_first_window/result.videoedit.json
最可能未对齐原因：<从下表选择并附证据>
```

### 8.1 先看 metadata

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path('outputs/compare/sglang_video_edit_first_window/result.videoedit.json')
meta = json.loads(p.read_text())
print('num_input_frames =', meta.get('num_input_frames'))
print('num_output_frames =', meta.get('num_output_frames'))
print('drop_reference_frame =', meta.get('drop_reference_frame'))
print('bbox =', meta.get('bbox'))
print('crop_h/crop_w =', meta.get('crop_h'), meta.get('crop_w'))
print('aligned_h/aligned_w =', meta.get('aligned_h'), meta.get('aligned_w'))
print('window_specs =')
for spec in meta.get('window_specs', []):
    print(spec)
print('window_materialize =')
for item in meta.get('window_materialize', []):
    print(item)
PY
```

重点判断：

- `num_input_frames != 81`：说明 `--num-frames 80` 或 reference prepend 语义没有对齐。
- `num_output_frames != 81`：说明 drop-reference、全窗口提交或保存阶段不对齐。
- `drop_reference_frame = true`：说明漏传 `--no-drop-reference-frame`。
- `window_specs` 不是单窗口：说明运行的不是首窗口复现语义。
- `bbox/crop/aligned` 与预期差异明显：优先回看 `bbox_padding`、`bbox_expand_scale`、`dilate_px`、`mask_scale`。

### 8.2 再看 SGLang 运行日志

```bash
rg -n \
  "server_args|Running pipeline stages|windowing|window_start|window_done|Output saved|video_input_path|mask_input_path|reference_image_path|num_frames|num_input_frames|num_output_frames|drop_reference|num_inference_steps|guidance|seed|generator|clip|teacache|bbox|crop|aligned|dilate|mask_scale|feather|decode|tail_padding|overlap_commit|error|warning|traceback" \
  outputs/compare/logs/sglang_video_edit_first_window.log
```

若日志只保留最终 JSON，仍应至少检查其中：

- `output_file_path` 是否指向 `outputs/compare/sglang_video_edit_first_window/result.mp4`。
- `metrics` 是否显示异常耗时、异常显存或中途 fallback。
- 是否出现 `warning`、`Failed to save video with reference profile`、`Falling back to default imageio writer`。

### 8.3 对照原因表

| 现象 | 最可能原因 | 证据位置 | 处理动作 |
| --- | --- | --- | --- |
| 候选 `80` 帧 | 丢掉 reference 首帧 | `result.videoedit.json` 的 `drop_reference_frame=true` | 重新运行并传 `--no-drop-reference-frame` |
| 候选 `>81` 帧 | 跑了全视频/多窗口 | `num_input_frames>81` 或 `window_specs` 多于 1 个 | 重新运行并传 `--num-frames 80` |
| 第 0 帧差异极大 | reference 图路径、resize、首帧保留不一致 | metadata 的 `reference_image_path`、第 0 帧人工检查 | 确认 `reg.png` 绝对路径和 `--no-drop-reference-frame` |
| 所有帧差异极大 | 采样参数或模型路径不一致 | 日志 `model_path`、`transformer_path`、`num_inference_steps`、`seed` | 对齐第 5 节命令，重点确认 transformer 权重 |
| mask 区域位置错 | bbox/mask 预处理不一致 | metadata `bbox/crop_h/crop_w/aligned_h/aligned_w` | 确认 `--dilate-px 0 --mask-scale 1 --bbox-expand-scale 2.5` |
| 边界过软或非 mask 区域改变 | paste-back feather 默认未覆盖 | 命令和日志中的 `feather_px` | 重新运行并传 `--feather-px 0` |
| 内容接近但高指标失败 | 编码、SP 并行、decode、CLIP 细节差异 | 高指标失败但诊断门槛通过 | 追加单卡 `--num-gpus 1 --sp-degree 1` 复测；检查保存 profile 与 CLIP 分支 |
| crop-only 接近但 paste-back 不接近 | 后处理 paste-back 未对齐 | SGLang `_crop_only.mp4` 与主视频比较差异 | 对照 `paste_back()` 的 mask、feather、AdaIN 参数 |
| 日志出现 TeaCache | 未关闭服务化优化 | 日志或参数中 `enable_teacache=true` | 重新运行并传 `--no-enable-teacache` |
| OOM 或显存峰值过高 | 低内存开关未全部启用，或 VAE 分片未生效 | 日志命令、`peak_memory_mb`、CUDA OOM 栈 | 重新运行并确认 `--dit-cpu-offload --text-encoder-cpu-offload --image-encoder-cpu-offload --vae-cpu-offload --pin-cpu-memory --vae-tiling --vae-slicing` 全部存在 |

### 8.4 单卡复测命令

若高指标不合格但诊断门槛通过，且日志显示参数已对齐，可用单卡复测排除 SP 并行数值差异：

```bash
python -m sglang.multimodal_gen.runtime.videoedit.cli repair \
  --model-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model \
  --transformer-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model/transformer \
  --prompt "一个男人在舞台演讲，背后有两行文字，背景保持不变。" \
  --video-input-path /mnt/shanhai-ai/liuh/VideoEdit-diffusers/test_videos/lh/1080.mp4 \
  --mask-input-path /mnt/shanhai-ai/liuh/VideoEdit-diffusers/test_videos/lh/mask_1080_merged.mp4 \
  --reference-image-path /mnt/shanhai-ai/liuh/VideoEdit-diffusers/test_videos/lh/reg.png \
  --output-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs/compare/sglang_video_edit_first_window_sp1 \
  --output-file-name result \
  --num-gpus 1 \
  --sp-degree 1 \
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
  --save-crop-only \
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
  --dit-cpu-offload \
  --text-encoder-cpu-offload \
  --image-encoder-cpu-offload \
  --vae-cpu-offload \
  --pin-cpu-memory \
  --vae-tiling \
  --vae-slicing \
  --no-enable-torch-compile \
  --no-enable-teacache \
  --no-enable-frame-interpolation \
  --no-enable-upscaling \
  2>&1 | tee outputs/compare/logs/sglang_video_edit_first_window_sp1.log
```

## 9. 暂不覆盖范围

- 不重新运行 `/mnt/shanhai-ai/liuh/VideoEdit-diffusers/infer.py`。
- 不把原始 `result_color.mp4` 作为主指标基线，因为 SGLang 当前没有完全等价的 `_color.mp4` sidecar 输出。
- 不比较多窗口长视频结果，除非先为 SGLang 增加 `--chunks` 或明确改用全视频语义。
- 不要求 bit-level 完全一致；当前计划以帧数一致、首帧语义一致、高指标门槛通过和人工检查通过为准。

## 10. 后续源码改进建议

1. 给 `python/sglang/multimodal_gen/runtime/videoedit/cli.py` 增加 `--chunks`，在 `WanVideoEditSamplingParams` 与 `build_videoedit_window_specs()` 后过滤窗口，复刻原始 `parse_chunks()` 的 `<=0 = all`、正数取前 N 个窗口语义。
2. 增加 `--save-color` 或 color-correct sidecar，复刻 `/mnt/shanhai-ai/liuh/VideoEdit-diffusers/utils/postprocess.py:85` 中 `color_correct=True` 路径，方便比较 `result_color.mp4`。
3. 增加 compare preset，例如 `--preset videoedit-diffusers-first-window`，自动设置 `num_frames=80`、`drop_reference_frame=False`、`num_inference_steps=10`、`dilate_px=0`、`mask_scale=1`、`feather_px=0`、`clip_preprocess=diffsynth`、`teacache=False`。
4. 增加 `--low-memory` preset，自动设置 `dit_cpu_offload=True`、`text_encoder_cpu_offload=True`、`image_encoder_cpu_offload=True`、`vae_cpu_offload=True`、`pin_cpu_memory=True`、`vae_tiling=True`、`vae_slicing=True`、`enable_torch_compile=False`，减少长视频对比命令中的重复参数。
5. 在 metadata 中追加采样参数快照，例如 `num_inference_steps`、`guidance_scale`、`seed`、`generator_device`、`clip_preprocess`、`enable_teacache`、`decode_mode`、`mask_scale`、`feather_px`、`vae_tiling`、`vae_slicing`、`vae_cpu_offload`，降低之后排查日志的成本。
6. 在 compare 工具中增加可选 mask-aware 指标：mask 区域、非 mask 区域、边界 ring 分别输出 SSIM/MSE/MAE，便于区分生成差异和 paste-back 差异。

## 11. 本计划自检

- 没有使用 `待定占位`、`未实现占位`、`后续补充` 作为执行占位。
- 所有命令均使用绝对路径或从 `/mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang` 可解析的相对路径。
- 明确区分主基线 `result.mp4`、crop-only sidecar、color sidecar。
- 明确高指标不合格时不能声称完成，必须结合 SGLang 运行日志和 `result.videoedit.json` 给出不合格原因。
