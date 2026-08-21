# VideoEdit-diffusers 与 SGLang VideoEdit 对齐测试

本文档使用同一输入、checkpoint 和采样参数，对比原始 `VideoEdit-diffusers`
与 SGLang VideoEdit。当前只跑 48 个源视频帧的单窗口基线；SGLang 使用 CPU
offload 降低显存占用，不启用缓存、插帧或超分。

主验收对象是两端生成的 `*_crop_only.mp4`：它对应 bbox 内真正由模型生成、尚未 paste-back 的区域。完整视频仅用于辅助检查回贴、尺寸和编码，不能用完整画面中的未编辑背景稀释 crop 误差。

## 1. 公共环境

在同一个 Bash 会话中执行：

```bash
export VE_CASE_DIR=/mnt/shanhai-ai/liuh/VideoEdit-diffusers/datas/edit_val_cases/0008
export VE_REFERENCE_IMAGE="$VE_CASE_DIR/reference.png"
export VE_PROMPT="$(tr -d '\r\n' < "$VE_CASE_DIR/prompt.txt")"

# case0008 的原 video/mask 分别为 210/209 帧。严格契约会先检查原始长度相等，
# 因此用无重编码 remux 得到等长测试副本，再显式取前 48 帧。1.04 秒覆盖 54 个
# 解码帧，避免在 B-frame 边界截断时漏掉第 48 个显示帧。
export VE_GOLDEN_INPUT_DIR=/mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs/video-edit-inputs-v2
mkdir -p "$VE_GOLDEN_INPUT_DIR"
ffmpeg -hide_banner -loglevel error -y -i "$VE_CASE_DIR/video.mp4" \
  -map 0:v:0 -t 1.04 -c copy -an "$VE_GOLDEN_INPUT_DIR/case0008_video_sync.mp4"
ffmpeg -hide_banner -loglevel error -y -i "$VE_CASE_DIR/mask.mp4" \
  -map 0:v:0 -t 1.04 -c copy -an "$VE_GOLDEN_INPUT_DIR/case0008_mask_sync.mp4"
export VE_INPUT_VIDEO="$VE_GOLDEN_INPUT_DIR/case0008_video_sync.mp4"
export VE_INPUT_MASK="$VE_GOLDEN_INPUT_DIR/case0008_mask_sync.mp4"

export VE_ORIGINAL_REPO=/mnt/shanhai-ai/liuh/VideoEdit-diffusers
export VE_ORIGINAL_MODEL=/mnt/shanhai-ai/shanhai-workspace/fanruidi/projects/VideoEdit-new/VideoEdit_diffusers/pretrain_models/Wan2.1-I2V-14B-480P-Diffusers
export VE_ORIGINAL_TRANSFORMER=/mnt/shanhai-ai/liuh/VideoEdit-diffusers/ckpts/step_47500

export VE_SGLANG_REPO=/mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang
export VE_SGLANG_PYTHON=/home/root/uv-envs/sglang-llm-diffusion/bin/python
export VE_SGLANG_MODEL=/mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model
export VE_SGLANG_TRANSFORMER="$VE_SGLANG_MODEL/transformer"

export VE_OUTPUT_ROOT="$VE_SGLANG_REPO/outputs"
export VE_ORIGINAL_OUTPUT="$VE_OUTPUT_ROOT/video-edit_outputs-step47500"
export VE_SGLANG_OUTPUT="$VE_OUTPUT_ROOT/sglang_outputs-step47500"
mkdir -p "$VE_ORIGINAL_OUTPUT" "$VE_SGLANG_OUTPUT"

export VE_SOURCE_FRAMES=48
export VE_INFER_LEN=49
export VE_OVERLAP=5
export VE_STEPS=40
export VE_GUIDANCE_SCALE=5.0
export VE_SEED=42
```

`VE_SOURCE_FRAMES=48` 是单窗口基线的显式限制；VideoEdit 未传 `num_frames` 时默认处理
完整视频。其余 case、checkpoint 和生成参数与当前 `infer.py` 的默认参数对齐。命令仍
显式传入这些参数，避免后续默认值变化影响复现。输出目录/文件名是本对齐测试的命名
覆盖；另外仅为减少冗余输出而显式使用 `--no_save_color`，不改变 crop 主验收对象。

两端当前内置的 negative prompt 相同，因此下面不重复传入，避免复制长中文字符串时
破坏 Shell 命令。如果以后修改了任一端的默认值，应在两条命令中显式传入同一个值。

运行前做一次路径检查：

```bash
for VE_FILE in \
  "$VE_INPUT_VIDEO" \
  "$VE_INPUT_MASK" \
  "$VE_REFERENCE_IMAGE" \
  "$VE_ORIGINAL_MODEL/model_index.json" \
  "$VE_ORIGINAL_TRANSFORMER/transformer/config.json" \
  "$VE_SGLANG_MODEL/model_index.json" \
  "$VE_SGLANG_TRANSFORMER/config.json"; do
  test -e "$VE_FILE" || { echo "missing: $VE_FILE"; exit 1; }
done

nvidia-smi -i 2,3
```

`VE_SGLANG_MODEL` 必须是完整的 Diffusers 模型根目录，并且根目录下直接包含
`transformer/`、`vae/`、`text_encoder/`、`tokenizer/`、`scheduler/` 和
`model_index.json`。否则模型注册阶段会报
`missing required component directories: transformer`；此时不能仅靠
`--transformer-path` 修复不完整的模型根目录。

同时确认 `$VE_SGLANG_TRANSFORMER` 是由 `$VE_ORIGINAL_TRANSFORMER` 的
`step_47500` 权重转换而来；路径存在只说明目录结构完整，不能证明两端 checkpoint
相同。若重新转换权重，应把转换来源一并记录到测试报告中。

## 2. 运行原始 VideoEdit-diffusers

使用物理 GPU 2：

```bash
cd "$VE_ORIGINAL_REPO"

CUDA_VISIBLE_DEVICES=2 \
conda run --no-capture-output -n VideoEdit python infer.py \
  --video_path "$VE_INPUT_VIDEO" \
  --mask_path "$VE_INPUT_MASK" \
  --img_path "$VE_REFERENCE_IMAGE" \
  --ref_frame_idx 0 \
  --bridge_overlap 5 \
  --model_path "$VE_ORIGINAL_MODEL" \
  --transformer_path "$VE_ORIGINAL_TRANSFORMER" \
  --output_dir "$VE_ORIGINAL_OUTPUT" \
  --output_name case0008_reference \
  --prompt "$VE_PROMPT" \
  --num_frames "$VE_SOURCE_FRAMES" \
  --infer_len "$VE_INFER_LEN" \
  --overlap "$VE_OVERLAP" \
  --num_inference_steps "$VE_STEPS" \
  --guidance_scale "$VE_GUIDANCE_SCALE" \
  --seed "$VE_SEED" \
  --dtype bf16 \
  --dynamic_cfg \
  --vae_tiling \
  --use_clip \
  --clip_preprocess diffuser \
  --bbox_padding 0 \
  --bbox_expand_scale 1.6 \
  --dilate_px 8 \
  --mask_scale 1.0 \
  --feather_px 8 \
  --adain_boundary_dilate 0 \
  --save_paste \
  --save_crop \
  --no_save_color
```

主对比文件为 `$VE_ORIGINAL_OUTPUT/case0008_reference_crop_only.mp4`；`case0008_reference.mp4` 仅作 full-frame 辅助检查。

## 3. 运行 SGLang VideoEdit

原始任务结束后，使用物理 GPU 3：

```bash
cd "$VE_SGLANG_REPO"

CUDA_VISIBLE_DEVICES=3 \
SGLANG_CACHE_DIT_ENABLED=false \
"$VE_SGLANG_PYTHON" -m sglang.multimodal_gen.runtime.videoedit.cli repair \
  --model-path "$VE_SGLANG_MODEL" \
  --transformer-path "$VE_SGLANG_TRANSFORMER" \
  --prompt "$VE_PROMPT" \
  --video-input-path "$VE_INPUT_VIDEO" \
  --mask-input-path "$VE_INPUT_MASK" \
  --reference-image-path "$VE_REFERENCE_IMAGE" \
  --output-path "$VE_SGLANG_OUTPUT" \
  --output-file-name case0008_sglang.mp4 \
  --num-frames "$VE_SOURCE_FRAMES" \
  --ref-frame-idx 0 \
  --bridge-overlap 5 \
  --infer-len "$VE_INFER_LEN" \
  --overlap "$VE_OVERLAP" \
  --num-inference-steps "$VE_STEPS" \
  --guidance-scale "$VE_GUIDANCE_SCALE" \
  --seed "$VE_SEED" \
  --dtype bf16 \
  --dynamic-cfg \
  --dynamic-cfg-max-step 15 \
  --dynamic-cfg-min 1.0 \
  --bbox-padding 0 \
  --bbox-expand-scale 0.3 \
  --dilate-px 8 \
  --mask-scale 1.0 \
  --feather-px 8 \
  --adain-boundary-dilate 0 \
  --enable-paste-back \
  --save-crop-only \
  --use-clip \
  --clip-preprocess diffuser \
  --decode-mode stream \
  --no-dit-cpu-offload \
  --dit-layerwise-offload \
  --dit-offload-prefetch-size 0 \
  --text-encoder-cpu-offload \
  --image-encoder-cpu-offload \
  --vae-cpu-offload \
  --pin-cpu-memory \
  --no-enable-teacache \
  --no-enable-frame-interpolation \
  --no-enable-upscaling \
  --num-gpus 1 \
  --tp-size 1 \
  --sp-degree 1 \
  --ulysses-degree 1 \
  --ring-degree 1 \
  --attention-backend torch_sdpa \
  --perf-dump-path "$VE_SGLANG_OUTPUT/case0008_sglang_perf.json"
```

主对比文件为 `$VE_SGLANG_OUTPUT/case0008_sglang_crop_only.mp4`；`case0008_sglang.mp4` 仅作 full-frame 辅助检查。

SGLang 使用逐层 DiT CPU offload，并把文本编码器、图像编码器和 VAE 卸载到 CPU。
`--dit-offload-prefetch-size 0` 只预取一层，优先降低峰值显存；不要同时开启
`--dit-cpu-offload`、Cache-DiT 或 FSDP。`--pin-cpu-memory` 会增加主机内存占用，
但通常能提高 CPU/GPU 传输效率。

`--no-enable-teacache` 是严格数值对齐使用的显式覆盖；VideoEdit 的正常运行默认仍开启
TeaCache。这样 golden 可以排除缓存误差，同时不改变常规推理的加速默认值。

严格对齐使用 `torch_sdpa`，对应原算法 Diffusers 默认的 native PyTorch attention。
FlashAttention 可以作为性能模式另行回归，但不进入本节的数值 golden。

两端的 bbox 参数数值不同但含义对齐：原始实现的 `1.6` 表示最终宽高倍率；SGLang
的 `0.3` 表示每边扩展 30%，最终宽高同样是 `1 + 2 × 0.3 = 1.6` 倍。

## 4. 检查和比较输出

```bash
export VE_REFERENCE_CROP="$VE_ORIGINAL_OUTPUT/case0008_reference_crop_only.mp4"
export VE_CANDIDATE_CROP="$VE_SGLANG_OUTPUT/case0008_sglang_crop_only.mp4"
export VE_REFERENCE_FULL="$VE_ORIGINAL_OUTPUT/case0008_reference.mp4"
export VE_CANDIDATE_FULL="$VE_SGLANG_OUTPUT/case0008_sglang.mp4"

for VE_RESULT in "$VE_REFERENCE_CROP" "$VE_CANDIDATE_CROP"; do
  test -s "$VE_RESULT" || { echo "missing output: $VE_RESULT"; exit 1; }
  ffprobe -v error -select_streams v:0 \
    -show_entries stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_frames \
    -of default=noprint_wrappers=1 "$VE_RESULT"
done

cd "$VE_SGLANG_REPO"

"$VE_SGLANG_PYTHON" \
  python/sglang/multimodal_gen/runtime/videoedit/compare.py \
  --reference "$VE_REFERENCE_CROP" \
  --candidate "$VE_CANDIDATE_CROP" \
  --report-json "$VE_OUTPUT_ROOT/case0008_step47500_compare_report_crop.json" \
  --min-ssim 0.97 \
  --max-mse 25.0 \
  --max-mae 2.5 \
  --allow-frame-count-delta 0 \
  --max-failed-frame-ratio 0.0
```

只有上面的 crop compare 是算法验收主门禁。需要检查 paste-back/编码回归时，再运行下面的辅助比较；其结果不得覆盖 crop 失败：

```bash
for VE_RESULT in "$VE_REFERENCE_FULL" "$VE_CANDIDATE_FULL"; do
  test -s "$VE_RESULT" || { echo "missing output: $VE_RESULT"; exit 1; }
  ffprobe -v error -select_streams v:0 \
    -show_entries stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_frames \
    -of default=noprint_wrappers=1 "$VE_RESULT"
done

"$VE_SGLANG_PYTHON" \
  python/sglang/multimodal_gen/runtime/videoedit/compare.py \
  --reference "$VE_REFERENCE_FULL" \
  --candidate "$VE_CANDIDATE_FULL" \
  --report-json "$VE_OUTPUT_ROOT/case0008_step47500_compare_report_full.json" \
  --min-ssim 0.98 \
  --max-mse 25.0 \
  --max-mae 2.5 \
  --allow-frame-count-delta 0 \
  --max-failed-frame-ratio 0.0
```

### 4.1 `ffprobe` 字段

`ffprobe` 负责检查容器/视频流元数据，不参与像素指标的计算。FFmpeg 把
`avg_frame_rate` 定义为平均帧率，而 `nb_frames` 是“已知时的流帧数”，未知时可能为
`0`/`N/A`；因此 `nb_frames` 只作快速检查，最终以 `compare.py` 实际成功解码到的帧数
为准。来源：[FFmpeg `AVStream` 字段定义](https://ffmpeg.org/doxygen/trunk/structAVStream.html)、
[ffprobe 官方文档](https://ffmpeg.org/ffprobe.html)。

| 字段 | 含义 | case0008 参考值 |
|---|---|---|
| `codec_name` | 视频流编码器名称；它不直接代表解码后的像素质量。 | 两端应相同；当前流程预期为 `h264`。 |
| `pix_fmt` | 编码像素格式。 | crop 的奇数高度要求两端都是 `yuv444p`，避免 `yuv420p` 静默补齐尺寸。 |
| `width` / `height` | 编码视频的显示帧宽高。 | crop 主验收两端必须都是 `1778 × 747`；full 辅助输出预期为 `1920 × 1080`。 |
| `avg_frame_rate` | 流的平均帧率，以有理数输出。 | 两端必须完全相同；case0008 预期为 `50/1`。 |
| `nb_frames` | 容器能够提供时的流帧数。 | 两端都应为 `48`；若显示 `N/A`，仍须由下方比较器解码确认 48 帧。 |

尺寸检查不能省略：比较器遇到两帧尺寸不同会先把 candidate resize 到 reference 再算
指标，因此单看 SSIM/MSE/MAE 可能漏掉分辨率回归。实际解码、resize 和逐帧配对逻辑见
[`compare.py` 第 25–35、75–96 行](../../python/sglang/multimodal_gen/runtime/videoedit/compare.py#L25-L96)。

### 4.2 逐帧指标的定义、方向和范围

以下理论范围以本脚本当前处理的 8-bit RGB 帧，即每个通道取值 `0..255` 为前提。
NumPy 的 `mean` 默认对未指定轴的整个数组求算术平均，`absolute` 逐元素取绝对值；这正是
这里 RGB 三通道共同参与 MSE/MAE 的语义。来源：
[NumPy `mean`](https://numpy.org/doc/stable/reference/generated/numpy.mean.html)、
[NumPy `absolute`](https://numpy.org/doc/stable/reference/generated/numpy.absolute.html)、
[`compare.py` 第 85–98 行](../../python/sglang/multimodal_gen/runtime/videoedit/compare.py#L85-L98)。
脚本先把两帧转为 `float32` 再相减，避免 `uint8` 减法溢出；`np.mean` 未另行指定
`dtype`，所以逐帧 MSE/MAE 使用 NumPy 文档所述的输入浮点精度累计。

| JSON 字段 | 精确定义 | 理论范围与方向 | 本项目门限/参考范围 |
|---|---|---|---|
| `index` | 成功解码并配对后的零基帧序号。 | 整数 `0..compared_frames-1`。 | 本例为 `0..47`。 |
| `ssim` | 先按 OpenCV `COLOR_RGB2GRAY`（灰度权重为 `Y=0.299R+0.587G+0.114B`）把 RGB 转成灰度，再以 `11 × 11`、`sigma=1.5` 的 Gaussian 局部统计量计算结构相似度；动态范围固定为 255，`C1=(0.01×255)^2`、`C2=(0.03×255)^2`，最后对整张 score map 求均值。 | 理想实数运算下约为 `[-1, 1]`，`1` 表示相同，越高越相似；相关视频通常落在 `0..1`。 | **crop 经验门限**：每帧 `>= 0.97`；旧 `step_46500` 10-step 校准 mean/min 为 `0.979944055 / 0.975248443`。 |
| `mse` | `mean((reference_rgb - candidate_rgb)^2)`，在所有像素和三个颜色通道上平均。 | `[0, 65025]`，`0` 最好，越低越相似；平方会放大少数大误差。 | **crop 经验门限**：每帧 `<= 25.0`，等价于 RMSE `<= 5` 个 8-bit 灰阶；旧 `step_46500` 10-step 校准 mean/max 为 `15.209595 / 15.380247`。 |
| `mae` | `mean(abs(reference_rgb - candidate_rgb))`，同样在所有像素和三个颜色通道上平均。 | `[0, 255]`，`0` 最好，越低越相似；比 MSE 更接近典型的平均像素偏差。 | **crop 经验门限**：每帧 `<= 2.5`，约为满量程的 `0.98%`；旧 `step_46500` 10-step 校准 mean/max 为 `2.270228 / 2.354623`。 |
| `psnr` | MSE 为 0 时是 `+∞`，否则为 `20 log10(255 / sqrt(MSE))`。 | 当前 8-bit 前提下为 `[0, +∞] dB`，越高越好；相同帧为 `+∞`。 | **仅观察，不参与 pass**。MSE 门限 25 对应 PSNR `>= 34.15 dB`；旧 `step_46500` 10-step crop 的有限值 mean 为 `36.309747 dB`。 |
| `max_abs_diff` | 单帧所有像素、所有 RGB 通道中的最大绝对差。 | 整数 `[0, 255]`，越低越好；对单个离群像素极敏感。 | **仅观察，不参与 pass**；旧 `step_46500` 10-step crop 全视频最大值为 `84`，不能把它误读为平均偏差。 |
| `pass_frame` | `ssim >= min_ssim AND mse <= max_mse AND mae <= max_mae`。 | 布尔值；三个门限必须同时满足。 | 本 golden 要求 48 帧全部为 `true`。 |

这里的 SSIM 是仓库内的 OpenCV/NumPy 实现，不是对
`skimage.metrics.structural_similarity` 的直接调用。scikit-image 官方文档可用于理解
SSIM 衡量“结构相似度”以及显式指定动态范围、Gaussian 权重和 `sigma` 的意义，但不同
实现的边界处理和聚合细节可能不同，不能直接共用阈值。来源：
[scikit-image SSIM 官方文档](https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_similarity)、
[OpenCV 颜色转换公式](https://docs.opencv.org/4.11.0/de/d25/imgproc_color_conversions.html#color-conversions)、
[`compare.py` 第 41–54 行](../../python/sglang/multimodal_gen/runtime/videoedit/compare.py#L41-L54)。
SSIM 只看灰度结构，MSE/MAE 则看 RGB 误差；三项联合可避免单一指标掩盖颜色偏移或局部
大误差。scikit-image 的官方示例也说明，相同 MSE 不一定代表相同的感知结构相似度：
[Structural similarity index 示例](https://scikit-image.org/docs/stable/auto_examples/transform/plot_ssim.html)。

### 4.3 视频级汇总和通过条件

| JSON 字段/派生量 | 含义和范围 | 本项目口径 |
|---|---|---|
| `compared_frames` | 两边可比帧数，即丢弃可选首帧后两边解码帧数的较小值；正整数。 | 必须为 `48`。本例不使用 `--drop-*-first-frame`。 |
| 帧数差 | `abs(reference_frames - candidate_frames)`，非负整数；脚本当前不把它写入 JSON。 | **契约门限** `--allow-frame-count-delta 0`：必须为 0。若超限，脚本在生成报告前直接抛出 `ValueError`，而不是写出 `pass_compare=false`。 |
| `ssim_mean` / `ssim_min` | 所有逐帧 SSIM 的均值/最小值，范围约 `[-1,1]`，越高越好。 | 仅用于观察整体水平和最差帧；pass 仍按每帧 `ssim` 判断。 |
| `mse_mean` / `mse_max` | 所有逐帧 MSE 的均值/最大值，范围 `[0,65025]`，越低越好。 | 仅用于观察；pass 仍按每帧 `mse` 判断。 |
| `mae_mean` / `mae_max` | 所有逐帧 MAE 的均值/最大值，范围 `[0,255]`，越低越好。 | 仅用于观察；pass 仍按每帧 `mae` 判断。 |
| `psnr_mean` | 只对 PSNR 有限的帧求均值；若所有帧都完全相同则为 `+∞`。 | 仅观察，不参与 pass。注意混有完全相同帧时，`+∞` 帧不会进入该均值。 |
| `max_abs_diff` | 所有帧 `max_abs_diff` 的最大值。 | 仅观察，不参与 pass。 |
| `failed_frames` | 所有 `pass_frame=false` 的零基帧序号列表。 | golden 期望空列表 `[]`。 |
| 失败帧比例 | `len(failed_frames) / compared_frames`，范围 `[0,1]`；脚本当前不单独写入 JSON。 | **经验门限** `--max-failed-frame-ratio 0.0`：必须为 0，即 48 帧不允许任何一帧失败。 |
| `thresholds` | 原样记录本次命令使用的五个门限，便于报告审计。 | 应与本节命令完全一致。 |
| `pass_compare` | 当且仅当失败帧比例 `<= max_failed_frame_ratio` 时为 `true`；帧数差超限则更早报错。 | golden 必须为 `true`，CLI 才返回 0；否则返回 1。 |

上述汇总和返回码的精确定义见
[`compare.py` 第 57–68、100–156 行](../../python/sglang/multimodal_gen/runtime/videoedit/compare.py#L57-L156)。
不要用 `ssim_mean`、`mse_mean` 或 `mae_mean` 代替逐帧门限：均值可能让少量坏帧被大量好帧
稀释。

### 4.4 为什么采用这组数值

这组数值是 **case0008 crop 对齐回归的项目经验门限，不是 SSIM/MSE/MAE 的行业通用质量
分级**。它来自旧 `step_46500`、10-step、seed 42 的 2026-08-20 crop 报告：SSIM mean/min
`0.979944055 / 0.975248443`，MSE mean/max `15.209595 / 15.380247`，MAE mean/max
`2.270228 / 2.354623`，有限 PSNR mean `36.309747 dB`，全 crop 视频
`max_abs_diff=84`，且 48/48 帧通过。

`0.97 / 25.0 / 2.5` 分别在这次旧 10-step crop 校准最差值之外留出约 `0.00525` SSIM、
`9.62` MSE 和 `0.145` MAE 的 guard band，用于容纳 MP4 重编码和 BF16 执行造成的小幅数值扰动，同时仍
明显严于此前已观察到的失配结果。帧数差和失败帧比例设为 0，是因为这里验证的是固定
seed、固定输入、固定 48 帧的实现对齐，而不是主观视频质量；丢帧或单帧越界都应立即
暴露。若更换 case、编码参数、像素格式、attention backend、dtype 或硬件，应先收集多次
已知正确运行的逐帧分布，再重新校准经验门限，不应把本组数值直接当作通用标准。

旧 `step_46500` 在同一输入上的 40-step crop 为 SSIM mean/min
`0.975073031 / 0.970572816`，但 MSE mean/max `26.036753 / 26.501226`、MAE mean/max
`2.823670 / 2.891105`，因此在上述联合门限下 48/48 帧失败。SSIM 通过不能抵消
MSE/MAE 失败。这些数值只记录旧 checkpoint 的门限来源，不能当作当前 `step_47500`
命令的预期结果；`step_47500` 首次正确对齐运行后应根据逐帧报告重新确认门限。

两边都把 edited reference 作为 out-of-band 条件帧；它没有 source global index，
天然不会进入输出。因此 crop 和 full 都应为 48 帧，比较时不需要额外丢首帧。crop 比较命令返回 0
且报告中 `pass_compare=true` 才表示通过。

如果 SGLang 仍然 OOM，先检查主机可用内存和 GPU 3 是否被其他进程占用；当前命令已
采用最低预取量的逐层卸载，再降低显存通常需要量化或多卡方案，这会引入新的对齐变量。
