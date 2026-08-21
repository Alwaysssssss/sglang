# VideoEdit-diffusers 与 SGLang 全视频推理对齐测试

> 状态：仅完成测试设计，**尚未执行**。本文档编写于 2026-08-20。

本文档在 [`env.md`](./env.md) 的 48 帧单窗口数值基线之上，验证同一输入的
**完整有效视频时间线**：两套实现都运行全部滑窗，比较最终 crop-only 和 full-frame
结果，并检查 SGLang full-frame 输出的视频属性是否与推理输入对齐。

本文中的命令是后续执行清单。本次编写文档时不得启动模型、不得运行视频推理、不得生成
比较报告。

## 1. 测试目标和验收边界

### 1.1 目标

1. 原始 `VideoEdit-diffusers/infer.py` 完整处理 case0008 的全部有效配对帧。
2. SGLang VideoEdit 使用相同输入、checkpoint 和采样参数完整处理同一时间线。
3. 以两端 `*_crop_only.mp4` 做算法结果主比较。
4. 以两端 paste-back full-frame MP4 做最终成片辅助比较。
5. 检查 SGLang full-frame 输出和实际推理输入的分辨率、帧率、解码帧数、时长、
   显示宽高比、编码及颜色属性是否对齐。
6. 检查 SGLang 的 `.videoedit.json` 是否证明所有源帧均已生成，且全视频滑窗没有被截断。

### 1.2 “完整视频”和“原视频”的定义

case0008 原始数据存在已知不一致：`video.mp4` 可解码 210 帧，`mask.mp4` 可解码
209 帧。SGLang 在应用 `--num-frames` 之前会严格检查 video/mask 原始长度相等，因此不能
直接把这两个文件作为全量输入。

本测试把两者的**最长公共有效前缀**定义为完整有效时间线：

```text
VE_FULL_FRAMES = min(raw video frames, raw mask frames) = 209
```

后续将两个原始文件逐帧解码，并各自只保留前 209 帧，生成等长的规范化输入副本。
本文的“推理输入原视频”指该 209 帧 `VE_INPUT_VIDEO`；SGLang full 输出必须与它的所有
硬门禁属性对齐。

原始素材 `VE_RAW_VIDEO` 仍进入属性报告。除帧数和时长外，它的静态显示/编码属性也应与
SGLang full 输出一致；帧数固定多 1、时长约多一帧是已知数据差异，不记为 SGLang 回归。
不得复制最后一帧 mask 来伪造第 210 帧语义。如果产品要求覆盖原始 video 的 210 帧，
必须先由数据提供方补齐真实的第 210 帧 mask，再建立另一条 golden。

### 1.3 主门禁与辅助检查

| 检查 | 对象 | 是否为硬门禁 |
|---|---|---|
| 全量完成性 | 两端 crop/full、SGLang metadata | 是 |
| 算法数值对齐 | original crop vs SGLang crop | 是 |
| 最终成片对齐 | original full vs SGLang full | 辅助；失败不能覆盖 crop 失败，也不能被 crop 通过掩盖 |
| SGLang 视频属性对齐 | `VE_INPUT_VIDEO` vs SGLang full | 是 |
| 原始 210 帧素材差异记录 | `VE_RAW_VIDEO` vs SGLang full | 是静态属性门禁；帧数/时长差 1 帧仅记录 |
| 主观观感和滑窗接缝 | 两端 full/crop | 人工辅助 |

## 2. 公共环境

在一个新的 Bash 会话中执行。为避免覆盖 48 帧基线，输入、输出和报告均使用独立目录。

```bash
set -o pipefail

export VE_CASE_DIR=/mnt/shanhai-ai/liuh/VideoEdit-diffusers/datas/edit_val_cases/0008
export VE_RAW_VIDEO="$VE_CASE_DIR/video.mp4"
export VE_RAW_MASK="$VE_CASE_DIR/mask.mp4"
export VE_REFERENCE_IMAGE="$VE_CASE_DIR/reference.png"
export VE_PROMPT="$(tr -d '\r\n' < "$VE_CASE_DIR/prompt.txt")"

export VE_ORIGINAL_REPO=/mnt/shanhai-ai/liuh/VideoEdit-diffusers
export VE_ORIGINAL_MODEL=/mnt/shanhai-ai/shanhai-workspace/fanruidi/projects/VideoEdit-new/VideoEdit_diffusers/pretrain_models/Wan2.1-I2V-14B-480P-Diffusers
export VE_ORIGINAL_TRANSFORMER=/mnt/shanhai-ai/liuh/VideoEdit-diffusers/ckpts/step_47500

export VE_SGLANG_REPO=/mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang
export VE_SGLANG_PYTHON=/home/root/uv-envs/sglang-llm-diffusion/bin/python
export VE_SGLANG_MODEL=/mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model
export VE_SGLANG_TRANSFORMER="$VE_SGLANG_MODEL/transformer"

export VE_RUN_NAME=case0008_full_step47500_seed42
export VE_INPUT_DIR="$VE_SGLANG_REPO/outputs/video-edit-full-inputs-step47500"
export VE_ORIGINAL_OUTPUT="$VE_SGLANG_REPO/outputs/video-edit-full-reference-step47500"
export VE_SGLANG_OUTPUT="$VE_SGLANG_REPO/outputs/video-edit-full-sglang-step47500"
export VE_REPORT_DIR="$VE_SGLANG_REPO/outputs/video-edit-full-reports-step47500"
mkdir -p "$VE_INPUT_DIR" "$VE_ORIGINAL_OUTPUT" "$VE_SGLANG_OUTPUT" "$VE_REPORT_DIR"

export VE_INFER_LEN=49
export VE_OVERLAP=5
export VE_STEPS=40
export VE_GUIDANCE_SCALE=5.0
export VE_SEED=42
```

记录待测代码版本和输入摘要；报告中必须保留这些值：

```bash
git -C "$VE_ORIGINAL_REPO" rev-parse HEAD
git -C "$VE_SGLANG_REPO" rev-parse HEAD
sha256sum "$VE_RAW_VIDEO" "$VE_RAW_MASK" "$VE_REFERENCE_IMAGE"
```

## 3. 构造严格等长的全量输入

### 3.1 统计真实解码帧数

不要以 `nb_frames` 单独决定长度；它可能为 `N/A`。以 `-count_frames` 得到的
`nb_read_frames` 为准：

```bash
count_decoded_frames() {
  ffprobe -v error -count_frames -select_streams v:0 \
    -show_entries stream=nb_read_frames -of default=nw=1:nk=1 "$1"
}

export VE_RAW_VIDEO_FRAMES="$(count_decoded_frames "$VE_RAW_VIDEO")"
export VE_RAW_MASK_FRAMES="$(count_decoded_frames "$VE_RAW_MASK")"

test "$VE_RAW_VIDEO_FRAMES" -eq 210
test "$VE_RAW_MASK_FRAMES" -eq 209

if [ "$VE_RAW_VIDEO_FRAMES" -le "$VE_RAW_MASK_FRAMES" ]; then
  export VE_FULL_FRAMES="$VE_RAW_VIDEO_FRAMES"
else
  export VE_FULL_FRAMES="$VE_RAW_MASK_FRAMES"
fi
test "$VE_FULL_FRAMES" -eq 209
```

`210/209` 是当前 case 和输入文件的契约。如果这里发生变化，应停止测试并确认数据版本，
不能静默沿用旧 golden。

### 3.2 生成规范化公共前缀

这里使用解码后的 `trim=end_frame`，而不是依赖 B-frame/GOP 边界的 `-t -c copy`，确保
两个规范化文件实际都只含前 209 个显示帧。两端推理共享同一份规范化视频，因此这次
重新编码不会造成实现间输入差异。

```bash
export VE_RAW_FPS="$(ffprobe -v error -select_streams v:0 \
  -show_entries stream=avg_frame_rate -of default=nw=1:nk=1 "$VE_RAW_VIDEO")"
export VE_MASK_FPS="$(ffprobe -v error -select_streams v:0 \
  -show_entries stream=avg_frame_rate -of default=nw=1:nk=1 "$VE_RAW_MASK")"
test "$VE_RAW_FPS" = "$VE_MASK_FPS"

export VE_INPUT_VIDEO="$VE_INPUT_DIR/case0008_video_full_${VE_FULL_FRAMES}.mp4"
export VE_INPUT_MASK="$VE_INPUT_DIR/case0008_mask_full_${VE_FULL_FRAMES}.mp4"

ffmpeg -hide_banner -loglevel error -y -i "$VE_RAW_VIDEO" \
  -map 0:v:0 \
  -vf "trim=end_frame=${VE_FULL_FRAMES},setpts=PTS-STARTPTS" \
  -frames:v "$VE_FULL_FRAMES" -fps_mode cfr -r "$VE_RAW_FPS" -an \
  -c:v libx264 -preset slow -b:v 10M -maxrate 10M -bufsize 20M \
  -pix_fmt yuv420p -movflags +faststart "$VE_INPUT_VIDEO"

ffmpeg -hide_banner -loglevel error -y -i "$VE_RAW_MASK" \
  -map 0:v:0 \
  -vf "trim=end_frame=${VE_FULL_FRAMES},setpts=PTS-STARTPTS" \
  -frames:v "$VE_FULL_FRAMES" -fps_mode cfr -r "$VE_RAW_FPS" -an \
  -c:v libx264 -preset slow -b:v 10M -maxrate 10M -bufsize 20M \
  -pix_fmt yuv420p -movflags +faststart "$VE_INPUT_MASK"

test "$(count_decoded_frames "$VE_INPUT_VIDEO")" -eq "$VE_FULL_FRAMES"
test "$(count_decoded_frames "$VE_INPUT_MASK")" -eq "$VE_FULL_FRAMES"
```

再检查两个规范化文件的显示几何和帧率完全一致：

```bash
for VE_FILE in "$VE_INPUT_VIDEO" "$VE_INPUT_MASK"; do
  ffprobe -v error -select_streams v:0 \
    -show_entries stream=codec_name,pix_fmt,width,height,avg_frame_rate \
    -of default=noprint_wrappers=1 "$VE_FILE"
done
```

## 4. 运行前门禁

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

test "$(count_decoded_frames "$VE_INPUT_VIDEO")" -eq "$VE_FULL_FRAMES"
test "$(count_decoded_frames "$VE_INPUT_MASK")" -eq "$VE_FULL_FRAMES"
nvidia-smi -i 2,3
```

除路径存在外，还必须确认 `$VE_SGLANG_TRANSFORMER` 确实由
`$VE_ORIGINAL_TRANSFORMER` 的 `step_47500` 转换而来，并在报告中记录转换来源。两边的
negative prompt 当前相同，命令沿用内置值；若任一实现修改默认值，必须在两条命令中显式
传入完全相同的文本。

## 5. 完整运行原始 VideoEdit-diffusers

物理 GPU 2 独占运行。`--num_frames` 必须显式为 209，因为原始入口的默认值是 48；
不要传 `--chunks`，省略它才表示运行全部窗口。

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
  --output_name "${VE_RUN_NAME}_reference" \
  --prompt "$VE_PROMPT" \
  --chunks -1 \
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
  --no_save_color \
  2>&1 | tee "$VE_REPORT_DIR/${VE_RUN_NAME}_reference.log"
```

预期产物：

```bash
export VE_REFERENCE_FULL="$VE_ORIGINAL_OUTPUT/${VE_RUN_NAME}_reference.mp4"
export VE_REFERENCE_CROP="$VE_ORIGINAL_OUTPUT/${VE_RUN_NAME}_reference_crop_only.mp4"
```

日志必须包含 `Generated 209 frames`，且退出码为 0。

## 6. 完整运行 SGLang VideoEdit

原始任务完全退出后，再让物理 GPU 3 独占运行。参数继续使用 `env.md` 的严格数值对齐
配置：关闭 TeaCache、插帧和超分，使用 `torch_sdpa`，并采用逐层 DiT CPU offload。

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
  --output-file-name "${VE_RUN_NAME}_sglang.mp4" \
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
  --perf-dump-path "$VE_SGLANG_OUTPUT/${VE_RUN_NAME}_sglang_perf.json" \
  2>&1 | tee "$VE_REPORT_DIR/${VE_RUN_NAME}_sglang.log"
```

预期产物：

```bash
export VE_CANDIDATE_FULL="$VE_SGLANG_OUTPUT/${VE_RUN_NAME}_sglang.mp4"
export VE_CANDIDATE_CROP="$VE_SGLANG_OUTPUT/${VE_RUN_NAME}_sglang_crop_only.mp4"
export VE_SGLANG_METADATA="$VE_SGLANG_OUTPUT/${VE_RUN_NAME}_sglang.videoedit.json"
```

## 7. 产物完整性和全视频滑窗检查

先确认所有产物存在且非空：

```bash
for VE_RESULT in \
  "$VE_REFERENCE_FULL" \
  "$VE_REFERENCE_CROP" \
  "$VE_CANDIDATE_FULL" \
  "$VE_CANDIDATE_CROP" \
  "$VE_SGLANG_METADATA"; do
  test -s "$VE_RESULT" || { echo "missing output: $VE_RESULT"; exit 1; }
done
```

在 `ref_frame_idx=0`、`num_frames=209`、`infer_len=49`、`overlap=5` 下，模型时间线为
`[reference] + 209 source frames`，共 210 个位置。stride 为 44，预期只执行 long pass，
共 5 个窗口：

| window | pass-local start | valid length | reverse-mirror padding | 新提交的 source global index |
|---:|---:|---:|---:|---|
| 0 | 0 | 49 | 0 | `0..47` |
| 1 | 44 | 49 | 0 | `48..91` |
| 2 | 88 | 49 | 0 | `92..135` |
| 3 | 132 | 49 | 0 | `136..179` |
| 4 | 176 | 34 | 15 | `180..208` |

最终 source global index 必须恰好为 `0..208`，不得包含 reference，不得缺帧或重复提交。
SGLang metadata 的硬要求如下：

- `num_input_frames == 209`；
- `num_output_frames == 209`；
- `drop_reference_frame == false`；
- `enable_paste_back == true`；
- `window_specs` 的 pass 全为 `long`；
- window start 恰好为 `[0, 44, 88, 132, 176]`；
- valid length 恰好为 `[49, 49, 49, 49, 34]`；
- reflected count 恰好为 `[0, 0, 0, 0, 15]`。

## 8. 比较两套实现的最终结果

### 8.1 crop-only 主门禁

```bash
cd "$VE_SGLANG_REPO"

"$VE_SGLANG_PYTHON" \
  python/sglang/multimodal_gen/runtime/videoedit/compare.py \
  --reference "$VE_REFERENCE_CROP" \
  --candidate "$VE_CANDIDATE_CROP" \
  --report-json "$VE_REPORT_DIR/${VE_RUN_NAME}_compare_crop.json" \
  --min-ssim 0.97 \
  --max-mse 25.0 \
  --max-mae 2.5 \
  --allow-frame-count-delta 0 \
  --max-failed-frame-ratio 0.0
```

通过条件：

- 比较器实际解码并比较 209 帧；
- `failed_frames == []`；
- `pass_compare == true`；
- CLI 返回 0。

这里沿用 `env.md` 的 case0008 crop 严格门限，但它此前只由 48 帧基线校准。首次全视频
运行不得为了让结果通过而在同一轮放宽门限；如果后 161 帧或窗口边界失败，应保存逐帧
报告，先判断是实现差异、滑窗传播差异还是门限需要独立校准。

### 8.2 full-frame 最终成片辅助比较

```bash
"$VE_SGLANG_PYTHON" \
  python/sglang/multimodal_gen/runtime/videoedit/compare.py \
  --reference "$VE_REFERENCE_FULL" \
  --candidate "$VE_CANDIDATE_FULL" \
  --report-json "$VE_REPORT_DIR/${VE_RUN_NAME}_compare_full.json" \
  --min-ssim 0.98 \
  --max-mse 25.0 \
  --max-mae 2.5 \
  --allow-frame-count-delta 0 \
  --max-failed-frame-ratio 0.0
```

full-frame 比较用于发现 paste-back、羽化、颜色和编码链路回归。它含有大量未编辑背景，
因此不能替代 crop 主门禁，也不能用较高的 full SSIM 掩盖 crop 失败。

`compare.py` 在尺寸不同时会先 resize candidate，所以必须先通过下一节的独立视频属性
门禁，不能只看 SSIM/MSE/MAE。

## 9. 检查 SGLang 输出与原视频的视频属性

### 9.1 属性口径

SGLang full 输出相对 209 帧推理输入的硬门禁：

- `codec_name`、`pix_fmt`；
- `width`、`height`；
- `sample_aspect_ratio`、`display_aspect_ratio`；
- `avg_frame_rate`；
- 实际解码帧数 `nb_read_frames`；
- 容器时长与 `frames / fps` 的差不超过一帧时长；
- `color_range`、`color_space`、`color_transfer`、`color_primaries`、`field_order`；
- MP4 容器格式。

`profile`、`bit_rate`、`time_base`、`r_frame_rate` 和容器提供的 `nb_frames` 只记录，不做
精确相等门禁。输出经过重新编码，码率会受内容复杂度和 muxer 影响；实际帧数必须由
`nb_read_frames` 判断。

crop-only 不要求与原视频同尺寸，但 original crop 和 SGLang crop 的宽高、帧率、帧数、
codec 和 pix_fmt 必须一致。

### 9.2 自动属性报告和 metadata 门禁

下面脚本生成一个可审计 JSON，并在任何硬门禁失败时返回 1。它只检查属性，不计算像素
指标：

```bash
export VE_PROPERTY_REPORT="$VE_REPORT_DIR/${VE_RUN_NAME}_video_properties.json"

"$VE_SGLANG_PYTHON" - <<'PY'
import json
import os
import subprocess
from fractions import Fraction
from pathlib import Path


def probe(path: str) -> dict:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        (
            "stream=codec_name,profile,pix_fmt,width,height,sample_aspect_ratio,"
            "display_aspect_ratio,avg_frame_rate,r_frame_rate,time_base,duration,"
            "nb_frames,nb_read_frames,color_range,color_space,color_transfer,"
            "color_primaries,field_order:format=format_name,duration,bit_rate"
        ),
        "-of",
        "json",
        path,
    ]
    payload = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
    if not payload.get("streams"):
        raise RuntimeError(f"no video stream: {path}")
    return {
        "path": path,
        "stream": payload["streams"][0],
        "format": payload.get("format", {}),
    }


paths = {
    "raw_video": os.environ["VE_RAW_VIDEO"],
    "input_video": os.environ["VE_INPUT_VIDEO"],
    "reference_full": os.environ["VE_REFERENCE_FULL"],
    "candidate_full": os.environ["VE_CANDIDATE_FULL"],
    "reference_crop": os.environ["VE_REFERENCE_CROP"],
    "candidate_crop": os.environ["VE_CANDIDATE_CROP"],
}
profiles = {name: probe(path) for name, path in paths.items()}
expected_frames = int(os.environ["VE_FULL_FRAMES"])
checks = []


def add_check(name: str, passed: bool, detail: str) -> None:
    checks.append({"name": name, "pass": bool(passed), "detail": detail})


def stream(name: str) -> dict:
    return profiles[name]["stream"]


def frame_count(name: str) -> int:
    value = stream(name).get("nb_read_frames")
    if value in (None, "N/A"):
        raise RuntimeError(f"ffprobe did not return nb_read_frames for {name}")
    return int(value)


def fps(name: str) -> Fraction:
    value = stream(name).get("avg_frame_rate")
    if not value or value == "0/0":
        raise RuntimeError(f"invalid avg_frame_rate for {name}: {value}")
    return Fraction(value)


def duration(name: str) -> float:
    value = profiles[name]["format"].get("duration") or stream(name).get("duration")
    if value in (None, "N/A"):
        raise RuntimeError(f"ffprobe did not return duration for {name}")
    return float(value)


for name in ("input_video", "reference_full", "candidate_full", "reference_crop", "candidate_crop"):
    actual = frame_count(name)
    add_check(f"{name}.decoded_frames", actual == expected_frames, f"actual={actual}, expected={expected_frames}")

input_fps = fps("input_video")
for name in ("reference_full", "candidate_full", "reference_crop", "candidate_crop"):
    actual = fps(name)
    add_check(f"{name}.fps", actual == input_fps, f"actual={actual}, expected={input_fps}")

expected_duration = expected_frames / float(input_fps)
duration_tolerance = 1.0 / float(input_fps)
for name in ("input_video", "reference_full", "candidate_full", "reference_crop", "candidate_crop"):
    actual = duration(name)
    delta = abs(actual - expected_duration)
    add_check(
        f"{name}.duration",
        delta <= duration_tolerance + 1e-9,
        f"actual={actual}, expected={expected_duration}, tolerance={duration_tolerance}",
    )

static_fields = (
    "codec_name",
    "pix_fmt",
    "width",
    "height",
    "sample_aspect_ratio",
    "display_aspect_ratio",
    "avg_frame_rate",
    "color_range",
    "color_space",
    "color_transfer",
    "color_primaries",
    "field_order",
)
for field in static_fields:
    actual = stream("candidate_full").get(field)
    expected = stream("input_video").get(field)
    add_check(
        f"candidate_full.input_video.{field}",
        actual == expected,
        f"actual={actual!r}, expected input_video={expected!r}",
    )
add_check(
    "candidate_full.input_video.format_name",
    profiles["candidate_full"]["format"].get("format_name")
    == profiles["input_video"]["format"].get("format_name"),
    (
        f"actual={profiles['candidate_full']['format'].get('format_name')!r}, "
        f"expected={profiles['input_video']['format'].get('format_name')!r}"
    ),
)

# 原始 210 帧素材只参与静态属性检查；长度差在报告中明确记录，不算回归。
for field in static_fields:
    actual = stream("candidate_full").get(field)
    expected = stream("raw_video").get(field)
    add_check(
        f"candidate_full.raw_video_static.{field}",
        actual == expected,
        f"actual={actual!r}, expected raw_video={expected!r}",
    )
raw_delta = frame_count("raw_video") - frame_count("candidate_full")

crop_equal_fields = (
    "codec_name",
    "pix_fmt",
    "width",
    "height",
    "sample_aspect_ratio",
    "display_aspect_ratio",
    "avg_frame_rate",
)
for field in crop_equal_fields:
    actual = stream("candidate_full").get(field)
    expected = stream("reference_full").get(field)
    add_check(
        f"candidate_full.reference_full.{field}",
        actual == expected,
        f"actual={actual!r}, expected reference_full={expected!r}",
    )
for field in crop_equal_fields:
    actual = stream("candidate_crop").get(field)
    expected = stream("reference_crop").get(field)
    add_check(
        f"candidate_crop.reference_crop.{field}",
        actual == expected,
        f"actual={actual!r}, expected reference_crop={expected!r}",
    )

with open(os.environ["VE_SGLANG_METADATA"], encoding="utf-8") as handle:
    metadata = json.load(handle)
specs = metadata.get("window_specs", [])
metadata_expectations = {
    "num_input_frames": expected_frames,
    "num_output_frames": expected_frames,
    "drop_reference_frame": False,
    "enable_paste_back": True,
}
for field, expected in metadata_expectations.items():
    actual = metadata.get(field)
    add_check(f"metadata.{field}", actual == expected, f"actual={actual!r}, expected={expected!r}")

metadata_fps = metadata.get("fps")
add_check(
    "metadata.fps",
    metadata_fps is not None and abs(float(metadata_fps) - float(input_fps)) <= 1e-9,
    f"actual={metadata_fps!r}, expected={float(input_fps)!r}",
)

add_check(
    "metadata.crop_geometry",
    (metadata.get("crop_w"), metadata.get("crop_h"))
    == (stream("candidate_crop").get("width"), stream("candidate_crop").get("height")),
    (
        f"metadata={(metadata.get('crop_w'), metadata.get('crop_h'))}, "
        f"candidate_crop={(stream('candidate_crop').get('width'), stream('candidate_crop').get('height'))}"
    ),
)
add_check("metadata.window_passes", [item.get("pass") for item in specs] == ["long"] * 5, str([item.get("pass") for item in specs]))
add_check("metadata.window_starts", [item.get("start_index") for item in specs] == [0, 44, 88, 132, 176], str([item.get("start_index") for item in specs]))
add_check("metadata.window_valid_len", [item.get("valid_len") for item in specs] == [49, 49, 49, 49, 34], str([item.get("valid_len") for item in specs]))
add_check("metadata.window_reflected_count", [item.get("reflected_count") for item in specs] == [0, 0, 0, 0, 15], str([item.get("reflected_count") for item in specs]))
add_check("raw_video.known_frame_delta", raw_delta == 1, f"actual={raw_delta}, expected=1")

failures = [item for item in checks if not item["pass"]]
report = {
    "status": "pass" if not failures else "fail",
    "expected_effective_frames": expected_frames,
    "known_raw_video_frame_delta": raw_delta,
    "known_raw_video_frame_delta_expected": 1,
    "profiles": profiles,
    "metadata_path": os.environ["VE_SGLANG_METADATA"],
    "checks": checks,
    "failures": failures,
}
report_path = Path(os.environ["VE_PROPERTY_REPORT"])
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({"status": report["status"], "failures": failures}, indent=2, ensure_ascii=False))

raise SystemExit(1 if failures else 0)
PY
```

只有脚本返回 0 且报告中 `status == "pass"` 才表示属性对齐。若 SGLang 保存 reference
profile 失败并退回默认 imageio writer，codec、pix_fmt 或颜色属性门禁可能失败；这应视为
真实的输出链路回归，不能只凭肉眼播放正常而忽略。

## 10. 人工检查滑窗边界

数值报告通过后，再逐帧或慢速播放以下全局边界附近的 crop/full：

```text
48, 92, 136, 180
```

建议检查每个边界前后至少 2 帧，即 `46..50`、`90..94`、`134..138`、`178..182`，关注：

- 编辑区域是否跳变、闪烁或漂移；
- paste-back 边缘是否突然变化；
- 背景是否出现非预期修改；
- 最后一窗 reverse-mirror padding 是否泄漏到输出；
- 第 0 帧和第 208 帧是否存在且顺序正确。

人工观感不能替代逐帧数值门禁，但应作为最终成片验收记录。

## 11. 最终通过条件

一次完整运行只有同时满足以下条件才算通过：

1. 两条推理命令退出码均为 0，日志和版本信息已归档。
2. 两端 crop/full 均能实际解码 209 帧。
3. SGLang metadata 记录 209 输入帧、209 输出帧和预期的 5 个完整窗口。
4. crop 报告 `compared_frames == 209`、`failed_frames == []`、`pass_compare == true`。
5. full 报告完成并作为独立辅助结果记录；不得用它覆盖 crop 结论。
6. 视频属性脚本返回 0，JSON 中 `status == "pass"`。
7. 属性报告明确记录 raw video 与有效推理时间线的已知 1 帧差异。
8. 人工检查四个窗口边界和首尾帧，没有新增异常。

## 12. 测试报告模板

执行完成后，在测试记录中填写：

```markdown
# case0008 full-video alignment result

- Date:
- Operator:
- Original repo commit:
- SGLang repo commit:
- Original checkpoint:
- SGLang converted checkpoint and conversion source:
- Raw input SHA256:
- Effective input SHA256:
- GPU model / driver / CUDA:
- Effective frames: 209
- infer_len / overlap / steps / seed: 49 / 5 / 40 / 42

## Execution

- Original inference exit code / elapsed time / peak GPU memory:
- SGLang inference exit code / elapsed time / peak GPU and host memory:
- Output paths:

## Crop comparison

- compared_frames:
- ssim_mean / ssim_min:
- mse_mean / mse_max:
- mae_mean / mae_max:
- failed_frames:
- pass_compare:

## Full comparison

- compared_frames:
- ssim_mean / ssim_min:
- mse_mean / mse_max:
- mae_mean / mae_max:
- failed_frames:
- pass_compare:

## Video properties

- property report:
- SGLang vs effective input: pass/fail
- raw-video known frame delta: expected 1 / actual
- unexpected differences:

## Manual seam review

- frame 48:
- frame 92:
- frame 136:
- frame 180:
- first/last frame:

## Final conclusion

- PASS / FAIL
- Failure classification and follow-up:
```
