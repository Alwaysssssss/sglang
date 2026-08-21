# VideoEdit DMD 蒸馏模型与非蒸馏模型对比测试

> 状态：测试方案，尚未执行。本文档编写于 2026-08-20。

本文档以 [`env.md`](./env.md) 的 case0008、48 源帧单窗口基线为基础，在
SGLang VideoEdit 中对比以下两套配置：

| 组别 | Transformer | 采样步数 | Guidance scale | Dynamic CFG |
|---|---|---:|---:|---|
| A：非蒸馏基线 | `step_47500` 转换权重 | 40 | 5.0 | 开启 |
| B：DMD 蒸馏 | `step-55000-dmd-200/transformer` | 4 | 1.0 | 关闭 |

B 组必须使用：

```text
FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
```

当前 SGLang VideoEdit pipeline 已固定使用这三个 scheduler 参数，因此命令只需显式传
`--num-inference-steps 4` 和 `--guidance-scale 1.0`，不要添加当前 CLI 不支持的
`--sigma-min` 或 `--extra-one-step`。对应实现见
[`wan_videoedit_pipeline.py`](../../python/sglang/multimodal_gen/runtime/pipelines/wan_videoedit_pipeline.py)
和
[`videoedit_flow_match.py`](../../python/sglang/multimodal_gen/runtime/models/schedulers/videoedit_flow_match.py)。

## 1. 测试目的与解释边界

本测试回答两个问题：

1. DMD 模型按其 4-step/CFG=1.0 配方能否完成与非蒸馏模型相同的 VideoEdit 请求，且
   输出帧数、尺寸、帧率、crop/paste-back 契约均正确。
2. 相比非蒸馏的 40-step/CFG=5.0 配方，DMD 的端到端耗时、去噪耗时、峰值显存和主观
   编辑质量如何变化。

这不是逐像素实现对齐测试。两组使用不同 transformer、步数和 CFG，预期像素结果不同；
[`env.md`](./env.md) 中 `SSIM >= 0.97 / MSE <= 25 / MAE <= 2.5` 的 golden 门限不适用。
本文仍生成 SSIM/MSE/MAE 报告，但它们只描述两组输出的差异，不能单独判定哪一组质量
更好。

还要注意，当前可用非蒸馏基线是 `step_47500`，蒸馏权重目录是
`step-55000-dmd-200`。两者训练 checkpoint 不同，因此观察到的质量差异同时包含
checkpoint、蒸馏和采样配方的影响，不能写成严格的“仅蒸馏收益”。若后续取得与 DMD
教师完全对应的 step-55000 非蒸馏权重，应以它替换 A 组再做正式消融。

## 2. 公共环境

在同一个新的 Bash 会话中执行。先按 [`env.md` 第 1 节](./env.md#1-公共环境) 构造
case0008 的 48 帧等长输入，然后设置以下变量：

```bash
set -o pipefail

export VE_CASE_DIR=/mnt/shanhai-ai/liuh/VideoEdit-diffusers/datas/edit_val_cases/0008
export VE_REFERENCE_IMAGE="$VE_CASE_DIR/reference.png"
export VE_PROMPT="$(tr -d '\r\n' < "$VE_CASE_DIR/prompt.txt")"
export VE_INPUT_DIR=/mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/outputs/video-edit-inputs-v2
export VE_INPUT_VIDEO="$VE_INPUT_DIR/case0008_video_sync.mp4"
export VE_INPUT_MASK="$VE_INPUT_DIR/case0008_mask_sync.mp4"

export VE_SGLANG_REPO=/mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang
export VE_SGLANG_PYTHON=/home/root/uv-envs/sglang-llm-diffusion/bin/python
export VE_MODEL_ROOT=/mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model
export VE_BASELINE_TRANSFORMER="$VE_MODEL_ROOT/transformer"
export VE_DMD_TRANSFORMER=/mnt/shanhai-ai/shanhai-workspace/zhouhao6/tyx/workspace/difusser-model/step-55000-dmd-200/transformer

export VE_RUN_ROOT="$VE_SGLANG_REPO/outputs/video-edit-dmd-compare-case0008"
export VE_BASELINE_OUTPUT="$VE_RUN_ROOT/nondistilled-step47500"
export VE_DMD_OUTPUT="$VE_RUN_ROOT/distilled-step55000-dmd200"
export VE_REPORT_DIR="$VE_RUN_ROOT/reports"
mkdir -p "$VE_BASELINE_OUTPUT" "$VE_DMD_OUTPUT" "$VE_REPORT_DIR"

export VE_SOURCE_FRAMES=48
export VE_INFER_LEN=49
export VE_OVERLAP=5
export VE_SEED=42
export VE_GPU=3
```

除 transformer、采样步数、guidance scale 和与其对应的 dynamic CFG 外，两组必须使用
同一输入、模型根目录中的 T5/CLIP/VAE、seed、dtype、attention backend、offload、bbox、
mask、paste-back 和编码配置。两组应在同一张空闲 GPU 上串行运行，不能并行抢占资源。

## 3. 运行前门禁

### 3.1 路径与模型结构

```bash
for VE_FILE in \
  "$VE_INPUT_VIDEO" \
  "$VE_INPUT_MASK" \
  "$VE_REFERENCE_IMAGE" \
  "$VE_MODEL_ROOT/model_index.json" \
  "$VE_BASELINE_TRANSFORMER/config.json" \
  "$VE_BASELINE_TRANSFORMER/diffusion_pytorch_model.safetensors.index.json" \
  "$VE_DMD_TRANSFORMER/config.json" \
  "$VE_DMD_TRANSFORMER/diffusion_pytorch_model.safetensors.index.json"; do
  test -e "$VE_FILE" || { echo "missing: $VE_FILE"; exit 1; }
done

test "$(find "$VE_BASELINE_TRANSFORMER" -maxdepth 1 -name '*.safetensors' | wc -l)" -eq 4
test "$(find "$VE_DMD_TRANSFORMER" -maxdepth 1 -name '*.safetensors' | wc -l)" -eq 4

"$VE_SGLANG_PYTHON" - "$VE_BASELINE_TRANSFORMER/config.json" \
  "$VE_DMD_TRANSFORMER/config.json" <<'PY'
import json
import sys

def load_config(path):
    with open(path, encoding="utf-8") as file:
        config = json.load(file)
    config.pop("_diffusers_version", None)
    return config

baseline = load_config(sys.argv[1])
dmd = load_config(sys.argv[2])
assert baseline == dmd, "transformer architecture config mismatch"
print("transformer architecture configs match")
PY
```

最后一条命令必须打印 `transformer architecture configs match` 并以 0 退出。它只验证
网络结构兼容，不证明两个 checkpoint 有训练来源关系。

记录代码、输入和权重摘要，正式报告必须保留这些文件：

```bash
git -C "$VE_SGLANG_REPO" rev-parse HEAD | tee "$VE_REPORT_DIR/sglang_commit.txt"
git -C "$VE_SGLANG_REPO" status --short | tee "$VE_REPORT_DIR/sglang_status.txt"
sha256sum "$VE_INPUT_VIDEO" "$VE_INPUT_MASK" "$VE_REFERENCE_IMAGE" \
  | tee "$VE_REPORT_DIR/input_sha256.txt"
sha256sum "$VE_BASELINE_TRANSFORMER"/*.safetensors \
  | tee "$VE_REPORT_DIR/nondistilled_sha256.txt"
sha256sum "$VE_DMD_TRANSFORMER"/*.safetensors \
  | tee "$VE_REPORT_DIR/dmd_sha256.txt"

nvidia-smi -i "$VE_GPU" | tee "$VE_REPORT_DIR/nvidia_smi_before.txt"
```

### 3.2 Scheduler 配置

下面的检查确认当前代码默认 `flow_shift=5.0`，并确认 4-step scheduler 使用
`sigma_min=0.0` 和 `extra_one_step=True`：

```bash
cd "$VE_SGLANG_REPO"

"$VE_SGLANG_PYTHON" - <<'PY'
from sglang.multimodal_gen.configs.pipeline_configs.videoedit_wan import (
    WanVideoEditPipelineConfig,
)
from sglang.multimodal_gen.runtime.models.schedulers.videoedit_flow_match import (
    VideoEditFlowMatchScheduler,
)

config = WanVideoEditPipelineConfig()
assert config.flow_shift == 5.0, config.flow_shift

scheduler = VideoEditFlowMatchScheduler(
    shift=config.flow_shift,
    sigma_min=0.0,
    extra_one_step=True,
)
scheduler.set_timesteps(4)
assert scheduler.shift == 5.0
assert scheduler.sigma_min == 0.0
assert scheduler.extra_one_step is True
assert len(scheduler.timesteps) == 4
print("scheduler:", scheduler.__class__.__name__)
print("timesteps:", scheduler.timesteps.tolist())
print("sigmas:", scheduler.sigmas.tolist())
PY
```

预期四个 sigma 约为 `1.0, 0.9375, 0.8333333, 0.625`；最后一次 `step()` 再积分到
`sigma=0`。检查失败时应停止测试，不能在未知 scheduler 配置下生成结果。

## 4. 运行 A 组：非蒸馏基线

此命令复用 [`env.md`](./env.md) 的非蒸馏 SGLang 配置。为比较纯模型配方，继续关闭
TeaCache、插帧和超分。

```bash
cd "$VE_SGLANG_REPO"

CUDA_VISIBLE_DEVICES="$VE_GPU" \
SGLANG_CACHE_DIT_ENABLED=false \
"$VE_SGLANG_PYTHON" -m sglang.multimodal_gen.runtime.videoedit.cli repair \
  --model-path "$VE_MODEL_ROOT" \
  --transformer-path "$VE_BASELINE_TRANSFORMER" \
  --prompt "$VE_PROMPT" \
  --video-input-path "$VE_INPUT_VIDEO" \
  --mask-input-path "$VE_INPUT_MASK" \
  --reference-image-path "$VE_REFERENCE_IMAGE" \
  --output-path "$VE_BASELINE_OUTPUT" \
  --output-file-name case0008_nondistilled.mp4 \
  --num-frames "$VE_SOURCE_FRAMES" \
  --ref-frame-idx 0 \
  --bridge-overlap 5 \
  --infer-len "$VE_INFER_LEN" \
  --overlap "$VE_OVERLAP" \
  --num-inference-steps 40 \
  --guidance-scale 5.0 \
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
  --perf-dump-path "$VE_BASELINE_OUTPUT/case0008_nondistilled_perf.json" \
  2>&1 | tee "$VE_REPORT_DIR/case0008_nondistilled.log"
```

等待进程完全退出并确认 GPU 内存释放后，再运行 B 组。

## 5. 运行 B 组：DMD 蒸馏模型

`guidance_scale=1.0` 时不执行 classifier-free guidance。命令仍显式传
`--no-dynamic-cfg`，避免以后实现或默认值变化导致配方含义漂移。

```bash
cd "$VE_SGLANG_REPO"

CUDA_VISIBLE_DEVICES="$VE_GPU" \
SGLANG_CACHE_DIT_ENABLED=false \
"$VE_SGLANG_PYTHON" -m sglang.multimodal_gen.runtime.videoedit.cli repair \
  --model-path "$VE_MODEL_ROOT" \
  --transformer-path "$VE_DMD_TRANSFORMER" \
  --prompt "$VE_PROMPT" \
  --video-input-path "$VE_INPUT_VIDEO" \
  --mask-input-path "$VE_INPUT_MASK" \
  --reference-image-path "$VE_REFERENCE_IMAGE" \
  --output-path "$VE_DMD_OUTPUT" \
  --output-file-name case0008_dmd.mp4 \
  --num-frames "$VE_SOURCE_FRAMES" \
  --ref-frame-idx 0 \
  --bridge-overlap 5 \
  --infer-len "$VE_INFER_LEN" \
  --overlap "$VE_OVERLAP" \
  --num-inference-steps 4 \
  --guidance-scale 1.0 \
  --seed "$VE_SEED" \
  --dtype bf16 \
  --no-dynamic-cfg \
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
  --perf-dump-path "$VE_DMD_OUTPUT/case0008_dmd_perf.json" \
  2>&1 | tee "$VE_REPORT_DIR/case0008_dmd.log"
```

## 6. 输出契约检查

```bash
export VE_BASELINE_FULL="$VE_BASELINE_OUTPUT/case0008_nondistilled.mp4"
export VE_BASELINE_CROP="$VE_BASELINE_OUTPUT/case0008_nondistilled_crop_only.mp4"
export VE_BASELINE_META="$VE_BASELINE_OUTPUT/case0008_nondistilled.videoedit.json"
export VE_BASELINE_PERF="$VE_BASELINE_OUTPUT/case0008_nondistilled_perf.json"

export VE_DMD_FULL="$VE_DMD_OUTPUT/case0008_dmd.mp4"
export VE_DMD_CROP="$VE_DMD_OUTPUT/case0008_dmd_crop_only.mp4"
export VE_DMD_META="$VE_DMD_OUTPUT/case0008_dmd.videoedit.json"
export VE_DMD_PERF="$VE_DMD_OUTPUT/case0008_dmd_perf.json"

for VE_RESULT in \
  "$VE_BASELINE_FULL" "$VE_BASELINE_CROP" "$VE_BASELINE_META" "$VE_BASELINE_PERF" \
  "$VE_DMD_FULL" "$VE_DMD_CROP" "$VE_DMD_META" "$VE_DMD_PERF"; do
  test -s "$VE_RESULT" || { echo "missing output: $VE_RESULT"; exit 1; }
done

"$VE_SGLANG_PYTHON" - <<'PY' | tee "$VE_REPORT_DIR/video_contracts.json"
import json
import os
import subprocess

fields = "codec_name,pix_fmt,width,height,avg_frame_rate,nb_read_frames"

def probe(path):
    output = subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", f"stream={fields}", "-of", "json", path,
        ],
        text=True,
    )
    stream = json.loads(output)["streams"][0]
    return {field: stream.get(field) for field in fields.split(",")}

paths = {
    "nondistilled_full": os.environ["VE_BASELINE_FULL"],
    "nondistilled_crop": os.environ["VE_BASELINE_CROP"],
    "dmd_full": os.environ["VE_DMD_FULL"],
    "dmd_crop": os.environ["VE_DMD_CROP"],
}
report = {name: probe(path) for name, path in paths.items()}
assert report["nondistilled_full"] == report["dmd_full"], report
assert report["nondistilled_crop"] == report["dmd_crop"], report
assert report["dmd_full"]["nb_read_frames"] == "48", report["dmd_full"]
assert report["dmd_crop"]["nb_read_frames"] == "48", report["dmd_crop"]
assert report["dmd_crop"]["avg_frame_rate"] == "50/1", report["dmd_crop"]
print(json.dumps(report, indent=2))
PY

"$VE_SGLANG_PYTHON" - "$VE_BASELINE_META" "$VE_DMD_META" <<'PY'
import json
import sys

for path in sys.argv[1:]:
    with open(path, encoding="utf-8") as file:
        metadata = json.load(file)
    assert metadata["num_input_frames"] == 48, (path, metadata["num_input_frames"])
    assert metadata["num_output_frames"] == 48, (path, metadata["num_output_frames"])
    print(path, "input=48 output=48")
PY
```

硬门禁如下：

- 两组进程退出码都是 0，八个预期产物均非空。
- baseline full 与 DMD full 的 codec、pixel format、宽高、帧率和解码帧数完全一致。
- baseline crop 与 DMD crop 的上述属性完全一致，且均为 `50/1`、48 帧。crop 的具体
  宽高还必须与本次非蒸馏基线一致，不能依赖比较器自动 resize 掩盖几何回归。
- 两份 `.videoedit.json` 都记录 48 帧输入和 48 帧输出。

任一硬门禁失败时，先判定为功能或输出契约回归，不进入质量优劣结论。

## 7. 结果对比

### 7.1 像素差异报告

crop-only 是模型实际生成区域，作为主对比对象。下面使用理论范围作为宽松门限，使脚本
只负责逐帧配对并记录差异，不把非蒸馏输出误当成 DMD 的数值 golden：

```bash
cd "$VE_SGLANG_REPO"

"$VE_SGLANG_PYTHON" \
  python/sglang/multimodal_gen/runtime/videoedit/compare.py \
  --reference "$VE_BASELINE_CROP" \
  --candidate "$VE_DMD_CROP" \
  --report-json "$VE_REPORT_DIR/case0008_nondistilled_vs_dmd_crop.json" \
  --min-ssim -1.0 \
  --max-mse 65025 \
  --max-mae 255 \
  --allow-frame-count-delta 0 \
  --max-failed-frame-ratio 0.0
```

报告中的 SSIM 越高、MSE/MAE 越低，只表示 DMD 更接近这一次非蒸馏输出；它不等价于
更符合 prompt、更忠于 reference 或主观质量更高。full-frame 含有大量未编辑背景，会
稀释模型区域差异，因此不作为质量指标。

### 7.2 生成左右对照视频

```bash
ffmpeg -hide_banner -loglevel error -y \
  -i "$VE_BASELINE_CROP" -i "$VE_DMD_CROP" \
  -filter_complex '[0:v][1:v]hstack=inputs=2[v]' \
  -map '[v]' -an -c:v libx264 -crf 18 -pix_fmt yuv444p \
  "$VE_REPORT_DIR/case0008_crop_left_nondistilled_right_dmd.mp4"

ffmpeg -hide_banner -loglevel error -y \
  -i "$VE_BASELINE_FULL" -i "$VE_DMD_FULL" \
  -filter_complex '[0:v][1:v]hstack=inputs=2[v]' \
  -map '[v]' -an -c:v libx264 -crf 18 -pix_fmt yuv420p \
  "$VE_REPORT_DIR/case0008_full_left_nondistilled_right_dmd.mp4"
```

左侧始终是非蒸馏基线，右侧始终是 DMD。人工审核应至少逐项记录：

| 维度 | 检查内容 | 非蒸馏（1-5） | DMD（1-5） | 备注/问题帧 |
|---|---|---:|---:|---|
| 编辑正确性 | 是否完成 prompt 指定的修复/编辑 | 待填 | 待填 | 待填 |
| 参考一致性 | 目标外观是否忠于 reference image | 待填 | 待填 | 待填 |
| 时序稳定性 | 是否闪烁、漂移、形变或突然跳变 | 待填 | 待填 | 待填 |
| 边界质量 | mask 边缘是否有接缝、光晕、错色 | 待填 | 待填 | 待填 |
| 细节与自然度 | 纹理、结构、清晰度是否自然 | 待填 | 待填 | 待填 |

crop 用于评价模型生成质量；full 用于检查 paste-back 后的边缘以及整体观感。发现问题时
必须写出零基帧号，不能只写“看起来更差”。若结论用于发布门禁，建议隐藏左右标签并由
至少两名审核者独立打分，再揭盲汇总。

### 7.3 性能汇总

```bash
"$VE_SGLANG_PYTHON" - <<'PY' | tee "$VE_REPORT_DIR/performance_summary.json"
import json
import os

def load(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)

def denoising_ms(perf):
    values = [
        step["duration_ms"]
        for step in perf["steps"]
        if step["name"] == "VideoEditDenoisingStage"
    ]
    assert len(values) == 1, values
    return values[0]

def summarize(perf):
    memory = perf["memory_checkpoints"]["after_forward"]
    return {
        "total_ms": perf["total_duration_ms"],
        "denoising_ms": denoising_ms(perf),
        "peak_allocated_mb": memory["peak_allocated_mb"],
        "peak_reserved_mb": memory["peak_reserved_mb"],
    }

baseline = load(os.environ["VE_BASELINE_PERF"])
dmd = load(os.environ["VE_DMD_PERF"])
baseline_summary = summarize(baseline)
dmd_summary = summarize(dmd)
report = {
    "nondistilled": baseline_summary,
    "dmd": dmd_summary,
    "speedup": {
        "total": baseline_summary["total_ms"] / dmd_summary["total_ms"],
        "denoising": baseline_summary["denoising_ms"] / dmd_summary["denoising_ms"],
    },
}
print(json.dumps(report, indent=2))
PY
```

4 vs 40 步本身只代表 10 倍去噪步数差；非蒸馏组还启用了 CFG，而 DMD 组不执行 CFG。
因此不要事先声称端到端必然 10 倍加速。T5、CLIP、condition VAE encode、decode、CPU
offload、FFmpeg 和写盘都是固定或弱相关开销。正式性能结论应在同一 GPU 上交替运行
A/B，至少各 3 次，报告中位数，并确认没有其他进程占用 GPU。

## 8. 报告模板与判定

| 项目 | A：非蒸馏 | B：DMD | B 相对 A | 结论 |
|---|---:|---:|---:|---|
| 输出契约 | 待填 | 待填 | - | 待填 |
| 总时延（ms，中位数） | 待填 | 待填 | 待填 x | 待填 |
| 去噪时延（ms，中位数） | 待填 | 待填 | 待填 x | 待填 |
| Peak allocated（MB） | 待填 | 待填 | 待填 % | 待填 |
| Peak reserved（MB） | 待填 | 待填 | 待填 % | 待填 |
| crop SSIM mean/min | 基准 | 待填 | 仅描述差异 | 不作质量门禁 |
| crop MSE mean/max | 基准 | 待填 | 仅描述差异 | 不作质量门禁 |
| crop MAE mean/max | 基准 | 待填 | 仅描述差异 | 不作质量门禁 |
| 人工质量均分 | 待填 | 待填 | 待填 | 待填 |
| 严重伪影/失败帧 | 待填 | 待填 | 待填 | 待填 |

最低通过条件是所有输出契约硬门禁通过，DMD 没有导致编辑失败、严重时序闪烁或不可接受
的边界伪影，并且性能汇总显示实际收益。主观分数允许下降多少属于产品质量门槛，必须由
项目负责人预先确定，不能在看到结果后调整标准。

最终结论必须同时写清：

- 本次对比的确切 transformer 路径和 SHA256；
- SGLang commit 与工作区是否干净；
- 两组的采样参数和 scheduler 配置；
- 输出契约、主观质量和性能三类结果；
- `step_47500` 与 `step-55000-dmd-200` 不同 checkpoint 带来的解释限制。
