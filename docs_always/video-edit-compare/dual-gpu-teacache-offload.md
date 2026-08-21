# VideoEdit 双卡 FlashAttention、TeaCache 与内存卸载命令行测试

本文档在 case0008、step_47500、48 个源视频帧和 40 个去噪步骤上测试以下组合：

- 物理 GPU 2、3；
- 2-way Ulysses 序列并行；
- `strict_videoedit_math=false`；
- FlashAttention backend；
- TeaCache 显式开启；
- DiT 逐层 CPU offload；
- 文本编码器、图像编码器和 VAE CPU offload；
- Cache-DiT、插帧和超分关闭。

测试沿用 [`env.md`](./env.md) 中已经生成的同步输入和原始算法 step_47500
输出。它不会重新运行原始算法，也不会覆盖关闭 TeaCache 的单双卡基线。

## 1. 参数口径

| 功能 | 命令行参数 | 本测试值 |
|---|---|---|
| 双卡并行 | `--num-gpus` / `--sp-degree` / `--ulysses-degree` | `2 / 2 / 2` |
| 张量并行 | `--tp-size` | `1` |
| Attention backend | `--attention-backend` | `fa`；`torch_sdpa` 作为对照组 |
| TeaCache | `--enable-teacache` | 开启 |
| TeaCache 阈值 | `--teacache-thresh` | `0.3` |
| TeaCache 跳过区间 | `--teacache-start-skipping` / `--teacache-end-skipping` | `5 / 1.0` |
| DiT 卸载 | `--dit-layerwise-offload` | 开启，预取量为 `0` |
| 其他模块卸载 | `--text-encoder-cpu-offload`、`--image-encoder-cpu-offload`、`--vae-cpu-offload` | 全部开启 |

`--no-dit-cpu-offload` 只关闭整模型 DiT CPU offload；本命令同时显式开启
`--dit-layerwise-offload`，所以 DiT 仍然按层在 CPU/GPU 之间卸载。这两种 DiT
offload 模式不能同时开启。

TeaCache 会跳过部分 DiT block 计算，属于性能模式而不是严格数值 golden。环境变量
`SGLANG_CACHE_DIT_ENABLED=false` 用于继续关闭 Cache-DiT，确保本测试只引入
TeaCache 一个缓存变量。

`fa` 是 SGLang-native pipeline 的 FlashAttention CLI 名称。当前 A100 环境使用
`sgl_kernel.flash_attn` 实现；运行日志必须出现 `Using FlashAttention`，若打印
`Using Torch SDPA` 则说明发生了回退，不能记为 FlashAttention 测试。

## 2. 运行命令（FlashAttention）

```bash
export VE_CASE_DIR=/mnt/shanhai-ai/liuh/VideoEdit-diffusers/datas/edit_val_cases/0008
export VE_PROMPT="$(tr -d '\r\n' < "$VE_CASE_DIR/prompt.txt")"
export VE_SGLANG_REPO=/mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang
export VE_SGLANG_PYTHON=/home/root/uv-envs/sglang-llm-diffusion/bin/python
export VE_SGLANG_MODEL=/mnt/shanhai-ai/shanhai-workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model
export VE_SGLANG_TRANSFORMER="$VE_SGLANG_MODEL/transformer"
export VE_INPUT_VIDEO="$VE_SGLANG_REPO/outputs/video-edit-inputs-v2/case0008_video_sync.mp4"
export VE_INPUT_MASK="$VE_SGLANG_REPO/outputs/video-edit-inputs-v2/case0008_mask_sync.mp4"
export VE_OUTPUT="$VE_SGLANG_REPO/outputs/sglang_outputs-step47500-strict-false-dual-gpu-teacache-offload-fa"

mkdir -p "$VE_OUTPUT/tmp" "$VE_OUTPUT/cache"
cd "$VE_SGLANG_REPO"

CUDA_VISIBLE_DEVICES=2,3 \
SGLANG_CACHE_DIT_ENABLED=false \
PYTHONUNBUFFERED=1 \
TMPDIR="$VE_OUTPUT/tmp" \
XDG_CACHE_HOME="$VE_OUTPUT/cache" \
TORCH_HOME="$VE_OUTPUT/cache/torch" \
"$VE_SGLANG_PYTHON" -m sglang.multimodal_gen.runtime.videoedit.cli repair \
  --model-path "$VE_SGLANG_MODEL" \
  --transformer-path "$VE_SGLANG_TRANSFORMER" \
  --prompt "$VE_PROMPT" \
  --video-input-path "$VE_INPUT_VIDEO" \
  --mask-input-path "$VE_INPUT_MASK" \
  --reference-image-path "$VE_CASE_DIR/reference.png" \
  --output-path "$VE_OUTPUT" \
  --output-file-name case0008_sglang_dual_gpu_teacache_fa.mp4 \
  --num-frames 48 \
  --ref-frame-idx 0 \
  --bridge-overlap 5 \
  --infer-len 49 \
  --overlap 5 \
  --num-inference-steps 40 \
  --guidance-scale 5.0 \
  --seed 42 \
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
  --enable-teacache \
  --teacache-thresh 0.3 \
  --teacache-start-skipping 5 \
  --teacache-end-skipping 1.0 \
  --no-enable-frame-interpolation \
  --no-enable-upscaling \
  --num-gpus 2 \
  --tp-size 1 \
  --sp-degree 2 \
  --ulysses-degree 2 \
  --ring-degree 1 \
  --attention-backend fa \
  --perf-dump-path "$VE_OUTPUT/case0008_sglang_dual_gpu_teacache_fa_perf.json"
```

复现 `torch_sdpa` 对照组时，只替换以下三项，其余参数必须保持不变：

```bash
export VE_OUTPUT="$VE_SGLANG_REPO/outputs/sglang_outputs-step47500-strict-false-dual-gpu-teacache-offload"
# --output-file-name case0008_sglang_dual_gpu_teacache.mp4
# --attention-backend torch_sdpa
# --perf-dump-path "$VE_OUTPUT/case0008_sglang_dual_gpu_teacache_perf.json"
```

## 3. 输出检查与比较

主输出应为 48 帧；本轮实际原始算法参考 crop 是 `1458 x 600 / yuv420p`，
因此 candidate 也必须保持这一媒体契约。

```bash
export VE_REFERENCE_CROP="$VE_SGLANG_REPO/outputs/video-edit_outputs-step47500/case0008_reference_crop_only.mp4"
export VE_REFERENCE_FULL="$VE_SGLANG_REPO/outputs/video-edit_outputs-step47500/case0008_reference.mp4"
export VE_SDPA_TEACACHE_CROP="$VE_SGLANG_REPO/outputs/sglang_outputs-step47500-strict-false-dual-gpu-teacache-offload/case0008_sglang_dual_gpu_teacache_crop_only.mp4"
export VE_CANDIDATE_CROP="$VE_OUTPUT/case0008_sglang_dual_gpu_teacache_fa_crop_only.mp4"
export VE_CANDIDATE_FULL="$VE_OUTPUT/case0008_sglang_dual_gpu_teacache_fa.mp4"

for VE_RESULT in "$VE_CANDIDATE_CROP" "$VE_CANDIDATE_FULL"; do
  ffprobe -v error -select_streams v:0 \
    -show_entries stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_frames \
    -of default=noprint_wrappers=1 "$VE_RESULT"
done

"$VE_SGLANG_PYTHON" \
  python/sglang/multimodal_gen/runtime/videoedit/compare.py \
  --reference "$VE_REFERENCE_CROP" \
  --candidate "$VE_CANDIDATE_CROP" \
  --report-json "$VE_SGLANG_REPO/outputs/case0008_step47500_compare_report_crop.teacache_offload_dual_gpu_fa.json" \
  --min-ssim 0.97 \
  --max-mse 25.0 \
  --max-mae 2.5 \
  --allow-frame-count-delta 0 \
  --max-failed-frame-ratio 0.0

"$VE_SGLANG_PYTHON" \
  python/sglang/multimodal_gen/runtime/videoedit/compare.py \
  --reference "$VE_SDPA_TEACACHE_CROP" \
  --candidate "$VE_CANDIDATE_CROP" \
  --report-json "$VE_SGLANG_REPO/outputs/case0008_step47500_compare_report_crop.teacache_fa_vs_sdpa_dual_gpu.json" \
  --min-ssim 0.97 \
  --max-mse 25.0 \
  --max-mae 2.5 \
  --allow-frame-count-delta 0 \
  --max-failed-frame-ratio 0.0

"$VE_SGLANG_PYTHON" \
  python/sglang/multimodal_gen/runtime/videoedit/compare.py \
  --reference "$VE_REFERENCE_FULL" \
  --candidate "$VE_CANDIDATE_FULL" \
  --report-json "$VE_SGLANG_REPO/outputs/case0008_step47500_compare_report_full.teacache_offload_dual_gpu_fa.json" \
  --min-ssim 0.98 \
  --max-mse 25.0 \
  --max-mae 2.5 \
  --allow-frame-count-delta 0 \
  --max-failed-frame-ratio 0.0
```

第一份报告衡量 FlashAttention + TeaCache 输出与原始算法的差异；第二份报告隔离
FlashAttention 相对同配置 `torch_sdpa` 所引入的差异；第三份报告辅助检查 full-frame paste-back。
由于 TeaCache 有意改变计算路径，严格门限失败时应保留报告并结合逐帧指标判断，
不能把它当作关闭缓存的 golden 回归。

## 4. 性能口径

使用 perf JSON 中的 `total_duration_ms` 和 `VideoEditDenoisingStage` 与以下
`torch_sdpa + TeaCache` 对照组比较：

```text
outputs/sglang_outputs-step47500-strict-false-dual-gpu-teacache-offload/case0008_sglang_dual_gpu_teacache_perf.json
```

模型加载发生在 pipeline 计时之前，因此 `total_duration_ms` 是请求 pipeline 耗时，
不是包含模型加载的 CLI 端到端墙钟时间。TeaCache 的主要收益应体现在
`VideoEditDenoisingStage`。条件编码、解码和模型加载不使用 DiT attention backend，
这些阶段的波动不得归因于 FlashAttention。

## 5. 2026-08-21 `torch_sdpa` 对照组实测结果

对照组命令成功返回 `0`，crop 和 full 都是 H.264、`yuv420p`、50 FPS、48 帧；crop 为
`1458 x 600`，full 为 `1920 x 1080`。

### 5.1 正确性

| 对比 | SSIM mean/min | MSE mean/max | MAE mean/max | 结果 |
|---|---:|---:|---:|---|
| 原始算法 crop vs TeaCache crop | `0.992841 / 0.991206` | `1.096368 / 1.278075` | `0.525651 / 0.587833` | 48/48 帧通过 |
| 无缓存双卡 crop vs TeaCache crop | `0.993243 / 0.992001` | `1.071441 / 1.250574` | `0.514350 / 0.571684` | 48/48 帧通过 |
| 原始算法 full vs TeaCache full | `0.988952 / 0.988037` | `2.904915 / 3.228683` | `0.799547 / 0.844109` | 48/48 帧通过 |

报告：

- `outputs/case0008_step47500_compare_report_crop.teacache_offload_dual_gpu.json`
- `outputs/case0008_step47500_compare_report_crop.teacache_vs_no_teacache_dual_gpu.json`
- `outputs/case0008_step47500_compare_report_full.teacache_offload_dual_gpu.json`

### 5.2 性能与显存

| 指标 | 无缓存双卡 | TeaCache + offload | 相对无缓存双卡 |
|---|---:|---:|---:|
| pipeline 总耗时 | `493.139 s` | `255.907 s` | `1.927x` 加速 |
| 去噪耗时 | `394.118 s` | `141.570 s` | `2.784x` 加速 |
| 解码耗时 | `45.938 s` | `48.674 s` | `0.944x`，慢 `5.96%` |
| peak allocated | `12384.56 MiB` | `12384.56 MiB` | 不变 |
| after-forward reserved | `15486 MiB` | `16308 MiB` | 增加 `822 MiB` |

显存字段来自 perf JSON 的 rank 0 快照，不是两张卡的求和。TeaCache 保留缓存使请求结束时
allocated/reserved 略有增加，但没有提高本次记录到的 peak allocated。相对无缓存单卡，
TeaCache 双卡的 pipeline 总耗时加速 `3.288x`，去噪阶段加速 `5.264x`。

## 6. 2026-08-21 FlashAttention 实测结果

启动日志记录 `attention_backend="fa"` 并多次打印 `Using FlashAttention`，没有回退到
`torch_sdpa`。命令返回 `0`，crop/full 媒体属性与第 5 节对照组一致。

### 6.1 正确性

| 对比 | SSIM mean/min | MSE mean/max | MAE mean/max | 结果 |
|---|---:|---:|---:|---|
| 原始算法 crop vs FA TeaCache crop | `0.992859 / 0.991328` | `1.097713 / 1.273113` | `0.525276 / 0.583708` | 48/48 帧通过 |
| SDPA TeaCache crop vs FA TeaCache crop | `0.993555 / 0.992324` | `0.991034 / 1.178311` | `0.492002 / 0.552047` | 48/48 帧通过 |
| 原始算法 full vs FA TeaCache full | `0.988976 / 0.987944` | `2.904109 / 3.246252` | `0.799583 / 0.848809` | 48/48 帧通过 |

报告：

- `outputs/case0008_step47500_compare_report_crop.teacache_offload_dual_gpu_fa.json`
- `outputs/case0008_step47500_compare_report_crop.teacache_fa_vs_sdpa_dual_gpu.json`
- `outputs/case0008_step47500_compare_report_full.teacache_offload_dual_gpu_fa.json`

### 6.2 FlashAttention 与 `torch_sdpa` 性能对照

| 阶段 | SDPA TeaCache | FA TeaCache | FA 相对 SDPA |
|---|---:|---:|---:|
| pipeline 总耗时 | `255.907 s` | `316.459 s` | `0.809x`，慢 `23.66%` |
| 文本编码 | `13.085 s` | `6.026 s` | `2.171x` |
| 图像编码 | `4.090 s` | `4.930 s` | `0.830x` |
| 条件编码 | `35.861 s` | `56.020 s` | `0.640x` |
| 去噪 | `141.570 s` | `140.922 s` | `1.0046x`，快 `0.46%` |
| 解码 | `48.674 s` | `84.179 s` | `0.578x` |
| peak allocated | `12384.56 MiB` | `12384.56 MiB` | 不变 |
| after-forward reserved | `16308 MiB` | `16308 MiB` | 不变 |

FlashAttention 只对去噪阶段有直接影响；本次去噪快 `0.46%`，不足以超出单次运行波动。
总 pipeline 变慢来自条件编码和解码，而不是 attention backend。TeaCache 还可能因 backend
的微小数值差异改变缓存刷新决策，因此要确认小于 1% 的差异，需要交错重复多轮并报告
中位数，不能根据本次单样本宣称 FlashAttention 有稳定加速。

按已有原始算法 209 帧、5 窗口、4533 秒的结果归一化为 `906.6 秒/窗口`，本轮 FA
pipeline 的 `316.459 秒` 约为原始算法的 `2.865x` 加速、耗时下降 `65.09%`。该数值仍是
跨运行的按窗口估算，不是同轮受控 benchmark。

## 7. 2026-08-21 `stream` 双窗口实测

本轮保持第 2 节的双卡、FlashAttention、TeaCache、逐层 DiT offload、其他模块 CPU
offload、`stream` 解码模式和随机种子不变，只把输入长度扩展为恰好两个完整窗口。
`infer_len=49`、`overlap=5` 时 stride 为 44；92 个源视频帧会产生起点为
`[0, 44]` 的两个窗口。

### 7.1 命令行差异

复用第 2 节的完整命令，只替换下列环境变量和参数：

```bash
export VE_INPUT_VIDEO="$VE_SGLANG_REPO/outputs/video-edit-full-inputs-step47500/case0008_video_full_209.mp4"
export VE_INPUT_MASK="$VE_SGLANG_REPO/outputs/video-edit-full-inputs-step47500/case0008_mask_full_209.mp4"
export VE_OUTPUT="$VE_SGLANG_REPO/outputs/sglang_outputs-step47500-strict-false-dual-gpu-teacache-offload-fa-stream-2win"

# 替换第 2 节命令中的对应参数：
# --output-file-name case0008_sglang_dual_gpu_teacache_fa_stream_2win.mp4
# --num-frames 92
# --perf-dump-path "$VE_OUTPUT/case0008_sglang_dual_gpu_teacache_fa_stream_2win_perf.json"
```

其余参数必须原样保留，特别是：

```text
--enable-teacache --teacache-thresh 0.3
--teacache-start-skipping 5 --teacache-end-skipping 1.0
--dit-layerwise-offload --dit-offload-prefetch-size 0
--text-encoder-cpu-offload --image-encoder-cpu-offload --vae-cpu-offload
--num-gpus 2 --tp-size 1 --sp-degree 2 --ulysses-degree 2 --ring-degree 1
--attention-backend fa
```

### 7.2 窗口和媒体契约

命令返回 `0`，日志出现三次 `Using FlashAttention`，没有 SDPA 回退、OOM 或异常。
metadata 记录了两个 forward 窗口：

| 窗口 | 起点 | 有效长度 | 传播 overlap | 提交范围 |
|---|---:|---:|---:|---|
| 0 | `0` | `49` | `0` | 全局帧 `0..47` |
| 1 | `44` | `49` | `5` | 全局帧 `48..91` |

提交范围无缺口、无重复，最终 crop/full 均为 H.264、`yuv420p`、50 FPS、92 帧；crop 为
`1458 x 600`，full 为 `1920 x 1080`。这说明 stream 模式完成了两窗即时解码、overlap
传播和最终拼接。

### 7.3 正确性和接缝

原始算法长视频参考共有 208 帧，比较工具直接读取其前 92 帧与本轮 92 帧输出比较。
不要先用 `-frames:v 92 -c copy` 截断带 B-frame 的参考视频；这种包级截断会污染末尾显示帧，
产生伪失败。

| 对比 | SSIM mean/min | MSE mean/max | MAE mean/max | 结果 |
|---|---:|---:|---:|---|
| 原始算法长参考前 92 帧 crop vs stream crop | `0.992966 / 0.990294` | `1.027427 / 1.295697` | `0.514014 / 0.609093` | 92/92 帧通过 |
| 原始算法长参考前 92 帧 full vs stream full | `0.991170 / 0.990371` | `2.098363 / 2.342373` | `0.692258 / 0.742192` | 92/92 帧通过 |
| 既有 FA eager crop vs stream 第一窗 crop | `0.992244 / 0.991168` | `1.199101 / 1.562764` | `0.557089 / 0.660435` | 48/48 帧通过 |

第三行同时包含公共输入编码方式的差异，因此只作为第一窗稳定性检查，不是严格隔离
`decode_mode` 的受控 A/B。

第二窗开始提交的接缝是 `47 -> 48`。接缝附近第 43--52 帧对原始参考的最低 SSIM 为
`0.994701`，最高 MSE/MAE 为 `0.798455 / 0.433768`，10/10 帧通过。候选视频的接缝
时序 MAE 相对附近转场中位数为 `2.243x`，原始参考同位置为 `2.190x`，仅高 `2.41%`；
逐帧接触图也未见跳变、重影或字幕位置漂移。因此本次两窗口 stream 结果正常。

报告：

- `outputs/case0008_step47500_compare_report_crop.teacache_offload_dual_gpu_fa_stream_2win_direct.json`
- `outputs/case0008_step47500_compare_report_full.teacache_offload_dual_gpu_fa_stream_2win_direct.json`
- `outputs/case0008_step47500_compare_report_crop.teacache_fa_stream_firstwin_vs_eager.json`

### 7.4 两窗口耗时与显存

当前 perf JSON 的逐阶段字段只保留最后一个窗口，因此逐窗口耗时从同一次运行日志提取；
`total_duration_ms` 仍覆盖完整请求。

| 阶段 | 窗口 0 | 窗口 1 |
|---|---:|---:|
| 文本编码 | `6.523 s` | `5.765 s` |
| 图像编码 | `2.275 s` | `1.429 s` |
| 条件编码 | `30.967 s` | `30.171 s` |
| 去噪 | `138.822 s` | `130.941 s` |
| stream 解码 | `48.979 s` | `49.467 s` |
| 可归属阶段合计 | `227.595 s` | `217.792 s` |

完整两窗口 pipeline 总耗时为 `516.716 s`；两个窗口的可归属阶段合计为 `445.387 s`，
其余约 `71.329 s` 是请求预处理、窗口物化和最终视频编码等开销。rank 0 的 peak allocated
为 `13107.15 MiB`，after-forward reserved 为 `16752 MiB`。与第 6 节 48 帧单窗口 eager
运行相比，分别增加 `722.59 MiB`（`5.83%`）和 `444 MiB`（`2.72%`）；由于窗口数也
同时改变，不能把这组显存差异单独归因于 `stream`。

按原始算法 209 帧、5 窗口、4533 秒归一化，本轮平均 `258.358 秒/窗口`，估算为
`3.509x` 加速、耗时下降 `71.50%`。这仍是跨运行归一化估算，不是同轮两窗口原始算法
受控 benchmark。
