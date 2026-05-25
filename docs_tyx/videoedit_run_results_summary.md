# VideoEdit 本轮运行结果汇总

生成时间：`2026-05-13 07:31:57 UTC`

代码版本：

```text
a37752bc756bc5403b49acc82d1f11477415c85c
```

本文只汇总本轮实际跑过或明确复现过的 VideoEdit API/serve 结果。`outputs/` 目录里还有大量 `cli_*`、`serve_*`、`bench_*`、`compare_*` 文件，那些是复制过来的历史 benchmark 文件，不纳入本文主表。

## 1. 公共输入

```text
模型:
/home/tyx/workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model

Transformer:
/home/tyx/workspace/zhouhao6/video_diffusers/pretrain_models/VideoEdit-diffusers-model/transformer

输入视频:
/home/tyx/workspace/zhouhao6/video_diffusers/pexel_test_data_0410/videos/15108907_3840_2160_50fps_short.mp4

输入 mask:
/home/tyx/workspace/zhouhao6/video_diffusers/pexel_test_data_0410/masks/15108907_3840_2160_50fps_No_bbox_mask.mp4
```

输入视频和 mask 都是：

```text
156 frames, 1920x1080, 25 fps, 6.24s
```

Prompt：

```text
A close-up of an orange flower with a yellow center, remaining in focus against a blurred green grass background throughout the video.
```

通用生成参数：

```text
infer_len = 81
overlap = 0
num_inference_steps = 20
guidance_scale = 5.0
dynamic_cfg = true
dynamic_cfg_max_step = 15
seed = 42
dtype = bf16
enable_paste_back = true
```

## 2. 成功结果总表

| 序号 | task_id | 配置 | 输入帧 | 输出帧 | 输出路径 | perf total | 日志端到端 | warmup | 备注 |
|---:|---|---|---:|---:|---|---:|---:|---:|---|
| 1 | `sp1_offload` | 单卡 SP1 + CPU/layerwise offload | 81 | 80 | `outputs/15108907_3840_2160_50fps_api_sp1_offload.mp4` | 310.32s | 405.17s | 93.14s | Stage 0 基线 |
| 2 | `sp1_offload_100f` | 单卡 SP1 + CPU/layerwise offload | 100 | 99 | `outputs/15108907_3840_2160_50fps_api_sp1_offload_100f.mp4` | 1113.29s | 1285.62s | 169.06s | 100 帧无 overlap |
| 3 | `sp2_no_offload_fa_100f_test` | 双卡 SP2 + no-offload + FA | 100 | 99 | `outputs/15108907_3840_2160_50fps_api_sp2_no_offload_fa_100f_test.mp4` | 345.79s | 430.19s | 82.09s | 双卡明显加速 |
| 4 | `sp1_offload_156f_all_gpu0` | 单卡 GPU0 + CPU/layerwise offload | 156 | 156 | `output_tyx/15108907_3840_2160_50fps_api_sp1_offload_156f_all_gpu0.mp4` | 627.77s | 723.29s | 92.23s | 全帧成功结果 |

说明：

- `perf total` 来自对应 `videoedit_perf_*.json` 的 `total_duration_ms`。
- `日志端到端` 来自 serve 终端里 `Pixel data generated successfully in ... seconds` 或 `Completed batch processing ...` 的时间。
- 多窗口请求的 perf JSON 里 `steps` 通常只保留最后一个窗口的 stage 明细；端到端时间和日志里的每个窗口进度更适合看完整耗时。

## 3. 成功结果细节

### 3.1 `sp1_offload`：81 帧单卡 offload 基线

输出：

```text
outputs/15108907_3840_2160_50fps_api_sp1_offload.mp4
```

输出视频信息：

```text
80 frames
1920x1088
25 fps
3.20s
文件大小: 460K
```

metadata：

```json
{
  "num_input_frames": 81,
  "drop_reference_frame": true,
  "window_specs": [
    {
      "window_index": 0,
      "start_index": 0,
      "end_index": 81,
      "reflected_count": 0
    }
  ]
}
```

时间：

```text
warmup: 93.14s
perf total_duration_ms: 310315.13 ms = 310.32s
serve batch log: 405.17s
VideoEditDenoisingStage: 282.58s
VideoEditDecodingStage: 8.03s
VideoEditConditionEncodingStage: 9.78s
peak_allocated_mb: 14224.12
peak_reserved_mb: 17944.0
```

备注：

- 因为 `drop_reference_frame=true`，输入 81 帧，最终输出 80 帧。
- 这是最早跑通的 Stage 0 baseline。

### 3.2 `sp1_offload_100f`：100 帧单卡 offload

输出：

```text
outputs/15108907_3840_2160_50fps_api_sp1_offload_100f.mp4
```

输出视频信息：

```text
99 frames
1920x1088
25 fps
3.96s
文件大小: 575K
```

metadata：

```json
{
  "num_input_frames": 100,
  "drop_reference_frame": true,
  "window_specs": [
    {
      "window_index": 0,
      "start_index": 0,
      "end_index": 81,
      "reflected_count": 0
    },
    {
      "window_index": 1,
      "start_index": 81,
      "end_index": 100,
      "reflected_count": 62
    }
  ]
}
```

时间：

```text
warmup: 169.06s
perf total_duration_ms: 1113291.61 ms = 1113.29s
serve batch log: 1285.62s
window 0 denoising: 446.01s
window 0 decoding: 17.04s
window 1 denoising: 573.69s
window 1 decoding: 17.62s
denoising sum: 1019.70s
perf JSON last-window VideoEditDenoisingStage: 573.69s
perf JSON last-window VideoEditDecodingStage: 17.62s
peak_allocated_mb: 14224.61
peak_reserved_mb: 17944.0
```

备注：

- 因为 `drop_reference_frame=true`，输入 100 帧，最终输出 99 帧。
- 第二个窗口只有真实帧 `81..99`，所以反射补了 62 帧来凑够 81 帧窗口。

### 3.3 `sp2_no_offload_fa_100f_test`：100 帧双卡 SP2 no-offload

输出：

```text
outputs/15108907_3840_2160_50fps_api_sp2_no_offload_fa_100f_test.mp4
```

输出视频信息：

```text
99 frames
1920x1088
25 fps
3.96s
文件大小: 584K
```

配置：

```text
CUDA_VISIBLE_DEVICES=0,1
num_gpus = 2
sp_degree = 2
ulysses_degree = 2
ring_degree = 1
dit_cpu_offload = false
dit_layerwise_offload = false
attention_backend = fa
```

metadata：

```json
{
  "num_input_frames": 100,
  "drop_reference_frame": true,
  "window_specs": [
    {
      "window_index": 0,
      "start_index": 0,
      "end_index": 81,
      "reflected_count": 0
    },
    {
      "window_index": 1,
      "start_index": 81,
      "end_index": 100,
      "reflected_count": 62
    }
  ]
}
```

时间：

```text
warmup: 82.09s
perf total_duration_ms: 345786.83 ms = 345.79s
serve batch log: 430.19s
window 0 denoising: 150.94s
window 0 decoding: 7.02s
window 1 denoising: 151.40s
window 1 decoding: 6.89s
perf JSON last-window VideoEditDenoisingStage: 151.40s
perf JSON last-window VideoEditDecodingStage: 6.89s
peak_allocated_mb: 43953.04
peak_reserved_mb: 45066.0
```

备注：

- 双卡 no-offload 比单卡 offload 的 100 帧快很多。
- 但 GPU0 有 26GB 残留显存时，全帧 no-offload 后面会 OOM，见失败记录。

### 3.4 `sp1_offload_156f_all_gpu0`：单卡 GPU0 全帧成功结果

输出：

```text
output_tyx/15108907_3840_2160_50fps_api_sp1_offload_156f_all_gpu0.mp4
```

输出视频信息：

```text
156 frames
1920x1088
25 fps
6.24s
文件大小: 890K
```

配置：

```text
CUDA_VISIBLE_DEVICES=0
num_gpus = 1
sp_degree = 1
ulysses_degree = 1
ring_degree = 1
dit_cpu_offload = true
dit_layerwise_offload = true
text_encoder_cpu_offload = true
image_encoder_cpu_offload = true
vae_cpu_offload = true
drop_reference_frame = false
```

metadata：

```json
{
  "num_input_frames": 156,
  "drop_reference_frame": false,
  "window_specs": [
    {
      "window_index": 0,
      "start_index": 0,
      "end_index": 81,
      "reflected_count": 0
    },
    {
      "window_index": 1,
      "start_index": 81,
      "end_index": 156,
      "reflected_count": 6
    }
  ]
}
```

时间：

```text
warmup: 92.23s
perf total_duration_ms: 627771.20 ms = 627.77s
serve batch log: 723.29s
window 0 denoising: 284.15s
window 0 decoding: 11.63s
window 1 denoising: 284.05s
window 1 decoding: 8.11s
perf JSON last-window VideoEditDenoisingStage: 284.05s
perf JSON last-window VideoEditDecodingStage: 8.11s
peak_allocated_mb: 14223.45
peak_reserved_mb: 16140.0
```

备注：

- 这是当前最重要的全帧成功结果。
- 因为 `drop_reference_frame=false`，输入 156 帧，输出也保留 156 帧。
- 第二个窗口用了 6 帧时间维度反射补帧，只用于凑满模型输入窗口，不会额外写入输出。

## 4. 失败和中断记录

### 4.1 双卡 no-offload + FA，全帧 156：decode 阶段 OOM

task_id：

```text
sp2_no_offload_fa_156f_all_retry
```

配置：

```text
CUDA_VISIBLE_DEVICES=1,0
num_gpus = 2
sp_degree = 2
ulysses_degree = 2
ring_degree = 1
dit_cpu_offload = false
dit_layerwise_offload = false
attention_backend = fa
num_frames = 156
```

结果：

```text
失败，没有输出视频和 perf JSON。
```

报错阶段：

```text
VideoEditDecodingStage
```

核心错误：

```text
CUDA out of memory. Tried to allocate 4.05 GiB.
```

原因判断：

- GPU0 有残留显存约 26GB。
- 全帧 no-offload 在 decode 阶段需要额外显存。
- 即使用 `CUDA_VISIBLE_DEVICES=1,0` 让空卡做 rank0，另一张带残留的卡仍然参与 SP2，最终还是在 decode 阶段 OOM。

结论：

```text
在 GPU0 残留显存未清理前，不建议继续用双卡 no-offload 跑全帧 156。
```

### 4.2 双卡 SP2 + layerwise offload，全帧 156：transformer 加载阶段 OOM

配置：

```text
CUDA_VISIBLE_DEVICES=1,0
num_gpus = 2
sp_degree = 2
ulysses_degree = 2
ring_degree = 1
dit_cpu_offload = false
dit_layerwise_offload = true
dit_offload_prefetch_size = 1
text_encoder_cpu_offload = true
vae_cpu_offload = true
attention_backend = fa
num_frames = 156
```

结果：

```text
失败，serve 没有成功启动。
```

报错阶段：

```text
transformer load / FSDP load
```

核心错误：

```text
CUDA out of memory. Tried to allocate 300.00 MiB.
```

随后 fallback native loader 又失败：

```text
AttributeError: module diffusers has no attribute WanVideoEditTransformer3DModel
```

原因判断：

- 这个组合在当前分支/当前显存状态下不稳定。
- layerwise offload 对单卡保守方案有效，但双卡 SP2 + layerwise offload 加载阶段出现了额外显存和 loader fallback 问题。

结论：

```text
当前不推荐用双卡 SP2 + layerwise offload 跑全帧。单卡 GPU0 offload 已经能跑通全帧。
```

## 5. GPU 残留显存记录

多次查询看到 GPU0 存在驱动残留 compute app：

```text
3642863, [Not Found], 3200 MiB
969625, [Not Found], 23702 MiB
```

这两个 PID 在 `/proc` 里已经不存在，但 NVIDIA 驱动仍然记录其显存占用。它们正好对应之前 OOM 日志里的 PID。

影响：

- GPU0 长期少约 26GB 可用显存。
- no-offload、compile、双卡 SP2 全帧更容易 OOM。
- 单卡 CPU/layerwise offload 可以绕开这个问题，已经成功跑出全帧结果。

清理方式通常需要管理员操作：

```bash
sudo nvidia-smi --gpu-reset -i 0
```

如果 reset 不允许，就需要重启节点或让管理员处理。不要随便 reset 共享机器上的 GPU。

## 6. 关于反射补帧

当前模型窗口固定是 81 帧，所以：

```text
156 帧全帧输入 = 2 个窗口
窗口 0: 0..80，一共 81 帧，不补
窗口 1: 81..155，一共 75 帧，需要补 6 帧
```

补帧是时间维度反射，不是画面左右/上下翻转：

```text
156 -> 154
157 -> 153
158 -> 152
159 -> 151
160 -> 150
161 -> 149
```

这些反射帧只用于模型输入，不会写入最终视频。`sp1_offload_156f_all_gpu0` 最终输出仍然是 156 帧。

## 7. 参考文件

成功输出：

```text
outputs/15108907_3840_2160_50fps_api_sp1_offload.mp4
outputs/15108907_3840_2160_50fps_api_sp1_offload_100f.mp4
outputs/15108907_3840_2160_50fps_api_sp2_no_offload_fa_100f_test.mp4
output_tyx/15108907_3840_2160_50fps_api_sp1_offload_156f_all_gpu0.mp4
```

perf：

```text
outputs/videoedit_perf_api_sp1_offload.json
outputs/videoedit_perf_api_sp1_offload_100f.json
outputs/videoedit_perf_api_sp2_no_offload_fa_100f_test.json
output_tyx/videoedit_perf_api_sp1_offload_156f_all_gpu0.json
```

metadata：

```text
outputs/15108907_3840_2160_50fps_api_sp1_offload.videoedit.json
outputs/15108907_3840_2160_50fps_api_sp1_offload_100f.videoedit.json
outputs/15108907_3840_2160_50fps_api_sp2_no_offload_fa_100f_test.videoedit.json
output_tyx/15108907_3840_2160_50fps_api_sp1_offload_156f_all_gpu0.videoedit.json
```

## 8. 当前建议

如果目标是稳定得到全帧结果：

```text
继续用 sp1_offload_156f_all_gpu0 这套单卡 GPU0 + offload/layerwise offload 命令。
```

如果目标是提速：

```text
先清理 GPU0 的 26GB 残留显存。
清理后再重新测试双卡 SP2 no-offload + FA 全帧。
```

如果目标是做质量和速度对比：

```text
固定 num_frames=156、seed=42、prompt、mask、drop_reference_frame=false。
每次只改一个变量，例如只改 offload、只改 TeaCache、只改 torch.compile。
```
