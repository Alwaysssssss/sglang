# VividVR Phase D 验收完成与 Phase E Benchmark 入口交接

更新时间：`2026-06-05 UTC`

## 1. 这份交接文档覆盖什么

本文档总结本轮对话中完成的几件关键工作：

- 重新确认 `Phase D` 的真实目标是“对齐原版 `Vivid-VR` 长视频主语义”，不是只做一个能跑的近似实现。
- 按 `docs_xzh/add_strategy/10_grouped_stage_acceptance.md` 和相关文档，对当前 `Phase D` 进行正式验收。
- 在 `step=6` 通过后，继续跑完整 `step=50` 对比验收，并确认通过。
- 为后续 `Phase E` 性能迭代，把日常 benchmark 默认档位收口到 `20 step`，并统一 `sglang` 侧 benchmark 入口。
- 增强原版 `/home/zhiheng/Vivid-VR` 的 benchmark 可观测性：补总进度条、补 runtime report、补可对比的时间字段。
- 实际跑通一轮 `GPU 0` 上的原版 `20 step` benchmark 和对应的 `sglang` benchmark，并通过验收。


## 2. 当前结论

### 2.1 Phase D 已通过正式验收

当前 `Phase D` 不再停留在“代码和 benchmark 工具链已具备，但公平验收未通过”的状态。本轮验证结果表明：

- `Phase D` 长视频主语义已经和原版 `Vivid-VR` 对齐到可验收状态。
- 不只是轻量 `step=6` 能过，完整 `step=50` 也能过。
- 后续做 `Phase E` 时，可以把 `Phase D` 视为“语义基线已收口”的前置条件。

### 2.2 当前默认 benchmark 口径

当前长视频 benchmark 的正式口径是：

- 输入视频固定为：
  - `/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.mp4`
- 长视频公平 benchmark 使用原版 caption sidecar 回放，不走 `sglang` live caption。
- 日常性能迭代默认使用 `20 step`。
- 最终回归和阶段验收保留 `50 step`。
- 所有这类推理都固定使用 `GPU 0`。


## 3. Phase D 语义到底对齐了什么

本轮确认的重点不是单 clip，而是原版 `Vivid-VR` 长视频路径的核心语义已经对齐。主要包括：

- `clip split` 语义：
  - 130 帧输入被切成 2 个 temporal clip，而不是旧的 `x3 duplicate` 基准。
- `timestep` 级 orchestration：
  - 多 clip 在每个 denoise step 上同步推进，而不是每个 clip 各自完整跑完再粗暴拼接。
- overlap 区域的 `latent merge / ownership`：
  - 跨 clip merge 的时机和重叠区域归属与原版一致。
- `trim / stitch` 语义：
  - overlap 和 padding 的裁剪、最终长视频 stitch 行为与原版对齐。
- caption 公平性：
  - `sglang` 长视频 benchmark 使用原版每个 temporal clip 的 raw caption sidecar 回放，避免把 caption backend 差异混进对比里。

当前 `20 step` 验收 JSON 的 debug 字段已经能直接反映出这一点：

- `execution_mode = temporal_windowed`
- `num_clips = 2`
- `clip_specs`：
  - clip0: `start=0`, `end=121`, `trim_back=30`
  - clip1: `start=60`, `end=130`, `pad=3`, `trim_front=31`
- `clip_latent_lengths = [32, 20]`
- `output_num_frames = 130`

对应产物：

- `sglang` `20 step` 指标：
  - `/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_d_130f_20step_metrics_seed42_20260605T083022Z.json`


## 4. 本轮在 sglang 侧完成的工作

### 4.1 增加通用推理脚本

新增通用脚本：

- `/home/zhiheng/sglang/python/sglang/multimodal_gen/tools/run_vividvr_inference.py`

这份脚本的定位是：

- 作为 `Phase D / Phase E` 共用的 `sglang` 原生推理入口。
- 不再依赖只为 `Phase D` 写的一次性长视频 helper 命令。
- 后续 benchmark 时，直接在启动命令里修改参数即可，例如：
  - `--input-video`
  - `--caption-file`
  - `--reference-video`
  - `--num-inference-steps`
  - `--artifact-prefix`

当前这份脚本支持两种主要模式：

- `prompt_file`：
  - 对应 `Phase C` 单 clip / 固定 prompt 基线。
- `caption_file`：
  - 对应 `Phase D` / `Phase E` 长视频公平 benchmark。

脚本默认仍保留 `50 step` 作为 canonical 默认值，符合“底层默认不轻易改成开发态参数”的原则；日常 benchmark 则在命令层显式传 `20 step`。

### 4.2 更新 benchmark 文档

文档已更新：

- `/home/zhiheng/sglang/docs_xzh/run_vivid_benchmark.md`

本轮主要改动：

- 把 `Phase D` 输入视频改为固定 `130f` 路径。
- 把日常 benchmark 默认 step 改为 `20`。
- 保留说明：`50 step` 只用于最终回归和阶段验收。
- 把 `sglang` `Phase D` 命令改成简洁的短命令，不再使用过长的内联 Python。
- 改为统一引用通用脚本 `run_vividvr_inference.py`。


## 5. 本轮在原版 Vivid-VR 侧完成的工作

### 5.1 原版之前缺少什么

在本轮修改前，原版 `Vivid-VR` 的直接推理入口并没有稳定输出和 `sglang` 对齐的两类关键时间：

- `total_runtime_seconds`
- `model_inference_runtime_seconds`

此外，长视频总进度条虽然存在，但会被内层 `tqdm` 刷新掉，不方便在终端里稳定观察整体进度。

### 5.2 已补强的文件

本轮补强的是原版外部仓库中的：

- `/home/zhiheng/Vivid-VR/VRDiT/inference.py`

增强内容包括：

- 长视频总进度条改成单独常驻一行，覆盖完整重路径：
  - `Long video cache`
  - `Long video denoise/merge`
  - `Long video decode`
  - `Long video done`
- 每次生成视频后，在结果视频同目录写 runtime report。
- report 中记录关键参数和时间字段，便于直接和 `sglang` 对比。

当前原版 `20 step` 直接 benchmark 的新 report 路径为：

- `/home/zhiheng/Vivid-VR/result/720p_long_up1_result_vivid_ori_20step/videos/test_video_long_960x720_130f_report.json`

其中已经包含：

- `total_runtime_seconds`
- `model_inference_runtime_seconds`
- `model_loading_runtime_seconds`
- `video_processing_runtime_seconds`
- `save_runtime_seconds`
- 以及 input / output / seed / step / guidance / frame count / size 等关键参数

需要注意：

- 原版现有 wrapper 脚本本来就会在 `run_reports/` 下写一层总 runtime report。
- 本轮新增的是 `VRDiT/inference.py` 直跑路径下、与输出视频同目录的 per-video report。
- 两层 report 并不冲突，但后续如果想继续收口原版 benchmark 体验，可以考虑把 wrapper 与 direct inference 的 report 字段进一步统一。


## 6. Phase D 验收结果

### 6.1 step=6 验收

产物：

- `/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_d_130f_6step_metrics_seed42_20260605T021208Z.json`

结果：

- `pass_compare = true`
- `ssim_mean = 0.948550`
- `ssim_min = 0.929455`
- `mse_mean = 41.263714`
- `mse_max = 58.775631`
- `mae_mean = 4.882232`
- `mae_max = 5.548004`
- `failed_frame_ratio = 0.0`
- `total_runtime_seconds = 459.164331`
- `model_inference_runtime_seconds = 433.700859`

### 6.2 step=50 完整回归验收

产物：

- `sglang` 指标：
  - `/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_d_130f_50step_metrics_seed42_20260605T030814Z.json`
- 原版 reference 日志：
  - `/home/zhiheng/sglang/Vivid_Acceptance/logs/vividvr_ori_phase_d_130f_50step_20260605T022243Z.log`
- 原版 wrapper report：
  - `/home/zhiheng/Vivid-VR/result/720p_long_up1_result_vivid_ori_50step/run_reports/test_video_long_960x720_130f_20260605T022247Z.json`

结果：

- `pass_compare = true`
- `ssim_mean = 0.982343`
- `ssim_min = 0.979458`
- `mse_mean = 12.196430`
- `mse_max = 14.904971`
- `mae_mean = 2.666520`
- `mae_max = 2.834451`
- `failed_frame_ratio = 0.0`
- `sglang total_runtime_seconds = 2496.806432`
- `sglang model_inference_runtime_seconds = 2461.438457`
- `original total_runtime_seconds = 2456.141355`

这说明 `Phase D` 不只是轻量档位通过，完整 `50 step` 下也已和原版对齐。

### 6.3 step=20 日常 benchmark 验收

这是本轮为后续 `Phase E` 日常性能迭代补跑的一轮中档 benchmark。

原版产物：

- 原版 reference 视频：
  - `/home/zhiheng/Vivid-VR/result/720p_long_up1_result_vivid_ori_20step/videos/test_video_long_960x720_130f.mp4`
- 原版 runtime report：
  - `/home/zhiheng/Vivid-VR/result/720p_long_up1_result_vivid_ori_20step/videos/test_video_long_960x720_130f_report.json`
- 原版日志：
  - `/home/zhiheng/sglang/Vivid_Acceptance/logs/vividvr_ori_phase_d_20step_20260605T081011Z.log`

`sglang` 产物：

- 指标：
  - `/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_d_130f_20step_metrics_seed42_20260605T083022Z.json`
- 候选视频：
  - `/home/zhiheng/sglang/Vivid_Acceptance/result_videos/phase_d_130f_20step_seed42_20260605T083022Z.mp4`
- 日志：
  - `/home/zhiheng/sglang/Vivid_Acceptance/logs/phase_d_130f_20step_20260605T083011Z.log`

结果：

- `pass_compare = true`
- `ssim_mean = 0.984745`
- `ssim_min = 0.979066`
- `mse_mean = 12.207597`
- `mse_max = 20.391026`
- `mae_mean = 2.663361`
- `mae_max = 2.909005`
- `failed_frame_ratio = 0.0`

时间：

- original:
  - `total_runtime_seconds = 1139.839039`
  - `model_inference_runtime_seconds = 1047.001905`
- sglang:
  - `total_runtime_seconds = 1101.002094`
  - `model_inference_runtime_seconds = 1075.420882`


## 7. 速度观察

目前有两组比较值得保留的时间数据：

### 7.1 step=50

- original total：`2456.141355s`
- sglang total：`2496.806432s`
- sglang 比 original 慢约 `40.665s`，约 `1.7%`

这个量级更像实现常数开销或环境噪声，不像明显的系统级资源争用。

### 7.2 step=20

- original total：`1139.839039s`
- original model inference：`1047.001905s`
- sglang total：`1101.002094s`
- sglang model inference：`1075.420882s`

对于后续 `Phase E`，`20 step` 是更合适的日常 profile 档位：

- 比 `50 step` 轻很多
- 比 `6 step` 更接近真实长视频负载


## 8. Caption sidecar 现状

当前 `130f` 基准实际使用的 caption sidecar 是：

- `20 step`：
  - `/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.txt`
- `50 step`：
  - `/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f_50step.txt`

本轮 `20 step` 原版 benchmark 跑完后，日志中的两段原始 caption 与现有 `test_video_long_960x720_130f.txt` 做过比对，去掉原版正向 prompt suffix 后完全一致，因此直接复用了这份 sidecar，没有重新改写内容。


## 9. 当前推荐命令入口

后续以文档为准：

- `/home/zhiheng/sglang/docs_xzh/run_vivid_benchmark.md`

当前推荐策略：

- `Phase C`：
  - 保持原有单 clip 验收入口。
- `Phase D / Phase E`：
  - 统一使用 `/home/zhiheng/sglang/python/sglang/multimodal_gen/tools/run_vividvr_inference.py`
- 日常 benchmark：
  - `20 step`
- 最终回归：
  - `50 step`

所有重推理都要求：

- 放在 `tmux` 中运行
- 固定 `CUDA_VISIBLE_DEVICES=0`


## 10. 当前仓库状态

### 10.1 sglang 仓库

当前快照：

- repo: `/home/zhiheng/sglang`
- branch: `sglang_Vivid`
- HEAD: `83064b778be3630942c5121b630549dd62ab936a`

在写这份交接文档前，`sglang` 工作区是干净的；因此当前这份 handover 文档本身会成为本轮新增改动的主体。

### 10.2 原版 Vivid-VR 仓库

当前快照：

- repo: `/home/zhiheng/Vivid-VR`
- branch: `main`
- HEAD: `9ef0d03dc8e77ff4c427c57256473aba9ca6223c`

需要注意：

- 原版仓库本来就不是干净工作区，存在较多既有改动和未跟踪文件。
- 本轮与 benchmark 直接相关、且需要特别记住的是：
  - `VRDiT/inference.py` 已被继续修改
  - 新增或保留了 `130f` 基准 caption sidecar
  - `20 step` 和 `50 step` 的原版 runtime/report 产物已经落盘


## 11. 后续 Phase E 建议

后续进入 `Phase E` 时，建议遵守以下节奏：

1. 日常 profile / 迭代默认跑 `20 step`。
2. 做性能实验时，不要破坏 `Phase C` 单 clip 已验收基线。
3. 不要再把精力放回 `caption fairness` 上，当前主 benchmark 路径已经收口。
4. 如果改动默认推理配置、attention backend、offload、tiling 或 compile 策略，记得同步更新：
   - `docs_xzh/run_vivid_benchmark.md`
   - 仓库根目录 `AGENTS.md`
5. 阶段性结论仍然要用 `50 step` 复核，不要只看 `20 step`。


## 12. 一句话总结

这轮对话之后，`Phase D` 已经从“长视频实现和公平 benchmark 工具链已具备，但仍未过线”，推进到“语义完成对齐并通过 `6 / 20 / 50 step` 验收”；同时，后续 `Phase E` 所需的通用 `sglang` benchmark 入口、原版可对比时间统计、原版常驻总进度条和更新后的 benchmark 文档也都已经到位。
