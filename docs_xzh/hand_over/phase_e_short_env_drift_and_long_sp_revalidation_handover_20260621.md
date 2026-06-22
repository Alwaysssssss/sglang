# VividVR Phase E：短视频环境漂移证据与双卡长视频 SP 复验交接

更新时间：`2026-06-21 UTC`

## 1. 文档目的

本文档面向下一位继续推进 `VividVR Phase E` 的 Codex，记录当前项目最重要的三条结论：

1. 当前 `sglang` 短视频 `Phase C` 路径已经收口到 `ssim_mean ≈ 0.97025`，并且在**同一运行环境**下与原版 `Vivid-VR` 的第一个 denoising step 做到了逐张量完全一致。
2. 原版 `Vivid-VR` 在它自己的 `.venv` 里可以对历史 short reference 做到**字节级复现**，但放到 `sglang/.venv` 后会掉到 `ssim_mean = 0.9711355802396912`，说明**底层运行时栈差异已经足以显著改变 short 结果**。
3. 双卡 `SP(pool=1)` 的长视频 `130f / 20 step` 正式质量复验已重新跑通，且结果高于 `2026-06-18` 的历史已验收值，当前长视频质量线稳定。

当前最重要的工作判断是：

- 如果后续继续追 short 的 `0.98+` 或更高目标，第一优先嫌疑已经不是“`sglang` 首步语义明显写错”，而是**运行环境 / 底层库版本漂移**。
- 如果后续继续做 release gate，长视频双卡 `SP(pool=1)` 这条线当前不需要重开质量回归调查。

---

## 2. 仓库与环境锚点

| 项 | 值 |
|----|-----|
| 分支 | `sglang_Vivid` |
| 当前 HEAD | `9cc2729d4d` |
| `sglang` Python | `/home/zhiheng/sglang/.venv/bin/python` |
| 原版 Python | `/home/zhiheng/Vivid-VR/.venv/bin/python` |
| GPU | `2 x NVIDIA A100-SXM4-80GB` |

当前工作区不是干净状态。`git status` 显示至少存在：

- `.gitignore`
- `python/sglang/multimodal_gen/runtime/videoedit/preprocess.py`

下一轮默认**不要**做工作区清理，也不要回退这些本地改动。

---

## 3. 当前已经被证实的事实

### 3.1 当前 `sglang` 短视频 `Phase C` 正式结果

最新 formal 指标文件：

- `/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_c_metrics_seed42_20260619T124936Z.json`

关键结果：

- `pass_compare = true`
- `ssim_mean = 0.9702456939196501`
- `ssim_min = 0.9622448299640508`
- `total_runtime_seconds = 788.145639`
- `model_inference_runtime_seconds = 769.817355`

这条结果对应的是前几轮 short 对齐之后的稳定状态，主要已经补齐过：

- control-video decode 对齐到原版 `decord`
- VAE slicing / tiling 显式生效
- text encoder 调用方式向原版 short 路径收紧

当前 short 正式结果仍然低于历史 reference 自身，但已经明显高于 `Phase C` 最早期 `0.9677` 左右的门线。

### 3.2 同环境下，`sglang` 与原版 step 0 语义严格一致

语义 trace 对比文件：

- `/home/zhiheng/sglang/Vivid_Acceptance/semantic_trace/20260619T124547Z/compare_original_step0_currentenv_vs_existing_sglang.json`

关键事实：

- `first_diverged_step = null`
- 下列张量 `max_abs_diff = 0`：
  - `control_latents`
  - `initial_latents`
  - `prompt_embeds`
  - `negative_prompt_embeds`
  - `timesteps`
  - `first_step_noise_pred_raw`
  - `first_step_noise_pred_guided`
  - `first_step_merged_latents`
  - `first_step_merged_old_pred`

这说明当前 `sglang` 路径至少在**同环境、同 seed、同输入**下的第一个 denoising step 上，不存在显式语义分叉。

### 3.3 原版 `Vivid-VR` 在自己的 `.venv` 里可字节级复现 short reference

原版复现产物目录：

- `/home/zhiheng/sglang/Vivid_Acceptance/original_short_recheck_20260621T041924Z`

compare 报告：

- `/home/zhiheng/sglang/Vivid_Acceptance/indicator/original_short_recheck_vs_reference_20260621T041924Z.json`

关键结果：

- `ssim_mean = 1.0`
- `ssim_min = 1.0`
- `mse_mean = 0.0`
- `mae_mean = 0.0`
- `pass_compare = true`

对应视频 SHA256：

- reference：`533b065bac4ef2f6431257de3cdf3e3e8cbb617cf8ea4c6e8f81c902d01f07e4`
- 原版复现输出：`533b065bac4ef2f6431257de3cdf3e3e8cbb617cf8ea4c6e8f81c902d01f07e4`

这说明：

- 历史 short reference 不是“不可复现的老产物”
- 原版仓库在其原始依赖栈下，确实可以精确复现该 short reference

### 3.4 原版 `Vivid-VR` 放到 `sglang/.venv` 后，不再能复现 short reference

原版代码在 `sglang/.venv` 下的复现产物目录：

- `/home/zhiheng/sglang/Vivid_Acceptance/original_on_sglang_env_short_20260621T043511Z`

compare 报告：

- `/home/zhiheng/sglang/Vivid_Acceptance/indicator/original_on_sglang_env_short_vs_reference_20260621T043511Z.json`

关键结果：

- `pass_compare = true`
- `ssim_mean = 0.9711355802396912`
- `ssim_min = 0.9637546286415938`
- `mse_mean = 32.1875938143049`
- `mae_mean = 2.974415523665292`

对应视频 SHA256：

- reference：`533b065bac4ef2f6431257de3cdf3e3e8cbb617cf8ea4c6e8f81c902d01f07e4`
- 原版在 `sglang/.venv` 输出：`bc6f0b2b74e78d2daeecbf7c3187f1d990dfded1d46e78fffec3185f6a215b83`

这条证据非常关键。它说明即使**完全不走 `sglang` 运行时代码**，只要把原版 `Vivid-VR` 放到当前 `sglang` 依赖环境中，short 结果也会显著偏离历史 reference。

### 3.5 两套环境的关键底层库版本确实不一致

| 包 | 原版 `.venv` | `sglang/.venv` |
|----|--------------|----------------|
| `torch` | `2.2.1+cu121` | `2.9.1+cu128` |
| `diffusers` | `0.31.0` | `0.37.0` |
| `transformers` | `4.42.4` | `5.3.0` |
| `decord` | `0.6.0` | `0.6.0` |
| `opencv-python` / `cv2` | `4.13.0` | `4.10.0` |
| `numpy` | `1.26.4` | `2.2.6` |

因此，当前把“底层库版本 / 运行时栈差异”列为 short gap 的高优先级嫌疑，是有直接证据支持的，不是猜测。

### 3.6 环境漂移已经解释了 short gap 的大部分量级

当前两条最接近的对照是：

- 原版代码 + `sglang/.venv`：`ssim_mean = 0.9711355802396912`
- 当前 `sglang Phase C`：`ssim_mean = 0.9702456939196501`

两者只差约 `0.00089`。

这意味着：

- `sglang` 当前 short 路径与“原版代码但同样跑在 `sglang` 依赖环境里”的结果已经非常接近
- short 从 `1.0` 掉到 `0.971` 的主要量级，已经可以由环境漂移单独解释掉大部分
- 剩余小差距仍然可能来自后续 denoising steps、decode、postprocess 或少量实现细节，但它已经不是当前最大的解释项

### 3.7 双卡 `SP(pool=1)` 长视频 `130f / 20 step` 已重新完成正式质量复验

本轮复验产物：

- 日志：`/home/zhiheng/sglang/Vivid_Acceptance/logs/phase_e41_recheck_sp_v2_default_pool1_130f_20step_compile_20260621T045304Z.log`
- 指标：`/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_e41_recheck_sp_v2_default_pool1_130f_20step_compile_metrics_seed42_20260621T045315Z.json`
- 视频：`/home/zhiheng/sglang/Vivid_Acceptance/result_videos/phase_e41_recheck_sp_v2_default_pool1_130f_20step_compile_seed42_20260621T045315Z.mp4`

关键结果：

- `pass_compare = true`
- `ssim_mean = 0.9871715024815744`
- `ssim_min = 0.9828873473700072`
- `total_runtime_seconds = 781.154236`
- `model_inference_runtime_seconds = 533.573864`

视频 profile 复核：

- `codec_name = h264`
- `width = 960`
- `height = 720`
- `pix_fmt = yuv420p`
- `fps = 25`
- `nb_frames = 130`

与 `2026-06-18` 的历史已验收双卡 `SP(pool=1)` 结果相比：

- 历史指标：`/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_e41_stage_executor_sp_v2_default_pool1_130f_20step_compile_metrics_seed42_20260618T074059Z.json`
- 历史 `ssim_mean = 0.9845430592776135`
- 历史 `ssim_min = 0.9783342201035932`

结论是：

- 当前双卡 `SP(pool=1)` 长视频质量没有回退
- 这条线当前可以继续视为稳定默认质量口径

---

## 4. 当前项目状态的最佳解释

截至本文档更新时，当前最稳妥的解释排序如下：

1. short 剩余质量差距的**头号嫌疑**是运行时栈差异，尤其是 `torch / diffusers / transformers / numpy / cv2` 等核心底层库版本漂移。
2. `sglang` 当前实现并未表现出“step 0 就明显语义错位”的证据；在同环境下，首个 denoising step 已经严格对齐。
3. 如果还存在代码层面的 residual mismatch，更可能出现在：
   - 后续多步 denoising 累积
   - decode 路径
   - `drop first 3 frames + crop padding + AdaIN/reference color fix` 一类 postprocess 细节
4. 长视频双卡 `SP(pool=1)` 主线当前更像“已验证稳定”，而不是“仍在质量异常状态”。

---

## 5. 下一轮推荐动作

### 5.1 如果目标是继续追 short 的更高对齐

建议优先按下面顺序推进：

1. 先把原版 `.venv` 视为 short gold environment，不要继续拿不同依赖栈下的结果直接推断语义回归。
2. 如果需要把 `sglang` 真正逼近历史 short reference，优先做**环境收敛 / 依赖 bisect**，而不是一开始就继续改主链代码。
3. 如果要继续做代码级语义追查，下一步应该把张量 trace 从 `step 0` 扩展到：
   - 若干中间 denoising steps
   - decode 前 latents
   - decode 后帧
   - postprocess 前后帧

### 5.2 如果目标是继续守住 release gate

1. 长视频双卡正式默认仍然使用 `SP(pool=1)`。
2. 当前 `130f / 20 step / seed=42 / fa_sp / compile` 口径已经重新验证，无需因为这轮讨论再重开长视频质量疑点。
3. 后续如需继续做性能实验，应将其与正式默认质量口径分开记录。

---

## 6. 本轮交接结论

一句话总结：

**当前 `VividVR` 的主要不确定性已经从“`sglang` 是否把原版语义写错”转向“当前依赖环境是否已经改变了原版 short 结果本身”；与此同时，双卡 `SP(pool=1)` 长视频正式质量线已再次确认稳定。**
