# Phase 5：解码对齐、端到端联调与验收

## 1. 阶段目标

本阶段的目标是把前几个阶段的组件真正串起来，完成：

1. 自定义 decode 路径
2. 端到端推理
3. 与 STAR 原始实现的结果对齐
4. 第一轮完整功能验收

阶段完成后，应满足：

1. SGLang native pipeline 可端到端生成结果视频
2. decode 路径复现 STAR 的时间窗口逻辑
3. 在固定输入、固定 seed 下，可与 STAR 原始实现做中间量和结果对齐
4. 已有明确的“通过/不通过”验收标准

---

## 2. 本阶段范围

### 本阶段处理

1. `STARCogVideoXSRDecodingStage` 完整实现
2. 时序分块 decode
3. 端到端 smoke run
4. 中间量对齐
5. 结果视频对齐

### 本阶段不处理

1. 高级性能优化
2. TP / SP 完整支持
3. 上游同步规范收尾

---

## 3. 自定义 decoding stage 的实现重点

## 3.1 为什么必须自定义

STAR 原始解码不是：

1. 一次性整段 VAE decode

而是：

1. 对 latent 按时间窗口切块
2. 分多次调用 `first_stage_model.decode`
3. 再沿时间维拼接

这意味着不能直接用通用 `DecodingStage.decode(batch.latents)` 替代。

## 3.2 推荐 stage 文件

1. `runtime/pipelines_core/stages/model_specific_stages/star_cogvideox_sr_decoding.py`

## 3.3 推荐职责拆分

建议该 stage 分成四步：

1. `prepare_latents_for_decode`
2. `build_decode_windows`
3. `decode_in_windows`
4. `postprocess_decoded_video`

---

## 4. 时间窗口 decode 逻辑

根据 STAR 原始 `sample_sr.py`，应优先保持其现有窗口策略。

## 4.1 当前推荐还原策略

原始逻辑的关键点是：

1. 第一段 decode 前 3 帧
2. 之后按 2 帧为单位继续 decode
3. 最后一段可能需要清 cache

建议先按原逻辑还原，不要提前抽象成“更通用”的窗口器。

### 推荐实现接口

```python
def build_decode_windows(num_frames: int) -> list[tuple[int, int, bool]]:
    ...
```

返回：

1. `start_frame`
2. `end_frame`
3. `clear_cache`

## 4.2 推荐伪代码

```python
def build_decode_windows(num_frames):
    windows = []
    loop_num = (num_frames - 1) // 2
    for i in range(loop_num):
        if i == 0:
            start, end = 0, 3
        else:
            start, end = i * 2 + 1, i * 2 + 3
        clear_cache = (i == loop_num - 1)
        windows.append((start, end, clear_cache))
    return windows
```

实现时如果发现某些 `num_frames` 下边界不够稳，再在本阶段修文档并加测试，不要默默改语义。

---

## 5. decode 前处理与后处理

## 5.1 decode 前处理

建议沿用标准逻辑，但放在自定义 decoding stage 中显式完成：

1. 调用 `pipeline_config.get_decode_scale_and_shift`
2. 对 latent 先反 scale / shift
3. 必要时调用 `pipeline_config.preprocess_decoding`

## 5.2 decode 输出后处理

解码后建议统一执行：

1. `(image / 2 + 0.5).clamp(0, 1)`
2. 输出格式整理为 `[B, T, C, H, W]` 或框架约定格式

## 5.3 color fix 策略

本阶段只建议做：

1. 在 sampling params 中保留 `enable_color_fix` 接口
2. 在 decoding stage 中预留 `apply_color_fix(...)`

MVP 推荐：

1. 默认关闭 color fix
2. 先把“无 color fix 的结果 parity”做通

原因：

1. color fix 容易掩盖 decode / latent 本身的对齐问题

---

## 6. 端到端联调顺序

建议按以下顺序联调：

1. 随机张量通路
   - 不依赖真实权重，只验证 shape 和 stage 串接
2. 真实权重 + 最小视频样本
   - 目标是能输出一段视频
3. 固定 seed 的中间量对齐
4. 最终视频对齐

### 第一步：随机张量通路

验证：

1. `condition_video -> image_latent -> concat -> transformer -> scheduler -> decoding`
2. 所有阶段 shape 稳定

### 第二步：真实权重最小样本

验证：

1. 真实条件视频能跑通
2. 能产出合法 mp4

### 第三步：中间量对齐

优先比对：

1. 文本 embedding shape
2. 条件 latent shape 和均值/方差
3. 初始噪声 latent shape 和统计量
4. 若干 timestep 的 noise_pred 统计量
5. decode 前 latent 统计量

### 第四步：最终结果对齐

比对：

1. 输出帧数
2. 分辨率
3. 主观观感
4. 简单客观指标

---

## 7. parity 验证方法

## 7.1 建议固定条件

每次 parity 必须固定：

1. prompt
2. negative prompt
3. condition video
4. seed
5. num_frames
6. height / width
7. num_inference_steps
8. guidance_scale

## 7.2 parity 的比对层级

本阶段建议把 parity 分成三个层级，不要只做最终 mp4 的肉眼检查：

1. **中间量 parity**
   对齐 `prompt_embeds`、`image_latent`、初始噪声、若干关键步 `noise_pred`、decode 前最终 latent。
2. **逐帧图像 parity**
   对齐 reference 与 candidate 的逐帧像素统计，作为本阶段的主验收口径。
3. **封装结果 parity**
   只检查最终 mp4 的帧数、fps、分辨率、时序是否稳定；这一层不能替代逐帧图像 parity。

如果第 1 层已经明显偏离，不要直接进入第 2 层或第 3 层排查。

## 7.3 建议至少保存以下中间信息

1. `prompt_embeds` shape / mean / std
2. `image_latent` shape / mean / std
3. `latents` 初始统计量
4. 第 1、N/2、最后一步 `noise_pred` 统计量
5. decode 前最终 latent 统计量
6. 输出视频前 3 帧快照

## 7.4 推荐 parity 工具

建议增加：

1. `python/sglang/multimodal_gen/test/manual/run_star_cogvideox_sr_smoke.py`
2. `python/sglang/multimodal_gen/test/manual/compare_star_sglang_outputs.py`

其中：

1. `run_star_cogvideox_sr_smoke.py`
   负责从 SGLang 跑出标准结果和中间摘要
2. `compare_star_sglang_outputs.py`
   负责读取 STAR 原结果与 SGLang 结果做逐帧与统计量比对

建议 `compare_star_sglang_outputs.py` 同时支持：

1. `--mode smoke`
2. `--mode baseline`
3. `--mode strict`

避免把阈值硬编码在脚本里。

---

## 8. 阶段内测试计划

建议新增：

1. `python/sglang/multimodal_gen/test/unit/test_star_decoding_windows.py`
2. `python/sglang/multimodal_gen/test/unit/test_star_decoding_stage.py`
3. `python/sglang/multimodal_gen/test/manual/run_star_cogvideox_sr_smoke.py`

### `test_star_decoding_windows.py`

至少覆盖：

1. 给定 `num_frames` 时窗口切分正确
2. 最后一段 `clear_cache` 标记正确

### `test_star_decoding_stage.py`

至少覆盖：

1. decode 前 scale / shift 路径正确
2. 分块输出能正确拼接为完整时间轴
3. `num_frames=7/9/11/13` 这类 STAR 常用长度下窗口边界稳定

### `run_star_cogvideox_sr_smoke.py`

至少支持：

1. 固定 `prompt + condition_video_path + seed`
2. 输出 candidate mp4
3. 输出中间统计 json
4. 可选落盘逐帧 png，便于排查编码器干扰

---

## 9. 阶段验收标准

本阶段完成标准：

1. SGLang native pipeline 能输出结果视频
2. 自定义 decode 路径稳定
3. 中间量比对链路建立完成
4. 与 STAR 原实现达到基本可接受对齐

### 建议的最小通过标准

1. shape 全对
2. 帧数全对
3. 结果无明显时序错位或解码断裂
4. 中间统计量在可解释范围内接近
5. 逐帧图像指标达到本阶段定义的 baseline 阈值

如果视觉差异很大，但统计量没有明显异常，需要优先回查 scheduler 和 transformer。

---

## 10. 失败信号与止损点

出现以下情况时，不要进入阶段 6：

1. decode 窗口逻辑还在频繁变化
2. 中间量对齐链路未建立
3. 输出视频还存在明显时序拼接错误
4. 无法判断问题来自 scheduler、DiT 还是 decode

这个阶段的重点不是“勉强能出视频”，而是“知道结果为什么对/为什么不对”。


## 11. Reference 逐帧对齐验收

本节定义 phase 5 的**最终结果验收口径**。
结论很简单：`SGLang` 集成结果必须和 `STAR_mg` reference 输出做自动化逐帧对齐，不能只看“能出视频”。

### 11.1 验收目标

本节要回答两个问题：

1. `SGLang` candidate 是否和 `STAR_mg` reference 在视觉结果上处于同一分布
2. 若不对齐，问题更像是编码器扰动，还是模型 / decode / scheduler 级错误

因此本节的验收重点不是 bit-exact，而是发现以下类型的问题：

1. 通道顺序错误
2. 数值尺度错误
3. decode 窗口错位
4. mask / latent packing 错误
5. scheduler 轨迹明显漂移
6. 解码后处理错误

### 11.2 Reference 资产定义

本阶段应冻结一份 reference case，不允许一边实现一边更换输入样本。

建议固定使用：

1. 输入目录：`/sgl-workspace/STAR_mg/input/cogvideox_test`
2. STAR 原始输出根目录：`/sgl-workspace/STAR_mg/cogvideox-based/sat/output/results`
3. 本次 reference 样例目录：
   `/sgl-workspace/STAR_mg/cogvideox-based/sat/output/results/0_A_serene_scene_of_a_panda_bear_playing_a_guitar_at_sunset_unfolds_by_a_tranquil_lake._The_panda,_with_its_black-and-whit`

建议把以下内容一并记录到 parity manifest 中：

1. prompt 文本
2. condition video 文件名
3. fps
4. num_frames
5. width / height
6. seed
7. num_inference_steps
8. guidance_scale
9. reference mp4 绝对路径

### 11.3 Reference 生成要求

Reference 必须来自 `STAR_mg` 原始推理链路，且推理时**不能使用 gt 进行颜色修复或后验增强**。

这里要明确保留原项目中的如下约束：

1. 不使用 `gt` 参与推理输出构造
2. 显式保持 color fix 关闭
3. 不允许把 `# samples = adain_color_fix(samples, gt)` 重新打开后再拿来做 reference

原因：

1. 我们当前接入目标是 STAR 的主推理结果，而不是“推理后再叠一层 reference aware 修复”
2. 如果 reference 开了 color fix，而 `SGLang` candidate 没开，会把 decode 偏差和后处理偏差混在一起

### 11.4 Reference 生成命令

原始文档中的命令带有容器内路径 `/workspace/STAR/...`。在当前工作区里，建议明确换成**本地实际路径**，避免后续执行口径不一致。

建议本地执行口径写死为：

```bash
cd /sgl-workspace/STAR_mg/cogvideox-based/sat
export STAR_COG_TEST_DATA_DIR=/sgl-workspace/STAR_mg/input/cogvideox_test
export STAR_COG_OUTPUT_DIR=/sgl-workspace/STAR_mg/cogvideox-based/sat/output/results
CUDA_VISIBLE_DEVICES=1 bash inference_sr.sh
```

如果后续在容器中复跑，可另外记录一份容器路径版命令，但不要和本地路径版混写在同一个验收步骤里。

### 11.5 Candidate 生成要求

`SGLang` 侧 candidate 输出必须满足：

1. 使用与 reference 相同的 prompt
2. 使用与 reference 相同的 condition video
3. 使用与 reference 相同的 seed
4. 使用与 reference 相同的 `num_frames / fps / width / height / num_inference_steps / guidance_scale`
5. 默认关闭 color fix
6. 输出 candidate mp4
7. 同时输出中间统计 json

如果 `SGLang` 侧还保留可选 color fix 开关，本节验收必须先走 `color_fix = false` 的基线口径。

### 11.6 比较输入与口径

逐帧对齐时，比较对象应是：

1. reference mp4
2. candidate mp4

主比较流程：

1. 逐帧读取 reference mp4 与 candidate mp4
2. 对齐帧索引后逐帧比较
3. 对每一帧计算：
   `SSIM`、`MSE`、`MAE`、`PSNR`、`max_abs_diff`
4. 输出全局统计：
   `ssim_mean`、`ssim_min`、`mse_mean`、`mse_max`、`mae_mean`、`mae_max`、`failed_frames`

这里的 `failed_frames` 建议记录完整结构，而不是只记帧号。至少包括：

1. `frame_index`
2. `ssim`
3. `mse`
4. `mae`
5. `psnr`
6. `max_abs_diff`
7. `failure_reasons`

### 11.7 帧对齐规则

逐帧比较前，必须先做以下一致性检查：

1. 分辨率一致
2. 通道数一致
3. reference 与 candidate 的帧数差不超过允许阈值

建议比较脚本显式支持：

1. `allow_frame_count_delta`
2. `drop_tail_frames`

默认策略建议为：

1. 若帧数完全一致，逐帧一一对齐
2. 若帧数只差 1 帧，默认允许截断尾帧后再比较
3. 若帧数差超过阈值，直接判失败，不进入逐帧指标阶段

不要在脚本里偷偷做重采样、补帧、光流对齐，这会掩盖真正的生成问题。

### 11.8 默认阈值分档

阈值不应只有一套。建议脚本至少提供三档：

1. `smoke`
2. `baseline`
3. `strict`

#### `baseline` 阈值

这是 phase 5 默认验收档，用于发现“质性偏差”，不是追求 bit-exact：

```text
min_ssim = 0.90
max_mse = 150.0
max_mae = 8.0
allow_frame_count_delta = 1
max_failed_frame_ratio = 0.05
```

#### `smoke` 阈值

只用于确认流程基本跑通，不用于最终验收：

```text
min_ssim = 0.80
max_mse = 400.0
max_mae = 15.0
allow_frame_count_delta = 1
max_failed_frame_ratio = 0.10
```

#### `strict` 阈值

只在同一台机器、同一 backend、同一视频编码器、同一 dtype 配置下使用，适合作为 release gate 候选：

```text
min_ssim = 0.95
max_mse = 60.0
max_mae = 5.0
allow_frame_count_delta = 0
max_failed_frame_ratio = 0.0
```

### 11.9 为什么不能要求 bit-exact

本项目当前不应把逐帧对齐定义成 bit-exact。原因包括：

1. H.264 / HEVC 编码是有损的
2. 不同 attention backend 可能带来小幅数值差异
3. 不同 GPU kernel / dtype / VAE 执行路径可能带来微小漂移
4. 同样的视觉结果，逐像素值也可能不完全一致

因此 phase 5 的目标是发现**大范围偏移**，而不是逐像素完全一致。

但无论选择哪一档阈值，都必须保留以下字段上报：

1. `ssim_min`
2. `mse_max`
3. `failed_frames`

这三项是回归排查时最有价值的信号。

### 11.10 输出产物要求

每次逐帧对齐验收，建议固定产出：

1. `candidate.mp4`
2. `reference.mp4`
3. `parity_metrics.json`
4. `failed_frames.json`
5. `frame_preview/`

其中 `frame_preview/` 建议至少保存：

1. 前 3 帧 reference png
2. 前 3 帧 candidate png
3. 若存在失败帧，保存对应失败帧的 reference / candidate / diff png

### 11.11 通过 / 不通过判定

建议 phase 5 的最终逐帧对齐判定规则写死为：

1. 帧数差不超过 `allow_frame_count_delta`
2. `ssim_min >= min_ssim`
3. `mse_max <= max_mse`
4. `mae_mean <= max_mae`
5. `failed_frame_ratio <= max_failed_frame_ratio`

默认情况下，phase 5 的“通过”应采用 `baseline` 档，而不是 `smoke` 档。

### 11.12 失败后的优先排查顺序

如果逐帧对齐不通过，建议按以下顺序排查：

1. 先查输入是否真的一致
   - prompt / seed / condition_video / num_frames / resolution / steps / guidance
2. 再查 decode 路径
   - scale / shift、窗口切分、拼接顺序、最后一段 cache 清理
3. 再查 scheduler 语义
   - timesteps、alpha/sigma 口径、step 更新公式
4. 再查 transformer 语义
   - latent concat、text 投影、AdaLN、RoPE、noise_pred 输出尺度
5. 最后再查编码器影响
   - mp4 codec、fps、像素格式、保存质量

不要在逐帧对齐失败后，第一反应就去调阈值；应先确认失败是噪声级差异，还是语义级偏差。
