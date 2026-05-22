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

## 7.2 建议至少保存以下中间信息

1. `prompt_embeds` shape / mean / std
2. `image_latent` shape / mean / std
3. `latents` 初始统计量
4. 第 1、N/2、最后一步 `noise_pred` 统计量
5. decode 前最终 latent 统计量
6. 输出视频前 3 帧快照

## 7.3 推荐 parity 工具

建议增加：

1. `python/sglang/multimodal_gen/test/manual/run_star_cogvideox_sr_smoke.py`
2. `python/sglang/multimodal_gen/test/manual/compare_star_sglang_outputs.py`

其中：

1. `run_star_cogvideox_sr_smoke.py`
   负责从 SGLang 跑出标准结果和中间摘要
2. `compare_star_sglang_outputs.py`
   负责读取 STAR 原结果与 SGLang 结果做比对

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

如果视觉差异很大，但统计量没有明显异常，需要优先回查 scheduler 和 transformer。

---

## 10. 失败信号与止损点

出现以下情况时，不要进入阶段 6：

1. decode 窗口逻辑还在频繁变化
2. 中间量对齐链路未建立
3. 输出视频还存在明显时序拼接错误
4. 无法判断问题来自 scheduler、DiT 还是 decode

这个阶段的重点不是“勉强能出视频”，而是“知道结果为什么对/为什么不对”。
