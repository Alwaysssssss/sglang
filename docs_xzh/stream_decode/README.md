# SGLang VideoEdit 流式解码改造计划

## 1. 目标

当前 `sglang` 集成的 VideoEdit 在进入窗口推理前，会先把整段输入视频和整段 mask 全量解码到 CPU 内存，再做全局裁剪、缩放和窗口切分。对长视频或高分辨率视频，这条路径会直接放大 CPU 内存占用，成为服务稳定性的主要风险。

本计划的目标是把当前实现改造成：

1. 输入视频不再全量解码到 CPU。
2. 按窗口顺序进行流式解码、裁剪、缩放和缓存。
3. 解码与当前窗口的 VAE/DiT 推理支持异步重叠，尽量隐藏 decode 耗时。
4. 对外请求协议、推理参数、窗口融合逻辑、输出行为保持兼容。
5. 改造后结果需要通过逐帧验收。

## 2. 当前基线

### 2.1 基线运行命令

当前基线推理命令位于：

- `/mnt/nas/xzh/project/VideoEdit/sglang/docs_xzh/run_edit.md`

本次改造以该文档中的这组参数作为首要验收基线：

- 输入视频：`/mnt/nas/models/DifusserEdit/pexel_test_data_0410/videos/15108907_3840_2160_50fps_short.mp4`
- 输入 mask：`/mnt/nas/models/DifusserEdit/pexel_test_data_0410/masks/15108907_3840_2160_50fps_No_bbox_mask.mp4`
- 输出视频：`/mnt/nas/xzh/project/VideoEdit/sglang/output_results/15108907_3840_2160_50fps_api_sp1_no_offload_fa_156f_all_gpu0.mp4`
- `num_frames=156`
- `infer_len=81`
- `overlap=0`
- `num_inference_steps=20`
- `guidance_scale=5.0`
- `seed=42`

### 2.2 基线输出

当前基线输出文件：

- `/mnt/nas/xzh/project/VideoEdit/sglang/output_results/15108907_3840_2160_50fps_api_sp1_no_offload_fa_156f_all_gpu0.mp4`

已确认该文件当前属性为：

- `156` 帧
- `25.0` fps
- `1920x1080`

### 2.3 当前问题定位

当前全量解码路径主要在以下文件：

- `python/sglang/multimodal_gen/runtime/videoedit/preprocess.py`
- `python/sglang/multimodal_gen/runtime/pipelines/wan_videoedit_pipeline.py`
- `python/sglang/multimodal_gen/runtime/videoedit/mask_io.py`

关键问题：

1. `prepare_global_inputs()` 会一次性加载全部 `original_frames` 和全部 `raw_mask_frames`。
2. 裁剪后会生成整段 `resized_video` 和整段 `resized_masks`。
3. 后续窗口只是在这些内存列表上按 index 取帧，并不是按窗口从文件流式读取。
4. 该结构会让 CPU 内存占用随 `num_frames * H * W` 线性增长。

## 3. 改造原则

### 3.1 保持不变

以下行为在第一阶段不改：

1. 请求协议不改。
2. `build_videoedit_window_specs()` 的窗口定义不改。
3. overlap 融合权重逻辑不改。
4. DiT、VAE、scheduler、CFG、paste-back 行为不改。
5. 最终输出文件格式和 sidecar metadata 行为不改。

### 3.1.1 新增模式选择参数

本次改造需要显式增加一个可输入参数，让调用方可以选择：

1. 旧模式：全量解码到 CPU 后再按窗口推理
2. 新模式：流式解码 + 有界缓存 + 可选异步预取

建议参数名：

- `decode_mode`

建议取值：

- `eager`
- `stream`

建议默认值：

- 当前代码默认值为 `stream`

原因：

1. 保证现网和已有脚本零行为变化。
2. 新路径先通过灰度和逐帧验收。
3. 验收稳定后再讨论是否切默认值。

### 3.2 第一阶段允许保守实现

第一阶段优先追求稳定和可验收，不追求最激进的吞吐优化：

1. 可以先实现顺序流式解码 + 有界缓存。
2. 再实现异步预取。
3. 不在第一阶段引入 GPU decode。
4. 不在第一阶段改动模型数值路径。

## 4. 目标架构

### 4.1 总体思路

把当前“全量帧列表”改成“窗口驱动的帧提供器”，但保留旧 eager 路径，通过 `decode_mode` 做运行时选择。

新的数据流：

1. 若 `decode_mode=eager`，继续走当前全量解码路径。
2. 若 `decode_mode=stream`，启动请求后先做一次全局信息探测。
3. `stream` 模式下只解析视频 metadata 和 mask metadata，不立刻把全部帧解到内存。
4. `stream` 模式下用一次顺序扫描得到全局 dilated mask bbox。
5. bbox 确定后，构建顺序流式解码器。
6. 管线按窗口推进时，只请求当前窗口需要的输入帧。
7. 帧提供器在后台预取后续窗口帧，主线程继续当前窗口推理。

### 4.1.1 模式切换入口

建议在以下层级统一接入 `decode_mode`：

1. OpenAI video API 请求参数
2. 本地 CLI 参数
3. `WanVideoEditSamplingParams`

建议优先级：

1. 显式请求参数
2. CLI 默认值
3. 代码内默认值 `stream`

建议行为：

1. `decode_mode=eager` 时，沿用 `prepare_global_inputs()`
2. `decode_mode=stream` 时，走新的 bbox scan + frame provider 路径
3. 未识别取值直接报错，不做 silent fallback

### 4.2 新增核心抽象

建议新增以下模块：

- `python/sglang/multimodal_gen/runtime/videoedit/stream_decoder.py`
- `python/sglang/multimodal_gen/runtime/videoedit/frame_provider.py`
- `python/sglang/multimodal_gen/runtime/videoedit/window_materializer.py`

建议抽象：

1. `SequentialVideoDecoder`
2. `SequentialMaskDecoder`
3. `WindowFrameProvider`
4. `PrefetchController`

### 4.3 SequentialVideoDecoder

职责：

1. 对单个视频文件维持一个常驻 ffmpeg 解码进程。
2. 顺序读取原始帧。
3. 将帧写入有界缓存。
4. 支持按连续 index 区间取帧。
5. 支持关闭、异常回收和提前结束。

建议接口：

```python
class SequentialVideoDecoder:
    def open(self) -> None: ...
    def start_prefetch(self) -> None: ...
    def get_frames(self, start: int, length: int) -> list[Image.Image]: ...
    def get_frame_count(self) -> int: ...
    def close(self) -> None: ...
```

约束：

1. 第一阶段只支持单调递增访问。
2. 不支持随机回跳。
3. 如遇窗口反射补帧，优先从已提交输出或已缓存原始帧解决，不回源随机 seek。

实现注记：

1. 当前 phase2/phase3 落地版本为保证与原 eager 路径逐帧一致，输入解码 backend 先采用顺序 `OpenCV VideoCapture`。
2. 抽象层已经收敛在 `stream_decoder.py` / `frame_provider.py`，后续如需切换到常驻 ffmpeg 进程，只需要替换 decoder backend，不需要再改窗口执行逻辑。

### 4.4 SequentialMaskDecoder

mask 源有三类：

1. 视频 mask
2. numpy mask
3. COCO JSON mask

建议统一成 `MaskFrameProvider` 接口，但实现分开：

1. 视频 mask 复用顺序 ffmpeg 解码逻辑。
2. numpy / COCO JSON 保持按索引读取，但只保留有界缓存，不全量转成 `list[PIL.Image]`。

### 4.5 WindowFrameProvider

职责：

1. 对外提供“窗口级帧物化”能力。
2. 内部持有 video decoder、mask decoder、bbox、resize 参数、缓存和预取线程。
3. 在 `window_spec.input_indices` 单调前进时返回窗口帧。
4. 当 `use_repaired_context=true` 时，优先用已经提交的修复结果覆盖窗口输入中的历史帧。

建议接口：

```python
class WindowFrameProvider:
    def materialize_window(
        self,
        input_indices: list[int],
        use_repaired_context: bool,
    ) -> tuple[list[Image.Image], list[Image.Image]]: ...

    def notify_window_committed(self, global_indices: list[int], frames: list[Image.Image]) -> None: ...
    def close(self) -> None: ...
```

## 5. 两阶段处理策略

### 5.1 阶段 A：全局 bbox 探测

当前 `prepare_global_inputs()` 依赖整段 dilated mask 才能得到全局 bbox。这个约束第一阶段不改。

因此建议拆成两步：

1. `scan_global_bbox()`
2. `stream_window_inputs()`

`scan_global_bbox()` 只做：

1. 顺序读取视频尺寸和 fps
2. 顺序读取 mask
3. 做 dilate / scale
4. 统计全局 bbox
5. 不保留完整帧列表

这样虽然 still 需要完整扫描一遍 mask，但内存只与单帧和小缓存有关，不与总帧数线性绑定。

`decode_mode=eager` 下不进入该阶段，直接保持现有 `prepare_global_inputs()` 行为。

### 5.2 阶段 B：窗口流式推理

bbox 确定后再正式进入推理期：

1. 新开视频 decoder 和 mask decoder
2. 顺序读取当前窗口需要的帧
3. 对每帧执行 crop + resize
4. 组装成 `runtime_window_frames/runtime_window_masks`
5. 继续走现有 `prepare_window_inputs()`、VAE encode、DiT、VAE decode

## 6. 异步预取设计

### 6.1 当前缺陷

当前 `MGErase` 虽然是流式分块解码，但每次缺帧时还是现场起 ffmpeg 子进程并同步读完。这会把 decode latency 暴露在关键路径上。

本次 `sglang` 改造不应复用这种“临时起 ffmpeg”的模式，而应直接做常驻解码 + 后台预取。

该能力只在 `decode_mode=stream` 下启用。

### 6.2 推荐实现

每个 decoder 有一个后台线程：

1. ffmpeg 持续输出 rawvideo 到 pipe
2. 后台线程按 frame size 从 `stdout` 拆帧
3. 每读到一帧就转成 `PIL.Image` 或 numpy array
4. 写入有界 ring buffer
5. 当缓存达到高水位时阻塞
6. 当消费到低水位时继续预取

### 6.3 缓存策略

建议参数：

1. `decode_prefetch_window_count`
2. `decode_prefetch_max_frames`
3. `decode_prefetch_low_watermark`
4. `decode_prefetch_high_watermark`

默认建议：

1. `max_frames = 2 * infer_len`
2. `low_watermark = infer_len // 2`
3. `high_watermark = 2 * infer_len`

在当前基线 `infer_len=81` 下，缓存上限可先设为 `162` 帧量级，不允许无限扩张。

### 6.4 与窗口执行重叠

主线程：

1. 取当前窗口帧
2. 做 VAE encode / DiT / VAE decode
3. 提交窗口输出

后台线程：

1. 提前读取后续帧
2. 提前完成 raw decode
3. 可选提前完成 crop + resize

第一阶段建议只异步做原始 decode；crop + resize 先留在主线程，降低并发复杂度。

## 7. repaired context 兼容策略

当前 `WanVideoEditPipeline._materialize_window_inputs()` 在 `use_repaired_context=true` 时，会优先从 `runtime_accum_frames` 取历史修复结果覆盖窗口输入。

流式改造后需要保留该能力，但不能再依赖整段 `runtime_resized_frames` 列表。

建议：

1. 维护一个 `original_frame_cache`
2. 维护一个 `repaired_frame_cache`
3. 对于当前窗口的历史 index：
   - 若该帧已被 commit，则优先用 repaired frame
   - 否则从 original decoder cache 取

这样可以保持原有“前窗结果反馈给后窗”的行为。

## 8. 反射补帧策略

当前 `build_videoedit_window_specs()` 可能在最后一个窗口产生反射索引。

建议实现：

1. 如果反射索引仍在 decoder 最近缓存范围内，直接复用缓存帧。
2. 如果反射索引对应的是历史已提交帧，优先取 repaired frame。
3. 第一阶段不支持向后随机 seek 回源。
4. 若缓存策略无法覆盖最后窗口反射场景，则在第一阶段允许保留一个“小型历史帧缓存”，至少覆盖 `infer_len` 范围。

## 9. 代码修改落点

### 9.1 必改文件

1. `python/sglang/multimodal_gen/runtime/videoedit/preprocess.py`
2. `python/sglang/multimodal_gen/runtime/videoedit/mask_io.py`
3. `python/sglang/multimodal_gen/runtime/pipelines/wan_videoedit_pipeline.py`
4. `python/sglang/multimodal_gen/configs/sample/videoedit_wan.py`
5. `python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py`
6. `python/sglang/multimodal_gen/runtime/videoedit/cli.py`

### 9.2 新增文件

1. `python/sglang/multimodal_gen/runtime/videoedit/stream_decoder.py`
2. `python/sglang/multimodal_gen/runtime/videoedit/frame_provider.py`
3. `python/sglang/multimodal_gen/runtime/videoedit/bbox_scan.py`

### 9.3 建议新增测试

1. `python/sglang/multimodal_gen/test/unit/test_videoedit_stream_decoder.py`
2. `python/sglang/multimodal_gen/test/unit/test_videoedit_frame_provider.py`
3. `python/sglang/multimodal_gen/test/unit/test_videoedit_bbox_scan.py`
4. `python/sglang/multimodal_gen/test/unit/test_videoedit_stream_vs_eager.py`
5. `python/sglang/multimodal_gen/test/unit/test_videoedit_decode_mode_params.py`

## 9.4 参数接入设计

### 请求与 CLI

建议在请求和 CLI 中增加：

```json
"decode_mode": "stream"
```

CLI 建议增加：

```bash
--decode-mode eager
--decode-mode stream
```

### Sampling Params

建议在 `WanVideoEditSamplingParams` 中新增字段：

```python
decode_mode: str = "stream"
```

并增加校验：

1. 只允许 `eager` / `stream`
2. 非法值直接抛 `ValueError`

### Pipeline 分支点

建议在 `WanVideoEditPipeline.forward()` 或 `_prepare_global_videoedit_context()` 内部做单一分支：

1. `eager`：`prepare_global_inputs()`
2. `stream`：`scan_global_bbox() + build WindowFrameProvider`

要求：

1. 两条路径共用相同的窗口执行、commit 和 finalize 逻辑
2. 差异尽量只收敛在“输入帧获取”这一层

## 10. 分阶段实施计划

### Phase 0：基线冻结

1. 固定 `run_edit.md` 当前命令。
2. 固定当前基线输出视频作为对比标准。
3. 增加 `decode_mode` 参数但默认值保持 `eager`。
4. 固定对比脚本输入参数和验收阈值。

产物：

1. baseline metadata
2. baseline frame metrics report format
3. `decode_mode=eager` 与当前分支完全一致的确认结果

### Phase 1：抽离 bbox 扫描

1. 从 `prepare_global_inputs()` 中拆出 bbox 扫描逻辑。
2. 实现只扫描 mask / 必要视频 metadata 的全局 bbox 计算。
3. 保持 `decode_mode=eager` 旧路径完全不变。
4. 新增 `decode_mode=stream` 但暂未接管正式窗口推理。

验收：

1. bbox 与旧实现完全一致。
2. crop_h/crop_w/aligned_h/aligned_w 一致。

### Phase 2：引入顺序 decoder 和窗口帧提供器

1. 实现常驻 ffmpeg decoder。
2. 实现窗口级 `materialize_window()`。
3. 用 provider 替代 `runtime_resized_frames/runtime_resized_masks` 全量列表。
4. 仅在 `decode_mode=stream` 下启用新路径。
5. 先不启用异步预取，只做同步按需顺序取帧。

验收：

1. `decode_mode=eager` 输出与当前基线完全一致。
2. `decode_mode=stream` 单窗口输入与旧路径逐帧一致。
3. `decode_mode=stream` 整条视频输出通过逐帧验收。
4. `decode_mode=stream` CPU 峰值内存显著下降。

### Phase 3：引入后台预取

1. 增加 decoder 后台线程。
2. 增加有界缓存和水位控制。
3. 在窗口推理期间预取下一批帧。
4. 增加关闭、异常、超时和取消清理逻辑。
5. 该能力仅作用于 `decode_mode=stream`。

验收：

1. `decode_mode=eager` 结果不变。
2. `decode_mode=stream` 结果不变。
3. `decode_mode=stream` decode 阶段平均等待下降。
4. 无线程泄漏、无子进程泄漏。

### Phase 4：压测与回归

1. 用基线样例做逐帧验收。
2. 用更长视频验证内存曲线。
3. 用 `num_frames=-1` 验证全帧模式。
4. 用 reference image 场景验证 `drop_reference_frame` 和首帧行为。

## 11. 逐帧验收方案

### 11.1 对比对象

对比视频：

1. baseline：`output_results/15108907_3840_2160_50fps_api_sp1_no_offload_fa_156f_all_gpu0.mp4`
2. candidate eager：加入 `decode_mode=eager` 后的新输出
3. candidate stream：加入 `decode_mode=stream` 后的新输出

### 11.2 验收指标

必须满足以下阈值：

```python
min_ssim = 0.95
max_mse = 60.0
max_mae = 5.0
max_failed_frame_ratio = 0.0
```

解释：

1. 每一帧都必须满足 `ssim >= 0.95`
2. 每一帧都必须满足 `mse <= 60.0`
3. 每一帧都必须满足 `mae <= 5.0`
4. 不允许任何失败帧，失败帧比例必须是 `0.0`

要求：

1. `candidate eager` 必须首先与 baseline 对齐，作为参数接入正确性的确认。
2. `candidate stream` 再与 baseline 做同样阈值验收。

### 11.3 建议对比脚本

建议新增：

- `python/sglang/multimodal_gen/runtime/videoedit/compare_video_frames.py`

输出：

1. 每帧 `ssim/mse/mae`
2. 最差帧列表
3. 总结 JSON
4. 失败即返回非零退出码

### 11.4 验收命令建议

建议增加一条单独的验收命令，输入为 baseline 和 candidate 两个视频路径，输出 JSON 报告和最差帧截图目录。

## 12. 性能与稳定性观测项

改造后需要重点记录：

1. 请求总耗时
2. `VideoEditConditionEncodingStage` 耗时
3. `VideoEditDenoisingStage` 耗时
4. `VideoEditDecodingStage` 耗时
5. decode 等待时间
6. CPU RSS 峰值
7. 后台解码缓存帧数水位
8. ffmpeg 子进程生命周期

## 13. 风险点

### 13.1 bbox 扫描与正式解码的二次读取

第一阶段会有两遍输入访问：

1. 扫描 bbox
2. 正式窗口推理

这会增加总 I/O，但可以显著降低峰值内存。该 tradeoff 在第一阶段可接受。

### 13.2 ffmpeg 管道阻塞

若后台预取过快、主线程消费过慢，pipe 和缓存都可能阻塞。需要：

1. 有界缓存
2. 明确水位
3. 明确 close / cancel / join 逻辑

### 13.3 异常回收

请求中途失败时必须保证：

1. ffmpeg 子进程退出
2. 后台线程退出
3. 临时缓存释放

### 13.4 数值一致性

任何 resize、颜色空间、mask 二值化路径的微小差异都可能影响最终视频逐帧指标。第一阶段必须复用当前 eager 路径中的：

1. `expand_mask_frames()`
2. `crop_frames()`
3. `resize_frames()`
4. `prepare_window_inputs()`

不能另写一套近似实现。

## 14. 回退策略

建议以请求参数和 CLI 参数为主，不建议只做环境变量开关。

行为：

1. 当前默认值已切换为 `decode_mode=stream`
2. 如需保留旧路径，可显式传入 `decode_mode=eager`
3. 若 `stream` 路径出现稳定性问题，调用方可以立即切回 `eager`
4. 如需调试或压测，可额外补充环境变量覆盖，但不作为主接口

## 15. 本次文档对应的落地范围

本次只落盘设计计划，不做代码修改。

后续代码修改应严格按以下顺序推进：

1. 先拆 bbox 扫描
2. 再接入窗口级 frame provider
3. 再加后台异步预取
4. 每一阶段都跑逐帧验收
