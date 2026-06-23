# VideoEdit 多窗口 reference 对齐修改计划

## 目标

把 `/home/tyx/workspace/zhouhao6/sglang` 里的 VideoEdit 多窗口逻辑对齐到原始推理 `/home/tyx/workspace/zhouhao6/VideoEdit-diffusers`。

本计划只针对 `infer_len=81` 且 `overlap>0` 的多窗口场景。核心要求是：第一个窗口跑完 81 帧后，后续窗口的 reference 帧选取和使用方式必须与原始 `infer.py` 完全一致。

## 原始实现的行为 contract

来源：

- `/home/tyx/workspace/zhouhao6/VideoEdit-diffusers/infer.py`
- `/home/tyx/workspace/zhouhao6/VideoEdit-diffusers/utils/preprocess.py`

原始实现的多窗口行为如下：

1. 窗口长度固定为 `infer_len=81`，`stride = infer_len - overlap`。
2. 窗口起点生成逻辑为：

```text
window_starts = [0]
next_start = stride
while next_start + overlap < total_frames:
    window_starts.append(next_start)
    next_start += stride
```

这等价于后续窗口起点为 `0, stride, 2*stride, ...`，直到前一个窗口的真实覆盖已经到达尾部。

3. 第 0 个窗口不使用上一窗口生成结果作为 reference。
4. 对第 `k>0` 个窗口，且 `overlap>0`，只有当第 `k-1` 个窗口确实刚刚执行过时，才从上一窗口生成结果里取：

```text
ref_local_idx = stride = infer_len - overlap
reference_frame_pil = prev_window_frames[ref_local_idx]
```

5. `prepare_window_inputs()` 会把当前窗口的 `window_video[0]` 替换成这张 `reference_frame_pil`。
6. 当前窗口的前 `overlap` 张 mask 全部强制为黑：

```text
for i in range(overlap):
    window_masks[i] = black
```

7. 替换后的 `window_video[0]` 同时用于三处：

- `input_image`，也就是 CLIP image conditioning。
- `masked_video_tensor` 的第 0 帧，进入 masked video VAE encode。
- `video_tensor` 的第 0 帧，进入 raw video VAE encode。

8. 非首窗口提交结果时跳过 overlap 段：

```text
take_start = 0 if w_idx == 0 else overlap
take_end = valid_len
```

所以当前窗口 `local 0..overlap-1` 只作为上下文参与推理，不写回最终 `gen_by_idx`。

9. 尾窗口不足 81 帧时，原始实现使用 native reverse mirror 补齐：

```text
N-1, N-2, N-3, ...
```

## 例子

以 `infer_len=81, overlap=5` 为例：

```text
stride = 76

window 0: global 0..80
  输出 81 帧，记为 prev_window_frames[0..80]

window 1: 原始输入 global 76..156
  window_video[0] = prev_window_frames[76]
  window_video[1..4] = 原始 global 77..80
  window_masks[0..4] = black
  CLIP image = window_video[0] = prev_window_frames[76]
  VAE masked/raw video 的第 0 帧 = prev_window_frames[76]
  提交输出时跳过 local 0..4，只提交 local >= 5
```

这个例子是本次修改和验证的最小判定模型。

## sglang 当前相关实现

sglang 侧相关文件：

- `python/sglang/multimodal_gen/runtime/videoedit/windowing.py`
- `python/sglang/multimodal_gen/runtime/pipelines/wan_videoedit_pipeline.py`
- `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/videoedit_wan.py`
- `python/sglang/multimodal_gen/runtime/videoedit/preprocess.py`
- `python/sglang/multimodal_gen/runtime/videoedit/frame_provider.py`
- `python/sglang/multimodal_gen/configs/sample/videoedit_wan.py`

当前已基本对齐的部分：

- `build_videoedit_window_specs()` 的窗口起点逻辑与原始 `while next_start + overlap < total_frames` 等价。
- `tail_padding_mode="native_reverse_mirror"` 对齐原始尾窗口补帧。
- `_materialize_window_inputs()` 在 `native_skip` 模式下会把非首窗口 `frames[0]` 替换成上一窗口 `frames[stride]`。
- `VideoEditImageEncodingStage` 使用 `params.runtime_window_frames[0]` 做 CLIP。
- `VideoEditConditionEncodingStage` 使用 `params.runtime_window_frames/masks` 构造 masked/raw video VAE 输入。
- `_commit_window_output()` 在 `native_skip` 模式下跳过非首窗口 `local_idx < overlap` 的输出。

需要补强或明确的风险点：

1. reference 注入逻辑目前挂在 `overlap_commit_mode == "native_skip"` 分支下。原始实现没有 weighted commit 分支；为了“完全对齐”，native 对齐路径必须明确只使用 `native_skip`，并把 reference 注入作为 native contract 固定下来。
2. `use_repaired_context=True` 会让窗口输入中已提交过的帧来自累计修复结果。原始实现除 `window_video[0]` 外，其他 overlap 帧仍来自原始 resized video，只是 mask 被置黑。因此对齐模式必须禁用 `use_repaired_context`，或把它标记为非 native 扩展。
3. eager 预处理里 reference image 的 raw mask 当前先建成白色，再依赖 `expand_mask_frames(... force_zero i==0)` 归零；streaming provider 也先建白色再 `force_zero`。原始实现直接前置全黑 mask。建议改成显式黑 mask，避免后续重构破坏 reference mask 语义。
4. 当前 metadata 没有记录每个窗口的 reference 来源。对齐问题排查时应该记录 `reference_from_previous_window`、`reference_prev_local_idx`、`reference_global_index`、`zeroed_overlap_mask_count`。

## 修改计划

### 1. 固化 native 多窗口 contract

修改 `python/sglang/multimodal_gen/runtime/videoedit/contracts.py` 和 `windowing.py`：

- 在 `VideoEditWindowSpec` 增加调试字段：
  - `stride`
  - `reference_prev_local_idx`
  - `reference_global_index`
  - `overlap_mask_zero_count`
  - `commit_start_local_idx`
- `window_index == 0` 时这些 reference 字段为 `None`。
- `window_index > 0 and overlap > 0` 时：
  - `reference_prev_local_idx = infer_len - overlap`
  - `reference_global_index = start_index`
  - `overlap_mask_zero_count = overlap`
  - `commit_start_local_idx = overlap`
- `overlap == 0` 时：
  - 不从上一窗口取 reference。
  - `commit_start_local_idx = 0`。

这些字段不改变推理行为，只让代码和测试能直接表达原始 contract。

### 2. 重构窗口物化逻辑

修改 `python/sglang/multimodal_gen/runtime/pipelines/wan_videoedit_pipeline.py`：

- 把 `_materialize_window_inputs()` 拆成几个小函数：
  - `_build_native_window_frames()`
  - `_apply_previous_window_reference()`
  - `_apply_native_overlap_masks()`
- 对 native 对齐路径执行以下固定顺序：
  1. 按 `window_spec.input_indices` 读取 source frames/masks。
  2. 如果 `use_repaired_context=True`，仅在非 native 模式允许；native 对齐模式保持 `False`。
  3. 如果 `window_index > 0 and overlap > 0`，要求上一窗口 index 连续，并从 `runtime_prev_window_output_frames[stride]` 取 reference。
  4. 用该 reference 覆盖 `frames[0]`。
  5. 把 `masks[0:overlap]` 全部置黑。
  6. 赋值给 `params.runtime_window_frames/masks`。
- 如果 `overlap>0` 但上一窗口未连续执行，行为要与原始 `selected_chunks` 逻辑一致：不注入 reference，但仍按当前窗口 source frames 构造输入。这个情况只应在调试运行部分 chunks 时出现。
- 在 metadata 或 stage dump 中记录：
  - 当前窗口 `start_index/end_index/valid_len`
  - `reference_from_previous_window`
  - `reference_prev_local_idx`
  - `reference_global_index`
  - `zeroed_overlap_mask_count`

### 3. 明确 native 对齐参数

修改或文档化 `WanVideoEditSamplingParams`：

native parity 必须使用：

```text
overlap_commit_mode = native_skip
tail_padding_mode = native_reverse_mirror
use_repaired_context = false
init_latent_mode = noise
generator_device = cpu
vary_seed_by_window = false
mask_downsample_mode = nearest
use_clip = true
```

如果保留 `weighted` 和 `use_repaired_context`，需要明确它们是非原始模型行为，不参与“完全对齐”验收。

### 4. 统一 reference mask 语义

修改 `python/sglang/multimodal_gen/runtime/videoedit/preprocess.py`：

- `prepare_global_inputs()` 中有 `reference_image` 时，前置 reference mask 直接构造为全黑，而不是白色。

修改 `python/sglang/multimodal_gen/runtime/videoedit/frame_provider.py`：

- `_decode_one()`、`_reopen_decode_entry()`、`paste_back_frames()` 中 `global_index == 0 and reference_offset == 1` 时，raw mask 直接用全黑。
- 保留 `force_zero=global_index == 0`，作为双保险。

这样 eager 和 stream 两条路径都直接对齐原始 `prepare_global_inputs()` 的前置黑 mask 行为。

### 5. 增加单元测试

新增 `python/sglang/multimodal_gen/test/unit/test_videoedit_windowing.py`：

- 复刻原始 `infer.py` 的窗口起点生成逻辑，和 `build_videoedit_window_specs()` 对比。
- 覆盖：
  - `num_frames=81, overlap=0`
  - `num_frames=156, overlap=0`
  - `num_frames=156, overlap=5`
  - `num_frames=162, overlap=5`
  - 尾窗口 reverse mirror 补帧。
- 校验非首窗口：
  - `reference_prev_local_idx == infer_len - overlap`
  - `commit_start_local_idx == overlap`
  - `input_indices[0] == start_index`

新增或扩展 pipeline 级轻量测试：

- 构造假 `prev_window_output_frames`，每帧用不同颜色编码 local index。
- 调用 `_materialize_window_inputs()`。
- 断言窗口 1：
  - `runtime_window_frames[0]` 等于上一窗口 `local=stride` 的颜色。
  - `runtime_window_frames[1:overlap]` 没有被替换成上一窗口输出。
  - `runtime_window_masks[0:overlap]` 全黑。
  - `runtime_window_masks[overlap]` 保持原 mask。

扩展现有测试：

- `test_videoedit_mask_sources.py`：reference image 前置后，第 0 张 `raw/dilated/resized` mask 都应为黑。
- `test_videoedit_frame_provider.py`：带 reference image 时，stream provider materialize 出来的第 0 张 mask 与 eager 完全一致。

### 6. 增加 stage dump 对齐验证

用同一组参数分别跑原始 `VideoEdit-diffusers/infer.py` 和 sglang CLI，建议先用 `num_inference_steps=1` 缩短验证时间。

对齐参数重点：

```text
infer_len=81
overlap=5 或 10
use_clip=true
init_latent_mode=noise
generator_device=cpu
overlap_commit_mode=native_skip
tail_padding_mode=native_reverse_mirror
use_repaired_context=false
vary_seed_by_window=false
mask_downsample_mode=nearest
no teacache
no frame interpolation
no upscaling
```

重点比较 window 1 的 stage dump：

- window materialize metadata：
  - `start_index == 81 - overlap`
  - `reference_from_previous_window == true`
  - `reference_prev_local_idx == 81 - overlap`
- `window_frame_000`：sglang 必须等于原始 window 0 输出的 `local=stride`。
- `window_mask_000..window_mask_{overlap-1}`：必须全黑。
- `window_mask_{overlap}`：必须等于原始对应 mask。
- CLIP 输入和 `image_embeds`：必须使用替换后的 `window_frame_000`。
- `masked_video_tensor`、`video_tensor`、`cond_masks`、`cond_latents`：window 1 对齐。

最终再比较输出视频：

- 先比较 crop-only 输出，排除 paste-back 对差异的放大。
- 再比较 pasted output。
- 如果要与原始 `infer.py` 当前保存行为对齐，sglang CLI 使用 `--no-drop-reference-frame`；如果 API 业务要丢 reference 首帧，则比较时两边都显式去掉第 0 帧。

## 验收标准

1. 对任意 `overlap>0` 的非首窗口，sglang 的 reference 选取满足：

```text
current_window_frame_0 == previous_window_output_frame[infer_len - overlap]
```

2. 这张 reference frame 同时进入：

- CLIP image conditioning。
- masked video VAE input 的 frame 0。
- raw video VAE input 的 frame 0。

3. 非首窗口 `local 0..overlap-1` 的 mask 全黑。
4. 非首窗口最终提交时跳过 `local 0..overlap-1`。
5. eager decode 和 stream decode 在 reference image、窗口输入、mask 输入上保持一致。
6. `overlap_commit_mode=native_skip`、`use_repaired_context=false` 下，window materialize 和主要 stage tensor 能与原始实现逐窗口对齐。

## 建议实施顺序

1. 先加 `test_videoedit_windowing.py`，锁住窗口起点、tail padding、commit 起点。
2. 修改/重构 `_materialize_window_inputs()`，加 reference 注入和 overlap mask 的单元测试。
3. 统一 reference mask 为显式黑 mask，并补 eager/stream 一致性测试。
4. 增加 metadata 字段，方便实际跑 stage dump 时定位。
5. 跑 unit tests。
6. 跑一组 `num_inference_steps=1` 的原始 vs sglang stage dump 对比。
7. 再跑业务参数的完整视频对比。
