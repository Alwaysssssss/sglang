# VideoEdit 支持 NPY/COCO Mask 输入的修改方案

## 背景

当前 VideoEdit 修复流程默认 `mask_input_path` 是一个视频文件，例如 mp4。实际对接方后续可能传入：

- Numpy mask 文件：`.npy` / `.npz`
- COCO RLE JSON mask 文件
- 仍然可能传入原来的 mask 视频

目标是让这些 mask 输入都能参与现有模型流程，尽量不改模型主体、窗口逻辑、denoise stage 和 paste-back 后处理。

参考文件：

- `docs_tyx/coco/video_info_coco.py`：COCO RLE mask 读取方式。
- `docs_tyx/coco/video_info_ffmpeg.py`：用 FFmpeg 读取视频元信息和帧。
- `docs_tyx/coco/video_stream_writer_ffmpeg.py`：参考原视频 profile 写出视频。

当前真正使用 mask 的入口主要在：

- `python/sglang/multimodal_gen/runtime/videoedit/preprocess.py`
- `python/sglang/multimodal_gen/runtime/pipelines/wan_videoedit_pipeline.py`
- `python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py`
- `python/sglang/multimodal_gen/runtime/videoedit/cli.py`

## 总体设计

新增一个 mask 输入适配层，把不同来源的 mask 统一转成现有 preprocess 已经能处理的 `list[PIL.Image.Image]` 灰度帧。

统一输出约定：

- 每帧是 PIL `L` 模式。
- 背景为 `0`，前景为 `255`。
- 帧序按输入顺序或 COCO `frame` 字段排序。
- 多对象 mask 默认做 union，生成单通道 mask。

现有后续流程保持不变：

- `expand_mask_frames()`
- `get_mask_bbox()`
- `crop_frames()`
- `prepare_window_inputs()`
- `paste_back()`

这样模型看到的仍然是 mask frames，不需要知道 mask 原始格式。

## 第一阶段：新增 mask_io.py

新增文件：

```text
python/sglang/multimodal_gen/runtime/videoedit/mask_io.py
```

建议提供这些接口：

```python
def probe_mask_frame_count(mask_path: str) -> int:
    ...

def load_mask_frames(
    mask_path: str,
    num_frames: int | None = None,
    target_size: tuple[int, int] | None = None,
) -> list[Image.Image]:
    ...
```

`target_size` 使用 PIL 的 `(width, height)`。如果 mask 尺寸和原视频不同，可以在 loader 里用 `Image.NEAREST` resize 到原视频尺寸，避免后续 bbox 和 paste-back 尺寸不一致。

### 格式识别策略

不要只依赖文件后缀，因为远程 `mask_url` 下载后当前可能会被保存成 `{request_id}_mask.mp4`。

建议按内容 magic 优先识别：

- NPY：文件头以 `b"\x93NUMPY"` 开始。
- NPZ：zip magic `b"PK\x03\x04"`，再用 `np.load()` 尝试读取。
- JSON/COCO：跳过空白后第一个字符是 `{` 或 `[`。
- 其他情况 fallback 为视频 mask，用当前 `load_video_frames()`。

这样即使远程 `.npy` 被临时保存成 `.mp4`，也能正确进入 numpy loader。

### Numpy mask 支持范围

建议支持以下 shape：

- `(T, H, W)`
- `(T, H, W, 1)`
- `(T, H, W, 3)`：转灰度或取 channel max。
- `(T, N, H, W)`：多对象 union。
- `(N, T, H, W)`：如果能通过参数或启发式识别，再支持；第一版可先不支持，报明确错误。

建议支持 dtype：

- `bool`
- `uint8`
- `float32` / `float64`
- 其他数值类型

二值化规则：

- `bool`：True 为前景。
- 整数：`> 0` 为前景。
- 浮点：默认 `> 0.5` 为前景；如果最大值大于 1，也可以按 `> 0` 处理或先归一化。建议第一版明确使用 `> 0.5`，并在错误信息里提示格式。

`.npz` key 读取优先级：

```text
masks -> mask -> arr_0 -> 第一个 array
```

如果 `np.load(..., allow_pickle=True)` 得到 dict/object，可以兼容：

- `{"masks": ndarray}`
- `{"mask": ndarray}`
- `{"video_segments": ...}`，但这一类结构复杂，建议放第二阶段。

### COCO JSON 支持范围

优先支持当前已有代码生成的结构：

```json
[
  {
    "frame": 0,
    "size": [height, width],
    "counts": [
      {"object_id": 1, "mask": "rle_counts"},
      {"object_id": 2, "mask": "rle_counts"}
    ]
  }
]
```

读取逻辑：

1. 读取 JSON list。
2. 按 `frame` 升序排序。
3. 对每帧的 `counts` 逐个 `pycocotools.mask.decode()`。
4. 多对象做 OR union。
5. 转成 `L` 模式 PIL mask。

注意点：

- `size` 是 `[height, width]`，PIL resize 目标是 `(width, height)`。
- RLE `counts` 如果是字符串，需要按 pycocotools 接口传入。
- 如果某帧没有 objects，返回全 0 mask。

后续如果对方传标准 COCO annotations 格式，再增加一个 parser 分支：

```json
{
  "images": [...],
  "annotations": [...]
}
```

第一版不建议把这个复杂格式混进来，先把当前对接格式跑通。

## 第二阶段：preprocess.py 最小改动

当前逻辑：

```python
def resolve_videoedit_num_frames(...):
    video_frames = probe_video_frame_count(video_input_path)
    mask_frames = probe_video_frame_count(mask_input_path)

def prepare_global_inputs(...):
    original_frames, fps = load_video_frames(input_video, num_frames)
    raw_mask_frames, _ = load_video_frames(mask_video, num_frames)
```

建议改为：

```python
from sglang.multimodal_gen.runtime.videoedit.mask_io import (
    load_mask_frames,
    probe_mask_frame_count,
)
```

```python
def resolve_videoedit_num_frames(...):
    video_frames = probe_video_frame_count(video_input_path)
    mask_frames = probe_mask_frame_count(mask_input_path)
```

```python
def prepare_global_inputs(...):
    original_frames, fps = load_video_frames(input_video, num_frames)
    raw_mask_frames = load_mask_frames(
        mask_video,
        num_frames=num_frames,
        target_size=original_frames[0].size,
    )
```

其余逻辑不动。

收益：

- 本地 CLI 可以直接传 `--mask-input-path xxx.npy` 或 `xxx.json`。
- API 可以直接传 `"mask_input_path": "/path/to/mask.npy"`。
- `num_frames=-1` 会正确取 `min(video_frames, mask_frames)`。

## 第三阶段：远程 mask_url 的兼容

当前 `video_api.py` 的 `_save_video_source_to_path()` 对 HTTP URL 在 target 没后缀时默认补 `.mp4`。

第一阶段可以不改 `video_api.py`，因为 `mask_io.py` 会按文件内容识别格式；即使 `.npy` 被保存成 `.mp4`，也能用 magic 读出。

如果要让临时文件名也正确，后续可以做很小改动：

```python
async def _save_media_source_to_path(
    source: str,
    target_path: str,
    default_ext: str = ".mp4",
) -> str:
    ...
```

调用时：

```python
video_input_path = await _save_media_source_to_path(
    req.video_url,
    os.path.join(uploads_dir, f"{request_id}_video"),
    default_ext=".mp4",
)

mask_input_path = await _save_media_source_to_path(
    req.mask_url,
    os.path.join(uploads_dir, f"{request_id}_mask"),
    default_ext="",
)
```

mask 的后缀推断优先级：

1. URL path 自带后缀：`.mp4` / `.npy` / `.npz` / `.json`
2. HTTP `Content-Type`
3. 内容 magic
4. fallback `.bin`

不建议第一版新增 `mask_format` 请求字段，除非对方 URL 完全没有后缀、Content-Type 也不可控，并且不希望按内容识别。

## 第四阶段：保持原视频格式和画质

mask 输入适配本身不需要把 `.npy/.json` 转成 mask mp4，因此不会引入 mask 视频压缩损伤。

但当前最终输出视频保存路径仍然走通用 `imageio.mimsave(... codec="libx264")`，这不能严格保持原视频 codec、pix_fmt、bitrate、color metadata。

如果要求输出视频尽量保持原视频格式和画质，需要增加 FFmpeg 写出适配。

新增文件建议：

```text
python/sglang/multimodal_gen/runtime/videoedit/ffmpeg_io.py
```

核心接口：

```python
def probe_video_profile(video_path: str) -> dict:
    ...

def save_video_frames_like_reference(
    frames: list[Image.Image] | list[np.ndarray],
    output_path: str,
    refer_file: str,
    fps: float | None = None,
) -> str:
    ...
```

实现参考：

- `docs_tyx/coco/video_info_ffmpeg.py`
- `docs_tyx/coco/video_stream_writer_ffmpeg.py`

需要尽量复用原视频：

- fps
- codec
- pix_fmt
- bit_rate
- color_space
- color_transfer
- color_primaries
- field_order
- prores profile

注意：

- 模型输出是新生成画面，不可能和原视频 bit-identical。
- “保持画质”实际含义是避免额外低质量转码，尽量保留原视频编码 profile 和颜色元数据。
- 如果原视频是特殊 codec 或高位深，需要 FFmpeg writer fallback 到安全编码参数并记录 warning。

最小落点有两个选择：

1. 只在 VideoEdit pipeline 内部保存最终结果，直接返回 `OutputBatch(output_file_paths=[path])`。
2. 改通用 `save_outputs()`，当 sampling params 带 `video_input_path` 时使用参考视频写出。

推荐第 1 种，影响面更小，只作用于 VideoEdit。

## 建议落地顺序

1. 新增 `mask_io.py`，先只支持本地 `.npy/.npz`、当前 COCO JSON、原视频 mask。
2. 修改 `preprocess.py` 的两个 mask 读取点。
3. 增加 unit test，覆盖：
   - `.npy` `(T,H,W)`。
   - `.npy` `(T,H,W,1)`。
   - `.npz` `masks` key。
   - 当前 COCO RLE JSON。
   - `num_frames=-1` 时 video/mask 取 min。
4. 用 CLI 跑一条小样例，确认 bbox、窗口、paste-back 不报错。
5. 再处理远程 `mask_url` 文件名和内容识别。
6. 最后单独做 FFmpeg 参考原视频写出，避免和 mask 输入适配混在一个风险点里。

## 需要避免的方案

不建议第一版把 `.npy/.json` 先转成临时 mask mp4 再交给当前流程。

原因：

- mask 是二值语义数据，视频编码会引入边缘压缩噪声。
- 需要额外考虑 codec、bitrate、pix_fmt，风险比直接读数组高。
- 当前后续流程本来就消费 PIL mask frames，没必要绕回视频。

## 验收标准

功能验收：

- 原来的 mp4 mask 输入行为不变。
- 本地 `.npy/.npz` mask 能通过 CLI/API 进入模型。
- 当前 COCO RLE JSON 能通过 CLI/API 进入模型。
- `num_frames=-1` 对非视频 mask 正常生效。
- mask 尺寸和视频尺寸不一致时，能 nearest resize 到视频尺寸，或报出清晰错误。

质量验收：

- mask loader 输出只有 0/255，不产生中间灰度。
- 多对象 COCO mask union 后 bbox 正常。
- FFmpeg 写出阶段完成后，输出视频 fps、codec、pix_fmt、颜色元数据尽量与原视频一致。

## 风险点

- 对方 NPY shape 未固定，需要先约定或在 loader 里做严格错误提示。
- COCO 标准 annotations 格式和当前内部 RLE list 格式不同，第一版建议只支持当前已确认格式。
- 远程 URL 没后缀时，必须靠内容识别，不能只看临时文件名。
- 输出视频“保持原格式和画质”不是 mask 输入问题，需要单独改保存链路。
