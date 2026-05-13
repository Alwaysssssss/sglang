# VideoEdit 支持 `num_frames=-1` 跑全视频的修改计划

## 背景

现在如果想编辑全部帧，需要先查输入视频帧数，然后在请求里手动写：

```json
"num_frames": 156
```

这不方便，而且容易写错。希望支持：

```json
"num_frames": -1
```

语义为：读取输入视频和 mask 的全部可用帧。

结论：可以做，而且建议只在 VideoEdit 入口层做，不改全局 `SamplingParams` 的正整数语义。

## 当前行为

当前链路：

- API 请求模型：`python/sglang/multimodal_gen/runtime/entrypoints/openai/protocol.py`
  - `VideoRepairRequest.num_frames: int = 81`
- API 创建 sampling params：`python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py`
  - 直接把 `req.num_frames` 传给 `WanVideoEditSamplingParams`
- CLI 参数：`python/sglang/multimodal_gen/runtime/videoedit/cli.py`
  - `--num-frames` 默认 `81`
  - 直接把 `args.num_frames` 传给 `WanVideoEditSamplingParams`
- VideoEdit sampling params：`python/sglang/multimodal_gen/configs/sample/videoedit_wan.py`
  - `num_frames` 必须大于 0
- 基类 sampling params：`python/sglang/multimodal_gen/configs/sample/sampling_params.py`
  - `__post_init__` 里有 `assert self.num_frames >= 1`
  - `_validate()` 里也要求 `num_frames` 是正整数
- 底层读取视频：`python/sglang/multimodal_gen/runtime/videoedit/preprocess.py`
  - `load_video_frames(video_path, num_frames=None)` 已经支持读到 EOF

所以现在直接传 `-1` 会在 sampling params 阶段被拒绝。

## 推荐方案

在 VideoEdit API/CLI 入口把 `num_frames=-1` 转成真实正整数，再进入 `WanVideoEditSamplingParams`。

不要让 `-1` 或 `None` 进入通用 `SamplingParams`，原因：

- `SamplingParams` 被很多图像/视频模型共用，放宽校验会扩大影响面。
- VideoEdit 后续窗口规划需要明确整数帧数。
- `prepare_global_inputs()` 会同时读取 video 和 mask，实际可处理帧数应取二者最小值。

## 用户语义

支持：

```json
"num_frames": -1
```

表示：

```text
num_frames = min(input_video_frame_count, input_mask_frame_count)
```

如果视频和 mask 帧数不同，使用较短者，避免越界。

保留默认行为：

```json
不传 num_frames -> 仍然默认 81
```

这样不会改变已有 benchmark 和脚本。

## 修改点

### 1. 增加帧数探测 helper

建议新增到：

```text
python/sglang/multimodal_gen/runtime/videoedit/preprocess.py
```

新增函数：

```python
def probe_video_frame_count(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if count <= 0:
        count = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            count += 1
    cap.release()
    if count <= 0:
        raise RuntimeError(f"No frames found in video file: {video_path}")
    return count
```

再加一个 VideoEdit 专用解析函数：

```python
def resolve_videoedit_num_frames(
    requested_num_frames: int,
    video_input_path: str,
    mask_input_path: str,
) -> int:
    if requested_num_frames != -1:
        return requested_num_frames

    video_frames = probe_video_frame_count(video_input_path)
    mask_frames = probe_video_frame_count(mask_input_path)
    resolved = min(video_frames, mask_frames)
    if resolved <= 0:
        raise RuntimeError(
            f"Could not resolve full-video frame count: "
            f"video={video_frames}, mask={mask_frames}"
        )
    return resolved
```

也可以把这两个 helper 放到新文件：

```text
python/sglang/multimodal_gen/runtime/videoedit/frame_count.py
```

但当前 `preprocess.py` 已经依赖 `cv2` 并包含 `load_video_frames()`，放在那里更直接。

### 2. API 入口使用 helper

修改：

```text
python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py
```

在 `create_video_repair()` 中，`video_input_path` 和 `mask_input_path` 都解析完成之后、构造 `WanVideoEditSamplingParams` 之前，加入：

```python
from sglang.multimodal_gen.runtime.videoedit.preprocess import (
    resolve_videoedit_num_frames,
)

resolved_num_frames = resolve_videoedit_num_frames(
    req.num_frames,
    video_input_path,
    mask_input_path,
)
```

然后把：

```python
num_frames=req.num_frames,
```

改成：

```python
num_frames=resolved_num_frames,
```

同时 `_video_repair_job_from_sampling()` 现在拿的是 `sampling_params`，最终 job 里会反映解析后的实际帧数。

### 3. CLI 入口使用 helper

修改：

```text
python/sglang/multimodal_gen/runtime/videoedit/cli.py
```

在 `repair_cmd()` 构造 `WanVideoEditSamplingParams` 之前加入：

```python
from sglang.multimodal_gen.runtime.videoedit.preprocess import (
    resolve_videoedit_num_frames,
)

resolved_num_frames = resolve_videoedit_num_frames(
    args.num_frames,
    args.video_input_path,
    args.mask_input_path,
)
```

然后把：

```python
num_frames=args.num_frames,
```

改成：

```python
num_frames=resolved_num_frames,
```

### 4. 错误提示更友好

修改：

```text
python/sglang/multimodal_gen/configs/sample/videoedit_wan.py
```

如果最后仍然有 `num_frames <= 0` 进入 validation，错误提示可以改得更明确：

```python
if self.num_frames is not None and self.num_frames <= 0:
    raise ValueError(
        "VideoEdit num_frames must be positive after request normalization. "
        "Use num_frames=-1 only at API/CLI entrypoints to mean all frames."
    )
```

这个不是必须，但有助于排查。

### 5. 文档更新

更新已有文档：

```text
docs_tyx/videoedit_over_81_frames.md
docs_tyx/videoedit_sp1_offload_serve_runbook.md
```

新增说明：

```text
num_frames=-1 表示读取 video 和 mask 的全部可用帧，实际使用 min(video_frames, mask_frames)。
```

示例请求：

```bash
curl --noproxy '*' -s -X POST http://127.0.0.1:30000/v1/videos/repairs \
  -H 'Content-Type: application/json' \
  -d '{
    "task_id": "sp1_offload_all_frames",
    "prompt": "A close-up of an orange flower with a yellow center, remaining in focus against a blurred green grass background throughout the video.",
    "video_input_path": "/home/tyx/workspace/zhouhao6/video_diffusers/pexel_test_data_0410/videos/15108907_3840_2160_50fps_short.mp4",
    "mask_input_path": "/home/tyx/workspace/zhouhao6/video_diffusers/pexel_test_data_0410/masks/15108907_3840_2160_50fps_No_bbox_mask.mp4",
    "output_storage": "local",
    "output_path": "/home/tyx/workspace/zhouhao6/sglang/outputs/15108907_3840_2160_50fps_api_sp1_offload_all_frames.mp4",
    "num_frames": -1,
    "infer_len": 81,
    "overlap": 0,
    "num_inference_steps": 20,
    "guidance_scale": 5.0,
    "dynamic_cfg": true,
    "dynamic_cfg_max_step": 15,
    "seed": 42,
    "dtype": "bf16",
    "enable_paste_back": true,
    "drop_reference_frame": false,
    "perf_dump_path": "/home/tyx/workspace/zhouhao6/sglang/outputs/videoedit_perf_api_sp1_offload_all_frames.json"
  }'
```

CLI 示例：

```bash
python -m sglang.multimodal_gen.runtime.videoedit.cli repair \
  --model-path "$MODEL_PATH" \
  --transformer-path "$TRANSFORMER_PATH" \
  --prompt "$PROMPT" \
  --video-input-path "$INPUT_VIDEO" \
  --mask-input-path "$INPUT_MASK" \
  --output-path "$OUT_DIR" \
  --output-file-name 15108907_3840_2160_50fps_cli_sp1_offload_all_frames.mp4 \
  --num-gpus 1 \
  --sp-degree 1 \
  --ulysses-degree 1 \
  --ring-degree 1 \
  --dit-cpu-offload \
  --dit-layerwise-offload \
  --text-encoder-cpu-offload \
  --image-encoder-cpu-offload \
  --vae-cpu-offload \
  --num-frames -1 \
  --infer-len 81 \
  --overlap 0 \
  --num-inference-steps 20 \
  --guidance-scale 5.0 \
  --dynamic-cfg \
  --dynamic-cfg-max-step 15 \
  --seed 42 \
  --dtype bf16 \
  --enable-paste-back \
  --no-drop-reference-frame \
  --warmup \
  --warmup-steps 1 \
  --perf-dump-path "$OUT_DIR/videoedit_perf_cli_sp1_offload_all_frames.json"
```

## 翻转/反射补齐解释

模型每个窗口固定吃 `81` 帧。全视频帧数通常不是 81 的整数倍，因此最后一个窗口可能不足 81 帧。

例子：当前视频 156 帧，`infer_len=81, overlap=0`。

窗口切分：

```text
window 0: 0..80，共 81 帧
window 1: 81..155，共 75 帧，还差 6 帧
```

为了让第二个窗口也达到 81 帧，会从结尾往回取帧补齐：

```text
真实帧: 81, 82, ..., 155
补齐帧: 154, 153, 152, 151, 150, 149
```

这就是“反射补齐”或“翻转填充”。

注意：补齐帧只用于模型输入凑够 81 帧，不会作为额外新帧写进最终视频。最终提交的仍然只是真实输入帧。

如果 `drop_reference_frame=false`，当前 156 帧输入最终应输出 156 帧。

如果 `drop_reference_frame=true`，最终会丢第 0 帧参考帧，输出 155 帧。

## 测试计划

### 单元测试

新增或扩展测试：

```text
python/sglang/multimodal_gen/test/unit/
```

建议覆盖：

1. `resolve_videoedit_num_frames(81, video, mask)` 返回 `81`。
2. `resolve_videoedit_num_frames(-1, video_156, mask_156)` 返回 `156`。
3. video 156 帧、mask 100 帧时返回 `100`。
4. 不存在的视频路径抛出 `FileNotFoundError`。
5. `WanVideoEditSamplingParams(num_frames=-1)` 仍然报错，确保只有入口层支持 `-1`。

### API 轻量测试

不需要跑完整推理，可以 mock `resolve_videoedit_num_frames()` 或构造小视频，确认：

```json
"num_frames": -1
```

最终传给 `WanVideoEditSamplingParams` 的是解析后的正整数。

### 手动冒烟

用当前输入视频提交：

```json
"num_frames": -1,
"infer_len": 81,
"overlap": 0,
"drop_reference_frame": false
```

预期：

```text
metadata.num_input_frames = 156
window_specs = [
  {"window_index": 0, "start_index": 0, "end_index": 81, "reflected_count": 0},
  {"window_index": 1, "start_index": 81, "end_index": 156, "reflected_count": 6}
]
输出视频 frames = 156
```

## 风险和注意事项

- 不建议把 API 默认 `num_frames` 从 `81` 改成 `-1`。默认改成全视频会让历史 benchmark 变慢，并改变已有行为。
- `num_frames=-1` 对长视频可能非常慢；文档里要提醒用户先确认输入长度和预计窗口数。
- 若输入 video 和 mask 帧数不一致，使用较短者是更安全的行为，但需要在日志里记录。
- 对 100 帧、156 帧这类长视频，质量 reference 也应该按同样帧数重新生成；不能继续用 80 帧 reference 做正式 compare。
