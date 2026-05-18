# SPDX-License-Identifier: Apache-2.0
from sglang.multimodal_gen.runtime.videoedit.contracts import VideoEditWindowSpec

# 反射补帧，返回对应帧的索引
def _reflect_index(index: int, num_frames: int) -> int:
    if num_frames <= 1:
        return 0
    if index < num_frames:
        return index
    period = 2 * num_frames - 2
    mod = index % period
    return mod if mod < num_frames else period - mod


def build_videoedit_window_specs(
    num_frames: int,
    infer_len: int = 81,
    overlap: int = 0,
) -> list[VideoEditWindowSpec]:
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if infer_len <= 0:
        raise ValueError(f"infer_len must be positive, got {infer_len}")
    if not (0 <= overlap < infer_len):
        raise ValueError(f"overlap must be in [0, {infer_len}), got {overlap}")

    stride = infer_len - overlap
    # 创建窗口的起始索引列表
    starts: list[int] = [0]
    while starts[-1] + infer_len < num_frames:
        starts.append(starts[-1] + stride)

    specs: list[VideoEditWindowSpec] = []
    for window_index, start_index in enumerate(starts):
        raw_indices = list(range(start_index, start_index + infer_len))
        input_indices = [_reflect_index(i, num_frames) for i in raw_indices]
        commit = {
            local_idx: global_idx
            for local_idx, global_idx in enumerate(raw_indices)
            if global_idx < num_frames
        }
        reflected_count = sum(1 for i in raw_indices if i >= num_frames)
        specs.append(
            VideoEditWindowSpec(
                window_index=window_index,
                start_index=start_index,
                end_index=min(start_index + infer_len, num_frames),
                input_indices=input_indices,
                commit_local_to_global=commit,
                reflected_count=reflected_count,
            )
        )
    return specs

