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


def _native_reverse_mirror_index(index: int, num_frames: int) -> int:
    if num_frames <= 1:
        return 0
    if index < num_frames:
        return index
    return max(num_frames - 1 - (index - num_frames), 0)


def build_videoedit_window_specs(
    num_frames: int,
    infer_len: int = 81,
    overlap: int = 0,
    tail_padding_mode: str = "reflect",
    overlap_commit_mode: str = "native_skip",
) -> list[VideoEditWindowSpec]:
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if infer_len <= 0:
        raise ValueError(f"infer_len must be positive, got {infer_len}")
    if not (0 <= overlap < infer_len):
        raise ValueError(f"overlap must be in [0, {infer_len}), got {overlap}")
    if tail_padding_mode not in {"native_reverse_mirror", "reflect"}:
        raise ValueError(
            "tail_padding_mode must be one of native_reverse_mirror/reflect, "
            f"got {tail_padding_mode!r}"
        )
    if overlap_commit_mode not in {"native_skip", "weighted"}:
        raise ValueError(
            "overlap_commit_mode must be one of native_skip/weighted, "
            f"got {overlap_commit_mode!r}"
        )
    if overlap_commit_mode == "weighted" and overlap >= infer_len - 1:
        raise ValueError(
            "weighted overlap_commit_mode requires overlap < infer_len - 1, "
            f"got overlap={overlap}, infer_len={infer_len}"
        )

    stride = infer_len - overlap
    weighted_stride = infer_len - overlap - 1
    # Match VideoEdit-diffusers/infer.py window start generation exactly.
    if num_frames <= infer_len:
        starts: list[int] = [0]
    else:
        starts = [0]
        next_start = stride
        while next_start + overlap < num_frames:
            starts.append(next_start)
            if overlap_commit_mode == "weighted" and overlap > 0:
                next_start += weighted_stride
            else:
                next_start += stride

    specs: list[VideoEditWindowSpec] = []
    for window_index, start_index in enumerate(starts):
        uses_previous_reference = window_index > 0 and overlap > 0
        if overlap_commit_mode == "weighted" and uses_previous_reference:
            raw_indices = [start_index - 1] + list(
                range(start_index, start_index + infer_len - 1)
            )
        else:
            raw_indices = list(range(start_index, start_index + infer_len))
        if tail_padding_mode == "native_reverse_mirror":
            input_indices = [
                _native_reverse_mirror_index(i, num_frames) for i in raw_indices
            ]
        else:
            input_indices = [_reflect_index(i, num_frames) for i in raw_indices]
        commit = {
            local_idx: global_idx
            for local_idx, global_idx in enumerate(raw_indices)
            if global_idx < num_frames
        }
        reflected_count = sum(1 for i in raw_indices if i >= num_frames)
        if uses_previous_reference and overlap_commit_mode == "weighted":
            reference_prev_local_idx = weighted_stride
            reference_global_index = start_index - 1
            overlap_mask_zero_count = 1
            commit_start_local_idx = 1
        elif uses_previous_reference:
            reference_prev_local_idx = stride
            reference_global_index = start_index
            overlap_mask_zero_count = overlap
            commit_start_local_idx = overlap
        else:
            reference_prev_local_idx = None
            reference_global_index = None
            overlap_mask_zero_count = 0
            commit_start_local_idx = 0
        specs.append(
            VideoEditWindowSpec(
                window_index=window_index,
                start_index=start_index,
                end_index=min(start_index + infer_len, num_frames),
                input_indices=input_indices,
                commit_local_to_global=commit,
                valid_len=max(0, min(start_index + infer_len, num_frames) - start_index),
                reflected_count=reflected_count,
                stride=stride,
                reference_prev_local_idx=reference_prev_local_idx,
                reference_global_index=reference_global_index,
                overlap_mask_zero_count=overlap_mask_zero_count,
                commit_start_local_idx=commit_start_local_idx,
            )
        )
    return specs
