# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass


@dataclass(frozen=True)
class VideoEditWindowSpec:
    window_index: int
    start_index: int
    end_index: int
    input_indices: list[int]
    commit_local_to_global: dict[int, int]
    reflected_count: int = 0

