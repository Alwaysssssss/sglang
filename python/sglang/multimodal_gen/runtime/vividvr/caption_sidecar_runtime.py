# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class CaptionClipResult:
    clip_index: int
    caption: str
    worker_index: int
    inference_seconds: float | None = None
    total_seconds: float | None = None


@dataclass(frozen=True)
class CaptionWorkerBatchResult:
    worker_index: int
    clip_results: list[CaptionClipResult]
    total_seconds: float | None = None


@dataclass(frozen=True)
class CaptionRequestMetrics:
    request_id: str
    total_clip_count: int
    worker_count: int
    assigned_clip_indices_by_worker: dict[int, tuple[int, ...]]
    read_seconds: float | None = None
    write_seconds: float | None = None
    total_seconds: float | None = None
    worker_batches: list[CaptionWorkerBatchResult] = field(default_factory=list)

    def to_response_dict(self) -> dict[str, object]:
        sorted_worker_batches = sorted(
            self.worker_batches,
            key=lambda worker_batch: worker_batch.worker_index,
        )
        return {
            "request_id": self.request_id,
            "total_clip_count": self.total_clip_count,
            "worker_count": self.worker_count,
            "assigned_clip_indices_by_worker": {
                str(worker_index): list(clip_indices)
                for worker_index, clip_indices in self.assigned_clip_indices_by_worker.items()
            },
            "timing": {
                "read_seconds": self.read_seconds,
                "write_seconds": self.write_seconds,
                "total_seconds": self.total_seconds,
                "worker_batches": [
                    {
                        "worker_index": worker_batch.worker_index,
                        "total_seconds": worker_batch.total_seconds,
                        "clip_results": [
                            {
                                "clip_index": clip_result.clip_index,
                                "inference_seconds": clip_result.inference_seconds,
                                "total_seconds": clip_result.total_seconds,
                            }
                            for clip_result in worker_batch.clip_results
                        ],
                    }
                    for worker_batch in sorted_worker_batches
                ],
            },
        }


def assign_clip_indices_round_robin(
    clip_indices: Sequence[int],
    *,
    num_workers: int,
) -> dict[int, list[int]]:
    if num_workers <= 0:
        raise ValueError(f"num_workers must be positive, got {num_workers}")

    assignments = {worker_index: [] for worker_index in range(num_workers)}
    for assignment_index, clip_index in enumerate(clip_indices):
        assignments[assignment_index % num_workers].append(clip_index)
    return assignments


def merge_caption_results_in_clip_order(
    worker_results: Sequence[CaptionWorkerBatchResult],
) -> list[CaptionClipResult]:
    merged_results = [
        clip_result
        for worker_result in worker_results
        for clip_result in worker_result.clip_results
    ]
    return sorted(merged_results, key=lambda item: item.clip_index)
