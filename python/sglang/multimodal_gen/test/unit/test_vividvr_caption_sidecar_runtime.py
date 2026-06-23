from sglang.multimodal_gen.runtime.vividvr.caption_sidecar_runtime import (
    CaptionClipResult,
    CaptionRequestMetrics,
    CaptionWorkerBatchResult,
    assign_clip_indices_round_robin,
    merge_caption_results_in_clip_order,
)


def test_assign_clip_indices_round_robin_balances_by_clip_order():
    assignments = assign_clip_indices_round_robin([0, 1, 2, 3, 4], num_workers=2)

    assert assignments == {
        0: [0, 2, 4],
        1: [1, 3],
    }


def test_merge_caption_results_in_clip_order_sorts_across_workers():
    merged = merge_caption_results_in_clip_order(
        [
            CaptionWorkerBatchResult(
                worker_index=1,
                clip_results=[
                    CaptionClipResult(
                        clip_index=3,
                        caption="clip-3",
                        worker_index=1,
                    ),
                    CaptionClipResult(
                        clip_index=1,
                        caption="clip-1",
                        worker_index=1,
                    ),
                ],
            ),
            CaptionWorkerBatchResult(
                worker_index=0,
                clip_results=[
                    CaptionClipResult(
                        clip_index=2,
                        caption="clip-2",
                        worker_index=0,
                    ),
                    CaptionClipResult(
                        clip_index=0,
                        caption="clip-0",
                        worker_index=0,
                    ),
                ],
            ),
        ]
    )

    assert [item.clip_index for item in merged] == [0, 1, 2, 3]
    assert [item.caption for item in merged] == [
        "clip-0",
        "clip-1",
        "clip-2",
        "clip-3",
    ]


def test_caption_request_metrics_to_response_dict_uses_stable_transport_shape():
    metrics = CaptionRequestMetrics(
        request_id="req-42",
        total_clip_count=5,
        worker_count=2,
        assigned_clip_indices_by_worker={
            0: (0, 2, 4),
            1: (1, 3),
        },
        read_seconds=0.4,
        write_seconds=0.02,
        total_seconds=1.2,
        worker_batches=[
            CaptionWorkerBatchResult(
                worker_index=0,
                total_seconds=0.6,
                clip_results=[
                    CaptionClipResult(
                        clip_index=0,
                        caption="clip-0",
                        worker_index=0,
                        inference_seconds=0.2,
                        total_seconds=0.25,
                    )
                ],
            )
        ],
    )

    response = metrics.to_response_dict()

    assert response == {
        "request_id": "req-42",
        "total_clip_count": 5,
        "worker_count": 2,
        "assigned_clip_indices_by_worker": {
            "0": [0, 2, 4],
            "1": [1, 3],
        },
        "timing": {
            "read_seconds": 0.4,
            "write_seconds": 0.02,
            "total_seconds": 1.2,
            "worker_batches": [
                {
                    "worker_index": 0,
                    "total_seconds": 0.6,
                    "clip_results": [
                        {
                            "clip_index": 0,
                            "inference_seconds": 0.2,
                            "total_seconds": 0.25,
                        }
                    ],
                }
            ],
        },
    }

    response["assigned_clip_indices_by_worker"]["0"].append(99)
    assert metrics.assigned_clip_indices_by_worker[0] == (0, 2, 4)
