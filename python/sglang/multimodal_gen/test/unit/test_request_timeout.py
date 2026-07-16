import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import sglang.multimodal_gen.runtime.entrypoints.openai.video_api as video_api_mod
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.openai.stores import AsyncDictStore
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import process_generation_batch
from sglang.multimodal_gen.runtime.entrypoints.openai.video_api import _dispatch_job_async
from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker
from sglang.multimodal_gen.runtime.managers.scheduler import Scheduler
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.request_timeout import (
    TASK_TIMEOUT_MESSAGE,
    TaskTimeoutError,
    check_request_timeout,
    request_timeout_deadline,
)


def _expired_req(request_id="expired"):
    return Req(
        sampling_params=SamplingParams(
            request_id=request_id,
            request_timeout_deadline=time.monotonic() - 1.0,
        )
    )


class TestRequestTimeout(unittest.TestCase):
    def test_request_timeout_deadline_defaults_to_none_for_no_limit(self):
        self.assertIsNone(request_timeout_deadline(-1))

    def test_check_request_timeout_raises_task_timeout(self):
        req = _expired_req()
        with self.assertRaisesRegex(TaskTimeoutError, TASK_TIMEOUT_MESSAGE):
            check_request_timeout(req)

    def test_scheduler_does_not_send_expired_request_to_worker(self):
        scheduler = object.__new__(Scheduler)
        scheduler.worker = MagicMock()

        output = scheduler._handle_generation([_expired_req("queued-timeout")])

        self.assertEqual(output.error, TASK_TIMEOUT_MESSAGE)
        scheduler.worker.execute_forward.assert_not_called()

    def test_gpu_worker_returns_timeout_error_without_running_pipeline(self):
        worker = object.__new__(GPUWorker)
        worker.rank = 0
        worker.pipeline = MagicMock()
        worker.server_args = SimpleNamespace()

        with patch(
            "sglang.multimodal_gen.runtime.managers.gpu_worker.torch.cuda.is_initialized",
            return_value=False,
        ):
            output = worker.execute_forward([_expired_req("running-timeout")])

        self.assertEqual(output.error, TASK_TIMEOUT_MESSAGE)
        worker.pipeline.forward.assert_not_called()


    def test_late_success_does_not_overwrite_timeout_failed_job(self):
        store = AsyncDictStore()
        upload_calls = []

        class FakeStorage:
            async def upload_and_cleanup(self, *args, **kwargs):
                upload_calls.append((args, kwargs))
                return "s3://bucket/result.mp4"

        async def fake_process_generation_batch(_client, batch):
            return ["/tmp/result.mp4"], OutputBatch(
                output_file_paths=["/tmp/result.mp4"],
                metrics=batch.metrics,
            )

        async def run_test():
            await store.upsert(
                "task-timeout",
                {
                    "id": "task-timeout",
                    "status": "failed",
                    "progress": 42,
                    "error": {"message": TASK_TIMEOUT_MESSAGE},
                    "reason": TASK_TIMEOUT_MESSAGE,
                },
            )
            batch = Req(
                sampling_params=SamplingParams(
                    request_id="task-timeout",
                    prompt="repair video",
                )
            )
            with patch.object(video_api_mod, "VIDEO_STORE", store), patch.object(
                video_api_mod, "process_generation_batch", fake_process_generation_batch
            ):
                await _dispatch_job_async(
                    "task-timeout",
                    batch,
                    request_storage=FakeStorage(),
                )
            return await store.get("task-timeout")

        import asyncio

        job = asyncio.run(run_test())
        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["reason"], TASK_TIMEOUT_MESSAGE)
        self.assertNotIn("completed_at", job)
        self.assertEqual(len(upload_calls), 1)

    def test_process_generation_batch_preserves_timeout_error(self):
        class FakeSchedulerClient:
            async def forward(self, _batch):
                return OutputBatch(error=TASK_TIMEOUT_MESSAGE)

        async def run_test():
            with self.assertRaisesRegex(TaskTimeoutError, TASK_TIMEOUT_MESSAGE):
                await process_generation_batch(FakeSchedulerClient(), _expired_req())

        import asyncio

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
