# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx

from sglang.multimodal_gen.runtime.managers.gpu_worker import (
    trim_layerwise_offload_device_cache,
)
from sglang.multimodal_gen.runtime.request_timeout import (
    TaskTimeoutError,
    check_request_timeout,
)
from sglang.multimodal_gen.runtime.scheduler_client import (
    _scheduler_response_timeout_ms,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.videoedit.dual_service_gateway import (
    GatewayConfig,
    GatewayRuntime,
    resolve_variant,
)
from sglang.multimodal_gen.runtime.videoedit.dual_service_store import (
    DuplicateTaskError,
    DualServiceStore,
)


class DualServiceStoreTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "queue.sqlite3")
        self.store = DualServiceStore(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    @staticmethod
    def payload(task_id, model):
        return {"task_id": task_id, "model": model, "prompt": "test"}

    def enqueue(self, task_id, variant):
        model = "videoedit-normal" if variant == "normal" else "videoedit-dmd"
        return self.store.enqueue(
            task_id=task_id,
            variant=variant,
            backend_url=f"http://127.0.0.1/{variant}",
            request_payload=self.payload(task_id, model),
        )

    def test_fifo_and_single_active_constraint(self):
        self.enqueue("first", "normal")
        self.enqueue("second", "dmd")
        first = self.store.claim_next()
        self.assertEqual(first["task_id"], "first")
        self.assertIsNone(self.store.claim_next())

        self.store.mark_terminal("first", "completed")
        second = self.store.claim_next()
        self.assertEqual(second["task_id"], "second")

    def test_two_store_instances_cannot_claim_two_tasks(self):
        self.enqueue("first", "normal")
        self.enqueue("second", "dmd")
        stores = [DualServiceStore(self.db_path), DualServiceStore(self.db_path)]
        barrier = threading.Barrier(2)
        results = []

        def claim(store):
            barrier.wait()
            results.append(store.claim_next())

        threads = [threading.Thread(target=claim, args=(store,)) for store in stores]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        claimed = [task for task in results if task is not None]
        self.assertEqual(len(claimed), 1)
        self.assertEqual(claimed[0]["task_id"], "first")

    def test_duplicate_task_is_rejected(self):
        self.enqueue("duplicate", "normal")
        with self.assertRaises(DuplicateTaskError):
            self.enqueue("duplicate", "dmd")

    def test_cancel_queued_does_not_touch_active(self):
        self.enqueue("active", "normal")
        self.enqueue("queued", "dmd")
        self.store.claim_next()
        self.assertTrue(self.store.cancel_queued("queued"))
        self.assertFalse(self.store.cancel_queued("active"))
        self.assertEqual(self.store.get("queued")["status"], "cancelled")

    def test_database_permissions_are_private(self):
        self.assertEqual(os.stat(self.db_path).st_mode & 0o777, 0o600)


class DualServiceHelpersTest(unittest.TestCase):
    def test_model_routing(self):
        for model in (None, "videoedit", "normal", "videoedit-normal"):
            self.assertEqual(resolve_variant(model), "normal")
        for model in ("dmd", "videoedit-dmd"):
            self.assertEqual(resolve_variant(model), "dmd")
        with self.assertRaises(ValueError):
            resolve_variant("unknown")

    def test_cancel_marker_is_checked_on_request_and_sampling_params(self):
        with tempfile.NamedTemporaryFile() as marker:
            with self.assertRaises(TaskTimeoutError):
                check_request_timeout(SimpleNamespace(request_cancel_path=marker.name))
            with self.assertRaises(TaskTimeoutError):
                check_request_timeout(
                    SimpleNamespace(
                        request_cancel_path=None,
                        sampling_params=SimpleNamespace(
                            request_cancel_path=marker.name,
                            request_timeout_deadline=None,
                        ),
                    )
                )

    def test_scheduler_timeout_minus_one_is_preserved(self):
        self.assertEqual(
            _scheduler_response_timeout_ms(
                SimpleNamespace(scheduler_response_timeout=-1)
            ),
            -1,
        )

    def test_scheduler_timeout_and_nccl_port_cli(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        args = parser.parse_args(
            ["--scheduler-response-timeout", "-1", "--nccl-port", "31655"]
        )
        self.assertEqual(args.scheduler_response_timeout, -1)
        self.assertEqual(args.nccl_port, 31655)

    @patch("sglang.multimodal_gen.runtime.managers.gpu_worker.gc.collect")
    @patch("sglang.multimodal_gen.runtime.managers.gpu_worker.torch.get_device_module")
    def test_cache_trim_records_both_sides(self, get_device_module, collect):
        device = MagicMock()
        device.memory_allocated.side_effect = [10, 8]
        device.memory_reserved.side_effect = [20, 9]
        get_device_module.return_value = device
        trim_layerwise_offload_device_cache(rank=1)
        collect.assert_called_once_with()
        device.empty_cache.assert_called_once_with()
        self.assertEqual(device.memory_allocated.call_count, 2)
        self.assertEqual(device.memory_reserved.call_count, 2)


class GatewayDispatcherTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config = GatewayConfig(
            queue_db=os.path.join(self.temp_dir.name, "queue.sqlite3"),
            normal_url="http://normal",
            dmd_url="http://dmd",
            poll_interval=0.01,
            health_timeout=0.1,
        )
        self.backend_status = {"normal": "processing", "dmd": "processing"}

        async def backend(request):
            variant = request.url.host
            if request.url.path == "/health":
                return httpx.Response(200, json={"status": "ok"})
            if request.method == "POST":
                payload = json.loads(request.content)
                return httpx.Response(
                    200,
                    json={
                        "code": 0,
                        "task_id": payload["task_id"],
                        "status": "submitted",
                    },
                )
            task_id = request.url.path.rsplit("/", 1)[-1]
            return httpx.Response(
                200,
                json={
                    "task_id": task_id,
                    "status": self.backend_status[variant],
                },
            )

        self.runtime = GatewayRuntime(self.config)
        await self.runtime.client.aclose()
        self.runtime.client = httpx.AsyncClient(transport=httpx.MockTransport(backend))

    async def asyncTearDown(self):
        await self.runtime.close()
        self.temp_dir.cleanup()

    def enqueue(self, task_id, variant):
        model = "videoedit-normal" if variant == "normal" else "videoedit-dmd"
        self.runtime.store.enqueue(
            task_id=task_id,
            variant=variant,
            backend_url=self.config.backend_url(variant),
            request_payload={"task_id": task_id, "model": model, "prompt": "test"},
        )

    async def test_enqueue_enforces_dmd_no_cfg_policy_only_for_dmd(self):
        common = {
            "prompt": "test",
            "video_input_path": "/tmp/video.mp4",
            "mask_input_path": "/tmp/mask.mp4",
            "reference_image_path": "/tmp/reference.png",
            "num_inference_steps": 20,
            "guidance_scale": 5.0,
            "dynamic_cfg": True,
            "negative_prompt": "low quality",
        }
        dmd = await self.runtime.enqueue(
            {"task_id": "dmd-policy", "model": "videoedit-dmd", **common}
        )
        dmd_request = dmd["request_json"]
        self.assertEqual(dmd_request["num_inference_steps"], 4)
        self.assertEqual(dmd_request["guidance_scale"], 1.0)
        self.assertFalse(dmd_request["dynamic_cfg"])
        self.assertIsNone(dmd_request["negative_prompt"])

        normal = await self.runtime.enqueue(
            {"task_id": "normal-policy", "model": "videoedit-normal", **common}
        )
        normal_request = normal["request_json"]
        self.assertEqual(normal_request["num_inference_steps"], 20)
        self.assertEqual(normal_request["guidance_scale"], 5.0)
        self.assertTrue(normal_request["dynamic_cfg"])
        self.assertEqual(normal_request["negative_prompt"], "low quality")

    async def test_dispatcher_serializes_normal_and_dmd(self):
        self.enqueue("normal-task", "normal")
        self.enqueue("dmd-task", "dmd")

        task = self.runtime.store.claim_next()
        await self.runtime._advance(task)
        self.assertEqual(self.runtime.store.get("normal-task")["status"], "running")
        self.assertEqual(self.runtime.store.get("dmd-task")["status"], "queued")

        self.backend_status["normal"] = "completed"
        task = self.runtime.store.get_active()
        await self.runtime._advance(task)
        self.assertEqual(self.runtime.store.get("normal-task")["status"], "completed")
        self.assertEqual(self.runtime.store.get("dmd-task")["status"], "queued")

        task = self.runtime.store.claim_next()
        await self.runtime._advance(task)
        self.assertEqual(self.runtime.store.get("dmd-task")["status"], "running")

    async def test_health_reports_normal_only_degradation(self):
        async def health_backend(request):
            status = 200 if request.url.host == "normal" else 503
            return httpx.Response(status, json={"status": "ok"})

        await self.runtime.client.aclose()
        self.runtime.client = httpx.AsyncClient(
            transport=httpx.MockTransport(health_backend)
        )
        health = await self.runtime.health_snapshot()
        self.assertEqual(health["status"], "degraded_normal_only")


if __name__ == "__main__":
    unittest.main()
