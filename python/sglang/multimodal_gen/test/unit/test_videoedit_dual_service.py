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
    create_app,
    resolve_variant,
)
from sglang.multimodal_gen.runtime.videoedit.dual_service_store import (
    BusyTaskError,
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

    def admit(self, task_id, variant):
        model = "videoedit-normal" if variant == "normal" else "videoedit-dmd"
        return self.store.admit(
            task_id=task_id,
            variant=variant,
            backend_url=f"http://127.0.0.1/{variant}",
            request_payload=self.payload(task_id, model),
        )

    def test_busy_rejected_until_terminal(self):
        first = self.admit("first", "normal")
        self.assertEqual(first["status"], "dispatching")
        for status in ("dispatching", "running", "cancelling"):
            self.store.update_task("first", status=status)
            with self.assertRaises(BusyTaskError):
                self.admit("second", "dmd")
            self.assertIsNone(self.store.get("second"))
        self.store.mark_terminal("first", "completed")
        self.assertEqual(self.admit("second", "dmd")["status"], "dispatching")
        self.assertEqual(self.store.counts()["queued"], 0)

    def test_two_store_instances_cannot_admit_two_tasks(self):
        stores = [DualServiceStore(self.db_path), DualServiceStore(self.db_path)]
        barrier = threading.Barrier(2)
        results = []

        def admit(store, task_id):
            barrier.wait()
            try:
                results.append(
                    store.admit(
                        task_id=task_id,
                        variant="normal",
                        backend_url="http://normal",
                        request_payload=self.payload(task_id, "normal"),
                    )
                )
            except BusyTaskError:
                results.append(None)

        threads = [
            threading.Thread(target=admit, args=(store, str(i)))
            for i, store in enumerate(stores)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(len(results), 2)
        self.assertEqual(sum(task is not None for task in results), 1)
        self.assertEqual(len(self.store.list_tasks()), 1)

    def test_duplicate_task_is_rejected(self):
        self.admit("duplicate", "normal")
        with self.assertRaises(DuplicateTaskError):
            self.admit("duplicate", "dmd")

    def test_upgrade_cancels_legacy_queue_and_preserves_active(self):
        self.admit("legacy", "dmd")
        self.store.update_task("legacy", status="queued")
        self.admit("active", "normal")
        reopened = DualServiceStore(self.db_path)
        self.assertEqual(reopened.get("legacy")["status"], "cancelled")
        self.assertIsNotNone(reopened.get("legacy")["completed_at"])
        self.assertEqual(reopened.get_active()["task_id"], "active")

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

    def admit(self, task_id, variant):
        model = "videoedit-normal" if variant == "normal" else "videoedit-dmd"
        self.runtime.store.admit(
            task_id=task_id,
            variant=variant,
            backend_url=self.config.backend_url(variant),
            request_payload={"task_id": task_id, "model": model, "prompt": "test"},
        )

    async def test_admit_enforces_dmd_no_cfg_policy_only_for_dmd(self):
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
        dmd = await self.runtime.admit(
            {"task_id": "dmd-policy", "model": "videoedit-dmd", **common}
        )
        dmd_request = dmd["request_json"]
        self.assertEqual(dmd_request["num_inference_steps"], 4)
        self.assertEqual(dmd_request["guidance_scale"], 1.0)
        self.assertFalse(dmd_request["dynamic_cfg"])
        self.assertIsNone(dmd_request["negative_prompt"])

        self.runtime.store.mark_terminal("dmd-policy", "completed")
        normal = await self.runtime.admit(
            {"task_id": "normal-policy", "model": "videoedit-normal", **common}
        )
        normal_request = normal["request_json"]
        self.assertEqual(normal_request["num_inference_steps"], 20)
        self.assertEqual(normal_request["guidance_scale"], 5.0)
        self.assertTrue(normal_request["dynamic_cfg"])
        self.assertEqual(normal_request["negative_prompt"], "low quality")

    async def test_dispatcher_uses_single_slot_for_normal_and_dmd(self):
        self.admit("normal-task", "normal")
        with self.assertRaises(BusyTaskError):
            self.admit("dmd-task", "dmd")
        await self.runtime._advance(self.runtime.store.get_active())
        self.assertEqual(self.runtime.store.get("normal-task")["status"], "running")
        self.backend_status["normal"] = "completed"
        await self.runtime._advance(self.runtime.store.get_active())
        self.admit("dmd-task", "dmd")
        await self.runtime._advance(self.runtime.store.get_active())
        self.assertEqual(self.runtime.store.get("dmd-task")["status"], "running")

    async def test_api_rejects_busy_without_storing_request(self):
        with patch(
            "sglang.multimodal_gen.runtime.videoedit.dual_service_gateway.GatewayRuntime",
            return_value=self.runtime,
        ):
            app = create_app(self.config)
        payload = {
            "task_id": "first",
            "prompt": "test",
            "model": "normal",
            "video_input_path": "/tmp/video.mp4",
            "mask_input_path": "/tmp/mask.mp4",
            "reference_image_path": "/tmp/reference.png",
        }
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
        ) as client:
            first = await client.post("/v1/videos/repairs", json=payload)
            self.assertEqual(first.status_code, 200)
            self.assertEqual(first.json()["code"], 0)
            self.assertEqual(first.json()["status"], "dispatching")
            duplicate = await client.post("/v1/videos/repairs", json=payload)
            self.assertEqual(duplicate.status_code, 409)
            payload.update(task_id="second", model="dmd")
            busy = await client.post("/v1/videos/repairs", json=payload)
            self.assertEqual(busy.status_code, 200)
            self.assertEqual(busy.json()["code"], 2)
            self.assertIsNone(self.runtime.store.get("second"))
            self.runtime.store.mark_terminal("first", "completed")
            second = await client.post("/v1/videos/repairs", json=payload)
            self.assertEqual(second.json()["code"], 0)
            self.assertEqual(self.runtime.store.counts()["queued"], 0)

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
