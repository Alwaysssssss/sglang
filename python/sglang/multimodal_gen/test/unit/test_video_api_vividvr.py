import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from sglang.multimodal_gen.configs.pipeline_configs.videoedit_wan import (
    WanVideoEditPipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.vividvr import (
    VividVRPipelineConfig,
)
from sglang.multimodal_gen.runtime.entrypoints.openai import video_api
from sglang.multimodal_gen.runtime.entrypoints.openai import video_repair_shared
from sglang.multimodal_gen.runtime.server_args import set_global_server_args


class TestVideoRepairAPI(unittest.TestCase):
    def setUp(self) -> None:
        self.input_dir = tempfile.TemporaryDirectory()
        self.output_dir = tempfile.TemporaryDirectory()
        self.prompt_file = Path(self.input_dir.name) / "prompt.txt"
        self.prompt_file.write_text("demo prompt\n", encoding="utf-8")
        self.video_file = Path(self.input_dir.name) / "input.mp4"
        self.video_file.write_bytes(b"fake mp4 data")

        self.app = FastAPI()
        self.app.include_router(video_api.router)
        video_api.VIDEO_STORE._items.clear()
        self.original_semaphore = video_repair_shared.VIDEOEDIT_SEMAPHORE

    def tearDown(self) -> None:
        self.input_dir.cleanup()
        self.output_dir.cleanup()
        video_api.VIDEO_STORE._items.clear()
        video_repair_shared.VIDEOEDIT_SEMAPHORE = self.original_semaphore

    def _make_server_args(
        self,
        pipeline_config,
        *,
        prompt_file_path=None,
        model_id=None,
        pipeline_class_name=None,
        **overrides,
    ):
        return SimpleNamespace(
            pipeline_config=pipeline_config,
            output_path=self.output_dir.name,
            input_save_path=self.input_dir.name,
            prompt_file_path=prompt_file_path,
            model_id=model_id,
            pipeline_class_name=pipeline_class_name,
            num_gpus=1,
            comfyui_mode=False,
            **overrides,
        )

    def test_vividvr_repair_rejects_shared_route_before_queue_acquire(self):
        pipeline_config = VividVRPipelineConfig()
        pipeline_config.default_prompt_file_path = str(self.prompt_file)
        set_global_server_args(
            self._make_server_args(
                pipeline_config,
                model_id="served-vividvr",
            )
        )

        class RejectIfAcquiredSemaphore:
            def locked(self):
                raise AssertionError("Vivid-VR shared route must reject before queue check")

            async def acquire(self):
                raise AssertionError("Vivid-VR shared route must not acquire semaphore")

        video_repair_shared.VIDEOEDIT_SEMAPHORE = RejectIfAcquiredSemaphore()

        with TestClient(self.app) as client:
            response = client.post(
                "/v1/videos/repairs",
                json={"video_input_path": str(self.video_file)},
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json()["detail"],
            "Vivid-VR video repair must use /v1/videos/repairs/flowcut",
        )

    def test_wan_repair_still_requires_mask(self):
        set_global_server_args(
            self._make_server_args(WanVideoEditPipelineConfig(), model_id="videoedit")
        )

        with TestClient(self.app) as client:
            response = client.post(
                "/v1/videos/repairs",
                json={
                    "prompt": "repair this video",
                    "video_input_path": str(self.video_file),
                },
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json()["detail"], "mask_input_path or mask_url is required"
        )

    def test_wan_repair_still_requires_prompt(self):
        set_global_server_args(
            self._make_server_args(WanVideoEditPipelineConfig(), model_id="videoedit")
        )
        mask_file = Path(self.input_dir.name) / "mask.mp4"
        mask_file.write_bytes(b"fake mask data")

        with TestClient(self.app) as client:
            response = client.post(
                "/v1/videos/repairs",
                json={
                    "video_input_path": str(self.video_file),
                    "mask_input_path": str(mask_file),
                },
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "prompt is required")
