import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from sglang.multimodal_gen.configs.pipeline_configs.videoedit_wan import (
    WanVideoEditPipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.vividvr import (
    VividVRPipelineConfig,
)
from sglang.multimodal_gen.runtime.entrypoints.openai import video_api
from sglang.multimodal_gen.runtime.server_args import set_global_server_args


class TestVideoRepairAPI(unittest.TestCase):
    def setUp(self) -> None:
        self.input_dir = tempfile.TemporaryDirectory()
        self.output_dir = tempfile.TemporaryDirectory()
        self.prompt_file = Path(self.input_dir.name) / "prompt.txt"
        self.prompt_file.write_text("demo prompt\n", encoding="utf-8")
        self.override_prompt_file = Path(self.input_dir.name) / "override_prompt.txt"
        self.override_prompt_file.write_text("override prompt\n", encoding="utf-8")
        self.video_file = Path(self.input_dir.name) / "input.mp4"
        self.video_file.write_bytes(b"fake mp4 data")
        self.caption_file = Path(self.input_dir.name) / "captions.txt"
        self.caption_file.write_text("clip one\nclip two\n", encoding="utf-8")
        self.reference_video_file = Path(self.input_dir.name) / "reference.mp4"
        self.reference_video_file.write_bytes(b"fake reference mp4 data")

        self.app = FastAPI()
        self.app.include_router(video_api.router)
        video_api.VIDEO_STORE._items.clear()
        video_api._VIDEOEDIT_SEMAPHORE = asyncio.Semaphore(1)

    def tearDown(self) -> None:
        self.input_dir.cleanup()
        self.output_dir.cleanup()
        video_api.VIDEO_STORE._items.clear()

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

    async def _fake_dispatch_video_repair_job_async(self, *args, **kwargs):
        return None

    def test_vividvr_repair_accepts_minimal_request_without_prompt_or_mask(self):
        pipeline_config = VividVRPipelineConfig()
        pipeline_config.default_prompt_file_path = str(self.prompt_file)
        set_global_server_args(
            self._make_server_args(
                pipeline_config,
                model_id="served-vividvr",
            )
        )

        captured_kwargs = {}
        original_from_user_kwargs = video_api.VividVRSamplingParams.from_user_kwargs

        def capture_from_user_kwargs(server_args, *args, **kwargs):
            captured_kwargs.update(kwargs)
            return original_from_user_kwargs(server_args, *args, **kwargs)

        with patch.object(video_api, "prepare_request", return_value="fake-batch"):
            with patch.object(
                video_api,
                "_dispatch_video_repair_job_async",
                side_effect=self._fake_dispatch_video_repair_job_async,
            ):
                with patch.object(
                    video_api.VividVRSamplingParams,
                    "from_user_kwargs",
                    side_effect=capture_from_user_kwargs,
                ):
                    with TestClient(self.app) as client:
                        response = client.post(
                            "/v1/videos/repairs",
                            json={"video_input_path": str(self.video_file)},
                        )
                        video_response = client.get(
                            f'/v1/videos/{response.json()["id"]}'
                        )
                        progress_response = client.get(
                            f'/v1/videos/{response.json()["id"]}/progress'
                        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(video_response.status_code, 200)
        self.assertEqual(progress_response.status_code, 200)
        body = response.json()
        retrieved_body = video_response.json()
        progress_body = progress_response.json()
        self.assertEqual(body["status"], "queued")
        self.assertEqual(body["model"], "served-vividvr")
        self.assertTrue(body["file_path"].endswith(f'{body["id"]}.mp4'))
        self.assertEqual(retrieved_body["id"], body["id"])
        self.assertEqual(retrieved_body["status"], "queued")
        self.assertEqual(progress_body["id"], body["id"])
        self.assertEqual(progress_body["status"], "queued")
        self.assertEqual(progress_body["progress"], 0)
        self.assertEqual(captured_kwargs["video_input_path"], str(self.video_file))
        self.assertEqual(captured_kwargs["prompt_file_path"], str(self.prompt_file))
        self.assertNotIn("num_frames", captured_kwargs)
        self.assertNotIn("num_inference_steps", captured_kwargs)
        self.assertNotIn("guidance_scale", captured_kwargs)
        self.assertNotIn("output_quality", captured_kwargs)
        self.assertNotIn("output_compression", captured_kwargs)
        self.assertIn(body["id"], video_api.VIDEO_STORE._items)

    def test_vividvr_repair_preserves_explicit_output_quality_override(self):
        pipeline_config = VividVRPipelineConfig()
        pipeline_config.default_prompt_file_path = str(self.prompt_file)
        set_global_server_args(self._make_server_args(pipeline_config, model_id="VividVR"))

        captured_kwargs = {}

        def fake_from_user_kwargs(_server_args, *args, **kwargs):
            captured_kwargs.update(kwargs)
            return SimpleNamespace(
                prompt=kwargs.get("prompt"),
                output_file_path=lambda: str(Path(self.output_dir.name) / "job.mp4"),
            )

        with patch.object(video_api, "prepare_request", return_value="fake-batch"):
            with patch.object(
                video_api,
                "_dispatch_video_repair_job_async",
                side_effect=self._fake_dispatch_video_repair_job_async,
            ):
                with patch.object(
                    video_api.VividVRSamplingParams,
                    "from_user_kwargs",
                    side_effect=fake_from_user_kwargs,
                ):
                    with TestClient(self.app) as client:
                        response = client.post(
                            "/v1/videos/repairs",
                            json={
                                "video_input_path": str(self.video_file),
                                "output_quality": "high",
                            },
                        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured_kwargs["output_quality"], "high")

    def test_vividvr_repair_prefers_server_prompt_file_override(self):
        pipeline_config = VividVRPipelineConfig()
        pipeline_config.default_prompt_file_path = str(self.prompt_file)
        set_global_server_args(
            self._make_server_args(
                pipeline_config,
                prompt_file_path=str(self.override_prompt_file),
            )
        )

        captured_kwargs = {}
        original_from_user_kwargs = video_api.VividVRSamplingParams.from_user_kwargs

        def capture_from_user_kwargs(server_args, *args, **kwargs):
            captured_kwargs.update(kwargs)
            return original_from_user_kwargs(server_args, *args, **kwargs)

        with patch.object(video_api, "prepare_request", return_value="fake-batch"):
            with patch.object(
                video_api,
                "_dispatch_video_repair_job_async",
                side_effect=self._fake_dispatch_video_repair_job_async,
            ):
                with patch.object(
                    video_api.VividVRSamplingParams,
                    "from_user_kwargs",
                    side_effect=capture_from_user_kwargs,
                ):
                    with TestClient(self.app) as client:
                        response = client.post(
                            "/v1/videos/repairs",
                            json={"video_input_path": str(self.video_file)},
                        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            captured_kwargs["prompt_file_path"], str(self.override_prompt_file)
        )

    def test_vividvr_repair_forwards_caption_and_reference_overrides(self):
        pipeline_config = VividVRPipelineConfig()
        pipeline_config.default_prompt_file_path = str(self.prompt_file)
        set_global_server_args(
            self._make_server_args(
                pipeline_config,
                model_id="served-vividvr",
            )
        )

        captured_kwargs = {}
        original_from_user_kwargs = video_api.VividVRSamplingParams.from_user_kwargs

        def capture_from_user_kwargs(server_args, *args, **kwargs):
            captured_kwargs.update(kwargs)
            return original_from_user_kwargs(server_args, *args, **kwargs)

        with patch.object(video_api, "prepare_request", return_value="fake-batch"):
            with patch.object(
                video_api,
                "_dispatch_video_repair_job_async",
                side_effect=self._fake_dispatch_video_repair_job_async,
            ):
                with patch.object(
                    video_api.VividVRSamplingParams,
                    "from_user_kwargs",
                    side_effect=capture_from_user_kwargs,
                ):
                    with TestClient(self.app) as client:
                        response = client.post(
                            "/v1/videos/repairs",
                            json={
                                "video_input_path": str(self.video_file),
                                "caption_file_path": str(self.caption_file),
                                "reference_video_path": str(
                                    self.reference_video_file
                                ),
                            },
                        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured_kwargs["caption_source"], "caption_file")
        self.assertEqual(captured_kwargs["caption_file_path"], str(self.caption_file))
        self.assertEqual(
            captured_kwargs["reference_video_path"],
            str(self.reference_video_file),
        )

    def test_vividvr_repair_generates_caption_when_bridge_enabled(self):
        pipeline_config = VividVRPipelineConfig()
        pipeline_config.default_prompt_file_path = str(self.prompt_file)
        set_global_server_args(
            self._make_server_args(
                pipeline_config,
                model_id="served-vividvr",
                vividvr_caption_bridge=True,
                vividvr_caption_sidecar_url="http://127.0.0.1:31200",
                vividvr_caption_work_dir=str(
                    Path(self.output_dir.name) / "caption_sidecars"
                ),
                vividvr_caption_sidecar_timeout=30.0,
            )
        )

        captured_kwargs = {}

        def fake_from_user_kwargs(_server_args, *args, **kwargs):
            captured_kwargs.update(kwargs)
            return SimpleNamespace(
                prompt=kwargs.get("prompt"),
                output_file_path=lambda: str(Path(self.output_dir.name) / "job.mp4"),
            )

        async def fake_request_caption_sidecar(**kwargs):
            output_path = Path(kwargs["output_caption_path"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("caption 0\n", encoding="utf-8")
            return SimpleNamespace(
                caption_file_path=str(output_path),
                caption_count=1,
            )

        with patch.object(
            video_api,
            "build_vividvr_caption_manifest_for_video_path",
            return_value=SimpleNamespace(
                expected_caption_count=1,
                write_json=lambda path: Path(path).write_text("{}", encoding="utf-8"),
            ),
        ):
            with patch.object(
                video_api,
                "request_vividvr_caption_sidecar",
                side_effect=fake_request_caption_sidecar,
            ):
                with patch.object(video_api, "prepare_request", return_value="fake-batch"):
                    with patch.object(
                        video_api,
                        "_dispatch_video_repair_job_async",
                        side_effect=self._fake_dispatch_video_repair_job_async,
                    ):
                        with patch.object(
                            video_api.VividVRSamplingParams,
                            "from_user_kwargs",
                            side_effect=fake_from_user_kwargs,
                        ):
                            with TestClient(self.app) as client:
                                response = client.post(
                                    "/v1/videos/repairs",
                                    json={
                                        "task_id": "repair-auto",
                                        "video_input_path": str(self.video_file),
                                    },
                                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured_kwargs["caption_source"], "caption_file")
        self.assertTrue(
            captured_kwargs["caption_file_path"].endswith("repair-auto.txt")
        )

    def test_vividvr_repair_returns_500_when_caption_bridge_fails(self):
        pipeline_config = VividVRPipelineConfig()
        pipeline_config.default_prompt_file_path = str(self.prompt_file)
        set_global_server_args(
            self._make_server_args(
                pipeline_config,
                model_id="served-vividvr",
                vividvr_caption_bridge=True,
                vividvr_caption_sidecar_url="http://127.0.0.1:31200",
                vividvr_caption_work_dir=str(
                    Path(self.output_dir.name) / "caption_sidecars"
                ),
                vividvr_caption_sidecar_timeout=30.0,
            )
        )

        with patch.object(
            video_api,
            "build_vividvr_caption_manifest_for_video_path",
            return_value=SimpleNamespace(
                expected_caption_count=1,
                write_json=lambda path: Path(path).write_text("{}", encoding="utf-8"),
            ),
        ):
            with patch.object(
                video_api,
                "request_vividvr_caption_sidecar",
                side_effect=RuntimeError("sidecar offline"),
            ):
                with TestClient(self.app) as client:
                    response = client.post(
                        "/v1/videos/repairs",
                        json={"video_input_path": str(self.video_file)},
                    )

        self.assertEqual(response.status_code, 500)
        self.assertIn("caption bridge failed", response.json()["detail"])

    def test_vividvr_repair_accepts_explicit_vividvr_pipeline_class(self):
        pipeline_config = SimpleNamespace(default_prompt_file_path=str(self.prompt_file))
        set_global_server_args(
            self._make_server_args(
                pipeline_config,
                model_id="VividVR",
                pipeline_class_name="CogVideoXVividVRControlNetPipeline",
            )
        )

        captured_kwargs = {}
        captured_prepare_prompt = {}

        def fake_from_user_kwargs(_server_args, *args, **kwargs):
            captured_kwargs.update(kwargs)
            return SimpleNamespace(
                prompt=kwargs.get("prompt"),
                output_file_path=lambda: str(Path(self.output_dir.name) / "job.mp4")
            )

        def fake_prepare_request(*, server_args, sampling_params):
            captured_prepare_prompt["prompt"] = sampling_params.prompt
            return "fake-batch"

        with patch.object(video_api, "prepare_request", side_effect=fake_prepare_request):
            with patch.object(
                video_api,
                "_dispatch_video_repair_job_async",
                side_effect=self._fake_dispatch_video_repair_job_async,
            ):
                with patch.object(
                    video_api.VividVRSamplingParams,
                    "from_user_kwargs",
                    side_effect=fake_from_user_kwargs,
                ):
                    with TestClient(self.app) as client:
                        response = client.post(
                            "/v1/videos/repairs",
                            json={"video_input_path": str(self.video_file)},
                        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured_kwargs["prompt"], "demo prompt")
        self.assertEqual(captured_kwargs["prompt_file_path"], str(self.prompt_file))
        self.assertEqual(captured_kwargs["video_input_path"], str(self.video_file))
        self.assertEqual(captured_prepare_prompt["prompt"], "demo prompt")

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
