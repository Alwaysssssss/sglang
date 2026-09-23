# SPDX-License-Identifier: Apache-2.0
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.multimodal_gen.configs.pipeline_configs.vsr import WanVSRPipelineConfig
from sglang.multimodal_gen.configs.sample.vsr import WanVRSamplingParams
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.vsr import (
    VSRRestoreStage,
)
from sglang.multimodal_gen.runtime.vsr.cli import main


class RequestParametersTest(unittest.TestCase):
    def setUp(self):
        patcher = patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.base.get_global_server_args",
            return_value=SimpleNamespace(),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_tiling_resets_between_requests(self):
        config = WanVSRPipelineConfig()
        restorer = SimpleNamespace()
        stage = VSRRestoreStage(restorer, config)
        stage._apply_tiling(WanVRSamplingParams(tile_h=640, temporal_overlap=10))
        self.assertEqual(restorer.tile_h, 640)
        stage._apply_tiling(WanVRSamplingParams())
        self.assertEqual(restorer.tile_h, config.tile_h)
        self.assertEqual(restorer.t_overlap, config.temporal_overlap)

    def test_request_long_edge_overrides_config_resolution(self):
        stage = VSRRestoreStage(
            None, WanVSRPipelineConfig(target_resolution="3840x2160")
        )
        self.assertEqual(
            stage._resolve_geometry(WanVRSamplingParams(long_edge=640), 100, 200),
            (320, 640),
        )

    def test_stage_rejects_precision_change_before_inference(self):
        stage = VSRRestoreStage(
            SimpleNamespace(dtype=torch.bfloat16), WanVSRPipelineConfig()
        )
        params = WanVRSamplingParams(
            video_input_path="input.mp4", output_path="out.mp4", dtype="float32"
        )
        batch = SimpleNamespace(sampling_params=params)
        module = "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.vsr"
        with (
            patch(module + ".probe_video", return_value=(25, 33, 320, 640)),
            patch(module + ".stream_restore", return_value=33) as restore,
        ):
            with self.assertRaisesRegex(ValueError, "loaded model precision"):
                stage.forward(batch, None)
            restore.assert_not_called()
            params.dtype = "bfloat16"
            self.assertIs(stage.forward(batch, None), batch)
            self.assertEqual(params.runtime_frames_written, 33)
            params.gpu_postprocess = True
            stage.forward(batch, None)
            self.assertTrue(restore.call_args.kwargs["gpu_postprocess"])
            params.gpu_postprocess = False
            stage.forward(batch, None)
            self.assertFalse(restore.call_args.kwargs["gpu_postprocess"])

    def test_native_cli_loads_requested_precision(self):
        class ReachedLoader(Exception):
            pass

        with (
            patch(
                "sglang.multimodal_gen.DiffGenerator.from_pretrained",
                side_effect=ReachedLoader,
            ) as load,
            self.assertRaises(ReachedLoader),
        ):
            main(
                [
                    "restore",
                    "--via-pipeline",
                    "--checkpoint_dir",
                    "/tmp/vsr-checkpoint",
                    "--wan_root",
                    "/tmp/wan",
                    "--input",
                    "/tmp/input.mp4",
                    "--output",
                    "/tmp/output.mp4",
                    "--dtype",
                    "float32",
                    "--cudnn-benchmark",
                    "--channels-last-3d",
                    "--compile-decoder",
                    "--compile-encoder",
                    "--decoder-implicit-padding",
                    "--cache-dit-condition",
                    "--vae-cpu-offload",
                    "--dit-layerwise-offload",
                    "--dit-offload-prefetch-size",
                    "2",
                ]
            )
        self.assertEqual(
            load.call_args.kwargs["server_args"].pipeline_config.precision, "float32"
        )

        config = load.call_args.kwargs["server_args"].pipeline_config
        self.assertTrue(config.cudnn_benchmark)
        self.assertTrue(config.channels_last_3d)
        self.assertTrue(config.compile_decoder)
        self.assertTrue(config.compile_encoder)
        self.assertTrue(config.decoder_implicit_padding)
        self.assertTrue(config.cache_dit_condition)
        server_args = load.call_args.kwargs["server_args"]
        self.assertTrue(server_args.vae_cpu_offload)
        self.assertTrue(server_args.dit_layerwise_offload)
        self.assertFalse(server_args.dit_cpu_offload)
        self.assertEqual(server_args.dit_offload_prefetch_size, 2)

    def test_cudnn_setting_restored_on_failure(self):
        from sglang.multimodal_gen.runtime.vsr.model import VSRRestorer

        model = VSRRestorer.__new__(VSRRestorer)
        torch.nn.Module.__init__(model)
        previous = torch.backends.cudnn.benchmark
        model.cudnn_benchmark = not previous
        model.device = torch.device("cpu")

        def fail(window):
            self.assertEqual(torch.backends.cudnn.benchmark, not previous)
            raise RuntimeError("inference failure")

        with (
            patch.object(model, "_restore_window", side_effect=fail),
            self.assertRaisesRegex(RuntimeError, "inference failure"),
        ):
            model.restore_window(None)
        self.assertEqual(torch.backends.cudnn.benchmark, previous)


if __name__ == "__main__":
    unittest.main()
