import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from diffusers.video_processor import VideoProcessor

from sglang.multimodal_gen.configs.pipeline_configs.vividvr import VividVRPipelineConfig
from sglang.multimodal_gen.configs.sample.vividvr import VividVRSamplingParams
from sglang.multimodal_gen.runtime.pipelines.vividvr_pipeline import VividVRPipeline
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.vividvr import (
    VividVRTilingPreparationStage,
)
from sglang.multimodal_gen.runtime.vividvr.captioning import (
    build_vividvr_caption_prompt_lists,
    build_vividvr_tiled_prompt_lists,
    prepare_vividvr_prompt_context,
    read_caption_file,
)
from sglang.multimodal_gen.runtime.vividvr.postprocess import (
    decoded_video_to_frame_tensor,
    run_optional_postprocess_modules,
)
from sglang.multimodal_gen.runtime.vividvr.windowing import (
    build_vividvr_temporal_latent_merge_plan,
    build_vividvr_temporal_window_plan,
    merge_vividvr_temporal_latent_states,
    stitch_vividvr_temporal_output_clips,
    trim_vividvr_temporal_output_clip,
)


class TestStageDVividVRTemporalOrchestration(unittest.TestCase):
    def _make_vividvr_params(self, **kwargs) -> VividVRSamplingParams:
        return VividVRSamplingParams(
            prompt=" ",
            video_input_path="/home/zhiheng/Vivid-VR/input/720p/test_video_960x720.mp4",
            prompt_file_path="/home/zhiheng/Vivid-VR/input/720p/prompt.txt",
            **kwargs,
        )

    def test_build_temporal_window_plan_matches_reference_math(self):
        plan = build_vividvr_temporal_window_plan(
            num_frames=200,
            num_temporal_process_frames=121,
        )

        self.assertEqual(plan.num_clips, 3)
        self.assertEqual(plan.num_temporal_overlapped_frames, 61)
        self.assertEqual(plan.temporal_frame_stride, 60)
        self.assertEqual(
            [clip.start_frame for clip in plan.clip_specs],
            [0, 60, 120],
        )
        self.assertEqual(
            [clip.original_num_frames for clip in plan.clip_specs],
            [121, 121, 80],
        )
        self.assertEqual(
            [clip.num_padding_frames for clip in plan.clip_specs],
            [0, 0, 1],
        )
        self.assertEqual(
            [
                (clip.trim_front_frames, clip.trim_back_frames)
                for clip in plan.clip_specs
            ],
            [(0, 30), (31, 30), (31, 0)],
        )

    def test_build_vividvr_tiled_prompt_lists_matches_tile_count(self):
        tiled_prompts = build_vividvr_tiled_prompt_lists(
            model_prompt_text="prompt",
            negative_prompt_text="negative",
            tile_count=3,
        )

        self.assertEqual(tiled_prompts["prompt_list"], ["prompt", "prompt", "prompt"])
        self.assertEqual(
            tiled_prompts["negative_prompt_list"],
            ["negative", "negative", "negative"],
        )

    def test_build_vividvr_caption_prompt_lists_consumes_requested_entries(self):
        prompt_lists = build_vividvr_caption_prompt_lists(
            caption_texts=["clip 0", "clip 1", "clip 2"],
            start_index=1,
            tile_count=2,
            negative_prompt_text="negative",
            pipeline_config=VividVRPipelineConfig(),
        )

        self.assertEqual(prompt_lists["clip_caption_text"], "clip 1")
        self.assertEqual(prompt_lists["caption_texts"], ["clip 1"])
        self.assertEqual(prompt_lists["next_index"], 2)
        self.assertTrue(prompt_lists["prompt_list"][0].startswith("clip 1 "))
        self.assertTrue(prompt_lists["prompt_list"][1].startswith("clip 1 "))
        self.assertEqual(prompt_lists["negative_prompt_list"], ["negative", "negative"])

    def test_build_vividvr_caption_prompt_lists_rejects_insufficient_entries(self):
        with self.assertRaisesRegex(ValueError, "does not contain enough entries"):
            build_vividvr_caption_prompt_lists(
                caption_texts=["only one"],
                start_index=1,
                tile_count=2,
                negative_prompt_text="negative",
                pipeline_config=VividVRPipelineConfig(),
            )

    def test_temporal_windowed_forward_repeats_one_clip_caption_across_tiles(self):
        params = self._make_vividvr_params(
            num_frames=130,
            num_temporal_process_frames=121,
            num_inference_steps=1,
            height=4,
            width=4,
            seed=42,
        )
        batch = SimpleNamespace(
            sampling_params=params,
            perf_dump_path=None,
            metrics={},
            extra={},
            output=None,
            fps=None,
        )

        def _prompt_stage(current_batch, _server_args):
            current_batch.sampling_params.runtime_model_prompt_text = "prompt"
            current_batch.sampling_params.runtime_negative_prompt_text = ""
            current_batch.sampling_params.runtime_caption_texts = ["clip 0", "clip 1"]
            return current_batch

        class _DummyProgressBar:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def update(self):
                return None

        class _DummyDenoisingStage:
            def prepare_denoising_state(self, _batch, _server_args, **kwargs):
                return {
                    "latents": kwargs["latents"],
                    "autocast_enabled": False,
                    "target_dtype": torch.float32,
                }

            def progress_bar(self, total):
                self.total = total
                return _DummyProgressBar()

            def run_denoising_step(
                self,
                current_batch,
                _server_args,
                denoising_state,
                timestep_index,
                *,
                guidance_scale,
                restoration_guidance_scale,
            ):
                return None

        encoded_prompt_calls = []

        def _encode_prompt_pair(**kwargs):
            encoded_prompt_calls.append(list(kwargs["prompt"]))
            return {
                "prompt_embeds": torch.zeros(len(kwargs["prompt"]), 226, 8),
                "negative_prompt_embeds": torch.zeros(len(kwargs["prompt"]), 226, 8),
            }

        pipeline = object.__new__(VividVRPipeline)
        pipeline.input_validation_stage = lambda current_batch, _server_args: current_batch
        pipeline.prompt_preparation_stage = _prompt_stage
        pipeline._attach_runtime_acceleration_debug = lambda _batch, _server_args: None
        pipeline.condition_encoding_stage = SimpleNamespace(
            prepare_condition_inputs=lambda *_args, **_kwargs: {
                "control_video": torch.zeros(121, 3, 4, 4),
                "control_latents": torch.zeros(1, 1, 1, 2, 2),
                "generator": torch.Generator(device="cpu").manual_seed(42),
            }
        )
        pipeline.latent_preparation_stage = SimpleNamespace(
            prepare_latents=lambda **_kwargs: (
                torch.zeros(1, 1, 1, 2, 2),
                torch.zeros(1, 1, 1, 2, 2),
                0,
            )
        )
        pipeline.tiling_preparation_stage = SimpleNamespace(
            build_tiling_infos=lambda **_kwargs: [
                SimpleNamespace(tile_index=0),
                SimpleNamespace(tile_index=1),
            ],
            prepare_tiling_state=lambda **kwargs: {
                "tiling_infos": kwargs["tiling_infos"],
                "tiled_prompt_embeds": kwargs["prompt_embeds"],
                "tiled_negative_prompt_embeds": kwargs["negative_prompt_embeds"],
                "tile_count": len(kwargs["tiling_infos"]),
            },
        )
        pipeline.text_encoding_stage = SimpleNamespace(
            encode_prompt_pair=_encode_prompt_pair
        )
        pipeline.timestep_preparation_stage = SimpleNamespace(
            prepare_timesteps=lambda _steps: torch.tensor([1.0], dtype=torch.float32)
        )
        pipeline.denoising_stage = _DummyDenoisingStage()
        pipeline.decoding_stage = SimpleNamespace(
            decode_latents=lambda _latents, _padding_frames, _server_args: torch.zeros(
                121, 3, 4, 4
            )
        )
        pipeline.video_processor = SimpleNamespace()
        pipeline.get_module = lambda _name: SimpleNamespace(
            config=SimpleNamespace(
                temporal_compression_ratio=4,
                block_out_channels=[128, 256, 256, 512],
            )
        )

        clip_specs = [
            SimpleNamespace(
                clip_index=0,
                start_frame=0,
                end_frame=121,
                original_num_frames=121,
                padded_num_frames=121,
                num_padding_frames=0,
                trim_front_frames=0,
                trim_back_frames=0,
            ),
            SimpleNamespace(
                clip_index=1,
                start_frame=60,
                end_frame=130,
                original_num_frames=70,
                padded_num_frames=73,
                num_padding_frames=3,
                trim_front_frames=31,
                trim_back_frames=0,
            ),
        ]
        window_plan = SimpleNamespace(
            clip_specs=clip_specs,
            num_clips=2,
            num_temporal_overlapped_frames=61,
            temporal_frame_stride=60,
        )
        input_video_info = {
            "reference_video": torch.zeros(130, 3, 4, 4),
            "original_num_frames": 130,
            "original_height": 4,
            "original_width": 4,
            "fps": 24,
        }

        with patch(
            "sglang.multimodal_gen.runtime.pipelines.vividvr_pipeline.get_local_torch_device",
            return_value=torch.device("cpu"),
        ), patch(
            "sglang.multimodal_gen.runtime.pipelines.vividvr_pipeline.build_vividvr_temporal_window_plan",
            return_value=window_plan,
        ), patch(
            "sglang.multimodal_gen.runtime.pipelines.vividvr_pipeline.build_vividvr_temporal_latent_merge_plan",
            return_value=SimpleNamespace(),
        ), patch(
            "sglang.multimodal_gen.runtime.pipelines.vividvr_pipeline.merge_vividvr_temporal_latent_states"
        ), patch(
            "sglang.multimodal_gen.runtime.pipelines.vividvr_pipeline.decoded_video_to_frame_tensor",
            side_effect=lambda video, **_kwargs: video,
        ), patch(
            "sglang.multimodal_gen.runtime.pipelines.vividvr_pipeline.trim_vividvr_temporal_output_clip",
            side_effect=lambda video, _clip_spec: video,
        ), patch(
            "sglang.multimodal_gen.runtime.pipelines.vividvr_pipeline.stitch_vividvr_temporal_output_clips",
            side_effect=lambda clips: clips[0],
        ), patch(
            "sglang.multimodal_gen.runtime.pipelines.vividvr_pipeline.apply_reference_color_fix",
            side_effect=lambda video, _reference: video,
        ), patch(
            "sglang.multimodal_gen.runtime.pipelines.vividvr_pipeline.run_optional_postprocess_modules",
            side_effect=lambda video, **_kwargs: video,
        ):
            result = pipeline._forward_temporal_windowed(
                batch,
                SimpleNamespace(pipeline_config=VividVRPipelineConfig()),
                input_video_info,
            )

        self.assertIs(result, batch)
        self.assertEqual(len(encoded_prompt_calls), 2)
        self.assertTrue(all(prompt.startswith("clip 0 ") for prompt in encoded_prompt_calls[0]))
        self.assertTrue(all(prompt.startswith("clip 1 ") for prompt in encoded_prompt_calls[1]))
        self.assertEqual([len(prompts) for prompts in encoded_prompt_calls], [2, 2])
        self.assertEqual(
            batch.extra["vividvr_debug"]["clip_caption_texts"],
            [
                {"clip_index": 0, "caption_text": "clip 0", "tile_count": 2},
                {"clip_index": 1, "caption_text": "clip 1", "tile_count": 2},
            ],
        )
        self.assertEqual(batch.height, 8)
        self.assertEqual(batch.width, 8)

    def test_build_temporal_latent_merge_plan_matches_reference_math(self):
        plan = build_vividvr_temporal_latent_merge_plan(
            [32, 32, 22],
            num_temporal_process_frames=121,
            vae_scale_factor_temporal=4,
        )

        self.assertEqual(plan.non_first_frame_latents_start_index, 2)
        self.assertEqual(plan.num_temporal_overlap_latents, 15)
        self.assertEqual(plan.temporal_latent_stride, 15)
        self.assertEqual(
            plan.clip_id_to_latent_id_map,
            {
                0: (1, 31),
                1: (16, 46),
                2: (31, 51),
            },
        )
        self.assertEqual(plan.valid_latent_id_to_clip_id_map[1], 0)
        self.assertEqual(plan.valid_latent_id_to_clip_id_map[22], 0)
        self.assertEqual(plan.valid_latent_id_to_clip_id_map[23], 1)
        self.assertEqual(plan.valid_latent_id_to_clip_id_map[37], 1)
        self.assertEqual(plan.valid_latent_id_to_clip_id_map[38], 2)
        self.assertEqual(plan.valid_latent_id_to_clip_id_map[50], 2)

    def test_merge_temporal_latent_states_copies_owner_values(self):
        plan = build_vividvr_temporal_latent_merge_plan(
            [32, 32, 22],
            num_temporal_process_frames=121,
            vae_scale_factor_temporal=4,
        )
        original_states = []
        clip_states = []
        for clip_index, temporal_length in enumerate([32, 32, 22]):
            values = torch.arange(temporal_length, dtype=torch.float32).view(
                1, temporal_length, 1, 1, 1
            )
            values = values + float((clip_index + 1) * 100)
            original_states.append(values.clone())
            clip_states.append(
                {
                    "latents": values.clone(),
                    "old_pred_original_sample": values.clone(),
                }
            )

        merge_vividvr_temporal_latent_states(clip_states, plan)

        for clip_index, clip_state in enumerate(clip_states):
            clip_begin, clip_end = plan.clip_id_to_latent_id_map[clip_index]
            clip_offset = clip_begin - plan.non_first_frame_latents_start_index
            for latent_id in range(clip_begin, clip_end):
                owner_index = plan.valid_latent_id_to_clip_id_map[latent_id]
                owner_begin, _ = plan.clip_id_to_latent_id_map[owner_index]
                owner_offset = owner_begin - plan.non_first_frame_latents_start_index
                expected_tensor_index = latent_id - owner_offset
                actual_tensor_index = latent_id - clip_offset
                expected_value = original_states[owner_index][
                    :,
                    expected_tensor_index,
                    ...,
                ]
                self.assertTrue(
                    torch.equal(
                        clip_state["latents"][:, actual_tensor_index, ...],
                        expected_value,
                    )
                )
                self.assertTrue(
                    torch.equal(
                        clip_state["old_pred_original_sample"][
                            :,
                            actual_tensor_index,
                            ...,
                        ],
                        expected_value,
                    )
                )

    def test_trim_and_stitch_temporal_output_clips_preserves_frame_count(self):
        plan = build_vividvr_temporal_window_plan(
            num_frames=200,
            num_temporal_process_frames=121,
        )
        clip_lengths = [121, 121, 81]
        trimmed_clips = []
        for clip_spec, clip_length in zip(plan.clip_specs, clip_lengths, strict=True):
            clip = torch.zeros(clip_length, 3, 4, 4)
            trimmed_clips.append(trim_vividvr_temporal_output_clip(clip, clip_spec))

        stitched = stitch_vividvr_temporal_output_clips(trimmed_clips)
        self.assertEqual(stitched.shape[0], 200)

    def test_tiling_state_accepts_pre_tiled_prompt_embeds(self):
        latents = torch.zeros(1, 31, 2, 4, 8)
        tiling_infos = VividVRTilingPreparationStage.build_tiling_infos(
            latents=latents,
            enable_spatial_tiling=True,
            enable_temporal_tiling=False,
            tile_size=4,
            tile_stride=4,
        )
        self.assertEqual(len(tiling_infos), 2)

        prompt_embeds = torch.randn(2, 226, 8)
        negative_prompt_embeds = torch.randn(2, 226, 8)
        tiling_state = VividVRTilingPreparationStage.prepare_tiling_state(
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            enable_spatial_tiling=True,
            enable_temporal_tiling=False,
            tile_size=4,
            tile_stride=4,
            tiling_infos=tiling_infos,
        )

        self.assertEqual(tiling_state["tile_count"], 2)
        self.assertTrue(torch.equal(tiling_state["tiled_prompt_embeds"], prompt_embeds))
        self.assertTrue(
            torch.equal(
                tiling_state["tiled_negative_prompt_embeds"],
                negative_prompt_embeds,
            )
        )

    def test_tiling_state_rejects_prompt_embed_batch_mismatch(self):
        latents = torch.zeros(1, 31, 2, 4, 8)

        with self.assertRaisesRegex(ValueError, "prompt_embeds batch size"):
            VividVRTilingPreparationStage.prepare_tiling_state(
                latents=latents,
                prompt_embeds=torch.randn(3, 226, 8),
                negative_prompt_embeds=None,
                enable_spatial_tiling=True,
                enable_temporal_tiling=False,
                tile_size=4,
                tile_stride=4,
            )

    def test_caption_module_can_be_disabled_without_changing_prompt_file_path(self):
        params = self._make_vividvr_params(enable_optional_caption_module=False)
        prompt_context = prepare_vividvr_prompt_context(
            params,
            pipeline_config=VividVRPipelineConfig(),
            debug={},
        )

        self.assertEqual(prompt_context["caption_backend"], "prompt_file")
        self.assertEqual(
            prompt_context["prompt_file_path"],
            "/home/zhiheng/Vivid-VR/input/720p/prompt.txt",
        )

    def test_caption_module_falls_back_to_prompt_file_when_enabled_helper_fails(self):
        params = self._make_vividvr_params(
            enable_optional_caption_module=True,
            allow_optional_module_fallback=True,
        )
        debug = {}
        fallback_context = {
            "prompt_file_path": "/home/zhiheng/Vivid-VR/input/720p/prompt.txt",
            "prompt_text": "prompt",
            "model_prompt_text": "model prompt",
            "negative_prompt_text": "negative prompt",
            "caption_backend": "prompt_file",
        }

        with patch(
            "sglang.multimodal_gen.runtime.vividvr.captioning._direct_prompt_file_caption_context",
            side_effect=[RuntimeError("caption placeholder failure"), fallback_context],
        ) as mocked_helper:
            prompt_context = prepare_vividvr_prompt_context(
                params,
                pipeline_config=VividVRPipelineConfig(),
                debug=debug,
            )

        self.assertEqual(prompt_context, fallback_context)
        self.assertEqual(mocked_helper.call_count, 2)
        self.assertEqual(
            debug["optional_module_warnings"],
            ["caption_module_fallback: caption placeholder failure"],
        )

    def test_caption_file_context_reads_non_empty_lines(self):
        caption_file = Path(self.id()).with_suffix(".txt")
        caption_file.write_text("\ncaption a\n\ncaption b\n", encoding="utf-8")
        self.addCleanup(caption_file.unlink)

        caption_texts = read_caption_file(str(caption_file))

        self.assertEqual(caption_texts, ["caption a", "caption b"])

    def test_prepare_prompt_context_uses_caption_file_when_requested(self):
        caption_file = Path(self.id()).with_suffix(".txt")
        caption_file.write_text("caption a\ncaption b\n", encoding="utf-8")
        self.addCleanup(caption_file.unlink)

        params = self._make_vividvr_params(
            caption_source="caption_file",
            caption_file_path=str(caption_file),
        )
        prompt_context = prepare_vividvr_prompt_context(
            params,
            pipeline_config=VividVRPipelineConfig(),
            debug={},
        )

        self.assertEqual(prompt_context["caption_backend"], "caption_file")
        self.assertEqual(prompt_context["caption_file_path"], str(caption_file))
        self.assertEqual(prompt_context["caption_texts"], ["caption a", "caption b"])
        self.assertTrue(prompt_context["model_prompt_text"].startswith("caption a "))

    def test_postprocess_module_fallback_returns_original_output(self):
        output_video = torch.rand(4, 3, 8, 8)
        reference_video = torch.rand(4, 3, 8, 8)
        debug = {}

        result = run_optional_postprocess_modules(
            output_video,
            reference_video=reference_video,
            enabled=True,
            allow_fallback=True,
            debug=debug,
            processor=lambda *_args: (_ for _ in ()).throw(RuntimeError("postprocess boom")),
        )

        self.assertTrue(torch.equal(result, output_video))
        self.assertEqual(
            debug["optional_module_warnings"],
            ["postprocess_module_fallback: postprocess boom"],
        )

    def test_postprocess_module_raises_when_fallback_is_disabled(self):
        output_video = torch.rand(4, 3, 8, 8)
        reference_video = torch.rand(4, 3, 8, 8)

        with self.assertRaisesRegex(RuntimeError, "postprocess boom"):
            run_optional_postprocess_modules(
                output_video,
                reference_video=reference_video,
                enabled=True,
                allow_fallback=False,
                debug={},
                processor=lambda *_args: (_ for _ in ()).throw(RuntimeError("postprocess boom")),
            )

    def test_decoded_video_to_frame_tensor_matches_reference_processor_path(self):
        decoded_video = torch.linspace(
            -1.0,
            1.0,
            steps=1 * 3 * 2 * 3 * 4,
            dtype=torch.float32,
        ).reshape(1, 3, 2, 3, 4)
        video_processor = VideoProcessor(vae_scale_factor=8)

        expected_resized = [
            F.interpolate(
                sample.permute(1, 0, 2, 3),
                size=(5, 6),
                mode="bilinear",
                align_corners=False,
            )
            for sample in decoded_video
        ]
        expected_resized = torch.stack(expected_resized, dim=0).permute(0, 2, 1, 3, 4)
        expected = video_processor.postprocess_video(
            video=expected_resized.float(),
            output_type="pt",
        )[0]

        actual = decoded_video_to_frame_tensor(
            decoded_video,
            video_processor=video_processor,
            original_height=5,
            original_width=6,
        )

        torch.testing.assert_close(actual, expected)

    def test_resolve_input_video_info_reuses_pipeline_cache_for_same_file(self):
        pipeline = object.__new__(VividVRPipeline)
        fake_video_info = {"original_num_frames": 121}

        with patch(
            "sglang.multimodal_gen.runtime.pipelines.vividvr_pipeline.os.stat",
            return_value=SimpleNamespace(st_mtime_ns=123, st_size=456),
        ), patch(
            "sglang.multimodal_gen.runtime.pipelines.vividvr_pipeline.load_control_video",
            return_value=fake_video_info,
        ) as mock_load_control_video:
            first = pipeline._resolve_input_video_info("/tmp/control.mp4", upscale=1.0)
            second = pipeline._resolve_input_video_info("/tmp/control.mp4", upscale=1.0)

        self.assertIs(first, fake_video_info)
        self.assertIs(second, fake_video_info)
        self.assertIs(first, second)
        mock_load_control_video.assert_called_once_with(
            "/tmp/control.mp4",
            upscale=1.0,
        )

    def test_temporal_windowed_forward_uses_unbound_step_profile_helper(self):
        params = self._make_vividvr_params(
            num_frames=4,
            num_temporal_process_frames=121,
            num_inference_steps=1,
            height=4,
            width=4,
            seed=42,
        )
        batch = SimpleNamespace(
            sampling_params=params,
            perf_dump_path=None,
            metrics={},
            extra={},
            output=None,
            fps=None,
        )

        def _prompt_stage(current_batch, _server_args):
            current_batch.sampling_params.runtime_model_prompt_text = "prompt"
            current_batch.sampling_params.runtime_negative_prompt_text = ""
            current_batch.sampling_params.runtime_caption_texts = None
            return current_batch

        class _DummyProgressBar:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def update(self):
                return None

        class _DummyDenoisingStage:
            def __init__(self):
                self.run_calls = 0
                self.last_call = None

            def prepare_denoising_state(self, _batch, _server_args, **kwargs):
                return {
                    "latents": kwargs["latents"],
                    "autocast_enabled": False,
                    "target_dtype": torch.float32,
                }

            def progress_bar(self, total):
                self.total = total
                return _DummyProgressBar()

            def run_denoising_step(
                self,
                current_batch,
                _server_args,
                denoising_state,
                timestep_index,
                *,
                guidance_scale,
                restoration_guidance_scale,
            ):
                self.run_calls += 1
                self.last_call = {
                    "timestep_index": timestep_index,
                    "guidance_scale": guidance_scale,
                    "restoration_guidance_scale": restoration_guidance_scale,
                    "latent_shape": tuple(denoising_state["latents"].shape),
                    "raw_latent_shape": tuple(current_batch.raw_latent_shape),
                }

        dummy_denoising_stage = _DummyDenoisingStage()
        pipeline = object.__new__(VividVRPipeline)
        pipeline.input_validation_stage = lambda current_batch, _server_args: current_batch
        pipeline.prompt_preparation_stage = _prompt_stage
        pipeline._attach_runtime_acceleration_debug = lambda _batch, _server_args: None
        pipeline.condition_encoding_stage = SimpleNamespace(
            prepare_condition_inputs=lambda *_args, **_kwargs: {
                "control_video": torch.zeros(4, 3, 4, 4),
                "control_latents": torch.zeros(1, 1, 1, 2, 2),
                "generator": torch.Generator(device="cpu").manual_seed(42),
            }
        )
        pipeline.latent_preparation_stage = SimpleNamespace(
            prepare_latents=lambda **_kwargs: (
                torch.zeros(1, 1, 1, 2, 2),
                torch.zeros(1, 1, 1, 2, 2),
                0,
            )
        )
        pipeline.tiling_preparation_stage = SimpleNamespace(
            build_tiling_infos=lambda **_kwargs: [SimpleNamespace(tile_index=0)],
            prepare_tiling_state=lambda **kwargs: {
                "tiling_infos": kwargs["tiling_infos"],
                "tiled_prompt_embeds": kwargs["prompt_embeds"],
                "tiled_negative_prompt_embeds": kwargs["negative_prompt_embeds"],
            },
        )
        pipeline.text_encoding_stage = SimpleNamespace(
            encode_prompt_pair=lambda **_kwargs: {
                "prompt_embeds": torch.zeros(1, 226, 8),
                "negative_prompt_embeds": torch.zeros(1, 226, 8),
            }
        )
        pipeline.timestep_preparation_stage = SimpleNamespace(
            prepare_timesteps=lambda _steps: torch.tensor([1.0], dtype=torch.float32)
        )
        pipeline.denoising_stage = dummy_denoising_stage
        pipeline.decoding_stage = SimpleNamespace(
            decode_latents=lambda _latents, _padding_frames, _server_args: torch.zeros(
                4, 3, 4, 4
            )
        )
        pipeline.video_processor = SimpleNamespace()
        pipeline.get_module = lambda _name: SimpleNamespace(
            config=SimpleNamespace(
                temporal_compression_ratio=4,
                block_out_channels=[128, 256, 256, 512],
            )
        )

        clip_spec = SimpleNamespace(
            clip_index=0,
            start_frame=0,
            end_frame=4,
            original_num_frames=4,
            padded_num_frames=4,
            num_padding_frames=0,
            trim_front_frames=0,
            trim_back_frames=0,
        )
        window_plan = SimpleNamespace(
            clip_specs=[clip_spec],
            num_clips=1,
            num_temporal_overlapped_frames=0,
            temporal_frame_stride=4,
        )
        input_video_info = {
            "reference_video": torch.zeros(4, 3, 4, 4),
            "original_num_frames": 4,
            "original_height": 4,
            "original_width": 4,
            "fps": 24,
        }

        with patch(
            "sglang.multimodal_gen.runtime.pipelines.vividvr_pipeline.get_local_torch_device",
            return_value=torch.device("cpu"),
        ), patch(
            "sglang.multimodal_gen.runtime.pipelines.vividvr_pipeline.build_vividvr_temporal_window_plan",
            return_value=window_plan,
        ), patch(
            "sglang.multimodal_gen.runtime.pipelines.vividvr_pipeline.build_vividvr_temporal_latent_merge_plan",
            return_value=SimpleNamespace(),
        ), patch(
            "sglang.multimodal_gen.runtime.pipelines.vividvr_pipeline.merge_vividvr_temporal_latent_states"
        ), patch(
            "sglang.multimodal_gen.runtime.pipelines.vividvr_pipeline.decoded_video_to_frame_tensor",
            side_effect=lambda video, **_kwargs: video,
        ), patch(
            "sglang.multimodal_gen.runtime.pipelines.vividvr_pipeline.trim_vividvr_temporal_output_clip",
            side_effect=lambda video, _clip_spec: video,
        ), patch(
            "sglang.multimodal_gen.runtime.pipelines.vividvr_pipeline.stitch_vividvr_temporal_output_clips",
            side_effect=lambda clips: clips[0],
        ), patch(
            "sglang.multimodal_gen.runtime.pipelines.vividvr_pipeline.apply_reference_color_fix",
            side_effect=lambda video, _reference: video,
        ), patch(
            "sglang.multimodal_gen.runtime.pipelines.vividvr_pipeline.run_optional_postprocess_modules",
            side_effect=lambda video, **_kwargs: video,
        ):
            result = pipeline._forward_temporal_windowed(
                batch,
                SimpleNamespace(pipeline_config=VividVRPipelineConfig()),
                input_video_info,
            )

        self.assertIs(result, batch)
        self.assertEqual(dummy_denoising_stage.run_calls, 1)
        self.assertEqual(dummy_denoising_stage.last_call["timestep_index"], 0)
        self.assertEqual(
            dummy_denoising_stage.last_call["raw_latent_shape"],
            dummy_denoising_stage.last_call["latent_shape"],
        )
        self.assertEqual(batch.height, 8)
        self.assertEqual(batch.width, 8)
        self.assertEqual(batch.output.shape, (3, 4, 4, 4))
        self.assertFalse(batch.extra["vividvr_debug"]["vae_tiling_enabled"])
        self.assertEqual(batch.sampling_params.runtime_progress, 1.0)


if __name__ == "__main__":
    unittest.main()
