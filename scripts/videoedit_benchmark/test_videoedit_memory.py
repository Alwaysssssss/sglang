"""Real CPU tensor/stage tests without importing the GPU serving stack.

Run with: python -m unittest scripts.videoedit_benchmark.test_videoedit_memory -v
CUDA cases automatically run when a GPU is available.
"""

import logging
import os
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from PIL import Image

from scripts.videoedit_benchmark.test_overlap_pipeline import (
    Event, Stream, load_module, memory,
)
from scripts.videoedit_benchmark import test_overlap_pipeline as overlap


MEMORY_PATH = "sglang.multimodal_gen.runtime.utils.videoedit_memory"
RESIDENCY_PATH = "sglang.multimodal_gen.runtime.utils.readonly_model_residency"
residency = load_module("residency_under_test", "utils/readonly_model_residency.py",
                        {MEMORY_PATH: memory})
contracts = load_module("contracts_under_test", "videoedit/contracts.py", {})
windowing = load_module("windowing_under_test", "videoedit/windowing.py", {
    "sglang.multimodal_gen.runtime.videoedit.contracts": contracts,
})
mask_io = load_module("mask_io_under_test", "videoedit/mask_io.py", {})
preprocess = load_module("preprocess_under_test", "videoedit/preprocess.py", {
    "sglang.multimodal_gen.runtime.videoedit.contracts": contracts,
    "sglang.multimodal_gen.runtime.videoedit.windowing": windowing,
    "sglang.multimodal_gen.runtime.videoedit.mask_io": mask_io,
})
progress = load_module("progress_under_test", "videoedit/progress.py", {})
scheduler_module = load_module("scheduler_under_test", "models/schedulers/videoedit_flow_match.py", {})


class TextStage:
    def __init__(self, encoders, tokenizers):
        self.text_encoders = encoders

    def encode_text(self, text, *args, **kwargs):
        return [self.text_encoders[0](torch.ones(1, 4))], None, None


stages = load_module("stages_under_test", "pipelines_core/stages/model_specific_stages/videoedit_wan.py", {
    "sglang.multimodal_gen.runtime.utils.logging_utils": SimpleNamespace(init_logger=logging.getLogger),
    MEMORY_PATH: memory,
    RESIDENCY_PATH: residency,
    "sglang.multimodal_gen.configs.sample.videoedit_wan": SimpleNamespace(
        WanVideoEditSamplingParams=SimpleNamespace,
        DEFAULT_VIDEOEDIT_NEGATIVE_PROMPT="default negative",
    ),
    "sglang.multimodal_gen.runtime.distributed": SimpleNamespace(
        get_local_torch_device=lambda: torch.device("cpu"),
        get_sp_group=lambda: None,
    ),
    "sglang.multimodal_gen.runtime.managers.forward_context": SimpleNamespace(set_forward_context=lambda **kw: nullcontext()),
    "sglang.multimodal_gen.runtime.pipelines_core.schedule_batch": SimpleNamespace(Req=object),
    "sglang.multimodal_gen.runtime.pipelines_core.stages.base": SimpleNamespace(PipelineStage=torch.nn.Module),
    "sglang.multimodal_gen.runtime.pipelines_core.stages.denoising": SimpleNamespace(DenoisingStage=torch.nn.Module),
    "sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding": SimpleNamespace(TextEncodingStage=TextStage),
    "sglang.multimodal_gen.runtime.pipelines_core.stages.validators": SimpleNamespace(V=object, VerificationResult=object),
    "sglang.multimodal_gen.runtime.platforms": SimpleNamespace(current_platform=SimpleNamespace(device_type="cpu")),
    "sglang.multimodal_gen.runtime.request_timeout": SimpleNamespace(check_request_timeout=lambda batch: None),
    "sglang.multimodal_gen.runtime.server_args": SimpleNamespace(ServerArgs=object),
    "sglang.multimodal_gen.runtime.videoedit.preprocess": preprocess,
    "sglang.multimodal_gen.runtime.videoedit.progress": progress,
    "sglang.multimodal_gen.utils": SimpleNamespace(PRECISION_TO_TYPE={"fp32": torch.float32, "bf16": torch.bfloat16}),
})


class VAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(z_dim=3, latents_mean=[0] * 3, latents_std=[1] * 3)
        self.cache = None
        self.fail = False
        self.use_feature_cache = True

    def encode(self, x):
        self.cache = x
        if self.fail:
            raise RuntimeError("encode failed")
        value = x[:, :, ::4, ::8, ::8] * self.weight
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: value))

    def decode(self, x):
        self.cache = x
        if self.fail:
            raise RuntimeError("decode failed")
        return SimpleNamespace(sample=x * self.weight)

    def clear_cache(self):
        self.cache = None

    def enable_tiling(self):
        pass

    def disable_tiling(self):
        pass


def request():
    params = SimpleNamespace(
        runtime_window_validated=True, dtype="fp32", mask_downsample_mode="nearest",
        runtime_window_frames=[Image.new("RGB", (16, 16), (i * 20, 50, 90)) for i in range(5)],
        runtime_window_masks=[Image.new("L", (16, 16), 255) for _ in range(5)],
    )
    args = SimpleNamespace(vae_cpu_offload=True, disable_autocast=True,
        pipeline_config=SimpleNamespace(vae_precision="fp32", vae_tiling=True))
    return params, SimpleNamespace(sampling_params=params), args


class MemoryTests(unittest.TestCase):
    def test_preprocessing_omits_unused_video_without_changing_condition(self):
        params, _, _ = request()
        full = preprocess.prepare_window_inputs(params.runtime_window_frames, params.runtime_window_masks,
                                               "cpu", torch.float32)
        lean = preprocess.prepare_window_inputs(params.runtime_window_frames, params.runtime_window_masks,
                                               "cpu", torch.float32, return_intermediates=False)
        self.assertIsNone(lean["video_tensor"])
        self.assertIsNone(lean["mask_video_tensor"])
        for key in ("masked_video_tensor", "cond_masks"):
            torch.testing.assert_close(full[key], lean[key], rtol=0, atol=0)

    def test_encode_decode_repeatedly_restore_weights_and_clear_tensors(self):
        vae = VAE()
        pointer = vae.weight.data_ptr()
        encoder, decoder = stages.VideoEditConditionEncodingStage(vae), stages.VideoEditDecodingStage(vae)
        for iteration in range(3):
            params, batch, args = request()
            with patch.dict(os.environ, SGLANG_VIDEOEDIT_KEEP_STAGE_TENSORS="0"):
                encoder.forward(batch, args)
                self.assertIsNone(params.runtime_masked_video_tensor)
                self.assertIsNone(params.runtime_mask_video_tensor)
                self.assertIsNone(params.runtime_condition_latent)
                self.assertIsNone(vae.cache)
                self.assertEqual(vae.weight.data_ptr(), pointer)
                params.runtime_latents = params.runtime_cond_latents
                expected = memory.decoded_frames_to_images(
                    ((params.runtime_latents * vae.weight / 2 + 0.5).clamp(0, 1))[0].permute(1, 0, 2, 3).detach())
                decoder.forward(batch, args)
                for actual, ref in zip(params.runtime_window_output_frames, expected):
                    np.testing.assert_array_equal(np.asarray(actual), np.asarray(ref))
                self.assertIsNone(params.runtime_latents)
                self.assertIsNone(batch.latents)
                self.assertIsNone(params.runtime_decoded_video_tensor)
                self.assertIsNone(vae.cache)
                self.assertTrue(vae.use_feature_cache)
                self.assertEqual(vae.weight.data_ptr(), pointer)
            with torch.no_grad():
                vae.weight.mul_(0.9)

    def test_vae_failure_restores_cpu_and_clears_feature_cache(self):
        for decoding in (False, True):
            vae = VAE()
            vae.fail = True
            pointer = vae.weight.data_ptr()
            params, batch, args = request()
            params.runtime_latents = torch.zeros(1, 3, 2, 2, 2)
            stage = stages.VideoEditDecodingStage(vae) if decoding else stages.VideoEditConditionEncodingStage(vae)
            with self.assertRaisesRegex(RuntimeError, "failed"):
                stage.forward(batch, args)
            self.assertIsNone(vae.cache)
            self.assertEqual(vae.weight.data_ptr(), pointer)
            self.assertTrue(vae.use_feature_cache)

    def test_decode_pixels_match_original_fp32_conversion(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            source = torch.linspace(-0.1, 1.1, 7 * 3 * 8 * 8).reshape(7, 8, 8, 3).permute(0, 3, 1, 2).to(dtype)
            expected = [(frame.clamp(0, 1).numpy() * 255).astype("uint8")
                        for frame in source.float().cpu().permute(0, 2, 3, 1)]
            for size in (1, 4, 16):
                actual = memory.decoded_frames_to_images(source, size)
                self.assertEqual(len(actual), len(expected))
                for image, reference in zip(actual, expected):
                    np.testing.assert_array_equal(np.asarray(image), reference)

    def test_budget_policy_covers_full_partial_and_no_lookahead(self):
        self.assertEqual(memory.select_weight_budget([10] * 8, 80, 4), (8, 1))
        self.assertEqual(memory.select_weight_budget([10] * 8, 10, 4), (0, 0))
        self.assertEqual(memory.select_weight_budget([10] * 8, 50, 2), (2, 2))
        with self.assertRaises(ValueError):
            memory.select_weight_budget([10, 20], 19, 1)
        for budget in (1088, 2176, 4352, 9792):
            overlap.SchedulingTests().check_manager(4, 0, False, budget=budget)

    def test_background_prefetch_is_joined_and_restored_on_success_and_failure(self):
        for fail in (False, True):
            model = torch.nn.Linear(4, 4)
            pointer = model.weight.data_ptr()
            params = SimpleNamespace()
            with patch.object(torch.cuda, "device", side_effect=lambda device: nullcontext()):
                params.runtime_encoder_prefetch = residency.StagePrefetch(model, torch.device("cpu"))
                if fail:
                    # Abandon the following stage, as on timeout/verification failure.
                    residency.close_encoder_prefetch(params)
                else:
                    with params.runtime_encoder_prefetch as loaded:
                        torch.testing.assert_close(loaded(torch.ones(1, 4)), model(torch.ones(1, 4)))
                    residency.close_encoder_prefetch(params)
            self.assertIsNone(params.runtime_encoder_prefetch)
            self.assertEqual(model.weight.data_ptr(), pointer)

    def test_full_cpu_stage_chain_matches_retained_intermediate_reference(self):
        class DiT(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.1))

            def forward(self, hidden_states, encoder_hidden_states, **kwargs):
                return hidden_states[:, :3] * self.weight + encoder_hidden_states.mean()

        results = []
        for keep in ("1", "0"):
            params, batch, args = request()
            params.__dict__.update(seed=17, num_inference_steps=3, runtime_do_cfg=True,
                runtime_prompt_embeds=torch.ones(1, 2), runtime_negative_prompt_embeds=torch.zeros(1, 2),
                runtime_image_embeds=None, guidance_scale=5., dynamic_cfg_max_step=2,
                dynamic_cfg_min=1., dynamic_cfg=True, progress_path=None,
                runtime_num_input_frames=5, infer_len=5, overlap=1, runtime_window_specs=None,
                runtime_progress_window_index=None, runtime_window_index=0)
            args.enable_cfg_parallel = False
            args.pipeline_config.dit_precision = "fp32"
            args.pipeline_config.flow_shift = 5.
            vae, dit = VAE(), DiT()
            scheduler = scheduler_module.VideoEditFlowMatchScheduler()
            denoise = SimpleNamespace(transformer=dit, scheduler=scheduler,
                _manage_device_placement=lambda *args: None,
                _maybe_enable_cache_dit=lambda *args: None,
                _build_attn_metadata=lambda *args, **kwargs: None,
                progress_bar=lambda **kwargs: nullcontext())
            denoise._forward_denoising = lambda batch, args: stages.VideoEditDenoisingStage._forward_denoising(denoise, batch, args)
            with patch.dict(os.environ, SGLANG_VIDEOEDIT_KEEP_STAGE_TENSORS=keep):
                stages.VideoEditConditionEncodingStage(vae).forward(batch, args)
                stages.VideoEditLatentPreparationStage().forward(batch, args)
                stages.VideoEditTimestepPreparationStage(scheduler).forward(batch, args)
                stages.VideoEditLatentInitStage(scheduler).forward(batch, args)
                stages.VideoEditDenoisingStage.forward(denoise, batch, args)
                latent = params.runtime_latents.clone()
                if keep == "0":
                    self.assertIsNone(params.runtime_cond_latents)
                    self.assertIsNone(params.runtime_prompt_embeds)
                    self.assertIsNone(params.runtime_noise)
                stages.VideoEditDecodingStage(vae).forward(batch, args)
                results.append((latent, [np.asarray(image) for image in params.runtime_window_output_frames]))
        torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
        for expected, actual in zip(results[0][1], results[1][1]):
            np.testing.assert_array_equal(actual, expected)

    def test_denoising_failure_releases_condition_and_weight_cache(self):
        params, batch, args = request()
        params.runtime_cond_latents = torch.ones(1)
        manager = SimpleNamespace(auto_residency=True, release_all=Mock())
        stage = SimpleNamespace(transformer=SimpleNamespace(layerwise_offload_managers=[manager]),
            _forward_denoising=Mock(side_effect=RuntimeError("denoise failed")))
        with self.assertRaisesRegex(RuntimeError, "denoise failed"):
            stages.VideoEditDenoisingStage.forward(stage, batch, args)
        manager.release_all.assert_called_once()
        self.assertIsNone(params.runtime_cond_latents)

    def test_pinned_pool_is_bounded_and_failure_is_observable(self):
        source = torch.arange(1000, dtype=torch.float32)
        target = torch.empty_like(source)
        pool = memory.PinnedStagingPool(256)
        empty = torch.empty
        events = []

        class CopyEvent(Event):
            def synchronize(self):
                self.waited = True

        def event():
            result = CopyEvent()
            events.append(result)
            return result

        def pageable_for_test(*args, **kwargs):
            kwargs.pop("pin_memory", None)
            return empty(*args, **kwargs)

        with patch.object(torch, "empty", side_effect=pageable_for_test), patch.object(torch.cuda, "Event", side_effect=event):
            pool.copy_to(target, source, Stream())
            torch.testing.assert_close(target, source, rtol=0, atol=0)
            self.assertLessEqual(sum(t.numel() for t in pool.slots), 256)
            pool.close()
            self.assertTrue(all(e.waited for e in events))
        pool = memory.PinnedStagingPool(256)
        with patch.object(torch, "empty", side_effect=RuntimeError("pin denied")):
            pool.copy_to(target, source, Stream())
        self.assertTrue(pool.disabled)
        self.assertEqual(pool.fallback_copies, 1)
        self.assertEqual(pool.pinned_bytes, 0)
        torch.testing.assert_close(target, source, rtol=0, atol=0)

    def test_residency_respects_aliases_dtype_and_cpu_updates(self):
        model = torch.nn.Linear(4, 4).eval()
        model.tied = model.weight
        model.register_buffer("counter", torch.tensor(2, dtype=torch.int64))
        model.register_buffer("shared_counter", model.counter)
        pointer = model.weight.data_ptr()
        for _ in range(3):
            with residency.readonly_model_on_device(model, torch.device("cpu"), dtype=torch.float64):
                self.assertEqual(model.weight.dtype, torch.float64)
                self.assertEqual(model.counter.dtype, torch.int64)
                self.assertIs(model.weight, model.tied)
                self.assertIs(model.counter, model.shared_counter)
            self.assertEqual(model.weight.data_ptr(), pointer)
            self.assertEqual(model.weight.dtype, torch.float32)
            with torch.no_grad():
                model.weight.add_(1)

    def test_consolidated_upload_preserves_values_with_bounded_staging(self):
        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
        model.register_buffer("counter", torch.tensor(3))
        x = torch.ones(1, 4)
        expected = model(x).detach()
        original_storage = model[0].weight.data_ptr()
        empty = torch.empty

        class CopyEvent(Event):
            def synchronize(self):
                pass

        def simulated_pin(*args, **kwargs):
            kwargs.pop("pin_memory", None)
            return empty(*args, **kwargs)

        with patch.dict(os.environ, SGLANG_VIDEOEDIT_STAGE_PINNED_MB="1"), \
             patch.object(torch.cuda, "Stream", side_effect=lambda **kw: Stream()), \
             patch.object(torch.cuda, "stream", side_effect=lambda stream: nullcontext()), \
             patch.object(torch.cuda, "current_stream", return_value=Stream()), \
             patch.object(torch.cuda, "Event", CopyEvent), \
             patch.object(torch.Tensor, "record_stream", return_value=None), \
             patch.object(torch, "empty", side_effect=simulated_pin), torch.no_grad():
            handle = residency.ModelResidency(model, torch.device("cpu"))
            # Simulate CUDA stream mechanics while retaining real CPU tensor math.
            handle.asynchronous = True
            with handle:
                torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
                self.assertEqual(model[0].weight.untyped_storage().data_ptr(),
                                 model[1].weight.untyped_storage().data_ptr())
                self.assertEqual(model.counter.item(), 3)
                self.assertLessEqual(sum(t.numel() for t in handle.pool.slots), 1024**2)
            self.assertEqual(model[0].weight.data_ptr(), original_storage)

    def test_partial_upload_and_forward_failure_restore_originals(self):
        for partial in (True, False):
            model = torch.nn.Linear(4, 4)
            pointer = model.weight.data_ptr()
            original_to = torch.Tensor.to
            calls = []

            def upload(tensor, *args, **kwargs):
                calls.append(tensor)
                if partial and len(calls) == 2:
                    raise RuntimeError("partial upload")
                return original_to(tensor, *args, **kwargs)

            with patch.object(torch.Tensor, "to", upload):
                with self.assertRaises(RuntimeError):
                    with residency.readonly_model_on_device(model, torch.device("cpu"), dtype=torch.float64):
                        raise RuntimeError("forward failed")
            self.assertEqual(model.weight.data_ptr(), pointer)
            self.assertEqual(model.weight.dtype, torch.float32)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class CUDAMemoryTests(unittest.TestCase):
    def test_async_residency_and_d2h_match_cpu_reference(self):
        model = torch.nn.Linear(1024, 1024).eval()
        cpu_x = torch.randn(8, 1024)
        expected = model(cpu_x).detach()
        pointer = model.weight.data_ptr()
        for _ in range(3):
            stream = torch.cuda.Stream()
            with patch.dict(os.environ, SGLANG_VIDEOEDIT_STAGE_PINNED_MB="1"), \
                 torch.no_grad(), torch.cuda.stream(stream):
                with residency.ModelResidency(model, torch.device("cuda"), asynchronous=True):
                    actual = model(cpu_x.cuda())
            stream.synchronize()
            torch.testing.assert_close(actual.cpu(), expected)
            self.assertEqual(model.weight.data_ptr(), pointer)
        source = torch.rand(9, 3, 16, 16, dtype=torch.bfloat16)
        expected_images = memory.decoded_frames_to_images(source)
        actual_images = memory.decoded_frames_to_images(source.cuda())
        for actual, expected_image in zip(actual_images, expected_images):
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected_image))

    def test_layerwise_staging_repeated_forwards_and_disable_offload(self):
        with patch.dict(os.environ, SGLANG_VIDEOEDIT_OFFLOAD_PINNED_MB="1"):
            overlap.check_cuda_buffer_lifetimes()
            reference = overlap.Model().cuda().eval()
            model = overlap.Model().eval()
            model.load_state_dict(reference.state_dict())
            with patch.dict(os.environ, SGLANG_VIDEOEDIT_OVERLAP_PREFETCH="1",
                            SGLANG_VIDEOEDIT_RESIDENT_LAYERS="0"):
                manager = overlap.offload.LayerwiseOffloadManager(
                    model, layers_attr_str="blocks", num_layers=9, enabled=True)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.inference_mode(), torch.cuda.stream(stream):
                manager.remove_forward_hooks()
                manager.load_all_layers()
                x = torch.randn(8, 16, device="cuda")
                actual, expected = model(x), reference(x)
            stream.synchronize()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertLessEqual(manager.staging_pool.pinned_bytes, 1024**2)
            self.assertEqual(manager.pinned_weight_bytes, 0)
            manager.release_all()
            manager.staging_pool.close()


if __name__ == "__main__":
    unittest.main()
