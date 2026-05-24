from types import SimpleNamespace
from unittest.mock import patch

import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker
from sglang.multimodal_gen.runtime.pipelines_core.stages.star_latent_preparation import (
    STARLatentPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.video_condition_vae_encoding import (
    STARConditionVideoVAEEncodingStage,
)


class _DummyModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.param = torch.nn.Parameter(torch.zeros(1))
        self.moves: list[str] = []

    def to(self, *args, **kwargs):
        device = args[0] if args else kwargs.get("device", "unknown")
        self.moves.append(str(device))
        return super().to(*args, **kwargs)


class _DummyVAE(_DummyModule):
    def encode(self, video: torch.Tensor) -> DiagonalGaussianDistribution:
        batch_size = video.shape[0]
        params = torch.zeros(
            batch_size,
            8,
            1,
            2,
            2,
            device=video.device,
            dtype=video.dtype,
        )
        return DiagonalGaussianDistribution(params)


class _DummyPipeline:
    def __init__(self, transformer: _DummyModule) -> None:
        self.transformer = transformer

    def get_module(self, module_name: str, default=None):
        if module_name == "transformer":
            return self.transformer
        return default


def _build_server_args(
    *,
    peak_memory_mode: str = "text_encoder_and_transformer",
    keep_transformer_gpu_resident_between_requests: bool = False,
) -> SimpleNamespace:
    pipeline_config = SimpleNamespace(
        vae_precision="fp32",
        vae_tiling=False,
        latent_channels=4,
        postprocess_vae_encode=lambda latent, vae: latent,
        normalize_vae_encode=lambda latent, vae: None,
        get_decode_scale_and_shift=lambda device, dtype, vae: (1.0, None),
        postprocess_image_latent=lambda latent, batch: latent,
        condition_video_vae_target_headroom_gb=2.0,
        keep_transformer_gpu_resident_between_requests=keep_transformer_gpu_resident_between_requests,
        resolve_condition_video_vae_peak_memory_mode=lambda: peak_memory_mode,
        release_text_encoder_after_prompt_encode=True,
        temporarily_offload_transformer_during_condition_vae_encode=True,
        dit_config=SimpleNamespace(
            arch_config=SimpleNamespace(num_channels_latents=4)
        ),
        vae_config=SimpleNamespace(
            arch_config=SimpleNamespace(
                temporal_compression_ratio=4,
                spatial_compression_ratio=8,
            ),
            use_temporal_scaling_frames=True,
            encode_sample_mode=lambda: "argmax",
        ),
        get_latent_dtype=lambda prompt_dtype: prompt_dtype,
    )
    return SimpleNamespace(
        pipeline_config=pipeline_config,
        disable_autocast=True,
        vae_cpu_offload=False,
        dit_cpu_offload=False,
        use_fsdp_inference=False,
    )


def test_condition_video_vae_stage_releases_peak_memory_modules():
    server_args = _build_server_args()
    transformer = _DummyModule()
    text_encoder = _DummyModule()
    with patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages.base.get_global_server_args",
        return_value=server_args,
    ):
        stage = STARConditionVideoVAEEncodingStage(
            vae=_DummyVAE(),
            transformer=transformer,
            text_encoders=[text_encoder],
        )
    stage.server_args = server_args

    batch = SimpleNamespace(
        condition_video=torch.zeros(1, 1, 3, 16, 16),
        condition_video_num_frames=1,
        batch_size=1,
        generator=None,
        height=16,
        width=16,
        num_frames=7,
        image_latent=None,
        extra={},
    )

    simulated_devices = {
        id(transformer): "cuda",
        id(text_encoder): "cuda",
    }

    with patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages.video_condition_vae_encoding.get_local_torch_device",
        return_value=torch.device("cpu"),
    ), patch.object(
        STARConditionVideoVAEEncodingStage,
        "_module_device",
        side_effect=lambda module: simulated_devices.get(id(module), "cpu"),
    ):
        result = stage.forward(batch, server_args)

    assert result.extra["star_reload_transformer_before_denoise"] is True
    assert result.condition_video is None
    assert tuple(result.image_latent.shape) == (1, 4, 1, 2, 2)
    assert "cpu" in text_encoder.moves
    assert "cpu" in transformer.moves


def test_latent_preparation_reloads_transformer_before_denoising():
    server_args = _build_server_args()
    transformer = _DummyModule()
    with patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages.base.get_global_server_args",
        return_value=server_args,
    ):
        stage = STARLatentPreparationStage(
            scheduler=SimpleNamespace(init_noise_sigma=1.0),
            transformer=transformer,
        )

    batch = SimpleNamespace(
        batch_size=1,
        prompt_embeds=[torch.zeros(1, 1, 4)],
        generator=torch.Generator(device="cpu"),
        extra={"star_reload_transformer_before_denoise": True},
        latents=torch.zeros(1, 4, 1, 2, 2),
        height=16,
        width=16,
        num_frames=1,
        raw_latent_shape=None,
    )

    with patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages.star_latent_preparation.get_local_torch_device",
        return_value=torch.device("cpu"),
    ):
        result = stage.forward(batch, server_args)

    assert "star_reload_transformer_before_denoise" not in result.extra
    assert "cpu" in transformer.moves


def test_condition_video_vae_stage_auto_mode_offloads_transformer_when_headroom_low():
    server_args = _build_server_args(peak_memory_mode="auto")
    transformer = _DummyModule()
    text_encoder = _DummyModule()
    with patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages.base.get_global_server_args",
        return_value=server_args,
    ):
        stage = STARConditionVideoVAEEncodingStage(
            vae=_DummyVAE(),
            transformer=transformer,
            text_encoders=[text_encoder],
        )
    stage.server_args = server_args

    batch = SimpleNamespace(
        condition_video=torch.zeros(1, 4, 3, 16, 16),
        condition_video_num_frames=4,
        batch_size=1,
        generator=None,
        height=16,
        width=16,
        num_frames=7,
        image_latent=None,
        extra={},
    )

    simulated_devices = {
        id(transformer): "cuda",
        id(text_encoder): "cuda",
    }

    with patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages.video_condition_vae_encoding.get_local_torch_device",
        return_value=torch.device("cpu"),
    ), patch.object(
        STARConditionVideoVAEEncodingStage,
        "_module_device",
        side_effect=lambda module: simulated_devices.get(id(module), "cpu"),
    ), patch.object(
        STARConditionVideoVAEEncodingStage,
        "_get_free_gpu_memory_bytes",
        return_value=64,
    ):
        result = stage.forward(batch, server_args)

    assert result.extra["star_reload_transformer_before_denoise"] is True
    assert result.extra["star_condition_vae_peak_memory"]["offloaded_transformer"] is True


def test_condition_video_vae_stage_auto_mode_keeps_transformer_when_headroom_is_enough():
    server_args = _build_server_args(peak_memory_mode="auto")
    transformer = _DummyModule()
    text_encoder = _DummyModule()
    with patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages.base.get_global_server_args",
        return_value=server_args,
    ):
        stage = STARConditionVideoVAEEncodingStage(
            vae=_DummyVAE(),
            transformer=transformer,
            text_encoders=[text_encoder],
        )
    stage.server_args = server_args

    batch = SimpleNamespace(
        condition_video=torch.zeros(1, 1, 3, 16, 16),
        condition_video_num_frames=1,
        batch_size=1,
        generator=None,
        height=16,
        width=16,
        num_frames=7,
        image_latent=None,
        extra={},
    )

    simulated_devices = {
        id(transformer): "cuda",
        id(text_encoder): "cuda",
    }

    with patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages.video_condition_vae_encoding.get_local_torch_device",
        return_value=torch.device("cpu"),
    ), patch.object(
        STARConditionVideoVAEEncodingStage,
        "_module_device",
        side_effect=lambda module: simulated_devices.get(id(module), "cpu"),
    ), patch.object(
        STARConditionVideoVAEEncodingStage,
        "_get_free_gpu_memory_bytes",
        return_value=8 * (1024**3),
    ):
        result = stage.forward(batch, server_args)

    assert "star_reload_transformer_before_denoise" not in result.extra
    assert result.extra["star_condition_vae_peak_memory"]["offloaded_transformer"] is False


def test_gpu_worker_respects_keep_transformer_resident_flag():
    transformer = _DummyModule().to("cpu")
    worker = SimpleNamespace(
        server_args=_build_server_args(
            keep_transformer_gpu_resident_between_requests=True
        ),
        pipeline=_DummyPipeline(transformer),
    )
    worker.server_args.dit_cpu_offload = True

    GPUWorker._reclaim_dit_gpu_residency_after_request(worker)

    assert transformer.moves == ["cpu"]
