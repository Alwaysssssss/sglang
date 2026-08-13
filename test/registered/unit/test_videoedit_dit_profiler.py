import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.multimodal_gen.configs.sample.videoedit_wan import (
    WanVideoEditSamplingParams,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    VideoRepairRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.video_api import (
    _validate_video_repair_request,
)
from sglang.multimodal_gen.runtime.layers.attention import layer as attention_layer
from sglang.multimodal_gen.runtime.pipelines_core.executors.pipeline_executor import (
    PipelineExecutor,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.profiler import SGLDiffusionProfiler

REPO_ROOT = Path(__file__).resolve().parents[3]
DIAGNOSE_SCRIPT = REPO_ROOT / "scripts" / "videoedit_phase15_diagnose.py"
ATTENTION_BENCH_SCRIPT = REPO_ROOT / "scripts" / "videoedit_attention_bench.py"


def load_diagnose_module():
    module_name = "videoedit_phase15_diagnose_test"
    spec = importlib.util.spec_from_file_location(module_name, DIAGNOSE_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_attention_bench_module():
    module_name = "videoedit_attention_bench_test"
    spec = importlib.util.spec_from_file_location(module_name, ATTENTION_BENCH_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_serialized_fp8_variant_uses_checkpoint_quantization(tmp_path):
    module = load_diagnose_module()
    transformer_path = tmp_path / "transformer"
    transformer_path.mkdir()
    (transformer_path / "config.json").write_text(
        json.dumps(
            {
                "quantization_config": {
                    "quant_method": "fp8",
                    "activation_scheme": "dynamic",
                    "weight_scale_granularity": "channel",
                }
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        serve_executable="sglang",
        model_path=tmp_path / "model",
        transformer_path=transformer_path,
        host="0.0.0.0",
        port=30000,
        server_extra_arg=["--transformer-fp8-gemm-backend", "triton"],
    )
    variant = module.VARIANTS["fp8_serialized_layerwise"]

    module.validate_variant_checkpoint(args, variant)
    command = module.build_service_command(args, variant, tmp_path / "run")

    assert "--transformer-quantization" not in command
    assert command[command.index("--transformer-path") + 1] == str(transformer_path)


def test_static_fp8_variant_requires_static_activation_checkpoint(tmp_path):
    module = load_diagnose_module()
    transformer_path = tmp_path / "transformer"
    transformer_path.mkdir()
    (transformer_path / "config.json").write_text(
        json.dumps(
            {
                "quantization_config": {
                    "quant_method": "fp8",
                    "activation_scheme": "static",
                    "weight_scale_granularity": "channel",
                }
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(transformer_path=transformer_path)

    module.validate_variant_checkpoint(args, module.VARIANTS["fp8_static_layerwise"])
    with pytest.raises(ValueError, match="activation_scheme='dynamic'"):
        module.validate_variant_checkpoint(
            args, module.VARIANTS["fp8_serialized_layerwise"]
        )


def test_serialized_fp8_variant_rejects_non_channel_checkpoint(tmp_path):
    module = load_diagnose_module()
    transformer_path = tmp_path / "transformer"
    transformer_path.mkdir()
    (transformer_path / "config.json").write_text(
        json.dumps(
            {
                "quantization_config": {
                    "quant_method": "fp8",
                    "activation_scheme": "dynamic",
                }
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(transformer_path=transformer_path)

    with pytest.raises(ValueError, match="per-channel"):
        module.validate_variant_checkpoint(
            args, module.VARIANTS["fp8_serialized_layerwise"]
        )


def test_video_repair_request_accepts_profiler_fields():
    request = VideoRepairRequest(
        task_id="profile-test",
        prompt="repair video",
        video_input_path="/tmp/video.mp4",
        mask_input_path="/tmp/mask.mp4",
        profile=True,
        num_profiled_timesteps=2,
        profile_all_stages=False,
    )

    assert request.profile is True
    assert request.num_profiled_timesteps == 2
    assert request.profile_all_stages is False


def test_video_repair_request_rejects_zero_profiled_timesteps():
    request = VideoRepairRequest(
        task_id="profile-test",
        prompt="repair video",
        video_input_path="/tmp/video.mp4",
        mask_input_path="/tmp/mask.mp4",
        profile=True,
        num_profiled_timesteps=0,
    )

    with pytest.raises(ValueError, match="must be positive or -1"):
        _validate_video_repair_request(request)


def test_pipeline_profiler_runs_only_on_dump_rank():
    batch = SimpleNamespace(profile=True, is_warmup=False, request_id="request-1")
    executor = SimpleNamespace(rank=1)

    with (
        patch(
            "sglang.multimodal_gen.runtime.pipelines_core.executors.pipeline_executor.SGLDiffusionProfiler"
        ) as profiler_cls,
    ):
        with PipelineExecutor.profile_execution(executor, batch, dump_rank=0):
            pass

    profiler_cls.assert_not_called()


def test_pipeline_profiler_does_not_start_nested_kineto_session():
    batch = SimpleNamespace(
        profile=True,
        is_warmup=False,
        request_id="request-1",
        profile_all_stages=False,
        num_profiled_timesteps=1,
        num_inference_steps=4,
    )
    executor = SimpleNamespace(rank=0, _active_profiler=None)

    with patch(
        "sglang.multimodal_gen.runtime.pipelines_core.executors.pipeline_executor.SGLDiffusionProfiler"
    ) as profiler_cls:
        with PipelineExecutor.profile_execution(executor, batch, dump_rank=0):
            with PipelineExecutor.profile_execution(executor, batch, dump_rank=0):
                pass

    profiler_cls.assert_called_once()
    profiler_cls.return_value.stop.assert_called_once_with(dump_rank=0)
    assert executor._active_profiler is None


def test_videoedit_warmup_copy_does_not_publish_request_progress():
    request = Req(
        sampling_params=WanVideoEditSamplingParams(
            progress_path="/tmp/videoedit-progress.json"
        )
    )

    warmup_request = request.copy_as_warmup(warmup_steps=1)

    assert request.progress_path == "/tmp/videoedit-progress.json"
    assert warmup_request.is_warmup is True
    assert warmup_request.progress_path is None


def test_profiler_exports_machine_readable_operator_summary(tmp_path):
    events = [
        SimpleNamespace(
            key="sglang.fp8.gemm",
            count=3,
            device_type=SimpleNamespace(name="CPU"),
            cpu_time_total=12.0,
            self_cpu_time_total=2.0,
            device_time_total=40.0,
            self_device_time_total=0.0,
        )
    ]
    profiler = object.__new__(SGLDiffusionProfiler)
    profiler.request_id = "request-1"
    profiler.rank = 0
    profiler.profile_mode_id = "1 steps"
    profiler.profiler = SimpleNamespace(key_averages=lambda: events)

    trace_path = tmp_path / "request-1.trace.json.gz"
    summary_path = profiler._export_operator_summary(str(trace_path))
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))

    assert summary["request_id"] == "request-1"
    assert summary["events"][0]["name"] == "sglang.fp8.gemm"
    assert summary["events"][0]["device_time_total_us"] == 40.0


def test_operator_breakdown_uses_non_overlapping_profile_ranges(tmp_path):
    module = load_diagnose_module()
    profile_path = tmp_path / "profile.profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "request_id": "request-1",
                "rank": 0,
                "profile_mode": "1 steps",
                "events": [
                    {
                        "name": "cuda_kernel_a",
                        "count": 1,
                        "device_type": "cuda",
                        "self_device_time_total_us": 100.0,
                    },
                    {
                        "name": "sglang.fp8.activation_quant",
                        "device_type": "cuda",
                        "self_device_time_total_us": 10.0,
                    },
                    {
                        "name": "sglang.fp8.gemm",
                        "device_type": "cuda",
                        "self_device_time_total_us": 40.0,
                    },
                    {
                        "name": "sglang.fp8.gemm",
                        "device_type": "cpu",
                        "device_time_total_us": 0.0,
                    },
                    {
                        "name": "sglang.dit.attention.compute",
                        "device_type": "cuda",
                        "self_device_time_total_us": 20.0,
                    },
                    {
                        "name": "sglang.dit.attention.sp_communication",
                        "device_type": "cuda",
                        "self_device_time_total_us": 10.0,
                    },
                    {
                        "name": "ProfilerStep*",
                        "device_type": "cuda",
                        "self_device_time_total_us": 1000.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = module.summarize_operator_profile(profile_path)

    assert summary["summed_gpu_kernel_time_ms"] == 0.1
    assert summary["categories"]["fp8_gemm"]["device_time_ms"] == 0.04
    assert summary["categories"]["other_gpu_kernels"]["device_time_ms"] == 0.02
    assert summary["missing_profile_ranges"] == []


def test_empty_operator_profile_is_invalid(tmp_path):
    module = load_diagnose_module()

    summary = module.summarize_profile_dir(tmp_path)

    assert summary["status"] == "invalid"
    assert "no operator profile sidecar" in summary["validation_errors"][0]
    assert "no CUDA kernel time" in summary["validation_errors"][1]


def test_operator_breakdown_splits_attention_roles(tmp_path):
    module = load_diagnose_module()
    profile_path = tmp_path / "profile.profile.json"
    events = [
        {
            "name": "cuda_kernel_a",
            "count": 1,
            "device_type": "cuda",
            "self_device_time_total_us": 100.0,
        },
        {
            "name": "sglang.dit.attention.self.compute",
            "device_type": "cuda",
            "self_device_time_total_us": 30.0,
        },
        {
            "name": "sglang.dit.attention.text_cross.compute",
            "device_type": "cuda",
            "self_device_time_total_us": 4.0,
        },
        {
            "name": "sglang.dit.attention.image_cross.compute",
            "device_type": "cuda",
            "self_device_time_total_us": 2.0,
        },
        {
            "name": "sglang.dit.attention.sp_communication",
            "device_type": "cuda",
            "self_device_time_total_us": 5.0,
        },
    ]
    profile_path.write_text(
        json.dumps(
            {
                "request_id": "request-1",
                "rank": 0,
                "profile_mode": "1 steps",
                "events": events,
            }
        ),
        encoding="utf-8",
    )

    summary = module.summarize_operator_profile(profile_path)

    assert summary["categories"]["attention_self_compute"]["device_time_ms"] == 0.03
    assert (
        summary["categories"]["attention_text_cross_compute"]["device_time_ms"] == 0.004
    )
    assert (
        summary["categories"]["attention_image_cross_compute"]["device_time_ms"]
        == 0.002
    )
    assert summary["categories"]["attention_compute"]["device_time_ms"] == 0.036
    assert summary["categories"]["other_gpu_kernels"]["device_time_ms"] == 0.059
    assert summary["missing_profile_ranges"] == []


def test_attention_bench_loads_rank_zero_runtime_shapes(tmp_path):
    module = load_attention_bench_module()
    audit_path = tmp_path / "attention_runtime_audits.json"
    audit_path.write_text(
        json.dumps(
            [
                {
                    "rank": 1,
                    "profile_kind": "self",
                    "q_shape": [1, 10, 2, 128],
                    "k_shape": [1, 10, 2, 128],
                    "v_shape": [1, 10, 2, 128],
                },
                {
                    "rank": 0,
                    "profile_kind": "self",
                    "backend": "fa",
                    "dtype": "bfloat16",
                    "q_shape": [1, 20, 4, 128],
                    "k_shape": [1, 20, 4, 128],
                    "v_shape": [1, 20, 4, 128],
                },
            ]
        ),
        encoding="utf-8",
    )

    shapes = module.load_shapes(
        audit_path,
        ["self"],
        allow_default_shapes=False,
    )

    assert len(shapes) == 1
    assert shapes[0].q_shape == (1, 20, 4, 128)
    assert shapes[0].backend == "fa"
    assert shapes[0].source == str(audit_path)


def test_attention_bench_projects_dit_speedup():
    module = load_attention_bench_module()

    projection = module.projected_speedups(
        2.4,
        attention_fraction=0.7486,
        current_fp8_over_bf16=1.116,
    )

    assert projection["dit_over_current_fp8"] == pytest.approx(1.775, rel=1e-3)
    assert projection["dit_over_bf16_estimate"] == pytest.approx(1.981, rel=1e-3)


def test_usp_attention_rejects_unhonored_backend_override(monkeypatch):
    selected = {}

    class WrongBackend:
        @staticmethod
        def get_enum():
            return AttentionBackendEnum.FA

    def get_wrong_backend(head_size, dtype, supported_attention_backends):
        selected["head_size"] = head_size
        selected["dtype"] = dtype
        selected["supported"] = supported_attention_backends
        return WrongBackend

    monkeypatch.setattr(attention_layer, "get_compute_dtype", lambda: torch.bfloat16)
    monkeypatch.setattr(attention_layer, "get_attn_backend", get_wrong_backend)

    with pytest.raises(RuntimeError, match="request was not honored"):
        attention_layer.USPAttention(
            num_heads=4,
            head_size=128,
            backend_override=AttentionBackendEnum.SAGE_ATTN,
        )

    assert selected == {
        "head_size": 128,
        "dtype": torch.bfloat16,
        "supported": {AttentionBackendEnum.SAGE_ATTN},
    }
