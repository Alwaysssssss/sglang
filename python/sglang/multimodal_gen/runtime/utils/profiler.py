import gzip
import json
import os
from contextlib import nullcontext

import torch

from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.logging_utils import CYAN, RESET, init_logger

if current_platform.is_npu():
    import torch_npu

    patches = [
        ["profiler.profile", torch_npu.profiler.profile],
        ["profiler.schedule", torch_npu.profiler.schedule],
    ]
    torch_npu._apply_patches(patches)

logger = init_logger(__name__)

_PROFILE_RANGES_ENABLED = os.getenv(
    "SGLANG_DIT_PROFILE_RANGES", ""
).strip().lower() in {"1", "true", "yes", "on"}


def diffusion_profile_range(name: str):
    """Create a torch profiler range only for explicit DiT profiling runs."""
    if not _PROFILE_RANGES_ENABLED:
        return nullcontext()
    return torch.profiler.record_function(name)


class SGLDiffusionProfiler:
    """
    A wrapper around torch.profiler to simplify usage in pipelines.
    Supports both full profiling and scheduled profiling.


    1. if profile_all_stages is on: profile all stages, including all denoising steps
    2. otherwise, if num_profiled_timesteps is specified: profile {num_profiled_timesteps} denoising steps. profile all steps if num_profiled_timesteps==-1
    """

    _instance = None

    def __init__(
        self,
        request_id: str | None = None,
        rank: int = 0,
        full_profile: bool = False,
        num_steps: int | None = None,
        num_inference_steps: int | None = None,
        log_dir: str | None = None,
    ):
        self.request_id = request_id or "profile_trace"
        self.rank = rank
        self.full_profile = full_profile

        self.log_dir = (
            log_dir
            if log_dir is not None
            else os.getenv("SGLANG_TORCH_PROFILER_DIR", "./logs")
        )

        try:
            os.makedirs(self.log_dir, exist_ok=True)
        except OSError:
            pass

        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available() or (
            hasattr(torch, "musa") and torch.musa.is_available()
        ):
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        if current_platform.is_npu():
            activities.append(torch_npu.profiler.ProfilerActivity.NPU)

        with_stack = os.getenv(
            "SGLANG_TORCH_PROFILER_WITH_STACK", "1"
        ).strip().lower() in {"1", "true", "yes", "on"}
        common_torch_profiler_args = dict(
            activities=activities,
            record_shapes=True,
            with_stack=with_stack,
            acc_events=True,
            on_trace_ready=(
                None
                if not current_platform.is_npu()
                else torch_npu.profiler.tensorboard_trace_handler(self.log_dir)
            ),
        )
        if self.full_profile:
            # profile all stages
            self.profiler = torch.profiler.profile(**common_torch_profiler_args)
            self.profile_mode_id = "full stages"
        else:
            # profile denoising stage only
            warmup = 0 if num_steps == -1 else 1
            num_actual_steps = num_inference_steps if num_steps == -1 else num_steps
            if num_actual_steps is None or num_actual_steps <= 0:
                raise ValueError("num_steps must be positive or -1")
            self.num_remaining_step_calls = num_actual_steps + warmup
            self.profiler = torch.profiler.profile(
                **common_torch_profiler_args,
                schedule=torch.profiler.schedule(
                    skip_first=0,
                    wait=0,
                    warmup=warmup,
                    active=num_actual_steps,
                    repeat=1,
                ),
            )
            self.profile_mode_id = f"{num_actual_steps} steps"

        logger.info(
            f"Profiling request: {request_id} for {self.profile_mode_id} "
            f"(rank={self.rank}, pid={os.getpid()})..."
        )

        self.has_stopped = False

        SGLDiffusionProfiler._instance = self
        self.start()

    def start(self):
        logger.info(
            f"Starting Profiler (rank={self.rank}, pid={os.getpid()})..."
        )
        self.profiler.start()

    def _step(self):
        self.profiler.step()

    def step_stage(self):
        if self.full_profile:
            self._step()

    def step_denoising_step(self):
        if not self.full_profile:
            if self.num_remaining_step_calls > 0:
                self._step()
                self.num_remaining_step_calls -= 1
            else:
                # early exit when enough steps are captured, to reduce the trace file size
                self.stop(dump_rank=0)

    @classmethod
    def get_instance(cls) -> "SGLDiffusionProfiler":
        return cls._instance

    def stop(self, export_trace: bool = True, dump_rank: int | None = None):
        if self.has_stopped:
            return
        self.has_stopped = True
        logger.info("Stopping Profiler...")
        if torch.cuda.is_available() or (
            hasattr(torch, "musa") and torch.musa.is_available()
        ):
            torch.cuda.synchronize()
        if current_platform.is_npu():
            torch.npu.synchronize()
            export_trace = False  # set to false because our internal torch_npu.profiler will generate trace file
        self.profiler.stop()

        if export_trace:
            if dump_rank is not None and dump_rank != self.rank:
                pass
            else:
                self._export_trace()

        SGLDiffusionProfiler._instance = None

    def _export_trace(self):

        try:
            os.makedirs(self.log_dir, exist_ok=True)
            sanitized_profile_mode_id = self.profile_mode_id.replace(" ", "_")
            trace_path = os.path.abspath(
                os.path.join(
                    self.log_dir,
                    f"{self.request_id}-{sanitized_profile_mode_id}-global-rank{self.rank}.trace.json.gz",
                )
            )
            self.profiler.export_chrome_trace(trace_path)
            summary_path = self._export_operator_summary(trace_path)

            if self._check_trace_integrity(trace_path):
                logger.info(f"Saved profiler traces to: {CYAN}{trace_path}{RESET}")
                logger.info(
                    f"Saved profiler operator summary to: {CYAN}{summary_path}{RESET}"
                )
            else:
                logger.warning(f"Trace file may be corrupted: {trace_path}")
        except Exception as e:
            logger.error(f"Failed to save trace: {e}")

    def _export_operator_summary(self, trace_path: str) -> str:
        suffix = ".trace.json.gz"
        base_path = (
            trace_path[: -len(suffix)] if trace_path.endswith(suffix) else trace_path
        )
        summary_path = f"{base_path}.profile.json"
        events = []
        for event in self.profiler.key_averages():
            events.append(
                {
                    "name": str(event.key),
                    "count": int(event.count),
                    "device_type": str(event.device_type).rsplit(".", 1)[-1].lower(),
                    "cpu_time_total_us": float(event.cpu_time_total),
                    "self_cpu_time_total_us": float(event.self_cpu_time_total),
                    "device_time_total_us": float(event.device_time_total),
                    "self_device_time_total_us": float(event.self_device_time_total),
                }
            )
        events.sort(
            key=lambda event: (
                event["self_device_time_total_us"],
                event["device_time_total_us"],
            ),
            reverse=True,
        )
        payload = {
            "request_id": self.request_id,
            "rank": self.rank,
            "profile_mode": self.profile_mode_id,
            "time_unit": "microseconds",
            "trace_path": trace_path,
            "events": events,
        }
        tmp_path = f"{summary_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=True, indent=2)
        os.replace(tmp_path, summary_path)
        return summary_path

    def _check_trace_integrity(self, trace_path: str) -> bool:
        try:
            if not os.path.exists(trace_path) or os.path.getsize(trace_path) == 0:
                return False

            with gzip.open(trace_path, "rb") as f:
                content = f.read()
                if content.count(b"\x1f\x8b") > 1:
                    logger.warning("Multiple gzip headers detected")
                    return False

            return True
        except Exception as e:
            logger.warning(f"Trace file integrity check failed: {e}")
            return False
