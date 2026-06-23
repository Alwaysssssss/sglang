# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import gc
import importlib
import importlib.util
import json
import logging
import multiprocessing
import os
import sys
import sysconfig
import time
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from pydantic import BaseModel, Field, ValidationError

try:
    import uvicorn
except ImportError:  # pragma: no cover - exercised in the original Vivid-VR env
    uvicorn = None

try:
    from fastapi import FastAPI
except ImportError:  # pragma: no cover - exercised in the original Vivid-VR env
    FastAPI = None


logger = logging.getLogger(__name__)


def _load_caption_backend_factory():
    try:
        return importlib.import_module(
            "sglang.multimodal_gen.runtime.vividvr.caption_sidecar_backend.captioner"
        ).create_captioner
    except ModuleNotFoundError:
        backend_dir = (
            Path(__file__).resolve().parents[1]
            / "runtime"
            / "vividvr"
            / "caption_sidecar_backend"
        )
        package_path = backend_dir / "__init__.py"
        spec = importlib.util.spec_from_file_location(
            "vividvr_caption_sidecar_backend",
            package_path,
            submodule_search_locations=[str(backend_dir)],
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(
                "Failed to load VividVR caption backend package: "
                f"{package_path}"
            )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module.create_captioner


create_captioner = _load_caption_backend_factory()


def _load_manifest_contract():
    try:
        from sglang.multimodal_gen.runtime.vividvr.caption_manifest import (
            VividVRCaptionManifest,
        )

        return VividVRCaptionManifest
    except ModuleNotFoundError:
        manifest_path = (
            Path(__file__).resolve().parents[1]
            / "runtime"
            / "vividvr"
            / "caption_manifest.py"
        )
        spec = importlib.util.spec_from_file_location(
            "vividvr_caption_manifest_contract",
            manifest_path,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(
                f"Failed to load VividVR caption manifest module: {manifest_path}"
            )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module.VividVRCaptionManifest


VividVRCaptionManifest = _load_manifest_contract()


def _load_sidecar_runtime_contract():
    try:
        from sglang.multimodal_gen.runtime.vividvr.caption_sidecar_runtime import (
            CaptionClipResult,
            CaptionRequestMetrics,
            CaptionWorkerBatchResult,
            assign_clip_indices_round_robin,
            merge_caption_results_in_clip_order,
        )

        return SimpleNamespace(
            CaptionClipResult=CaptionClipResult,
            CaptionRequestMetrics=CaptionRequestMetrics,
            CaptionWorkerBatchResult=CaptionWorkerBatchResult,
            assign_clip_indices_round_robin=assign_clip_indices_round_robin,
            merge_caption_results_in_clip_order=merge_caption_results_in_clip_order,
        )
    except ModuleNotFoundError:
        runtime_path = (
            Path(__file__).resolve().parents[1]
            / "runtime"
            / "vividvr"
            / "caption_sidecar_runtime.py"
        )
        spec = importlib.util.spec_from_file_location(
            "vividvr_caption_sidecar_runtime_contract",
            runtime_path,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(
                f"Failed to load VividVR caption sidecar runtime module: {runtime_path}"
            )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module


_SIDECAR_RUNTIME = _load_sidecar_runtime_contract()
CaptionClipResult = _SIDECAR_RUNTIME.CaptionClipResult
CaptionRequestMetrics = _SIDECAR_RUNTIME.CaptionRequestMetrics
CaptionWorkerBatchResult = _SIDECAR_RUNTIME.CaptionWorkerBatchResult
assign_clip_indices_round_robin = _SIDECAR_RUNTIME.assign_clip_indices_round_robin
merge_caption_results_in_clip_order = _SIDECAR_RUNTIME.merge_caption_results_in_clip_order


class CaptionSidecarRequest(BaseModel):
    manifest_path: str
    output_caption_path: str
    expected_caption_count: int = Field(gt=0)


class CaptionSidecarResponse(BaseModel):
    caption_file_path: str
    caption_count: int
    manifest_path: str
    mode: str | None = None
    worker_count: int | None = None
    fallback_used: bool | None = None
    request_id: str | None = None
    total_clip_count: int | None = None
    assigned_clip_indices_by_worker: dict[str, list[int]] | None = None
    timing: dict[str, object] | None = None


@dataclass(frozen=True)
class CaptionWorkerClipJob:
    clip_index: int
    video: torch.Tensor
    fps: float


@dataclass(frozen=True)
class CaptionExecutionMetadata:
    mode: str
    fallback_used: bool
    request_metrics: CaptionRequestMetrics


class ParallelCaptionWorkerError(RuntimeError):
    pass


@dataclass
class CaptionSidecarState:
    captioner: object | None = None
    device: str = "cuda"
    worker_count: int = 1
    worker_devices: tuple[str, ...] = ()
    allow_serial_fallback: bool = True
    executors: tuple[ProcessPoolExecutor, ...] | None = None
    cogvlm2_ckpt_path: str = (
        "/home/zhiheng/Vivid-VR/ckpts/cogvlm2-llama3-caption"
    )


_WORKER_STATE: SimpleNamespace | None = None


def _prepend_env_path(name: str, value: str | os.PathLike[str]) -> None:
    value_str = os.fspath(value)
    existing = os.environ.get(name)
    if not existing:
        os.environ[name] = value_str
        return
    entries = existing.split(os.pathsep)
    if value_str in entries:
        return
    os.environ[name] = f"{value_str}{os.pathsep}{existing}"


def _candidate_python_dev_include_dirs() -> list[Path]:
    configured = sysconfig.get_path("include")
    major_minor = f"{sys.version_info.major}.{sys.version_info.minor}"
    version_tag = f"python{major_minor}"
    compact_version = f"{sys.version_info.major}{sys.version_info.minor}"

    candidates = []
    if configured:
        candidates.append(Path(configured))

    home = Path.home()
    candidates.extend(
        [
            home / f"tmp_py{compact_version}dev" / "extracted" / "usr" / "include" / version_tag,
            home
            / f"tmp_py{compact_version}_headers"
            / "extracted"
            / f"libpython{major_minor}-dev"
            / "usr"
            / "include"
            / version_tag,
        ]
    )

    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate_str not in seen:
            seen.add(candidate_str)
            deduped.append(candidate)
    return deduped


def _python_dev_extra_include_dirs(include_dir: Path) -> list[Path]:
    extras = [include_dir]
    multiarch = sysconfig.get_config_var("MULTIARCH")
    if multiarch:
        multiarch_dir = include_dir.parent / multiarch / include_dir.name
        if multiarch_dir.is_dir() and (multiarch_dir / "pyconfig.h").is_file():
            extras.append(include_dir.parent)
    return extras


def _ensure_python_dev_headers_for_sidecar() -> Path | None:
    configured_include = sysconfig.get_config_var("INCLUDEPY")
    if configured_include:
        configured_path = Path(configured_include)
        if configured_path.is_dir() and (configured_path / "Python.h").is_file():
            return configured_path

    for candidate in _candidate_python_dev_include_dirs():
        if candidate.is_dir() and (candidate / "Python.h").is_file():
            for include_dir in _python_dev_extra_include_dirs(candidate):
                _prepend_env_path("CPATH", include_dir)
                _prepend_env_path("C_INCLUDE_PATH", include_dir)
            print(
                "[VividVR Caption Sidecar] python_include="
                f"{os.environ.get('CPATH')}"
            )
            return candidate

    print(
        "[VividVR Caption Sidecar] warning: Python headers were not found; "
        "Triton caption kernels may fail to compile."
    )
    return None


def _load_video_tensor_cv2(video_path: str) -> tuple[torch.Tensor, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    finally:
        cap.release()
    if not frames:
        raise ValueError(f"No frames found in video file: {video_path}")
    array = np.stack(frames, axis=0).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(0, 3, 1, 2).contiguous()
    return tensor, fps


def _load_video_tensor_decord(video_path: str) -> tuple[torch.Tensor, float]:
    import decord

    # Match the upstream Vivid-VR control-video decode path when available.
    decord.bridge.set_bridge("torch")
    video_reader = decord.VideoReader(uri=video_path, num_threads=1)
    total_frames = len(video_reader)
    if total_frames <= 0:
        raise RuntimeError(f"No frames found in video file: {video_path}")

    batch = video_reader.get_batch(list(range(total_frames)))
    if not isinstance(batch, torch.Tensor):
        arrays = batch.asnumpy() if hasattr(batch, "asnumpy") else np.asarray(batch)
        batch = torch.from_numpy(arrays)
    tensor = batch.float().div(255.0).permute(0, 3, 1, 2).contiguous()
    fps = float(video_reader.get_avg_fps() or 24.0)
    return tensor, fps


def _load_video_tensor(video_path: str) -> tuple[torch.Tensor, float]:
    try:
        return _load_video_tensor_decord(video_path)
    except Exception as exc:
        logger.warning(
            "VividVR caption decord decode failed for %s; falling back to OpenCV: %s",
            video_path,
            exc,
        )
        return _load_video_tensor_cv2(video_path)


def _clip_tensor(video: torch.Tensor, *, start: int, end: int, padded_frames: int) -> torch.Tensor:
    clip = video[start:end]
    if clip.numel() == 0:
        raise ValueError(f"empty clip range start={start} end={end}")
    if clip.shape[0] < padded_frames:
        padding = clip[-1:].repeat(padded_frames - clip.shape[0], 1, 1, 1)
        clip = torch.cat([clip, padding], dim=0)
    return clip


def _caption_manifest_serial(
    state: CaptionSidecarState, manifest: VividVRCaptionManifest
) -> tuple[list[str], float, list[CaptionWorkerBatchResult]]:
    read_start = time.perf_counter()
    video, fps = _load_video_tensor(manifest.video_path)
    read_seconds = time.perf_counter() - read_start
    effective_fps = manifest.fps or fps
    captions: list[str] = []
    clip_results: list[CaptionClipResult] = []

    worker_start = time.perf_counter()
    captioner = _get_serial_captioner(state)
    captioner.to(state.device)
    try:
        for clip in manifest.clips:
            clip_video = _clip_tensor(
                video,
                start=clip.start_frame,
                end=clip.end_frame,
                padded_frames=clip.padded_num_frames,
            )
            clip_start = time.perf_counter()
            caption = str(captioner(clip_video, fps=effective_fps)).strip()
            clip_elapsed = time.perf_counter() - clip_start
            captions.append(caption)
            clip_results.append(
                CaptionClipResult(
                    clip_index=clip.clip_index,
                    caption=caption,
                    worker_index=0,
                    inference_seconds=clip_elapsed,
                    total_seconds=clip_elapsed,
                )
            )
    finally:
        captioner.to(torch.device("cpu"))
        _release_cuda_memory(state.device)

    return captions, read_seconds, [
        CaptionWorkerBatchResult(
            worker_index=0,
            clip_results=clip_results,
            total_seconds=time.perf_counter() - worker_start,
        )
    ]


def _build_request_metrics(
    state: CaptionSidecarState,
    manifest: VividVRCaptionManifest,
    *,
    request_id: str,
) -> CaptionRequestMetrics:
    clip_indices = [clip.clip_index for clip in manifest.clips]
    worker_count = max(1, int(state.worker_count))
    if worker_count > 1:
        assignments = assign_clip_indices_round_robin(
            clip_indices,
            num_workers=worker_count,
        )
    else:
        assignments = {0: list(clip_indices)}
    return CaptionRequestMetrics(
        request_id=request_id,
        total_clip_count=len(clip_indices),
        worker_count=worker_count,
        assigned_clip_indices_by_worker={
            worker_index: tuple(worker_clip_indices)
            for worker_index, worker_clip_indices in assignments.items()
        },
    )


def _build_parallel_worker_jobs(
    manifest: VividVRCaptionManifest,
    *,
    video: torch.Tensor,
    effective_fps: float,
    request_metrics: CaptionRequestMetrics,
) -> dict[int, list[CaptionWorkerClipJob]]:
    clips_by_index = {clip.clip_index: clip for clip in manifest.clips}
    worker_jobs_by_worker: dict[int, list[CaptionWorkerClipJob]] = {}
    for worker_index, clip_indices in request_metrics.assigned_clip_indices_by_worker.items():
        jobs: list[CaptionWorkerClipJob] = []
        for clip_index in clip_indices:
            clip = clips_by_index[clip_index]
            jobs.append(
                CaptionWorkerClipJob(
                    clip_index=clip_index,
                    video=_clip_tensor(
                        video,
                        start=clip.start_frame,
                        end=clip.end_frame,
                        padded_frames=clip.padded_num_frames,
                    ),
                    fps=effective_fps,
                )
            )
        worker_jobs_by_worker[worker_index] = jobs
    return worker_jobs_by_worker


def _collect_parallel_worker_results(
    state: CaptionSidecarState,
    worker_jobs_by_worker: dict[int, list[CaptionWorkerClipJob]],
) -> list[CaptionWorkerBatchResult]:
    if state.executors is None:
        raise RuntimeError("parallel workers requested but no executors are configured")

    future_to_worker_index = {}
    for worker_index, jobs in worker_jobs_by_worker.items():
        if not jobs:
            continue
        if worker_index >= len(state.executors):
            raise ValueError(
                "parallel worker index exceeds configured executor count: "
                f"worker_index={worker_index} executors={len(state.executors)}"
            )
        future = state.executors[worker_index].submit(
            _run_worker_caption_job,
            worker_index,
            jobs,
        )
        future_to_worker_index[future] = worker_index

    worker_results: list[CaptionWorkerBatchResult] = []
    for future in as_completed(future_to_worker_index):
        worker_index = future_to_worker_index[future]
        try:
            worker_results.append(future.result())
        except Exception as exc:
            raise ParallelCaptionWorkerError(
                "caption worker execution failed: "
                f"worker_index={worker_index}"
            ) from exc
    return worker_results


def _caption_manifest_parallel(
    state: CaptionSidecarState,
    manifest: VividVRCaptionManifest,
    request_metrics: CaptionRequestMetrics,
) -> tuple[list[str], float, list[CaptionWorkerBatchResult]]:
    read_start = time.perf_counter()
    video, fps = _load_video_tensor(manifest.video_path)
    effective_fps = manifest.fps or fps
    worker_jobs_by_worker = _build_parallel_worker_jobs(
        manifest,
        video=video,
        effective_fps=effective_fps,
        request_metrics=request_metrics,
    )
    read_seconds = time.perf_counter() - read_start
    worker_results = _collect_parallel_worker_results(state, worker_jobs_by_worker)
    merged_results = merge_caption_results_in_clip_order(worker_results)
    actual_clip_indices = [clip_result.clip_index for clip_result in merged_results]
    expected_clip_indices = [clip.clip_index for clip in manifest.clips]
    if actual_clip_indices != expected_clip_indices:
        raise ValueError(
            "parallel caption worker results do not match manifest clip order: "
            f"expected={expected_clip_indices} actual={actual_clip_indices}"
        )
    return [clip_result.caption for clip_result in merged_results], read_seconds, worker_results


def _generate_captions(
    state: CaptionSidecarState,
    manifest: VividVRCaptionManifest,
) -> tuple[list[str], CaptionExecutionMetadata]:
    total_start = time.perf_counter()
    request_metrics = _build_request_metrics(
        state,
        manifest,
        request_id=uuid.uuid4().hex,
    )

    if request_metrics.worker_count <= 1:
        captions, read_seconds, worker_batches = _caption_manifest_serial(state, manifest)
        request_metrics = replace(
            request_metrics,
            read_seconds=read_seconds,
            total_seconds=time.perf_counter() - total_start,
            worker_batches=worker_batches,
        )
        return captions, CaptionExecutionMetadata(
            mode="serial",
            fallback_used=False,
            request_metrics=request_metrics,
        )

    _ensure_parallel_executors(state)
    try:
        captions, read_seconds, worker_batches = _caption_manifest_parallel(
            state,
            manifest,
            request_metrics,
        )
        request_metrics = replace(
            request_metrics,
            read_seconds=read_seconds,
            total_seconds=time.perf_counter() - total_start,
            worker_batches=worker_batches,
        )
        return captions, CaptionExecutionMetadata(
            mode="parallel",
            fallback_used=False,
            request_metrics=request_metrics,
        )
    except ParallelCaptionWorkerError as exc:
        if not state.allow_serial_fallback:
            raise
        _shutdown_parallel_executors(state, wait=True)
        print(
            "[VividVR Caption Sidecar] parallel caption failed; falling back to serial. "
            f"request_id={request_metrics.request_id} error={exc!r}",
            file=sys.stderr,
        )
        traceback.print_exc()
        captions, read_seconds, worker_batches = _caption_manifest_serial(state, manifest)
        _release_serial_captioner(state)
        request_metrics = replace(
            request_metrics,
            read_seconds=read_seconds,
            total_seconds=time.perf_counter() - total_start,
            worker_batches=worker_batches,
        )
        try:
            _ensure_parallel_executors(state)
        except Exception as pool_exc:
            print(
                "[VividVR Caption Sidecar] failed to recreate parallel caption workers "
                f"after fallback; request_id={request_metrics.request_id} error={pool_exc!r}",
                file=sys.stderr,
            )
            traceback.print_exc()
        return captions, CaptionExecutionMetadata(
            mode="serial",
            fallback_used=True,
            request_metrics=request_metrics,
        )


def _generate_caption_sidecar_output(
    *,
    state: CaptionSidecarState,
    manifest_path: str,
    output_caption_path: str,
    expected_caption_count: int,
) -> CaptionSidecarResponse:
    manifest = VividVRCaptionManifest.read_json(manifest_path)
    if manifest.expected_caption_count != expected_caption_count:
        raise ValueError(
            "expected_caption_count mismatch: "
            f"request={expected_caption_count} manifest={manifest.expected_caption_count}"
        )

    output_path = Path(output_caption_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    output_start = time.perf_counter()
    captions, execution_metadata = _generate_captions(state, manifest)
    if len(captions) != expected_caption_count:
        raise ValueError(
            f"caption sidecar expected {expected_caption_count} captions, "
            f"got {len(captions)}"
        )

    write_start = time.perf_counter()
    tmp_path.write_text(
        "".join(f"{caption}\n" for caption in captions),
        encoding="utf-8",
    )
    os.replace(tmp_path, output_path)
    request_metrics = replace(
        execution_metadata.request_metrics,
        write_seconds=time.perf_counter() - write_start,
        total_seconds=time.perf_counter() - output_start,
    )
    execution_metadata = CaptionExecutionMetadata(
        mode=execution_metadata.mode,
        fallback_used=execution_metadata.fallback_used,
        request_metrics=request_metrics,
    )
    metrics_payload = execution_metadata.request_metrics.to_response_dict()
    return CaptionSidecarResponse(
        caption_file_path=str(output_path),
        caption_count=len(captions),
        manifest_path=manifest_path,
        mode=execution_metadata.mode,
        worker_count=execution_metadata.request_metrics.worker_count,
        fallback_used=execution_metadata.fallback_used,
        request_id=execution_metadata.request_metrics.request_id,
        total_clip_count=execution_metadata.request_metrics.total_clip_count,
        assigned_clip_indices_by_worker=metrics_payload[
            "assigned_clip_indices_by_worker"
        ],
        timing=metrics_payload["timing"],
    )


def create_app(state: CaptionSidecarState) -> FastAPI:
    if FastAPI is None:
        raise RuntimeError(
            "fastapi is not installed; use main() to run the stdlib fallback server"
        )
    app = FastAPI(title="VividVR Caption Sidecar")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/v1/vividvr/captions", response_model=CaptionSidecarResponse)
    async def create_captions(req: CaptionSidecarRequest):
        return _generate_caption_sidecar_output(
            state=state,
            manifest_path=req.manifest_path,
            output_caption_path=req.output_caption_path,
            expected_caption_count=req.expected_caption_count,
        )

    return app


def _write_json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict) -> None:
    response_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(response_body)))
    handler.end_headers()
    handler.wfile.write(response_body)


def _build_fallback_handler(state: CaptionSidecarState):
    class CaptionSidecarHTTPRequestHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:
            return None

        def do_GET(self):
            if self.path != "/health":
                _write_json_response(
                    self,
                    HTTPStatus.NOT_FOUND,
                    {"detail": "Not Found"},
                )
                return
            _write_json_response(self, HTTPStatus.OK, {"status": "ok"})

        def do_POST(self):
            if self.path != "/v1/vividvr/captions":
                _write_json_response(
                    self,
                    HTTPStatus.NOT_FOUND,
                    {"detail": "Not Found"},
                )
                return
            try:
                try:
                    content_length = int(self.headers.get("Content-Length") or "0")
                except ValueError as exc:
                    raise ValidationError.from_exception_data(
                        "CaptionSidecarRequest",
                        [
                            {
                                "type": "int_parsing",
                                "loc": ("headers", "Content-Length"),
                                "msg": "Input should be a valid integer",
                                "input": self.headers.get("Content-Length"),
                            }
                        ],
                    ) from exc
                body = self.rfile.read(content_length)
                req = CaptionSidecarRequest.model_validate_json(body)
                response = _generate_caption_sidecar_output(
                    state=state,
                    manifest_path=req.manifest_path,
                    output_caption_path=req.output_caption_path,
                    expected_caption_count=req.expected_caption_count,
                )
            except ValidationError as exc:
                _write_json_response(
                    self,
                    HTTPStatus.BAD_REQUEST,
                    {"detail": str(exc)},
                )
                return
            except Exception as exc:
                _write_json_response(
                    self,
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"detail": str(exc)},
                )
                return

            _write_json_response(
                self,
                HTTPStatus.OK,
                response.model_dump(),
            )

    return CaptionSidecarHTTPRequestHandler


def _run_fallback_http_server(
    *,
    state: CaptionSidecarState,
    host: str,
    port: int,
) -> None:
    server = ThreadingHTTPServer((host, port), _build_fallback_handler(state))
    try:
        server.serve_forever()
    finally:
        server.server_close()


def _build_cogvlm2_captioner(cogvlm2_ckpt_path: str):
    captioner_args = SimpleNamespace(
        caption_backend="cogvlm2",
        cogvlm2_ckpt_path=str(Path(cogvlm2_ckpt_path).expanduser()),
    )
    return create_captioner(captioner_args)


def _get_serial_captioner(state: CaptionSidecarState):
    if state.captioner is None:
        state.captioner = _build_cogvlm2_captioner(state.cogvlm2_ckpt_path)
    return state.captioner


def _release_serial_captioner(state: CaptionSidecarState) -> None:
    state.captioner = None


def _release_cuda_memory(device: str | torch.device) -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return
    device_obj = torch.device(device)
    if device_obj.type != "cuda":
        return
    with torch.cuda.device(device_obj):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _init_worker_state(
    cogvlm2_ckpt_path: str,
    device: str,
) -> None:
    global _WORKER_STATE
    _WORKER_STATE = SimpleNamespace(
        captioner=_build_cogvlm2_captioner(cogvlm2_ckpt_path),
        device=device,
    )


def _run_worker_caption_job(
    worker_index: int,
    clip_jobs: list[CaptionWorkerClipJob],
) -> CaptionWorkerBatchResult:
    if _WORKER_STATE is None:
        raise RuntimeError("worker state is not initialized")

    captioner = _WORKER_STATE.captioner
    captioner.to(_WORKER_STATE.device)
    try:
        worker_start = time.perf_counter()
        clip_results: list[CaptionClipResult] = []
        for clip_job in clip_jobs:
            clip_start = time.perf_counter()
            caption = str(captioner(clip_job.video, fps=clip_job.fps)).strip()
            clip_elapsed = time.perf_counter() - clip_start
            clip_results.append(
                CaptionClipResult(
                    clip_index=clip_job.clip_index,
                    caption=caption,
                    worker_index=worker_index,
                    inference_seconds=clip_elapsed,
                    total_seconds=clip_elapsed,
                )
            )
    finally:
        captioner.to(torch.device("cpu"))
        _release_cuda_memory(_WORKER_STATE.device)
    return CaptionWorkerBatchResult(
        worker_index=worker_index,
        clip_results=clip_results,
        total_seconds=time.perf_counter() - worker_start,
    )


def _parse_worker_devices(value: str) -> tuple[str, ...]:
    devices = tuple(part.strip() for part in value.split(",") if part.strip())
    if not devices:
        raise argparse.ArgumentTypeError("worker device list must not be empty")
    return devices


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VividVR caption sidecar service.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=31200)
    parser.add_argument(
        "--cogvlm2-ckpt-path",
        default="/home/zhiheng/Vivid-VR/ckpts/cogvlm2-llama3-caption",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--parallel-workers", type=int, default=1)
    parser.add_argument("--worker-devices", type=_parse_worker_devices, default=None)
    parser.add_argument(
        "--disable-serial-fallback",
        dest="allow_serial_fallback",
        action="store_false",
    )
    parser.set_defaults(allow_serial_fallback=True)
    args = parser.parse_args(argv)
    if args.parallel_workers <= 0:
        parser.error("--parallel-workers must be a positive integer")
    if args.worker_devices is not None and len(args.worker_devices) != args.parallel_workers:
        parser.error(
            "--worker-devices count must match --parallel-workers when provided"
        )
    return args


def _create_parallel_executors(
    *,
    cogvlm2_ckpt_path: str,
    worker_devices: tuple[str, ...],
) -> tuple[ProcessPoolExecutor, ...]:
    spawn_context = multiprocessing.get_context("spawn")
    return tuple(
        ProcessPoolExecutor(
            max_workers=1,
            mp_context=spawn_context,
            initializer=_init_worker_state,
            initargs=(cogvlm2_ckpt_path, worker_device),
        )
        for worker_device in worker_devices
    )


def _ensure_parallel_executors(state: CaptionSidecarState) -> None:
    if state.worker_count <= 1:
        return
    if state.executors is not None:
        return
    state.executors = _create_parallel_executors(
        cogvlm2_ckpt_path=state.cogvlm2_ckpt_path,
        worker_devices=state.worker_devices,
    )


def _shutdown_parallel_executors(
    state: CaptionSidecarState,
    *,
    wait: bool = False,
) -> None:
    executors = state.executors
    state.executors = None
    if executors is None:
        return
    for executor in executors:
        executor.shutdown(wait=wait, cancel_futures=True)


def main() -> None:
    args = parse_args()
    _ensure_python_dev_headers_for_sidecar()
    worker_devices = args.worker_devices
    if worker_devices is None:
        worker_devices = tuple(args.device for _ in range(args.parallel_workers))
    state = CaptionSidecarState(
        captioner=(
            _build_cogvlm2_captioner(args.cogvlm2_ckpt_path)
            if args.parallel_workers <= 1
            else None
        ),
        device=args.device,
        worker_count=args.parallel_workers,
        worker_devices=worker_devices,
        allow_serial_fallback=args.allow_serial_fallback,
        cogvlm2_ckpt_path=args.cogvlm2_ckpt_path,
    )
    if args.parallel_workers > 1:
        state.executors = _create_parallel_executors(
            cogvlm2_ckpt_path=args.cogvlm2_ckpt_path,
            worker_devices=worker_devices,
        )
    try:
        if FastAPI is not None and uvicorn is not None:
            uvicorn.run(create_app(state), host=args.host, port=args.port)
            return
        _run_fallback_http_server(
            state=state,
            host=args.host,
            port=args.port,
        )
    finally:
        _shutdown_parallel_executors(state)


if __name__ == "__main__":
    main()
