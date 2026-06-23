# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import sysconfig
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from pydantic import BaseModel, Field

try:
    import uvicorn
except ImportError:  # pragma: no cover - exercised in the original Vivid-VR env
    uvicorn = None

try:
    from fastapi import FastAPI
except ImportError:  # pragma: no cover - exercised in the original Vivid-VR env
    FastAPI = None


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


class CaptionSidecarRequest(BaseModel):
    manifest_path: str
    output_caption_path: str
    expected_caption_count: int = Field(gt=0)


class CaptionSidecarResponse(BaseModel):
    caption_file_path: str
    caption_count: int
    manifest_path: str


@dataclass
class CaptionSidecarState:
    captioner: object
    device: str = "cuda"


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


def _load_video_tensor(video_path: str) -> tuple[torch.Tensor, float]:
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


def _clip_tensor(video: torch.Tensor, *, start: int, end: int, padded_frames: int) -> torch.Tensor:
    clip = video[start:end]
    if clip.numel() == 0:
        raise ValueError(f"empty clip range start={start} end={end}")
    if clip.shape[0] < padded_frames:
        padding = clip[-1:].repeat(padded_frames - clip.shape[0], 1, 1, 1)
        clip = torch.cat([clip, padding], dim=0)
    return clip


def _caption_manifest(state: CaptionSidecarState, manifest: VividVRCaptionManifest) -> list[str]:
    video, fps = _load_video_tensor(manifest.video_path)
    effective_fps = manifest.fps or fps
    captions: list[str] = []

    state.captioner.to(state.device)
    try:
        for clip in manifest.clips:
            clip_video = _clip_tensor(
                video,
                start=clip.start_frame,
                end=clip.end_frame,
                padded_frames=clip.padded_num_frames,
            )
            captions.append(str(state.captioner(clip_video, fps=effective_fps)).strip())
    finally:
        state.captioner.to(torch.device("cpu"))

    return captions


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

    captions = _caption_manifest(state, manifest)
    if len(captions) != expected_caption_count:
        raise ValueError(
            f"caption sidecar expected {expected_caption_count} captions, "
            f"got {len(captions)}"
        )

    tmp_path.write_text(
        "".join(f"{caption}\n" for caption in captions),
        encoding="utf-8",
    )
    os.replace(tmp_path, output_path)
    return CaptionSidecarResponse(
        caption_file_path=str(output_path),
        caption_count=len(captions),
        manifest_path=manifest_path,
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
                content_length = int(self.headers.get("Content-Length") or "0")
                body = self.rfile.read(content_length)
                req = CaptionSidecarRequest.model_validate_json(body)
                response = _generate_caption_sidecar_output(
                    state=state,
                    manifest_path=req.manifest_path,
                    output_caption_path=req.output_caption_path,
                    expected_caption_count=req.expected_caption_count,
                )
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


def _load_original_captioner(args: argparse.Namespace):
    vividvr_root = Path(args.vividvr_root).expanduser()
    for path in (vividvr_root, vividvr_root / "src"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    from VRDiT.captioner import create_captioner

    captioner_args = SimpleNamespace(
        caption_backend="cogvlm2",
        cogvlm2_ckpt_path=str(Path(args.cogvlm2_ckpt_path).expanduser()),
        caption_sglang_base_url="http://127.0.0.1:30000/v1",
        caption_sglang_model=None,
        caption_sglang_api_key="None",
        caption_sglang_max_tokens=256,
        caption_sglang_timeout=300,
        caption_sglang_max_frames=8,
        caption_sglang_max_pixels=1280 * 720,
    )
    return create_captioner(captioner_args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VividVR caption sidecar service.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=31200)
    parser.add_argument("--vividvr-root", default="/home/zhiheng/Vivid-VR")
    parser.add_argument(
        "--cogvlm2-ckpt-path",
        default="/home/zhiheng/Vivid-VR/ckpts/cogvlm2-llama3-caption",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_python_dev_headers_for_sidecar()
    state = CaptionSidecarState(
        captioner=_load_original_captioner(args),
        device=args.device,
    )
    if FastAPI is not None and uvicorn is not None:
        uvicorn.run(create_app(state), host=args.host, port=args.port)
        return
    _run_fallback_http_server(
        state=state,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
