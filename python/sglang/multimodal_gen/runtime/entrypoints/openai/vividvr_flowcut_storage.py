import shutil
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import httpx

from sglang.multimodal_gen.runtime.entrypoints.openai.flowcut import (
    upload_to_flowcut_minio,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.vividvr_flowcut_protocol import (
    FlowCutMinIOConfig,
)


def normalize_flowcut_object_key(key: str) -> str:
    normalized = key.strip().lstrip("/")
    if not normalized:
        raise ValueError("S3 object key must not be empty")
    if any(part == ".." for part in normalized.split("/")):
        raise ValueError("S3 object key must not contain '..'")
    return normalized


def default_flowcut_output_object_key(
    request_id: str,
    *,
    now: datetime | None = None,
    extension: str = ".mp4",
) -> str:
    now = now or datetime.now()
    extension = extension if extension.startswith(".") else f".{extension}"
    return f"{now:%Y/%m/%d}/{now:%H%M%S}_{request_id}{extension}"


class VividVRFlowCutStorage:
    """Request-scoped file storage for FlowCut Vivid-VR jobs."""

    def __init__(self, *, base_dir: str | Path, request_id: str):
        self.base_dir = Path(base_dir).resolve()
        workdir = (self.base_dir / request_id).resolve()
        if workdir == self.base_dir or not self._is_path_inside(workdir, self.base_dir):
            raise ValueError(f"request_id escapes base_dir: {request_id!r}")

        self.request_id = request_id
        self.workdir = workdir
        self.inputs_dir = self.workdir / "inputs"
        self.outputs_dir = self.workdir / "outputs"
        self.manifests_dir = self.workdir / "manifests"

        for path in (self.inputs_dir, self.outputs_dir, self.manifests_dir):
            path.mkdir(parents=True, exist_ok=True)

    async def materialize_video(
        self,
        source: str | Path,
        filename_hint: str = "input",
    ) -> str:
        source_text = str(source)
        target_path = self.inputs_dir / self._normalize_video_filename(
            filename_hint,
            source_extension=self._source_extension(source_text),
        )
        if self._is_http_url(source_text):
            async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
                response = await client.get(source_text)
                response.raise_for_status()
            target_path.write_bytes(response.content)
            return str(target_path)

        source_path = Path(source).expanduser()
        if not source_path.exists():
            raise FileNotFoundError(str(source_path))

        if source_path.resolve() == target_path.resolve():
            return str(target_path)

        shutil.copy2(source_path, target_path)
        return str(target_path)

    def output_file_path(self, filename: str | None = None) -> str:
        output_filename = filename or f"{self.request_id}.mp4"
        return str(self.outputs_dir / Path(output_filename).name)

    async def upload_result(
        self,
        local_path: str | Path,
        minio_config: FlowCutMinIOConfig | None,
        *,
        object_key: str | None = None,
        bucket_name: str | None = None,
    ) -> str:
        local_path_str = str(local_path)
        if minio_config is None:
            return local_path_str

        local_path_resolved = Path(local_path).resolve()
        outputs_dir_resolved = self.outputs_dir.resolve()
        if not self._is_path_inside(local_path_resolved, outputs_dir_resolved):
            raise ValueError(
                f"local_path must be inside outputs_dir: {local_path_resolved}"
            )

        resolved_object_key = normalize_flowcut_object_key(
            object_key
            or default_flowcut_output_object_key(
                self.request_id,
                extension=local_path_resolved.suffix or ".mp4",
            )
        )
        result_url = await upload_to_flowcut_minio(
            local_path=str(local_path_resolved),
            object_key=resolved_object_key,
            config=minio_config,
            bucket_name=bucket_name,
        )
        local_path_resolved.unlink()
        return result_url

    def finalize_local_result(
        self,
        local_path: str | Path,
        *,
        persistent_output_path: str | Path | None = None,
    ) -> str:
        local_path_resolved = Path(local_path).resolve()
        if persistent_output_path is None:
            return str(local_path_resolved)

        target_path = Path(persistent_output_path).expanduser().resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if local_path_resolved != target_path:
            shutil.move(str(local_path_resolved), str(target_path))
        return str(target_path)

    def cleanup(self) -> None:
        shutil.rmtree(self.workdir, ignore_errors=True)

    @staticmethod
    def _is_http_url(source: str) -> bool:
        return urlparse(source).scheme in {"http", "https"}

    @staticmethod
    def _normalize_video_filename(
        filename_hint: str,
        *,
        source_extension: str | None = None,
    ) -> str:
        filename = Path(filename_hint).name or "input"
        path = Path(filename)
        if path.suffix:
            return filename
        extension = source_extension or ".mp4"
        if not extension.startswith("."):
            extension = f".{extension}"
        return f"{filename}{extension}"

    @staticmethod
    def _source_extension(source: str) -> str:
        parsed = urlparse(source)
        candidate = parsed.path if parsed.scheme else source
        suffix = Path(candidate).suffix.lower()
        return suffix or ".mp4"

    @staticmethod
    def _is_path_inside(path: Path, parent: Path) -> bool:
        try:
            path.relative_to(parent)
        except ValueError:
            return False
        return True
