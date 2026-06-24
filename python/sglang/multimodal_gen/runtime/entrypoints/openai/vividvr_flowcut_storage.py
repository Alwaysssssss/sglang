import shutil
from pathlib import Path
from urllib.parse import urlparse

import httpx

from sglang.multimodal_gen.runtime.entrypoints.openai.flowcut import (
    upload_to_flowcut_minio,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.vividvr_flowcut_protocol import (
    FlowCutMinIOConfig,
)


class VividVRFlowCutStorage:
    """Request-scoped file storage for FlowCut Vivid-VR jobs."""

    def __init__(self, *, base_dir: str | Path, request_id: str):
        self.base_dir = Path(base_dir)
        self.request_id = request_id
        self.workdir = self.base_dir / request_id
        self.inputs_dir = self.workdir / "inputs"
        self.outputs_dir = self.workdir / "outputs"
        self.manifests_dir = self.workdir / "manifests"

        for path in (self.inputs_dir, self.outputs_dir, self.manifests_dir):
            path.mkdir(parents=True, exist_ok=True)

    async def materialize_video(
        self,
        source: str | Path,
        filename_hint: str = "input.mp4",
    ) -> str:
        target_path = self.inputs_dir / self._normalize_video_filename(filename_hint)
        source_text = str(source)
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
    ) -> str:
        local_path_str = str(local_path)
        if minio_config is None:
            return local_path_str

        result_url = await upload_to_flowcut_minio(
            local_path=local_path_str,
            object_key=f"outputs/{self.request_id}.mp4",
            config=minio_config,
        )
        Path(local_path).unlink()
        return result_url

    def cleanup(self) -> None:
        shutil.rmtree(self.workdir, ignore_errors=True)

    @staticmethod
    def _is_http_url(source: str) -> bool:
        return urlparse(source).scheme in {"http", "https"}

    @staticmethod
    def _normalize_video_filename(filename_hint: str) -> str:
        filename = Path(filename_hint).name or "input.mp4"
        path = Path(filename)
        if path.suffix:
            return filename
        return f"{filename}.mp4"
