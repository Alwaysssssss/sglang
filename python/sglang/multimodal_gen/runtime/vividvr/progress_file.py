import json
import os
from pathlib import Path


def write_vividvr_runtime_progress(
    progress_path: str | os.PathLike[str] | None,
    *,
    request_id: str | None,
    runtime_progress: float,
) -> None:
    if not progress_path:
        return

    path = Path(progress_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "request_id": request_id,
        "runtime_progress": float(runtime_progress),
    }
    temp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temp_path.replace(path)


def read_vividvr_runtime_progress(
    progress_path: str | os.PathLike[str] | None,
) -> float | None:
    if not progress_path:
        return None

    try:
        payload = json.loads(Path(progress_path).read_text(encoding="utf-8"))
        return float(payload["runtime_progress"])
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
