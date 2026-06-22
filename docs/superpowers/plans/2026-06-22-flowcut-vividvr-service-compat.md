# FlowCut Vivid-VR 服务兼容实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 在 `/home/zhiheng/sglang` 当前仓库内，为 Vivid-VR serve 增加 FlowCut 下游接口兼容能力，满足 `docs_xzh/downstream-endpoint-spec.html` 中的接单、并发、异步回调和中间进度要求。

**架构：** 保留现有 `/v1/videos/repairs` OpenAI 风格接口不变，新增 FlowCut 专用入口 `POST /v1/videos/repairs/flowcut`。该入口解析 FlowCut 的 `taskId/callbackUrl/timeout/minioConfig` 和 Vivid-VR 业务字段，内部复用当前 Vivid-VR repair sampling/job 派发链路，后台任务通过 FlowCut callback mapper 上报 `running/succeeded/failed`。不复制或直接依赖 `/home/zhiheng/sglang_serve` 的任何代码。

**技术栈：** FastAPI、Pydantic、asyncio、httpx、boto3（仅当 `minioConfig` 存在且需要上传时）、pytest、FastAPI TestClient/httpx mock。

---

## 设计边界和文件结构

**新增文件**

- `python/sglang/multimodal_gen/runtime/entrypoints/openai/flowcut.py`
  - FlowCut 专用 request/response/callback payload helper。
  - FlowCut progress reporter。
  - 可选 per-request MinIO 上传 helper。
  - 不包含 Vivid-VR sampling 逻辑，避免模型逻辑和协议适配耦合。

- `python/sglang/multimodal_gen/test/unit/test_flowcut_protocol.py`
  - 覆盖 FlowCut Pydantic alias、响应码、callback payload、timeout 解析、MinIO 配置解析。

- `python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py`
  - 覆盖 FastAPI endpoint 的成功接单、并发满、参数错误和回调调度。

**修改文件**

- `python/sglang/multimodal_gen/runtime/entrypoints/openai/protocol.py`
  - 增加 `FlowCutMinIOConfig`、`FlowCutVideoRepairRequest`、`FlowCutResponse`。
  - 不修改 `VideoRepairRequest` 的现有字段默认语义。

- `python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py`
  - 抽出当前 Vivid-VR repair job 构建逻辑供两个 endpoint 复用。
  - 新增 `@router.post("/repairs/flowcut")`。
  - 为后台派发增加 FlowCut callback 模式。
  - 保持 `/v1/videos/repairs` 的返回模型、HTTP 429 行为和 OpenAI callback payload 不变。

**不修改**

- 不修改 `/home/zhiheng/sglang_serve`。
- 不修改 Vivid-VR runtime preprocessing/postprocessing 语义。
- 不修改 Phase C/D/E 默认配置、benchmark 命令或模型权重路径。

---

## 任务 1：协议模型和 FlowCut helper

**文件：**
- 修改：`python/sglang/multimodal_gen/runtime/entrypoints/openai/protocol.py`
- 创建：`python/sglang/multimodal_gen/runtime/entrypoints/openai/flowcut.py`
- 测试：`python/sglang/multimodal_gen/test/unit/test_flowcut_protocol.py`

- [ ] **步骤 1：编写失败的协议测试**

在 `python/sglang/multimodal_gen/test/unit/test_flowcut_protocol.py` 添加：

```python
import json

from sglang.multimodal_gen.runtime.entrypoints.openai.flowcut import (
    build_flowcut_final_callback_payload,
    build_flowcut_running_callback_payload,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    FlowCutResponse,
    FlowCutVideoRepairRequest,
)


def test_flowcut_request_accepts_camel_case_system_fields():
    req = FlowCutVideoRepairRequest.model_validate(
        {
            "taskId": "task-1",
            "timeout": -1,
            "callbackUrl": "http://127.0.0.1:9000/tasks/task-1/callback",
            "minioConfig": {
                "endpoint": "minio.example.com:9000",
                "bucket_name": "flowcut",
                "access_key": "ak",
                "secret_key": "sk",
                "secure": False,
                "region": "us-east-1",
            },
            "video_url": "https://example.com/in.mp4",
            "caption_file_path": "/tmp/caption.txt",
            "num_inference_steps": 20,
        }
    )

    assert req.task_id == "task-1"
    assert req.timeout == -1
    assert req.callback_url == "http://127.0.0.1:9000/tasks/task-1/callback"
    assert req.minio_config is not None
    assert req.minio_config.bucket_name == "flowcut"
    assert req.video_url == "https://example.com/in.mp4"
    assert req.caption_file_path == "/tmp/caption.txt"
    assert req.num_inference_steps == 20


def test_flowcut_response_uses_numeric_code():
    accepted = FlowCutResponse(code=0, message="ok")
    busy = FlowCutResponse(code=2, message="A task is running.")

    assert accepted.model_dump() == {"code": 0, "message": "ok"}
    assert busy.model_dump()["code"] == 2
    assert isinstance(busy.model_dump()["code"], int)


def test_flowcut_running_callback_payload():
    payload = build_flowcut_running_callback_payload(
        task_id="task-1",
        progress=45.5,
        reason="processing",
    )

    assert payload == {
        "status": "running",
        "progress": 45.5,
        "reason": "processing",
        "output": "",
    }


def test_flowcut_final_callback_payload_success_output_is_json_string():
    payload = build_flowcut_final_callback_payload(
        status="succeeded",
        progress=100,
        reason="",
        output={"result_url": "http://storage/out.mp4", "duration": 12.5},
    )

    assert payload["status"] == "succeeded"
    assert payload["progress"] == 100
    assert payload["reason"] == ""
    assert json.loads(payload["output"]) == {
        "result_url": "http://storage/out.mp4",
        "duration": 12.5,
    }


def test_flowcut_final_callback_payload_failed_omits_output_data():
    payload = build_flowcut_final_callback_payload(
        status="failed",
        progress=0,
        reason="invalid video input",
        output=None,
    )

    assert payload == {
        "status": "failed",
        "progress": 0,
        "reason": "invalid video input",
        "output": "",
    }
```

- [ ] **步骤 2：运行测试确认失败**

运行：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_flowcut_protocol.py -q
```

预期：FAIL，导入 `FlowCutVideoRepairRequest` 或 `build_flowcut_running_callback_payload` 失败。

- [ ] **步骤 3：实现协议模型**

在 `protocol.py` 的 `VideoRepairRequest` 后添加：

```python
class FlowCutMinIOConfig(BaseModel):
    endpoint: str
    bucket_name: str
    access_key: str
    secret_key: str
    secure: bool = False
    region: Optional[str] = None


class FlowCutVideoRepairRequest(VideoRepairRequest):
    task_id: str = Field(alias="taskId")
    timeout: int = -1
    callback_url: str = Field(alias="callbackUrl")
    minio_config: Optional[FlowCutMinIOConfig] = Field(
        default=None, alias="minioConfig"
    )

    model_config = {
        "populate_by_name": True,
        "extra": "allow",
    }


class FlowCutResponse(BaseModel):
    code: int
    message: str = "ok"
```

保留 `VideoRepairRequest` 原有字段，避免影响 `/v1/videos/repairs`。

- [ ] **步骤 4：实现 FlowCut callback helper**

创建 `flowcut.py`，初始内容：

```python
import asyncio
import json
import os
import time
from typing import Any, Dict, Optional

import httpx

from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    FlowCutMinIOConfig,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

FLOWCUT_PROGRESS_INTERVAL_SECONDS = float(
    os.environ.get("SGLANG_FLOWCUT_PROGRESS_INTERVAL_SECONDS", "30")
)


def build_flowcut_running_callback_payload(
    *, task_id: str, progress: float, reason: str
) -> Dict[str, Any]:
    return {
        "status": "running",
        "progress": float(progress),
        "reason": reason,
        "output": "",
    }


def build_flowcut_final_callback_payload(
    *,
    status: str,
    progress: float,
    reason: str,
    output: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if status not in {"succeeded", "failed"}:
        raise ValueError(f"Unsupported FlowCut final status: {status}")
    return {
        "status": status,
        "progress": float(progress),
        "reason": reason,
        "output": json.dumps(output, ensure_ascii=False) if output else "",
    }


async def post_flowcut_callback(
    callback_url: str,
    payload: Dict[str, Any],
    *,
    timeout: float = 10.0,
    max_retries: int = 3,
) -> None:
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
                response = await client.post(callback_url, json=payload)
                response.raise_for_status()
            return
        except Exception as e:
            last_error = str(e)
            logger.warning(
                "FlowCut callback failed attempt=%s/%s url=%s: %s",
                attempt,
                max_retries,
                callback_url,
                last_error,
            )
            if attempt < max_retries:
                await asyncio.sleep(min(2 ** (attempt - 1), 5))
    raise RuntimeError(f"FlowCut callback failed after {max_retries} attempts: {last_error}")
```

MinIO 上传函数在任务 4 添加，避免本任务范围过大。

- [ ] **步骤 5：运行协议测试确认通过**

运行：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_flowcut_protocol.py -q
```

预期：PASS。

- [ ] **步骤 6：格式检查**

运行：

```bash
git diff --check -- \
  python/sglang/multimodal_gen/runtime/entrypoints/openai/protocol.py \
  python/sglang/multimodal_gen/runtime/entrypoints/openai/flowcut.py \
  python/sglang/multimodal_gen/test/unit/test_flowcut_protocol.py
```

预期：无输出。

---

## 任务 2：抽出 Vivid-VR repair job 构建逻辑

**文件：**
- 修改：`python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py`
- 测试：`python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py`

- [ ] **步骤 1：编写失败的 helper 测试**

在 `test_flowcut_video_repair_api.py` 添加：

```python
from types import SimpleNamespace

import pytest

from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import VideoRepairRequest
from sglang.multimodal_gen.runtime.entrypoints.openai import video_api


def test_build_vividvr_kwargs_keeps_phase_e_defaults_optional(monkeypatch, tmp_path):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("restore the video", encoding="utf-8")
    server_args = SimpleNamespace(
        prompt_file_path=str(prompt_file),
        pipeline_config=SimpleNamespace(default_prompt_file_path=str(prompt_file)),
    )
    req = VideoRepairRequest(
        task_id="job-1",
        video_input_path="/tmp/input.mp4",
        caption_file_path="/tmp/caption.txt",
        reference_video_path="/tmp/reference.mp4",
        num_inference_steps=20,
        seed=42,
    )

    kwargs = video_api._build_vividvr_repair_kwargs(
        request_id="job-1",
        req=req,
        server_args=server_args,
        video_input_path="/tmp/input.mp4",
        output_dir=str(tmp_path),
        output_file_name="job-1.mp4",
    )

    assert kwargs["request_id"] == "job-1"
    assert kwargs["video_input_path"] == "/tmp/input.mp4"
    assert kwargs["prompt"] == "restore the video"
    assert kwargs["caption_source"] == "caption_file"
    assert kwargs["caption_file_path"] == "/tmp/caption.txt"
    assert kwargs["reference_video_path"] == "/tmp/reference.mp4"
    assert kwargs["num_inference_steps"] == 20
    assert kwargs["seed"] == 42
```

- [ ] **步骤 2：运行测试确认失败**

运行：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py::test_build_vividvr_kwargs_keeps_phase_e_defaults_optional -q
```

预期：FAIL，`_build_vividvr_repair_kwargs` 不存在。

- [ ] **步骤 3：抽出 Vivid-VR kwargs helper**

在 `video_api.py` 中 `_resolve_vividvr_prompt_file_path` 后添加：

```python
def _build_vividvr_repair_kwargs(
    *,
    request_id: str,
    req: VideoRepairRequest,
    server_args,
    video_input_path: str,
    output_dir: str,
    output_file_name: str,
) -> Dict[str, Any]:
    vividvr_prompt_file_path = _resolve_vividvr_prompt_file_path(server_args)
    vividvr_kwargs = {
        "request_id": request_id,
        "video_input_path": video_input_path,
        "prompt": read_prompt_file(vividvr_prompt_file_path),
        "prompt_file_path": vividvr_prompt_file_path,
        "output_path": output_dir,
        "output_file_name": output_file_name,
        "seed": req.seed,
        "dtype": req.dtype,
        "enable_teacache": req.enable_teacache,
        "enable_frame_interpolation": req.enable_frame_interpolation,
        "frame_interpolation_exp": req.frame_interpolation_exp,
        "frame_interpolation_scale": req.frame_interpolation_scale,
        "enable_upscaling": req.enable_upscaling,
        "upscaling_scale": req.upscaling_scale,
        "perf_dump_path": req.perf_dump_path,
    }
    if req.output_quality not in (None, "default"):
        vividvr_kwargs["output_quality"] = req.output_quality
    if req.negative_prompt is not None:
        vividvr_kwargs["negative_prompt"] = req.negative_prompt
    if req.caption_file_path is not None:
        vividvr_kwargs["caption_source"] = "caption_file"
        vividvr_kwargs["caption_file_path"] = req.caption_file_path
    if req.reference_video_path is not None:
        vividvr_kwargs["reference_video_path"] = req.reference_video_path
    if req.num_frames is not None:
        vividvr_kwargs["num_frames"] = req.num_frames
    if req.num_inference_steps is not None:
        vividvr_kwargs["num_inference_steps"] = req.num_inference_steps
    if req.guidance_scale is not None:
        vividvr_kwargs["guidance_scale"] = req.guidance_scale
    if req.generator_device is not None:
        vividvr_kwargs["generator_device"] = req.generator_device
    if req.num_temporal_process_frames is not None:
        vividvr_kwargs["num_temporal_process_frames"] = req.num_temporal_process_frames
    if req.restoration_guidance_scale is not None:
        vividvr_kwargs["restoration_guidance_scale"] = req.restoration_guidance_scale
    if req.frame_interpolation_model_path is not None:
        vividvr_kwargs["frame_interpolation_model_path"] = req.frame_interpolation_model_path
    if req.upscaling_model_path is not None:
        vividvr_kwargs["upscaling_model_path"] = req.upscaling_model_path
    if req.output_compression is not None:
        vividvr_kwargs["output_compression"] = req.output_compression
    return vividvr_kwargs
```

替换 `create_video_repair` 中 Vivid-VR 分支的内联 `vividvr_kwargs` 构造：

```python
vividvr_kwargs = _build_vividvr_repair_kwargs(
    request_id=request_id,
    req=req,
    server_args=server_args,
    video_input_path=video_input_path,
    output_dir=output_dir,
    output_file_name=output_file_name,
)
sampling_params = VividVRSamplingParams.from_user_kwargs(
    server_args,
    **vividvr_kwargs,
)
```

- [ ] **步骤 4：运行 helper 测试确认通过**

运行：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py::test_build_vividvr_kwargs_keeps_phase_e_defaults_optional -q
```

预期：PASS。

- [ ] **步骤 5：运行现有 repair 相关轻量测试**

运行：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_inference_tool.py::TestVividVRInferenceTool::test_load_video_frames_reuses_compare_cache -q
```

预期：PASS。该测试不是 endpoint 测试，但能快速确认本轮没有影响已有 Vivid-VR 工具链导入。

---

## 任务 3：新增 FlowCut 接单 endpoint 和响应码语义

**文件：**
- 修改：`python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py`
- 修改：`python/sglang/multimodal_gen/runtime/entrypoints/openai/protocol.py`
- 测试：`python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py`

- [ ] **步骤 1：编写 endpoint 协议测试**

继续在 `test_flowcut_video_repair_api.py` 添加：

```python
import asyncio

from fastapi import FastAPI
from fastapi.testclient import TestClient

from sglang.multimodal_gen.runtime.entrypoints.openai import video_api


def _make_test_client():
    app = FastAPI()
    app.include_router(video_api.router)
    return TestClient(app)


def test_flowcut_endpoint_returns_code_2_when_queue_full(monkeypatch):
    client = _make_test_client()

    async def locked():
        return True

    class LockedSemaphore:
        def locked(self):
            return True

    monkeypatch.setattr(video_api, "_VIDEOEDIT_SEMAPHORE", LockedSemaphore())

    response = client.post(
        "/v1/videos/repairs/flowcut",
        json={
            "taskId": "busy-task",
            "timeout": -1,
            "callbackUrl": "http://127.0.0.1:9000/callback",
            "video_input_path": "/tmp/in.mp4",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"code": 2, "message": "A task is running."}


def test_flowcut_endpoint_returns_code_1_for_missing_input(monkeypatch):
    class AvailableSemaphore:
        def locked(self):
            return False

        async def acquire(self):
            raise AssertionError("should not acquire semaphore for invalid request")

    monkeypatch.setattr(video_api, "_VIDEOEDIT_SEMAPHORE", AvailableSemaphore())
    client = _make_test_client()

    response = client.post(
        "/v1/videos/repairs/flowcut",
        json={
            "taskId": "invalid-task",
            "timeout": -1,
            "callbackUrl": "http://127.0.0.1:9000/callback",
        },
    )

    assert response.status_code == 200
    assert response.json()["code"] == 1
    assert "video_input_path or video_url is required" in response.json()["message"]
```

- [ ] **步骤 2：运行 endpoint 测试确认失败**

运行：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py::test_flowcut_endpoint_returns_code_2_when_queue_full \
  python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py::test_flowcut_endpoint_returns_code_1_for_missing_input -q
```

预期：FAIL，路由不存在或响应不是 FlowCut code。

- [ ] **步骤 3：导入 FlowCut 模型并新增 endpoint 骨架**

在 `video_api.py` 导入：

```python
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    FlowCutResponse,
    FlowCutVideoRepairRequest,
    VideoGenerationsRequest,
    VideoListResponse,
    VideoRepairRequest,
    VideoResponse,
)
```

添加 endpoint：

```python
@router.post("/repairs/flowcut", response_model=FlowCutResponse)
async def create_flowcut_video_repair(req: FlowCutVideoRepairRequest):
    if _VIDEOEDIT_SEMAPHORE.locked():
        return FlowCutResponse(code=2, message="A task is running.")

    if not req.video_input_path and not req.video_url:
        return FlowCutResponse(
            code=1,
            message="video_input_path or video_url is required",
        )

    return FlowCutResponse(code=0, message="ok")
```

这一步只让协议测试通过，后续任务接入真实后台派发。

- [ ] **步骤 4：运行 endpoint 协议测试确认通过**

运行：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py::test_flowcut_endpoint_returns_code_2_when_queue_full \
  python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py::test_flowcut_endpoint_returns_code_1_for_missing_input -q
```

预期：PASS。

---

## 任务 4：FlowCut 后台派发、running 进度和最终回调

**文件：**
- 修改：`python/sglang/multimodal_gen/runtime/entrypoints/openai/flowcut.py`
- 修改：`python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py`
- 测试：`python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py`

- [ ] **步骤 1：编写回调调度测试**

继续在 `test_flowcut_video_repair_api.py` 添加：

```python
def test_flowcut_endpoint_accepts_and_schedules_background_job(monkeypatch, tmp_path):
    scheduled = {}
    acquired = {"value": False}

    class AvailableSemaphore:
        def locked(self):
            return False

        async def acquire(self):
            acquired["value"] = True

        def release(self):
            acquired["value"] = False

    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("restore the video", encoding="utf-8")

    monkeypatch.setattr(video_api, "_VIDEOEDIT_SEMAPHORE", AvailableSemaphore())
    monkeypatch.setattr(
        video_api,
        "get_global_server_args",
        lambda: type(
            "Args",
            (),
            {
                "input_save_path": str(tmp_path / "inputs"),
                "output_path": str(tmp_path / "outputs"),
                "prompt_file_path": str(prompt_file),
                "pipeline_config": type(
                    "Cfg",
                    (),
                    {"default_prompt_file_path": str(prompt_file)},
                )(),
                "model_id": "vividvr",
                "pipeline_class_name": "CogVideoXVividVRControlNetPipeline",
            },
        )(),
    )
    monkeypatch.setattr(
        video_api.VividVRSamplingParams,
        "from_user_kwargs",
        staticmethod(lambda server_args, **kwargs: type(
            "Sampling",
            (),
            {
                "output_file_path": lambda self: str(tmp_path / "outputs" / "task-1.mp4"),
            },
        )()),
    )
    monkeypatch.setattr(
        video_api,
        "prepare_request",
        lambda server_args, sampling_params: "prepared-batch",
    )

    def fake_create_task(coro):
        scheduled["coro_name"] = coro.cr_code.co_name
        coro.close()
        return None

    monkeypatch.setattr(video_api.asyncio, "create_task", fake_create_task)

    client = _make_test_client()
    response = client.post(
        "/v1/videos/repairs/flowcut",
        json={
            "taskId": "task-1",
            "timeout": -1,
            "callbackUrl": "http://127.0.0.1:9000/tasks/task-1/callback",
            "video_input_path": "/tmp/in.mp4",
            "caption_file_path": "/tmp/caption.txt",
            "reference_video_path": "/tmp/ref.mp4",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"code": 0, "message": "ok"}
    assert acquired["value"] is True
    assert scheduled["coro_name"] == "_dispatch_flowcut_video_repair_job_async"
```

- [ ] **步骤 2：运行测试确认失败**

运行：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py::test_flowcut_endpoint_accepts_and_schedules_background_job -q
```

预期：FAIL，endpoint 没有真实 acquire、prepare_request 或 schedule 后台任务。

- [ ] **步骤 3：实现 progress reporter**

在 `flowcut.py` 添加：

```python
def progress_from_elapsed(started_at: float) -> float:
    elapsed = max(0.0, time.monotonic() - started_at)
    if elapsed < 1:
        return 1.0
    return min(89.0, 5.0 + elapsed / 30.0)


async def report_flowcut_running_until_done(
    *,
    task_id: str,
    callback_url: str,
    done_event: asyncio.Event,
    interval_seconds: float = FLOWCUT_PROGRESS_INTERVAL_SECONDS,
) -> None:
    started_at = time.monotonic()
    await post_flowcut_callback(
        callback_url,
        build_flowcut_running_callback_payload(
            task_id=task_id,
            progress=1,
            reason="accepted",
        ),
    )
    while not done_event.is_set():
        try:
            await asyncio.wait_for(done_event.wait(), timeout=interval_seconds)
        except asyncio.TimeoutError:
            await post_flowcut_callback(
                callback_url,
                build_flowcut_running_callback_payload(
                    task_id=task_id,
                    progress=progress_from_elapsed(started_at),
                    reason="running",
                ),
            )
```

说明：`timeout=-1` 表示不设推理 deadline；progress reporter 只负责 heartbeat，不取消后台推理。

- [ ] **步骤 4：实现 FlowCut 后台派发 wrapper**

在 `video_api.py` 导入 FlowCut helpers：

```python
from sglang.multimodal_gen.runtime.entrypoints.openai.flowcut import (
    build_flowcut_final_callback_payload,
    post_flowcut_callback,
    report_flowcut_running_until_done,
)
```

添加后台 wrapper：

```python
async def _dispatch_flowcut_video_repair_job_async(
    job_id: str,
    batch: Req,
    *,
    callback_url: str,
    temp_dirs: list[str] | None = None,
    output_persistent: bool = True,
) -> None:
    done_event = asyncio.Event()
    reporter_task = asyncio.create_task(
        report_flowcut_running_until_done(
            task_id=job_id,
            callback_url=callback_url,
            done_event=done_event,
        )
    )
    try:
        await VIDEO_STORE.update_fields(job_id, {"status": "running", "progress": 1})
        await _dispatch_job_async(
            job_id,
            batch,
            temp_dirs=None,
            output_persistent=output_persistent,
            callback_url=None,
        )
        job = await VIDEO_STORE.get(job_id) or {}
        status = job.get("status")
        if status == "completed":
            payload = build_flowcut_final_callback_payload(
                status="succeeded",
                progress=100,
                reason="",
                output={
                    "result_url": job.get("url"),
                    "file_path": job.get("file_path"),
                    "duration": job.get("inference_time_s"),
                },
            )
        else:
            error = job.get("error") or {}
            payload = build_flowcut_final_callback_payload(
                status="failed",
                progress=0,
                reason=error.get("message") or "video repair failed",
                output=None,
            )
        await post_flowcut_callback(callback_url, payload)
    finally:
        done_event.set()
        reporter_task.cancel()
        try:
            await reporter_task
        except asyncio.CancelledError:
            pass
        _VIDEOEDIT_SEMAPHORE.release()
        for td in temp_dirs or []:
            shutil.rmtree(td, ignore_errors=True)
```

- [ ] **步骤 5：接入 FlowCut endpoint 的真实派发**

把 `create_flowcut_video_repair` 的 `code:0` 骨架替换为与 `create_video_repair` 等价的 Vivid-VR 分支：

```python
await _VIDEOEDIT_SEMAPHORE.acquire()
server_args = get_global_server_args()
request_id = req.task_id
temp_dirs: list[str] = []
try:
    uploads_dir = server_args.input_save_path
    if uploads_dir is None:
        uploads_dir = tempfile.mkdtemp(prefix="sglang_flowcut_input_")
        temp_dirs.append(uploads_dir)
    os.makedirs(uploads_dir, exist_ok=True)

    video_input_path = req.video_input_path
    if req.video_url:
        video_input_path = await _save_video_source_to_path(
            req.video_url, os.path.join(uploads_dir, f"{request_id}_video")
        )
    if not video_input_path:
        _VIDEOEDIT_SEMAPHORE.release()
        for td in temp_dirs:
            shutil.rmtree(td, ignore_errors=True)
        return FlowCutResponse(
            code=1,
            message="video_input_path or video_url is required",
        )

    output_dir, output_file_name = _split_output_path(
        req.output_path, request_id, server_args.output_path
    )
    output_persistent = output_dir is not None
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="sglang_flowcut_output_")
        temp_dirs.append(output_dir)
        output_persistent = False

    if not _is_vividvr_video_repair_pipeline(server_args):
        _VIDEOEDIT_SEMAPHORE.release()
        for td in temp_dirs:
            shutil.rmtree(td, ignore_errors=True)
        return FlowCutResponse(
            code=1,
            message="FlowCut repair endpoint requires Vivid-VR pipeline",
        )

    vividvr_kwargs = _build_vividvr_repair_kwargs(
        request_id=request_id,
        req=req,
        server_args=server_args,
        video_input_path=video_input_path,
        output_dir=output_dir,
        output_file_name=output_file_name,
    )
    sampling_params = VividVRSamplingParams.from_user_kwargs(
        server_args,
        **vividvr_kwargs,
    )
    job = _video_repair_job_from_sampling(request_id, req, sampling_params)
    job["model"] = _resolve_video_repair_model_name(req, server_args, "VividVR")
    await VIDEO_STORE.upsert(request_id, job)
    batch = prepare_request(server_args=server_args, sampling_params=sampling_params)
    asyncio.create_task(
        _dispatch_flowcut_video_repair_job_async(
            request_id,
            batch,
            temp_dirs=temp_dirs or None,
            output_persistent=output_persistent,
            callback_url=req.callback_url,
        )
    )
    return FlowCutResponse(code=0, message="ok")
except Exception as e:
    _VIDEOEDIT_SEMAPHORE.release()
    for td in temp_dirs:
        shutil.rmtree(td, ignore_errors=True)
    return FlowCutResponse(code=1, message=str(e))
```

实现时要避免双重 release：所有 `return code:1` 路径在 acquire 后必须 release；后台任务启动成功后只由 `_dispatch_flowcut_video_repair_job_async` release。

- [ ] **步骤 6：运行 FlowCut endpoint 调度测试**

运行：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py::test_flowcut_endpoint_accepts_and_schedules_background_job -q
```

预期：PASS。

- [ ] **步骤 7：运行本阶段单测集合**

运行：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_flowcut_protocol.py \
  python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py -q
```

预期：PASS。

---

## 任务 5：MinIO 输出兼容

**文件：**
- 修改：`python/sglang/multimodal_gen/runtime/entrypoints/openai/flowcut.py`
- 修改：`python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py`
- 测试：`python/sglang/multimodal_gen/test/unit/test_flowcut_protocol.py`

- [ ] **步骤 1：编写 MinIO helper 测试**

在 `test_flowcut_protocol.py` 添加：

```python
import asyncio

from sglang.multimodal_gen.runtime.entrypoints.openai.flowcut import (
    build_minio_result_url,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import FlowCutMinIOConfig


def test_build_minio_result_url_http():
    cfg = FlowCutMinIOConfig(
        endpoint="minio.example.com:9000",
        bucket_name="flowcut",
        access_key="ak",
        secret_key="sk",
        secure=False,
        region="us-east-1",
    )

    assert (
        build_minio_result_url(cfg, "outputs/task-1.mp4")
        == "http://minio.example.com:9000/flowcut/outputs/task-1.mp4"
    )


def test_build_minio_result_url_https():
    cfg = FlowCutMinIOConfig(
        endpoint="minio.example.com",
        bucket_name="flowcut",
        access_key="ak",
        secret_key="sk",
        secure=True,
        region="us-east-1",
    )

    assert (
        build_minio_result_url(cfg, "outputs/task-1.mp4")
        == "https://minio.example.com/flowcut/outputs/task-1.mp4"
    )
```

- [ ] **步骤 2：运行 MinIO 测试确认失败**

运行：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_flowcut_protocol.py::test_build_minio_result_url_http \
  python/sglang/multimodal_gen/test/unit/test_flowcut_protocol.py::test_build_minio_result_url_https -q
```

预期：FAIL，`build_minio_result_url` 不存在。

- [ ] **步骤 3：实现 MinIO URL 和上传 helper**

在 `flowcut.py` 添加：

```python
def build_minio_result_url(config: FlowCutMinIOConfig, object_key: str) -> str:
    scheme = "https" if config.secure else "http"
    endpoint = config.endpoint.rstrip("/")
    return f"{scheme}://{endpoint}/{config.bucket_name}/{object_key.lstrip('/')}"


async def upload_to_flowcut_minio(
    *,
    local_path: str,
    object_key: str,
    config: FlowCutMinIOConfig,
) -> str:
    import boto3

    endpoint_url = f"{'https' if config.secure else 'http'}://{config.endpoint.rstrip('/')}"

    def _sync_upload() -> None:
        client = boto3.client(
            "s3",
            aws_access_key_id=config.access_key,
            aws_secret_access_key=config.secret_key,
            endpoint_url=endpoint_url,
            region_name=config.region,
        )
        client.upload_file(local_path, config.bucket_name, object_key)

    await asyncio.get_running_loop().run_in_executor(None, _sync_upload)
    return build_minio_result_url(config, object_key)
```

- [ ] **步骤 4：让 FlowCut 后台最终输出优先使用请求级 MinIO**

修改 `_dispatch_flowcut_video_repair_job_async` 签名：

```python
async def _dispatch_flowcut_video_repair_job_async(
    job_id: str,
    batch: Req,
    *,
    callback_url: str,
    minio_config=None,
    temp_dirs: list[str] | None = None,
    output_persistent: bool = True,
) -> None:
```

在成功分支中：

```python
result_url = job.get("url")
file_path = job.get("file_path")
if minio_config is not None and file_path:
    object_key = f"outputs/{job_id}.mp4"
    result_url = await upload_to_flowcut_minio(
        local_path=file_path,
        object_key=object_key,
        config=minio_config,
    )
payload = build_flowcut_final_callback_payload(
    status="succeeded",
    progress=100,
    reason="",
    output={
        "result_url": result_url,
        "file_path": file_path,
        "duration": job.get("inference_time_s"),
    },
)
```

把 endpoint 创建后台任务处改为：

```python
asyncio.create_task(
    _dispatch_flowcut_video_repair_job_async(
        request_id,
        batch,
        temp_dirs=temp_dirs or None,
        output_persistent=output_persistent,
        callback_url=req.callback_url,
        minio_config=req.minio_config,
    )
)
```

并在 `video_api.py` 导入 `upload_to_flowcut_minio`。

- [ ] **步骤 5：运行 MinIO 单测**

运行：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_flowcut_protocol.py -q
```

预期：PASS。

---

## 任务 6：端到端 mock callback 测试

**文件：**
- 修改：`python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py`

- [ ] **步骤 1：编写后台 wrapper 直接测试**

添加：

```python
import pytest


@pytest.mark.asyncio
async def test_dispatch_flowcut_job_posts_running_and_final_callbacks(monkeypatch, tmp_path):
    callbacks = []

    async def fake_post_flowcut_callback(callback_url, payload, **kwargs):
        callbacks.append(payload)

    async def fake_dispatch_job_async(job_id, batch, **kwargs):
        await video_api.VIDEO_STORE.update_fields(
            job_id,
            {
                "status": "completed",
                "progress": 100,
                "file_path": str(tmp_path / "out.mp4"),
                "url": None,
                "inference_time_s": 1.25,
            },
        )

    monkeypatch.setattr(video_api, "post_flowcut_callback", fake_post_flowcut_callback)
    monkeypatch.setattr(video_api, "_dispatch_job_async", fake_dispatch_job_async)
    monkeypatch.setattr(
        video_api,
        "report_flowcut_running_until_done",
        lambda task_id, callback_url, done_event: fake_post_flowcut_callback(
            callback_url,
            {
                "status": "running",
                "progress": 1,
                "reason": "accepted",
                "output": "",
            },
        ),
    )

    class ReleaseTrackingSemaphore:
        def __init__(self):
            self.released = False

        def release(self):
            self.released = True

    semaphore = ReleaseTrackingSemaphore()
    monkeypatch.setattr(video_api, "_VIDEOEDIT_SEMAPHORE", semaphore)

    await video_api.VIDEO_STORE.upsert(
        "task-1",
        {
            "id": "task-1",
            "object": "video",
            "model": "VividVR",
            "status": "queued",
            "progress": 0,
            "created_at": 1,
            "size": "",
            "seconds": "",
            "quality": "standard",
            "file_path": str(tmp_path / "out.mp4"),
        },
    )

    await video_api._dispatch_flowcut_video_repair_job_async(
        "task-1",
        batch="prepared",
        callback_url="http://127.0.0.1:9000/callback",
    )

    assert callbacks[0]["status"] == "running"
    assert callbacks[-1]["status"] == "succeeded"
    assert callbacks[-1]["progress"] == 100
    assert semaphore.released is True
```

- [ ] **步骤 2：运行 wrapper 测试**

运行：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py::test_dispatch_flowcut_job_posts_running_and_final_callbacks -q
```

预期：PASS。

- [ ] **步骤 3：补失败回调测试**

添加：

```python
@pytest.mark.asyncio
async def test_dispatch_flowcut_job_posts_failed_callback(monkeypatch, tmp_path):
    callbacks = []

    async def fake_post_flowcut_callback(callback_url, payload, **kwargs):
        callbacks.append(payload)

    async def fake_dispatch_job_async(job_id, batch, **kwargs):
        await video_api.VIDEO_STORE.update_fields(
            job_id,
            {
                "status": "failed",
                "error": {"message": "GPU out of memory during inference"},
            },
        )

    monkeypatch.setattr(video_api, "post_flowcut_callback", fake_post_flowcut_callback)
    monkeypatch.setattr(video_api, "_dispatch_job_async", fake_dispatch_job_async)
    monkeypatch.setattr(
        video_api,
        "report_flowcut_running_until_done",
        lambda task_id, callback_url, done_event: fake_post_flowcut_callback(
            callback_url,
            {
                "status": "running",
                "progress": 1,
                "reason": "accepted",
                "output": "",
            },
        ),
    )

    class ReleaseTrackingSemaphore:
        def release(self):
            pass

    monkeypatch.setattr(video_api, "_VIDEOEDIT_SEMAPHORE", ReleaseTrackingSemaphore())

    await video_api.VIDEO_STORE.upsert(
        "task-fail",
        {
            "id": "task-fail",
            "object": "video",
            "model": "VividVR",
            "status": "queued",
            "progress": 0,
            "created_at": 1,
            "size": "",
            "seconds": "",
            "quality": "standard",
            "file_path": str(tmp_path / "out.mp4"),
        },
    )

    await video_api._dispatch_flowcut_video_repair_job_async(
        "task-fail",
        batch="prepared",
        callback_url="http://127.0.0.1:9000/callback",
    )

    assert callbacks[-1] == {
        "status": "failed",
        "progress": 0.0,
        "reason": "GPU out of memory during inference",
        "output": "",
    }
```

- [ ] **步骤 4：运行失败回调测试**

运行：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py::test_dispatch_flowcut_job_posts_failed_callback -q
```

预期：PASS。

---

## 任务 7：文档和运行命令

**文件：**
- 修改：`docs_xzh/run_command/vividvr_default_run_and_serve_commands.md`
- 创建：`docs_xzh/hand_over/flowcut_vividvr_service_compat_handover_20260622.md`

- [ ] **步骤 1：更新 serve 调用文档**

在 `docs_xzh/run_command/vividvr_default_run_and_serve_commands.md` 的 Vivid-VR serve/curl 部分增加 FlowCut curl 示例：

```bash
curl -X POST http://127.0.0.1:31190/v1/videos/repairs/flowcut \
  -H 'Content-Type: application/json' \
  -d '{
    "taskId": "flowcut-vividvr-smoke-001",
    "timeout": -1,
    "callbackUrl": "http://127.0.0.1:39090/tasks/flowcut-vividvr-smoke-001/callback",
    "video_input_path": "/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.mp4",
    "caption_file_path": "/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.txt",
    "reference_video_path": "/home/zhiheng/Vivid-VR/result/720p_long_up1_result_vivid_ori_20step/videos/test_video_long_960x720_130f.mp4",
    "num_inference_steps": 20,
    "seed": 42
  }'
```

说明中写明：`timeout=-1` 表示 Vivid-VR 服务侧不对长推理设置超时；同步接单仍需 30 秒内返回。

- [ ] **步骤 2：新增 handover 文档**

创建 `docs_xzh/hand_over/flowcut_vividvr_service_compat_handover_20260622.md`，包含：

```markdown
# FlowCut Vivid-VR Service Compatibility Handover - 2026-06-22

## Scope

- Adds FlowCut-compatible endpoint `POST /v1/videos/repairs/flowcut`.
- Keeps existing `POST /v1/videos/repairs` behavior unchanged.
- Uses only `/home/zhiheng/sglang` code; no direct dependency on `/home/zhiheng/sglang_serve`.

## Contract

- `code:0`: task accepted and runs asynchronously.
- `code:1`: permanent business failure, returned as HTTP 200 JSON.
- `code:2`: queue full, returned as HTTP 200 JSON.
- `timeout:-1`: no service-side inference timeout.
- Callback statuses: `running`, `succeeded`, `failed`.

## Verification

- Unit tests:
  - `python/sglang/multimodal_gen/test/unit/test_flowcut_protocol.py`
  - `python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py`
- Serve smoke test must run in tmux.
```

- [ ] **步骤 3：文档格式检查**

运行：

```bash
git diff --check -- \
  docs_xzh/run_command/vividvr_default_run_and_serve_commands.md \
  docs_xzh/hand_over/flowcut_vividvr_service_compat_handover_20260622.md
```

预期：无输出。

---

## 任务 8：验证矩阵

**文件：**
- 不新增代码文件。

- [ ] **步骤 1：运行新增单测**

运行：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_flowcut_protocol.py \
  python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py -q
```

预期：全部 PASS。

- [ ] **步骤 2：运行 Vivid-VR 现有轻量回归**

运行：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_stage_c_vividvr_single_clip.py::TestStageCVividVRContracts::test_control_video_padding_contract_matches_reference_wrapper \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_inference_tool.py::TestVividVRInferenceTool::test_load_video_frames_reuses_compare_cache -q
```

预期：全部 PASS。

- [ ] **步骤 3：运行完整静态检查**

运行：

```bash
git diff --check
```

预期：无输出。

- [ ] **步骤 4：启动 mock callback server**

新增临时验证命令，不提交临时脚本。该 receiver 接收任意 POST 路径，将请求体逐行写入 `Vivid_Acceptance/logs/flowcut_callback_mock.jsonl`：

```bash
tmux new-session -d -s flowcut_callback_mock \
  "cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/logs && /home/zhiheng/sglang/.venv/bin/python - <<'PY'
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

log_path = Path('Vivid_Acceptance/logs/flowcut_callback_mock.jsonl')

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get('Content-Length', '0'))
        body = self.rfile.read(length).decode('utf-8')
        with log_path.open('a', encoding='utf-8') as f:
            f.write(body + '\n')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(b'{\"code\":0,\"msg\":\"callback processed\"}')

    def log_message(self, fmt, *args):
        return

HTTPServer(('127.0.0.1', 39090), Handler).serve_forever()
PY"
```

查看 callback receiver：

```bash
tmux attach -r -t flowcut_callback_mock
```

- [ ] **步骤 5：启动 Vivid-VR serve smoke test**

所有推理验证必须在 tmux 中启动。使用当前单卡默认 `single_gpu_fa_compile` serve 配置，并额外设置 FlowCut progress 上报间隔为 10 秒：

```bash
tmux new-session -d -s vividvr_flowcut_serve \
  'cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/logs Vivid_Acceptance/result_videos/service_benchmark && export PYTHONUNBUFFERED=1 && export PYTHONPATH=python && export SGLANG_FLOWCUT_PROGRESS_INTERVAL_SECONDS=10 && export SGLANG_VIVIDVR_CONNECTOR_CONTROL_POOL_SIZE=1 && CUDA_VISIBLE_DEVICES=0 /home/zhiheng/sglang/.venv/bin/sglang serve \
    --model-path /home/zhiheng/Vivid-VR/ckpts/CogVideoX1.5-5B \
    --model-id VividVR \
    --pipeline-class-name CogVideoXVividVRControlNetPipeline \
    --component-paths.vividvr /home/zhiheng/Vivid-VR/ckpts/Vivid-VR \
    --attention-backend fa \
    --num-gpus 1 \
    --tp-size 1 \
    --sp-degree 1 \
    --ulysses-degree 1 \
    --ring-degree 1 \
    --enable-torch-compile \
    --dist-timeout 3600 \
    --host 127.0.0.1 \
    --port 31190 \
    --master-port 30190 \
    --scheduler-port 56190 \
    --strict-ports \
    --output-path /home/zhiheng/sglang/Vivid_Acceptance/result_videos/service_benchmark \
    --prompt-file-path /home/zhiheng/Vivid-VR/input/720p/prompt.txt \
    2>&1 | tee Vivid_Acceptance/logs/vividvr_flowcut_serve_$(date -u +%Y%m%dT%H%M%SZ).log'
```

启动后告知用户：

```bash
tmux attach -r -t vividvr_flowcut_serve
```

- [ ] **步骤 6：发送 FlowCut smoke 请求**

在 serve 健康后运行：

```bash
curl -sS -X POST http://127.0.0.1:31190/v1/videos/repairs/flowcut \
  -H 'Content-Type: application/json' \
  -d '{
    "taskId": "flowcut-vividvr-smoke-001",
    "timeout": -1,
    "callbackUrl": "http://127.0.0.1:39090/tasks/flowcut-vividvr-smoke-001/callback",
    "video_input_path": "/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.mp4",
    "caption_file_path": "/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.txt",
    "reference_video_path": "/home/zhiheng/Vivid-VR/result/720p_long_up1_result_vivid_ori_20step/videos/test_video_long_960x720_130f.mp4",
    "num_inference_steps": 20,
    "seed": 42
  }'
```

预期：30 秒内返回：

```json
{"code":0,"message":"ok"}
```

- [ ] **步骤 7：验证 FlowCut 回调目标**

检查 callback receiver 日志，必须至少包含：

```json
{"status":"running","progress":1,"reason":"accepted","output":""}
```

最终成功时必须包含：

```json
{"status":"succeeded","progress":100,"reason":"","output":"{\"result_url\":null,\"file_path\":\"...\",\"duration\":...}"}
```

如果生成失败，必须包含：

```json
{"status":"failed","progress":0,"reason":"...","output":""}
```

- [ ] **步骤 8：并发满实测**

在第一个长任务运行期间，再发第二个 FlowCut 请求：

```bash
curl -sS -X POST http://127.0.0.1:31190/v1/videos/repairs/flowcut \
  -H 'Content-Type: application/json' \
  -d '{
    "taskId": "flowcut-vividvr-smoke-002",
    "timeout": -1,
    "callbackUrl": "http://127.0.0.1:39090/tasks/flowcut-vividvr-smoke-002/callback",
    "video_input_path": "/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.mp4"
  }'
```

预期：

```json
{"code":2,"message":"A task is running."}
```

- [ ] **步骤 9：验收记录**

最终总结必须包含：

- 修改文件清单。
- 新增 endpoint：`POST /v1/videos/repairs/flowcut`。
- 单测命令与结果。
- tmux session 名称和 attach 命令。
- callback 日志路径。
- 是否跑了真实 Vivid-VR 推理；如未跑，说明未跑原因。

---

## 自检清单

- `docs_xzh/downstream-endpoint-spec.html` 的 `taskId`、`callbackUrl`、`timeout`、`minioConfig` 均有对应模型字段。
- `code:0/1/2` 均为 HTTP 200 JSON，且 `code` 是整数。
- 并发满不使用 HTTP 429。
- 业务参数错误不使用 HTTP 400。
- 接单成功后后台异步执行，不等待推理完成。
- 接单成功后必须回调 `running`，最终必须回调 `succeeded` 或 `failed`。
- `timeout=-1` 明确表示无服务侧推理超时。
- 不修改 `/v1/videos/repairs` 现有响应契约。
- 不使用 `/home/zhiheng/sglang_serve` 代码。
- 所有长推理验证均在 tmux 中运行。
