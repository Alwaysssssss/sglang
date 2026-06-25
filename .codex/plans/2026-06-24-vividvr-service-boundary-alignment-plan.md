# Vivid-VR 服务边界与 FlowCut 契约对齐实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 将当前嵌在共享 `video_api.py` 中的 Vivid-VR 服务路径拆成独立实现，在保持 `/v1/videos/repairs/flowcut` 外部路由不变的前提下，对齐 `docs_xzh/downstream-endpoint-spec.html` 的提交、回调、`result_url`、`progress`、`timeout` 与文件生命周期语义。

**架构：** 新增 `vividvr_flowcut_*` 模块承载 Vivid 自己的 request/response schema、request-scoped storage、stage-based progress、dispatch 和 router；共享层只保留队列、`VIDEO_STORE`、底层 callback retry、对象存储 helper 和纯生成执行器。`video_api.py` 不再通过 `if _is_vividvr...` 承载 Vivid 契约，只负责通用 OpenAI 风格视频服务，Vivid 走独立 router 但继续复用底层生成基础设施。

**技术栈：** Python 3.10、FastAPI、Pydantic v2、httpx、boto3、pytest、tmux、SGLang multimodal runtime

---

## 实施前必读

- `AGENTS.md`
- `docs_xzh/downstream-endpoint-spec.html`
- `docs_xzh/hand_over/flowcut_vividvr_service_compat_handover_20260622.md`
- `docs_xzh/hand_over/vividvr_service_external_access_and_caption_next_handover_20260622.md`
- `docs_xzh/hand_over/phase_e_default_configs_and_serve_followups_handover_20260622.md`

## 文件结构

- 创建：`python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_protocol.py`
  - Vivid 专属 FlowCut request / response / callback schema，固定 `result_url` 语义，接管 `taskId` / `timeout` / `callbackUrl` / `minioConfig` 的解析规则。
- 创建：`python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_storage.py`
  - Vivid request-scoped storage，对单请求输入下载、输出上传、本地 staging 清理负责。
- 创建：`python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_progress.py`
  - Vivid stage-based progress 映射与 callback payload builder，保证 `running` 上报单调、可信、可解释。
- 创建：`python/sglang/multimodal_gen/runtime/entrypoints/openai/video_job_runner.py`
  - 纯生成执行层，只负责调用 `process_generation_batch(...)`、收集产物路径和推理指标，不掺杂 callback / cloud upload / FlowCut 语义。
- 创建：`python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_api.py`
  - Vivid 独立 router、校验、参数映射、dispatch、timeout、callback 与清理责任。
- 修改：`python/sglang/multimodal_gen/runtime/entrypoints/openai/protocol.py`
  - 删除 FlowCut 为“通用视频协议”的事实地位；保留指向 `vividvr_flowcut_protocol.py` 的兼容别名，避免一次性打断旧 import。
- 修改：`python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py`
  - 移除 `/repairs/flowcut` 实现；移除 `create_video_repair` 里的 Vivid 分支；改用 `video_job_runner.py` 的纯执行层；当服务器处于 Vivid pipeline 时拒绝共享 `/repairs` 契约。
- 修改：`python/sglang/multimodal_gen/runtime/entrypoints/http_server.py`
  - 显式注册 `vividvr_flowcut_api.router`。
- 修改：`python/sglang/multimodal_gen/test/unit/test_flowcut_protocol.py`
  - 切到新协议模块，覆盖 `timeout=0 -> 300`、`result_url` 规范、无 `gen_video_url`。
- 创建：`python/sglang/multimodal_gen/test/unit/test_vividvr_flowcut_storage.py`
  - 覆盖输入 staging、输出上传、本地清理、上传失败保留文件、远端对象不删除。
- 创建：`python/sglang/multimodal_gen/test/unit/test_vividvr_flowcut_progress.py`
  - 覆盖 stage progress 单调递增、`reason` 显式、success callback 只返回 `result_url`。
- 修改：`python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py`
  - 挂载新 router，覆盖 `code=0/1/2`、Vivid-only 校验、callback payload、timeout 行为。
- 创建：`python/sglang/multimodal_gen/test/unit/test_video_job_runner.py`
  - 覆盖纯执行层不触发 callback、不触发 cloud upload，且能返回生成产物路径与指标。
- 修改：`python/sglang/multimodal_gen/test/unit/test_video_api_vividvr.py`
  - 共享 `/v1/videos/repairs` 在 Vivid server args 下应拒绝并指向专用路由；Wan/通用路径保持原行为。
- 修改：`python/sglang/multimodal_gen/test/unit/test_flowcut_service_acceptance_tool.py`
  - 固定 acceptance 工具对 `result_url` 和 `code=2` 重试行为的预期。
- 创建：`docs_xzh/hand_over/vividvr_service_boundary_alignment_handover_20260624.md`
  - 记录新服务边界、callback 语义、文件清理策略、验证命令与已知限制。

## 文件生命周期矩阵

- 远端原视频对象：默认不删除。
- 远端结果视频对象：默认不删除。
- 服务机本地输入副本：任务进入终态后删除。
- 服务机本地输出副本：上传成功后删除；上传失败时保留，便于排障。
- caption manifest / caption txt / request workdir：任务进入终态后删除，除非显式配置保留目录。

## 统一契约决策

- 提交响应只允许 `{"code": 0|1|2, "message": str}`。
- 成功 callback `output` 只允许 `{"result_url": "...", "duration": ...}`，禁止 `gen_video_url`、禁止 `file_path`。
- `timeout=0` 解释为 `300`，`timeout<0` 仅允许 `-1` 代表不启用服务侧超时。
- `running` callback 改为基于真实服务阶段上报；不再使用 elapsed time 伪造百分比。
- Vivid server 开启时，共享 `/v1/videos/repairs` 不再处理 Vivid 请求。

### 任务 1：冻结 Vivid 专属 FlowCut 协议并建立兼容别名

**文件：**
- 创建：`python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_protocol.py`
- 修改：`python/sglang/multimodal_gen/runtime/entrypoints/openai/protocol.py`
- 修改：`python/sglang/multimodal_gen/test/unit/test_flowcut_protocol.py`

- [ ] **步骤 1：编写失败的协议单测**

```python
import json

from sglang.multimodal_gen.runtime.entrypoints.openai.vividvr_flowcut_protocol import (
    VividVRFlowCutCallbackPayload,
    VividVRFlowCutRequest,
    VividVRFlowCutSubmitResponse,
)


def test_vividvr_flowcut_request_normalizes_timeout_zero_to_default():
    req = VividVRFlowCutRequest.model_validate(
        {
            "taskId": "task-1",
            "timeout": 0,
            "callbackUrl": "http://127.0.0.1:39090/tasks/task-1/callback",
            "video_input_path": "/tmp/input.mp4",
        }
    )

    assert req.task_id == "task-1"
    assert req.timeout == 300


def test_vividvr_flowcut_callback_payload_uses_result_url_only():
    payload = VividVRFlowCutCallbackPayload.succeeded(
        result_url="http://storage.example.com/out.mp4",
        duration=12.5,
    ).model_dump()

    assert payload["status"] == "succeeded"
    assert payload["progress"] == 100.0
    assert json.loads(payload["output"]) == {
        "result_url": "http://storage.example.com/out.mp4",
        "duration": 12.5,
    }
    assert "gen_video_url" not in payload["output"]
    assert "file_path" not in payload["output"]


def test_vividvr_flowcut_submit_response_uses_numeric_codes():
    response = VividVRFlowCutSubmitResponse(code=2, message="A task is running.")
    assert response.model_dump() == {"code": 2, "message": "A task is running."}
```

- [ ] **步骤 2：运行测试验证失败**

运行：`/home/zhiheng/sglang/.venv/bin/pytest python/sglang/multimodal_gen/test/unit/test_flowcut_protocol.py -v`

预期：FAIL，报错 `ModuleNotFoundError: No module named 'sglang.multimodal_gen.runtime.entrypoints.openai.vividvr_flowcut_protocol'`

- [ ] **步骤 3：实现 Vivid 专属协议模块并在通用 protocol 中保留兼容别名**

```python
# python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_protocol.py
import json
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class VividVRFlowCutMinIOConfig(BaseModel):
    endpoint: str
    bucket_name: str
    access_key: str
    secret_key: str
    secure: bool = False
    region: Optional[str] = None


class VividVRFlowCutRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    task_id: Optional[str] = Field(default=None, alias="taskId")
    timeout: int = 300
    callback_url: Optional[str] = Field(default=None, alias="callbackUrl")
    minio_config: Optional[VividVRFlowCutMinIOConfig] = Field(
        default=None,
        alias="minioConfig",
    )

    video_input_path: Optional[str] = None
    video_url: Optional[str] = None
    caption_file_path: Optional[str] = None
    reference_video_path: Optional[str] = None
    output_path: Optional[str] = None
    prompt: Optional[str] = None
    num_inference_steps: Optional[int] = None
    seed: int = 42

    @field_validator("timeout", mode="before")
    @classmethod
    def _normalize_timeout(cls, value):
        if value in (None, 0):
            return 300
        return value


class VividVRFlowCutSubmitResponse(BaseModel):
    code: int
    message: str = "ok"


class VividVRFlowCutCallbackPayload(BaseModel):
    status: Literal["running", "succeeded", "failed"]
    progress: float
    reason: str = ""
    output: str = ""

    @classmethod
    def running(cls, *, progress: float, reason: str) -> "VividVRFlowCutCallbackPayload":
        return cls(status="running", progress=progress, reason=reason, output="")

    @classmethod
    def succeeded(
        cls,
        *,
        result_url: str,
        duration: float | None,
    ) -> "VividVRFlowCutCallbackPayload":
        output = {"result_url": result_url}
        if duration is not None:
            output["duration"] = duration
        return cls(
            status="succeeded",
            progress=100.0,
            reason="",
            output=json.dumps(output, ensure_ascii=False),
        )

    @classmethod
    def failed(
        cls,
        *,
        reason: str,
        progress: float = 0.0,
    ) -> "VividVRFlowCutCallbackPayload":
        return cls(status="failed", progress=progress, reason=reason, output="")
```

```python
# python/sglang/multimodal_gen/runtime/entrypoints/openai/protocol.py
from sglang.multimodal_gen.runtime.entrypoints.openai.vividvr_flowcut_protocol import (
    VividVRFlowCutMinIOConfig as FlowCutMinIOConfig,
    VividVRFlowCutRequest as FlowCutVideoRepairRequest,
    VividVRFlowCutSubmitResponse as FlowCutResponse,
)
```

- [ ] **步骤 4：运行测试验证通过**

运行：`/home/zhiheng/sglang/.venv/bin/pytest python/sglang/multimodal_gen/test/unit/test_flowcut_protocol.py -v`

预期：PASS

- [ ] **步骤 5：Commit**

```bash
git add python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_protocol.py \
        python/sglang/multimodal_gen/runtime/entrypoints/openai/protocol.py \
        python/sglang/multimodal_gen/test/unit/test_flowcut_protocol.py
git commit -m "refactor: introduce vividvr flowcut protocol module"
```

### 任务 2：实现 Vivid request-scoped storage 与文件生命周期

**文件：**
- 创建：`python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_storage.py`
- 创建：`python/sglang/multimodal_gen/test/unit/test_vividvr_flowcut_storage.py`

- [ ] **步骤 1：编写失败的 storage 单测**

```python
import asyncio
from pathlib import Path

from sglang.multimodal_gen.runtime.entrypoints.openai.vividvr_flowcut_protocol import (
    VividVRFlowCutMinIOConfig,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.vividvr_flowcut_storage import (
    VividVRFlowCutStorage,
)


def test_storage_materializes_input_and_cleans_workdir(tmp_path):
    source = tmp_path / "input.mp4"
    source.write_bytes(b"fake mp4")

    storage = VividVRFlowCutStorage(job_id="job-1", base_dir=tmp_path / "jobs")
    local_path = asyncio.run(storage.materialize_video(str(source)))

    assert Path(local_path).exists()
    assert str(local_path).startswith(str(tmp_path / "jobs"))

    storage.cleanup()
    assert not (tmp_path / "jobs" / "job-1").exists()


def test_upload_result_removes_local_output_after_success(monkeypatch, tmp_path):
    output = tmp_path / "job.mp4"
    output.write_bytes(b"video bytes")
    uploaded = {}

    async def fake_upload_to_flowcut_minio(*, local_path, object_key, config):
        uploaded["local_path"] = local_path
        uploaded["object_key"] = object_key
        uploaded["config"] = config
        return "http://minio.example.com/flowcut/outputs/job-1.mp4"

    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.entrypoints.openai.vividvr_flowcut_storage.upload_to_flowcut_minio",
        fake_upload_to_flowcut_minio,
    )

    storage = VividVRFlowCutStorage(
        job_id="job-1",
        base_dir=tmp_path / "jobs",
        minio_config=VividVRFlowCutMinIOConfig(
            endpoint="minio.example.com",
            bucket_name="flowcut",
            access_key="ak",
            secret_key="sk",
            secure=False,
            region="us-east-1",
        ),
    )

    result_url = asyncio.run(storage.upload_result(str(output), "outputs/job-1.mp4"))

    assert result_url == "http://minio.example.com/flowcut/outputs/job-1.mp4"
    assert uploaded["local_path"] == str(output)
    assert not output.exists()


def test_upload_failure_preserves_local_output(monkeypatch, tmp_path):
    output = tmp_path / "job.mp4"
    output.write_bytes(b"video bytes")

    async def fake_upload_to_flowcut_minio(*, local_path, object_key, config):
        raise RuntimeError("minio unavailable")

    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.entrypoints.openai.vividvr_flowcut_storage.upload_to_flowcut_minio",
        fake_upload_to_flowcut_minio,
    )

    storage = VividVRFlowCutStorage(job_id="job-1", base_dir=tmp_path / "jobs")

    try:
        asyncio.run(storage.upload_result(str(output), "outputs/job-1.mp4"))
    except RuntimeError:
        pass

    assert output.exists()
```

- [ ] **步骤 2：运行测试验证失败**

运行：`/home/zhiheng/sglang/.venv/bin/pytest python/sglang/multimodal_gen/test/unit/test_vividvr_flowcut_storage.py -v`

预期：FAIL，报错 `ModuleNotFoundError: No module named 'sglang.multimodal_gen.runtime.entrypoints.openai.vividvr_flowcut_storage'`

- [ ] **步骤 3：实现 request-scoped storage**

```python
# python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_storage.py
import os
import shutil
from pathlib import Path

import httpx

from sglang.multimodal_gen.runtime.entrypoints.openai.flowcut import (
    upload_to_flowcut_minio,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.vividvr_flowcut_protocol import (
    VividVRFlowCutMinIOConfig,
)


class VividVRFlowCutStorage:
    def __init__(
        self,
        *,
        job_id: str,
        base_dir: str | Path,
        minio_config: VividVRFlowCutMinIOConfig | None = None,
    ) -> None:
        self.job_id = job_id
        self.base_dir = Path(base_dir)
        self.job_dir = self.base_dir / job_id
        self.inputs_dir = self.job_dir / "inputs"
        self.outputs_dir = self.job_dir / "outputs"
        self.minio_config = minio_config
        self.inputs_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    async def materialize_video(self, source: str) -> str:
        suffix = Path(source.split("?", 1)[0]).suffix or ".mp4"
        target = self.inputs_dir / f"input{suffix}"
        if source.lower().startswith(("http://", "https://")):
            async with httpx.AsyncClient(follow_redirects=True) as client:
                response = await client.get(source, timeout=60.0)
                response.raise_for_status()
            target.write_bytes(response.content)
            return str(target)

        shutil.copyfile(source, target)
        return str(target)

    def output_file_path(self, filename: str) -> str:
        return str(self.outputs_dir / filename)

    async def upload_result(self, local_path: str, object_key: str) -> str:
        if self.minio_config is None:
            return local_path

        result_url = await upload_to_flowcut_minio(
            local_path=local_path,
            object_key=object_key,
            config=self.minio_config,
        )
        Path(local_path).unlink(missing_ok=True)
        return result_url

    def cleanup(self) -> None:
        shutil.rmtree(self.job_dir, ignore_errors=True)
```

- [ ] **步骤 4：运行测试验证通过**

运行：`/home/zhiheng/sglang/.venv/bin/pytest python/sglang/multimodal_gen/test/unit/test_vividvr_flowcut_storage.py -v`

预期：PASS

- [ ] **步骤 5：Commit**

```bash
git add python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_storage.py \
        python/sglang/multimodal_gen/test/unit/test_vividvr_flowcut_storage.py
git commit -m "feat: add vividvr request scoped storage"
```

### 任务 3：实现 Vivid 专属 progress 与 callback payload 语义

**文件：**
- 创建：`python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_progress.py`
- 创建：`python/sglang/multimodal_gen/test/unit/test_vividvr_flowcut_progress.py`

- [ ] **步骤 1：编写失败的 progress 单测**

```python
import json

from sglang.multimodal_gen.runtime.entrypoints.openai.vividvr_flowcut_progress import (
    VividVRFlowCutProgressReporter,
    build_vividvr_failed_payload,
    build_vividvr_success_payload,
)


def test_progress_reporter_is_monotonic_and_stage_driven():
    reporter = VividVRFlowCutProgressReporter()

    accepted = reporter.mark("accepted")
    input_ready = reporter.mark("input_ready")
    editing = reporter.mark("editing")
    uploading = reporter.mark("uploading_result")

    assert accepted.progress == 1.0
    assert input_ready.progress > accepted.progress
    assert editing.progress > input_ready.progress
    assert uploading.progress > editing.progress
    assert uploading.reason == "uploading result"


def test_success_payload_uses_result_url_without_local_path():
    payload = build_vividvr_success_payload(
        result_url="http://storage.example.com/out.mp4",
        duration=9.8,
    )

    assert payload["status"] == "succeeded"
    assert payload["progress"] == 100.0
    assert json.loads(payload["output"]) == {
        "result_url": "http://storage.example.com/out.mp4",
        "duration": 9.8,
    }
    assert "file_path" not in payload["output"]


def test_failed_payload_keeps_last_known_progress():
    payload = build_vividvr_failed_payload(
        reason="task timeout",
        progress=70.0,
    )

    assert payload == {
        "status": "failed",
        "progress": 70.0,
        "reason": "task timeout",
        "output": "",
    }
```

- [ ] **步骤 2：运行测试验证失败**

运行：`/home/zhiheng/sglang/.venv/bin/pytest python/sglang/multimodal_gen/test/unit/test_vividvr_flowcut_progress.py -v`

预期：FAIL，报错 `ModuleNotFoundError: No module named 'sglang.multimodal_gen.runtime.entrypoints.openai.vividvr_flowcut_progress'`

- [ ] **步骤 3：实现 stage-based progress 与 payload builder**

```python
# python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_progress.py
from dataclasses import dataclass

from sglang.multimodal_gen.runtime.entrypoints.openai.vividvr_flowcut_protocol import (
    VividVRFlowCutCallbackPayload,
)


_STAGE_PROGRESS = {
    "accepted": (1.0, "accepted"),
    "input_ready": (8.0, "input ready"),
    "caption_ready": (15.0, "caption ready"),
    "editing": (60.0, "editing video"),
    "merging": (90.0, "merging clips"),
    "uploading_result": (98.0, "uploading result"),
}


@dataclass(frozen=True)
class VividVRProgressEvent:
    stage: str
    progress: float
    reason: str


class VividVRFlowCutProgressReporter:
    def __init__(self) -> None:
        self._last_progress = 0.0

    def mark(self, stage: str) -> VividVRProgressEvent:
        progress, reason = _STAGE_PROGRESS[stage]
        if progress < self._last_progress:
            progress = self._last_progress
        self._last_progress = progress
        return VividVRProgressEvent(stage=stage, progress=progress, reason=reason)


def build_vividvr_running_payload(event: VividVRProgressEvent) -> dict:
    return VividVRFlowCutCallbackPayload.running(
        progress=event.progress,
        reason=event.reason,
    ).model_dump()


def build_vividvr_success_payload(*, result_url: str, duration: float | None) -> dict:
    return VividVRFlowCutCallbackPayload.succeeded(
        result_url=result_url,
        duration=duration,
    ).model_dump()


def build_vividvr_failed_payload(*, reason: str, progress: float = 0.0) -> dict:
    return VividVRFlowCutCallbackPayload.failed(
        reason=reason,
        progress=progress,
    ).model_dump()
```

- [ ] **步骤 4：运行测试验证通过**

运行：`/home/zhiheng/sglang/.venv/bin/pytest python/sglang/multimodal_gen/test/unit/test_vividvr_flowcut_progress.py -v`

预期：PASS

- [ ] **步骤 5：Commit**

```bash
git add python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_progress.py \
        python/sglang/multimodal_gen/test/unit/test_vividvr_flowcut_progress.py
git commit -m "feat: add vividvr flowcut progress semantics"
```

### 任务 4：抽取纯生成执行层，去掉共享 dispatch 中的契约耦合

**文件：**
- 创建：`python/sglang/multimodal_gen/runtime/entrypoints/openai/video_job_runner.py`
- 创建：`python/sglang/multimodal_gen/test/unit/test_video_job_runner.py`
- 修改：`python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py`

- [ ] **步骤 1：编写失败的纯执行层单测**

```python
import asyncio
from types import SimpleNamespace

from sglang.multimodal_gen.runtime.entrypoints.openai.video_job_runner import (
    run_video_generation_job,
)


def test_run_video_generation_job_returns_artifact_without_callback_or_upload(monkeypatch, tmp_path):
    output = tmp_path / "job.mp4"
    output.write_bytes(b"fake video")

    async def fake_process_generation_batch(async_client, batch):
        return [str(output)], {"peak_memory_mb": 111.0, "inference_time_s": 22.5}

    def fail_upload_and_cleanup(*args, **kwargs):
        raise AssertionError("pure runner must not upload")

    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.entrypoints.openai.video_job_runner.process_generation_batch",
        fake_process_generation_batch,
    )
    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.entrypoints.openai.video_api.cloud_storage.upload_and_cleanup",
        fail_upload_and_cleanup,
    )

    run = asyncio.run(run_video_generation_job(batch=SimpleNamespace()))

    assert run.save_file_path == str(output)
    assert run.metrics["peak_memory_mb"] == 111.0
    assert run.metrics["inference_time_s"] == 22.5
```

- [ ] **步骤 2：运行测试验证失败**

运行：`/home/zhiheng/sglang/.venv/bin/pytest python/sglang/multimodal_gen/test/unit/test_video_job_runner.py -v`

预期：FAIL，报错 `ModuleNotFoundError: No module named 'sglang.multimodal_gen.runtime.entrypoints.openai.video_job_runner'`

- [ ] **步骤 3：实现纯执行层并让共享 `video_api.py` 复用它**

```python
# python/sglang/multimodal_gen/runtime/entrypoints/openai/video_job_runner.py
import time
from dataclasses import dataclass
from typing import Any

from sglang.multimodal_gen.runtime.entrypoints.openai.common_api import (
    add_common_data_to_response,
)
from sglang.multimodal_gen.runtime.managers.io_struct import Req
from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client
from sglang.multimodal_gen.runtime.utils import process_generation_batch


@dataclass
class VideoJobRunResult:
    save_file_path: str
    metrics: dict[str, Any]
    completed_at: int


async def run_video_generation_job(*, batch: Req) -> VideoJobRunResult:
    save_file_path_list, result = await process_generation_batch(
        async_scheduler_client,
        batch,
    )
    metrics = add_common_data_to_response({}, request_id="video-job", result=result)
    return VideoJobRunResult(
        save_file_path=save_file_path_list[0],
        metrics=metrics,
        completed_at=int(time.time()),
    )
```

```python
# python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py
from sglang.multimodal_gen.runtime.entrypoints.openai.video_job_runner import (
    run_video_generation_job,
)


async def _dispatch_job_async(...):
    try:
        run = await run_video_generation_job(batch=batch)
        save_file_path = run.save_file_path
        cloud_url = await cloud_storage.upload_and_cleanup(save_file_path)
        update_fields = {
            "status": "completed",
            "progress": 100,
            "completed_at": run.completed_at,
            "url": cloud_url,
            "file_path": save_file_path if not cloud_url and output_persistent else None,
            **run.metrics,
        }
```

- [ ] **步骤 4：运行测试验证通过**

运行：`/home/zhiheng/sglang/.venv/bin/pytest python/sglang/multimodal_gen/test/unit/test_video_job_runner.py -v`

预期：PASS

- [ ] **步骤 5：Commit**

```bash
git add python/sglang/multimodal_gen/runtime/entrypoints/openai/video_job_runner.py \
        python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py \
        python/sglang/multimodal_gen/test/unit/test_video_job_runner.py
git commit -m "refactor: extract pure video job runner"
```

### 任务 5：实现独立的 Vivid FlowCut router 与 dispatch

**文件：**
- 创建：`python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_api.py`
- 修改：`python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py`

- [ ] **步骤 1：编写失败的 Vivid FlowCut API 单测**

```python
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from sglang.multimodal_gen.runtime.entrypoints.openai import vividvr_flowcut_api


def _make_client():
    app = FastAPI()
    app.include_router(vividvr_flowcut_api.router)
    return TestClient(app)


def test_flowcut_endpoint_accepts_timeout_zero_and_schedules_vivid_job(monkeypatch, tmp_path):
    scheduled = {}

    class AvailableSemaphore:
        def locked(self):
            return False

        async def acquire(self):
            return None

        def release(self):
            return None

    monkeypatch.setattr(vividvr_flowcut_api, "_VIDEOEDIT_SEMAPHORE", AvailableSemaphore())
    monkeypatch.setattr(
        vividvr_flowcut_api,
        "get_global_server_args",
        lambda: SimpleNamespace(
            output_path=str(tmp_path / "outputs"),
            input_save_path=str(tmp_path / "inputs"),
            model_id="vividvr",
            pipeline_class_name="CogVideoXVividVRControlNetPipeline",
            prompt_file_path=str(tmp_path / "prompt.txt"),
            pipeline_config=SimpleNamespace(default_prompt_file_path=str(tmp_path / "prompt.txt")),
        ),
    )
    monkeypatch.setattr(
        vividvr_flowcut_api,
        "prepare_request",
        lambda server_args, sampling_params: "prepared-batch",
    )
    monkeypatch.setattr(
        vividvr_flowcut_api.VividVRSamplingParams,
        "from_user_kwargs",
        staticmethod(lambda server_args, **kwargs: SimpleNamespace(output_file_path=lambda: str(tmp_path / "outputs" / "task-1.mp4"))),
    )

    def fake_create_task(coro):
        scheduled["coro_name"] = coro.cr_code.co_name
        coro.close()
        return None

    monkeypatch.setattr(vividvr_flowcut_api.asyncio, "create_task", fake_create_task)

    client = _make_client()
    response = client.post(
        "/v1/videos/repairs/flowcut",
        json={
            "taskId": "task-1",
            "timeout": 0,
            "callbackUrl": "http://127.0.0.1:39090/tasks/task-1/callback",
            "video_input_path": "/tmp/input.mp4",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"code": 0, "message": "ok"}
    assert scheduled["coro_name"] == "_dispatch_vividvr_flowcut_job_async"


def test_flowcut_endpoint_rejects_non_vivid_pipeline(monkeypatch):
    class AvailableSemaphore:
        def locked(self):
            return False

        async def acquire(self):
            return None

        def release(self):
            return None

    monkeypatch.setattr(vividvr_flowcut_api, "_VIDEOEDIT_SEMAPHORE", AvailableSemaphore())
    monkeypatch.setattr(
        vividvr_flowcut_api,
        "get_global_server_args",
        lambda: SimpleNamespace(model_id="wan", pipeline_class_name="WanVideoEditPipeline"),
    )

    client = _make_client()
    response = client.post(
        "/v1/videos/repairs/flowcut",
        json={
            "taskId": "task-1",
            "callbackUrl": "http://127.0.0.1:39090/tasks/task-1/callback",
            "video_input_path": "/tmp/input.mp4",
        },
    )

    assert response.status_code == 200
    assert response.json()["code"] == 1
```

- [ ] **步骤 2：运行测试验证失败**

运行：`/home/zhiheng/sglang/.venv/bin/pytest python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py -v`

预期：FAIL，报错 `ImportError: cannot import name 'vividvr_flowcut_api'`

- [ ] **步骤 3：实现独立 router、dispatch、timeout 和 callback**

```python
# python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_api.py
import asyncio
import os
import tempfile

from fastapi import APIRouter, Request

from sglang.multimodal_gen.runtime.entrypoints.openai.flowcut import post_flowcut_callback
from sglang.multimodal_gen.runtime.entrypoints.openai.video_api import (
    VIDEO_STORE,
    _VIDEOEDIT_SEMAPHORE,
    _build_vividvr_repair_kwargs,
    _copy_video_repair_request_with_caption,
    _ensure_vividvr_caption_file,
    _is_vividvr_video_repair_pipeline,
    _resolve_video_repair_model_name,
    _video_repair_job_from_sampling,
    get_global_server_args,
    prepare_request,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.video_job_runner import (
    run_video_generation_job,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.vividvr_flowcut_progress import (
    VividVRFlowCutProgressReporter,
    build_vividvr_failed_payload,
    build_vividvr_running_payload,
    build_vividvr_success_payload,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.vividvr_flowcut_protocol import (
    VividVRFlowCutRequest,
    VividVRFlowCutSubmitResponse,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.vividvr_flowcut_storage import (
    VividVRFlowCutStorage,
)
from sglang.multimodal_gen.runtime.video.vividvr_params import VividVRSamplingParams


router = APIRouter(prefix="/v1/videos", tags=["videos"])


async def _dispatch_vividvr_flowcut_job_async(
    *,
    job_id: str,
    batch,
    callback_url: str,
    storage: VividVRFlowCutStorage,
    timeout: int,
) -> None:
    reporter = VividVRFlowCutProgressReporter()
    last_progress = 0.0
    try:
        accepted = reporter.mark("accepted")
        last_progress = accepted.progress
        await post_flowcut_callback(callback_url, build_vividvr_running_payload(accepted))
        await VIDEO_STORE.update_fields(job_id, {"status": "running", "progress": accepted.progress})

        editing = reporter.mark("editing")
        last_progress = editing.progress
        await post_flowcut_callback(callback_url, build_vividvr_running_payload(editing))
        run = await asyncio.wait_for(run_video_generation_job(batch=batch), timeout=timeout) if timeout != -1 else await run_video_generation_job(batch=batch)

        uploading = reporter.mark("uploading_result")
        last_progress = uploading.progress
        await post_flowcut_callback(callback_url, build_vividvr_running_payload(uploading))
        result_url = await storage.upload_result(
            run.save_file_path,
            f"outputs/{job_id}.mp4",
        )
        if result_url == run.save_file_path:
            result_url = run.save_file_path

        await VIDEO_STORE.update_fields(
            job_id,
            {
                "status": "completed",
                "progress": 100,
                "completed_at": run.completed_at,
                "url": result_url,
                "file_path": None,
                **run.metrics,
            },
        )
        await post_flowcut_callback(
            callback_url,
            build_vividvr_success_payload(
                result_url=result_url,
                duration=run.metrics.get("inference_time_s"),
            ),
        )
    except asyncio.TimeoutError:
        await VIDEO_STORE.update_fields(
            job_id,
            {"status": "failed", "progress": last_progress, "error": {"message": "task timeout"}},
        )
        await post_flowcut_callback(
            callback_url,
            build_vividvr_failed_payload(reason="task timeout", progress=last_progress),
        )
    except Exception as exc:
        await VIDEO_STORE.update_fields(
            job_id,
            {"status": "failed", "progress": last_progress, "error": {"message": str(exc)}},
        )
        await post_flowcut_callback(
            callback_url,
            build_vividvr_failed_payload(reason=str(exc), progress=last_progress),
        )
    finally:
        storage.cleanup()
        _VIDEOEDIT_SEMAPHORE.release()


@router.post("/repairs/flowcut", response_model=VividVRFlowCutSubmitResponse)
async def create_vividvr_flowcut_repair(request: Request):
    payload = await request.json()
    req = VividVRFlowCutRequest.model_validate(payload)

    if _VIDEOEDIT_SEMAPHORE.locked():
        return VividVRFlowCutSubmitResponse(code=2, message="A task is running.")
    if not req.task_id:
        return VividVRFlowCutSubmitResponse(code=1, message="taskId is required")
    if not req.callback_url:
        return VividVRFlowCutSubmitResponse(code=1, message="callbackUrl is required")
    if not (req.video_input_path or req.video_url):
        return VividVRFlowCutSubmitResponse(code=1, message="video_input_path or video_url is required")

    server_args = get_global_server_args()
    if not _is_vividvr_video_repair_pipeline(server_args):
        return VividVRFlowCutSubmitResponse(code=1, message="FlowCut repair endpoint requires Vivid-VR pipeline")

    await _VIDEOEDIT_SEMAPHORE.acquire()

    base_dir = server_args.output_path or server_args.input_save_path or tempfile.gettempdir()
    storage = VividVRFlowCutStorage(job_id=req.task_id, base_dir=base_dir, minio_config=req.minio_config)
    input_path = await storage.materialize_video(req.video_input_path or req.video_url)
    req_for_sampling = _copy_video_repair_request_with_caption(
        req,
        caption_file_path=await _ensure_vividvr_caption_file(server_args=server_args, request_id=req.task_id, video_input_path=input_path, caption_file_path=req.caption_file_path),
    )
    sampling_params = VividVRSamplingParams.from_user_kwargs(
        server_args,
        **_build_vividvr_repair_kwargs(
            request_id=req.task_id,
            req=req_for_sampling,
            server_args=server_args,
            video_input_path=input_path,
            output_dir=os.path.dirname(storage.output_file_path(f"{req.task_id}.mp4")),
            output_file_name=f"{req.task_id}.mp4",
        ),
    )
    batch = prepare_request(server_args, sampling_params)
    job = _video_repair_job_from_sampling(req.task_id, req_for_sampling, sampling_params)
    job["model"] = _resolve_video_repair_model_name(req_for_sampling, server_args, "VividVR")
    await VIDEO_STORE.upsert(req.task_id, job)

    asyncio.create_task(
        _dispatch_vividvr_flowcut_job_async(
            job_id=req.task_id,
            batch=batch,
            callback_url=req.callback_url,
            storage=storage,
            timeout=req.timeout,
        )
    )
    return VividVRFlowCutSubmitResponse(code=0, message="ok")
```

- [ ] **步骤 4：运行测试验证通过**

运行：`/home/zhiheng/sglang/.venv/bin/pytest python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py -v`

预期：PASS

- [ ] **步骤 5：Commit**

```bash
git add python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_api.py \
        python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py
git commit -m "feat: add dedicated vividvr flowcut service router"
```

### 任务 6：把共享 `video_api.py` 收口为通用 OpenAI 服务，并注册新 router

**文件：**
- 修改：`python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py`
- 修改：`python/sglang/multimodal_gen/runtime/entrypoints/http_server.py`
- 修改：`python/sglang/multimodal_gen/test/unit/test_video_api_vividvr.py`

- [ ] **步骤 1：编写失败的边界单测**

```python
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sglang.multimodal_gen.runtime.entrypoints.openai import video_api


def test_shared_video_repair_route_rejects_vivid_pipeline(monkeypatch):
    app = FastAPI()
    app.include_router(video_api.router)

    monkeypatch.setattr(
        video_api,
        "get_global_server_args",
        lambda: type(
            "Args",
            (),
            {
                "model_id": "vividvr",
                "pipeline_class_name": "CogVideoXVividVRControlNetPipeline",
                "pipeline_config": type("Cfg", (), {"default_prompt_file_path": None})(),
                "output_path": "/tmp",
                "input_save_path": "/tmp",
            },
        )(),
    )

    with TestClient(app) as client:
        response = client.post("/v1/videos/repairs", json={"video_input_path": "/tmp/input.mp4"})

    assert response.status_code == 400
    assert "/v1/videos/repairs/flowcut" in response.json()["detail"]
```

- [ ] **步骤 2：运行测试验证失败**

运行：`/home/zhiheng/sglang/.venv/bin/pytest python/sglang/multimodal_gen/test/unit/test_video_api_vividvr.py -v`

预期：FAIL，现有实现仍会在 Vivid pipeline 下接受共享 `/v1/videos/repairs`

- [ ] **步骤 3：删除共享路由中的 Vivid 契约并在 http_server 注册专用 router**

```python
# python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py
from fastapi import HTTPException


@router.post("/repairs", response_model=VideoResponse)
async def create_video_repair(req: VideoRepairRequest):
    server_args = get_global_server_args()
    if _is_vividvr_video_repair_pipeline(server_args):
        raise HTTPException(
            status_code=400,
            detail="Vivid-VR uses dedicated /v1/videos/repairs/flowcut service contract",
        )

    if not (req.video_input_path or req.video_url):
        raise HTTPException(status_code=400, detail="video_input_path or video_url is required")
    if not (req.mask_input_path or req.mask_url):
        raise HTTPException(status_code=400, detail="mask_input_path or mask_url is required")
    # 其余保留 Wan / 通用修复逻辑，不再进入 Vivid 参数分支
```

```python
# python/sglang/multimodal_gen/runtime/entrypoints/http_server.py
from sglang.multimodal_gen.runtime.entrypoints.openai import vividvr_flowcut_api

app.include_router(video_api.router)
app.include_router(vividvr_flowcut_api.router)
```

- [ ] **步骤 4：运行测试验证通过**

运行：`/home/zhiheng/sglang/.venv/bin/pytest python/sglang/multimodal_gen/test/unit/test_video_api_vividvr.py -v`

预期：PASS

- [ ] **步骤 5：Commit**

```bash
git add python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py \
        python/sglang/multimodal_gen/runtime/entrypoints/http_server.py \
        python/sglang/multimodal_gen/test/unit/test_video_api_vividvr.py
git commit -m "refactor: remove vividvr contract from shared video api"
```

### 任务 7：更新 acceptance、交接文档并完成验证

**文件：**
- 修改：`python/sglang/multimodal_gen/test/unit/test_flowcut_service_acceptance_tool.py`
- 创建：`docs_xzh/hand_over/vividvr_service_boundary_alignment_handover_20260624.md`

- [ ] **步骤 1：补 acceptance 工具断言与交接文档草稿**

```python
# python/sglang/multimodal_gen/test/unit/test_flowcut_service_acceptance_tool.py
def test_local_callback_server_records_result_url_payload(tmp_path):
    callback_log = tmp_path / "callback.jsonl"
    recorder = _FlowCutCallbackRecorder(str(callback_log))

    with _LocalFlowCutCallbackServer(
        host="127.0.0.1",
        port=0,
        task_id="task-1",
        recorder=recorder,
    ) as server:
        with httpx.Client(trust_env=False) as client:
            response = client.post(
                server.callback_url,
                json={
                    "status": "succeeded",
                    "progress": 100,
                    "reason": "",
                    "output": "{\"result_url\":\"http://storage.example.com/out.mp4\"}",
                },
            )
            response.raise_for_status()

        final_payload = recorder.wait_for_final(timeout=1.0)

    assert "result_url" in final_payload["output"]
    assert "gen_video_url" not in final_payload["output"]
```

```markdown
# docs_xzh/hand_over/vividvr_service_boundary_alignment_handover_20260624.md

## Scope

- Vivid-VR FlowCut 服务从共享 `video_api.py` 中拆分为独立 router。
- `/v1/videos/repairs/flowcut` 继续作为对外入口。
- callback success payload 统一只输出 `result_url`。
- `running` progress 改为 stage-based 真实阶段上报，不再使用 elapsed time heartbeat。
- request-scoped storage 负责本地 staging 文件清理；远端 MinIO 对象默认不删除。

## Verification

- `python/sglang/multimodal_gen/test/unit/test_flowcut_protocol.py`
- `python/sglang/multimodal_gen/test/unit/test_vividvr_flowcut_storage.py`
- `python/sglang/multimodal_gen/test/unit/test_vividvr_flowcut_progress.py`
- `python/sglang/multimodal_gen/test/unit/test_video_job_runner.py`
- `python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py`
- `python/sglang/multimodal_gen/test/unit/test_video_api_vividvr.py`
- `python/sglang/multimodal_gen/test/unit/test_flowcut_service_acceptance_tool.py`
```

- [ ] **步骤 2：运行目标单测**

运行：

```bash
/home/zhiheng/sglang/.venv/bin/pytest \
  python/sglang/multimodal_gen/test/unit/test_flowcut_protocol.py \
  python/sglang/multimodal_gen/test/unit/test_vividvr_flowcut_storage.py \
  python/sglang/multimodal_gen/test/unit/test_vividvr_flowcut_progress.py \
  python/sglang/multimodal_gen/test/unit/test_video_job_runner.py \
  python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py \
  python/sglang/multimodal_gen/test/unit/test_video_api_vividvr.py \
  python/sglang/multimodal_gen/test/unit/test_flowcut_service_acceptance_tool.py \
  -v
```

预期：PASS

- [ ] **步骤 3：运行静态校验**

运行：

```bash
/home/zhiheng/sglang/.venv/bin/python -m py_compile \
  python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_protocol.py \
  python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_storage.py \
  python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_progress.py \
  python/sglang/multimodal_gen/runtime/entrypoints/openai/video_job_runner.py \
  python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_api.py
git diff --check
```

预期：全部通过，无语法错误、无尾随空格、无 patch 冲突标记

- [ ] **步骤 4：Commit**

```bash
git add python/sglang/multimodal_gen/test/unit/test_flowcut_service_acceptance_tool.py \
        docs_xzh/hand_over/vividvr_service_boundary_alignment_handover_20260624.md
git commit -m "docs: record vividvr service boundary alignment"
```

## 计划自检

- 规格覆盖度：
  - 服务边界：任务 5、任务 6。
  - request/response schema：任务 1。
  - `result_url` 语义：任务 1、任务 3、任务 7。
  - progress 对齐：任务 3、任务 5。
  - request-scoped storage 与本地清理：任务 2、任务 5。
  - 纯共享基础设施抽取：任务 4。
  - 验收与交接：任务 7。
- 占位符扫描：
  - 本计划不使用 `TODO`、`待定`、`后续实现`、`类似任务 N`。
- 类型一致性：
  - 新类型统一使用 `VividVRFlowCut*` 命名。
  - 共享执行器统一使用 `run_video_generation_job(...)` / `VideoJobRunResult`。
  - callback success 字段统一为 `result_url`。

## 执行交接

计划已完成并保存到 `.codex/plans/2026-06-24-vividvr-service-boundary-alignment-plan.md`。两种执行方式：

**1. 子代理驱动（推荐）** - 每个任务调度一个新的子代理，任务间进行审查，快速迭代

**2. 内联执行** - 在当前会话中使用 executing-plans 执行任务，批量执行并设有检查点

选哪种方式？
