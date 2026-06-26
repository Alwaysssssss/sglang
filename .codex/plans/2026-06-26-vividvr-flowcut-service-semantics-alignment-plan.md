# Vivid-VR FlowCut 服务语义对齐实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 保留 Vivid-VR 独立的 FlowCut 协议、storage、progress 和 caption bridge，实现与 `origin/share-tyx` `video_edit` 在请求契约、生命周期、对象存储、回调和超时上的服务语义对齐。

**架构：** 继续以 `vividvr_flowcut_api.py` 为独立入口，不复用 `video_api.py` 的主处理链；只迁移 `share-tyx` 中对请求端可见的服务行为。Vivid-VR 模型执行链、caption bridge、runtime progress 来源和 FlowCut callback payload 继续保持独立，实现“服务契约对齐，模型语义分离”。

**技术栈：** Python 3.10、FastAPI、Pydantic v2、httpx、boto3、pytest、SGLang multimodal runtime

---

## 实施前必读

- `AGENTS.md`
- `docs_xzh/hand_over/vividvr_compile_stabilization_and_env_packaging_handover_20260625.md`
- `docs_xzh/hand_over/flowcut_vividvr_service_compat_handover_20260622.md`
- `docs_xzh/hand_over/vividvr_service_external_access_and_caption_next_handover_20260622.md`
- `python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_api.py`
- `python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_protocol.py`
- `python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_storage.py`
- `origin/share-tyx:python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py`

## 文件结构

- 修改：`python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_protocol.py`
  - 固定 Vivid-VR FlowCut 的请求字段、alias 归一化、`timeout` 语义和响应约束。
- 修改：`python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_api.py`
  - 负责请求归一化、failed submission 入库、caption bridge 前后的状态推进、timeout、callback bookkeeping、cleanup 策略。
- 修改：`python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_storage.py`
  - 负责 request workdir 结构、输入 materialize、输出上传、对象 key/bucket 解析和终态清理策略。
- 修改：`python/sglang/multimodal_gen/runtime/entrypoints/openai/flowcut.py`
  - 负责 callback retry 复用和 MinIO 上传 helper 扩展，使 Vivid FlowCut 能消费 `output_object_key` / `output_bucket`。
- 修改：`python/sglang/multimodal_gen/runtime/entrypoints/openai/protocol.py`
  - 只保留从 `vividvr_flowcut_protocol.py` re-export 的兼容别名，不重新定义 Vivid 请求语义。
- 修改：`python/sglang/multimodal_gen/test/unit/test_flowcut_protocol.py`
  - 覆盖请求 alias、`timeout`、响应 payload 和字段约束。
- 修改：`python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py`
  - 覆盖 failed submission、callback bookkeeping、timeout reason、output object key 透传和 Vivid 专属队列语义。
- 创建：`python/sglang/multimodal_gen/test/unit/test_vividvr_flowcut_storage.py`
  - 覆盖 request workdir、临时目录清理、持久目录保留、上传成功/失败后的文件生命周期。
- 修改：`docs_xzh/run_command/mock_test.md`
  - 同步示例请求中的输出存储、失败响应和 callback 语义。
- 创建：`docs_xzh/hand_over/vividvr_flowcut_service_semantics_alignment_handover_20260626.md`
  - 记录对齐后的契约、清理矩阵、回归命令和剩余风险。

## 对齐边界

- **必须对齐：**
  - 请求字段 alias / 默认值 / 非法请求处理
  - `timeout` 语义
  - failed submission 是否入库
  - `output_object_key` / `output_bucket` / MinIO 上传行为
  - callback bookkeeping
  - 输入下载与临时目录清理策略
- **明确不对齐：**
  - `video_api.py` 主入口
  - `WanVideoEditSamplingParams`
  - `mask` / `reference_image` / `bbox` 等编辑模型语义
  - Vivid-VR caption bridge 和 runtime progress 文件格式

## 生命周期决策

- `input_save_path` 或 `output_path` 已配置时：保留 request workdir，行为与 `video_edit` 的持久目录模式对齐。
- 未配置持久目录且结果已经外部化时：清理临时 request workdir。
- “结果已经外部化” 定义为：
  - MinIO 上传成功，或
  - `output_path` 指向持久落盘位置。
- 未配置持久目录且未上传 MinIO 时：保留 request workdir，避免 FlowCut callback 中的本地 `result_url` 立即失效。

### 任务 1：对齐请求契约与 failed submission 语义

**文件：**
- 修改：`python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_protocol.py`
- 修改：`python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_api.py`
- 修改：`python/sglang/multimodal_gen/test/unit/test_flowcut_protocol.py`
- 修改：`python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py`

- [ ] **步骤 1：编写失败的请求契约测试**

```python
def test_vividvr_flowcut_request_accepts_nested_minio_aliases():
    req = VividVRFlowCutRequest.model_validate(
        {
            "taskId": "task-1",
            "callbackUrl": "http://127.0.0.1:9000/callback",
            "video_url": "https://example.com/in.mp4",
            "minioConfig": {
                "endpoint": "minio.example.com:9000",
                "bucketName": "bucket-a",
                "accessKey": "ak",
                "secretKey": "sk",
            },
        }
    )

    assert req.minio_config is not None
    assert req.minio_config.bucket_name == "bucket-a"


def test_vividvr_flowcut_request_rejects_timeout_less_than_minus_one():
    with pytest.raises(ValidationError):
        VividVRFlowCutRequest.model_validate({"timeout": -2})


@pytest.mark.asyncio
async def test_invalid_flowcut_request_with_task_id_is_persisted_as_failed(client):
    response = await client.post(
        "/v1/videos/repairs/flowcut",
        json={"taskId": "bad-task", "timeout": -2},
    )

    assert response.json()["code"] == 1
    job = await VIDEO_STORE.get("bad-task")
    assert job["status"] == "failed"
    assert "timeout" in job["reason"]
```

- [ ] **步骤 2：运行测试验证失败**

运行：`PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest python/sglang/multimodal_gen/test/unit/test_flowcut_protocol.py python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py -k "minio_aliases or timeout_less_than_minus_one or persisted_as_failed" -q`

预期：FAIL，当前协议还不接受 `bucketName/accessKey/secretKey` alias，且无效请求不会写入 `VIDEO_STORE`。

- [ ] **步骤 3：实现最少请求归一化与 failed submission 逻辑**

```python
_FLOWCUT_FIELD_ALIASES = {
    "taskId": "task_id",
    "callbackUrl": "callback_url",
    "minioConfig": "minio_config",
    "outputObjectKey": "output_object_key",
}

_FLOWCUT_MINIO_FIELD_ALIASES = {
    "bucketName": "bucket_name",
    "accessKey": "access_key",
    "secretKey": "secret_key",
}


def _normalize_vividvr_flowcut_payload(body: Any) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise ValueError("request body must be a JSON object")
    payload = _normalize_aliases(body, _FLOWCUT_FIELD_ALIASES)
    minio_config = payload.get("minio_config")
    if isinstance(minio_config, dict):
        payload["minio_config"] = _normalize_aliases(
            minio_config, _FLOWCUT_MINIO_FIELD_ALIASES
        )
    return payload


async def _store_failed_flowcut_submission(
    request_id: str, reason: str, *, body: dict[str, Any] | None = None
) -> None:
    job = {
        "id": request_id,
        "object": "video",
        "model": "VividVR",
        "status": "failed",
        "progress": 0,
        "created_at": int(time.time()),
        "reason": reason,
        "error": {"message": reason},
        "callback_url": (body or {}).get("callback_url") or (body or {}).get("callbackUrl"),
        "callback_status": None,
        "callback_error": None,
        "output_object_key": (body or {}).get("output_object_key") or (body or {}).get("outputObjectKey"),
    }
    await VIDEO_STORE.upsert(request_id, job)
```

- [ ] **步骤 4：运行测试验证通过**

运行：`PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest python/sglang/multimodal_gen/test/unit/test_flowcut_protocol.py python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py -k "minio_aliases or timeout_less_than_minus_one or persisted_as_failed" -q`

预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_protocol.py \
        python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_api.py \
        python/sglang/multimodal_gen/test/unit/test_flowcut_protocol.py \
        python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py
git commit -m "feat: align vividvr flowcut request submission semantics"
```

### 任务 2：对齐对象存储契约与输出目标解析

**文件：**
- 修改：`python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_api.py`
- 修改：`python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_storage.py`
- 修改：`python/sglang/multimodal_gen/runtime/entrypoints/openai/flowcut.py`
- 修改：`python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py`
- 创建：`python/sglang/multimodal_gen/test/unit/test_vividvr_flowcut_storage.py`

- [ ] **步骤 1：编写失败的对象存储测试**

```python
def test_upload_result_uses_explicit_output_object_key(tmp_path):
    storage = VividVRFlowCutStorage(base_dir=tmp_path, request_id="task-1")
    local_path = Path(storage.output_file_path("task-1.mp4"))
    local_path.write_bytes(b"video")

    with patch(
        "sglang.multimodal_gen.runtime.entrypoints.openai.vividvr_flowcut_storage.upload_to_flowcut_minio",
        new=AsyncMock(return_value="http://minio/bucket/custom/key.mp4"),
    ) as mock_upload:
        result = asyncio.run(
            storage.upload_result(
                local_path,
                minio_config=FlowCutMinIOConfig(
                    endpoint="minio.example.com:9000",
                    bucket_name="bucket-a",
                    access_key="ak",
                    secret_key="sk",
                ),
                object_key="custom/key.mp4",
                bucket_name="bucket-b",
            )
        )

    assert result == "http://minio/bucket/custom/key.mp4"
    assert mock_upload.await_args.kwargs["object_key"] == "custom/key.mp4"
    assert mock_upload.await_args.kwargs["bucket_name"] == "bucket-b"


@pytest.mark.asyncio
async def test_flowcut_request_persists_default_output_object_key(client):
    response = await client.post(
        "/v1/videos/repairs/flowcut",
        json={
            "taskId": "task-1",
            "callbackUrl": "http://127.0.0.1:9000/callback",
            "video_input_path": "/tmp/in.mp4",
            "minioConfig": {
                "endpoint": "minio.example.com:9000",
                "bucket_name": "bucket-a",
                "access_key": "ak",
                "secret_key": "sk",
            },
        },
    )

    assert response.json()["code"] == 0
    job = await VIDEO_STORE.get("task-1")
    assert job["output_object_key"].endswith("_task-1.mp4")
```

- [ ] **步骤 2：运行测试验证失败**

运行：`PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest python/sglang/multimodal_gen/test/unit/test_vividvr_flowcut_storage.py python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py -k "output_object_key or bucket_name or default_output_object_key" -q`

预期：FAIL，当前上传逻辑写死 `outputs/{request_id}.mp4`，并忽略 `output_bucket`。

- [ ] **步骤 3：实现最少对象目标解析与上传透传**

```python
def _resolve_flowcut_output_object_key(req: FlowCutVideoRepairRequest, request_id: str) -> str | None:
    if req.minio_config is None:
        return None
    if req.output_object_key:
        return req.output_object_key.lstrip("/")
    return default_video_repair_output_object_key(request_id, extension=".mp4")


async def upload_result(
    self,
    local_path: str | Path,
    minio_config: FlowCutMinIOConfig | None,
    *,
    object_key: str | None = None,
    bucket_name: str | None = None,
) -> str:
    if minio_config is None:
        return str(local_path)

    result_url = await upload_to_flowcut_minio(
        local_path=str(Path(local_path).resolve()),
        object_key=object_key or f"outputs/{self.request_id}.mp4",
        bucket_name=bucket_name,
        config=minio_config,
    )
    Path(local_path).unlink()
    return result_url
```

- [ ] **步骤 4：运行测试验证通过**

运行：`PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest python/sglang/multimodal_gen/test/unit/test_vividvr_flowcut_storage.py python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py -k "output_object_key or bucket_name or default_output_object_key" -q`

预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_api.py \
        python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_storage.py \
        python/sglang/multimodal_gen/runtime/entrypoints/openai/flowcut.py \
        python/sglang/multimodal_gen/test/unit/test_vividvr_flowcut_storage.py \
        python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py
git commit -m "feat: align vividvr flowcut object storage semantics"
```

### 任务 3：对齐输入下载与 request workdir 生命周期

**文件：**
- 修改：`python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_api.py`
- 修改：`python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_storage.py`
- 创建：`python/sglang/multimodal_gen/test/unit/test_vividvr_flowcut_storage.py`
- 修改：`python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py`

- [ ] **步骤 1：编写失败的清理策略测试**

```python
def test_cleanup_temp_workdir_after_uploaded_result(tmp_path):
    storage = VividVRFlowCutStorage(base_dir=tmp_path, request_id="task-1")
    inputs_dir = storage.inputs_dir
    inputs_dir.mkdir(parents=True, exist_ok=True)
    (inputs_dir / "input.mp4").write_bytes(b"in")

    storage.cleanup_if_transient(result_externalized=True, base_dir_is_temp=True)

    assert not storage.workdir.exists()


def test_keep_temp_workdir_when_result_is_local_only(tmp_path):
    storage = VividVRFlowCutStorage(base_dir=tmp_path, request_id="task-2")
    (storage.inputs_dir / "input.mp4").write_bytes(b"in")

    storage.cleanup_if_transient(result_externalized=False, base_dir_is_temp=True)

    assert storage.workdir.exists()
```

- [ ] **步骤 2：运行测试验证失败**

运行：`PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest python/sglang/multimodal_gen/test/unit/test_vividvr_flowcut_storage.py -k "cleanup_temp_workdir or keep_temp_workdir" -q`

预期：FAIL，当前 storage 只有 `cleanup()`，没有按结果是否外部化区分清理行为。

- [ ] **步骤 3：实现最少生命周期策略**

```python
def cleanup_if_transient(
    self,
    *,
    result_externalized: bool,
    base_dir_is_temp: bool,
) -> None:
    if not base_dir_is_temp:
        return
    if not result_externalized:
        return
    shutil.rmtree(self.workdir, ignore_errors=True)


result_externalized = bool(minio_config is not None or output_is_persistent)
storage.cleanup_if_transient(
    result_externalized=result_externalized,
    base_dir_is_temp=base_dir_is_temp,
)
```

- [ ] **步骤 4：运行测试验证通过**

运行：`PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest python/sglang/multimodal_gen/test/unit/test_vividvr_flowcut_storage.py -k "cleanup_temp_workdir or keep_temp_workdir" -q`

预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_api.py \
        python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_storage.py \
        python/sglang/multimodal_gen/test/unit/test_vividvr_flowcut_storage.py \
        python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py
git commit -m "feat: align vividvr flowcut request lifecycle cleanup"
```

### 任务 4：对齐 callback bookkeeping 与 timeout 失败语义

**文件：**
- 修改：`python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_api.py`
- 修改：`python/sglang/multimodal_gen/runtime/entrypoints/openai/flowcut.py`
- 修改：`python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py`

- [ ] **步骤 1：编写失败的 callback / timeout 测试**

```python
@pytest.mark.asyncio
async def test_flowcut_callback_status_is_recorded_after_success(client):
    response = await client.post("/v1/videos/repairs/flowcut", json=VALID_FLOWCUT_BODY)
    assert response.json()["code"] == 0

    await wait_for_background_tasks()
    job = await VIDEO_STORE.get("task-1")
    assert job["callback_status"] == "succeeded"
    assert job["callback_error"] is None
    assert job["callback_attempts"] == 1


@pytest.mark.asyncio
async def test_flowcut_timeout_reason_matches_shared_contract(client):
    response = await client.post(
        "/v1/videos/repairs/flowcut",
        json={**VALID_FLOWCUT_BODY, "timeout": 1},
    )
    assert response.json()["code"] == 0

    await wait_for_background_tasks()
    job = await VIDEO_STORE.get("task-1")
    assert job["status"] == "failed"
    assert job["reason"] == TASK_TIMEOUT_MESSAGE
```

- [ ] **步骤 2：运行测试验证失败**

运行：`PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py -k "callback_status_is_recorded or timeout_reason_matches_shared_contract" -q`

预期：FAIL，当前 FlowCut callback 不会回写 `callback_status/callback_error/callback_attempts`，timeout reason 也不是统一文案。

- [ ] **步骤 3：实现最少 callback bookkeeping 与 timeout 收口**

```python
async def _post_stage_callback(task_id: str, callback_url: str, payload: dict[str, Any]) -> None:
    attempts = 0
    try:
        attempts = await post_flowcut_callback(
            callback_url, payload, timeout=5.0, max_retries=1, return_attempts=True
        )
    except Exception as e:
        await VIDEO_STORE.update_fields(
            task_id,
            {
                "callback_status": "failed",
                "callback_error": str(e),
                "callback_attempts": max(1, attempts),
                "callback_completed_at": int(time.time()),
            },
        )
        raise
    else:
        await VIDEO_STORE.update_fields(
            task_id,
            {
                "callback_status": "succeeded",
                "callback_error": None,
                "callback_attempts": attempts,
                "callback_completed_at": int(time.time()),
            },
        )


except asyncio.TimeoutError:
    reason = TASK_TIMEOUT_MESSAGE
    await VIDEO_STORE.update_fields(
        job_id,
        {"status": "failed", "error": {"message": reason}, "reason": reason},
    )
```

- [ ] **步骤 4：运行测试验证通过**

运行：`PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py -k "callback_status_is_recorded or timeout_reason_matches_shared_contract" -q`

预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_api.py \
        python/sglang/multimodal_gen/runtime/entrypoints/openai/flowcut.py \
        python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py
git commit -m "feat: align vividvr flowcut callback and timeout semantics"
```

### 任务 5：收口回归测试与文档

**文件：**
- 修改：`python/sglang/multimodal_gen/test/unit/test_flowcut_protocol.py`
- 修改：`python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py`
- 创建：`python/sglang/multimodal_gen/test/unit/test_vividvr_flowcut_storage.py`
- 修改：`docs_xzh/run_command/mock_test.md`
- 创建：`docs_xzh/hand_over/vividvr_flowcut_service_semantics_alignment_handover_20260626.md`

- [ ] **步骤 1：补齐回归测试与文档断言**

```python
def test_vividvr_flowcut_submit_response_uses_numeric_codes_only():
    assert VividVRFlowCutSubmitResponse(code=0).model_dump() == {
        "code": 0,
        "message": "ok",
    }


@pytest.mark.asyncio
async def test_vivid_pipeline_rejects_shared_video_repairs_endpoint(client):
    response = await client.post("/v1/videos/repairs", json={"taskId": "task-1"})
    assert response.status_code == 400
    assert "must use /v1/videos/repairs/flowcut" in response.json()["detail"]
```

文档补充片段：

```md
- `outputObjectKey`：当配置 `minioConfig` 时可选；未提供时服务端自动生成 `YYYY/MM/DD/HHMMSS_<taskId>.mp4`
- `timeout=0`：按 `300` 处理
- 非法请求且包含 `taskId` 时：会创建 failed task 记录，并可能触发失败 callback
```

- [ ] **步骤 2：运行完整单测回归**

运行：`PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest python/sglang/multimodal_gen/test/unit/test_flowcut_protocol.py python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py python/sglang/multimodal_gen/test/unit/test_vividvr_flowcut_storage.py -q`

预期：PASS。

- [ ] **步骤 3：人工检查文档与 handover 一致性**

运行：`rg -n "outputObjectKey|timeout=0|failed task|callback_status" docs_xzh/run_command/mock_test.md docs_xzh/hand_over/vividvr_flowcut_service_semantics_alignment_handover_20260626.md`

预期：命中文档中的新契约说明，且没有旧的“固定 outputs/{request_id}.mp4”描述。

- [ ] **步骤 4：运行相关 API 回归**

运行：`PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest python/sglang/multimodal_gen/test/unit/test_video_api_vividvr.py -q`

预期：PASS，确认共享 `/v1/videos/repairs` 仍拒绝 Vivid，请求端必须走独立 FlowCut 路由。

- [ ] **步骤 5：Commit**

```bash
git add python/sglang/multimodal_gen/test/unit/test_flowcut_protocol.py \
        python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py \
        python/sglang/multimodal_gen/test/unit/test_vividvr_flowcut_storage.py \
        docs_xzh/run_command/mock_test.md \
        docs_xzh/hand_over/vividvr_flowcut_service_semantics_alignment_handover_20260626.md
git commit -m "docs: document vividvr flowcut service semantics alignment"
```

## 自检

- **规格覆盖度：**
  - 请求契约：任务 1
  - 输出对象存储：任务 2
  - 清理策略：任务 3
  - callback / timeout：任务 4
  - 文档与回归：任务 5
- **占位符扫描：**
  - 未使用 `TODO`、`待定`、`后续实现`、`类似任务 N`。
- **类型一致性：**
  - 统一使用 `VividVRFlowCutRequest`、`FlowCutMinIOConfig`、`output_object_key`、`callback_status`、`TASK_TIMEOUT_MESSAGE`。

计划已完成并保存到 `.codex/plans/2026-06-26-vividvr-flowcut-service-semantics-alignment-plan.md`。两种执行方式：

**1. 子代理驱动（推荐）** - 每个任务调度一个新的子代理，任务间进行审查，快速迭代

**2. 内联执行** - 在当前会话中使用 executing-plans 执行任务，批量执行并设有检查点

选哪种方式？
