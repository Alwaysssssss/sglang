# Vivid-VR FlowCut 输入清理、输出格式与取消语义对齐实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 在保留 Vivid-VR FlowCut 独立入口的前提下，对齐 `share-tyx` / `online_videoedit` 的三项请求端语义：输入缓存删除、输出扩展名继承输入、`DELETE` 取消任务。

**架构：** 保留 [vividvr_flowcut_api.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_api.py)、[vividvr_flowcut_storage.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_storage.py)、[vividvr_flowcut_progress.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_progress.py) 的独立服务链，不并回 `video_api.py`。服务层新增 FlowCut 自己的输入生命周期判定、输出命名解析和取消任务骨架，并把取消信号传入 Vivid-VR runtime 的长循环，确保 `DELETE` 能真实停止推理。

**技术栈：** FastAPI、asyncio、Pydantic、tmux、pytest、Vivid-VR pipeline、MinIO/mock S3

---

## 文件结构

**修改：**
- `python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_api.py`
  FlowCut 入口、任务注册、取消接口、输入/输出语义判定。
- `python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_storage.py`
  输入视频落盘命名、输出扩展名解析辅助、cleanup 边界。
- `python/sglang/multimodal_gen/runtime/entrypoints/openai/video_repair_shared.py`
  Vivid-VR caption sidecar 的 workdir 归属与显式 caption 生命周期约束。
- `python/sglang/multimodal_gen/configs/sample/sampling_params.py`
  为通用 sampling params 增加 `request_cancel_path` / `request_timeout_deadline` 字段，供 FlowCut 和 runtime 传递取消信号。
- `python/sglang/multimodal_gen/configs/sample/vividvr.py`
  确保 Vivid-VR sampling params 继承取消相关字段，不覆盖现有校验。
- `python/sglang/multimodal_gen/runtime/request_timeout.py`
  新增与 `origin/online_videoedit` 对齐的取消/超时检查 helper。
- `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py`
  在 denoise 主循环中接入取消检查，确保 `DELETE` 后能真实停下。
- `python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py`
  覆盖 FlowCut 取消接口、任务状态、cleanup 语义、输出命名。
- `python/sglang/multimodal_gen/test/unit/test_vividvr_flowcut_storage.py`
  覆盖输入扩展名保留、输出扩展名生成和 cleanup 边界。
- `python/sglang/multimodal_gen/test/unit/test_flowcut_protocol.py`
  如需补充请求字段行为断言，仅收口与本次三项语义相关的测试。
- `docs_xzh/hand_over/vividvr_flowcut_input_format_cancel_alignment_handover_20260626.md`
  记录对齐结果、服务启动命令、验收路径和已知边界。

**不修改：**
- `python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py`
  仅作为对齐参照，不把 FlowCut 重新并入通用 `video_edit` 主入口。

---

### 任务 1：对齐输入缓存生命周期与 caption 清理边界

**文件：**
- 修改：`python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_api.py`
- 修改：`python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_storage.py`
- 修改：`python/sglang/multimodal_gen/runtime/entrypoints/openai/video_repair_shared.py`
- 测试：`python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py`

- [ ] **步骤 1：编写失败的测试**

```python
def test_flowcut_temp_input_cache_is_cleaned_after_externalized_result(...):
    server_args = _make_server_args(input_save_path=None, output_path=None)
    req = _make_flowcut_request(
        task_id="cleanup-temp",
        video_url="https://example.com/input.mov",
        minio_config=_make_minio(),
    )
    ...
    assert not storage.workdir.exists()


def test_flowcut_persistent_input_cache_is_kept_when_input_save_path_is_configured(...):
    server_args = _make_server_args(input_save_path=str(tmp_path / "uploads"))
    ...
    assert storage.workdir.exists()


def test_explicit_caption_file_path_is_never_deleted(...):
    caption_file = tmp_path / "captions" / "manual.txt"
    caption_file.write_text("clip caption\n")
    ...
    assert caption_file.exists()


def test_bridge_caption_under_temp_workdir_is_deleted_with_request_workdir(...):
    server_args = _make_server_args(
        input_save_path=None,
        output_path=None,
        vividvr_caption_bridge=True,
        vividvr_caption_work_dir=None,
    )
    ...
    assert not temp_caption_path.exists()
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py \
  -k "temp_input_cache or persistent_input_cache or explicit_caption_file_path or bridge_caption_under_temp_workdir" -q
```

预期：FAIL。当前实现会因为 `_flowcut_work_base_dir()` 仍由 `input_save_path or output_path` 共同决定，且 caption workdir 没有显式 cleanup 语义而失败。

- [ ] **步骤 3：编写最少实现代码**

在 `vividvr_flowcut_api.py` 中把“输入缓存是否临时”改成只由 `input_save_path` 决定，并把 `output_path` 仅用于结果落点：

```python
def _flowcut_input_base_dir(server_args) -> tuple[str, bool]:
    input_save_path = getattr(server_args, "input_save_path", None)
    if input_save_path:
        return str(input_save_path), False
    return tempfile.mkdtemp(prefix="sglang_vividvr_flowcut_input_"), True
```

在 `video_repair_shared.py` 中把 bridge caption 的默认 workdir 绑定到 request-local `output_dir/caption_sidecars`，并依赖 request workdir cleanup：

```python
if req.caption_file_path:
    return req.caption_file_path

work_dir = getattr(server_args, "vividvr_caption_work_dir", None)
if not work_dir:
    work_dir = os.path.join(output_dir, "caption_sidecars")
```

在 `vividvr_flowcut_api.py` 的异步 finally 中，仅当输入 base dir 为临时目录且结果已外部化时清理 request workdir。

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py \
  -k "temp_input_cache or persistent_input_cache or explicit_caption_file_path or bridge_caption_under_temp_workdir" -q
```

预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add \
  python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_api.py \
  python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_storage.py \
  python/sglang/multimodal_gen/runtime/entrypoints/openai/video_repair_shared.py \
  python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py
git commit -m "feat: align flowcut input cache cleanup semantics"
```

---

### 任务 2：对齐输出文件扩展名继承输入视频格式

**文件：**
- 修改：`python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_storage.py`
- 修改：`python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_api.py`
- 修改：`python/sglang/multimodal_gen/runtime/entrypoints/openai/video_repair_shared.py`
- 测试：`python/sglang/multimodal_gen/test/unit/test_vividvr_flowcut_storage.py`
- 测试：`python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py`

- [ ] **步骤 1：编写失败的测试**

```python
def test_materialize_video_preserves_source_extension_for_url_download(...):
    path = await storage.materialize_video("https://example.com/source.mov")
    assert path.endswith(".mov")


def test_flowcut_default_output_file_name_inherits_input_extension(...):
    output_file_name = vividvr_flowcut_api._flowcut_output_file_name(
        req_without_output_path,
        "job-1",
        source_video_path="/tmp/input.mov",
    )
    assert output_file_name == "job-1.mov"


def test_flowcut_explicit_output_path_base_name_keeps_input_extension(...):
    output_file_name = vividvr_flowcut_api._flowcut_output_file_name(
        req_with_output_path("/tmp/result.mp4"),
        "job-1",
        source_video_path="/tmp/input.mov",
    )
    assert output_file_name == "result.mov"


def test_flowcut_default_output_object_key_uses_input_extension(...):
    object_key = vividvr_flowcut_api._resolve_flowcut_output_object_key(
        req_with_minio(),
        "job-1",
        output_file_name="job-1.mov",
    )
    assert object_key.endswith(".mov")
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_vividvr_flowcut_storage.py \
  python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py \
  -k "preserves_source_extension or inherits_input_extension or keeps_input_extension or uses_input_extension" -q
```

预期：FAIL。当前默认路径仍以 `.mp4` 为中心。

- [ ] **步骤 3：编写最少实现代码**

在 `vividvr_flowcut_storage.py` 中保留下载源扩展名：

```python
@staticmethod
def _infer_source_extension(source: str) -> str | None:
    parsed = urlparse(source)
    candidate = parsed.path if parsed.scheme in {"http", "https"} else source
    suffix = Path(candidate).suffix.lower()
    return suffix or None
```

在 `materialize_video(...)` 中根据源扩展名改写目标文件名：

```python
source_ext = self._infer_source_extension(source_text)
target_path = self.inputs_dir / self._normalize_video_filename(
    filename_hint,
    source_ext=source_ext,
)
```

在 `vividvr_flowcut_api.py` 中新增 FlowCut 自己的输出命名解析 helper，不复用当前只支持 `.mp4` 的 `split_output_path()`：

```python
def _flowcut_split_output_path(
    output_path: str | None,
    job_id: str,
    server_output_path: str | None,
    *,
    source_video_path: str | None,
) -> tuple[str | None, str]:
    source_ext = _source_video_extension(source_video_path) or ".mp4"
    if output_path and Path(output_path).suffix.lower() in VIDEO_OUTPUT_EXTENSIONS:
        return str(Path(output_path).resolve().parent), f"{Path(output_path).stem}{source_ext}"
    return output_path or server_output_path, f"{job_id}{source_ext}"
```

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_vividvr_flowcut_storage.py \
  python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py \
  -k "preserves_source_extension or inherits_input_extension or keeps_input_extension or uses_input_extension" -q
```

预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add \
  python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_storage.py \
  python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_api.py \
  python/sglang/multimodal_gen/runtime/entrypoints/openai/video_repair_shared.py \
  python/sglang/multimodal_gen/test/unit/test_vividvr_flowcut_storage.py \
  python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py
git commit -m "feat: align flowcut output extension semantics"
```

---

### 任务 3：对齐 `online_videoedit` 的取消请求语义

**文件：**
- 修改：`python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_api.py`
- 创建：`python/sglang/multimodal_gen/runtime/request_timeout.py`
- 修改：`python/sglang/multimodal_gen/configs/sample/sampling_params.py`
- 修改：`python/sglang/multimodal_gen/configs/sample/vividvr.py`
- 修改：`python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py`
- 测试：`python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py`

- [ ] **步骤 1：编写失败的测试**

```python
@pytest.mark.asyncio
async def test_delete_flowcut_job_marks_failed_and_posts_failed_callback(...):
    job_id = "cancel-me"
    await VIDEO_STORE.upsert(
        job_id,
        {
            "id": job_id,
            "status": "running",
            "progress": 37,
            "callback_url": "http://callback.local/test",
            "request_cancel_path": str(tmp_path / "cancel.marker"),
        },
    )
    response = await vividvr_flowcut_api.delete_vividvr_flowcut_video(job_id)
    assert response.status == "failed"
    assert response.reason == vividvr_flowcut_api.TASK_TIMEOUT_MESSAGE
    assert Path(response.request_cancel_path).exists()


@pytest.mark.asyncio
async def test_dispatch_flowcut_job_handles_cancelled_error_as_timeout(...):
    ...
    task.cancel()
    await registered_task
    job = await VIDEO_STORE.get(job_id)
    assert job["status"] == "failed"
    assert job["reason"] == vividvr_flowcut_api.TASK_TIMEOUT_MESSAGE


def test_check_request_timeout_raises_when_cancel_marker_exists(tmp_path):
    cancel_path = tmp_path / "job.cancel"
    cancel_path.write_text("1")
    req = SimpleNamespace(request_cancel_path=str(cancel_path))
    with pytest.raises(TaskTimeoutError, match="Request timed out."):
        check_request_timeout(req)
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py \
  -k "delete_flowcut_job_marks_failed or handles_cancelled_error_as_timeout or check_request_timeout_raises" -q
```

预期：FAIL。当前 FlowCut 没有取消接口、没有任务注册表、也没有 request-timeout helper。

- [ ] **步骤 3：编写最少实现代码**

在 `runtime/request_timeout.py` 中引入与 `online_videoedit` 对齐的 helper：

```python
TASK_TIMEOUT_MESSAGE = "Request timed out."


class TaskTimeoutError(TimeoutError):
    pass


def check_request_timeout(request: Any) -> None:
    sampling_params = getattr(request, "sampling_params", None)
    cancel_path = getattr(request, "request_cancel_path", None)
    if cancel_path is None:
        cancel_path = getattr(sampling_params, "request_cancel_path", None)
    if cancel_path and os.path.exists(cancel_path):
        raise TaskTimeoutError(TASK_TIMEOUT_MESSAGE)
```

在 `vividvr_flowcut_api.py` 中新增 FlowCut 自己的取消骨架：

```python
_FLOWCUT_TASKS: dict[str, asyncio.Task] = {}
_FLOWCUT_TASKS_LOCK = asyncio.Lock()


async def _create_registered_flowcut_task(job_id: str, coro) -> asyncio.Task:
    task = asyncio.create_task(coro)
    async with _FLOWCUT_TASKS_LOCK:
        _FLOWCUT_TASKS[job_id] = task
    return task
```

新增删除接口并完全对齐 `online_videoedit` 语义：

```python
@router.delete("/repairs/flowcut/{video_id}")
async def delete_vividvr_flowcut_video(video_id: str = Path(...)):
    job = await VIDEO_STORE.get(video_id)
    if not job:
        raise HTTPException(status_code=404, detail="Video not found")
    if job.get("status") not in {"completed", "failed"}:
        job = await _mark_flowcut_job_cancelled(video_id, job)
    return job
```

在 `vividvr_pipeline.py` 的 denoise 循环里加入取消检查：

```python
from sglang.multimodal_gen.runtime.request_timeout import check_request_timeout

with self.denoising_stage.progress_bar(total=len(params.runtime_timesteps)) as progress_bar:
    for timestep_index, _ in enumerate(params.runtime_timesteps):
        check_request_timeout(batch)
        ...
```

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py -k "delete_flowcut_job_marks_failed or handles_cancelled_error_as_timeout" -q
```

运行：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py \
  -k "check_request_timeout_raises" -q
```

预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add \
  python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_api.py \
  python/sglang/multimodal_gen/runtime/request_timeout.py \
  python/sglang/multimodal_gen/configs/sample/sampling_params.py \
  python/sglang/multimodal_gen/configs/sample/vividvr.py \
  python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py \
  python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py
git commit -m "feat: add flowcut cancellation semantics"
```

---

### 任务 4：回归测试、双卡服务验收与 handover 收口

**文件：**
- 修改：`python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py`
- 修改：`python/sglang/multimodal_gen/test/unit/test_vividvr_flowcut_storage.py`
- 创建：`docs_xzh/hand_over/vividvr_flowcut_input_format_cancel_alignment_handover_20260626.md`

- [ ] **步骤 1：运行单元测试回归**

运行：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_flowcut_protocol.py \
  python/sglang/multimodal_gen/test/unit/test_vividvr_flowcut_storage.py \
  python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py \
  python/sglang/multimodal_gen/test/unit/test_video_api_vividvr.py -q
```

预期：PASS。

- [ ] **步骤 2：在 tmux 中启动双卡 FlowCut 服务**

运行：

```bash
tmux new-session -d -s vividvr_flowcut_align_dual \
  'cd /home/zhiheng/sglang && export PYTHONPATH=python && \
   export SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE=eager_global && \
   export SGLANG_VIVIDVR_CONNECTOR_CONTROL_POOL_SIZE=1 && \
   CUDA_VISIBLE_DEVICES=0,1 /home/zhiheng/sglang/.venv/bin/python -m sglang.launch_server \
   --model-path vividvr \
   --attention-backend fa \
   --tp-size 1 \
   --sp-degree 2 \
   --ulysses-degree 2 \
   --ring-degree 1 \
   --enable-torch-compile \
   --host 127.0.0.1 \
   --port 31240 \
   --master-port 30240 \
   --scheduler-port 56240 \
   --strict-ports 2>&1 | tee Vivid_Acceptance/logs/vividvr_flowcut_align_dual_$(date -u +%Y%m%dT%H%M%SZ).log'
```

预期：服务健康检查通过。只读查看：

```bash
tmux attach -r -t vividvr_flowcut_align_dual
```

- [ ] **步骤 3：验收输入扩展名与 cleanup**

运行 mock S3 + callback receiver，并发送一个 `.mov` 输入请求：

```bash
curl -X POST http://127.0.0.1:31240/v1/videos/repairs/flowcut \
  -H 'Content-Type: application/json' \
  -d '{
    "taskId": "flowcut-align-mov-e2e",
    "callbackUrl": "http://127.0.0.1:18080/callback",
    "videoUrl": "http://127.0.0.1:19090/input.mov",
    "referenceVideoPath": "/home/zhiheng/Vivid-VR/result/720p_long_up1_result_vivid_ori_20step/videos/test_video_long_960x720_130f.mp4",
    "captionFilePath": "/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.txt",
    "minioConfig": {
      "endpoint": "http://127.0.0.1:4566",
      "bucketName": "flowcut",
      "accessKey": "test",
      "secretKey": "testtest"
    }
  }'
```

预期：
- FlowCut 输出对象 key 扩展名为 `.mov`
- callback 成功
- 若 `input_save_path=None`，任务结束后 request workdir 已清理
- 显式 `captionFilePath` 原文件仍存在

- [ ] **步骤 4：验收取消语义**

运行：

```bash
curl -X DELETE http://127.0.0.1:31240/v1/videos/repairs/flowcut/flowcut-align-cancel
```

随后轮询：

```bash
curl http://127.0.0.1:31240/v1/videos/flowcut/flowcut-align-cancel/progress
```

预期：
- 状态为 `failed`
- `reason` 为 `Request timed out.`
- 失败 callback 已发送
- GPU 任务不再继续跑完

- [ ] **步骤 5：写 handover 并 Commit**

handover 文档至少包含：

```md
# Vivid-VR FlowCut 输入/格式/取消语义对齐交接

- 输入缓存删除语义与 share-tyx 对齐结果
- 输出扩展名继承输入格式的最终行为
- DELETE 取消语义与 online_videoedit 对齐结果
- mock S3 / callback / 双卡验收命令与产物路径
- 已知边界：显式 caption_file_path 不删除
```

提交：

```bash
git add \
  python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py \
  python/sglang/multimodal_gen/test/unit/test_vividvr_flowcut_storage.py \
  docs_xzh/hand_over/vividvr_flowcut_input_format_cancel_alignment_handover_20260626.md
git commit -m "docs: hand over flowcut input format cancel alignment"
```

---

## 自检

- 本计划覆盖了 3 项明确需求：
  - 输入缓存删除语义
  - 输出扩展名继承输入
  - `online_videoedit` 风格取消任务
- 没有把 FlowCut 并回 `video_api.py`，保持了 Vivid-VR 独立入口边界。
- 没有把“取消”设计成新状态 `cancelled`，而是按你的要求完全对齐为 `failed + Request timed out.`。
- 显式 `caption_file_path` 不删除、bridge 自动生成 caption 随临时 workdir 清理，这个边界已在任务 1 中写死。
