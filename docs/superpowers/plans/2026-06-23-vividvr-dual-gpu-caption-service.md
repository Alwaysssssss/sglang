# VividVR Dual-GPU Caption Service 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 在不改变现有 caption bridge 输入输出契约的前提下，把 Vivid-VR caption sidecar 升级为同机双卡并行服务，并保证 `caption.txt` 对同一输入逐行逐字与当前单卡基线完全一致。

**架构：** 主服务继续只调用一个 `/v1/vividvr/captions`。sidecar 内部实现为一个 controller HTTP 进程加两个常驻单 worker 执行器，每个执行器固定绑定一张 GPU，并保留一份 CPU 常驻 caption 模型；请求到来时 controller 按 `clip_index` 把不同 temporal clip 分发给两个 worker，最后按 `clip_index` 顺序聚合并原子写回 `caption.txt`。任何并行失败都回退到当前已验收的串行单卡路径，不引入 TP/模型并行，不改主推理链。

**技术栈：** Python、FastAPI、Pydantic、`concurrent.futures.ProcessPoolExecutor`、PyTorch、pytest、tmux、现有 `caption_manifest` / `caption_bridge` / `video_api`、原版 `/home/zhiheng/Vivid-VR/.venv`。

---

## 范围与假设

- 这个计划基于本轮已确认方案，不再引入“同一个 clip 双卡模型并行”。
- 双卡加速来自“不同 clip 并行”，不是“同一个 clip 两卡共同生成”。
- `expected_caption_count` 继续表示 temporal clip 数，一行一个 clip caption。
- 主服务接口、`manifest.json` 结构和 `caption.txt` 消费方式保持不变。
- 当前 `video_api` 已有 `_VIDEOEDIT_SEMAPHORE=1`，本计划默认继续用现有请求串行化约束，不额外扩展新的全局 GPU 调度器。
- sidecar 运行环境仍是 `/home/zhiheng/Vivid-VR/.venv/bin/python`；所有单元测试仍在 `/home/zhiheng/sglang/.venv/bin/python` 下执行。

## 文件结构

- 创建 `python/sglang/multimodal_gen/runtime/vividvr/caption_sidecar_runtime.py`
  - 轻量 sidecar runtime helper。
  - 负责 clip 分发、结果排序、计时数据结构、worker 结果聚合。
  - 只依赖 stdlib / torch / 当前 `caption_manifest`，避免把重型主推理依赖带入原版环境。
- 修改 `python/sglang/multimodal_gen/tools/run_vividvr_caption_sidecar.py`
  - 保留 HTTP 入口和现有串行路径。
  - 新增双 worker 执行器、worker 初始化、并行调度、串行回退、扩展响应元数据、CLI 参数。
- 修改 `python/sglang/multimodal_gen/runtime/vividvr/caption_bridge.py`
  - 在保持旧契约兼容的前提下，解析 sidecar 返回的可选 `mode / worker_count / fallback_used / timing` 元数据。
- 修改 `python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py`
  - 日志输出 sidecar 模式、耗时、是否走回退，便于后续端到端定位。
- 创建 `python/sglang/multimodal_gen/tools/run_vividvr_caption_sidecar_benchmark.py`
  - 独立 benchmark 入口。
  - 负责构造 manifest、请求 sidecar、比对基线 caption 文件、落盘 JSON 指标。
- 创建 `python/sglang/multimodal_gen/test/unit/test_vividvr_caption_sidecar_runtime.py`
  - 覆盖 round-robin clip 分发、按 `clip_index` 排序聚合、worker timing 汇总。
- 修改 `python/sglang/multimodal_gen/test/unit/test_vividvr_caption_sidecar_tool.py`
  - 覆盖 sidecar HTTP 响应、双 worker 任务分发、失败回退、CLI 参数解析。
- 修改 `python/sglang/multimodal_gen/test/unit/test_vividvr_caption_bridge.py`
  - 覆盖可选元数据解析且不破坏旧响应兼容性。
- 创建 `python/sglang/multimodal_gen/test/unit/test_vividvr_caption_sidecar_benchmark.py`
  - 覆盖 benchmark CLI 参数、基线 caption 比对、指标 JSON 结构。
- 修改 `docs_xzh/run_command/vividvr_default_run_and_serve_commands.md`
  - 增加双 worker sidecar 启动命令和独立 benchmark 命令。
- 修改 `docs_xzh/run_vivid_benchmark.md`
  - 增加 caption sidecar 独立 benchmark 与验收说明。
- 创建 `docs_xzh/hand_over/vividvr_dual_gpu_caption_service_handover_20260623.md`
  - 记录双 worker 部署、基准结果、回退语义、tmux 验收命令和已知风险。

---

### 任务 1：建立双 worker runtime 契约与调度 helper

**文件：**
- 创建：`python/sglang/multimodal_gen/runtime/vividvr/caption_sidecar_runtime.py`
- 创建：`python/sglang/multimodal_gen/test/unit/test_vividvr_caption_sidecar_runtime.py`

- [ ] **步骤 1：先写 runtime 失败测试，锁定任务分发和排序语义**

修改文件：
- `python/sglang/multimodal_gen/test/unit/test_vividvr_caption_sidecar_runtime.py`

测试代码：

```python
from sglang.multimodal_gen.runtime.vividvr.caption_sidecar_runtime import (
    CaptionClipResult,
    CaptionWorkerBatchResult,
    CaptionRequestMetrics,
    assign_clip_indices_round_robin,
    merge_caption_results_in_clip_order,
)


def test_assign_clip_indices_round_robin_two_workers():
    assignments = assign_clip_indices_round_robin(
        clip_indices=[0, 1, 2, 3, 4],
        worker_count=2,
    )

    assert assignments == [[0, 2, 4], [1, 3]]


def test_merge_caption_results_preserves_clip_index_order():
    merged = merge_caption_results_in_clip_order(
        [
            CaptionWorkerBatchResult(
                worker_id=1,
                device="cuda:1",
                clips=[
                    CaptionClipResult(
                        clip_index=1,
                        caption="clip-1",
                        device="cuda:1",
                        decode_seconds=0.1,
                        inference_seconds=1.0,
                        total_seconds=1.1,
                    )
                ],
                total_seconds=1.1,
            ),
            CaptionWorkerBatchResult(
                worker_id=0,
                device="cuda:0",
                clips=[
                    CaptionClipResult(
                        clip_index=0,
                        caption="clip-0",
                        device="cuda:0",
                        decode_seconds=0.1,
                        inference_seconds=1.0,
                        total_seconds=1.1,
                    )
                ],
                total_seconds=1.1,
            ),
        ]
    )

    assert [item.clip_index for item in merged] == [0, 1]
    assert [item.caption for item in merged] == ["clip-0", "clip-1"]


def test_request_metrics_exposes_parallel_summary():
    metrics = CaptionRequestMetrics(
        mode="dual_worker",
        worker_count=2,
        fallback_used=False,
        read_seconds=0.4,
        write_seconds=0.01,
        total_seconds=10.0,
        worker_batches=[],
    )

    payload = metrics.to_response_dict()
    assert payload["mode"] == "dual_worker"
    assert payload["worker_count"] == 2
    assert payload["fallback_used"] is False
```

测试命令：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_vividvr_caption_sidecar_runtime.py -q
```

预期结果：失败，报错包含 `No module named 'sglang.multimodal_gen.runtime.vividvr.caption_sidecar_runtime'` 或缺少 helper / dataclass。

- [ ] **步骤 2：实现 runtime helper、数据结构和排序逻辑**

修改文件：
- `python/sglang/multimodal_gen/runtime/vividvr/caption_sidecar_runtime.py`

实现骨架：

```python
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class CaptionClipResult:
    clip_index: int
    caption: str
    device: str
    decode_seconds: float
    inference_seconds: float
    total_seconds: float


@dataclass(frozen=True)
class CaptionWorkerBatchResult:
    worker_id: int
    device: str
    clips: list[CaptionClipResult]
    total_seconds: float


@dataclass(frozen=True)
class CaptionRequestMetrics:
    mode: str
    worker_count: int
    fallback_used: bool
    read_seconds: float
    write_seconds: float
    total_seconds: float
    worker_batches: list[CaptionWorkerBatchResult] = field(default_factory=list)

    def to_response_dict(self) -> dict:
        return {
            "mode": self.mode,
            "worker_count": self.worker_count,
            "fallback_used": self.fallback_used,
            "read_seconds": self.read_seconds,
            "write_seconds": self.write_seconds,
            "total_seconds": self.total_seconds,
            "worker_batches": [asdict(batch) for batch in self.worker_batches],
        }


def assign_clip_indices_round_robin(
    *, clip_indices: list[int], worker_count: int
) -> list[list[int]]:
    assignments = [[] for _ in range(worker_count)]
    for idx, clip_index in enumerate(clip_indices):
        assignments[idx % worker_count].append(clip_index)
    return [assignment for assignment in assignments if assignment]


def merge_caption_results_in_clip_order(
    worker_batches: list[CaptionWorkerBatchResult],
) -> list[CaptionClipResult]:
    flattened = [clip for batch in worker_batches for clip in batch.clips]
    return sorted(flattened, key=lambda item: item.clip_index)
```

测试命令：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_vividvr_caption_sidecar_runtime.py -q
```

预期结果：`3 passed`。

- [ ] **步骤 3：提交 runtime 契约基线**

修改文件：
- `python/sglang/multimodal_gen/runtime/vividvr/caption_sidecar_runtime.py`
- `python/sglang/multimodal_gen/test/unit/test_vividvr_caption_sidecar_runtime.py`

验证命令：

```bash
git diff --cached --name-only
```

预期结果：只包含本任务的 runtime helper 与对应测试文件。

提交命令：

```bash
git add python/sglang/multimodal_gen/runtime/vividvr/caption_sidecar_runtime.py \
  python/sglang/multimodal_gen/test/unit/test_vividvr_caption_sidecar_runtime.py
git commit -m "feat(vividvr): add caption sidecar runtime helpers"
```

---

### 任务 2：把现有 sidecar 串行逻辑包装成可回退的 worker 执行单元

**文件：**
- 修改：`python/sglang/multimodal_gen/tools/run_vividvr_caption_sidecar.py`
- 修改：`python/sglang/multimodal_gen/test/unit/test_vividvr_caption_sidecar_tool.py`

- [ ] **步骤 1：先写失败测试，固定 worker job 与串行回退语义**

修改文件：
- `python/sglang/multimodal_gen/test/unit/test_vividvr_caption_sidecar_tool.py`

测试代码：

```python
def test_worker_job_returns_clip_results_for_requested_indices(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sglang.multimodal_gen.tools.run_vividvr_caption_sidecar._load_video_tensor",
        lambda video_path: (__import__("torch").zeros((9, 3, 64, 64)), 24.0),
    )

    class FakeCaptioner:
        def __init__(self):
            self.devices = []

        def to(self, device):
            self.devices.append(str(device))
            return self

        def __call__(self, video, fps=None):
            return f"frames={video.shape[0]}"

    state = FakeCaptioner()
    monkeypatch.setattr(
        "sglang.multimodal_gen.tools.run_vividvr_caption_sidecar._WORKER_STATE",
        {"captioner": state, "device": "cuda:0"},
    )

    results = run_vividvr_caption_sidecar._run_worker_caption_job(
        manifest_path=str(_write_test_manifest(tmp_path)),
        clip_indices=[1],
    )

    assert [item.clip_index for item in results.clips] == [1]
    assert results.device == "cuda:0"
    assert state.devices[0] == "cuda:0"
    assert state.devices[-1] == "cpu"


def test_parallel_failure_falls_back_to_serial_caption(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sglang.multimodal_gen.tools.run_vividvr_caption_sidecar._run_parallel_caption_jobs",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(
        "sglang.multimodal_gen.tools.run_vividvr_caption_sidecar._caption_manifest_serial",
        lambda *args, **kwargs: (
            ["serial-0", "serial-1"],
            {"mode": "serial", "fallback_used": True, "worker_count": 1},
        ),
    )

    captions, metrics = run_vividvr_caption_sidecar._caption_manifest_with_fallback(
        state=SimpleNamespace(),
        manifest=_read_test_manifest(tmp_path),
    )

    assert captions == ["serial-0", "serial-1"]
    assert metrics["fallback_used"] is True
```

测试命令：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_vividvr_caption_sidecar_tool.py \
  -k "worker_job or fallback" -q
```

预期结果：失败，报错包含 `_WORKER_STATE`、`_run_worker_caption_job` 或 `_caption_manifest_with_fallback` 未定义。

- [ ] **步骤 2：实现 worker job、串行 path 和显式回退入口**

修改文件：
- `python/sglang/multimodal_gen/tools/run_vividvr_caption_sidecar.py`

实现骨架：

```python
_WORKER_STATE: dict[str, object] | None = None


def _init_worker_state(vividvr_root: str, ckpt_path: str, device: str) -> None:
    global _WORKER_STATE
    _ensure_python_dev_headers_for_sidecar()
    _WORKER_STATE = {
        "captioner": _load_original_captioner(
            SimpleNamespace(
                vividvr_root=vividvr_root,
                cogvlm2_ckpt_path=ckpt_path,
            )
        ),
        "device": device,
    }


def _run_worker_caption_job(
    *,
    manifest_path: str,
    clip_indices: list[int],
) -> CaptionWorkerBatchResult:
    manifest = VividVRCaptionManifest.read_json(manifest_path)
    video, fps = _load_video_tensor(manifest.video_path)
    effective_fps = manifest.fps or fps
    captioner = _WORKER_STATE["captioner"]
    device = _WORKER_STATE["device"]
    captioner.to(device)
    try:
        clips = []
        for clip_index in clip_indices:
            clip = manifest.clips[clip_index]
            clip_video = _clip_tensor(
                video,
                start=clip.start_frame,
                end=clip.end_frame,
                padded_frames=clip.padded_num_frames,
            )
            caption = str(captioner(clip_video, fps=effective_fps)).strip()
            clips.append(
                CaptionClipResult(
                    clip_index=clip.clip_index,
                    caption=caption,
                    device=str(device),
                    decode_seconds=...,
                    inference_seconds=...,
                    total_seconds=...,
                )
            )
    finally:
        captioner.to(torch.device("cpu"))
    return CaptionWorkerBatchResult(...)


def _caption_manifest_serial(...):
    # 保留当前已验收串行逻辑；返回 (captions, metrics)


def _caption_manifest_with_fallback(...):
    try:
        return _run_parallel_caption_jobs(...)
    except Exception:
        return _caption_manifest_serial(...)
```

测试命令：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_vividvr_caption_sidecar_tool.py \
  -k "worker_job or fallback" -q
```

预期结果：对应测试通过；至少显示 `2 passed`。

- [ ] **步骤 3：提交 worker job 与回退基线**

修改文件：
- `python/sglang/multimodal_gen/tools/run_vividvr_caption_sidecar.py`
- `python/sglang/multimodal_gen/test/unit/test_vividvr_caption_sidecar_tool.py`

验证命令：

```bash
git diff --cached --name-only
```

预期结果：只包含 sidecar tool 与其测试文件。

提交命令：

```bash
git add python/sglang/multimodal_gen/tools/run_vividvr_caption_sidecar.py \
  python/sglang/multimodal_gen/test/unit/test_vividvr_caption_sidecar_tool.py
git commit -m "feat(vividvr): add caption worker job and serial fallback"
```

---

### 任务 3：实现双 worker 并行 controller、扩展 sidecar HTTP 响应

**文件：**
- 修改：`python/sglang/multimodal_gen/tools/run_vividvr_caption_sidecar.py`
- 修改：`python/sglang/multimodal_gen/test/unit/test_vividvr_caption_sidecar_tool.py`

- [ ] **步骤 1：先写失败测试，锁定双 worker 顺序、并行结果聚合和响应元数据**

修改文件：
- `python/sglang/multimodal_gen/test/unit/test_vividvr_caption_sidecar_tool.py`

测试代码：

```python
def test_sidecar_parallel_mode_writes_output_in_clip_index_order(tmp_path, monkeypatch):
    manifest_path = _write_test_manifest(tmp_path)
    output_path = tmp_path / "captions.txt"

    async def fake_parallel(*args, **kwargs):
        return (
            ["clip-0", "clip-1"],
            {
                "mode": "dual_worker",
                "worker_count": 2,
                "fallback_used": False,
                "total_seconds": 12.3,
                "worker_batches": [
                    {"worker_id": 1, "device": "cuda:1"},
                    {"worker_id": 0, "device": "cuda:0"},
                ],
            },
        )

    monkeypatch.setattr(
        "sglang.multimodal_gen.tools.run_vividvr_caption_sidecar._caption_manifest_with_fallback",
        lambda **kwargs: fake_parallel(),
    )

    response = run_vividvr_caption_sidecar._generate_caption_sidecar_output(
        state=SimpleNamespace(),
        manifest_path=str(manifest_path),
        output_caption_path=str(output_path),
        expected_caption_count=2,
    )

    assert output_path.read_text(encoding="utf-8").splitlines() == ["clip-0", "clip-1"]
    assert response.mode == "dual_worker"
    assert response.worker_count == 2
    assert response.fallback_used is False


def test_parallel_dispatch_assigns_even_and_odd_clips_to_different_workers():
    assignments = run_vividvr_caption_sidecar._build_parallel_assignments(
        clip_count=5,
        worker_devices=["cuda:0", "cuda:1"],
    )

    assert assignments == [
        ("cuda:0", [0, 2, 4]),
        ("cuda:1", [1, 3]),
    ]
```

测试命令：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_vividvr_caption_sidecar_tool.py \
  -k "parallel_mode or parallel_dispatch" -q
```

预期结果：失败，报错包含 `mode` / `worker_count` 字段缺失，或 `_build_parallel_assignments` / 双 worker 入口未定义。

- [ ] **步骤 2：实现双 worker controller、`ProcessPoolExecutor` 常驻 worker 和扩展响应**

修改文件：
- `python/sglang/multimodal_gen/tools/run_vividvr_caption_sidecar.py`

实现骨架：

```python
class CaptionSidecarResponse(BaseModel):
    caption_file_path: str
    caption_count: int
    manifest_path: str
    mode: str = "serial"
    worker_count: int = 1
    fallback_used: bool = False
    read_seconds: float | None = None
    write_seconds: float | None = None
    total_seconds: float | None = None
    worker_batches: list[dict] = Field(default_factory=list)


@dataclass
class CaptionSidecarState:
    vividvr_root: str
    cogvlm2_ckpt_path: str
    worker_devices: list[str]
    executors: list[ProcessPoolExecutor]
    allow_serial_fallback: bool = True


def _build_parallel_assignments(*, clip_count: int, worker_devices: list[str]):
    clip_indices = list(range(clip_count))
    grouped = assign_clip_indices_round_robin(
        clip_indices=clip_indices,
        worker_count=len(worker_devices),
    )
    return list(zip(worker_devices, grouped))


def _build_parallel_executors(args: argparse.Namespace) -> list[ProcessPoolExecutor]:
    ctx = multiprocessing.get_context("spawn")
    executors = []
    for device in args.worker_devices:
        executors.append(
            ProcessPoolExecutor(
                max_workers=1,
                mp_context=ctx,
                initializer=_init_worker_state,
                initargs=(args.vividvr_root, args.cogvlm2_ckpt_path, device),
            )
        )
    return executors


def _run_parallel_caption_jobs(...):
    loop = asyncio.new_event_loop()
    assignments = _build_parallel_assignments(...)
    futures = [
        loop.run_in_executor(
            state.executors[i],
            functools.partial(
                _run_worker_caption_job,
                manifest_path=manifest_path,
                clip_indices=clip_indices,
            ),
        )
        for i, (_device, clip_indices) in enumerate(assignments)
        if clip_indices
    ]
    worker_batches = loop.run_until_complete(asyncio.gather(*futures))
    ordered = merge_caption_results_in_clip_order(worker_batches)
    captions = [item.caption for item in ordered]
    metrics = CaptionRequestMetrics(...)
    return captions, metrics.to_response_dict()
```

测试命令：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_vividvr_caption_sidecar_tool.py -q
```

预期结果：现有 sidecar tool 测试和新增并行测试全部通过。

- [ ] **步骤 3：提交双 worker controller 主改动**

修改文件：
- `python/sglang/multimodal_gen/tools/run_vividvr_caption_sidecar.py`
- `python/sglang/multimodal_gen/test/unit/test_vividvr_caption_sidecar_tool.py`

验证命令：

```bash
git diff --cached --name-only
```

预期结果：只包含 sidecar controller 相关代码和测试。

提交命令：

```bash
git add python/sglang/multimodal_gen/tools/run_vividvr_caption_sidecar.py \
  python/sglang/multimodal_gen/test/unit/test_vividvr_caption_sidecar_tool.py
git commit -m "feat(vividvr): parallelize caption sidecar across two gpus"
```

---

### 任务 4：扩展 bridge 结果、主服务日志与独立 benchmark 工具

**文件：**
- 修改：`python/sglang/multimodal_gen/runtime/vividvr/caption_bridge.py`
- 修改：`python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py`
- 创建：`python/sglang/multimodal_gen/tools/run_vividvr_caption_sidecar_benchmark.py`
- 修改：`python/sglang/multimodal_gen/test/unit/test_vividvr_caption_bridge.py`
- 创建：`python/sglang/multimodal_gen/test/unit/test_vividvr_caption_sidecar_benchmark.py`

- [ ] **步骤 1：先写失败测试，锁定 bridge 兼容性和 benchmark 指标结构**

修改文件：
- `python/sglang/multimodal_gen/test/unit/test_vividvr_caption_bridge.py`
- `python/sglang/multimodal_gen/test/unit/test_vividvr_caption_sidecar_benchmark.py`

测试代码：

```python
def test_request_caption_sidecar_preserves_optional_metrics(monkeypatch, tmp_path):
    output = tmp_path / "caption.txt"
    output.write_text("caption 0\n", encoding="utf-8")

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "caption_file_path": str(output),
                "caption_count": 1,
                "mode": "dual_worker",
                "worker_count": 2,
                "fallback_used": False,
                "total_seconds": 10.5,
            }

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, json):
            return FakeResponse()

    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.vividvr.caption_bridge.httpx.AsyncClient",
        lambda **kwargs: FakeClient(),
    )

    result = asyncio.run(
        request_vividvr_caption_sidecar(
            config=VividVRCaptionBridgeConfig(
                enabled=True,
                base_url="http://127.0.0.1:31200",
                timeout_s=30.0,
            ),
            manifest_path=str(tmp_path / "manifest.json"),
            output_caption_path=str(output),
            expected_caption_count=1,
        )
    )

    assert result.mode == "dual_worker"
    assert result.worker_count == 2
    assert result.total_seconds == 10.5


def test_benchmark_writes_summary_json(tmp_path):
    summary = build_benchmark_summary(
        request_id="bench-1",
        manifest_path="/tmp/manifest.json",
        caption_file_path="/tmp/caption.txt",
        expected_caption_file="/tmp/base.txt",
        response_payload={
            "caption_count": 2,
            "mode": "dual_worker",
            "worker_count": 2,
            "fallback_used": False,
            "total_seconds": 35.2,
        },
        captions_match_baseline=True,
    )

    assert summary["mode"] == "dual_worker"
    assert summary["captions_match_baseline"] is True
    assert summary["worker_count"] == 2
```

测试命令：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_vividvr_caption_bridge.py \
  python/sglang/multimodal_gen/test/unit/test_vividvr_caption_sidecar_benchmark.py -q
```

预期结果：失败，报错包含 `mode` / `worker_count` / benchmark helper 缺失。

- [ ] **步骤 2：实现 bridge 元数据透传、主服务日志增强与 benchmark CLI**

修改文件：
- `python/sglang/multimodal_gen/runtime/vividvr/caption_bridge.py`
- `python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py`
- `python/sglang/multimodal_gen/tools/run_vividvr_caption_sidecar_benchmark.py`

实现骨架：

```python
@dataclass(frozen=True)
class VividVRCaptionBridgeResult:
    caption_file_path: str
    caption_count: int
    mode: str | None = None
    worker_count: int | None = None
    fallback_used: bool | None = None
    total_seconds: float | None = None


return VividVRCaptionBridgeResult(
    caption_file_path=caption_file_path,
    caption_count=int(data.get("caption_count") or expected_caption_count),
    mode=data.get("mode"),
    worker_count=data.get("worker_count"),
    fallback_used=data.get("fallback_used"),
    total_seconds=data.get("total_seconds"),
)
```

```python
logger.info(
    "VividVR caption bridge generated captions request_id=%s path=%s count=%s mode=%s workers=%s fallback=%s total_s=%s",
    request_id,
    result.caption_file_path,
    result.caption_count,
    result.mode,
    result.worker_count,
    result.fallback_used,
    result.total_seconds,
)
```

```python
def build_benchmark_summary(...):
    return {
        "request_id": request_id,
        "manifest_path": manifest_path,
        "caption_file_path": caption_file_path,
        "expected_caption_file": expected_caption_file,
        "caption_count": response_payload["caption_count"],
        "mode": response_payload.get("mode"),
        "worker_count": response_payload.get("worker_count"),
        "fallback_used": response_payload.get("fallback_used"),
        "total_seconds": response_payload.get("total_seconds"),
        "captions_match_baseline": captions_match_baseline,
    }


def main():
    # 1. build manifest
    # 2. call sidecar /v1/vividvr/captions
    # 3. compare caption file against --expected-caption-file when provided
    # 4. write summary json to --report-path
```

测试命令：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_vividvr_caption_bridge.py \
  python/sglang/multimodal_gen/test/unit/test_vividvr_caption_sidecar_benchmark.py -q
```

预期结果：新增 bridge / benchmark 测试通过；旧 bridge 测试不回退。

- [ ] **步骤 3：验证 sidecar CLI 在原版环境可启动并提交 bridge/benchmark 代码**

修改文件：
- `python/sglang/multimodal_gen/runtime/vividvr/caption_bridge.py`
- `python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py`
- `python/sglang/multimodal_gen/tools/run_vividvr_caption_sidecar_benchmark.py`
- `python/sglang/multimodal_gen/test/unit/test_vividvr_caption_bridge.py`
- `python/sglang/multimodal_gen/test/unit/test_vividvr_caption_sidecar_benchmark.py`

验证命令：

```bash
PYTHONPATH=python /home/zhiheng/Vivid-VR/.venv/bin/python \
  python/sglang/multimodal_gen/tools/run_vividvr_caption_sidecar.py --help
```

预期结果：退出码 `0`，且帮助信息包含新增的 `--parallel-workers`、`--worker-devices`、`--allow-serial-fallback`。

提交命令：

```bash
git add python/sglang/multimodal_gen/runtime/vividvr/caption_bridge.py \
  python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py \
  python/sglang/multimodal_gen/tools/run_vividvr_caption_sidecar_benchmark.py \
  python/sglang/multimodal_gen/test/unit/test_vividvr_caption_bridge.py \
  python/sglang/multimodal_gen/test/unit/test_vividvr_caption_sidecar_benchmark.py
git commit -m "feat(vividvr): add caption sidecar benchmark and bridge metrics"
```

---

### 任务 5：补齐 sidecar CLI、运行文档和部署命令

**文件：**
- 修改：`python/sglang/multimodal_gen/tools/run_vividvr_caption_sidecar.py`
- 修改：`docs_xzh/run_command/vividvr_default_run_and_serve_commands.md`
- 修改：`docs_xzh/run_vivid_benchmark.md`
- 修改：`python/sglang/multimodal_gen/test/unit/test_vividvr_caption_sidecar_tool.py`

- [ ] **步骤 1：先写失败测试，锁定 sidecar 新 CLI 参数**

修改文件：
- `python/sglang/multimodal_gen/test/unit/test_vividvr_caption_sidecar_tool.py`

测试代码：

```python
def test_parse_args_supports_dual_worker_flags(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_vividvr_caption_sidecar.py",
            "--host",
            "127.0.0.1",
            "--port",
            "31200",
            "--parallel-workers",
            "2",
            "--worker-devices",
            "cuda:0,cuda:1",
            "--allow-serial-fallback",
        ],
    )

    args = run_vividvr_caption_sidecar.parse_args()

    assert args.parallel_workers == 2
    assert args.worker_devices == ["cuda:0", "cuda:1"]
    assert args.allow_serial_fallback is True
```

测试命令：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_vividvr_caption_sidecar_tool.py \
  -k "parse_args_supports_dual_worker_flags" -q
```

预期结果：失败，报错包含 `parallel_workers`、`worker_devices` 或 `allow_serial_fallback` 未定义。

- [ ] **步骤 2：实现 CLI 参数并更新默认运行文档**

修改文件：
- `python/sglang/multimodal_gen/tools/run_vividvr_caption_sidecar.py`
- `docs_xzh/run_command/vividvr_default_run_and_serve_commands.md`
- `docs_xzh/run_vivid_benchmark.md`

实现与文档要求：

```python
parser.add_argument("--parallel-workers", type=int, default=1)
parser.add_argument("--worker-devices", default="cuda")
parser.add_argument("--allow-serial-fallback", action="store_true")

args = parser.parse_args()
args.worker_devices = [
    item.strip() for item in args.worker_devices.split(",") if item.strip()
]
if args.parallel_workers != len(args.worker_devices):
    raise ValueError("parallel_workers must match worker_devices length")
```

文档里新增的 sidecar 启动命令应类似：

```bash
tmux new-session -d -s vividvr_caption_sidecar_dual \
  'cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/logs && \
   PYTHONPATH=python /home/zhiheng/Vivid-VR/.venv/bin/python \
   python/sglang/multimodal_gen/tools/run_vividvr_caption_sidecar.py \
   --host 127.0.0.1 --port 31200 \
   --parallel-workers 2 \
   --worker-devices cuda:0,cuda:1 \
   --allow-serial-fallback \
   2>&1 | tee Vivid_Acceptance/logs/vividvr_caption_sidecar_dual_$(date -u +%Y%m%dT%H%M%SZ).log'
```

测试命令：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_vividvr_caption_sidecar_tool.py -q
```

预期结果：sidecar tool 单测全部通过。

文档验证命令：

```bash
rg -n "vividvr_caption_sidecar_dual|parallel-workers|worker-devices" \
  docs_xzh/run_command/vividvr_default_run_and_serve_commands.md \
  docs_xzh/run_vivid_benchmark.md -S
```

预期结果：命中文档中的双 worker sidecar 启动命令和 benchmark 说明。

- [ ] **步骤 3：提交 CLI 与文档收口**

修改文件：
- `python/sglang/multimodal_gen/tools/run_vividvr_caption_sidecar.py`
- `docs_xzh/run_command/vividvr_default_run_and_serve_commands.md`
- `docs_xzh/run_vivid_benchmark.md`
- `python/sglang/multimodal_gen/test/unit/test_vividvr_caption_sidecar_tool.py`

验证命令：

```bash
git diff --cached --name-only
```

预期结果：只包含 sidecar CLI、运行文档和相关测试。

提交命令：

```bash
git add python/sglang/multimodal_gen/tools/run_vividvr_caption_sidecar.py \
  docs_xzh/run_command/vividvr_default_run_and_serve_commands.md \
  docs_xzh/run_vivid_benchmark.md \
  python/sglang/multimodal_gen/test/unit/test_vividvr_caption_sidecar_tool.py
git commit -m "docs(vividvr): add dual gpu caption sidecar commands"
```

---

### 任务 6：执行独立 benchmark、端到端回归，并回填交接文档

**文件：**
- 创建：`docs_xzh/hand_over/vividvr_dual_gpu_caption_service_handover_20260623.md`
- 产物目录：`Vivid_Acceptance/logs/`、`Vivid_Acceptance/captions/service_sidecars/`、`Vivid_Acceptance/indicator/`、`Vivid_Acceptance/result_videos/service_benchmark/`

- [ ] **步骤 1：先做 caption sidecar 独立 benchmark，验证文本逐字一致**

修改文件：
- 无源码修改；生成 benchmark 产物到 `Vivid_Acceptance/indicator/` 和 `Vivid_Acceptance/captions/service_sidecars/`

启动 sidecar：

```bash
tmux new-session -d -s vividvr_caption_sidecar_dual \
  'cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/logs && \
   PYTHONPATH=python /home/zhiheng/Vivid-VR/.venv/bin/python \
   python/sglang/multimodal_gen/tools/run_vividvr_caption_sidecar.py \
   --host 127.0.0.1 --port 31200 \
   --parallel-workers 2 \
   --worker-devices cuda:0,cuda:1 \
   --allow-serial-fallback \
   2>&1 | tee Vivid_Acceptance/logs/vividvr_caption_sidecar_dual_$(date -u +%Y%m%dT%H%M%SZ).log'
```

运行 benchmark：

```bash
tmux new-session -d -s vividvr_caption_bench \
  'cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/indicator Vivid_Acceptance/captions/service_sidecars Vivid_Acceptance/logs && \
   export PYTHONPATH=python && \
   /home/zhiheng/sglang/.venv/bin/python \
   python/sglang/multimodal_gen/tools/run_vividvr_caption_sidecar_benchmark.py \
   --sidecar-url http://127.0.0.1:31200 \
   --video-path /home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.mp4 \
   --expected-caption-file /home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.txt \
   --num-temporal-process-frames 121 \
   --tile-size 128 \
   --tile-stride 64 \
   --artifact-prefix vividvr_dual_caption_bench_130f \
   --caption-output-dir /home/zhiheng/sglang/Vivid_Acceptance/captions/service_sidecars \
   --report-path /home/zhiheng/sglang/Vivid_Acceptance/indicator/vividvr_dual_caption_bench_130f.json \
   2>&1 | tee Vivid_Acceptance/logs/vividvr_dual_caption_bench_$(date -u +%Y%m%dT%H%M%SZ).log'
```

校验命令：

```bash
cmp -s \
  /home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.txt \
  /home/zhiheng/sglang/Vivid_Acceptance/captions/service_sidecars/vividvr_dual_caption_bench_130f.txt
```

预期结果：`cmp` 退出码 `0`；benchmark JSON 中 `captions_match_baseline=true`、`mode="dual_worker"`、`worker_count=2`、`fallback_used=false`。

- [ ] **步骤 2：再做双卡 `serve` 端到端 FlowCut 回归**

修改文件：
- 无源码修改；生成 service benchmark 产物和 callback 日志

启动双卡主服务：

```bash
tmux new-session -d -s vividvr_serve_dual_default \
  'cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/logs Vivid_Acceptance/result_videos/service_benchmark Vivid_Acceptance/captions/service_sidecars && \
   export PYTHONUNBUFFERED=1 && export PYTHONPATH=python && \
   export SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE=eager_global && \
   export SGLANG_VIVIDVR_CONNECTOR_CONTROL_POOL_SIZE=1 && \
   CUDA_VISIBLE_DEVICES=0,1 /home/zhiheng/sglang/.venv/bin/sglang serve \
   --model-path /home/zhiheng/Vivid-VR/ckpts/CogVideoX1.5-5B \
   --model-id VividVR \
   --pipeline-class-name CogVideoXVividVRControlNetPipeline \
   --component-paths.vividvr /home/zhiheng/Vivid-VR/ckpts/Vivid-VR \
   --attention-backend fa \
   --num-gpus 2 --tp-size 1 --sp-degree 2 --ulysses-degree 2 --ring-degree 1 \
   --enable-torch-compile --dist-timeout 3600 \
   --host 127.0.0.1 --port 31191 --master-port 30191 --scheduler-port 56191 --strict-ports \
   --output-path /home/zhiheng/sglang/Vivid_Acceptance/result_videos/service_benchmark \
   --prompt-file-path /home/zhiheng/Vivid-VR/input/720p/prompt.txt \
   --vividvr-caption-bridge \
   --vividvr-caption-sidecar-url http://127.0.0.1:31200 \
   --vividvr-caption-work-dir /home/zhiheng/sglang/Vivid_Acceptance/captions/service_sidecars \
   --vividvr-caption-sidecar-timeout 1800 \
   2>&1 | tee Vivid_Acceptance/logs/vividvr_serve_dual_default_$(date -u +%Y%m%dT%H%M%SZ).log'
```

提交 FlowCut 验收：

```bash
tmux new-session -d -s vividvr_flowcut_dual_caption_accept \
  'cd /home/zhiheng/sglang && export PYTHONPATH=python && \
   export BASE_URL=http://127.0.0.1:31191 && \
   export TASK_ID=flowcut-dual-caption-$(date -u +%Y%m%dT%H%M%SZ) && \
   export INPUT_VIDEO=/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.mp4 && \
   export OUTPUT_PATH=/home/zhiheng/sglang/Vivid_Acceptance/result_videos/service_benchmark/${TASK_ID}.mp4 && \
   export PERF_DUMP_PATH=/home/zhiheng/sglang/Vivid_Acceptance/indicator/${TASK_ID}.json && \
   export CALLBACK_LOG=/home/zhiheng/sglang/Vivid_Acceptance/logs/${TASK_ID}_callback.jsonl && \
   /home/zhiheng/sglang/.venv/bin/python \
   python/sglang/multimodal_gen/tools/run_flowcut_vividvr_service_acceptance.py \
   --base-url "${BASE_URL}" \
   --task-id "${TASK_ID}" \
   --callback-log "${CALLBACK_LOG}" \
   --video-input-path "${INPUT_VIDEO}" \
   --num-inference-steps 20 \
   --seed 42 \
   --num-temporal-process-frames 121 \
   --output-path "${OUTPUT_PATH}" \
   --perf-dump-path "${PERF_DUMP_PATH}" \
   --submit-timeout-s 2400 \
   --poll-timeout-s 2400 \
   2>&1 | tee /home/zhiheng/sglang/Vivid_Acceptance/logs/${TASK_ID}.log'
```

校验命令：

```bash
python - <<'PY'
import json, pathlib
callback_log = max(pathlib.Path("/home/zhiheng/sglang/Vivid_Acceptance/logs").glob("flowcut-dual-caption-*callback.jsonl"))
lines = [json.loads(line) for line in callback_log.read_text(encoding="utf-8").splitlines() if line.strip()]
assert any(item.get("status") == "running" for item in lines)
assert any(item.get("status") == "succeeded" for item in lines)
print("callback_ok")
PY
```

预期结果：FlowCut 接单成功；callback 日志同时包含 `running` 和 `succeeded`；自动生成的 sidecar 文件行数等于 manifest `expected_caption_count`；最终视频和 perf JSON 落盘。

- [ ] **步骤 3：把 benchmark / 回归结果写入交接文档并提交**

修改文件：
- `docs_xzh/hand_over/vividvr_dual_gpu_caption_service_handover_20260623.md`

文档至少包含：

```markdown
- sidecar 启动命令和 tmux session 名
- benchmark JSON 路径、caption sidecar 路径、日志路径
- 双 worker 与单卡基线的耗时对比
- `caption.txt` 与基线 `cmp -s` 一致结论
- FlowCut callback 结果、最终视频路径、perf JSON 路径
- 是否触发 fallback、已知风险和下一步建议
```

验证命令：

```bash
rg -n "dual_worker|captions_match_baseline|fallback_used|FlowCut" \
  docs_xzh/hand_over/vividvr_dual_gpu_caption_service_handover_20260623.md -S
```

预期结果：交接文档包含 benchmark、文本一致性、FlowCut 验收和 fallback 信息。

提交命令：

```bash
git add docs_xzh/hand_over/vividvr_dual_gpu_caption_service_handover_20260623.md
git commit -m "docs(vividvr): hand off dual gpu caption service acceptance"
```

---

## 自检

- 规格覆盖度：
  - 双 worker 调度、串行回退、可选响应元数据、独立 benchmark、端到端回归都已有独立任务覆盖。
  - “逐字完全一致”通过 `cmp -s` 和 benchmark JSON 的 `captions_match_baseline` 进行显式验收。
- 占位符扫描：
  - 没有 `TODO`、`待定`、`后续实现` 之类占位符。
  - 所有步骤都给出了文件、命令和预期结果。
- 类型一致性：
  - 计划内统一使用 `CaptionClipResult`、`CaptionWorkerBatchResult`、`CaptionRequestMetrics`、`VividVRCaptionBridgeResult` 这些名称，没有混用别名。

## 执行交接

计划已完成并保存到 `docs/superpowers/plans/2026-06-23-vividvr-dual-gpu-caption-service.md`。两种执行方式：

**1. 子代理驱动（推荐）** - 每个任务调度一个新的子代理，任务间进行审查，快速迭代

**2. 内联执行** - 在当前会话中使用 executing-plans 执行任务，批量执行并设有检查点

选哪种方式？
