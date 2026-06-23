# VividVR Caption Sidecar Service 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 让 Vivid-VR `serve` 对外调用时只需要用户传入视频，服务端自动通过原版 `/home/zhiheng/Vivid-VR/.venv` 中的 caption 模型生成 sidecar caption，并由 `sglang` 原生 Vivid-VR 推理链消费。

**架构：** `sglang` 主服务仍运行在 `/home/zhiheng/sglang/.venv`，只负责下载/定位输入视频、生成 caption manifest、调用本机常驻 caption sidecar，并把生成的 caption 文件路径写入 `VividVRSamplingParams`。caption sidecar 是独立 HTTP 服务，必须用 `/home/zhiheng/Vivid-VR/.venv/bin/python` 启动，只读取 manifest 和视频，输出一行一个 temporal clip caption 的 sidecar 文件；manifest 里的 spatial tile 信息只作为调试和语义对齐元数据保留，不参与 sidecar 行数契约。

**技术栈：** Python、FastAPI/httpx、Pydantic、pytest、tmux、原版 Vivid-VR `VRDiT.captioner.CogVLM2_Captioner`、现有 `sglang.multimodal_gen.runtime.vividvr` windowing/tiling/captioning helpers。

---

## 文件结构

- 创建 `python/sglang/multimodal_gen/runtime/vividvr/caption_manifest.py`
  - 负责在 `sglang` 环境中构造稳定 JSON manifest。
  - 只包含轻量逻辑：视频帧元信息、Phase D temporal clip 计划、每个 clip 的空间 tile 计数、期望 caption 总行数。
  - 使用现有 `load_control_video_frames`、`build_vividvr_temporal_window_plan`、`prepare_tiling_infos_generator` 保持与 Phase C/D 语义一致。
- 创建 `python/sglang/multimodal_gen/runtime/vividvr/caption_bridge.py`
  - 负责主服务到 sidecar 的 HTTP 客户端、配置解析、输出 caption 文件校验。
  - 不导入原版 `/home/zhiheng/Vivid-VR` 代码。
- 创建 `python/sglang/multimodal_gen/tools/run_vividvr_caption_sidecar.py`
  - 作为独立 sidecar 服务入口。
  - 由原版 `/home/zhiheng/Vivid-VR/.venv/bin/python` 启动。
  - 导入原版 `VRDiT.captioner.create_captioner` 或 `CogVLM2_Captioner`，按 manifest 的 temporal clip 顺序生成 caption 文件。
- 修改 `python/sglang/multimodal_gen/runtime/server_args.py`
  - 增加 caption bridge 配置：启用开关、sidecar URL、manifest/output 目录、请求超时。
- 修改 `python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py`
  - 在 Vivid-VR `repair` 和 `flowcut` 路径中，如果请求未传 `caption_file_path` 且 bridge 已启用，则先生成 sidecar caption，再构造 `VividVRSamplingParams`。
- 修改 `python/sglang/multimodal_gen/runtime/vividvr/__init__.py`
  - 导出新 helper，便于单元测试和工具复用。
- 创建 `python/sglang/multimodal_gen/test/unit/test_vividvr_caption_manifest.py`
  - 覆盖 manifest 的 clip/tile 元数据顺序、clip 级期望 caption 数、JSON 可序列化。
- 创建 `python/sglang/multimodal_gen/test/unit/test_vividvr_caption_bridge.py`
  - 覆盖 HTTP 请求、超时/失败映射、caption 行数校验。
- 修改 `python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py`
  - 覆盖 FlowCut 请求不传 caption 时，自动生成 caption 文件并接入 sampling 参数。
- 修改 `python/sglang/multimodal_gen/test/unit/test_server_args.py`
  - 覆盖新增 CLI 参数和路径展开。
- 修改 `docs_xzh/run_command/vividvr_default_run_and_serve_commands.md`
  - 增加 sidecar 启动命令、主服务启用 bridge 的单卡/双卡 serve 命令。
- 创建 `docs_xzh/hand_over/vividvr_caption_sidecar_service_handover_20260622.md`
  - 记录方案、部署顺序、验收命令和已知风险。

---

## 任务 1：新增 caption manifest 构造与单元测试

**文件：**
- 创建：`python/sglang/multimodal_gen/runtime/vividvr/caption_manifest.py`
- 修改：`python/sglang/multimodal_gen/runtime/vividvr/__init__.py`
- 测试：`python/sglang/multimodal_gen/test/unit/test_vividvr_caption_manifest.py`

- [ ] **步骤 1：编写失败测试**

修改文件：`python/sglang/multimodal_gen/test/unit/test_vividvr_caption_manifest.py`

测试代码要覆盖：

```python
from pathlib import Path

import torch

from sglang.multimodal_gen.runtime.vividvr.caption_manifest import (
    VividVRCaptionManifest,
    build_vividvr_caption_manifest_from_video_info,
)


def test_manifest_counts_temporal_clips_and_spatial_tiles():
    manifest = build_vividvr_caption_manifest_from_video_info(
        video_path="/tmp/input.mp4",
        fps=24.0,
        num_frames=130,
        height=720,
        width=1280,
        num_temporal_process_frames=121,
        tile_size=128,
        tile_stride=64,
    )

    assert isinstance(manifest, VividVRCaptionManifest)
    assert manifest.video_path == "/tmp/input.mp4"
    assert manifest.num_frames == 130
    assert manifest.num_temporal_process_frames == 121
    assert len(manifest.clips) == 2
    assert manifest.expected_caption_count == len(manifest.clips)
    assert manifest.clips[0].clip_index == 0
    assert manifest.clips[1].clip_index == 1


def test_manifest_round_trips_json(tmp_path):
    manifest = build_vividvr_caption_manifest_from_video_info(
        video_path="/tmp/input.mp4",
        fps=24.0,
        num_frames=9,
        height=64,
        width=64,
        num_temporal_process_frames=9,
        tile_size=128,
        tile_stride=64,
    )
    path = tmp_path / "manifest.json"

    manifest.write_json(path)
    loaded = VividVRCaptionManifest.read_json(path)

    assert loaded == manifest
    assert loaded.expected_caption_count == 1
```

测试命令：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_vividvr_caption_manifest.py -q
```

预期结果：测试失败，报错包含 `No module named 'sglang.multimodal_gen.runtime.vividvr.caption_manifest'`。

Commit 信息：不提交，本步骤只提交失败测试给下一步验证。

- [ ] **步骤 2：实现 manifest 数据结构与 helper**

修改文件：`python/sglang/multimodal_gen/runtime/vividvr/caption_manifest.py`、`python/sglang/multimodal_gen/runtime/vividvr/__init__.py`

实现要求：

```python
@dataclass(frozen=True)
class VividVRCaptionTileSpec:
    tile_index: int
    t_start: int
    t_end: int
    h_start: int
    h_end: int
    w_start: int
    w_end: int


@dataclass(frozen=True)
class VividVRCaptionClipSpec:
    clip_index: int
    start_frame: int
    end_frame: int
    original_num_frames: int
    padded_num_frames: int
    tile_count: int
    tiles: list[VividVRCaptionTileSpec]


@dataclass(frozen=True)
class VividVRCaptionManifest:
    version: int
    video_path: str
    fps: float
    num_frames: int
    height: int
    width: int
    num_temporal_process_frames: int
    tile_size: int
    tile_stride: int
    expected_caption_count: int
    clips: list[VividVRCaptionClipSpec]
```

`build_vividvr_caption_manifest_from_video_info(...)` 必须：

- 用 `build_vividvr_temporal_window_plan` 计算 temporal clip。
- 对每个 clip 用 `torch.empty((1, padded_num_frames, 3, height, width), device="meta")` 或 CPU 小 tensor 生成 tile 切片。
- 调用 `prepare_tiling_infos_generator(enable_spatial_tiling=True, enable_temporal_tiling=False, tile_size=tile_size, tile_stride=tile_stride)`。
- 按 clip 顺序、tile 顺序记录 tile spec。
- `expected_caption_count` 等于 temporal clip 数；spatial tile 数继续保留在每个 clip 的 `tile_count` 里，仅用于调试和语义对齐。
- 提供 `write_json(path)` / `read_json(path)`。

测试命令：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_vividvr_caption_manifest.py -q
```

预期结果：`2 passed`。

Commit 信息：

```bash
git add python/sglang/multimodal_gen/runtime/vividvr/caption_manifest.py \
  python/sglang/multimodal_gen/runtime/vividvr/__init__.py \
  python/sglang/multimodal_gen/test/unit/test_vividvr_caption_manifest.py
git commit -m "feat(vividvr): add caption manifest contract"
```

---

## 任务 2：新增 caption bridge 客户端与校验

**文件：**
- 创建：`python/sglang/multimodal_gen/runtime/vividvr/caption_bridge.py`
- 测试：`python/sglang/multimodal_gen/test/unit/test_vividvr_caption_bridge.py`

- [ ] **步骤 1：编写失败测试**

修改文件：`python/sglang/multimodal_gen/test/unit/test_vividvr_caption_bridge.py`

测试代码要覆盖：

```python
import pytest

from sglang.multimodal_gen.runtime.vividvr.caption_bridge import (
    VividVRCaptionBridgeConfig,
    request_vividvr_caption_sidecar,
    validate_caption_sidecar_file,
)


def test_validate_caption_sidecar_file_requires_exact_line_count(tmp_path):
    caption_file = tmp_path / "caption.txt"
    caption_file.write_text("clip a\nclip b\n", encoding="utf-8")

    validate_caption_sidecar_file(caption_file, expected_count=2)

    with pytest.raises(ValueError, match="expected 3 captions"):
        validate_caption_sidecar_file(caption_file, expected_count=3)


def test_request_caption_sidecar_posts_manifest(monkeypatch, tmp_path):
    calls = {}
    output = tmp_path / "caption.txt"
    output.write_text("caption 0\n", encoding="utf-8")

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"caption_file_path": str(output), "caption_count": 1}

    class FakeClient:
        def __init__(self, **kwargs):
            calls["client_kwargs"] = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, json):
            calls["url"] = url
            calls["json"] = json
            return FakeResponse()

    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.vividvr.caption_bridge.httpx.AsyncClient",
        FakeClient,
    )

    config = VividVRCaptionBridgeConfig(
        enabled=True,
        base_url="http://127.0.0.1:31200",
        timeout_s=30.0,
    )

    result = pytest.run(asyncio=True)(
        request_vividvr_caption_sidecar(
            config=config,
            manifest_path=str(tmp_path / "manifest.json"),
            output_caption_path=str(output),
            expected_caption_count=1,
        )
    )

    assert result.caption_file_path == str(output)
    assert calls["url"] == "http://127.0.0.1:31200/v1/vividvr/captions"
    assert calls["client_kwargs"]["trust_env"] is False
```

如果本仓库没有 `pytest.run(asyncio=True)` helper，则使用 `asyncio.run(...)`。

测试命令：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_vividvr_caption_bridge.py -q
```

预期结果：测试失败，报错包含 `No module named 'sglang.multimodal_gen.runtime.vividvr.caption_bridge'`。

Commit 信息：不提交，本步骤只提交失败测试给下一步验证。

- [ ] **步骤 2：实现 bridge 客户端**

修改文件：`python/sglang/multimodal_gen/runtime/vividvr/caption_bridge.py`

实现要求：

```python
@dataclass(frozen=True)
class VividVRCaptionBridgeConfig:
    enabled: bool = False
    base_url: str | None = None
    timeout_s: float = 1800.0


@dataclass(frozen=True)
class VividVRCaptionBridgeResult:
    caption_file_path: str
    caption_count: int
```

`request_vividvr_caption_sidecar(...)` 必须：

- 当 `enabled=False` 时抛 `RuntimeError("VividVR caption bridge is disabled")`。
- 用 `httpx.AsyncClient(timeout=config.timeout_s, trust_env=False)`。
- POST 到 `${base_url.rstrip("/")}/v1/vividvr/captions`。
- JSON 请求体包含 `manifest_path`、`output_caption_path`、`expected_caption_count`。
- sidecar 返回后调用 `validate_caption_sidecar_file`。
- 错误信息包含 sidecar URL 和任务路径，方便服务日志定位。

测试命令：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_vividvr_caption_bridge.py -q
```

预期结果：`2 passed`。

Commit 信息：

```bash
git add python/sglang/multimodal_gen/runtime/vividvr/caption_bridge.py \
  python/sglang/multimodal_gen/test/unit/test_vividvr_caption_bridge.py
git commit -m "feat(vividvr): add caption sidecar bridge client"
```

---

## 任务 3：增加 ServerArgs caption bridge 配置

**文件：**
- 修改：`python/sglang/multimodal_gen/runtime/server_args.py`
- 测试：`python/sglang/multimodal_gen/test/unit/test_server_args.py`

- [ ] **步骤 1：编写失败测试**

修改文件：`python/sglang/multimodal_gen/test/unit/test_server_args.py`

新增测试：

```python
class TestVividVRCaptionBridgeArgs(unittest.TestCase):
    def test_caption_bridge_cli_args_are_parsed(self):
        parser = FlexibleArgumentParser()
        ServerArgs.add_cli_args(parser)
        argv = [
            "--model-path",
            "/tmp/vividvr",
            "--model-id",
            "VividVR",
            "--pipeline-class-name",
            "CogVideoXVividVRControlNetPipeline",
            "--vividvr-caption-bridge",
            "--vividvr-caption-sidecar-url",
            "http://127.0.0.1:31200",
            "--vividvr-caption-work-dir",
            "~/vividvr_caption_sidecars",
            "--vividvr-caption-sidecar-timeout",
            "120",
        ]

        with patch.object(
            PipelineConfig,
            "from_kwargs",
            return_value=VividVRPipelineConfig(),
        ):
            args, unknown_args = parser.parse_known_args(argv)
            server_args = ServerArgs.from_cli_args(args, unknown_args)

        self.assertTrue(server_args.vividvr_caption_bridge)
        self.assertEqual(
            server_args.vividvr_caption_sidecar_url,
            "http://127.0.0.1:31200",
        )
        self.assertEqual(server_args.vividvr_caption_sidecar_timeout, 120.0)
        self.assertFalse(server_args.vividvr_caption_work_dir.startswith("~"))
```

测试命令：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_server_args.py::TestVividVRCaptionBridgeArgs -q
```

预期结果：测试失败，报错包含 `unrecognized arguments: --vividvr-caption-bridge`。

Commit 信息：不提交，本步骤只提交失败测试给下一步验证。

- [ ] **步骤 2：实现 CLI 参数**

修改文件：`python/sglang/multimodal_gen/runtime/server_args.py`

实现要求：

- 在 `ServerArgs` dataclass 中增加：
  - `vividvr_caption_bridge: bool = False`
  - `vividvr_caption_sidecar_url: str | None = None`
  - `vividvr_caption_work_dir: str | None = None`
  - `vividvr_caption_sidecar_timeout: float = 1800.0`
- 在 `add_cli_args` 中增加：
  - `--vividvr-caption-bridge`
  - `--vividvr-caption-sidecar-url`
  - `--vividvr-caption-work-dir`
  - `--vividvr-caption-sidecar-timeout`
- 复用现有 `expand_path_fields(self)` 处理 `vividvr_caption_work_dir` 的 `~` 展开。
- `_validate_parameters` 中增加校验：启用 bridge 时必须提供 sidecar URL。

测试命令：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_server_args.py::TestVividVRCaptionBridgeArgs -q
```

预期结果：`1 passed`。

Commit 信息：

```bash
git add python/sglang/multimodal_gen/runtime/server_args.py \
  python/sglang/multimodal_gen/test/unit/test_server_args.py
git commit -m "feat(vividvr): add caption bridge server args"
```

---

## 任务 4：实现 caption sidecar 服务入口

**文件：**
- 创建：`python/sglang/multimodal_gen/tools/run_vividvr_caption_sidecar.py`
- 测试：`python/sglang/multimodal_gen/test/unit/test_vividvr_caption_sidecar_tool.py`

- [ ] **步骤 1：编写失败测试**

修改文件：`python/sglang/multimodal_gen/test/unit/test_vividvr_caption_sidecar_tool.py`

测试代码要覆盖：

```python
from fastapi.testclient import TestClient

from sglang.multimodal_gen.tools.run_vividvr_caption_sidecar import (
    CaptionSidecarState,
    create_app,
)


def test_sidecar_writes_captions_in_manifest_order(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "captions.txt"
    manifest_path.write_text(
        '''
        {
          "version": 1,
          "video_path": "/tmp/in.mp4",
          "fps": 24.0,
          "num_frames": 9,
          "height": 64,
          "width": 64,
          "num_temporal_process_frames": 9,
          "tile_size": 128,
          "tile_stride": 64,
          "expected_caption_count": 1,
          "clips": [
            {
              "clip_index": 0,
              "start_frame": 0,
              "end_frame": 9,
              "original_num_frames": 9,
              "padded_num_frames": 9,
              "tile_count": 1,
              "tiles": [
                {"tile_index": 0, "t_start": 0, "t_end": 9, "h_start": 0, "h_end": 64, "w_start": 0, "w_end": 64}
              ]
            }
          ]
        }
        ''',
        encoding="utf-8",
    )

    class FakeCaptioner:
        def to(self, device):
            return self

        def __call__(self, video, fps=None):
            return "caption from fake model"

    state = CaptionSidecarState(captioner=FakeCaptioner(), device="cpu")
    app = create_app(state)
    client = TestClient(app)

    response = client.post(
        "/v1/vividvr/captions",
        json={
            "manifest_path": str(manifest_path),
            "output_caption_path": str(output_path),
            "expected_caption_count": 1,
        },
    )

    assert response.status_code == 200
    assert response.json()["caption_count"] == 1
    assert output_path.read_text(encoding="utf-8").splitlines() == [
        "caption from fake model"
    ]
```

测试命令：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_vividvr_caption_sidecar_tool.py -q
```

预期结果：测试失败，报错包含 `No module named 'sglang.multimodal_gen.tools.run_vividvr_caption_sidecar'`。

Commit 信息：不提交，本步骤只提交失败测试给下一步验证。

- [ ] **步骤 2：实现 sidecar 应用和 CLI**

修改文件：`python/sglang/multimodal_gen/tools/run_vividvr_caption_sidecar.py`

实现要求：

- CLI 参数：
  - `--host`，默认 `127.0.0.1`
  - `--port`，默认 `31200`
  - `--vividvr-root`，默认 `/home/zhiheng/Vivid-VR`
  - `--cogvlm2-ckpt-path`，默认 `/home/zhiheng/Vivid-VR/ckpts/cogvlm2-llama3-caption`
  - `--device`，默认 `cuda`
- 启动时把 `vividvr_root` 和 `vividvr_root/src` 加入 `sys.path`，导入 `VRDiT.captioner.create_captioner`。
- 提供 `GET /health`，返回 `{"status": "ok"}`。
- 提供 `POST /v1/vividvr/captions`。
- 对每个 manifest temporal clip，按顺序从输入视频加载对应 clip tensor，调用 captioner，写入 `output_caption_path`。
- 输出文件写入时先写 `${output_caption_path}.tmp`，全部成功后原子替换为最终路径。
- 返回 JSON：`caption_file_path`、`caption_count`、`manifest_path`。

测试命令：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_vividvr_caption_sidecar_tool.py -q
```

预期结果：`1 passed`。

Commit 信息：

```bash
git add python/sglang/multimodal_gen/tools/run_vividvr_caption_sidecar.py \
  python/sglang/multimodal_gen/test/unit/test_vividvr_caption_sidecar_tool.py
git commit -m "feat(vividvr): add caption sidecar service tool"
```

---

## 任务 5：在 Vivid-VR serve 请求中接入 caption bridge

**文件：**
- 修改：`python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py`
- 测试：`python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py`
- 测试：`python/sglang/multimodal_gen/test/unit/test_video_api_vividvr.py`

- [ ] **步骤 1：编写失败测试**

修改文件：`python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py`

新增测试：

```python
def test_flowcut_endpoint_generates_caption_when_bridge_enabled(monkeypatch, tmp_path):
    scheduled = {}

    class AvailableSemaphore:
        def locked(self):
            return False

        async def acquire(self):
            pass

        def release(self):
            pass

    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("restore the video", encoding="utf-8")
    caption_file = tmp_path / "caption_sidecars" / "task-auto.txt"
    manifest_file = tmp_path / "caption_sidecars" / "task-auto.json"

    monkeypatch.setattr(video_api, "_VIDEOEDIT_SEMAPHORE", AvailableSemaphore())
    monkeypatch.setattr(
        video_api,
        "get_global_server_args",
        lambda: SimpleNamespace(
            input_save_path=str(tmp_path / "inputs"),
            output_path=str(tmp_path / "outputs"),
            prompt_file_path=str(prompt_file),
            pipeline_config=SimpleNamespace(default_prompt_file_path=str(prompt_file)),
            model_id="vividvr",
            pipeline_class_name="CogVideoXVividVRControlNetPipeline",
            vividvr_caption_bridge=True,
            vividvr_caption_sidecar_url="http://127.0.0.1:31200",
            vividvr_caption_work_dir=str(tmp_path / "caption_sidecars"),
            vividvr_caption_sidecar_timeout=30.0,
        ),
    )
    monkeypatch.setattr(
        video_api,
        "build_vividvr_caption_manifest_for_video_path",
        lambda **kwargs: SimpleNamespace(
            expected_caption_count=1,
            write_json=lambda path: Path(path).write_text("{}", encoding="utf-8"),
        ),
    )

    async def fake_request_caption_sidecar(**kwargs):
        Path(kwargs["output_caption_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs["output_caption_path"]).write_text("caption 0\n", encoding="utf-8")
        return SimpleNamespace(
            caption_file_path=kwargs["output_caption_path"],
            caption_count=1,
        )

    monkeypatch.setattr(video_api, "request_vividvr_caption_sidecar", fake_request_caption_sidecar)

    captured_kwargs = {}
    monkeypatch.setattr(
        video_api.VividVRSamplingParams,
        "from_user_kwargs",
        staticmethod(
            lambda server_args, **kwargs: captured_kwargs.update(kwargs)
            or SimpleNamespace(output_file_path=lambda: str(tmp_path / "outputs" / "task-auto.mp4"))
        ),
    )
    monkeypatch.setattr(video_api, "prepare_request", lambda server_args, sampling_params: "prepared-batch")

    def fake_create_task(coro):
        scheduled["coro_name"] = coro.cr_code.co_name
        coro.close()
        return None

    monkeypatch.setattr(video_api.asyncio, "create_task", fake_create_task)

    client = _make_test_client()
    response = client.post(
        "/v1/videos/repairs/flowcut",
        json={
            "taskId": "task-auto",
            "timeout": -1,
            "callbackUrl": "http://127.0.0.1:9000/callback",
            "video_input_path": "/tmp/in.mp4",
        },
    )

    assert response.status_code == 200
    assert response.json()["code"] == 0
    assert captured_kwargs["caption_source"] == "caption_file"
    assert captured_kwargs["caption_file_path"].endswith("task-auto.txt")
```

测试命令：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py::test_flowcut_endpoint_generates_caption_when_bridge_enabled -q
```

预期结果：测试失败，原因是 `video_api` 尚未调用 caption bridge。

Commit 信息：不提交，本步骤只提交失败测试给下一步验证。

- [ ] **步骤 2：实现 video_api 接线**

修改文件：`python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py`

实现要求：

- 新增 async helper `_ensure_vividvr_caption_file(...)`：
  - 如果 `req.caption_file_path` 非空，直接返回原请求。
  - 如果 `server_args.vividvr_caption_bridge` 为 False，保持原行为，不自动生成。
  - 若 bridge 启用，创建 work dir：`server_args.vividvr_caption_work_dir or <output_dir>/caption_sidecars`。
  - manifest 路径：`${work_dir}/${request_id}.manifest.json`。
  - caption 路径：`${work_dir}/${request_id}.txt`。
  - 调用 `build_vividvr_caption_manifest_for_video_path(...)` 写 manifest。
  - 调用 `request_vividvr_caption_sidecar(...)` 生成 caption。
  - 返回 caption 文件路径。
- 在 `create_flowcut_video_repair` 和普通 `create_video_repair` 的 `_build_vividvr_repair_kwargs` 前调用该 helper，并把返回路径写入 `req.caption_file_path` 或传入 `_build_vividvr_repair_kwargs`。
- bridge 失败时：
  - FlowCut 返回 `FlowCutResponse(code=1, message="caption bridge failed: ...")`，不接单。
  - 普通 `/repairs` 抛 HTTP 500 或 400，错误 detail 包含 `caption bridge failed`。
- 不改变显式 `caption_file_path` 的行为，已有用户 sidecar 优先。

测试命令：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py::test_flowcut_endpoint_generates_caption_when_bridge_enabled \
  python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py::test_flowcut_endpoint_accepts_and_schedules_background_job \
  python/sglang/multimodal_gen/test/unit/test_video_api_vividvr.py -q
```

预期结果：新增 bridge 测试通过，既有显式 caption_file_path 测试继续通过。

Commit 信息：

```bash
git add python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py \
  python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py \
  python/sglang/multimodal_gen/test/unit/test_video_api_vividvr.py
git commit -m "feat(vividvr): auto-generate captions in serve requests"
```

---

## 任务 6：补充文档和启动命令

**文件：**
- 修改：`docs_xzh/run_command/vividvr_default_run_and_serve_commands.md`
- 创建：`docs_xzh/hand_over/vividvr_caption_sidecar_service_handover_20260622.md`

- [ ] **步骤 1：更新默认命令文档**

修改文件：`docs_xzh/run_command/vividvr_default_run_and_serve_commands.md`

文档必须增加：

```bash
tmux new-session -d -s vividvr_caption_sidecar \
  'cd /home/zhiheng/sglang && PYTHONPATH=python /home/zhiheng/Vivid-VR/.venv/bin/python python/sglang/multimodal_gen/tools/run_vividvr_caption_sidecar.py --host 127.0.0.1 --port 31200 2>&1 | tee Vivid_Acceptance/logs/vividvr_caption_sidecar_$(date -u +%Y%m%dT%H%M%SZ).log'
```

主服务命令必须增加：

```bash
--vividvr-caption-bridge \
--vividvr-caption-sidecar-url http://127.0.0.1:31200 \
--vividvr-caption-work-dir /home/zhiheng/sglang/Vivid_Acceptance/captions/service_sidecars \
--vividvr-caption-sidecar-timeout 1800
```

测试命令：

```bash
rg -n "vividvr_caption_sidecar|--vividvr-caption-bridge|31200" \
  docs_xzh/run_command/vividvr_default_run_and_serve_commands.md
```

预期结果：输出包含 sidecar tmux session、bridge 参数和 `31200`。

Commit 信息：不提交，本步骤和交接文档一起提交。

- [ ] **步骤 2：新增 handover 文档**

修改文件：`docs_xzh/hand_over/vividvr_caption_sidecar_service_handover_20260622.md`

文档必须记录：

- 用户请求仍然只需要 `video_input_path` 或 `video_url`。
- `caption_file_path` 仍作为可选 override，显式传入时优先。
- sidecar 必须由 `/home/zhiheng/Vivid-VR/.venv/bin/python` 启动。
- 主服务必须由 `/home/zhiheng/sglang/.venv/bin/python` 启动。
- sidecar 输出文件固定为一行一个 temporal clip caption，行数必须等于 manifest `expected_caption_count`。
- FlowCut bridge 失败返回 `code=1`，不进入推理队列。
- 重型验收必须在 tmux 中运行。

测试命令：

```bash
rg -n "caption_file_path|Vivid-VR/.venv|sglang/.venv|expected_caption_count|code=1|tmux" \
  docs_xzh/hand_over/vividvr_caption_sidecar_service_handover_20260622.md
```

预期结果：每个关键词至少出现一次。

Commit 信息：

```bash
git add docs_xzh/run_command/vividvr_default_run_and_serve_commands.md \
  docs_xzh/hand_over/vividvr_caption_sidecar_service_handover_20260622.md
git commit -m "docs(vividvr): document caption sidecar service"
```

---

## 任务 7：轻量回归与重型 serve 验收

**文件：**
- 修改：无代码文件；产物写入 `Vivid_Acceptance/logs`、`Vivid_Acceptance/indicator`、`Vivid_Acceptance/result_videos`
- 测试：单元测试、py_compile、tmux serve E2E

- [ ] **步骤 1：运行单元测试集合**

修改文件：无。

测试命令：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_vividvr_caption_manifest.py \
  python/sglang/multimodal_gen/test/unit/test_vividvr_caption_bridge.py \
  python/sglang/multimodal_gen/test/unit/test_vividvr_caption_sidecar_tool.py \
  python/sglang/multimodal_gen/test/unit/test_server_args.py::TestVividVRCaptionBridgeArgs \
  python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py \
  python/sglang/multimodal_gen/test/unit/test_stage_d_vividvr_temporal_orchestration.py \
  python/sglang/multimodal_gen/test/unit/test_sampling_params.py::TestSamplingParamsSubclass::test_vividvr_rejects_live_cogvlm2_caption_contract \
  -q
```

预期结果：全部 PASS；`test_vividvr_rejects_live_cogvlm2_caption_contract` 继续证明主推理环境没有重新启用 live CogVLM2。

Commit 信息：不提交，本步骤验证前面提交。

- [ ] **步骤 2：运行静态编译检查**

修改文件：无。

测试命令：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m py_compile \
  python/sglang/multimodal_gen/runtime/vividvr/caption_manifest.py \
  python/sglang/multimodal_gen/runtime/vividvr/caption_bridge.py \
  python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py \
  python/sglang/multimodal_gen/runtime/server_args.py \
  python/sglang/multimodal_gen/tools/run_vividvr_caption_sidecar.py
```

预期结果：命令退出码为 0，无语法错误输出。

Commit 信息：不提交，本步骤验证前面提交。

- [ ] **步骤 3：在 tmux 启动 caption sidecar**

修改文件：无。

测试命令：

```bash
tmux new-session -d -s vividvr_caption_sidecar \
  'cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/logs && PYTHONPATH=python /home/zhiheng/Vivid-VR/.venv/bin/python python/sglang/multimodal_gen/tools/run_vividvr_caption_sidecar.py --host 127.0.0.1 --port 31200 2>&1 | tee Vivid_Acceptance/logs/vividvr_caption_sidecar_$(date -u +%Y%m%dT%H%M%SZ).log'

curl --noproxy '*' --silent --show-error --fail http://127.0.0.1:31200/health
```

预期结果：

```json
{"status":"ok"}
```

补充说明：

- 如果宿主机没有系统级 `/usr/include/python3.10/Python.h`，sidecar 需要能自动探测已解压的 Python 3.10 dev headers；本轮实现默认会额外尝试 `~/tmp_py310dev/extracted/usr/include/python3.10` 和 `~/tmp_py310_headers/extracted/libpython3.10-dev/usr/include/python3.10`。
- sidecar 日志里出现 `[VividVR Caption Sidecar] python_include=...` 代表 headers 已就位。

用户查看命令：

```bash
tmux attach -r -t vividvr_caption_sidecar
```

Commit 信息：不提交，本步骤验证部署命令。

- [ ] **步骤 4：在 tmux 启动启用 bridge 的 Vivid-VR serve**

修改文件：无。

测试命令按单卡默认 `single_gpu_fa_compile` 执行，若明确验收双卡则替换为 `dual_gpu_fa_eager_compile`：

```bash
tmux new-session -d -s vividvr_serve_caption_bridge \
  'cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/logs Vivid_Acceptance/result_videos/service_benchmark Vivid_Acceptance/captions/service_sidecars && export PYTHONUNBUFFERED=1 && export PYTHONPATH=python && export SGLANG_VIVIDVR_CONNECTOR_CONTROL_POOL_SIZE=1 && CUDA_VISIBLE_DEVICES=0 /home/zhiheng/sglang/.venv/bin/sglang serve --model-path /home/zhiheng/Vivid-VR/ckpts/CogVideoX1.5-5B --model-id VividVR --pipeline-class-name CogVideoXVividVRControlNetPipeline --component-paths.vividvr /home/zhiheng/Vivid-VR/ckpts/Vivid-VR --attention-backend fa --num-gpus 1 --tp-size 1 --sp-degree 1 --ulysses-degree 1 --ring-degree 1 --enable-torch-compile --dist-timeout 3600 --host 127.0.0.1 --port 31191 --master-port 30191 --scheduler-port 56191 --strict-ports --output-path /home/zhiheng/sglang/Vivid_Acceptance/result_videos/service_benchmark --prompt-file-path /home/zhiheng/Vivid-VR/input/720p/prompt.txt --vividvr-caption-bridge --vividvr-caption-sidecar-url http://127.0.0.1:31200 --vividvr-caption-work-dir /home/zhiheng/sglang/Vivid_Acceptance/captions/service_sidecars --vividvr-caption-sidecar-timeout 1800 2>&1 | tee Vivid_Acceptance/logs/vividvr_serve_caption_bridge_$(date -u +%Y%m%dT%H%M%SZ).log'

curl --noproxy '*' --silent --show-error --fail http://127.0.0.1:31191/health
```

预期结果：

```json
{"status":"ok"}
```

用户查看命令：

```bash
tmux attach -r -t vividvr_serve_caption_bridge
```

Commit 信息：不提交，本步骤验证部署命令。

- [ ] **步骤 5：运行 FlowCut E2E，不传 caption_file_path**

修改文件：无；产物写入 `Vivid_Acceptance`。

测试命令：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python \
  python/sglang/multimodal_gen/tools/run_flowcut_vividvr_service_acceptance.py \
  --base-url http://127.0.0.1:31191 \
  --task-id flowcut-caption-bridge-$(date -u +%Y%m%dT%H%M%SZ) \
  --callback-log /home/zhiheng/sglang/Vivid_Acceptance/logs/flowcut_caption_bridge_callback_$(date -u +%Y%m%dT%H%M%SZ).jsonl \
  --video-input-path /home/zhiheng/Vivid-VR/input/720p/input.mp4 \
  --output-path /home/zhiheng/sglang/Vivid_Acceptance/result_videos/service_caption_bridge \
  --num-inference-steps 20 \
  --num-temporal-process-frames 121 \
  --submit-timeout-s 2400 \
  --poll-timeout-s 2400
```

预期结果：

- 接口返回 `code=0`。
- callback 最终状态为 `succeeded`。
- `Vivid_Acceptance/captions/service_sidecars/<task_id>.txt` 存在。
- `--submit-timeout-s` 需要和 `--poll-timeout-s` 一起放大，因为 bridge 路径的首次提交会同步等待 sidecar 先生成完整 caption sidecar 文件。
- caption 文件非空，行数等于对应 manifest 的 `expected_caption_count`。
- `Vivid_Acceptance/result_videos/service_caption_bridge/<task_id>.mp4` 存在。

Commit 信息：

```bash
git add Vivid_Acceptance/logs Vivid_Acceptance/indicator Vivid_Acceptance/result_videos \
  Vivid_Acceptance/captions/service_sidecars
git commit -m "test(vividvr): accept caption sidecar serve flow"
```

执行提交前必须检查产物大小，避免把不应纳入仓库的大视频或临时日志混入；如果仓库约定不提交 `Vivid_Acceptance` 产物，则只提交 handover 文档中记录的产物路径。

---

## 自检

- 规格覆盖度：计划覆盖了 sidecar 协议、主服务配置、manifest 契约、FlowCut 自动 caption、普通 repair 兼容、文档命令和 tmux E2E。
- 占位符扫描：计划没有使用未定义函数名作为最终状态；每个新增函数都在对应任务中定义了职责和测试。
- 类型一致性：`VividVRCaptionBridgeConfig`、`VividVRCaptionManifest`、`expected_caption_count`、`caption_file_path` 在测试和实现任务中命名一致。
- 范围控制：不修改 `/home/zhiheng/sglang/.venv` 依赖，不把原版 caption 模型导入主推理进程，不改变显式 `caption_file_path` 的现有优先级。
