# Vivid-VR 运行时代码解耦实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 让 `sglang` 的 Vivid-VR caption 能力在运行时代码和服务形态上完全脱离 `/home/zhiheng/Vivid-VR`，仅允许继续复用原版仓库中的静态资源（checkpoint、prompt、输入视频、reference）。

**架构：** 保留现有独立 HTTP sidecar 形态，但 sidecar 只能运行 `sglang` 仓库内代码。主推理继续使用 `/home/zhiheng/sglang/.venv`，caption sidecar 改为使用 `/home/zhiheng/sglang/.venv-vividvr-caption`；CogVLM2 caption 相关 Python 代码全部 vendor 到 `sglang` 的 `runtime/vividvr/captioning/` 下，禁止通过 `sys.path` 注入原版仓库，禁止通过 checkpoint 目录中的 `trust_remote_code` 执行原版 Python 文件。

**技术栈：** Python 3.10、FastAPI/uvicorn、PyTorch、Transformers 4.42.4（caption env）、SGLang、pytest、tmux

---

## 文件结构

- 创建：`python/sglang/multimodal_gen/runtime/vividvr/captioning/__init__.py`
  - repo 内部 caption runtime 包入口。
- 创建：`python/sglang/multimodal_gen/runtime/vividvr/captioning/captioner.py`
  - 本地 captioner 工厂，替代 `VRDiT.captioner`。
- 创建：`python/sglang/multimodal_gen/runtime/vividvr/captioning/cogvlm2.py`
  - 本地 `CogVLM2Captioner` 加载与推理逻辑，替代 `VRDiT.cogvlm2`。
- 创建：`python/sglang/multimodal_gen/runtime/vividvr/captioning/cogvlm2_vendor/__init__.py`
  - vendor 子包入口。
- 创建：`python/sglang/multimodal_gen/runtime/vividvr/captioning/cogvlm2_vendor/configuration_cogvlm.py`
- 创建：`python/sglang/multimodal_gen/runtime/vividvr/captioning/cogvlm2_vendor/modeling_cogvlm.py`
- 创建：`python/sglang/multimodal_gen/runtime/vividvr/captioning/cogvlm2_vendor/model_config.py`
- 创建：`python/sglang/multimodal_gen/runtime/vividvr/captioning/cogvlm2_vendor/util.py`
- 创建：`python/sglang/multimodal_gen/runtime/vividvr/captioning/cogvlm2_vendor/visual.py`
  - vendor 自 checkpoint remote-code 和原版补丁后的实现，确保运行时不从 `/home/zhiheng/Vivid-VR` 或 checkpoint 目录导入 Python 代码。
- 修改：`python/sglang/multimodal_gen/tools/run_vividvr_caption_sidecar.py`
  - 删掉 `sys.path` 注入与 `VRDiT.*` 依赖，改用本地 captioning 包。
- 创建：`python/sglang/multimodal_gen/test/unit/test_vividvr_captioning_loader.py`
  - 校验本地 caption loader 不访问原版路径、不启用 `trust_remote_code=True`。
- 修改：`python/sglang/multimodal_gen/test/unit/test_vividvr_caption_sidecar_tool.py`
  - 校验 sidecar 使用本地 caption 工厂、CLI 不再需要 `--vividvr-root`。
- 创建：`python/requirements-vividvr-caption.txt`
  - repo 内 caption sidecar 专用依赖锁定文件。
- 创建：`python/sglang/multimodal_gen/tools/setup_vividvr_caption_env.sh`
  - 用于创建 `/home/zhiheng/sglang/.venv-vividvr-caption` 的安装脚本。
- 修改：`docs_xzh/run_command/vividvr_default_run_and_serve_commands.md`
  - 更新正式启动命令与 `.venv-vividvr-caption` 用法。
- 修改：`docs_xzh/run_vivid_benchmark.md`
  - 更新 caption benchmark/serve smoke 命令。
- 修改：`AGENTS.md`
  - 把 caption sidecar 的 repo 内独立环境纳入仓库契约。
- 创建：`docs_xzh/hand_over/vividvr_runtime_decoupling_handover_20260623.md`
  - 记录新环境、命令、已知限制和验收结果。

### 任务 1：迁移 caption 运行时代码到 `sglang`

**文件：**
- 创建：`python/sglang/multimodal_gen/runtime/vividvr/captioning/__init__.py`
- 创建：`python/sglang/multimodal_gen/runtime/vividvr/captioning/captioner.py`
- 创建：`python/sglang/multimodal_gen/runtime/vividvr/captioning/cogvlm2.py`
- 创建：`python/sglang/multimodal_gen/runtime/vividvr/captioning/cogvlm2_vendor/__init__.py`
- 创建：`python/sglang/multimodal_gen/runtime/vividvr/captioning/cogvlm2_vendor/configuration_cogvlm.py`
- 创建：`python/sglang/multimodal_gen/runtime/vividvr/captioning/cogvlm2_vendor/modeling_cogvlm.py`
- 创建：`python/sglang/multimodal_gen/runtime/vividvr/captioning/cogvlm2_vendor/model_config.py`
- 创建：`python/sglang/multimodal_gen/runtime/vividvr/captioning/cogvlm2_vendor/util.py`
- 创建：`python/sglang/multimodal_gen/runtime/vividvr/captioning/cogvlm2_vendor/visual.py`
- 测试：`python/sglang/multimodal_gen/test/unit/test_vividvr_captioning_loader.py`

- [ ] **步骤 1：编写失败的 loader 单测**

```python
from pathlib import Path

import pytest

from sglang.multimodal_gen.runtime.vividvr.captioning.cogvlm2 import (
    CogVLM2Captioner,
)


def test_cogvlm2_captioner_uses_local_vendor_code(monkeypatch, tmp_path):
    calls = {}

    class FakeTokenizer:
        pass

    class FakeModel:
        def eval(self):
            return self

    def fake_tokenizer_from_pretrained(model_path, **kwargs):
        calls["tokenizer"] = {"model_path": model_path, **kwargs}
        return FakeTokenizer()

    def fake_model_from_pretrained(model_path, **kwargs):
        calls["model"] = {"model_path": model_path, **kwargs}
        return FakeModel()

    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.vividvr.captioning.cogvlm2.AutoTokenizer.from_pretrained",
        fake_tokenizer_from_pretrained,
    )
    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.vividvr.captioning.cogvlm2.CogVLMVideoForCausalLM.from_pretrained",
        fake_model_from_pretrained,
    )

    CogVLM2Captioner(model_path=str(tmp_path / "ckpt"))

    assert calls["tokenizer"]["trust_remote_code"] is False
    assert "config" in calls["model"]
    assert str(Path("/home/zhiheng/Vivid-VR")) not in __import__("sys").path
```

- [ ] **步骤 2：运行测试验证失败**

运行：`/home/zhiheng/sglang/.venv/bin/pytest python/sglang/multimodal_gen/test/unit/test_vividvr_captioning_loader.py -v`
预期：FAIL，报错 `ModuleNotFoundError: No module named 'sglang.multimodal_gen.runtime.vividvr.captioning'`

- [ ] **步骤 3：添加本地 captioning 包与 vendor 代码**

```python
# python/sglang/multimodal_gen/runtime/vividvr/captioning/cogvlm2.py
from transformers import AutoTokenizer

from sglang.multimodal_gen.runtime.vividvr.captioning.cogvlm2_vendor.configuration_cogvlm import (
    CogVLMConfig,
)
from sglang.multimodal_gen.runtime.vividvr.captioning.cogvlm2_vendor.modeling_cogvlm import (
    CogVLMVideoForCausalLM,
)


class CogVLM2Captioner:
    def __init__(self, model_path, torch_type=torch.bfloat16):
        self.torch_type = torch_type
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=False,
        )
        self.config = CogVLMConfig.from_pretrained(model_path)
        self.model = CogVLMVideoForCausalLM.from_pretrained(
            model_path,
            config=self.config,
            torch_dtype=self.torch_type,
        ).eval()
        self.device = "cpu"
```

```python
# python/sglang/multimodal_gen/runtime/vividvr/captioning/captioner.py
from sglang.multimodal_gen.runtime.vividvr.captioning.cogvlm2 import (
    CogVLM2Captioner,
)


def create_captioner(args):
    if args.caption_backend != "cogvlm2":
        raise ValueError(f"Unsupported caption backend: {args.caption_backend}")
    return CogVLM2Captioner(model_path=args.cogvlm2_ckpt_path)
```

- [ ] **步骤 4：复制并裁剪 vendor 依赖文件**

```bash
cp /home/zhiheng/Vivid-VR/ckpts/cogvlm2-llama3-caption/configuration_cogvlm.py \
  python/sglang/multimodal_gen/runtime/vividvr/captioning/cogvlm2_vendor/
cp /home/zhiheng/Vivid-VR/ckpts/cogvlm2-llama3-caption/model_config.py \
  python/sglang/multimodal_gen/runtime/vividvr/captioning/cogvlm2_vendor/
cp /home/zhiheng/Vivid-VR/ckpts/cogvlm2-llama3-caption/util.py \
  python/sglang/multimodal_gen/runtime/vividvr/captioning/cogvlm2_vendor/
cp /home/zhiheng/Vivid-VR/ckpts/cogvlm2-llama3-caption/visual.py \
  python/sglang/multimodal_gen/runtime/vividvr/captioning/cogvlm2_vendor/
cp /home/zhiheng/Vivid-VR/VRDiT/cogvlm2-llama3-caption/modeling_cogvlm.py \
  python/sglang/multimodal_gen/runtime/vividvr/captioning/cogvlm2_vendor/
```

预期：vendor 包只保留 caption 推理需要的文件；后续 import 全部走 `sglang.multimodal_gen.runtime.vividvr.captioning.cogvlm2_vendor.*`

- [ ] **步骤 5：运行测试验证通过**

运行：`/home/zhiheng/sglang/.venv/bin/pytest python/sglang/multimodal_gen/test/unit/test_vividvr_captioning_loader.py -v`
预期：PASS

- [ ] **步骤 6：Commit**

```bash
git add python/sglang/multimodal_gen/runtime/vividvr/captioning \
        python/sglang/multimodal_gen/test/unit/test_vividvr_captioning_loader.py
git commit -m "feat: vendor vividvr caption runtime into sglang"
```

### 任务 2：重写 sidecar loader，移除原版仓库导入与旧 CLI 契约

**文件：**
- 修改：`python/sglang/multimodal_gen/tools/run_vividvr_caption_sidecar.py`
- 修改：`python/sglang/multimodal_gen/test/unit/test_vividvr_caption_sidecar_tool.py`

- [ ] **步骤 1：编写失败的 sidecar 契约测试**

```python
def test_sidecar_uses_local_captioner_factory(monkeypatch):
    calls = {}

    class FakeCaptioner:
        def to(self, device):
            return self

    def fake_create_captioner(args):
        calls["args"] = args
        return FakeCaptioner()

    monkeypatch.setattr(
        "sglang.multimodal_gen.tools.run_vividvr_caption_sidecar.create_captioner",
        fake_create_captioner,
    )

    state = sidecar_tool.CaptionSidecarState(
        captioner=None,
        device="cpu",
        worker_count=1,
        worker_devices=("cpu",),
        cogvlm2_ckpt_path="/tmp/cogvlm2",
    )
    sidecar_tool._get_serial_captioner(state)

    assert calls["args"].caption_backend == "cogvlm2"
    assert not hasattr(state, "vividvr_root")
```

```python
def test_parse_args_rejects_legacy_vividvr_root_flag():
    with pytest.raises(SystemExit):
        parse_args(["--vividvr-root", "/tmp/legacy"])
```

- [ ] **步骤 2：运行测试验证失败**

运行：`/home/zhiheng/sglang/.venv/bin/pytest python/sglang/multimodal_gen/test/unit/test_vividvr_caption_sidecar_tool.py -k 'local_captioner_factory or legacy_vividvr_root' -v`
预期：FAIL，当前实现仍存在 `vividvr_root` 字段并接受 `--vividvr-root`

- [ ] **步骤 3：改写 sidecar 的 captioner 构造路径**

```python
# run_vividvr_caption_sidecar.py
from sglang.multimodal_gen.runtime.vividvr.captioning.captioner import (
    create_captioner,
)


@dataclass
class CaptionSidecarState:
    captioner: object | None = None
    device: str = "cuda"
    worker_count: int = 1
    worker_devices: tuple[str, ...] = ()
    allow_serial_fallback: bool = True
    executors: tuple[ProcessPoolExecutor, ...] | None = None
    cogvlm2_ckpt_path: str = "/home/zhiheng/Vivid-VR/ckpts/cogvlm2-llama3-caption"


def _build_cogvlm2_captioner(cogvlm2_ckpt_path: str):
    captioner_args = SimpleNamespace(
        caption_backend="cogvlm2",
        cogvlm2_ckpt_path=str(Path(cogvlm2_ckpt_path).expanduser()),
    )
    return create_captioner(captioner_args)
```

- [ ] **步骤 4：删除 `sys.path` 注入和原版 import 逻辑**

```python
# 删除这些函数
def _load_original_captioner_from_paths(...): ...
def _load_original_captioner(...): ...

# 替换调用点
def _get_serial_captioner(state: CaptionSidecarState):
    if state.captioner is None:
        state.captioner = _build_cogvlm2_captioner(state.cogvlm2_ckpt_path)
    return state.captioner

def _init_worker_state(cogvlm2_ckpt_path: str, device: str) -> None:
    global _WORKER_STATE
    _WORKER_STATE = SimpleNamespace(
        captioner=_build_cogvlm2_captioner(cogvlm2_ckpt_path),
        device=device,
    )
```

- [ ] **步骤 5：收紧 CLI 参数**

```python
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run VividVR caption sidecar service.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=31200)
    parser.add_argument(
        "--cogvlm2-ckpt-path",
        default="/home/zhiheng/Vivid-VR/ckpts/cogvlm2-llama3-caption",
    )
    # 不再暴露 --vividvr-root
```

- [ ] **步骤 6：运行测试验证通过**

运行：`/home/zhiheng/sglang/.venv/bin/pytest python/sglang/multimodal_gen/test/unit/test_vividvr_caption_sidecar_tool.py -v`
预期：PASS

- [ ] **步骤 7：Commit**

```bash
git add python/sglang/multimodal_gen/tools/run_vividvr_caption_sidecar.py \
        python/sglang/multimodal_gen/test/unit/test_vividvr_caption_sidecar_tool.py
git commit -m "refactor: decouple vividvr caption sidecar from original repo imports"
```

### 任务 3：建立 repo 内 caption 专用虚拟环境契约

**文件：**
- 创建：`python/requirements-vividvr-caption.txt`
- 创建：`python/sglang/multimodal_gen/tools/setup_vividvr_caption_env.sh`

- [ ] **步骤 1：先写环境契约文件**

```text
# python/requirements-vividvr-caption.txt
torch==2.2.1+cu121
transformers==4.42.4
accelerate==1.11.0
bitsandbytes==0.44.1
decord==0.6.0
opencv-python==4.13.0.92
imageio==2.37.0
imageio-ffmpeg==0.6.0
sentencepiece==0.2.1
tokenizers==0.19.1
einops
requests
fastapi
uvicorn
pydantic
safetensors
torchvision
```

- [ ] **步骤 2：添加环境安装脚本**

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/home/zhiheng/sglang
ENV_PYTHON="${REPO_ROOT}/.venv-vividvr-caption/bin/python"

python3 -m venv "${REPO_ROOT}/.venv-vividvr-caption"
"${ENV_PYTHON}" -m pip install --upgrade pip wheel
"${ENV_PYTHON}" -m pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
  -r "${REPO_ROOT}/python/requirements-vividvr-caption.txt"
```

- [ ] **步骤 3：对脚本做最小校验**

运行：`bash -n /home/zhiheng/sglang/python/sglang/multimodal_gen/tools/setup_vividvr_caption_env.sh`
预期：无输出，退出码 `0`

- [ ] **步骤 4：验证 requirements 中关键版本未漂移**

运行：`rg -n "transformers==4.42.4|torch==2.2.1\\+cu121|bitsandbytes==0.44.1" /home/zhiheng/sglang/python/requirements-vividvr-caption.txt`
预期：输出 3 行，分别命中这 3 个固定版本

- [ ] **步骤 5：Commit**

```bash
git add python/requirements-vividvr-caption.txt \
        python/sglang/multimodal_gen/tools/setup_vividvr_caption_env.sh
git commit -m "build: add vividvr caption sidecar environment contract"
```

### 任务 4：更新运行命令、仓库契约和文档

**文件：**
- 修改：`docs_xzh/run_command/vividvr_default_run_and_serve_commands.md`
- 修改：`docs_xzh/run_vivid_benchmark.md`
- 修改：`AGENTS.md`

- [ ] **步骤 1：先改仓库契约**

```markdown
# AGENTS.md
- 默认使用 `/home/zhiheng/sglang/.venv` 作为主推理环境。
- `Vivid-VR` caption sidecar 是唯一例外，必须使用 `/home/zhiheng/sglang/.venv-vividvr-caption`。
- caption sidecar 运行时只允许 import `sglang` 仓库内的代码；禁止再使用 `/home/zhiheng/Vivid-VR/.venv` 或向 `sys.path` 注入原版仓库。
```

- [ ] **步骤 2：更新 sidecar 启动命令**

```bash
tmux new-session -d -s vividvr_caption_sidecar \
  'cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/logs && export PYTHONPATH=python && CUDA_VISIBLE_DEVICES=0,1 /home/zhiheng/sglang/.venv-vividvr-caption/bin/python python/sglang/multimodal_gen/tools/run_vividvr_caption_sidecar.py --host 127.0.0.1 --port 31200 --parallel-workers 2 --worker-devices cuda:0,cuda:1 2>&1 | tee Vivid_Acceptance/logs/vividvr_caption_sidecar_$(date -u +%Y%m%dT%H%M%SZ).log'
```

- [ ] **步骤 3：更新 benchmark 文档中的 caption 环境说明**

```markdown
- caption sidecar 服务固定使用 `/home/zhiheng/sglang/.venv-vividvr-caption`。
- sidecar 代码和 HTTP 服务均位于 `sglang` 仓库；`/home/zhiheng/Vivid-VR` 仅继续提供 checkpoint、prompt、输入视频、reference 等静态资源。
- 不允许再以 `/home/zhiheng/Vivid-VR/.venv/bin/python` 启动 `run_vividvr_caption_sidecar.py`。
```

- [ ] **步骤 4：检查文档中是否还有旧环境引用**

运行：`rg -n "/home/zhiheng/Vivid-VR/.venv/bin/python|caption sidecar .*原版|--vividvr-root" /home/zhiheng/sglang/AGENTS.md /home/zhiheng/sglang/docs_xzh`
预期：不再命中 sidecar 新命令和 sidecar 新说明中的旧环境表述；原版公平 benchmark 文档允许保留“原版 benchmark 用原版环境”描述

- [ ] **步骤 5：Commit**

```bash
git add AGENTS.md docs_xzh/run_command/vividvr_default_run_and_serve_commands.md \
        docs_xzh/run_vivid_benchmark.md
git commit -m "docs: switch vividvr caption sidecar docs to repo-local env"
```

### 任务 5：补齐解耦回归测试

**文件：**
- 修改：`python/sglang/multimodal_gen/test/unit/test_vividvr_caption_bridge.py`
- 修改：`python/sglang/multimodal_gen/test/unit/test_vividvr_caption_sidecar_tool.py`
- 测试：`python/sglang/multimodal_gen/test/unit/test_vividvr_captioning_loader.py`

- [ ] **步骤 1：给 bridge 增加“保持 HTTP 边界不变”的测试**

```python
def test_request_caption_sidecar_contract_is_stable(monkeypatch, tmp_path):
    output = tmp_path / "caption.txt"
    output.write_text("caption 0\n", encoding="utf-8")

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "caption_file_path": str(output),
                "caption_count": 1,
                "mode": "serial",
                "worker_count": 1,
                "fallback_used": False,
            }

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, json):
            return FakeResponse()
```

- [ ] **步骤 2：运行相关测试验证当前实现覆盖不完整**

运行：`/home/zhiheng/sglang/.venv/bin/pytest python/sglang/multimodal_gen/test/unit/test_vividvr_caption_bridge.py python/sglang/multimodal_gen/test/unit/test_vividvr_caption_sidecar_tool.py python/sglang/multimodal_gen/test/unit/test_vividvr_captioning_loader.py -v`
预期：至少 1 个 FAIL，直到本地 captioning + sidecar 改造全部合入

- [ ] **步骤 3：补齐断言，避免回归到原版代码路径**

```python
assert "/home/zhiheng/Vivid-VR" not in "".join(sys.path)
assert result.mode in {"serial", "parallel"}
assert result.worker_count >= 1
```

- [ ] **步骤 4：运行测试验证通过**

运行：`/home/zhiheng/sglang/.venv/bin/pytest python/sglang/multimodal_gen/test/unit/test_vividvr_caption_bridge.py python/sglang/multimodal_gen/test/unit/test_vividvr_caption_sidecar_tool.py python/sglang/multimodal_gen/test/unit/test_vividvr_captioning_loader.py -v`
预期：PASS

- [ ] **步骤 5：Commit**

```bash
git add python/sglang/multimodal_gen/test/unit/test_vividvr_caption_bridge.py \
        python/sglang/multimodal_gen/test/unit/test_vividvr_caption_sidecar_tool.py \
        python/sglang/multimodal_gen/test/unit/test_vividvr_captioning_loader.py
git commit -m "test: lock vividvr caption decoupling contracts"
```

### 任务 6：做独立 benchmark、`serve` smoke 和交接文档

**文件：**
- 创建：`docs_xzh/hand_over/vividvr_runtime_decoupling_handover_20260623.md`
- 产物：`Vivid_Acceptance/caption_sidecar_benchmark/*.json`
- 产物：`Vivid_Acceptance/captions/service_sidecars/*`
- 产物：`Vivid_Acceptance/logs/*.log`

- [ ] **步骤 1：创建 caption 专用环境**

运行：`bash /home/zhiheng/sglang/python/sglang/multimodal_gen/tools/setup_vividvr_caption_env.sh`
预期：生成 `/home/zhiheng/sglang/.venv-vividvr-caption/bin/python`

- [ ] **步骤 2：在 tmux 中启动 sidecar**

运行：

```bash
tmux new-session -d -s vividvr_caption_sidecar_decouple \
  'cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/logs && export PYTHONPATH=python && CUDA_VISIBLE_DEVICES=0,1 /home/zhiheng/sglang/.venv-vividvr-caption/bin/python python/sglang/multimodal_gen/tools/run_vividvr_caption_sidecar.py --host 127.0.0.1 --port 31206 --parallel-workers 2 --worker-devices cuda:0,cuda:1 2>&1 | tee Vivid_Acceptance/logs/vividvr_caption_sidecar_decouple_$(date -u +%Y%m%dT%H%M%SZ).log'
```

预期：`tmux attach -r -t vividvr_caption_sidecar_decouple` 可看到 sidecar 正常监听

- [ ] **步骤 3：运行独立 caption benchmark**

运行：

```bash
tmux new-session -d -s vividvr_caption_bench_decouple \
  'cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/caption_sidecar_benchmark && export PYTHONPATH=python && /home/zhiheng/sglang/.venv/bin/python python/sglang/multimodal_gen/tools/run_vividvr_caption_sidecar_benchmark.py --video-path /home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.mp4 --baseline-caption-path /home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.txt --sidecar-base-url http://127.0.0.1:31206 --manifest-path /home/zhiheng/sglang/Vivid_Acceptance/caption_sidecar_benchmark/manifest_decouple.json --output-caption-path /home/zhiheng/sglang/Vivid_Acceptance/caption_sidecar_benchmark/generated_decouple.txt --metrics-json-path /home/zhiheng/sglang/Vivid_Acceptance/caption_sidecar_benchmark/metrics_decouple.json 2>&1 | tee Vivid_Acceptance/logs/vividvr_caption_bench_decouple_$(date -u +%Y%m%dT%H%M%SZ).log'
```

预期：`captions_match = true`，`sidecar_mode = "parallel"`，`fallback_used = false`

- [ ] **步骤 4：运行 `serve` smoke 验收**

运行：

```bash
tmux new-session -d -s vividvr_serve_decouple \
  'cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/logs Vivid_Acceptance/captions/service_sidecars Vivid_Acceptance/result_videos/service_benchmark && export PYTHONPATH=python && export SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE=eager_global && export SGLANG_VIVIDVR_CONNECTOR_CONTROL_POOL_SIZE=1 && /home/zhiheng/sglang/.venv/bin/sglang serve --model-path /home/zhiheng/Vivid-VR/ckpts/CogVideoX1.5-5B --model-id VividVR --pipeline-class-name CogVideoXVividVRControlNetPipeline --component-paths.vividvr /home/zhiheng/Vivid-VR/ckpts/Vivid-VR --attention-backend fa --num-gpus 2 --tp-size 1 --sp-degree 2 --ulysses-degree 2 --ring-degree 1 --enable-torch-compile --dist-timeout 3600 --host 127.0.0.1 --port 31196 --master-port 30196 --scheduler-port 56196 --strict-ports --output-path /home/zhiheng/sglang/Vivid_Acceptance/result_videos/service_benchmark --prompt-file-path /home/zhiheng/Vivid-VR/input/720p/prompt.txt --vividvr-caption-bridge --vividvr-caption-sidecar-url http://127.0.0.1:31206 --vividvr-caption-work-dir /home/zhiheng/sglang/Vivid_Acceptance/captions/service_sidecars --vividvr-caption-sidecar-timeout 1800 2>&1 | tee Vivid_Acceptance/logs/vividvr_serve_decouple_$(date -u +%Y%m%dT%H%M%SZ).log'
```

然后提交 1-step smoke：

```bash
TASK_ID=vividvr-bridge-decouple-20260623T1200Z
curl --noproxy '*' -sS -X POST 'http://127.0.0.1:31196/v1/videos/repairs' \
  -H 'Content-Type: application/json' \
  --data-binary @- <<JSON
{
  "model": "VividVR",
  "task_id": "${TASK_ID}",
  "video_input_path": "/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.mp4",
  "num_inference_steps": 1,
  "seed": 42,
  "num_temporal_process_frames": 121,
  "output_path": "/home/zhiheng/sglang/Vivid_Acceptance/result_videos/service_benchmark/${TASK_ID}.mp4",
  "perf_dump_path": "/home/zhiheng/sglang/Vivid_Acceptance/indicator/${TASK_ID}_perf.json"
}
JSON
```

预期：主服务日志出现 `mode=parallel worker_count=2 fallback_used=False`，且 sidecar Python 路径来自 `/home/zhiheng/sglang/.venv-vividvr-caption`

- [ ] **步骤 5：写交接文档**

```markdown
# Vivid-VR 运行时代码解耦交接

- caption sidecar 已迁移到 `sglang` 仓库内代码
- caption sidecar 环境固定为 `/home/zhiheng/sglang/.venv-vividvr-caption`
- `/home/zhiheng/Vivid-VR` 仅继续提供静态资源
- benchmark 与 serve smoke 均已通过，附日志路径、tmux session 名称、已知限制
```

- [ ] **步骤 6：Commit**

```bash
git add docs_xzh/hand_over/vividvr_runtime_decoupling_handover_20260623.md
git commit -m "docs: add vividvr runtime decoupling handover"
```
