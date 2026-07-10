# VividVR CFG Parallel v2 等价加速实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 在不改变 VividVR v2 语义的前提下，先实现并验收 CFG-only 并行闭环，再实现并验收四卡 `CFG=2 x SP=2` 组合加速。

**架构：** 当前 VividVR denoising 在每个 tile 内把负向和正向 prompt 拼成 batch=2，每个 SP rank 都同时计算 uncond/cond。新方案保持 v2/eager_global 的 control context、timestep 编排、scheduler、tiling、merge、decode/postprocess 完全不变，只把 CFG 的两个分支拆到 CFG parallel group 的两个 rank：CFG rank 0 只跑正向 cond，CFG rank 1 只跑负向 uncond，然后用 CFG group all-reduce 合成与串行公式等价的 `noise_pred = uncond + guidance * (cond - uncond)`。第一阶段只开 `CFG=2, SP=1`，第二阶段再开 `CFG=2, SP=2`。

**技术栈：** Python、PyTorch、torch.distributed、SGLang multimodal_gen pipeline stages、VividVR DiT/ControlNet、tmux、mock Flowcut service、`python/sglang/multimodal_gen/runtime/videoedit/compare.py`。

## 2026-07-10 执行状态

本计划已按“先 CFG-only v2 等价闭环，再四卡 `CFG=2 x SP=2`”路线完成实现与 mock 服务验收。

- CFG-only formal：`vividvr-cfg-only-formal-20260709T092339Z`，SSIM `mean=0.9845679069864866`，`min=0.9805834440842233`，`pass_compare=true`。
- 四卡 `cfg_sp` formal：`vividvr-cfg2-sp2-formal-20260710T014318Z`，`total_runtime_seconds=356.31927194446325`，`model_inference_runtime_seconds=194.2424`，SSIM `mean=0.984805283930388`，`min=0.9802324220300491`，`pass_compare=true`。
- 四卡 formal 验收 summary：`/home/zhiheng/sglang/Vivid_Acceptance/indicator/service_benchmark/vividvr-cfg2-sp2-formal-20260710T014318Z_acceptance_summary.json`。
- 详细交接记录：`/home/zhiheng/sglang/docs_xzh/hand_over/vividvr_cfg_parallel_v2_equivalence_handover_20260709.md`。

---

## 硬约束

1. 本计划所有实现都以 v2 语义等价为硬门槛。不得为了速度回退到历史 fast native SP、`deferred_global` 默认语义、local-only control context，或改变 prompt/caption、VAE tiling、temporal clip split/merge、scheduler step、decode/postprocess。
2. 正式质量验收必须按 `/home/zhiheng/sglang/docs_xzh/run_command/mock_test.md` 的 mock 服务链路执行，包括 moto S3、callback receiver、caption sidecar mock、Flowcut bridge service、服务任务提交、progress 查询、callback 检查、S3 下载结果。
3. 服务启动后的第一次完整推理只作为 torch compile warmup，不记录为正式性能结果；第二次完整推理才记录 `total_runtime_seconds` 和 `model_inference_runtime_seconds`。
4. 正式质量对比使用上一轮四卡 reference 视频：

```bash
export VIVIDVR_CFG_REFERENCE_VIDEO=/home/zhiheng/sglang/Vivid_Acceptance/result_videos/service_benchmark/downloads/quad-test-video-long-960x720-130f-run2-20260708T060202Z.bridge-downloaded.mp4
```

执行前必须确认该文件存在。如果文件不存在，停止执行并让用户提供“之前的四卡 reference 视频”的准确路径，不允许临时改用单卡或双卡结果作为 reference。

5. 本计划按严格 SSIM 口径执行：正式 compare 报告必须满足 `summary.ssim_mean > 0.98` 且 `summary.ssim_min >= 0.98`，并且 `summary.pass_compare == true`。`compare.py` 命令用 `--min-ssim 0.98 --max-failed-frame-ratio 0` 保证每个被比较帧都不低于 0.98。
6. 两卡不能同时使用 `SP=2` 和 `CFG=2`。SGLang 的有效 GPU 需求是 `dp * tp * cfg * sp`，所以两卡阶段只能做 CFG-only 或 SP-only；四卡阶段才允许 `CFG=2 x SP=2`。
7. 必须保留纯 SP 并行方案，不允许用 CFG parallel 实现直接替换或删除 SP-only 路径。VividVR 并行组合必须由服务启动参数显式决定；请求进入服务后不能动态切换并行组合，因为 process group 和模型并行拓扑已经在服务启动阶段初始化。

## 文件结构

- 修改：`python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py`
  - 增加 VividVR denoising stage 的 CFG parallel 声明、rank 分支选择、CFG group 等价合成。
  - 保持原串行 batch=2 代码路径作为默认路径，未显式 `--enable-cfg-parallel` 时行为不变。

- 修改：`python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py`
  - 在 `vividvr_debug` 中记录 `cfg_parallel_enabled`、`cfg_rank`、`cfg_branch`、`cfg_world_size`、`cfg_combine_formula`，用于服务 perf dump 和回归排查。

- 创建：`test/srt/multimodal_gen/test_vividvr_cfg_parallel_stage.py`
  - 单元测试 CFG branch 选择、串行公式、CFG all-reduce 公式、非 CFG parallel 回退路径。

- 修改：`python/sglang/multimodal_gen/runtime/server_args.py`
  - 增加 VividVR 专用启动参数 `--vividvr-parallel-mode`，默认 `auto`，可选 `single`、`sp`、`cfg`、`cfg_sp`。
  - 参数只表达和校验人类意图，不替代底层现有并行参数；实际 process group 仍由 `--sp-degree`、`--ulysses-degree`、`--ring-degree`、`--enable-cfg-parallel`、`--num-gpus` 初始化。
  - 保证 `sp` 模式不启用 CFG，`cfg` 模式不启用 SP，`cfg_sp` 模式同时启用 CFG 与 SP。

- 修改：`docs_xzh/run_command/mock_test.md`
  - 增补 CFG-only mock service 和 `CFG=2 x SP=2` mock service 的启动命令、warmup/formal 两次请求、compare 命令和验收阈值。

- 创建：`docs_xzh/hand_over/vividvr_cfg_parallel_v2_equivalence_handover_20260709.md`
  - 实现完成后记录 commit、服务验收日志、perf dump、compare report、warmup/formal 任务 ID、风险项。

## 实现原则

并行模式选择固定为服务启动级参数：

```text
--vividvr-parallel-mode auto
--vividvr-parallel-mode single
--vividvr-parallel-mode sp
--vividvr-parallel-mode cfg
--vividvr-parallel-mode cfg_sp
```

参数语义：

```text
auto: 保持历史兼容，由现有 --enable-cfg-parallel 和 --sp-degree 推导有效模式
single: 要求 --enable-cfg-parallel=false 且 --sp-degree=1
sp: 要求 --enable-cfg-parallel=false 且 --sp-degree>1
cfg: 要求 --enable-cfg-parallel=true 且 --sp-degree=1
cfg_sp: 要求 --enable-cfg-parallel=true 且 --sp-degree>1
```

这不是请求级参数。原因是 CFG group 和 SP group 在服务启动时创建，单个请求不能在同一服务进程里安全切换并行拓扑。

CFG parallel 的数学等价式固定为：

```python
serial = uncond + guidance_scale * (cond - uncond)
serial = guidance_scale * cond + (1.0 - guidance_scale) * uncond
```

CFG rank 分配固定为：

```python
cfg_rank == 0 -> positive / cond branch
cfg_rank == 1 -> negative / uncond branch
```

CFG parallel 合成固定为：

```python
if cfg_rank == 0:
    partial = guidance_scale * noise_pred_cond
else:
    partial = (1.0 - guidance_scale) * noise_pred_uncond
noise_pred = cfg_model_parallel_all_reduce(partial)
```

这个公式是 v2 串行 batch=2 的代数重排，不允许替换成近似公式、异步近似、不同 guidance scale schedule，或只同步部分 tile。

---

## 任务 0：增加 VividVR 并行模式参数守护

**文件：**
- 修改：`python/sglang/multimodal_gen/runtime/server_args.py`
- 修改：`python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py`
- 测试：`test/srt/multimodal_gen/test_vividvr_cfg_parallel_stage.py`

- [ ] **步骤 1：编写失败测试，固定 mode 与底层参数的关系**

在 `test/srt/multimodal_gen/test_vividvr_cfg_parallel_stage.py` 创建或追加：

```python
from types import SimpleNamespace

import pytest

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.vividvr import (
    _resolve_vividvr_parallel_mode,
)


@pytest.mark.parametrize(
    ("mode", "enable_cfg_parallel", "sp_degree", "expected"),
    [
        ("auto", False, 1, "single"),
        ("auto", False, 2, "sp"),
        ("auto", True, 1, "cfg"),
        ("auto", True, 2, "cfg_sp"),
        ("single", False, 1, "single"),
        ("sp", False, 2, "sp"),
        ("cfg", True, 1, "cfg"),
        ("cfg_sp", True, 2, "cfg_sp"),
    ],
)
def test_vividvr_parallel_mode_resolves_valid_combinations(
    mode,
    enable_cfg_parallel,
    sp_degree,
    expected,
):
    server_args = SimpleNamespace(
        vividvr_parallel_mode=mode,
        enable_cfg_parallel=enable_cfg_parallel,
        sp_degree=sp_degree,
    )
    assert _resolve_vividvr_parallel_mode(server_args) == expected


@pytest.mark.parametrize(
    ("mode", "enable_cfg_parallel", "sp_degree"),
    [
        ("single", True, 1),
        ("single", False, 2),
        ("sp", True, 2),
        ("sp", False, 1),
        ("cfg", False, 1),
        ("cfg", True, 2),
        ("cfg_sp", False, 2),
        ("cfg_sp", True, 1),
    ],
)
def test_vividvr_parallel_mode_rejects_mismatched_flags(
    mode,
    enable_cfg_parallel,
    sp_degree,
):
    server_args = SimpleNamespace(
        vividvr_parallel_mode=mode,
        enable_cfg_parallel=enable_cfg_parallel,
        sp_degree=sp_degree,
    )
    with pytest.raises(ValueError, match="vividvr_parallel_mode"):
        _resolve_vividvr_parallel_mode(server_args)
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
cd /home/zhiheng/sglang
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  test/srt/multimodal_gen/test_vividvr_cfg_parallel_stage.py::test_vividvr_parallel_mode_resolves_valid_combinations \
  test/srt/multimodal_gen/test_vividvr_cfg_parallel_stage.py::test_vividvr_parallel_mode_rejects_mismatched_flags \
  -q
```

预期：FAIL，错误为 `_resolve_vividvr_parallel_mode` 不存在。

- [ ] **步骤 3：在 server args 中增加启动参数**

在 `python/sglang/multimodal_gen/runtime/server_args.py` 的 parser 定义中增加：

```python
parser.add_argument(
    "--vividvr-parallel-mode",
    type=str,
    choices=["auto", "single", "sp", "cfg", "cfg_sp"],
    default="auto",
    help=(
        "VividVR parallel mode guard. auto preserves existing behavior. "
        "single requires no CFG parallel and sp-degree=1; sp requires no CFG "
        "parallel and sp-degree>1; cfg requires CFG parallel and sp-degree=1; "
        "cfg_sp requires CFG parallel and sp-degree>1."
    ),
)
```

如果当前 `ServerArgs` 使用 dataclass 字段而不是直接 parser 字段，同时补充：

```python
vividvr_parallel_mode: str = "auto"
```

- [ ] **步骤 4：增加 VividVR mode 解析 helper**

在 `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py` 顶层 helper 区域加入：

```python
def _resolve_vividvr_parallel_mode(server_args: ServerArgs) -> str:
    requested = getattr(server_args, "vividvr_parallel_mode", "auto")
    enable_cfg_parallel = bool(getattr(server_args, "enable_cfg_parallel", False))
    sp_degree = int(getattr(server_args, "sp_degree", 1))

    if requested == "auto":
        if enable_cfg_parallel and sp_degree > 1:
            return "cfg_sp"
        if enable_cfg_parallel:
            return "cfg"
        if sp_degree > 1:
            return "sp"
        return "single"

    expected = {
        "single": (False, False),
        "sp": (False, True),
        "cfg": (True, False),
        "cfg_sp": (True, True),
    }[requested]
    actual = (enable_cfg_parallel, sp_degree > 1)
    if actual != expected:
        raise ValueError(
            "Invalid vividvr_parallel_mode configuration: "
            f"vividvr_parallel_mode={requested!r}, "
            f"enable_cfg_parallel={enable_cfg_parallel}, sp_degree={sp_degree}"
        )
    return requested
```

- [ ] **步骤 5：在 VividVR 入口记录并校验 effective mode**

在 `VividVRInputValidationStage.forward` 中，`params._validate_with_pipeline_config(...)` 之后加入：

```python
        debug = batch.extra.setdefault("vividvr_debug", {})
        debug["vividvr_parallel_mode"] = _resolve_vividvr_parallel_mode(server_args)
```

这样服务启动参数和 runtime perf dump 都能明确区分 `sp`、`cfg`、`cfg_sp`，且不需要删除任何旧 SP 路径。

- [ ] **步骤 6：运行测试验证通过**

运行：

```bash
cd /home/zhiheng/sglang
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  test/srt/multimodal_gen/test_vividvr_cfg_parallel_stage.py \
  -q
```

预期：mode 解析测试通过；如果其他测试尚未写入，至少本任务新增的两个测试通过。

- [ ] **步骤 7：Commit**

```bash
git add python/sglang/multimodal_gen/runtime/server_args.py \
  python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py \
  test/srt/multimodal_gen/test_vividvr_cfg_parallel_stage.py
git commit -m "feat: guard vividvr parallel mode selection"
```

---

## 任务 1：为 VividVR CFG 公式写失败测试

**文件：**
- 创建：`test/srt/multimodal_gen/test_vividvr_cfg_parallel_stage.py`

- [ ] **步骤 1：创建测试文件，覆盖 CFG 等价公式**

使用 `apply_patch` 创建以下文件：

```python
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch


def _serial_cfg(cond: torch.Tensor, uncond: torch.Tensor, scale: float) -> torch.Tensor:
    return uncond + scale * (cond - uncond)


def _cfg_rank_partial(
    *,
    cfg_rank: int,
    cond: torch.Tensor,
    uncond: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    if cfg_rank == 0:
        return scale * cond
    if cfg_rank == 1:
        return (1.0 - scale) * uncond
    raise AssertionError(f"unexpected cfg_rank={cfg_rank}")


def test_vividvr_cfg_parallel_formula_matches_serial_batch2_formula():
    cond = torch.tensor(
        [[[[[0.25, -0.50], [1.25, 2.00]]]]],
        dtype=torch.float32,
    )
    uncond = torch.tensor(
        [[[[[-1.00, 0.75], [0.50, -0.25]]]]],
        dtype=torch.float32,
    )
    scale = 7.5

    serial = _serial_cfg(cond, uncond, scale)
    cfg_parallel = _cfg_rank_partial(
        cfg_rank=0,
        cond=cond,
        uncond=uncond,
        scale=scale,
    ) + _cfg_rank_partial(
        cfg_rank=1,
        cond=cond,
        uncond=uncond,
        scale=scale,
    )

    torch.testing.assert_close(cfg_parallel, serial, rtol=0.0, atol=0.0)
```

- [ ] **步骤 2：运行测试验证当前测试文件可执行**

运行：

```bash
cd /home/zhiheng/sglang
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  test/srt/multimodal_gen/test_vividvr_cfg_parallel_stage.py::test_vividvr_cfg_parallel_formula_matches_serial_batch2_formula \
  -q
```

预期：`1 passed`。这一步不是验证实现，只是固定数学等价公式，防止后续代码改动把 rank 分配或符号写反。

- [ ] **步骤 3：Commit**

```bash
git add test/srt/multimodal_gen/test_vividvr_cfg_parallel_stage.py
git commit -m "test: lock vividvr cfg parallel formula"
```

---

## 任务 2：让 VividVR denoising stage 支持 CFG_PARALLEL 调度

**文件：**
- 修改：`python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py`
- 测试：`test/srt/multimodal_gen/test_vividvr_cfg_parallel_stage.py`

- [ ] **步骤 1：补充失败测试，要求 VividVR stage 在启用 CFG parallel 时声明 `CFG_PARALLEL`**

在 `test/srt/multimodal_gen/test_vividvr_cfg_parallel_stage.py` 追加：

```python
from unittest.mock import patch

from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.vividvr import (
    VividVRDenoisingStage,
)


class _FakeServerArgs:
    enable_cfg_parallel = True


def test_vividvr_denoising_stage_declares_cfg_parallel_when_enabled():
    stage = VividVRDenoisingStage.__new__(VividVRDenoisingStage)
    with patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.vividvr.get_global_server_args",
        return_value=_FakeServerArgs(),
    ):
        assert stage.parallelism_type is StageParallelismType.CFG_PARALLEL
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
cd /home/zhiheng/sglang
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  test/srt/multimodal_gen/test_vividvr_cfg_parallel_stage.py::test_vividvr_denoising_stage_declares_cfg_parallel_when_enabled \
  -q
```

预期：FAIL，错误表现为 `VividVRDenoisingStage` 没有返回 `StageParallelismType.CFG_PARALLEL`，或缺少导入。

- [ ] **步骤 3：修改 VividVR imports**

在 `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py` 中，把 distributed imports 改为包含 CFG rank：

```python
from sglang.multimodal_gen.runtime.distributed import (
    cfg_model_parallel_all_reduce,
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
    get_local_torch_device,
    get_sp_group,
    get_world_group,
)
```

把 stage base import 改为：

```python
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    PipelineStage,
    StageParallelismType,
)
```

增加全局 server args import：

```python
from sglang.multimodal_gen.runtime.server_args import (
    ServerArgs,
    get_global_server_args,
)
```

- [ ] **步骤 4：在 `VividVRDenoisingStage` 内增加 parallelism_type**

在 `VividVRDenoisingStage` class 的 `__init__` 之后加入：

```python
    @property
    def parallelism_type(self) -> StageParallelismType:
        if get_global_server_args().enable_cfg_parallel:
            return StageParallelismType.CFG_PARALLEL
        return StageParallelismType.REPLICATED
```

如果 `VividVRDenoisingStage` 当前 class 定义位置与 `__init__` 不相邻，只把 property 放在 class 内、`prepare_denoising_state` 前，避免移动现有逻辑。

- [ ] **步骤 5：运行测试验证通过**

运行：

```bash
cd /home/zhiheng/sglang
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  test/srt/multimodal_gen/test_vividvr_cfg_parallel_stage.py \
  -q
```

预期：`2 passed`。

- [ ] **步骤 6：Commit**

```bash
git add python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py \
  test/srt/multimodal_gen/test_vividvr_cfg_parallel_stage.py
git commit -m "feat: route vividvr denoising through cfg parallel stage"
```

---

## 任务 3：实现 CFG branch 选择辅助函数

**文件：**
- 修改：`python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py`
- 测试：`test/srt/multimodal_gen/test_vividvr_cfg_parallel_stage.py`

- [ ] **步骤 1：补充失败测试，固定 prompt branch 和 batch 行为**

在测试文件追加：

```python
def test_vividvr_cfg_parallel_branch_uses_single_batch_prompt_slice():
    prompt = torch.full((1, 4, 3), 2.0)
    negative = torch.full((1, 4, 3), -3.0)

    positive_embeds, positive_branch = VividVRDenoisingStage._select_cfg_prompt_embeds(
        prompt_embeds=prompt,
        negative_prompt_embeds=negative,
        prompt_slice=slice(0, 1),
        do_classifier_free_guidance=True,
        enable_cfg_parallel=True,
        cfg_rank=0,
    )
    negative_embeds, negative_branch = VividVRDenoisingStage._select_cfg_prompt_embeds(
        prompt_embeds=prompt,
        negative_prompt_embeds=negative,
        prompt_slice=slice(0, 1),
        do_classifier_free_guidance=True,
        enable_cfg_parallel=True,
        cfg_rank=1,
    )
    serial_embeds, serial_branch = VividVRDenoisingStage._select_cfg_prompt_embeds(
        prompt_embeds=prompt,
        negative_prompt_embeds=negative,
        prompt_slice=slice(0, 1),
        do_classifier_free_guidance=True,
        enable_cfg_parallel=False,
        cfg_rank=0,
    )

    assert positive_branch == "positive"
    assert negative_branch == "negative"
    assert serial_branch == "serial_batch2"
    torch.testing.assert_close(positive_embeds, prompt)
    torch.testing.assert_close(negative_embeds, negative)
    torch.testing.assert_close(serial_embeds, torch.cat([negative, prompt], dim=0))
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
cd /home/zhiheng/sglang
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  test/srt/multimodal_gen/test_vividvr_cfg_parallel_stage.py::test_vividvr_cfg_parallel_branch_uses_single_batch_prompt_slice \
  -q
```

预期：FAIL，错误为 `_select_cfg_prompt_embeds` 不存在。

- [ ] **步骤 3：在 VividVRDenoisingStage 中增加 prompt branch helper**

加入以下 staticmethod：

```python
    @staticmethod
    def _select_cfg_prompt_embeds(
        *,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        prompt_slice: slice,
        do_classifier_free_guidance: bool,
        enable_cfg_parallel: bool,
        cfg_rank: int,
    ) -> tuple[torch.Tensor, str]:
        tile_prompt_embeds = prompt_embeds[prompt_slice]
        if not do_classifier_free_guidance:
            return tile_prompt_embeds, "none"
        if negative_prompt_embeds is None:
            raise ValueError("VividVR negative prompt embeds are required for CFG")
        tile_negative_prompt_embeds = negative_prompt_embeds[prompt_slice]
        if enable_cfg_parallel:
            if cfg_rank == 0:
                return tile_prompt_embeds, "positive"
            if cfg_rank == 1:
                return tile_negative_prompt_embeds, "negative"
            raise ValueError(f"VividVR CFG parallel requires cfg_rank 0 or 1, got {cfg_rank}")
        return torch.cat([tile_negative_prompt_embeds, tile_prompt_embeds], dim=0), "serial_batch2"
```

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
cd /home/zhiheng/sglang
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  test/srt/multimodal_gen/test_vividvr_cfg_parallel_stage.py \
  -q
```

预期：`3 passed`。

- [ ] **步骤 5：Commit**

```bash
git add python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py \
  test/srt/multimodal_gen/test_vividvr_cfg_parallel_stage.py
git commit -m "feat: add vividvr cfg branch prompt selection"
```

---

## 任务 4：实现 CFG parallel noise 合成辅助函数

**文件：**
- 修改：`python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py`
- 测试：`test/srt/multimodal_gen/test_vividvr_cfg_parallel_stage.py`

- [ ] **步骤 1：补充失败测试，固定串行 chunk 顺序和 parallel partial 公式**

在测试文件追加：

```python
def test_vividvr_cfg_combine_serial_uses_negative_then_positive_chunk_order():
    negative = torch.tensor([[[[[1.0, 2.0]]]]])
    positive = torch.tensor([[[[[3.0, 5.0]]]]])
    batch2 = torch.cat([negative, positive], dim=0)

    result = VividVRDenoisingStage._combine_cfg_noise_pred(
        noise_pred=batch2,
        do_classifier_free_guidance=True,
        enable_cfg_parallel=False,
        cfg_rank=0,
        guidance_scale=7.5,
    )

    expected = negative + 7.5 * (positive - negative)
    torch.testing.assert_close(result, expected, rtol=0.0, atol=0.0)


def test_vividvr_cfg_combine_parallel_returns_rank_partial_before_all_reduce():
    positive = torch.tensor([[[[[3.0, 5.0]]]]])
    negative = torch.tensor([[[[[1.0, 2.0]]]]])

    pos_partial = VividVRDenoisingStage._combine_cfg_noise_pred(
        noise_pred=positive,
        do_classifier_free_guidance=True,
        enable_cfg_parallel=True,
        cfg_rank=0,
        guidance_scale=7.5,
        all_reduce_fn=lambda x: x,
    )
    neg_partial = VividVRDenoisingStage._combine_cfg_noise_pred(
        noise_pred=negative,
        do_classifier_free_guidance=True,
        enable_cfg_parallel=True,
        cfg_rank=1,
        guidance_scale=7.5,
        all_reduce_fn=lambda x: x,
    )

    expected = negative + 7.5 * (positive - negative)
    torch.testing.assert_close(pos_partial + neg_partial, expected, rtol=0.0, atol=0.0)
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
cd /home/zhiheng/sglang
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  test/srt/multimodal_gen/test_vividvr_cfg_parallel_stage.py::test_vividvr_cfg_combine_serial_uses_negative_then_positive_chunk_order \
  test/srt/multimodal_gen/test_vividvr_cfg_parallel_stage.py::test_vividvr_cfg_combine_parallel_returns_rank_partial_before_all_reduce \
  -q
```

预期：FAIL，错误为 `_combine_cfg_noise_pred` 不存在。

- [ ] **步骤 3：在 VividVRDenoisingStage 中增加 noise combine helper**

加入以下 staticmethod：

```python
    @staticmethod
    def _combine_cfg_noise_pred(
        *,
        noise_pred: torch.Tensor,
        do_classifier_free_guidance: bool,
        enable_cfg_parallel: bool,
        cfg_rank: int,
        guidance_scale: float,
        all_reduce_fn=cfg_model_parallel_all_reduce,
    ) -> torch.Tensor:
        noise_pred = noise_pred.float()
        if not do_classifier_free_guidance:
            return noise_pred
        if enable_cfg_parallel:
            if cfg_rank == 0:
                partial = float(guidance_scale) * noise_pred
            elif cfg_rank == 1:
                partial = (1.0 - float(guidance_scale)) * noise_pred
            else:
                raise ValueError(f"VividVR CFG parallel requires cfg_rank 0 or 1, got {cfg_rank}")
            return all_reduce_fn(partial)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        return noise_pred_uncond + float(guidance_scale) * (
            noise_pred_text - noise_pred_uncond
        )
```

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
cd /home/zhiheng/sglang
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  test/srt/multimodal_gen/test_vividvr_cfg_parallel_stage.py \
  -q
```

预期：`5 passed`。

- [ ] **步骤 5：Commit**

```bash
git add python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py \
  test/srt/multimodal_gen/test_vividvr_cfg_parallel_stage.py
git commit -m "feat: add vividvr cfg parallel noise combine"
```

---

## 任务 5：把 helper 接入 VividVR tile denoising

**文件：**
- 修改：`python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py`
- 测试：`test/srt/multimodal_gen/test_vividvr_cfg_parallel_stage.py`

- [ ] **步骤 1：补充失败测试，固定 latent/control batch 选择**

在测试文件追加：

```python
def test_vividvr_cfg_parallel_model_inputs_keep_batch_one_per_rank():
    latents = torch.randn(1, 2, 3, 4, 4)
    control = torch.randn(1, 2, 3, 4, 4)

    serial_latents, serial_control = VividVRDenoisingStage._prepare_cfg_model_inputs(
        tile_latents=latents,
        tile_control_latents=control,
        do_classifier_free_guidance=True,
        enable_cfg_parallel=False,
    )
    parallel_latents, parallel_control = VividVRDenoisingStage._prepare_cfg_model_inputs(
        tile_latents=latents,
        tile_control_latents=control,
        do_classifier_free_guidance=True,
        enable_cfg_parallel=True,
    )

    assert serial_latents.shape[0] == 2
    assert serial_control.shape[0] == 2
    assert parallel_latents.shape[0] == 1
    assert parallel_control.shape[0] == 1
    torch.testing.assert_close(parallel_latents, latents)
    torch.testing.assert_close(parallel_control, control)
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
cd /home/zhiheng/sglang
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  test/srt/multimodal_gen/test_vividvr_cfg_parallel_stage.py::test_vividvr_cfg_parallel_model_inputs_keep_batch_one_per_rank \
  -q
```

预期：FAIL，错误为 `_prepare_cfg_model_inputs` 不存在。

- [ ] **步骤 3：新增 model input helper**

在 `VividVRDenoisingStage` 中加入：

```python
    @staticmethod
    def _prepare_cfg_model_inputs(
        *,
        tile_latents: torch.Tensor,
        tile_control_latents: torch.Tensor,
        do_classifier_free_guidance: bool,
        enable_cfg_parallel: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if do_classifier_free_guidance and not enable_cfg_parallel:
            return torch.cat([tile_latents] * 2), torch.cat([tile_control_latents] * 2)
        return tile_latents, tile_control_latents
```

- [ ] **步骤 4：修改 `run_denoising_step`**

在 `run_denoising_step` 中读取 CFG parallel 状态：

```python
        enable_cfg_parallel = bool(server_args.enable_cfg_parallel)
        cfg_rank = get_classifier_free_guidance_rank() if enable_cfg_parallel else 0
        cfg_world_size = (
            get_classifier_free_guidance_world_size() if enable_cfg_parallel else 1
        )
        if enable_cfg_parallel and cfg_world_size != 2:
            raise ValueError(
                f"VividVR CFG parallel expects cfg_world_size=2, got {cfg_world_size}"
            )
        debug.update(
            {
                "cfg_parallel_enabled": enable_cfg_parallel,
                "cfg_world_size": int(cfg_world_size),
                "cfg_rank": int(cfg_rank),
                "cfg_combine_formula": (
                    "guidance*cond + (1-guidance)*uncond"
                    if enable_cfg_parallel
                    else "uncond + guidance*(cond-uncond)"
                ),
            }
        )
```

用 helper 替换当前 batch=2 拼接：

```python
            latent_model_input, control_model_input = self._prepare_cfg_model_inputs(
                tile_latents=tile_latents,
                tile_control_latents=tile_control_latents,
                do_classifier_free_guidance=do_classifier_free_guidance,
                enable_cfg_parallel=enable_cfg_parallel,
            )
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input,
                timestep,
            )
```

用 helper 替换 prompt 拼接：

```python
            tile_prompt_embeds, cfg_branch = self._select_cfg_prompt_embeds(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                prompt_slice=prompt_slice,
                do_classifier_free_guidance=do_classifier_free_guidance,
                enable_cfg_parallel=enable_cfg_parallel,
                cfg_rank=cfg_rank,
            )
            debug["cfg_branch"] = cfg_branch
            batch.is_cfg_negative = cfg_branch == "negative"
```

修正 `timestep_expand` 的 batch 大小，必须在 model input shape 确定后执行：

```python
            timestep_expand = timestep.expand(latent_model_input.shape[0])
```

用 combine helper 替换当前 chunk 逻辑：

```python
            noise_pred = self._combine_cfg_noise_pred(
                noise_pred=noise_pred,
                do_classifier_free_guidance=do_classifier_free_guidance,
                enable_cfg_parallel=enable_cfg_parallel,
                cfg_rank=cfg_rank,
                guidance_scale=guidance_scale,
            )
```

- [ ] **步骤 5：检查不要破坏 scheduler 输入 shape**

确认 helper 接入后，传给 scheduler 的 `noise_pred` batch size 与 `tile_latents` batch size 一致：

```python
            if noise_pred.shape[0] != tile_latents.shape[0]:
                raise RuntimeError(
                    "VividVR denoising produced mismatched batch size: "
                    f"noise_pred={tuple(noise_pred.shape)}, "
                    f"tile_latents={tuple(tile_latents.shape)}"
                )
```

这段检查放在调用 scheduler step 之前。串行 CFG 路径 chunk 后是 batch=1；CFG parallel all-reduce 后也是 batch=1。

- [ ] **步骤 6：运行单元测试**

运行：

```bash
cd /home/zhiheng/sglang
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  test/srt/multimodal_gen/test_vividvr_cfg_parallel_stage.py \
  -q
```

预期：全部通过。

- [ ] **步骤 7：Commit**

```bash
git add python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py \
  test/srt/multimodal_gen/test_vividvr_cfg_parallel_stage.py
git commit -m "feat: run vividvr cfg branches on cfg parallel ranks"
```

---

## 任务 6：CFG-only v2 等价闭环验收

**文件：**
- 修改：`docs_xzh/run_command/mock_test.md`
- 产物：`/home/zhiheng/sglang/Vivid_Acceptance/logs`
- 产物：`/home/zhiheng/sglang/Vivid_Acceptance/indicator/service_benchmark`
- 产物：`/home/zhiheng/sglang/Vivid_Acceptance/result_videos/service_benchmark`

- [ ] **步骤 1：准备 mock test 环境变量**

运行：

```bash
cd /home/zhiheng/sglang
export PYTHONPATH=python
export NO_PROXY=127.0.0.1,localhost
export LOG_DIR=/home/zhiheng/sglang/Vivid_Acceptance/logs
export OUTPUT_DIR=/home/zhiheng/sglang/Vivid_Acceptance/result_videos/service_benchmark
export INDICATOR_DIR=/home/zhiheng/sglang/Vivid_Acceptance/indicator/service_benchmark
export CAPTION_SIDECAR_DIR=/home/zhiheng/sglang/Vivid_Acceptance/captions/service_sidecars
export INPUT_VIDEO_130F=/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.mp4
export CAPTION_FILE=/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.txt
export PROMPT_FILE=/home/zhiheng/Vivid-VR/input/720p/prompt.txt
export MOTO_S3_ENDPOINT=127.0.0.1:4566
export MOTO_S3_BUCKET=flowcut
export MOTO_S3_ACCESS_KEY=test
export MOTO_S3_SECRET_KEY=test
export BRIDGE_BASE_URL=http://127.0.0.1:31231
export CALLBACK_BASE_URL=http://127.0.0.1:39090
export VIVIDVR_CFG_REFERENCE_VIDEO=/home/zhiheng/sglang/Vivid_Acceptance/result_videos/service_benchmark/downloads/quad-test-video-long-960x720-130f-run2-20260708T060202Z.bridge-downloaded.mp4
mkdir -p "$LOG_DIR" "$OUTPUT_DIR" "$INDICATOR_DIR" "$CAPTION_SIDECAR_DIR" "$OUTPUT_DIR/downloads"
test -f "$INPUT_VIDEO_130F"
test -f "$CAPTION_FILE"
test -f "$VIVIDVR_CFG_REFERENCE_VIDEO"
```

预期：所有 `test -f` 返回 0。

- [ ] **步骤 2：按 mock_test.md 启动依赖服务**

启动本地 S3 模拟服务：

```bash
tmux new-session -d -s vividvr_moto_s3 \
  'cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/logs && /home/zhiheng/sglang/.venv/bin/moto_server -H 127.0.0.1 -p 4566 2>&1 | tee Vivid_Acceptance/logs/vividvr_moto_s3_$(date -u +%Y%m%dT%H%M%SZ).log'
```

创建 bucket：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python - <<'PY'
import boto3

s3 = boto3.client(
    "s3",
    endpoint_url="http://127.0.0.1:4566",
    aws_access_key_id="test",
    aws_secret_access_key="test",
    region_name="us-east-1",
)
try:
    s3.create_bucket(Bucket="flowcut")
except s3.exceptions.BucketAlreadyOwnedByYou:
    pass
print([b["Name"] for b in s3.list_buckets()["Buckets"]])
PY
```

启动 callback receiver：

```bash
tmux new-session -d -s vividvr_flowcut_callback_receiver \
  'cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/logs && export CALLBACK_LOG=Vivid_Acceptance/logs/mock_callback_$(date -u +%Y%m%dT%H%M%SZ).jsonl && /home/zhiheng/sglang/.venv/bin/python - <<'"'"'PY'"'"'
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

log_path = os.environ["CALLBACK_LOG"]

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception as exc:
            payload = {"invalid_json": str(exc), "raw": body.decode("utf-8", "replace")}
        with open(log_path, "a", encoding="utf-8") as fout:
            fout.write(json.dumps(payload, ensure_ascii=False))
            fout.write("\n")
        response = b"{\"code\":0}"
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format, *args):
        return

server = ThreadingHTTPServer(("127.0.0.1", 39090), Handler)
print(json.dumps({"callback_url": "http://127.0.0.1:39090/tasks/mock/callback", "log_path": log_path}, ensure_ascii=False), flush=True)
server.serve_forever()
PY'
```

启动 caption sidecar：

```bash
tmux new-session -d -s vividvr_caption_sidecar_mock \
  'cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/logs && CUDA_VISIBLE_DEVICES=0,1 /home/zhiheng/sglang/.venv-vividvr-caption/bin/python python/sglang/multimodal_gen/tools/run_vividvr_caption_sidecar.py --host 127.0.0.1 --port 31200 --parallel-workers 2 --worker-devices cuda:0,cuda:1 --cogvlm2-ckpt-path /home/zhiheng/ckpts/cogvlm2-llama3-caption 2>&1 | tee Vivid_Acceptance/logs/vividvr_caption_sidecar_mock_$(date -u +%Y%m%dT%H%M%SZ).log'
```

健康检查：

```bash
curl --noproxy '*' --silent --show-error --fail http://127.0.0.1:31200/health
```

如果这些 session 已经存在，先用对应 health check 确认服务可用；不要杀掉正在跑正式验收的 session。

只读 attach 命令：

```bash
tmux attach -r -t vividvr_moto_s3
tmux attach -r -t vividvr_flowcut_callback_receiver
tmux attach -r -t vividvr_caption_sidecar_mock
```

- [ ] **步骤 3：启动 CFG-only service**

运行：

```bash
cd /home/zhiheng/sglang
tmux new-session -d -s vividvr_flowcut_cfg_only_mock_service \
  'cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/logs Vivid_Acceptance/captions/service_sidecars && export CUDA_VISIBLE_DEVICES=0,1 && export PYTHONUNBUFFERED=1 && export PYTHONPATH=python && unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY && export NO_PROXY=127.0.0.1,localhost && export no_proxy=127.0.0.1,localhost && export AWS_EC2_METADATA_DISABLED=true && export SGLANG_FLOWCUT_PROGRESS_INTERVAL_SECONDS=5 && export SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE=eager_global && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && /home/zhiheng/sglang/.venv/bin/sglang serve --model-path /home/zhiheng/ckpts/CogVideoX1.5-5B --model-id VividVR --pipeline-class-name CogVideoXVividVRControlNetPipeline --component-paths.vividvr /home/zhiheng/ckpts/Vivid-VR --num-gpus 2 --tp-size 1 --sp-degree 1 --ulysses-degree 1 --ring-degree 1 --enable-cfg-parallel --vividvr-parallel-mode cfg --enable-torch-compile --dist-timeout 3600 --attention-backend fa --host 127.0.0.1 --port 31231 --master-port 30231 --scheduler-port 56231 --strict-ports --input-save-path "" --output-path /home/zhiheng/sglang/Vivid_Acceptance/result_videos/service_benchmark --vividvr-caption-bridge --vividvr-caption-sidecar-url http://127.0.0.1:31200 --vividvr-caption-work-dir /home/zhiheng/sglang/Vivid_Acceptance/captions/service_sidecars --vividvr-caption-sidecar-timeout 1800 2>&1 | tee Vivid_Acceptance/logs/vividvr_flowcut_cfg_only_mock_service_$(date -u +%Y%m%dT%H%M%SZ).log'
```

只读 attach 命令：

```bash
tmux attach -r -t vividvr_flowcut_cfg_only_mock_service
```

健康检查：

```bash
curl --noproxy '*' --silent --show-error --fail http://127.0.0.1:31231/health
```

预期：服务监听 `http://127.0.0.1:31231`，日志显示 `enable_cfg_parallel=True`、`sp_degree=1`、`vividvr_parallel_mode=cfg`。

- [ ] **步骤 4：提交第一次 warmup 请求**

运行：

```bash
cd /home/zhiheng/sglang
export BRIDGE_BASE_URL=http://127.0.0.1:31231
export TASK_ID=vividvr-cfg-only-warmup-$(date -u +%Y%m%dT%H%M%SZ)
curl -sS -X POST "$BRIDGE_BASE_URL/v1/videos/repairs/flowcut" \
  -H 'Content-Type: application/json' \
  -d "{
    \"taskId\": \"${TASK_ID}\",
    \"callbackUrl\": \"${CALLBACK_BASE_URL}/callback/${TASK_ID}\",
    \"video_input_path\": \"${INPUT_VIDEO_130F}\",
    \"num_inference_steps\": 20,
    \"seed\": 42,
    \"num_temporal_process_frames\": 121,
    \"upscale\": 1.0,
    \"output_path\": \"${OUTPUT_DIR}/${TASK_ID}.mp4\",
    \"outputObjectKey\": \"bridge-semantic-check/${TASK_ID}\",
    \"perf_dump_path\": \"${INDICATOR_DIR}/${TASK_ID}_perf.json\",
    \"minioConfig\": {
      \"endpoint\": \"${MOTO_S3_ENDPOINT}\",
      \"bucket_name\": \"${MOTO_S3_BUCKET}\",
      \"access_key\": \"${MOTO_S3_ACCESS_KEY}\",
      \"secret_key\": \"${MOTO_S3_SECRET_KEY}\",
      \"secure\": false,
      \"region\": \"us-east-1\"
    }
  }"
```

等待任务完成：

```bash
curl -sS "$BRIDGE_BASE_URL/v1/videos/repairs/flowcut/${TASK_ID}"
curl -sS "$BRIDGE_BASE_URL/v1/videos/repairs/flowcut/${TASK_ID}/progress"
```

预期：最终 `status` 为 `completed`，progress 为 `100`。本次只作为 compile warmup，不记录正式用时。

- [ ] **步骤 5：提交第二次 formal 请求并记录 CFG-only 用时**

运行：

```bash
cd /home/zhiheng/sglang
export BRIDGE_BASE_URL=http://127.0.0.1:31231
export TASK_ID=vividvr-cfg-only-formal-$(date -u +%Y%m%dT%H%M%SZ)
curl -sS -X POST "$BRIDGE_BASE_URL/v1/videos/repairs/flowcut" \
  -H 'Content-Type: application/json' \
  -d "{
    \"taskId\": \"${TASK_ID}\",
    \"callbackUrl\": \"${CALLBACK_BASE_URL}/callback/${TASK_ID}\",
    \"video_input_path\": \"${INPUT_VIDEO_130F}\",
    \"num_inference_steps\": 20,
    \"seed\": 42,
    \"num_temporal_process_frames\": 121,
    \"upscale\": 1.0,
    \"output_path\": \"${OUTPUT_DIR}/${TASK_ID}.mp4\",
    \"outputObjectKey\": \"bridge-semantic-check/${TASK_ID}\",
    \"perf_dump_path\": \"${INDICATOR_DIR}/${TASK_ID}_perf.json\",
    \"minioConfig\": {
      \"endpoint\": \"${MOTO_S3_ENDPOINT}\",
      \"bucket_name\": \"${MOTO_S3_BUCKET}\",
      \"access_key\": \"${MOTO_S3_ACCESS_KEY}\",
      \"secret_key\": \"${MOTO_S3_SECRET_KEY}\",
      \"secure\": false,
      \"region\": \"us-east-1\"
    }
  }"
```

等待任务完成后读取 perf：

```bash
/home/zhiheng/sglang/.venv/bin/python - <<'PY'
import json
import os
task_id = os.environ["TASK_ID"]
path = f"/home/zhiheng/sglang/Vivid_Acceptance/indicator/service_benchmark/{task_id}_perf.json"
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
print(json.dumps({
    "task_id": task_id,
    "total_runtime_seconds": data.get("total_runtime_seconds"),
    "model_inference_runtime_seconds": data.get("model_inference_runtime_seconds"),
    "vividvr_debug": data.get("vividvr_debug", {}),
}, indent=2))
assert data.get("total_runtime_seconds") is not None
assert data.get("model_inference_runtime_seconds") is not None
debug = data.get("vividvr_debug", {})
assert debug.get("vividvr_parallel_mode") == "cfg"
assert debug.get("cfg_parallel_enabled") is True
assert debug.get("cfg_world_size") == 2
assert debug.get("connector_context_mode") in {
    "single_rank_full_sequence",
    "sp_exact_global_control_attention",
}
PY
```

预期：formal perf JSON 存在，并包含 `total_runtime_seconds`、`model_inference_runtime_seconds`、`cfg_parallel_enabled=true`、`cfg_world_size=2`。

- [ ] **步骤 6：下载 mock S3 结果并比较四卡 reference**

从 bridge 详情接口取回 `result_url`，并下载到固定目标路径：

```bash
export CANDIDATE_VIDEO="${OUTPUT_DIR}/downloads/${TASK_ID}.bridge-downloaded.mp4"
export RESULT_URL=$(curl --noproxy '*' --silent "${BRIDGE_BASE_URL}/v1/videos/repairs/flowcut/${TASK_ID}" | /home/zhiheng/sglang/.venv/bin/python -c 'import json,sys; print(json.load(sys.stdin)["url"])')
curl --noproxy '*' --silent --show-error --fail -L \
  -o "$CANDIDATE_VIDEO" \
  "$RESULT_URL"
test -s "$CANDIDATE_VIDEO"
```

运行质量比较：

```bash
cd /home/zhiheng/sglang
export COMPARE_JSON="${INDICATOR_DIR}/${TASK_ID}_compare_vs_quad_reference.json"
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python \
  python/sglang/multimodal_gen/runtime/videoedit/compare.py \
  --reference "$VIVIDVR_CFG_REFERENCE_VIDEO" \
  --candidate "$CANDIDATE_VIDEO" \
  --report-json "$COMPARE_JSON" \
  --min-ssim 0.98 \
  --max-mse 1000000000 \
  --max-mae 1000000000 \
  --max-failed-frame-ratio 0
/home/zhiheng/sglang/.venv/bin/python - <<'PY'
import json
import os
path = os.environ["COMPARE_JSON"]
with open(path, "r", encoding="utf-8") as f:
    report = json.load(f)
summary = report["summary"]
print(json.dumps(summary, indent=2))
assert summary["pass_compare"] is True
assert summary["ssim_mean"] > 0.98
assert summary["ssim_min"] >= 0.98
PY
```

预期：compare 命令返回 0，且 `ssim_mean > 0.98`、`ssim_min >= 0.98`。

- [ ] **步骤 7：更新 mock_test.md 增补 CFG-only 命令**

把本任务中 CFG-only service、warmup/formal 两次请求、compare 命令整理进：

```bash
docs_xzh/run_command/mock_test.md
```

保留原有双卡 SP2 和四卡 SP4 mock 命令，不删除历史命令。

- [ ] **步骤 8：Commit**

```bash
git add docs_xzh/run_command/mock_test.md
git commit -m "docs: add vividvr cfg-only mock acceptance commands"
```

---

## 任务 7：实现并验收四卡 CFG2 x SP2

**文件：**
- 修改：`python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py`
- 修改：`docs_xzh/run_command/mock_test.md`
- 产物：`/home/zhiheng/sglang/Vivid_Acceptance/logs`
- 产物：`/home/zhiheng/sglang/Vivid_Acceptance/indicator/service_benchmark`
- 产物：`/home/zhiheng/sglang/Vivid_Acceptance/result_videos/service_benchmark`

- [ ] **步骤 1：补充运行时 debug，确认 CFG 与 SP 同时生效**

在 `prepare_denoising_state` 已有 debug 中加入 CFG group 信息：

```python
        enable_cfg_parallel = bool(server_args.enable_cfg_parallel)
        cfg_world_size = 1
        cfg_rank = 0
        if enable_cfg_parallel:
            cfg_world_size = int(get_classifier_free_guidance_world_size())
            cfg_rank = int(get_classifier_free_guidance_rank())
        debug.update(
            {
                "cfg_parallel_enabled": enable_cfg_parallel,
                "cfg_world_size": cfg_world_size,
                "cfg_rank": cfg_rank,
            }
        )
```

保留已有 `sp_world_size`、`sp_rank`、`enable_sequence_shard`、`connector_context_mode`、`control_context_shape_global` 字段。四卡验收依赖这些字段判断是 `CFG=2 x SP=2`，不是退化成 CFG-only 或 SP-only。

- [ ] **步骤 2：运行单元测试**

运行：

```bash
cd /home/zhiheng/sglang
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  test/srt/multimodal_gen/test_vividvr_cfg_parallel_stage.py \
  -q
```

预期：全部通过。

- [ ] **步骤 3：启动四卡 CFG2 x SP2 service**

运行：

```bash
cd /home/zhiheng/sglang
export BRIDGE_BASE_URL=http://127.0.0.1:31232
tmux new-session -d -s vividvr_flowcut_cfg2_sp2_mock_service \
  'cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/logs Vivid_Acceptance/captions/service_sidecars && export CUDA_VISIBLE_DEVICES=0,1,2,3 && export PYTHONUNBUFFERED=1 && export PYTHONPATH=python && unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY && export NO_PROXY=127.0.0.1,localhost && export no_proxy=127.0.0.1,localhost && export AWS_EC2_METADATA_DISABLED=true && export SGLANG_FLOWCUT_PROGRESS_INTERVAL_SECONDS=5 && export SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE=eager_global && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && /home/zhiheng/sglang/.venv/bin/sglang serve --model-path /home/zhiheng/ckpts/CogVideoX1.5-5B --model-id VividVR --pipeline-class-name CogVideoXVividVRControlNetPipeline --component-paths.vividvr /home/zhiheng/ckpts/Vivid-VR --num-gpus 4 --tp-size 1 --sp-degree 2 --ulysses-degree 2 --ring-degree 1 --enable-cfg-parallel --vividvr-parallel-mode cfg_sp --enable-torch-compile --dist-timeout 3600 --attention-backend fa --host 127.0.0.1 --port 31232 --master-port 30232 --scheduler-port 56232 --strict-ports --input-save-path "" --output-path /home/zhiheng/sglang/Vivid_Acceptance/result_videos/service_benchmark --vividvr-caption-bridge --vividvr-caption-sidecar-url http://127.0.0.1:31200 --vividvr-caption-work-dir /home/zhiheng/sglang/Vivid_Acceptance/captions/service_sidecars --vividvr-caption-sidecar-timeout 1800 2>&1 | tee Vivid_Acceptance/logs/vividvr_flowcut_cfg2_sp2_mock_service_$(date -u +%Y%m%dT%H%M%SZ).log'
```

只读 attach 命令：

```bash
tmux attach -r -t vividvr_flowcut_cfg2_sp2_mock_service
```

健康检查：

```bash
curl --noproxy '*' --silent --show-error --fail http://127.0.0.1:31232/health
```

预期：服务监听 `http://127.0.0.1:31232`，日志显示 `enable_cfg_parallel=True`、`sp_degree=2`、`vividvr_parallel_mode=cfg_sp`、effective backend 为 `fa_sp` 或同等 Ulysses distributed attention 描述。

- [ ] **步骤 4：提交第一次 warmup 请求**

运行：

```bash
cd /home/zhiheng/sglang
export BRIDGE_BASE_URL=http://127.0.0.1:31232
export TASK_ID=vividvr-cfg2-sp2-warmup-$(date -u +%Y%m%dT%H%M%SZ)
curl -sS -X POST "$BRIDGE_BASE_URL/v1/videos/repairs/flowcut" \
  -H 'Content-Type: application/json' \
  -d "{
    \"taskId\": \"${TASK_ID}\",
    \"callbackUrl\": \"${CALLBACK_BASE_URL}/callback/${TASK_ID}\",
    \"video_input_path\": \"${INPUT_VIDEO_130F}\",
    \"num_inference_steps\": 20,
    \"seed\": 42,
    \"num_temporal_process_frames\": 121,
    \"upscale\": 1.0,
    \"output_path\": \"${OUTPUT_DIR}/${TASK_ID}.mp4\",
    \"outputObjectKey\": \"bridge-semantic-check/${TASK_ID}\",
    \"perf_dump_path\": \"${INDICATOR_DIR}/${TASK_ID}_perf.json\",
    \"minioConfig\": {
      \"endpoint\": \"${MOTO_S3_ENDPOINT}\",
      \"bucket_name\": \"${MOTO_S3_BUCKET}\",
      \"access_key\": \"${MOTO_S3_ACCESS_KEY}\",
      \"secret_key\": \"${MOTO_S3_SECRET_KEY}\",
      \"secure\": false,
      \"region\": \"us-east-1\"
    }
  }"
```

等待任务 completed。本次只用于 compile，不进入正式性能表。

- [ ] **步骤 5：提交第二次 formal 请求并记录四卡用时**

运行：

```bash
cd /home/zhiheng/sglang
export BRIDGE_BASE_URL=http://127.0.0.1:31232
export TASK_ID=vividvr-cfg2-sp2-formal-$(date -u +%Y%m%dT%H%M%SZ)
curl -sS -X POST "$BRIDGE_BASE_URL/v1/videos/repairs/flowcut" \
  -H 'Content-Type: application/json' \
  -d "{
    \"taskId\": \"${TASK_ID}\",
    \"callbackUrl\": \"${CALLBACK_BASE_URL}/callback/${TASK_ID}\",
    \"video_input_path\": \"${INPUT_VIDEO_130F}\",
    \"num_inference_steps\": 20,
    \"seed\": 42,
    \"num_temporal_process_frames\": 121,
    \"upscale\": 1.0,
    \"output_path\": \"${OUTPUT_DIR}/${TASK_ID}.mp4\",
    \"outputObjectKey\": \"bridge-semantic-check/${TASK_ID}\",
    \"perf_dump_path\": \"${INDICATOR_DIR}/${TASK_ID}_perf.json\",
    \"minioConfig\": {
      \"endpoint\": \"${MOTO_S3_ENDPOINT}\",
      \"bucket_name\": \"${MOTO_S3_BUCKET}\",
      \"access_key\": \"${MOTO_S3_ACCESS_KEY}\",
      \"secret_key\": \"${MOTO_S3_SECRET_KEY}\",
      \"secure\": false,
      \"region\": \"us-east-1\"
    }
  }"
```

等待 completed 后读取 perf：

```bash
/home/zhiheng/sglang/.venv/bin/python - <<'PY'
import json
import os
task_id = os.environ["TASK_ID"]
path = f"/home/zhiheng/sglang/Vivid_Acceptance/indicator/service_benchmark/{task_id}_perf.json"
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
debug = data.get("vividvr_debug", {})
print(json.dumps({
    "task_id": task_id,
    "total_runtime_seconds": data.get("total_runtime_seconds"),
    "model_inference_runtime_seconds": data.get("model_inference_runtime_seconds"),
    "vividvr_parallel_mode": debug.get("vividvr_parallel_mode"),
    "cfg_world_size": debug.get("cfg_world_size"),
    "sp_world_size": debug.get("sp_world_size"),
    "connector_context_mode": debug.get("connector_context_mode"),
    "control_context_shape_global": debug.get("control_context_shape_global"),
}, indent=2))
assert data.get("total_runtime_seconds") is not None
assert data.get("model_inference_runtime_seconds") is not None
assert debug.get("vividvr_parallel_mode") == "cfg_sp"
assert debug.get("cfg_parallel_enabled") is True
assert debug.get("cfg_world_size") == 2
assert debug.get("sp_world_size") == 2
assert debug.get("enable_sequence_shard") is True
assert debug.get("connector_context_mode") == "sp_exact_global_control_attention"
assert debug.get("control_context_shape_global") is not None
PY
```

预期：formal perf JSON 证明 `CFG=2` 和 `SP=2` 同时生效，且仍是 eager global control context。

- [ ] **步骤 6：下载四卡 formal 结果并做 SSIM 验收**

从 bridge 详情接口取回 `result_url`，并下载到固定目标路径：

```bash
export CANDIDATE_VIDEO="${OUTPUT_DIR}/downloads/${TASK_ID}.bridge-downloaded.mp4"
export RESULT_URL=$(curl --noproxy '*' --silent "${BRIDGE_BASE_URL}/v1/videos/repairs/flowcut/${TASK_ID}" | /home/zhiheng/sglang/.venv/bin/python -c 'import json,sys; print(json.load(sys.stdin)["url"])')
curl --noproxy '*' --silent --show-error --fail -L \
  -o "$CANDIDATE_VIDEO" \
  "$RESULT_URL"
test -s "$CANDIDATE_VIDEO"
```

运行 compare：

```bash
cd /home/zhiheng/sglang
export COMPARE_JSON="${INDICATOR_DIR}/${TASK_ID}_compare_vs_quad_reference.json"
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python \
  python/sglang/multimodal_gen/runtime/videoedit/compare.py \
  --reference "$VIVIDVR_CFG_REFERENCE_VIDEO" \
  --candidate "$CANDIDATE_VIDEO" \
  --report-json "$COMPARE_JSON" \
  --min-ssim 0.98 \
  --max-mse 1000000000 \
  --max-mae 1000000000 \
  --max-failed-frame-ratio 0
/home/zhiheng/sglang/.venv/bin/python - <<'PY'
import json
import os
path = os.environ["COMPARE_JSON"]
with open(path, "r", encoding="utf-8") as f:
    report = json.load(f)
summary = report["summary"]
print(json.dumps(summary, indent=2))
assert summary["pass_compare"] is True
assert summary["ssim_mean"] > 0.98
assert summary["ssim_min"] >= 0.98
PY
```

预期：命令返回 0，`summary.ssim_mean > 0.98` 且 `summary.ssim_min >= 0.98`。

- [ ] **步骤 7：记录性能判定**

把以下结果写入 handover：

使用下面的命令从正式验收产物生成记录片段，再把输出写入 handover：

```bash
/home/zhiheng/sglang/.venv/bin/python - <<'PY'
import json
import os

cfg_only_task_id = os.environ["CFG_ONLY_FORMAL_TASK_ID"]
cfg2_sp2_task_id = os.environ["CFG2_SP2_FORMAL_TASK_ID"]
indicator_dir = "/home/zhiheng/sglang/Vivid_Acceptance/indicator/service_benchmark"
reference_video = "/home/zhiheng/sglang/Vivid_Acceptance/result_videos/service_benchmark/downloads/quad-test-video-long-960x720-130f-run2-20260708T060202Z.bridge-downloaded.mp4"

def load_perf(task_id: str) -> dict:
    with open(f"{indicator_dir}/{task_id}_perf.json", "r", encoding="utf-8") as f:
        return json.load(f)

cfg_only_perf = load_perf(cfg_only_task_id)
cfg2_sp2_perf = load_perf(cfg2_sp2_task_id)
compare_report = f"{indicator_dir}/{cfg2_sp2_task_id}_compare_vs_quad_reference.json"
print(f"CFG-only formal task id: {cfg_only_task_id}")
print(f"CFG-only formal model_inference_runtime_seconds: {cfg_only_perf['model_inference_runtime_seconds']}")
print(f"CFG2 x SP2 formal task id: {cfg2_sp2_task_id}")
print(f"CFG2 x SP2 formal model_inference_runtime_seconds: {cfg2_sp2_perf['model_inference_runtime_seconds']}")
print(f"Reference video: {reference_video}")
print(f"Compare report: {compare_report}")
PY
```

如果 `CFG2 x SP2` 比当前双卡 v2 baseline 没有明显下降，也不允许改动语义去追求速度；先保留正确实现，再单独分析瓶颈。

- [ ] **步骤 8：更新 mock_test.md 增补 CFG2 x SP2 命令**

把四卡 `CFG=2 x SP=2` service、warmup/formal 请求、perf 校验、compare 命令整理进：

```bash
docs_xzh/run_command/mock_test.md
```

保留“第一次为 compile warmup、第二次才计时”的说明。

- [ ] **步骤 9：Commit**

```bash
git add python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py \
  docs_xzh/run_command/mock_test.md
git commit -m "feat: enable vividvr cfg2 sp2 acceleration path"
```

---

## 任务 8：回归守护与交接

**文件：**
- 创建：`docs_xzh/hand_over/vividvr_cfg_parallel_v2_equivalence_handover_20260709.md`
- 检查：`docs_xzh/run_command/mock_test.md`
- 检查：`Vivid_Acceptance/indicator/service_benchmark/*.json`

- [ ] **步骤 1：运行最小非服务单元回归**

运行：

```bash
cd /home/zhiheng/sglang
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  test/srt/multimodal_gen/test_vividvr_cfg_parallel_stage.py \
  -q
```

预期：全部通过。

- [ ] **步骤 2：确认默认路径没有被改变**

运行：

```bash
cd /home/zhiheng/sglang
git diff -- python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py \
  docs_xzh/run_command/mock_test.md \
  test/srt/multimodal_gen/test_vividvr_cfg_parallel_stage.py
```

检查点：

```text
未传 --enable-cfg-parallel 时，VividVR 仍使用原 batch=2 串行 CFG。
未传 --enable-cfg-parallel 时，parallelism_type 仍为 REPLICATED。
双卡默认 dual_gpu_fa_eager_compile 不自动变成 CFG-only。
四卡 CFG2 x SP2 必须显式传 --enable-cfg-parallel。
```

- [ ] **步骤 3：创建交接文档**

创建 `docs_xzh/hand_over/vividvr_cfg_parallel_v2_equivalence_handover_20260709.md`，内容包含：

```markdown
# VividVR CFG Parallel v2 等价闭环交接 2026-07-09

## 结论

- CFG-only v2 等价闭环：通过或失败，并写明证据路径。
- CFG2 x SP2 四卡验收：通过或失败，并写明证据路径。
- 默认 single_gpu_fa_compile、dual_gpu_fa_eager_compile、dual_gpu_sdpa_eager_compile 是否保持不变。

## 代码变更

- `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py`
- `test/srt/multimodal_gen/test_vividvr_cfg_parallel_stage.py`
- `docs_xzh/run_command/mock_test.md`

## 验收产物

- CFG-only warmup task id:
- CFG-only formal task id:
- CFG-only perf JSON:
- CFG-only compare JSON:
- CFG2 x SP2 warmup task id:
- CFG2 x SP2 formal task id:
- CFG2 x SP2 perf JSON:
- CFG2 x SP2 compare JSON:
- Reference video:

## 关键指标

- CFG-only formal `model_inference_runtime_seconds`:
- CFG2 x SP2 formal `model_inference_runtime_seconds`:
- CFG2 x SP2 `ssim_mean`:
- CFG2 x SP2 `ssim_min`:

## 风险与后续

- 若质量未过 0.98，优先检查 CFG branch prompt 顺序、CFG all-reduce、SP eager_global control context，不允许降级语义。
- 若质量通过但速度收益不足，下一轮只做 profiling，不改变 v2 语义。
```

把实际路径和数值填入，不保留空项。

- [ ] **步骤 4：最终 commit**

```bash
git add docs_xzh/hand_over/vividvr_cfg_parallel_v2_equivalence_handover_20260709.md
git commit -m "docs: hand over vividvr cfg parallel acceptance"
```

---

## 最终验收条件

1. 单元测试通过：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  test/srt/multimodal_gen/test_vividvr_cfg_parallel_stage.py \
  -q
```

2. CFG-only mock 服务验收通过：

```text
service: http://127.0.0.1:31231
num_gpus: 2
vividvr_parallel_mode: cfg
cfg_world_size: 2
sp_world_size: 1
first request: warmup only
second request: formal timing
formal perf has total_runtime_seconds and model_inference_runtime_seconds
formal compare summary.ssim_mean > 0.98
formal compare summary.ssim_min >= 0.98
```

3. 四卡 `CFG=2 x SP=2` mock 服务验收通过：

```text
service: http://127.0.0.1:31232
num_gpus: 4
vividvr_parallel_mode: cfg_sp
cfg_world_size: 2
sp_world_size: 2
connector_context_mode: sp_exact_global_control_attention
control_context_shape_global: not null
first request: warmup only
second request: formal timing
formal perf has total_runtime_seconds and model_inference_runtime_seconds
formal compare summary.ssim_mean > 0.98
formal compare summary.ssim_min >= 0.98
```

4. 默认正式配置不被改写：

```text
single_gpu_fa_compile 仍不启用 CFG parallel
dual_gpu_fa_eager_compile 仍是 SP=2 + eager_global，不启用 CFG parallel，可显式标记 vividvr_parallel_mode=sp 或保持 auto 推导为 sp
dual_gpu_sdpa_eager_compile 仍是 SP=2 + eager_global，不启用 CFG parallel，可显式标记 vividvr_parallel_mode=sp 或保持 auto 推导为 sp
CFG2 x SP2 是显式新增实验/加速入口
纯 SP 路径必须保留，不能被 CFG parallel 分支替换或删除
```

5. 文档更新完成：

```text
docs_xzh/run_command/mock_test.md 包含 CFG-only 与 CFG2 x SP2 mock 验收命令
docs_xzh/hand_over/vividvr_cfg_parallel_v2_equivalence_handover_20260709.md 包含实际任务 ID、日志、perf、compare 路径和指标
```

## 不通过时的定位顺序

1. 如果单元测试失败，先修 CFG branch 顺序或公式，不启动服务验收。
2. 如果 CFG-only 质量失败，优先检查正负 prompt 分支是否反了、`noise_pred` 合成符号是否反了、`timestep_expand` batch size 是否仍为 2。
3. 如果 CFG-only 通过但 CFG2 x SP2 质量失败，优先检查 `connector_context_mode` 是否仍是 `sp_exact_global_control_attention`、`control_context_shape_global` 是否为空、SP shard metadata 是否和双卡 v2 相同。
4. 如果质量通过但速度不达预期，不修改语义；下一轮只做 profiling，重点看 controlnet/transformer 每 step 时间、CFG all-reduce 开销、compile graph break、SP communication overlap。
