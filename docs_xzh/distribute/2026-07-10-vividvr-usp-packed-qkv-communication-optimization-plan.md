# VividVR USP Packed QKV 通信优化实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 在不改变 VividVR CFG2×SP2 数值语义和正式服务契约的前提下，将四卡 `130f / 20 step` 的 `model_inference_runtime_seconds` 从基线 `194.2424s` 降到 `175s` 以下。

**架构：** 保留现有 CFG2×SP2、Ulysses `fa_sp`、`eager_global`、BF16 主计算、CFG FP32 合并与 compile 路径，只优化 USP 通信发起方式。第一阶段把 Q/K/V 三次输入 `all_to_all_single` 打包成一次 collective，并把 replicated-prefix 输出从 Python tensor-list `all_gather` 改成 functional `all_gather_into_tensor` 路径；两个优化各自受默认关闭的独立开关控制，以便逐项消融和立即回退。

**技术栈：** Python 3.11、PyTorch distributed functional collectives、NCCL、Diffusers CogVideoX attention processor、pytest、torchrun、torch profiler、tmux。

---

## 硬约束与成功标准

- 固定硬件：`4 × NVIDIA A100`，拓扑保持现有正式四卡环境，不通过增加 GPU 达标。
- 固定并行：`CFG degree = 2`、`SP/Ulysses degree = 2`、`ring degree = 1`。
- 固定正式语义：请求 backend 为 `fa`，有效 backend 为 `fa_sp`，connector context 为 `eager_global`，torch compile 开启。
- 固定精度：模型主计算保持 BF16；CFG cond/uncond 合并保持现有 FP32 路径。
- 固定输入：`130f / 20 step` 长视频正式 reference、caption、seed、tile 参数和现有服务请求字段不变。
- 性能门槛：预热后正式 run 的 `model_inference_runtime_seconds < 175.0`。
- 质量门槛：相对正式 reference 的 `SSIM mean >= 0.98`、`SSIM min >= 0.98`、`failed_frame_ratio = 0`。
- 回归门槛：`single_gpu_fa_compile`、`dual_gpu_fa_eager_compile`、`dual_gpu_sdpa_eager_compile` 的已有单元测试继续通过。
- 回退要求：两个新开关默认均为 `False`；关闭后必须执行当前 legacy collective 路径。
- 错误策略：在非 Ulysses 或 `ulysses_degree <= 1` 场景显式请求新开关时立即报错，不做静默降级。

## 数据布局合同

Packed QKV 只改变 collective 的调用次数，不改变每个 rank 收到的数据顺序。输入布局固定为：

```text
q, k, v: [B, S_local, H_global, D]
stack:   [3, B, S_local, H_global, D]
permute: [H_global, 3, B, S_local, D]
A2A 后:  [SP, H_local, 3, B, S_local, D]
permute: [3, B, SP, S_local, H_local, D]
reshape: [3, B, S_global, H_local, D]
unbind:  q', k', v': [B, S_global, H_local, D]
```

其中 `H_local = H_global / SP`，`S_global = S_local × SP`。正式 workload 为 `2 clips × 20 steps × 48 attention blocks`；输入 Q/K/V collective 次数由约 `5760` 次降为 `1920` 次，连同每层一次输出 A2A，总大 collective 次数由约 `7680` 次降为 `3840` 次。传输字节总量理论上不变，收益来自减少 NCCL launch、functional collective 调度和 Python 调用开销。

Replicated-prefix 输出合同固定为：

```text
out_rep local:  [B, T_text, H_local, D]
gather dim=2:   [B, T_text, H_global, D]
rank/head 顺序: 按 rank 递增排列，每个 rank 内保持本地 head 顺序
```

## 文件结构

### 创建

- `python/sglang/multimodal_gen/test/unit/test_usp_packed_collectives.py`：CPU 级布局、校验、legacy 等价性和开关选择测试。
- `test/srt/multimodal_gen/test_usp_packed_collectives_distributed.py`：真实两卡 NCCL collective 精确等价测试入口。
- `docs_xzh/distribute/vividvr_usp_collective_ablation_20260710.md`：记录 B0/P1/P2/P3 命令、profiler 指标、正式耗时和质量结果。

### 修改

- `python/sglang/multimodal_gen/runtime/server_args.py`：增加两个默认关闭的服务参数和 CLI 参数。
- `python/sglang/multimodal_gen/tools/run_vividvr_inference.py`：直跑 CLI、`ServerArgs` 传播、runtime snapshot 和报告字段。
- `python/sglang/multimodal_gen/runtime/layers/usp.py`：packed QKV input A2A 与 tensor-form prefix gather 原语。
- `python/sglang/multimodal_gen/runtime/layers/attention/layer.py`：USPAttention 开关、legacy/optimized 分支和 prefix 输出接入。
- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_attention_backend.py`：将开关保存在 processor，并纳入 lazy USP cache key。
- `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py`：在 backend 应用后、compile 前配置 transformer/controlnet processor，并暴露 requested/effective debug。
- `python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_attention_backend.py`：processor 配置、cache key、pipeline fail-fast 与 debug 测试。
- `python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_inference_tool.py`：CLI、ServerArgs、snapshot、报告字段测试。
- `docs_xzh/run_command/vividvr_default_run_and_serve_commands.md`：仅在正式验收通过后补充可选优化开关；不改当前默认配置。
- `docs_xzh/hand_over/vividvr_cfg_parallel_v2_equivalence_handover_20260709.md`：仅在正式验收通过后追加优化结果、回退命令和产物路径。

## 任务 1：增加显式参数、CLI 传播与运行快照

**文件：**
- 修改：`python/sglang/multimodal_gen/runtime/server_args.py:176-185`
- 修改：`python/sglang/multimodal_gen/runtime/server_args.py:728-758`
- 修改：`python/sglang/multimodal_gen/tools/run_vividvr_inference.py:178-270`
- 修改：`python/sglang/multimodal_gen/tools/run_vividvr_inference.py:354-400`
- 修改：`python/sglang/multimodal_gen/tools/run_vividvr_inference.py:689-735`
- 修改：`python/sglang/multimodal_gen/tools/run_vividvr_inference.py:930-983`
- 测试：`python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_inference_tool.py`

- [ ] **步骤 1：编写默认值和传播失败测试**

在现有 inference tool 测试类中加入以下测试，并在测试使用的 `argparse.Namespace` fixture 中明确补齐两个字段：

```python
def test_usp_collective_flags_default_disabled(self):
    argv = ["run_vividvr_inference.py", "--input-video", "/tmp/input.mp4"]
    with patch.object(sys, "argv", argv):
        args = parse_args()
    self.assertFalse(args.enable_usp_packed_qkv_a2a)
    self.assertFalse(args.enable_usp_prefix_all_gather_into_tensor)

def test_build_server_args_propagates_usp_collective_flags(self):
    argv = [
        "run_vividvr_inference.py",
        "--input-video",
        "/tmp/input.mp4",
        "--enable-usp-packed-qkv-a2a",
        "--enable-usp-prefix-all-gather-into-tensor",
    ]
    with patch.object(sys, "argv", argv):
        args = parse_args()
    server_args = build_server_args(args)
    self.assertTrue(server_args.enable_usp_packed_qkv_a2a)
    self.assertTrue(server_args.enable_usp_prefix_all_gather_into_tensor)

def test_runtime_snapshot_records_usp_collective_flags(self):
    args = Namespace(
        attention_backend="fa",
        use_runai_model_streamer=None,
        use_vividvr_vae_decode_tiling=None,
    )
    server_args = ServerArgs(
        model_path="/tmp/model",
        enable_usp_packed_qkv_a2a=True,
        enable_usp_prefix_all_gather_into_tensor=False,
    )
    snapshot = build_runtime_config_snapshot(args=args, server_args=server_args)
    self.assertTrue(snapshot["enable_usp_packed_qkv_a2a"])
    self.assertFalse(snapshot["enable_usp_prefix_all_gather_into_tensor"])
```

- [ ] **步骤 2：运行测试并确认新接口尚不存在**

运行：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_inference_tool.py \
  -k 'usp_collective' -q
```

预期：FAIL，错误包含 `unrecognized arguments`、`AttributeError` 或 `unexpected keyword argument`，证明两个参数尚未接入。

- [ ] **步骤 3：在 ServerArgs 和两个 parser 中加入默认关闭参数**

在 `ServerArgs` compilation 字段附近加入：

```python
enable_usp_packed_qkv_a2a: bool = False
enable_usp_prefix_all_gather_into_tensor: bool = False
```

在 `ServerArgs.add_cli_args()` 中加入：

```python
parser.add_argument(
    "--enable-usp-packed-qkv-a2a",
    action=StoreBoolean,
    default=ServerArgs.enable_usp_packed_qkv_a2a,
    help="Pack USP Q/K/V input all-to-all into one functional collective.",
)
parser.add_argument(
    "--enable-usp-prefix-all-gather-into-tensor",
    action=StoreBoolean,
    default=ServerArgs.enable_usp_prefix_all_gather_into_tensor,
    help="Use functional all_gather_into_tensor for USP replicated-prefix output.",
)
```

在 `run_vividvr_inference.py` 的 parser 中加入对应 `BooleanOptionalAction`：

```python
parser.add_argument(
    "--enable-usp-packed-qkv-a2a",
    action=argparse.BooleanOptionalAction,
    default=False,
)
parser.add_argument(
    "--enable-usp-prefix-all-gather-into-tensor",
    action=argparse.BooleanOptionalAction,
    default=False,
)
```

- [ ] **步骤 4：传播到 ServerArgs、snapshot 和最终报告**

在 `build_server_args()` 构造参数中加入：

```python
enable_usp_packed_qkv_a2a=args.enable_usp_packed_qkv_a2a,
enable_usp_prefix_all_gather_into_tensor=(
    args.enable_usp_prefix_all_gather_into_tensor
),
```

在 `build_runtime_config_snapshot()` 和最终报告字典中分别加入：

```python
"enable_usp_packed_qkv_a2a": bool(server_args.enable_usp_packed_qkv_a2a),
"enable_usp_prefix_all_gather_into_tensor": bool(
    server_args.enable_usp_prefix_all_gather_into_tensor
),
```

最终报告直接记录 CLI requested 值：

```python
"enable_usp_packed_qkv_a2a": args.enable_usp_packed_qkv_a2a,
"enable_usp_prefix_all_gather_into_tensor": (
    args.enable_usp_prefix_all_gather_into_tensor
),
```

- [ ] **步骤 5：运行定向测试和完整 inference tool 测试**

运行：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_inference_tool.py -q
```

预期：该文件全部 PASS，且无 fixture 缺字段错误。

- [ ] **步骤 6：提交参数面变更**

```bash
git add \
  python/sglang/multimodal_gen/runtime/server_args.py \
  python/sglang/multimodal_gen/tools/run_vividvr_inference.py \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_inference_tool.py
git commit -m "feat(vividvr): expose usp collective optimization flags"
```

## 任务 2：实现 packed QKV 输入 A2A 原语

**文件：**
- 创建：`python/sglang/multimodal_gen/test/unit/test_usp_packed_collectives.py`
- 修改：`python/sglang/multimodal_gen/runtime/layers/usp.py:36-103`

- [ ] **步骤 1：创建 world-size=1 和布局等价失败测试**

```python
from unittest.mock import patch

import pytest
import torch

from sglang.multimodal_gen.runtime.layers.usp import (
    _usp_input_all_to_all,
    _usp_input_all_to_all_qkv,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_packed_qkv_world_size_one_returns_inputs(dtype):
    q = torch.randn(1, 5, 4, 8, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    with patch(
        "sglang.multimodal_gen.runtime.layers.usp.get_ulysses_parallel_world_size",
        return_value=1,
    ):
        actual = _usp_input_all_to_all_qkv(q, k, v)
    assert actual[0] is q
    assert actual[1] is k
    assert actual[2] is v


def test_packed_qkv_matches_three_legacy_calls():
    q = torch.arange(1 * 3 * 4 * 2, dtype=torch.float32).reshape(1, 3, 4, 2)
    k = q + 1000
    v = q + 2000

    def fake_a2a(x):
        return x.reshape(2, 2, *x.shape[1:]).flip(0).reshape_as(x)

    with patch(
        "sglang.multimodal_gen.runtime.layers.usp.get_ulysses_parallel_world_size",
        return_value=2,
    ), patch(
        "sglang.multimodal_gen.runtime.layers.usp._usp_all_to_all_single",
        side_effect=fake_a2a,
    ):
        expected = tuple(
            _usp_input_all_to_all(x, head_dim=2) for x in (q, k, v)
        )
    with patch(
        "sglang.multimodal_gen.runtime.layers.usp.get_ulysses_parallel_world_size",
        return_value=2,
    ), patch(
        "sglang.multimodal_gen.runtime.layers.usp._usp_all_to_all_single",
        side_effect=fake_a2a,
    ):
        actual = _usp_input_all_to_all_qkv(q, k, v)
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)
```

- [ ] **步骤 2：运行测试并确认 helper 缺失**

运行：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_usp_packed_collectives.py \
  -k 'packed_qkv' -q
```

预期：测试收集失败，错误包含 `cannot import name '_usp_input_all_to_all_qkv'`。

- [ ] **步骤 3：实现最小 packed helper**

在 `_usp_input_all_to_all()` 后加入：

```python
def _usp_input_all_to_all_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    world_size = get_ulysses_parallel_world_size()
    if world_size <= 1:
        return q, k, v

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("Packed USP Q/K/V inputs must all be 4D tensors.")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(
            f"Packed USP Q/K/V shapes must match, got {q.shape}, {k.shape}, {v.shape}."
        )
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("Packed USP Q/K/V dtypes must match.")
    if q.device != k.device or q.device != v.device:
        raise ValueError("Packed USP Q/K/V devices must match.")

    batch, seq_local, heads_global, head_size = q.shape
    if heads_global % world_size != 0:
        raise ValueError(
            f"heads_global ({heads_global}) must be divisible by world_size ({world_size})."
        )

    heads_local = heads_global // world_size
    packed = torch.stack((q, k, v), dim=0)
    packed = packed.permute(3, 0, 1, 2, 4).contiguous()
    packed = _usp_all_to_all_single(packed)
    packed = packed.reshape(
        world_size,
        heads_local,
        3,
        batch,
        seq_local,
        head_size,
    )
    packed = packed.permute(2, 3, 0, 4, 1, 5).contiguous()
    packed = packed.reshape(
        3,
        batch,
        seq_local * world_size,
        heads_local,
        head_size,
    )
    q_out, k_out, v_out = packed.unbind(dim=0)
    return q_out, k_out, v_out
```

- [ ] **步骤 4：增加错误合同测试**

```python
@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda q: (q, q[:, :-1], q), "shapes must match"),
        (lambda q: (q, q.float(), q), "dtypes must match"),
        (lambda q: (q.squeeze(0), q, q), "must all be 4D"),
    ],
)
def test_packed_qkv_rejects_incompatible_inputs(mutate, message):
    q = torch.randn(1, 3, 4, 8, dtype=torch.bfloat16)
    with patch(
        "sglang.multimodal_gen.runtime.layers.usp.get_ulysses_parallel_world_size",
        return_value=2,
    ):
        with pytest.raises(ValueError, match=message):
            _usp_input_all_to_all_qkv(*mutate(q))


def test_packed_qkv_requires_divisible_heads():
    q = torch.randn(1, 3, 3, 8)
    with patch(
        "sglang.multimodal_gen.runtime.layers.usp.get_ulysses_parallel_world_size",
        return_value=2,
    ):
        with pytest.raises(ValueError, match="must be divisible"):
            _usp_input_all_to_all_qkv(q, q, q)
```

- [ ] **步骤 5：运行 helper 全部测试**

运行：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_usp_packed_collectives.py -q
```

预期：全部 PASS；packed/legacy 的比较为 bitwise exact。

- [ ] **步骤 6：提交 packed 原语**

```bash
git add \
  python/sglang/multimodal_gen/runtime/layers/usp.py \
  python/sglang/multimodal_gen/test/unit/test_usp_packed_collectives.py
git commit -m "feat(vividvr): add packed usp qkv all-to-all primitive"
```

## 任务 3：在 USPAttention 接入两个独立优化分支

**文件：**
- 修改：`python/sglang/multimodal_gen/runtime/layers/usp.py:26-46`
- 修改：`python/sglang/multimodal_gen/runtime/layers/attention/layer.py:330-395`
- 修改：`python/sglang/multimodal_gen/runtime/layers/attention/layer.py:443-525`
- 测试：`python/sglang/multimodal_gen/test/unit/test_usp_packed_collectives.py`

- [ ] **步骤 1：编写开关选择和 prefix gather 失败测试**

```python
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.multimodal_gen.runtime.layers.attention.layer import USPAttention
from sglang.multimodal_gen.runtime.layers.usp import _usp_prefix_all_gather


def test_usp_input_selector_uses_packed_helper_when_enabled():
    attention = SimpleNamespace(use_packed_qkv_a2a=True)
    q = torch.randn(1, 3, 4, 8)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    with patch(
        "sglang.multimodal_gen.runtime.layers.attention.layer._usp_input_all_to_all_qkv",
        return_value=(q + 1, k + 1, v + 1),
    ) as packed:
        actual = USPAttention._input_all_to_all_qkv(attention, q, k, v)
    packed.assert_called_once_with(q, k, v)
    torch.testing.assert_close(actual[0], q + 1)


def test_usp_input_selector_keeps_three_legacy_calls_when_disabled():
    attention = SimpleNamespace(use_packed_qkv_a2a=False)
    q = torch.randn(1, 3, 4, 8)
    with patch(
        "sglang.multimodal_gen.runtime.layers.attention.layer._usp_input_all_to_all",
        side_effect=lambda x, head_dim: x + 1,
    ) as legacy:
        actual = USPAttention._input_all_to_all_qkv(attention, q, q, q)
    assert legacy.call_count == 3
    assert all(call.kwargs == {"head_dim": 2} for call in legacy.call_args_list)
    assert all(torch.equal(x, q + 1) for x in actual)


def test_prefix_all_gather_uses_functional_collective_on_head_dim():
    x = torch.randn(1, 5, 2, 8)
    expected = torch.randn(1, 5, 4, 8)
    fake_group = MagicMock()
    fake_sp_group = MagicMock(ulysses_group=fake_group)
    with patch(
        "sglang.multimodal_gen.runtime.layers.usp.get_sp_group",
        return_value=fake_sp_group,
    ), patch(
        "sglang.multimodal_gen.runtime.layers.usp.ft_c.all_gather_tensor",
        return_value=expected,
    ) as gather:
        actual = _usp_prefix_all_gather(x)
    gather.assert_called_once_with(x.contiguous(), gather_dim=2, group=fake_group)
    assert actual is expected
```

- [ ] **步骤 2：运行测试并确认 selector/helper 缺失**

运行：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_usp_packed_collectives.py \
  -k 'selector or prefix_all_gather' -q
```

预期：FAIL，错误指出 `_input_all_to_all_qkv` 或 `_usp_prefix_all_gather` 不存在。

- [ ] **步骤 3：实现 functional prefix gather**

在 `usp.py` 中加入：

```python
def _usp_prefix_all_gather(x: torch.Tensor) -> torch.Tensor:
    ulysses_pg = get_sp_group().ulysses_group
    if ulysses_pg is None:
        raise RuntimeError("Ulysses process group is not initialized.")
    gathered = ft_c.all_gather_tensor(
        x.contiguous(),
        gather_dim=2,
        group=ulysses_pg,
    )
    return _maybe_wait(gathered)
```

- [ ] **步骤 4：给 USPAttention 增加开关和唯一选择器**

在构造函数显式参数中加入：

```python
use_packed_qkv_a2a: bool = False,
use_prefix_all_gather_into_tensor: bool = False,
```

保存字段并实现选择器：

```python
self.use_packed_qkv_a2a = use_packed_qkv_a2a
self.use_prefix_all_gather_into_tensor = use_prefix_all_gather_into_tensor

def _input_all_to_all_qkv(self, q, k, v):
    if self.use_packed_qkv_a2a:
        return _usp_input_all_to_all_qkv(q, k, v)
    return (
        _usp_input_all_to_all(q, head_dim=2),
        _usp_input_all_to_all(k, head_dim=2),
        _usp_input_all_to_all(v, head_dim=2),
    )
```

将普通 forward 和 replicated-prefix 路径中的三次 legacy 调用统一替换为：

```python
q, k, v = self._input_all_to_all_qkv(q, k, v)
```

replicated-prefix 分片变量版本使用：

```python
q_shard, k_shard, v_shard = self._input_all_to_all_qkv(
    q_shard,
    k_shard,
    v_shard,
)
```

- [ ] **步骤 5：仅在 prefix 开关开启时替换输出 gather**

```python
if self.use_prefix_all_gather_into_tensor:
    out_rep = _usp_prefix_all_gather(out_rep)
else:
    gathered = [torch.empty_like(out_rep) for _ in range(sp_size)]
    torch.distributed.all_gather(
        gathered,
        out_rep.contiguous(),
        group=get_sp_group().ulysses_group,
    )
    out_rep = torch.cat(gathered, dim=2)
```

不要改变 `_usp_output_all_to_all`，也不要把 prefix 优化扩展到 suffix 以外的新语义；suffix 继续复用已验收的 prefix rotation 路径。

- [ ] **步骤 6：运行定向测试和 sequence-shard 回归**

运行：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_usp_packed_collectives.py \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_sequence_shard.py \
  -q
```

预期：全部 PASS；legacy 默认分支的测试继续观察到三次输入 A2A。

- [ ] **步骤 7：提交 USP 接入**

```bash
git add \
  python/sglang/multimodal_gen/runtime/layers/usp.py \
  python/sglang/multimodal_gen/runtime/layers/attention/layer.py \
  python/sglang/multimodal_gen/test/unit/test_usp_packed_collectives.py \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_sequence_shard.py
git commit -m "feat(vividvr): select optimized usp collective paths"
```

## 任务 4：把开关传播到 CogVideoX lazy USP cache

**文件：**
- 修改：`python/sglang/multimodal_gen/runtime/models/dits/cogvideox_attention_backend.py:506-638`
- 测试：`python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_attention_backend.py`

`_get_cogvideox_sp_usp_attention()` 当前由 processor 懒创建，生成的 `USPAttention` 不是 transformer/controlnet 的注册子模块。因此不能在 pipeline 中遍历并直接修改 USP 实例；配置必须保存在 `CogVideoXSPAttnProcessor`，并作为 lazy factory 的 cache key 传入。

- [ ] **步骤 1：编写 processor 属性和 factory cache-key 失败测试**

在 attention backend 测试中导入 `CogVideoXSPAttnProcessor` 和 `_get_cogvideox_sp_usp_attention`，加入：

```python
def test_sp_processor_stores_usp_collective_flags(self):
    processor = CogVideoXSPAttnProcessor(
        kernel="fa",
        use_packed_qkv_a2a=True,
        use_prefix_all_gather_into_tensor=True,
    )
    self.assertTrue(processor.use_packed_qkv_a2a)
    self.assertTrue(processor.use_prefix_all_gather_into_tensor)

@patch(
    "sglang.multimodal_gen.runtime.models.dits.cogvideox_attention_backend.USPAttention"
)
def test_usp_factory_forwards_collective_flags(self, usp_cls):
    _get_cogvideox_sp_usp_attention.cache_clear()
    _get_cogvideox_sp_usp_attention(
        num_heads=48,
        head_size=64,
        kernel="fa",
        use_packed_qkv_a2a=True,
        use_prefix_all_gather_into_tensor=False,
    )
    self.assertTrue(usp_cls.call_args.kwargs["use_packed_qkv_a2a"])
    self.assertFalse(
        usp_cls.call_args.kwargs["use_prefix_all_gather_into_tensor"]
    )
```

- [ ] **步骤 2：运行测试并确认构造参数未支持**

运行：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_attention_backend.py \
  -k 'collective_flags' -q
```

预期：FAIL，错误包含 `unexpected keyword argument`。

- [ ] **步骤 3：扩展 processor 和 lazy factory**

将 processor 构造函数改为：

```python
def __init__(
    self,
    kernel: str = "fa",
    *,
    use_packed_qkv_a2a: bool = False,
    use_prefix_all_gather_into_tensor: bool = False,
):
    normalized_kernel = normalize_cogvideox_attention_backend(kernel)
    if normalized_kernel not in {"fa", "sdpa"}:
        raise ValueError(
            f"Unsupported CogVideoX SP attention kernel {kernel!r}; "
            "expected 'fa' or 'sdpa'."
        )
    self.kernel = normalized_kernel
    self._attention_backend = f"{self.kernel}_sp"
    self.use_packed_qkv_a2a = use_packed_qkv_a2a
    self.use_prefix_all_gather_into_tensor = (
        use_prefix_all_gather_into_tensor
    )
```

processor 调用 factory 时传入两个字段：

```python
usp_attn = _get_cogvideox_sp_usp_attention(
    num_heads=num_heads,
    head_size=head_dim,
    kernel=self.kernel,
    use_packed_qkv_a2a=self.use_packed_qkv_a2a,
    use_prefix_all_gather_into_tensor=(
        self.use_prefix_all_gather_into_tensor
    ),
)
```

factory 签名和构造调用改为：

```python
@lru_cache(maxsize=16)
def _get_cogvideox_sp_usp_attention(
    *,
    num_heads: int,
    head_size: int,
    kernel: str,
    use_packed_qkv_a2a: bool = False,
    use_prefix_all_gather_into_tensor: bool = False,
) -> USPAttention:
    if kernel == "fa":
        supported_attention_backends = {
            AttentionBackendEnum.FA,
            AttentionBackendEnum.FA2,
        }
    elif kernel == "sdpa":
        supported_attention_backends = {AttentionBackendEnum.TORCH_SDPA}
    else:
        raise ValueError(
            f"Unsupported CogVideoX SP attention kernel {kernel!r}; "
            "expected 'fa' or 'sdpa'."
        )
    return USPAttention(
        num_heads=num_heads,
        head_size=head_size,
        softmax_scale=None,
        causal=False,
        supported_attention_backends=supported_attention_backends,
        prefix=f"cogvideox_sp_attn_{kernel}_{num_heads}_{head_size}",
        use_packed_qkv_a2a=use_packed_qkv_a2a,
        use_prefix_all_gather_into_tensor=(
            use_prefix_all_gather_into_tensor
        ),
    )
```

这里保留现有 kernel 分支的完整实现，用上面的新签名和新增参数包住原逻辑；`maxsize=16` 覆盖 `2 kernels × 2 packed states × 2 gather states` 并为测试尺寸留余量。

- [ ] **步骤 4：增加 processor 配置与检查 helper**

在同一文件加入：

```python
def configure_cogvideox_usp_collectives(
    module: nn.Module,
    *,
    use_packed_qkv_a2a: bool,
    use_prefix_all_gather_into_tensor: bool,
) -> int:
    applied = 0
    for child in module.modules():
        if not isinstance(child, Attention):
            continue
        processor = child.processor
        if not isinstance(processor, CogVideoXSPAttnProcessor):
            continue
        processor.use_packed_qkv_a2a = use_packed_qkv_a2a
        processor.use_prefix_all_gather_into_tensor = (
            use_prefix_all_gather_into_tensor
        )
        applied += 1
    if applied == 0:
        raise ValueError("No CogVideoX SP attention processors were found.")
    return applied


def inspect_cogvideox_usp_collectives(
    module: nn.Module | None,
) -> dict[str, bool] | None:
    if module is None:
        return None
    states = {
        (
            child.processor.use_packed_qkv_a2a,
            child.processor.use_prefix_all_gather_into_tensor,
        )
        for child in module.modules()
        if isinstance(child, Attention)
        and isinstance(child.processor, CogVideoXSPAttnProcessor)
    }
    if not states:
        return None
    if len(states) != 1:
        raise RuntimeError("CogVideoX SP collective configuration is inconsistent.")
    packed, prefix_gather = states.pop()
    return {
        "packed_qkv_a2a": packed,
        "prefix_all_gather_into_tensor": prefix_gather,
    }
```

- [ ] **步骤 5：测试 helper 只配置 SP processor**

使用该测试文件现有的 `_DummyCogVideoXAttentionModule` 创建两个真实 Diffusers `Attention` 子模块，然后断言：

```python
module = nn.Module()
module.sp_attention = _DummyCogVideoXAttentionModule()
module.local_attention = _DummyCogVideoXAttentionModule()
set_cogvideox_attention_backend(module.sp_attention, "fa_sp")
set_cogvideox_attention_backend(module.local_attention, "sdpa")

applied = configure_cogvideox_usp_collectives(
    module,
    use_packed_qkv_a2a=True,
    use_prefix_all_gather_into_tensor=False,
)
self.assertEqual(applied, 1)
self.assertEqual(
    inspect_cogvideox_usp_collectives(module),
    {
        "packed_qkv_a2a": True,
        "prefix_all_gather_into_tensor": False,
    },
)
self.assertIsInstance(
    module.local_attention.attn.processor,
    CogVideoXSDPAAttnProcessor,
)
```

测试文件 import 列表同时加入 `CogVideoXSDPAAttnProcessor`、`configure_cogvideox_usp_collectives` 和 `inspect_cogvideox_usp_collectives`。

- [ ] **步骤 6：运行 attention backend 测试**

运行：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_attention_backend.py -q
```

预期：全部 PASS，现有 `fa`、`sdpa`、`fa_sp`、`sdpa_sp` backend 选择测试不变。

- [ ] **步骤 7：提交 CogVideoX 配置传播**

```bash
git add \
  python/sglang/multimodal_gen/runtime/models/dits/cogvideox_attention_backend.py \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_attention_backend.py
git commit -m "feat(vividvr): propagate usp collective options to cogvideox"
```

## 任务 5：在 VividVR pipeline 中 fail-fast 并记录 effective 状态

**文件：**
- 修改：`python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py:382-455`
- 修改：`python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py:527-600`
- 修改：`python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py:904-912`
- 测试：`python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_attention_backend.py`
- 测试：`python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_inference_tool.py`

- [ ] **步骤 1：编写非法拓扑和 debug 失败测试**

在现有 pipeline `__new__` 测试模式中加入：

```python
def test_usp_collective_optimization_rejects_single_rank(self):
    pipeline = object.__new__(VividVRPipeline)
    pipeline.modules = {
        "transformer": _PipelineHookModule(),
        "controlnet": _PipelineHookModule(),
    }
    args = ServerArgs(
        model_path="/tmp/model",
        sp_degree=1,
        ulysses_degree=1,
        enable_usp_packed_qkv_a2a=True,
    )
    with self.assertRaisesRegex(ValueError, "requires Ulysses degree greater than 1"):
        pipeline._apply_usp_collective_optimizations(args)

def test_runtime_debug_reports_requested_and_effective_collectives(self):
    pipeline = object.__new__(VividVRPipeline)
    pipeline.modules = {
        "transformer": _PipelineHookModule(),
        "controlnet": _PipelineHookModule(),
    }
    args = ServerArgs(
        model_path="/tmp/model",
        attention_backend="fa",
        sp_degree=2,
        ulysses_degree=2,
        enable_usp_packed_qkv_a2a=True,
        enable_usp_prefix_all_gather_into_tensor=False,
    )
    pipeline._apply_attention_backend(args)
    pipeline._apply_usp_collective_optimizations(args)
    debug = pipeline._build_runtime_acceleration_debug(args)
    self.assertTrue(debug["usp_packed_qkv_a2a_requested"])
    self.assertTrue(debug["usp_transformer"]["packed_qkv_a2a"])
    self.assertFalse(
        debug["usp_controlnet"]["prefix_all_gather_into_tensor"]
    )
```

- [ ] **步骤 2：运行测试并确认 pipeline 方法缺失**

运行：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_attention_backend.py \
  -k 'usp_collective_optimization or effective_collectives' -q
```

预期：FAIL，错误包含 `has no attribute '_apply_usp_collective_optimizations'`。

- [ ] **步骤 3：实现 pipeline 应用方法**

在 pipeline 中导入任务 4 的 configure/inspect helper，并加入：

```python
def _apply_usp_collective_optimizations(self, server_args: ServerArgs) -> None:
    use_packed = bool(server_args.enable_usp_packed_qkv_a2a)
    use_prefix_gather = bool(
        server_args.enable_usp_prefix_all_gather_into_tensor
    )
    if not use_packed and not use_prefix_gather:
        return

    ulysses_degree = getattr(server_args, "ulysses_degree", None) or 1
    if ulysses_degree <= 1:
        raise ValueError(
            "USP collective optimizations require Ulysses degree greater than 1."
        )
    runtime_choice = resolve_cogvideox_attention_runtime_choice(
        server_args.attention_backend,
        sp_enabled=True,
    )
    if runtime_choice.semantics != "ulysses_sp":
        raise ValueError(
            "USP collective optimizations require Ulysses SP attention semantics."
        )

    for component_name in ("transformer", "controlnet"):
        component = self.get_module(component_name)
        if component is None:
            continue
        applied = configure_cogvideox_usp_collectives(
            component,
            use_packed_qkv_a2a=use_packed,
            use_prefix_all_gather_into_tensor=use_prefix_gather,
        )
        logger.info(
            "Configured %d %s USP attention processors: packed_qkv_a2a=%s, "
            "prefix_all_gather_into_tensor=%s.",
            applied,
            component_name,
            use_packed,
            use_prefix_gather,
        )
```

- [ ] **步骤 4：固定应用顺序**

在 `load_modules()` 中把调用放在 backend 已设置、所有 fusion 和 compile 尚未开始的位置：

```python
self._apply_attention_backend(server_args)
self._apply_usp_collective_optimizations(server_args)
self._apply_qk_norm_fusion(server_args)
self._apply_qk_norm_rope_fusion(server_args)
self._apply_modulation_fusion(server_args)
self._apply_qkv_fusion(server_args)
self._apply_torch_compile(server_args)
```

这个顺序保证 processor 已经是 `fa_sp`/`sdpa_sp`，同时 compile 捕获的是最终配置。

- [ ] **步骤 5：扩展 runtime debug 和 snapshot**

在 `_build_runtime_acceleration_debug()` 的返回字典加入：

```python
"usp_packed_qkv_a2a_requested": bool(
    server_args.enable_usp_packed_qkv_a2a
),
"usp_prefix_all_gather_into_tensor_requested": bool(
    server_args.enable_usp_prefix_all_gather_into_tensor
),
"usp_transformer": inspect_cogvideox_usp_collectives(transformer),
"usp_controlnet": inspect_cogvideox_usp_collectives(controlnet),
```

在 `build_runtime_config_snapshot()` 加入 effective 字段：

```python
"usp_packed_qkv_a2a_requested": bool(
    server_args.enable_usp_packed_qkv_a2a
),
"usp_prefix_all_gather_into_tensor_requested": bool(
    server_args.enable_usp_prefix_all_gather_into_tensor
),
"usp_transformer": _json_ready(debug.get("usp_transformer")),
"usp_controlnet": _json_ready(debug.get("usp_controlnet")),
```

- [ ] **步骤 6：运行 pipeline 和 snapshot 测试**

运行：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_attention_backend.py \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_inference_tool.py \
  -q
```

预期：全部 PASS；默认关闭时 debug 的 requested 为 `False`，effective 为 processor 中的 `False`。

- [ ] **步骤 7：提交 pipeline 配置和可观测性**

```bash
git add \
  python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py \
  python/sglang/multimodal_gen/tools/run_vividvr_inference.py \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_attention_backend.py \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_inference_tool.py
git commit -m "feat(vividvr): configure and report usp collective optimizations"
```

## 任务 6：用真实两卡 NCCL 验证 collective 精确等价

**文件：**
- 创建：`test/srt/multimodal_gen/test_usp_packed_collectives_distributed.py`
- 测试：`python/sglang/multimodal_gen/test/unit/test_usp_packed_collectives.py`

- [ ] **步骤 1：创建可由 torchrun 直接执行的两卡测试**

```python
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist

import sglang.multimodal_gen.runtime.layers.usp as usp


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))

    usp.get_ulysses_parallel_world_size = lambda: world_size
    usp.get_sp_group = lambda: SimpleNamespace(ulysses_group=dist.group.WORLD)

    base = torch.arange(
        1 * 7 * 8 * 16,
        device=device,
        dtype=torch.bfloat16,
    ).reshape(1, 7, 8, 16)
    q = base + rank * 10000
    k = base + rank * 10000 + 1000
    v = base + rank * 10000 + 2000

    legacy = tuple(
        usp._usp_input_all_to_all(x, head_dim=2) for x in (q, k, v)
    )
    packed = usp._usp_input_all_to_all_qkv(q, k, v)
    for actual, expected in zip(packed, legacy, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    out_rep = packed[0][:, :3].contiguous()
    gathered_list = [torch.empty_like(out_rep) for _ in range(world_size)]
    dist.all_gather(gathered_list, out_rep, group=dist.group.WORLD)
    expected_prefix = torch.cat(gathered_list, dim=2)
    actual_prefix = usp._usp_prefix_all_gather(out_rep)
    torch.testing.assert_close(actual_prefix, expected_prefix, rtol=0, atol=0)

    dist.barrier()
    if rank == 0:
        print("PASS: packed QKV A2A and prefix gather are bitwise exact")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

- [ ] **步骤 2：运行新建的真实 NCCL 等价测试**

运行：

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=python \
  /home/zhiheng/sglang/.venv/bin/torchrun \
  --standalone --nproc-per-node=2 \
  test/srt/multimodal_gen/test_usp_packed_collectives_distributed.py
```

预期：rank 0 输出 `PASS: packed QKV A2A and prefix gather are bitwise exact`，两个进程退出码均为 0。该任务只增加跨进程集成覆盖，不再修改 collective 实现。

- [ ] **步骤 3：重复运行一次排除偶发 collective 顺序问题**

运行同一条 torchrun 命令。

预期：第二次仍输出相同 PASS，两个进程退出码均为 0，无 NCCL hang。

- [ ] **步骤 4：运行 CFG/SP 结构回归**

运行：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  test/srt/multimodal_gen/test_vividvr_cfg_parallel_stage.py \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_sequence_shard.py \
  -q
```

预期：全部 PASS；CFG 分组、SP 分组、cond/uncond 合并与 temporal shard 合同不变。

- [ ] **步骤 5：运行相关单元测试集合**

运行：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_usp_packed_collectives.py \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_attention_backend.py \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_inference_tool.py \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_sequence_shard.py \
  test/srt/multimodal_gen/test_vividvr_cfg_parallel_stage.py \
  -q
```

预期：全部 PASS。

- [ ] **步骤 6：提交分布式测试**

```bash
git add test/srt/multimodal_gen/test_usp_packed_collectives_distributed.py
git commit -m "test(vividvr): verify usp collectives on two nccl ranks"
```

## 任务 7：执行 B0/P1/P2/P3 profiler 消融和 compile smoke

**文件：**
- 修改：`python/sglang/multimodal_gen/tools/run_vividvr_inference.py:403-437`
- 修改：`python/sglang/multimodal_gen/tools/run_vividvr_inference.py:580-735`
- 修改：`python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_inference_tool.py`
- 创建：`docs_xzh/distribute/vividvr_usp_collective_ablation_20260710.md`

四组配置固定为：

| 配置 | packed QKV A2A | prefix tensor gather |
|---|---:|---:|
| B0 | 关闭 | 关闭 |
| P1 | 开启 | 关闭 |
| P2 | 关闭 | 开启 |
| P3 | 开启 | 开启 |

- [ ] **步骤 1：编写直跑 profiler 参数失败测试**

在 inference tool 测试中加入：

```python
def test_parse_args_and_request_support_profiler_fields(self):
    argv = [
        "run_vividvr_inference.py",
        "--input-video",
        "/tmp/input.mp4",
        "--profile",
        "--num-profiled-timesteps",
        "3",
    ]
    with patch.object(sys, "argv", argv):
        args = parse_args()
    self.assertTrue(args.profile)
    self.assertEqual(args.num_profiled_timesteps, 3)

    server_args = ServerArgs(model_path="/tmp/model")
    with patch(
        "sglang.multimodal_gen.tools.run_vividvr_inference.prepare_request",
        side_effect=lambda _server_args, params: params,
    ):
        params = build_request(
            server_args=server_args,
            args=args,
            output_file_name="profile.mp4",
        )
    self.assertTrue(params.profile)
    self.assertEqual(params.num_profiled_timesteps, 3)
```

- [ ] **步骤 2：运行测试并确认 profiler CLI 缺失**

运行：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_inference_tool.py \
  -k 'profiler_fields' -q
```

预期：FAIL，错误包含 `unrecognized arguments: --profile --num-profiled-timesteps 3`。

- [ ] **步骤 3：接入现有 SGLDiffusionProfiler 请求字段**

在 parser 中加入：

```python
parser.add_argument(
    "--profile",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Enable the existing denoising-stage torch profiler.",
)
parser.add_argument(
    "--num-profiled-timesteps",
    type=int,
    default=5,
    help="Number of denoising timesteps captured after one profiler warmup step.",
)
```

在 `build_request()` 的 `request_kwargs` 加入：

```python
"profile": args.profile,
"num_profiled_timesteps": args.num_profiled_timesteps,
```

在 snapshot/最终报告加入同名字段，确保 profiler 产物可追溯到运行参数。

- [ ] **步骤 4：运行 inference tool 完整测试**

运行：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_inference_tool.py -q
```

预期：全部 PASS。

- [ ] **步骤 5：检查四卡空闲和固定输入文件**

运行：

```bash
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
test -f /home/zhiheng/input/test_video_long_960x720_130f.mp4
test -f /home/zhiheng/sglang/Vivid_Acceptance/captions/service_sidecars/quad-test-video-long-960x720-130f-run2-20260708T060202Z.txt
```

预期：第一条无计算进程输出；两条 `test -f` 退出码均为 0。若 GPU 有占用，等待占用释放后重新执行，不并发启动 benchmark。

- [ ] **步骤 6：在单个 tmux session 中顺序运行四组 5-step profiler smoke**

```bash
tmux new-session -d -s vividvr_usp_ablation \
  'cd /home/zhiheng/sglang && \
   set -euo pipefail && \
   export CUDA_VISIBLE_DEVICES=0,1,2,3 && \
   export PYTHONPATH=python && \
   export PYTHONUNBUFFERED=1 && \
   export SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE=eager_global && \
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
   mkdir -p Vivid_Acceptance/logs/usp_ablation Vivid_Acceptance/profiles/usp_ablation Vivid_Acceptance/result_videos/usp_ablation Vivid_Acceptance/indicator/usp_ablation && \
   for spec in "B0|" "P1|--enable-usp-packed-qkv-a2a" "P2|--enable-usp-prefix-all-gather-into-tensor" "P3|--enable-usp-packed-qkv-a2a --enable-usp-prefix-all-gather-into-tensor"; do \
     name=${spec%%|*}; flags=${spec#*|}; \
     export SGLANG_TORCH_PROFILER_DIR=/home/zhiheng/sglang/Vivid_Acceptance/profiles/usp_ablation/${name}; \
     mkdir -p "$SGLANG_TORCH_PROFILER_DIR"; \
     /home/zhiheng/sglang/.venv/bin/torchrun \
       --nproc_per_node=4 --master_port=30310 \
       python/sglang/multimodal_gen/tools/run_vividvr_inference.py \
       --cogvideox-ckpt-path /home/zhiheng/ckpts/CogVideoX1.5-5B \
       --vividvr-ckpt-path /home/zhiheng/ckpts/Vivid-VR \
       --input-video /home/zhiheng/input/test_video_long_960x720_130f.mp4 \
       --caption-file /home/zhiheng/sglang/Vivid_Acceptance/captions/service_sidecars/quad-test-video-long-960x720-130f-run2-20260708T060202Z.txt \
       --output-dir /home/zhiheng/sglang/Vivid_Acceptance/result_videos/usp_ablation \
       --report-dir /home/zhiheng/sglang/Vivid_Acceptance/indicator/usp_ablation \
       --artifact-prefix vividvr_usp_${name}_5step \
       --phase-label E \
       --mode-label cfg2_sp2_${name} \
       --num-temporal-process-frames 121 \
       --num-inference-steps 5 \
       --upscale 1.0 \
       --seed 42 \
       --num-gpus 4 --tp-size 1 --sp-degree 2 --ulysses-degree 2 --ring-degree 1 \
       --enable-cfg-parallel --attention-backend fa --enable-torch-compile \
       --profile --num-profiled-timesteps 3 \
       $flags \
       2>&1 | tee Vivid_Acceptance/logs/usp_ablation/${name}.log; \
   done'
```

只读查看：

```bash
tmux attach -r -t vividvr_usp_ablation
```

预期：B0、P1、P2、P3 均完成 5 steps，P3 完成即证明两个开关可以和 `torch.compile` 共存；每组 profiler 目录至少有一个 `global-rank0.trace.json.gz`。

- [ ] **步骤 7：汇总 collective count 和 trace duration**

运行：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python - <<'PY'
import gzip
import json
from pathlib import Path

root = Path("Vivid_Acceptance/profiles/usp_ablation")
summary = {}
for variant in ("B0", "P1", "P2", "P3"):
    traces = sorted((root / variant).glob("*global-rank0.trace.json.gz"))
    assert traces, f"missing trace for {variant}"
    with gzip.open(traces[-1], "rt") as f:
        events = json.load(f)["traceEvents"]
    selected = {
        "all_to_all": [],
        "all_gather": [],
    }
    for event in events:
        name = str(event.get("name", "")).lower()
        duration = float(event.get("dur", 0.0))
        if "all_to_all" in name or "alltoall" in name:
            selected["all_to_all"].append(duration)
        if "all_gather" in name or "allgather" in name:
            selected["all_gather"].append(duration)
    summary[variant] = {
        key: {"count": len(values), "duration_us": sum(values)}
        for key, values in selected.items()
    }
print(json.dumps(summary, indent=2, sort_keys=True))
PY
```

预期关系：

- P1 的 input A2A 发起次数相对 B0 减少约三分之二，总 A2A 数约减半。
- P2 的 A2A count 与 B0 相同，prefix gather 实现名称/count 发生变化。
- P3 同时满足 P1 的 A2A count 和 P2 的 gather 路径。
- P1/P3 若没有降低 A2A count，停止正式 benchmark，先修正 packed 分支是否实际生效。

- [ ] **步骤 8：写入消融文档并提交 profiler 接入**

`docs_xzh/distribute/vividvr_usp_collective_ablation_20260710.md` 必须记录：commit、GPU 型号/拓扑、四组完整命令、trace 路径、A2A count/duration、gather count/duration、5-step model inference 时间、是否 compile 成功。然后提交：

```bash
git add \
  python/sglang/multimodal_gen/tools/run_vividvr_inference.py \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_inference_tool.py \
  docs_xzh/distribute/vividvr_usp_collective_ablation_20260710.md
git commit -m "perf(vividvr): profile usp collective variants"
```

## 任务 8：执行四卡正式服务验收并收口文档

**文件：**
- 修改：`docs_xzh/distribute/vividvr_usp_collective_ablation_20260710.md`
- 修改：`docs_xzh/run_command/vividvr_default_run_and_serve_commands.md`
- 修改：`docs_xzh/hand_over/vividvr_cfg_parallel_v2_equivalence_handover_20260709.md`
- 产物：`Vivid_Acceptance/indicator/service_benchmark/${FORMAL_TASK_ID}_perf.json`
- 产物：`Vivid_Acceptance/indicator/service_benchmark/${FORMAL_TASK_ID}_compare.json`
- 产物：`Vivid_Acceptance/indicator/service_benchmark/${FORMAL_TASK_ID}_acceptance_summary.json`
- 产物：`Vivid_Acceptance/result_videos/service_benchmark/${FORMAL_TASK_ID}.mp4`
- 产物：`Vivid_Acceptance/result_videos/service_benchmark/downloads/${FORMAL_TASK_ID}.bridge-downloaded.mp4`
- 产物：`Vivid_Acceptance/logs/mock_callback_*.jsonl`

**正式验收执行合同：**

- 任务 8 的服务与请求验收必须以 `docs_xzh/run_command/mock_test.md` 为准，不能只运行内部 acceptance helper 代替正式服务链路。
- 严格按该文档第 3、4 节启动 Moto S3 和 callback receiver，并确认 `flowcut` bucket 已创建。
- 严格按第 5.3 节启动固定 caption sidecar mock，再使用同节的四卡 `CFG=2 x SP=2` `sglang serve` 命令启动主服务；P3 只在该命令上追加 `--enable-usp-packed-qkv-a2a` 与 `--enable-usp-prefix-all-gather-into-tensor`。
- warmup 和 formal 都必须按第 6 节通过外部 `POST /v1/videos/repairs/flowcut` 提交，包含必填 `callbackUrl`、`outputObjectKey` 和 `minioConfig`，轮询到 completed，并验证 callback 终态与 Moto S3 对象上传。
- 正式质量对比必须使用从 Moto S3 下载的 `bridge-downloaded.mp4`，而不是只使用服务本地 `output_path`，从而覆盖正式 FlowCut 对外契约。
- 上述 Moto、callback、caption mock、P3 主服务、warmup 和 formal 推理进程均在命名清晰的独立 `tmux` session 中启动；启动前记录只读 attach 命令。

- [ ] **步骤 1：执行最终静态和自动化回归**

运行：

```bash
git diff --check
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_usp_packed_collectives.py \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_attention_backend.py \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_inference_tool.py \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_sequence_shard.py \
  test/srt/multimodal_gen/test_vividvr_cfg_parallel_stage.py \
  -q
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=python \
  /home/zhiheng/sglang/.venv/bin/torchrun \
  --standalone --nproc-per-node=2 \
  test/srt/multimodal_gen/test_usp_packed_collectives_distributed.py
```

预期：`git diff --check` 无输出；pytest 全部 PASS；torchrun 输出 bitwise exact PASS。

- [ ] **步骤 2：按 mock 服务链启动依赖和 P3 四卡正式服务**

先严格执行 `docs_xzh/run_command/mock_test.md` 第 3、4、5.3 节：启动 `vividvr_moto_s3`、`vividvr_flowcut_callback_receiver`、固定 caption 的 `vividvr_caption_sidecar_mock`，创建 `flowcut` bucket，并分别通过 S3 列表、callback 监听端口与 caption `/health` 检查。然后确认 `31232`、`30232`、`56232` 未占用且 GPU 无本任务外计算进程，再运行：

```bash
tmux new-session -d -s vividvr_usp_p3_formal_service \
  'cd /home/zhiheng/sglang && \
   mkdir -p Vivid_Acceptance/logs Vivid_Acceptance/result_videos/service_benchmark Vivid_Acceptance/indicator/service_benchmark && \
   export CUDA_VISIBLE_DEVICES=0,1,2,3 && \
   export PYTHONUNBUFFERED=1 && \
   export PYTHONPATH=python && \
   export NO_PROXY=127.0.0.1,localhost && \
   export AWS_EC2_METADATA_DISABLED=true && \
   export SGLANG_FLOWCUT_PROGRESS_INTERVAL_SECONDS=5 && \
   export SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE=eager_global && \
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
   /home/zhiheng/sglang/.venv/bin/sglang serve \
     --model-path /home/zhiheng/ckpts/CogVideoX1.5-5B \
     --model-id VividVR \
     --pipeline-class-name CogVideoXVividVRControlNetPipeline \
     --component-paths.vividvr /home/zhiheng/ckpts/Vivid-VR \
     --num-gpus 4 --tp-size 1 --sp-degree 2 --ulysses-degree 2 --ring-degree 1 \
     --enable-cfg-parallel --vividvr-parallel-mode cfg_sp \
     --enable-torch-compile --attention-backend fa \
     --enable-usp-packed-qkv-a2a \
     --enable-usp-prefix-all-gather-into-tensor \
     --dist-timeout 3600 \
     --host 127.0.0.1 --port 31232 --master-port 30232 --scheduler-port 56232 \
     --strict-ports --input-save-path "" \
     --output-path /home/zhiheng/sglang/Vivid_Acceptance/result_videos/service_benchmark \
     --vividvr-caption-bridge \
     --vividvr-caption-sidecar-url http://127.0.0.1:31200 \
     --vividvr-caption-work-dir /home/zhiheng/sglang/Vivid_Acceptance/captions/service_sidecars \
     --vividvr-caption-sidecar-timeout 1800 \
     2>&1 | tee Vivid_Acceptance/logs/vividvr_usp_p3_formal_service_$(date -u +%Y%m%dT%H%M%SZ).log'
```

只读查看：

```bash
tmux attach -r -t vividvr_usp_p3_formal_service
```

预期：服务 ready；启动日志显示 effective backend `fa_sp`，两个 USP 优化在 transformer/controlnet 均为 `True`。

- [ ] **步骤 3：按外部 FlowCut 契约提交一次完整 warmup 请求**

服务 ready 后，在 `vividvr_usp_p3_warmup` tmux session 中严格执行 `docs_xzh/run_command/mock_test.md` 第 6.1 至 6.4 节。请求必须使用下面的任务 ID 与路径，并包含文档中的 `callbackUrl`、`outputObjectKey`、`perf_dump_path` 和完整 `minioConfig`；固定 caption 由 sidecar mock 经 HTTP bridge 产出，请求中不直接传 `caption_file`：

```bash
export WARMUP_TASK_ID=vividvr-usp-p3-warmup-$(date -u +%Y%m%dT%H%M%SZ)
export BRIDGE_BASE_URL=http://127.0.0.1:31232
export CALLBACK_BASE_URL=http://127.0.0.1:39090
export MOTO_S3_ENDPOINT=127.0.0.1:4566
export MOTO_S3_BUCKET=flowcut
export MOTO_S3_ACCESS_KEY=test
export MOTO_S3_SECRET_KEY=test
```

只读查看：`tmux attach -r -t vividvr_usp_p3_warmup`。

预期：任务 completed，输出 130 帧，callback 最终 `status=succeeded`，S3 中存在带 `.mp4` 后缀的对象；该请求只用于 compile/cache warmup，不计入正式性能。

- [ ] **步骤 4：按外部 FlowCut 契约提交正式请求**

确认 warmup session 已退出，再在 `vividvr_usp_p3_formal` tmux session 中重复 `mock_test.md` 第 6.1 至 6.4 节。除了任务 ID 和产物路径外，请求字段必须与 warmup 完全一致；完成后按第 6.4 节从 Moto S3 下载对象：

```bash
export FORMAL_TASK_ID=vividvr-usp-p3-formal-$(date -u +%Y%m%dT%H%M%SZ)
export DOWNLOADED_RESULT_VIDEO=/home/zhiheng/sglang/Vivid_Acceptance/result_videos/service_benchmark/downloads/${FORMAL_TASK_ID}.bridge-downloaded.mp4
```

只读查看：`tmux attach -r -t vividvr_usp_p3_formal`。

预期：任务 completed；callback 最终 succeeded；Moto S3 对象存在且可下载；perf JSON 的 debug 同时满足 `cfg_world_size=2`、`sp_world_size=2`、`attention_backend_transformer=fa_sp`、两个 USP effective 开关为 `True`。

- [ ] **步骤 5：执行固定 reference SSIM 对比**

```bash
export VIVIDVR_CFG_REFERENCE_VIDEO=/home/zhiheng/sglang/Vivid_Acceptance/result_videos/service_benchmark/downloads/quad-test-video-long-960x720-130f-run2-20260708T060202Z.bridge-downloaded.mp4
export FORMAL_VIDEO=/home/zhiheng/sglang/Vivid_Acceptance/result_videos/service_benchmark/downloads/${FORMAL_TASK_ID}.bridge-downloaded.mp4
export FORMAL_COMPARE=/home/zhiheng/sglang/Vivid_Acceptance/indicator/service_benchmark/${FORMAL_TASK_ID}_compare.json
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python \
  python/sglang/multimodal_gen/runtime/videoedit/compare.py \
  --reference "${VIVIDVR_CFG_REFERENCE_VIDEO}" \
  --candidate "${FORMAL_VIDEO}" \
  --output-json "${FORMAL_COMPARE}" \
  --min-ssim 0.98 \
  --max-failed-frame-ratio 0
```

预期：退出码 0，`summary.ssim_mean >= 0.98`、`summary.ssim_min >= 0.98`、`summary.pass_compare = true`。

- [ ] **步骤 6：生成标准 acceptance summary 并执行性能门禁**

```bash
export FORMAL_PERF=/home/zhiheng/sglang/Vivid_Acceptance/indicator/service_benchmark/${FORMAL_TASK_ID}_perf.json
export FORMAL_SUMMARY=/home/zhiheng/sglang/Vivid_Acceptance/indicator/service_benchmark/${FORMAL_TASK_ID}_acceptance_summary.json
export WARMUP_TASK_ID FORMAL_TASK_ID FORMAL_PERF FORMAL_COMPARE FORMAL_VIDEO FORMAL_SUMMARY VIVIDVR_CFG_REFERENCE_VIDEO
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

perf = json.loads(Path(os.environ["FORMAL_PERF"]).read_text())
compare = json.loads(Path(os.environ["FORMAL_COMPARE"]).read_text())["summary"]
debug = perf["meta"]["vividvr_debug"]
denoising_ms = next(
    step["duration_ms"]
    for step in perf["steps"]
    if step["name"] == "VividVRMultiClipDenoisingStage"
)
summary = {
    "task_id": os.environ["FORMAL_TASK_ID"],
    "warmup_task_id": os.environ["WARMUP_TASK_ID"],
    "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "parallel_mode": debug["vividvr_parallel_mode"],
    "cfg_parallel_enabled": debug["cfg_parallel_enabled"],
    "cfg_world_size": debug["cfg_world_size"],
    "sp_world_size": debug["sp_world_size"],
    "reference_video": os.environ["VIVIDVR_CFG_REFERENCE_VIDEO"],
    "candidate_video": os.environ["FORMAL_VIDEO"],
    "perf_json": os.environ["FORMAL_PERF"],
    "compare_json": os.environ["FORMAL_COMPARE"],
    "total_runtime_seconds": perf["total_duration_ms"] / 1000.0,
    "model_inference_runtime_seconds": denoising_ms / 1000.0,
    "denoising_runtime_seconds": denoising_ms / 1000.0,
    "output_num_frames": debug["output_num_frames"],
    "prompt_embed_shape": debug["prompt_embed_shape"],
    "ssim_mean": compare["ssim_mean"],
    "ssim_min": compare["ssim_min"],
    "compared_frames": compare["compared_frames"],
    "pass_compare": compare["pass_compare"],
    "failed_frames": compare["failed_frames"],
    "thresholds": compare["thresholds"],
    "usp_transformer": debug["usp_transformer"],
    "usp_controlnet": debug["usp_controlnet"],
}
Path(os.environ["FORMAL_SUMMARY"]).write_text(
    json.dumps(summary, indent=2) + "\n"
)
assert summary["model_inference_runtime_seconds"] < 175.0, summary
assert summary["ssim_mean"] >= 0.98, summary
assert summary["ssim_min"] >= 0.98, summary
assert summary["pass_compare"] is True, summary
assert summary["cfg_world_size"] == 2, summary
assert summary["sp_world_size"] == 2, summary
print(json.dumps(summary, indent=2))
PY
```

预期：脚本退出码 0，summary 字段集合不小于现有正式基线，并包含两个 USP effective 状态。任何 assert 失败都表示第一阶段尚未验收，不得修改默认配置表述。

- [ ] **步骤 7：根据门禁结果收口文档**

门禁全部通过时：

- 在 `docs_xzh/distribute/vividvr_usp_collective_ablation_20260710.md` 写入 formal summary、相对 `194.2424s` 的绝对/百分比收益及 SSIM。
- 在 `docs_xzh/run_command/vividvr_default_run_and_serve_commands.md` 增加“可选 USP 通信优化”命令段；本计划不把开关改成全局默认值。
- 在 handover 末尾追加 commit、warmup/formal task id、perf/compare/summary/video 路径、回退方式。
- 回退命令为删除 `--enable-usp-packed-qkv-a2a` 和 `--enable-usp-prefix-all-gather-into-tensor`，无需代码回退。

性能或质量任一门禁失败时：只把 B0/P1/P2/P3 和 formal 实测数据写入 distribute 消融文档，保留两个开关默认关闭，不改默认命令文档和 handover 的正式基线结论。

- [ ] **步骤 8：提交验收文档并推送阶段提交**

门禁通过时执行：

```bash
git add \
  docs_xzh/distribute/vividvr_usp_collective_ablation_20260710.md \
  docs_xzh/run_command/vividvr_default_run_and_serve_commands.md \
  docs_xzh/hand_over/vividvr_cfg_parallel_v2_equivalence_handover_20260709.md
git commit -m "docs(vividvr): record usp collective acceptance"
git status --short
git log --oneline -8
git push
```

预期：提交只包含本计划相关文件；`git push` 成功。若门禁失败，提交范围只包含 distribute 消融文档，commit message 使用 `docs(vividvr): record usp collective experiment`。

## 第二阶段边界

只有在 P3 已证明 collective count 正确、数值质量通过，但正式 `model_inference_runtime_seconds` 仍不低于 `175s`，并且 profiler 显示 A2A CUDA duration 仍有可隐藏空间时，才另写独立计划评估按 head bucket 的异步 A2A/attention overlap。该独立计划必须重新定义 buffer 生命周期、stream/event 同步、compile graph break、bucket 大小消融和数值验收。

本计划明确不包含 Ring Attention、TP、custom CUDA/NCCL extension、低精度通信、改变 CFG 公式、改变 connector context、改变 clip orchestration 或更换正式 backend；这些方向的风险和验证面与本次 launch-count 优化不同。
