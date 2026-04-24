# 03 · Python 门面与上层模块接入（重点 B）

本章回答"上层 SGLang 怎么用 `sgl-kernel`"。分两层：

- `python/sgl_kernel/` 里的**门面层**（薄包装的 Python API）：它在 `torch.ops.sgl_kernel.*` 之上做哪些封装、为什么这样封装。
- `python/sglang/srt/`（以及 `python/sglang/multimodal_gen/`）里的**消费层**：以什么模式 import、如何在多硬件后端里分发、如何与 FlashInfer/aiter/vllm 等其它 kernel 库共存。

---

## 1. Python 门面层的通用模式

整个 `python/sgl_kernel/` 是**薄包装**。`__init__.py` 的主要职责是：

```1:12:sgl-kernel/python/sgl_kernel/__init__.py
import torch
from sgl_kernel.debug_utils import maybe_wrap_debug_kernel
from sgl_kernel.load_utils import _load_architecture_specific_ops, _preload_cuda_library

# Initialize the ops library based on current GPU
common_ops = _load_architecture_specific_ops()

if torch.version.cuda is not None:
    _preload_cuda_library()
```

然后逐一从子模块重新导出：

```13:109:sgl-kernel/python/sgl_kernel/__init__.py
from sgl_kernel.allreduce import *
from sgl_kernel.attention import (
    cutlass_mla_decode,
    cutlass_mla_get_workspace_size,
    merge_state_v2,
)
from sgl_kernel.cutlass_moe import cutlass_w4a8_moe_mm, get_cutlass_w4a8_moe_mm_data
from sgl_kernel.elementwise import (
    concat_mla_absorb_q,
    concat_mla_k,
    copy_to_gpu_no_ce,
    fused_add_rmsnorm,
    gelu_and_mul,
    gelu_tanh_and_mul,
    gemma_fused_add_rmsnorm,
    gemma_rmsnorm,
    rmsnorm,
    rotary_embedding,
    silu_and_mul,
)
...
```

之后把一批常用算子用 `maybe_wrap_debug_kernel` 包一层：

```115:191:sgl-kernel/python/sgl_kernel/__init__.py
_DEBUG_EXPORT_NAMES = [ "apply_shuffle_mul_sum", ..., "weak_ref_tensor" ]

if torch.version.hip is not None:
    _DEBUG_EXPORT_NAMES.append("gelu_quick")

for _name in _DEBUG_EXPORT_NAMES:
    if _name in globals():
        globals()[_name] = maybe_wrap_debug_kernel(
            globals()[_name], f"sgl_kernel.{_name}"
        )
```

`maybe_wrap_debug_kernel` 依赖环境变量：

```7:23:sgl-kernel/python/sgl_kernel/debug_utils.py
def _wrap_debug_kernel(func: F, op_name: str | None = None) -> F:
    try:
        if int(os.environ.get("SGLANG_KERNEL_API_LOGLEVEL", "0")) == 0:
            return func
    except Exception:
        return func
    try:
        from sglang.kernel_api_logging import debug_kernel_api
    except Exception:
        return func
    ...
```

所以默认无开销；`SGLANG_KERNEL_API_LOGLEVEL=1` 时会通过 `sglang.kernel_api_logging.debug_kernel_api` 把每次 op 的入参 shape/dtype/device 打印出来，配合 `debug-cuda-crash` 等 skill 排查"到底是哪个 kernel 进去炸的"。

以下是子模块里反复出现的 5 个模式。

### 1.1 输出 buffer 惰性分配

```258:270:sgl-kernel/python/sgl_kernel/elementwise.py
def silu_and_mul(input: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
    if input.shape[-1] * input.dtype.itemsize % 16 != 0:
        raise ValueError("The pointers must be multiple of 16 bytes.")
    if out is not None:
        _check_shape(input, out)
    else:
        out = torch.empty(
            input.shape[:-1] + (input.shape[-1] // 2,),
            device=input.device,
            dtype=input.dtype,
        )
    torch.ops.sgl_kernel.silu_and_mul.default(out, input)
    return out
```

- CUDA kernel 一律**要求调用方传 `out`**（in-place 风格），Python 门面负责"没传就 `torch.empty` 一个"。
- 同时做对齐检查（`* itemsize % 16 != 0`），因为 kernel 内部用 16-byte 向量化 load/store，不对齐会崩。
- 这样上层既能写 `y = silu_and_mul(x)`（用默认 out），也能写 `silu_and_mul(x, out=preallocated)`（复用 buffer），两种风格都自然。

### 1.2 FlashInfer / 自研 kernel 双路径 fallback

这是最重要的一种模式：`sgl-kernel` 里的某些算子在 FlashInfer 里有更新、更性能的 JIT 版本；`sgl-kernel` 的态度是"**有 FlashInfer 就用 FlashInfer，否则兜底到自己的 AOT 版本**"。

```107:122:sgl-kernel/python/sgl_kernel/elementwise.py
    # torch.compiler.is_dynamo_compiling(): FlashInfer norm paths are not safe under
    # torch.compile(..., fullgraph=True). Dynamo traces into FlashInfer's JIT module
    # loading path, which calls Path.exists() / os.stat() — both untraceable — causing
    # the entire compilation to fail. We fall back to the internal implementation while
    # tracing as a temporary workaround.
    if (
        _has_flashinfer
        and input.dtype in _FLASHINFER_NORM_SUPPORTED_DTYPES
        and not torch.compiler.is_dynamo_compiling()
    ):
        return _flashinfer_norm.rmsnorm(input, weight, eps, out, enable_pdl)
    else:
        return _rmsnorm_internal(input, weight, eps, out, enable_pdl)
```

采样也是一样：

```56:59:sgl-kernel/python/sgl_kernel/sampling.py
    if probs.device.type == "musa" or not _has_flashinfer:
        return _top_k_renorm_probs_internal(probs, *_to_tensor_scalar_tuple(top_k))
    else:
        return _flashinfer_sampling.top_k_renorm_probs(probs, top_k)
```

三点值得注意：

1. **dtype 白名单**：FlashInfer 只对 `fp16/bf16` 有高性能 kernel，`float32` 走自研版本。
2. **Dynamo 保护**：`torch.compiler.is_dynamo_compiling()` 时绕开 FlashInfer，因为 FlashInfer 的 JIT 触发 `Path.exists()` / `os.stat()`，会让 `torch.compile(fullgraph=True)` 失败。
3. **硬件豁免**：摩尔线程 `device.type == "musa"` 走自研版本，因为 FlashInfer 不支持 MUSA。

### 1.3 PDL / 架构能力自动开关

```57:66:sgl-kernel/python/sgl_kernel/utils.py
@cache_once
def is_arch_support_pdl() -> bool:
    if bool(torch.version.hip):
        return False
    try:
        device = torch.cuda.current_device()
        major, _ = torch.cuda.get_device_capability(device)
    except Exception:
        return False
    return major >= 9
```

```23:28:sgl-kernel/python/sgl_kernel/elementwise.py
def _rmsnorm_internal(...):
    if out is None:
        out = torch.empty_like(input)
    if enable_pdl is None:
        enable_pdl = is_arch_support_pdl()
    torch.ops.sgl_kernel.rmsnorm.default(out, input, weight, eps, enable_pdl)
```

所以上层完全可以写 `rmsnorm(x, w)`（`enable_pdl=None`），门面层会自动：Hopper+ 开 PDL（Programmatic Dependent Launch），Ampere/HIP 关 PDL，统一一套调用代码。

`cache_once` 用自实现而不是 `functools.lru_cache`，因为后者和 `torch.compile` 不兼容：

```40:54:sgl-kernel/python/sgl_kernel/utils.py
def cache_once(fn):
    """
    NOTE: `functools.lru_cache` is not compatible with `torch.compile`
    So we manually implement a simple cache_once decorator to replace it.
    """
    result_map = {}

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        if key not in result_map:
            result_map[key] = fn(*args, **kwargs)
        return result_map[key]

    return wrapper
```

### 1.4 Workspace 缓存

```21:30:sgl-kernel/python/sgl_kernel/utils.py
_cache_buf: Dict[Tuple[str, torch.device], torch.Tensor] = {}

def _get_cache_buf(name: str, bytes: int, device: torch.device) -> torch.Tensor:
    key = (name, device)
    buf = _cache_buf.get(key)
    if buf is None:
        buf = torch.empty(bytes, dtype=torch.uint8, device=device)
        _cache_buf[key] = buf
    return buf
```

典型用途是 `bmm_fp8` 的 cuBLAS workspace：

```65:81:sgl-kernel/python/sgl_kernel/gemm.py
def bmm_fp8(A, B, A_scale, B_scale, dtype, out=None):
    if out is None:
        out = torch.empty((A.shape[0], A.shape[1], B.shape[2]), device=A.device, dtype=dtype)
    workspace_buffer = _get_cache_buf("bmm_fp8_workspace", 32 * 1024 * 1024, A.device)
    _bmm_fp8_internal(workspace_buffer, A, B, out, A_scale, B_scale)
    return out
```

32 MB 的 cuBLAS workspace 进程级单例，避免每次调用都 alloc/free。

### 1.5 `scalar_type.py` — 把 Python 枚举桥到 C++

`sgl-kernel/python/sgl_kernel/scalar_type.py` 提供 `ScalarType` 枚举（fp8_e4m3 / fp8_e5m2 / int4 / uint4 等），对应 C++ 侧 `include/scalar_type.hpp`。它解决了 PyTorch 本身 `torch.dtype` 不覆盖 W4A8/FP8 子类型的问题——这些"伪 dtype"只在 kernel 侧用，不走 PyTorch tensor 的 dtype 系统。

---

## 2. 上层模块的 5 种接入模式

对 `python/sglang/` 主体代码 grep `from sgl_kernel` / `import sgl_kernel` 会得到 100+ 个文件。这些使用方式可以归为 5 类。

### 2.1 直接 `from sgl_kernel import <op>`（最常见）

适用于已知 device 是 CUDA、不需要多后端切换的场景。在 `sglang.srt.layers.*` 里俯拾即是：

**LayerNorm**（`python/sglang/srt/layers/layernorm.py`）：

```62:67:python/sglang/srt/layers/layernorm.py
    from sgl_kernel import (
        fused_add_rmsnorm,
        gemma_fused_add_rmsnorm,
        gemma_rmsnorm,
        rmsnorm,
    )
```

**forward 里直接用**：

```217:219:python/sglang/srt/layers/layernorm.py
            fused_add_rmsnorm(x, residual, self.weight.data, self.variance_epsilon)
            ...
        out = rmsnorm(x, self.weight.data, self.variance_epsilon)
```

**Attention backend**：

- `python/sglang/srt/layers/attention/flashattention_backend.py` → `from sgl_kernel.flash_attn import flash_attn_with_kvcache, flash_attn_varlen_func`
- `python/sglang/srt/layers/attention/cutlass_mla_backend.py` → `from sgl_kernel import cutlass_mla_decode, cutlass_mla_get_workspace_size`
- `python/sglang/srt/layers/attention/flashmla_backend.py` → `from sgl_kernel.flash_mla import ...`
- `python/sglang/srt/layers/attention/mamba/causal_conv1d.py` → `from sgl_kernel import causal_conv1d_fwd, causal_conv1d_update`

**MoE & 量化**：

- `python/sglang/srt/layers/moe/topk.py` → `moe_fused_gate, topk_softmax, topk_sigmoid`
- `python/sglang/srt/layers/moe/moe_runner/deep_gemm.py` → `import deep_gemm`（独立顶层包，由 sgl-kernel wheel 附带安装）
- `python/sglang/srt/layers/quantization/fp8_kernel.py` → `sgl_per_token_group_quant_fp8, fp8_blockwise_scaled_mm, ...`
- `python/sglang/srt/layers/quantization/gguf.py` → `ggml_dequantize, ggml_mul_mat_a8, ggml_moe_a8_vec`
- `python/sglang/srt/layers/quantization/awq.py` / `gptq.py` / `int8_kernel.py` → 同理

**Sampler / Speculative / Grammar**：

- `python/sglang/srt/layers/sampler.py` → `top_k_renorm_prob, top_p_renorm_prob`
- `python/sglang/srt/speculative/eagle_utils.py` → `build_tree_kernel_efficient, verify_tree_greedy, tree_speculative_sampling_target_only`
- `python/sglang/srt/constrained/xgrammar_backend.py` → `apply_token_bitmask_inplace_cuda`

### 2.2 直接 `torch.ops.sgl_kernel.*`（热路径 / `torch.compile` 首选）

当 op 会被 `torch.compile` 编进图，或者调用者想省一层 Python 函数开销，就直接走 dispatcher 版本：

```347:349:python/sglang/srt/layers/layernorm.py
                torch.ops.sgl_kernel.fused_add_rmsnorm_cpu(
                    ...
                )
```

这种写法对 Dynamo 是"原生理解"的——schema 已注册，alias/mutation 信息齐全。在 `flashattention_backend.py`、attention inner loop、`breakable_cuda_graph` 等热路径里经常见。

另外，CPU 实现（`fused_add_rmsnorm_cpu` / `gemma_fused_add_rmsnorm_cpu` 等）在门面层没有专门的 Python 包装，只能通过 `torch.ops.sgl_kernel.*_cpu` 调到——因此一些"按 device 分支"的代码会混写两种路径。

### 2.3 `MultiPlatformOp` 多后端派发

SGLang 在 `python/sglang/srt/layers/utils.py` 提供 `MultiPlatformOp`，封装"同一语义的 op、不同硬件不同实现"。`layernorm.py` 开头那一大段平台检测：

```41:80:python/sglang/srt/layers/layernorm.py
_is_cuda = is_cuda()
_is_flashinfer_available = is_flashinfer_available()
_is_hip = is_hip()
_is_npu = is_npu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()
_is_xpu = is_xpu()

if _is_cuda or _is_xpu:
    ...
    from sgl_kernel import (fused_add_rmsnorm, gemma_fused_add_rmsnorm, gemma_rmsnorm, rmsnorm)

if _use_aiter:
    from aiter import layernorm2d_fwd as layer_norm
    from aiter import rmsnorm2d_fwd as rms_norm
    from aiter import rmsnorm2d_fwd_with_add as fused_add_rms_norm
elif _is_hip:
    try:
        from vllm._custom_ops import fused_add_rms_norm, rms_norm
```

等价的调用列表：

| 硬件/路径 | 来自哪里 |
| --- | --- |
| CUDA（默认） | `sgl_kernel` |
| HIP + aiter | `aiter` |
| HIP（无 aiter） | `vllm._custom_ops` |
| CPU AMX | `torch.ops.sgl_kernel.rmsnorm_cpu` |
| XPU | `sgl_kernel`（有 XPU impl 的那部分）或其它 backend |
| Ascend NPU | 自带 `hardware_backend/npu/` |

**`sgl-kernel` 的角色**：CUDA/CPU 主后端；其它硬件后端（aiter/vllm/ascend/musa）作为并列备选。

### 2.4 `sgl_kernel.testing` — UT 辅助

单测不仅用算子，也用 `sgl_kernel.testing` 里的：

- FP8 参考量化 / 反量化
- 按容差做误差断言
- 伪造 MoE / attention 张量

这是 `sgl-kernel` 单向服务于上层测试的一块小 API。

### 2.5 特殊产物：`deep_gemm` / `triton_kernels`

```521:567:sgl-kernel/CMakeLists.txt
# Create a separate library for DeepGEMM's Python API.
Python_add_library(deep_gemm_cpp MODULE USE_SABI ${SKBUILD_SABI_VERSION} WITH_SOABI ${DEEPGEMM_SOURCES})
target_link_libraries(deep_gemm_cpp PRIVATE ${TORCH_LIBRARIES} c10 cuda nvrtc mscclpp_static)
...
install(TARGETS deep_gemm_cpp LIBRARY DESTINATION deep_gemm)

install(
    DIRECTORY ${repo-deepgemm_SOURCE_DIR}/deep_gemm/
    DESTINATION deep_gemm
)
...

install(DIRECTORY "${repo-triton_SOURCE_DIR}/python/triton_kernels/triton_kernels/"
        DESTINATION "triton_kernels"
        ...)
```

- **`deep_gemm`**：CMake 把 DeepGEMM 的 Python 源码整体 install 到顶层 `deep_gemm/` 包；`deep_gemm_cpp.so` 链 nvrtc，运行时 JIT。上层用 `import deep_gemm`，不走 `sgl_kernel.*` 命名空间。调用者：`python/sglang/srt/layers/moe/moe_runner/deep_gemm.py` 等。
- **`triton_kernels`**：把 `triton` 仓库里的 `triton_kernels/` Python 模块整体装到顶层，同样独立。

这就是为什么在 README 里看到 `sgl-kernel` wheel 会"附带安装多个顶层包"——它们不是 `sgl_kernel` 的子模块，但由 `sgl-kernel` wheel 统一分发。

---

## 3. 一个"双门面"的例子：`rmsnorm`

为了把前面的机制落到具体场景，对比 `rmsnorm` 的两种调用方式。

### 上层调用

```219:219:python/sglang/srt/layers/layernorm.py
        out = rmsnorm(x, self.weight.data, self.variance_epsilon)
```

### 门面层（有 FlashInfer 就用 FlashInfer）

```107:122:sgl-kernel/python/sgl_kernel/elementwise.py
    if (
        _has_flashinfer
        and input.dtype in _FLASHINFER_NORM_SUPPORTED_DTYPES
        and not torch.compiler.is_dynamo_compiling()
    ):
        return _flashinfer_norm.rmsnorm(input, weight, eps, out, enable_pdl)
    else:
        return _rmsnorm_internal(input, weight, eps, out, enable_pdl)
```

```16:28:sgl-kernel/python/sgl_kernel/elementwise.py
def _rmsnorm_internal(...):
    if out is None:
        out = torch.empty_like(input)
    if enable_pdl is None:
        enable_pdl = is_arch_support_pdl()
    torch.ops.sgl_kernel.rmsnorm.default(out, input, weight, eps, enable_pdl)
    return out
```

### 注册层

```64:65:sgl-kernel/csrc/common_extension.cc
  m.def("rmsnorm(Tensor! output, Tensor input, Tensor weight, float eps, bool enable_pdl) -> ()");
  m.impl("rmsnorm", torch::kCUDA, &rmsnorm);
```

### kernel 实现（`include/sgl_kernel_ops.h` 里声明 → `csrc/elementwise/...` / FlashInfer 头文件里实现）

调用 `flashinfer::norm::RMSNorm<c_type>` 模板 → 在 `at::cuda::getCurrentCUDAStream()` 上发射。

### 顺序图

```
sglang/srt/layers/layernorm.py::RMSNorm.forward
    └── rmsnorm(x, w, eps)                                 [Python 门面]
          ├─ 条件 A：走 FlashInfer  →  _flashinfer_norm.rmsnorm(...)
          └─ 条件 B：走 sgl-kernel  →  torch.ops.sgl_kernel.rmsnorm.default(out, x, w, eps, enable_pdl)
                                            │
                                            ▼  PyTorch dispatcher
                                      m.impl("rmsnorm", torch::kCUDA, &rmsnorm)
                                            │
                                            ▼  C++ rmsnorm(torch::Tensor, ...)
                                      flashinfer::norm::RMSNorm<c_type>(...)   on at::cuda::getCurrentCUDAStream()
```

两条路径都合法且正确；调用方代码**一行都不变**。这就是 `sgl-kernel` 对上层代码的价值：**把"选 kernel"的逻辑（dtype/硬件/FlashInfer 可用性/PDL/编译模式）都藏在门面里，让 `sglang.srt.layers` 保持简洁**。

---

## 4. 与其它 kernel 库的分工

| kernel 库 | 定位 | 和 sgl-kernel 的关系 |
| --- | --- | --- |
| **FlashInfer** | JIT 版 attention / norm / sampling | sgl-kernel 的 `.cu` 里大量 `#include <flashinfer/...>` 调其 device-side template；同时 Python 门面在"FlashInfer 装了 + dtype 支持 + 非编译模式"时优先用 FlashInfer Python 接口 |
| **FA3 (`sgl-attn` fork)** | Hopper 高性能 attention | 完全被 sgl-kernel 编进自己的 `flash_ops.so` |
| **FlashMLA** | MLA 解码专用 | 同上，独立 `.so` |
| **CUTLASS** | 模板库 | 构建期 `FetchContent` 拉源码，不产生独立 `.so` |
| **mscclpp** | GPU-native 通信 | 编进 `common_ops` + 独立 static lib |
| **DeepGEMM** | NVRTC 驱动的 JIT GEMM | 独立顶层包 `deep_gemm`，不走 `sgl_kernel.*` |
| **triton_kernels** | Triton 官方 kernel 集合 | 独立顶层包 |
| **aiter** (AMD) | ROCm 高性能 kernel | `sgl-kernel` 之外的 HIP 后端选择（`SGLANG_USE_AITER=1`） |
| **vllm._custom_ops** | vLLM 的 kernel | HIP 平台在 aiter 不可用时的 fallback |
| **ascend** (NPU) | 华为 NPU | 完全独立 backend，不走 `sgl_kernel` |
| **jit_kernel** (`python/sglang/jit_kernel/`) | Triton / CuTe DSL 的 JIT kernel | SGLang 主仓的轻量 JIT，和 `sgl-kernel` AOT 互补 |

**简言之**：`sgl-kernel` 在 NVIDIA CUDA 平台是"事实上的主要 kernel 提供者"，在 CPU 平台也是主要提供者；在 ROCm 由 aiter/vllm 补位；在 NPU/MUSA 等硬件由各自后端补位。所有这些后端在上层 `layers/*.py` 通过 `MultiPlatformOp` 或显式 `if is_xxx()` 分支粘合起来。

---

## 5. 小结

- `python/sgl_kernel/` 的子模块是"薄包装"，核心责任是：**默认 out 分配 / dtype 检查 / PDL 自动开关 / workspace 缓存 / FlashInfer 双路径 fallback / Debug 日志**。
- 上层在 `sglang.srt.*` 里用 5 种模式消费：直接 `from sgl_kernel import ...`、`torch.ops.sgl_kernel.*`、`MultiPlatformOp` 分发、`sgl_kernel.testing` 测试辅助、附带包（`deep_gemm` / `triton_kernels`）。
- `sgl-kernel` 在全局 kernel 生态里扮演"CUDA/CPU 主后端"角色，通过统一的 `torch.ops.sgl_kernel.*` 命名空间让上层对硬件差异保持无感。
