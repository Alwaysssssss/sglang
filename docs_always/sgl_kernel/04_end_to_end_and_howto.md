# 04 · 端到端调用链 + 新增 Kernel 指南

本章把 02 / 03 章的机制拼到一次真实调用里，然后给出"新增一个 AOT kernel"的完整流程，方便开发者照抄。

---

## 1. 端到端调用链：`fused_add_rmsnorm` in LLaMA DecoderLayer

以 LLaMA 的 "post-attention RMSNorm + residual add" 为例，走一条完整链。

### 1.1 模型层 —— 写法

上层模型调用长这样（语义等价简写）：

```python
# python/sglang/srt/layers/layernorm.py::RMSNorm.forward
fused_add_rmsnorm(x, residual, self.weight.data, self.variance_epsilon)
```

### 1.2 入口在哪里

```62:67:python/sglang/srt/layers/layernorm.py
    from sgl_kernel import (
        fused_add_rmsnorm,
        gemma_fused_add_rmsnorm,
        gemma_rmsnorm,
        rmsnorm,
    )
```

这一步触发 `sgl_kernel/__init__.py`：
1. 调 `_load_architecture_specific_ops()`，按当前 GPU 从 `sgl_kernel/sm90/` 或 `sgl_kernel/sm100/` 加载 `common_ops.abi3.so`。
2. 该 `.so` 加载副作用是执行 `TORCH_LIBRARY_FRAGMENT(sgl_kernel, m)` 中所有 `m.def` / `m.impl`，把 op 注册到 PyTorch dispatcher。
3. 可选：`_preload_cuda_library()` 兜底 `libcudart` 加载。
4. 各子模块 `elementwise/gemm/moe/...` 被导入，`fused_add_rmsnorm` 引用的是 `sgl_kernel/elementwise.py` 里的 Python 函数。

### 1.3 Python 门面层做什么

```125:162:sgl-kernel/python/sgl_kernel/elementwise.py
def fused_add_rmsnorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
) -> None:
    ...
    if (
        _has_flashinfer
        and input.dtype in _FLASHINFER_NORM_SUPPORTED_DTYPES
        and not torch.compiler.is_dynamo_compiling()
    ):
        _flashinfer_norm.fused_add_rmsnorm(input, residual, weight, eps, enable_pdl)
    else:
        _fused_add_rmsnorm_internal(input, residual, weight, eps, enable_pdl)
```

```31:43:sgl-kernel/python/sgl_kernel/elementwise.py
def _fused_add_rmsnorm_internal(...):
    if enable_pdl is None:
        enable_pdl = is_arch_support_pdl()        # Hopper+ 自动开 PDL
    torch.ops.sgl_kernel.fused_add_rmsnorm.default(
        input, residual, weight, eps, enable_pdl
    )
```

假设走自研路径（例如 dtype=fp32，或启用了 `torch.compile(fullgraph=True)`），进入 `torch.ops.sgl_kernel.fused_add_rmsnorm.default`。

### 1.4 PyTorch Dispatcher 层

根据 `input.device().type() == CUDA` 选择 `torch::kCUDA` 实现：

```67:68:sgl-kernel/csrc/common_extension.cc
  m.def("fused_add_rmsnorm(Tensor! input, Tensor! residual, Tensor weight, float eps, bool enable_pdl) -> ()");
  m.impl("fused_add_rmsnorm", torch::kCUDA, &sgl_fused_add_rmsnorm);
```

dispatcher 跳到 C++ 函数 `sgl_fused_add_rmsnorm`。

### 1.5 C++ 实现层

```24:59:sgl-kernel/csrc/elementwise/fused_add_rms_norm_kernel.cu
void sgl_fused_add_rmsnorm(
    torch::Tensor input, torch::Tensor residual, torch::Tensor weight, double eps, bool enable_pdl) {
  CHECK_INPUT(input);
  CHECK_INPUT(residual);
  CHECK_INPUT(weight);
  auto device = input.device();
  CHECK_EQ(residual.device(), device);
  CHECK_EQ(weight.device(), device);
  CHECK_DIM(2, input);
  CHECK_DIM(2, residual);
  CHECK_DIM(1, weight);
  CHECK_EQ(input.size(0), residual.size(0));
  CHECK_EQ(input.size(1), residual.size(1));
  CHECK_EQ(input.size(1), weight.size(0));
  unsigned int batch_size = input.size(0);
  unsigned int hidden_size = input.size(1);

  cudaStream_t torch_current_stream = at::cuda::getCurrentCUDAStream();
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), c_type, [&] {
    cudaError_t status = norm::FusedAddRMSNorm(
        static_cast<c_type*>(input.data_ptr()),
        static_cast<c_type*>(residual.data_ptr()),
        static_cast<c_type*>(weight.data_ptr()),
        batch_size, hidden_size,
        input.stride(0), residual.stride(0),
        eps, enable_pdl,
        torch_current_stream);
    TORCH_CHECK(
        status == cudaSuccess, "FusedAddRMSNorm failed with error code " + std::string(cudaGetErrorString(status)));
    return true;
  });
}
```

关键点回顾：
- `CHECK_INPUT` / `CHECK_DIM` / `CHECK_EQ` 来自 `include/utils.h`，做 contiguous & cuda & shape 检查。
- `at::cuda::getCurrentCUDAStream()` 取模型当前 stream，保证 kernel 跑在主 stream。
- `DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16` 根据 `input.scalar_type()` 实例化 fp32/fp16/bf16 三份 kernel。
- `norm::FusedAddRMSNorm` 是 FlashInfer 提供的 device-side 模板（头文件来自 `FetchContent` 拉的 `flashinfer` 仓库）。

### 1.6 `torch.compile` 下的行为

如果 RMSNorm 的 forward 被 `torch.compile` 包住：

- Dynamo 扫到 `torch.ops.sgl_kernel.fused_add_rmsnorm` —— 这是**注册过 schema 的 custom op**，被当作不透明节点保留。
- Schema `(Tensor! input, Tensor! residual, Tensor weight, float eps, bool enable_pdl) -> ()` 告诉 Inductor：**input 和 residual 会被就地修改**，不要把后续读这两个 tensor 的 op 提前、也不要把它折叠。
- Inductor 输出的 C++ 代码里直接 `c10::Dispatcher::singleton().findSchemaOrThrow("sgl_kernel::fused_add_rmsnorm", "").typed<...>()` 调用，等价于 eager 模式路径。

这就是 02 章里所谓的"schema 是 `torch.compile` 的硬门槛"的具体落地。

### 1.7 Debug 模式（`SGLANG_KERNEL_API_LOGLEVEL=1`）

`__init__.py` 在 import 阶段已经把 `fused_add_rmsnorm` 替换成 `debug_kernel_api(fn, op_name="sgl_kernel.fused_add_rmsnorm")` 包装过的版本：

```115:188:sgl-kernel/python/sgl_kernel/__init__.py
_DEBUG_EXPORT_NAMES = [ ..., "fused_add_rmsnorm", ... ]

for _name in _DEBUG_EXPORT_NAMES:
    if _name in globals():
        globals()[_name] = maybe_wrap_debug_kernel(
            globals()[_name], f"sgl_kernel.{_name}"
        )
```

每次调用都会先把 shape/dtype/device 打到日志，然后再真正执行。这对定位"哪个 kernel 炸了"非常有用，详见 `.claude/skills/debug-cuda-crash/SKILL.md`。

---

## 2. 全链路示意图

```
┌────────────────────────────────────────────────────────────────────┐
│  python/sglang/srt/layers/layernorm.py                             │
│      fused_add_rmsnorm(x, residual, w, eps)                        │
└──────────────────────────┬─────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────────┐
│  python/sgl_kernel/elementwise.py::fused_add_rmsnorm               │
│      ├── 选择路径（FlashInfer Python vs 自研）                      │
│      ├── dtype / PDL / out 自动化                                  │
│      └── torch.ops.sgl_kernel.fused_add_rmsnorm.default(...)       │
└──────────────────────────┬─────────────────────────────────────────┘
                           │  （Dynamo 能识别的 schema'd op）
                           ▼
┌────────────────────────────────────────────────────────────────────┐
│  PyTorch Dispatcher                                                │
│      Library: sgl_kernel,  Op: fused_add_rmsnorm                   │
│      Key: torch::kCUDA  →  &sgl_fused_add_rmsnorm                  │
└──────────────────────────┬─────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────────┐
│  sgl-kernel/csrc/elementwise/fused_add_rms_norm_kernel.cu          │
│      ├── CHECK_INPUT / CHECK_DIM / CHECK_EQ                        │
│      ├── at::cuda::getCurrentCUDAStream()                          │
│      ├── DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16                │
│      └── flashinfer::norm::FusedAddRMSNorm<c_type>(...)            │
└──────────────────────────┬─────────────────────────────────────────┘
                           │
                           ▼
                     CUDA kernel 发射
                     （SGLang 主 stream 上）
```

这条链条同时穿过了：
- Python 门面（默认参数 / PDL / FlashInfer fallback）
- torch.ops 注册（schema 驱动的 Dynamo 行为）
- dispatcher（按 device 分发）
- Torch shim（本例没用到；FA3 用；详见 02 章）
- CUDA stream 协同（`at::cuda::getCurrentCUDAStream()`）
- dtype 调度宏
- 第三方 kernel 复用（FlashInfer 模板）

---

## 3. 新增一个 AOT kernel 的 7 步流程

`sgl-kernel/README.md` 给出了简版 6 步流程，下面按真实常见步骤扩到 7 步。

> 以假设新增一个 `my_elementwise_op(input, out, scale)` 为例。

### Step 1 · 写 kernel 代码

在 `sgl-kernel/csrc/elementwise/my_elementwise.cu`：

```cpp
#include <ATen/cuda/CUDAContext.h>
#include "utils.h"

template <typename scalar_t>
__global__ void my_elementwise_kernel(const scalar_t* __restrict__ x,
                                      scalar_t* __restrict__ y,
                                      float scale,
                                      int64_t n) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  y[idx] = static_cast<scalar_t>(static_cast<float>(x[idx]) * scale);
}

void my_elementwise_op(torch::Tensor input, torch::Tensor output, double scale) {
  CHECK_INPUT(input);
  CHECK_INPUT(output);
  CHECK_EQ(input.numel(), output.numel());

  int64_t n = input.numel();
  constexpr int THREADS = 256;
  int64_t blocks = (n + THREADS - 1) / THREADS;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16,
      input.scalar_type(), "my_elementwise_op", [&] {
    my_elementwise_kernel<scalar_t><<<blocks, THREADS, 0, stream>>>(
        input.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        static_cast<float>(scale),
        n);
  });
}
```

### Step 2 · 在头文件声明

在 `sgl-kernel/include/sgl_kernel_ops.h` 找一个合适分区（例如 `From csrc/elementwise`），加：

```cpp
void my_elementwise_op(torch::Tensor input, torch::Tensor output, double scale);
```

### Step 3 · Torch 绑定

在 `sgl-kernel/csrc/common_extension.cc`，找到 `From csrc/elementwise` 段，加：

```cpp
m.def("my_elementwise_op(Tensor input, Tensor! output, float scale) -> ()");
m.impl("my_elementwise_op", torch::kCUDA, &my_elementwise_op);
```

Schema 要点：
- `Tensor input` → 只读
- `Tensor! output` → 就地写
- `float scale` → Python 里 `float`；C++ 侧会自动映射成 `double`（见 02 章 shim 章节）
- 返回 `()` → void

如果你的函数签名用了 `int` / `float` / `c10::optional<T>&` 等"非 binding 合法"类型，改用：

```cpp
#include "sgl_kernel_torch_shim.h"
m.impl("my_elementwise_op", torch::kCUDA, make_pytorch_shim(&my_elementwise_op));
```

### Step 4 · 加入 CMake SOURCES

在 `sgl-kernel/CMakeLists.txt` 的 `set(SOURCES ...)` 里按字母序插入一行：

```cmake
"csrc/elementwise/my_elementwise.cu"
```

这样两份 `common_ops` 构建（sm90 / sm100+）都会包含它。

### Step 5 · 写 Python 门面

在 `sgl-kernel/python/sgl_kernel/elementwise.py` 加：

```python
def my_elementwise_op(
    input: torch.Tensor,
    scale: float,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(input)
    torch.ops.sgl_kernel.my_elementwise_op.default(input, out, scale)
    return out
```

然后在 `sgl-kernel/python/sgl_kernel/__init__.py`：

```python
from sgl_kernel.elementwise import (
    ...
    my_elementwise_op,
)
```

如果想让 `SGLANG_KERNEL_API_LOGLEVEL=1` 也能打到日志，把名字加进 `_DEBUG_EXPORT_NAMES` 列表。

### Step 6 · 测试 & benchmark

`sgl-kernel/tests/test_my_elementwise.py`：

```python
import pytest
import torch
from sgl_kernel import my_elementwise_op

@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_my_elementwise_op_correctness(dtype):
    x = torch.randn(1024, 768, dtype=dtype, device="cuda")
    scale = 0.125
    y_ref = x * scale
    y = my_elementwise_op(x, scale)
    torch.testing.assert_close(y, y_ref, rtol=1e-2, atol=1e-2)
```

`sgl-kernel/benchmark/bench_my_elementwise.py`：

```python
import torch
import triton
import triton.testing
from sgl_kernel import my_elementwise_op

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"], x_vals=[2**i for i in range(14, 22)],
        line_arg="provider", line_vals=["sgl_kernel", "eager"],
        line_names=["sgl_kernel", "torch eager"],
        ylabel="TFLOPS", args={},
        plot_name="my_elementwise",
    )
)
def benchmark(N, provider):
    x = torch.randn(N, dtype=torch.bfloat16, device="cuda")
    if provider == "sgl_kernel":
        fn = lambda: my_elementwise_op(x, 0.125)
    else:
        fn = lambda: x * 0.125
    ms = triton.testing.do_bench_cudagraph(fn)
    return 2 * N / ms * 1e-9

if __name__ == "__main__":
    benchmark.run(save_path=".", print_data=True)
```

> `README.md` 推荐用 `do_bench_cudagraph` 而不是 `do_bench`：能把 PDL 等 SM90+ 特性的收益算进来，CPU 开销影响也小。

### Step 7 · 构建 + 本地验证

```bash
cd sgl-kernel
make build                       # 默认用所有核心
# 或者受限机器：
# make build MAX_JOBS=4 CMAKE_ARGS="-DSGL_KERNEL_COMPILE_THREADS=1"

pytest -xvs tests/test_my_elementwise.py
python benchmark/bench_my_elementwise.py
```

装载时可以打开日志看加载哪份 `.so`：

```bash
python -c "import logging; logging.basicConfig(level=logging.DEBUG); import sgl_kernel"
```

---

## 4. 扩展到多后端

| 场景 | 要额外做的事 |
| --- | --- |
| 想支持 CPU | 在 `csrc/cpu/` 另写 `my_elementwise_cpu.cpp`，在 `csrc/cpu/torch_extension_cpu.cpp` 里 `m.def(...)` + `m.impl("my_elementwise_op", torch::kCPU, &my_elementwise_cpu)`；同时在 `pyproject_cpu.toml` 对应构建入口包含新文件 |
| 想支持 ROCm | 确保 kernel 可以被 hipify；如果能直接编过就往 `csrc/common_extension_rocm.cc` 里加一份 `m.def/impl`；否则单独写 HIP kernel |
| 想接 `MultiPlatformOp` | 在 `python/sglang/srt/layers/<domain>.py` 里按 `_is_cuda / _is_hip / _is_cpu` 分支 import |
| 想让它能被 `torch.compile` 编图 | schema 一定要写完整，尤其 `Tensor!` / `Tensor?` / `SymInt`；不要用 `m.def("foo", &foo);`（无 schema）写法 |
| 第三方仓库签名有冲突 | 用 `sgl_kernel_torch_shim.h::make_pytorch_shim` 包一层 |

---

## 5. 常见坑

1. **schema 漏写 `!`**：kernel 会 in-place 写但 schema 写成 `Tensor input`，结果 `torch.compile` 把后面的读操作提前执行，跑出来值不对。一定要用 `Tensor!`。
2. **自己 `cudaStreamCreate`**：不要，用 `at::cuda::getCurrentCUDAStream()`。否则 CUDA Graph capture、`torch.cuda.current_stream()` 上下文、NCCL 多流都会错乱。
3. **没加 `CHECK_INPUT`**：用户传 non-contiguous / 非 CUDA tensor 时，`data_ptr()` 会访问到错误内存，晚期报错难查。
4. **多架构未覆盖**：`-gencode=arch=compute_XX` 漏掉目标 GPU，`.so` 加载后调用直接 `cudaErrorNoKernelImageForDevice`。看 `CMakeLists.txt` 中 `SGL_KERNEL_CUDA_FLAGS` 列表。
5. **ABI 不匹配**：Docker 镜像里 PyTorch 用的是 old C++ ABI，你本地编 kernel 用的是 new ABI → `std::string` 崩；受 `CMakeLists.txt` 里 `_GLIBCXX_USE_CXX11_ABI` 自动检测保护，但如果手动写 `setup.py` 要留意。
6. **cubin 过大**：模板实例化爆炸会导致 `.so` 特别大（几百 MB）。用 `analyze_whl_kernel_sizes.py path/to/wheel.whl` 可定位哪个 symbol 最胖（需要装 `cubloaty`）。
7. **两份 `.so` 不一致**：你只改了 sm100 分支没改 sm90（或者反过来），运行时一台机器正常另一台崩。`CMakeLists.txt` 其实是**同一份 SOURCES 编两次**，除非你特地给 sm90/sm100 设了独立宏。
8. **FlashInfer 版本不兼容**：`FetchContent` 锁的是特定 commit；本地如果装了更旧/更新的 `pip install flashinfer`，Python 门面 fallback 分支可能行为不一致。遇到一致性问题可以先 `unset FLASHINFER_*` 或卸载 flashinfer，走 sgl-kernel 自研路径。

---

## 6. 相关 Skill 与扩展阅读

仓库里已沉淀的流程化 Skill：

- `.claude/skills/add-sgl-kernel/SKILL.md` — 重量级 AOT kernel 的完整新增流程（含 CI 注册）
- `.claude/skills/add-jit-kernel/SKILL.md` — 对应 `python/sglang/jit_kernel/` 的轻量 JIT 版本
- `.claude/skills/debug-cuda-crash/SKILL.md` — 利用 `SGLANG_KERNEL_API_LOGLEVEL` 定位炸 kernel
- `.claude/skills/sglang-torch-profiler-analysis/SKILL.md` — 对现有 kernel 做热点 / overlap 分析

同域文档：

- [`docs_zh/path.md`](../path.md) — 仓库 5 层结构总览（`sgl-kernel/` 在第 12 节）
- [`docs_zh/multimodal_gen/`](../multimodal_gen/README.md) — 多模态子系统（使用 `sgl-kernel` 的另一个大客户，`csrc/` 里有它专用的 attention/render kernel 通过 `jit_kernel` 通道走）

---

## 7. TL;DR

- `sgl-kernel` 的全部对外 API 都是 `torch.ops.sgl_kernel.*`，由 `TORCH_LIBRARY_FRAGMENT` 注册，由运行时按 SM 架构挑 `.so` 实现。
- Python 门面层只是**默认参数 + 双路径 fallback + PDL 自动化**的薄包装，真正的热点代码在 CUDA kernel 里。
- 上层模块以 `from sgl_kernel import ...` 或 `torch.ops.sgl_kernel.*` 两种方式消费；多硬件通过 `MultiPlatformOp` / `if is_xxx()` 统一分发。
- 新增 kernel 的 7 步流程：写 `.cu` → 在 `sgl_kernel_ops.h` 声明 → 在 `common_extension.cc` 绑定（写 schema！）→ 在 `CMakeLists.txt` 的 `SOURCES` 加行 → 写 Python 门面 → `__init__.py` 导出 → 测试/bench。
