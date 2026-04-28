# 02 · 与 PyTorch 结合的机制（重点 A）

`sgl-kernel` 自始至终把"PyTorch Custom Op + Dispatcher"当作对外的**唯一接口**。不管底层是自己写的 CUDA kernel、FlashInfer 的模板，还是 FA3 的 WGMMA pipeline，最终都要经过 `torch.ops.sgl_kernel.*` 这条通道。

本章拆解这条通道的 **5 个关键机制**：

1. `TORCH_LIBRARY_FRAGMENT` 把 op 注册进 PyTorch dispatcher
2. `sgl_kernel_torch_shim.h` 解决上游类型和 PyTorch 绑定类型不兼容
3. `REGISTER_EXTENSION` 让 `.so` 具备 Python 模块身份
4. `load_utils.py` 按 SM 架构挑 `.so`
5. 三种等价的 Python 调用路径（含 `torch.compile` 友好性）

---

## 1. `TORCH_LIBRARY_FRAGMENT` — 注册到全局 Dispatcher

### 1.1 为什么用 `_FRAGMENT`

PyTorch 对 custom op 的注册有两个宏：
- `TORCH_LIBRARY(ns, m)`：为命名空间 `ns` **创建** 一个 library，整个进程只能执行一次。
- `TORCH_LIBRARY_FRAGMENT(ns, m)`：向已经存在的 `ns` **追加**注册，可执行多次。

`sgl-kernel` 会同时装出 `common_ops.so`、`flash_ops.so`、`spatial_ops.so`、`flashmla.so`、（ROCm 下还有 `common_extension_rocm.cc` 产生的 `.so`）、（CPU 下还有 `cpu/torch_extension_cpu.cpp`），它们都往 **同一个** `sgl_kernel` 命名空间注册。所以必须用 `FRAGMENT`，否则第二次加载就会因 "duplicate library" 报错。

最终效果：**不管分了多少 `.so`，对外都只看到一个统一的 `torch.ops.sgl_kernel.*` 命名空间**。

### 1.2 注册模板

以 `csrc/common_extension.cc` 开头几行为例：

```15:70:sgl-kernel/csrc/common_extension.cc
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/all.h>
#include <torch/library.h>

#include "sgl_kernel_ops.h"

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  /*
   * From csrc/allreduce
   */
  m.def("get_graph_buffer_ipc_meta", &get_graph_buffer_ipc_meta);
  m.def("register_graph_buffers", &register_graph_buffers);
  m.def("dispose", &dispose);
  m.def("meta_size", &meta_size);
  m.def("register_buffer", &register_buffer);

  m.def(
      "init_custom_ar(int[] ipc_tensors, Tensor rank_data, "
      "int rank, bool full_nvlink) -> int");
  m.impl("init_custom_ar", torch::kCUDA, &init_custom_ar);

  m.def(
      "all_reduce(int fa, Tensor inp, Tensor! out, int reg_buffer, "
      "int reg_buffer_sz_bytes) -> ()");
  m.impl("all_reduce", torch::kCUDA, &all_reduce);
  ...
```

两种写法并存：

- **无 schema 的函数导出**：`m.def("meta_size", &meta_size);`
  - 简单函数对象直接挂到 library，`torch.ops.sgl_kernel.meta_size()` 可用。
  - 代价：没有 alias/mutation 信息，`torch.compile` 不认，Dynamo 会把它作为 graph break 点或拒绝编译。
- **带 schema 的标准 op**：`m.def("xxx(...) -> ...")` + `m.impl("xxx", dispatch_key, &impl)`
  - Schema 是 PyTorch native functions 的那套语法（见 `pytorch/aten/src/ATen/native/README.md`）：
    - `Tensor!`：该参数会被就地写（mutation）
    - `Tensor?`：可选
    - `int[]`：`List[int]`
    - `SymInt`：符号 int（支持动态形状）
    - `ScalarType`：`torch.dtype`
  - 这样 Dynamo/Inductor 才能理解 op 的副作用与形状，做正确的图变换。
  - **`common_extension.cc` 里 178 条注册大部分是这种带 schema 的形式**，正是为了支持 `torch.compile` 走全图。

### 1.3 Schema 的实战价值：以 `fused_add_rmsnorm` 为例

```67:68:sgl-kernel/csrc/common_extension.cc
  m.def("fused_add_rmsnorm(Tensor! input, Tensor! residual, Tensor weight, float eps, bool enable_pdl) -> ()");
  m.impl("fused_add_rmsnorm", torch::kCUDA, &sgl_fused_add_rmsnorm);
```

- `Tensor! input, Tensor! residual` 告诉编译器：这两个 tensor 会被就地修改。
- `torch.compile` 看到这条 op 时，会在图里**保留 input/residual 的 alias 约束**，不会错把它"折叠掉"，也不会把后续依赖 `input` 的 op 提前。
- 如果写成 `Tensor input, Tensor residual`（不带 `!`），`torch.compile` 会当作纯函数，内部执行会读到 stale 数据。

### 1.4 多种 dispatch key

```396:408:sgl-kernel/csrc/cpu/torch_extension_cpu.cpp
TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  ...
  m.impl("silu_and_mul_cpu", torch::kCPU, &silu_and_mul_cpu);
  m.impl("gelu_tanh_and_mul_cpu", torch::kCPU, &gelu_tanh_and_mul_cpu);
  m.impl("gelu_and_mul_cpu", torch::kCPU, &gelu_and_mul_cpu);
  ...
  m.impl("rmsnorm_cpu", torch::kCPU, &rmsnorm_cpu);
```

同一个 library 可以为同名 op 注册多个 dispatch key：`torch::kCUDA`、`torch::kCPU`、`torch::kMeta`（形状推导）等。PyTorch dispatcher 会在运行时根据输入 tensor 的 device 自动选择。**上层代码只写一次 `torch.ops.sgl_kernel.foo(...)`，不需要关心当前是 CPU 还是 CUDA。**

---

## 2. `sgl_kernel_torch_shim.h` — 类型适配

### 2.1 问题背景

PyTorch 的 `torch::Library::impl` 对函数签名类型有严格限制：

| 上游代码用的类型 | PyTorch 绑定要求 | 直接绑会怎样 |
| --- | --- | --- |
| `int` | `int64_t` | 编译失败 |
| `float` | `double` | 编译失败 |
| `c10::optional<T>&` | `const c10::optional<T>&` | 可编译但会隐式 `const` 掉 |
| `c10::optional<const at::Tensor>&` | `const c10::optional<at::Tensor>&` | 编译失败 |

FA3 的 `mha_fwd` 来自 Dao-AILab 的 flash-attention 仓库，使用上游自然类型；如果每同步一次上游就去改一遍函数签名，维护不了。

### 2.2 解决方案：编译期类型映射 + Lambda 包装

```44:122:sgl-kernel/include/sgl_kernel_torch_shim.h
template <typename T>
struct pytorch_library_compatible_type {
  using type = T;
  static T convert_from_type(T arg) { return arg; }
};

template <typename T>
using pytorch_library_compatible_type_t = typename pytorch_library_compatible_type<T>::type;

// Map `c10::optional<T> &` -> `const c10::optional<T>&`
template <typename T>
struct pytorch_library_compatible_type<c10::optional<T>&> {
  using type = const c10::optional<T>&;
  ...
};

// Map `c10::optional<T>` -> `c10::optional<pytorch_library_compatible_type_t<T>>`
template <typename T>
struct pytorch_library_compatible_type<c10::optional<T>> { ... };

// Map `c10::optional<const at::Tensor>&` -> `const c10::optional<at::Tensor>&`
template <>
struct pytorch_library_compatible_type<c10::optional<const at::Tensor>&> { ... };

// Map `int` -> `int64_t`
template <>
struct pytorch_library_compatible_type<int> {
  using type = int64_t;
  static int convert_from_type(int64_t arg) {
    TORCH_CHECK(arg <= std::numeric_limits<int>::max(), "int64_t value is too large to be converted to int");
    TORCH_CHECK(arg >= std::numeric_limits<int>::min(), "int64_t value is too small to be converted to int");
    return arg;
  }
};

// Map `float` -> `double`
template <>
struct pytorch_library_compatible_type<float> { ... };

//  Shim Utils
template <typename Ret, typename... Args>
auto make_pytorch_shim(Ret (*fun)(Args... args)) {
  return [fun](pytorch_library_compatible_type_t<Args>... args) {
    return fun(convert_from_pytorch_compatible_type<Args>(args)...);
  };
}
```

工作原理：

- `pytorch_library_compatible_type<T>::type` 给出一个"合法的 binding 类型"。
- `make_pytorch_shim(&f)` 在编译期生成一个 lambda：其参数是 binding 合法类型，内部调用时再转回原 `f` 的类型。
- 对用户代码来说，`make_pytorch_shim(&mha_fwd)` **和直接传 `&mha_fwd` 在语义上等价**，只是参数类型换了一层皮。

### 2.3 使用示例

```63:63:sgl-kernel/csrc/flash_extension.cc
  m.impl("fwd", torch::kCUDA, make_pytorch_shim(&mha_fwd));
```

```97:97:sgl-kernel/csrc/flash_extension.cc
  m.impl("get_scheduler_metadata", torch::kCUDA, make_pytorch_shim(&mha_fwd_get_scheduler_metadata));
```

于是 **FA3 源码基本不用改**，同步上游只需 pull；`sgl-kernel` 通过 shim 承接所有类型差异。这也是 `sgl-kernel` 能作为"多上游融合胶水层"的关键招数。

对自己写的 kernel 一般不需要用 shim——只要你的 C++ 函数一开始就写 `int64_t` / `double` / `const c10::optional<at::Tensor>&` 就行。shim 主要服务于"上游代码不能改"的场景。

---

## 3. `REGISTER_EXTENSION` — 让 `.so` 成为合法 Python 模块

```38:42:sgl-kernel/include/sgl_kernel_ops.h
#define REGISTER_EXTENSION(NAME)                                                                      \
  PyMODINIT_FUNC CONCAT(PyInit_, NAME)() {                                                            \
    static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, STRINGIFY(NAME), nullptr, 0, nullptr}; \
    return PyModule_Create(&module);                                                                  \
  }
```

每个 `*_extension.cc` 最后调用一次：

```498:498:sgl-kernel/csrc/common_extension.cc
REGISTER_EXTENSION(common_ops)
```

```100:100:sgl-kernel/csrc/flash_extension.cc
REGISTER_EXTENSION(flash_ops)
```

```29:29:sgl-kernel/csrc/spatial_extension.cc
REGISTER_EXTENSION(spatial_ops)
```

展开后就是 `PyMODINIT_FUNC PyInit_common_ops()`，生成 ABI3 Python 扩展必须的入口符号。

**关键点：这个 Python 模块内部是空的**（`PyModuleDef { ..., nullptr /* methods */ }`）——真正的注册不靠 Python 绑定，靠的是 `.so` 加载时 `TORCH_LIBRARY_FRAGMENT` 触发的全局构造函数。

所以当 Python 执行：

```python
import common_ops   # 或 importlib.util.spec_from_file_location(...).exec_module(...)
```

会发生什么？
1. 动态加载 `common_ops.abi3.so`。
2. Linker 执行 `.init_array`，里面包含 `TORCH_LIBRARY_FRAGMENT` 展开后生成的静态初始化对象。
3. 静态初始化把 `m.def`/`m.impl` 里的 op 写进 PyTorch 全局 dispatcher。
4. `PyInit_common_ops()` 被调用，返回一个空 Python 模块给 `sys.modules["common_ops"]`。
5. 之后 `torch.ops.sgl_kernel.rmsnorm` 等才能找到实现。

换句话说：**`import common_ops` 的副作用就是注册算子**，模块对象本身几乎用不上。

---

## 4. 运行时按 GPU 架构选 `.so`

`python/sgl_kernel/__init__.py` 的开头就是：

```1:11:sgl-kernel/python/sgl_kernel/__init__.py
import torch
from sgl_kernel.debug_utils import maybe_wrap_debug_kernel
from sgl_kernel.load_utils import _load_architecture_specific_ops, _preload_cuda_library

# Initialize the ops library based on current GPU
common_ops = _load_architecture_specific_ops()

# Preload the CUDA library to avoid the issue of libcudart.so.12 not found
if torch.version.cuda is not None:
    _preload_cuda_library()
```

### 4.1 按 SM 挑 .so

```48:101:sgl-kernel/python/sgl_kernel/load_utils.py
def _load_architecture_specific_ops():
    compute_capability = _get_compute_capability()           # 90, 100, 120, ...
    sgl_kernel_dir = Path(__file__).parent

    if compute_capability == 90:
        ops_subdir = "sm90"
        variant_name = "SM90 (Hopper/H100 with fast math optimization)"
    elif compute_capability is not None:
        ops_subdir = "sm100"
        variant_name = f"SM{compute_capability} (precise math for compatibility)"
    else:
        ops_subdir = "sm100"
        variant_name = "CPU/No GPU detected (using precise math)"

    ops_pattern = str(sgl_kernel_dir / ops_subdir / "common_ops.*")
    raw_matching_files = glob.glob(ops_pattern)
    matching_files = _filter_compiled_extensions(raw_matching_files)
    ...
    if matching_files:
        ops_path = Path(matching_files[0])
        spec = importlib.util.spec_from_file_location("common_ops", str(ops_path))
        common_ops = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(common_ops)        # ★ 触发 PyInit + TORCH_LIBRARY_FRAGMENT
        return common_ops
```

对应 01 章里 `CMakeLists.txt` 的"双胞胎产物"：

- H100（SM90）→ 加载 `sgl_kernel/sm90/common_ops.abi3.so`（带 `-use_fast_math`）
- Blackwell/SM100+ → 加载 `sgl_kernel/sm100/common_ops.abi3.so`（精确数学）
- 纯 CPU 或检测不到 GPU → fallback 到 sm100 版本

`_load_architecture_specific_ops` 还有两层 fallback：
1. 优先按 `sm90/sm100` 子目录加载；失败了看 `sgl_kernel/common_ops.*`。
2. 再失败看标准 Python import `import common_ops`。
3. 都失败打印详细诊断（pattern / 文件列表 / CUDA 版本 / 错误链）再抛 `ImportError`。

这种"按架构分 .so + 运行时挑"的组织方式，避免在同一个 `.so` 里对两套不同的 `__device__` 函数实现做 `#if __CUDA_ARCH__ >= ...`，也让 `ptxas` 不必同时处理互斥的宏分支。

### 4.2 预加载 `libcudart`

```216:245:sgl-kernel/python/sgl_kernel/load_utils.py
def _preload_cuda_library():
    cuda_home = Path(_find_cuda_home())
    candidate_dirs = [
        cuda_home / "lib", cuda_home / "lib64",
        Path("/usr/lib/x86_64-linux-gnu"), Path("/usr/lib/aarch64-linux-gnu"),
        Path("/usr/lib64"), Path("/usr/lib"),
    ]
    cuda_major = torch.version.cuda.split(".")[0] if torch.version.cuda else "12"
    lib_versions = list(dict.fromkeys([cuda_major, "13", "12"]))

    for base in candidate_dirs:
        for lib_version in lib_versions:
            candidate = base / f"libcudart.so.{lib_version}"
            if candidate.exists():
                try:
                    ctypes.CDLL(str(candidate.resolve()), mode=ctypes.RTLD_GLOBAL)
                    return
                ...
```

某些系统只装了 CUDA 13 的 `libcudart.so.13`，而 PyTorch wheel 期望 `libcudart.so.12`（反之亦然）；`sgl-kernel` 的 `.so` 里会引用到 CUDA runtime 的符号，如果找不到就 ImportError。这里先 `RTLD_GLOBAL` 把 runtime 挂进来，后续的 `.so` 加载就能解析到符号。

---

## 5. Python 侧的三种等价调用路径

一旦 `.so` 加载完成，Python 端访问同一个算子有三种方式。

### 5.1 `torch.ops.sgl_kernel.<op>.default(...)` — Dynamo 首选

```40:42:sgl-kernel/python/sgl_kernel/elementwise.py
    torch.ops.sgl_kernel.fused_add_rmsnorm.default(
        input, residual, weight, eps, enable_pdl
    )
```

这是直达 PyTorch dispatcher 的路径。`torch.compile` 能识别，Inductor 也把它当作一个 opaque op 编进图里。所有 schema 级别的 alias / mutation 信息都会被正确看到。

### 5.2 `from sgl_kernel import <func>` — 人友好门面

```62:67:python/sglang/srt/layers/layernorm.py
    from sgl_kernel import (
        fused_add_rmsnorm,
        gemma_fused_add_rmsnorm,
        gemma_rmsnorm,
        rmsnorm,
    )
```

调用时看起来就是普通 Python 函数，带默认值、dtype/shape 检查、必要时自动分配 out buffer、FlashInfer/自研双路径 fallback（详见 03 章）。

### 5.3 `common_ops.<op>(...)` — 极少用

直接用 `_load_architecture_specific_ops()` 返回的模块对象。因为模块里**也**挂了同名函数（PyTorch 注册时会绑一份 C++ 函数指针），但通常没必要走这条路，因为失去了 Python 门面的便利性。

---

## 6. 与 `at::cuda::getCurrentCUDAStream()` 的协同

看一个具体 kernel 的实现末端：

```41:54:sgl-kernel/csrc/elementwise/fused_add_rms_norm_kernel.cu
  cudaStream_t torch_current_stream = at::cuda::getCurrentCUDAStream();
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), c_type, [&] {
    cudaError_t status = norm::FusedAddRMSNorm(
        static_cast<c_type*>(input.data_ptr()),
        static_cast<c_type*>(residual.data_ptr()),
        static_cast<c_type*>(weight.data_ptr()),
        batch_size,
        hidden_size,
        input.stride(0),
        residual.stride(0),
        eps,
        enable_pdl,
        torch_current_stream);
    ...
```

两个与 PyTorch 结合的关键点：

1. **`at::cuda::getCurrentCUDAStream()`**：直接取 PyTorch 当前线程所在 CUDA stream。**这保证 kernel 跑在调用方（模型 forward）的主 stream 上，避免 stream 错配。** 如果这里自己 `cudaStreamCreate` 一个新 stream，就会失去同步保证、和 `torch.cuda.graph` / CUDA Graph capture 也会不兼容。
2. **`DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16`**：风格上模仿 PyTorch 官方的 `AT_DISPATCH_FLOATING_TYPES`，根据 `input.scalar_type()` 在运行时选择 `c_type = float / half / nv_bfloat16`，内部用模板实例化出多套 CUDA kernel。这一层 dtype 调度是"PyTorch 友好"的标志——不要求上层调用时做任何显式 dtype 分支。

其他 kernel 也大多遵循这两条规则。

---

## 7. 小结：一次 `torch.ops.sgl_kernel.fused_add_rmsnorm(...)` 的完整过程

1. Python 调用进入 PyTorch C++ dispatcher（`ATen/core/dispatch/Dispatcher`）。
2. dispatcher 根据 op 名 `sgl_kernel::fused_add_rmsnorm` 找到注册表项，根据 `input.device()` 选择 `torch::kCUDA` 分支。
3. 该分支对应 `csrc/common_extension.cc` 里 `m.impl("fused_add_rmsnorm", torch::kCUDA, &sgl_fused_add_rmsnorm)` 绑定的函数指针。
4. 进入 `sgl_fused_add_rmsnorm`（`csrc/elementwise/fused_add_rms_norm_kernel.cu`），做 `CHECK_INPUT` + dtype 分发 + 取 `at::cuda::getCurrentCUDAStream()` + 调 FlashInfer `norm::FusedAddRMSNorm` 模板。
5. CUDA kernel 在模型主 stream 上执行，就地写回 `input` / `residual`。
6. 返回 `void`，Python 侧可以继续链式做事。

如果上层套了 `torch.compile`，整个图会把这个 op 作为不透明节点保留（因为 schema 告诉它有 mutation），Inductor 会谨慎地避免破坏 alias 约束。

下一章讨论 Python 门面层与上层模块如何使用这一切。
