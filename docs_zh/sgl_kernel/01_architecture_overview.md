# 01 · 顶层架构与构建体系

本章讲 `sgl-kernel` 的**静态结构**：源码如何组织、如何被编译成多份 `.so`、这些 `.so` 如何被装进最终 wheel。

---

## 1. 目录与分层

```
sgl-kernel/
├── CMakeLists.txt          # 顶层构建脚本（约 570 行，驱动所有产物）
├── cmake/
│   ├── flashmla.cmake      # FlashMLA 子构建
│   └── utils.cmake         # clear_cuda_arches 等辅助
├── include/                # 对外头文件
│   ├── sgl_kernel_ops.h        # 所有 GPU kernel C++ 声明（~845 行）
│   ├── sgl_flash_kernel_ops.h  # FA3 接口声明
│   ├── sgl_kernel_torch_shim.h # PyTorch 绑定的类型适配（详见 02 章）
│   ├── scalar_type.hpp         # ScalarType 包装
│   ├── utils.h                 # CHECK_INPUT / DISPATCH 宏
│   ├── hip/impl/               # HIP 兼容头
│   └── pytorch_extension_utils_rocm.h
├── csrc/                   # 所有 kernel 源码
│   ├── common_extension.cc         # ★ CUDA 主注册入口（178 条 m.def/m.impl）
│   ├── flash_extension.cc          # ★ FA3 注册入口
│   ├── spatial_extension.cc        # ★ Green Context Stream 注册入口
│   ├── flashmla_extension.cc       # ★ FlashMLA 注册入口
│   ├── common_extension_rocm.cc    # ROCm 变体（84 条）
│   ├── common_extension_musa.cc    # 摩尔线程 MUSA 变体
│   ├── cpu/torch_extension_cpu.cpp # ★ CPU 注册入口（107 条）
│   ├── allreduce/                  # 自定义 all-reduce + mscclpp
│   ├── attention/                  # cutlass_mla_decode / merge_state / vertical_slash_index
│   ├── elementwise/                # rmsnorm / silu_and_mul / rope / concat_mla / fast_topk / copy
│   ├── gemm/                       # fp8 / int8 / awq / gptq / qserve / dsv3 各种 GEMM
│   ├── moe/                        # moe_align / fused_gate / topk_softmax / blockwise_moe
│   ├── quantization/gguf/          # GGUF 反量化
│   ├── speculative/                # eagle_utils / ngram / tree_sampling / packbit
│   ├── mamba/                      # causal_conv1d
│   ├── kvcacheio/                  # KV 传输
│   ├── memory/                     # weak_ref_tensor
│   ├── spatial/                    # greenctx_stream
│   ├── grammar/                    # apply_token_bitmask
│   ├── expert_specialization/      # es_fp8_blockwise / es_sm100_mxfp8
│   └── cutlass_extensions/         # 自研 cutlass 模板（epilogue / collective）
├── python/sgl_kernel/      # Python 发行包（薄包装）
│   ├── __init__.py         # 加载 .so + 重导出
│   ├── load_utils.py       # 按 SM 选 sm90 / sm100
│   ├── debug_utils.py      # API 日志包装
│   ├── version.py / utils.py / test_utils.py
│   ├── elementwise.py gemm.py moe.py attention.py
│   ├── flash_attn.py flash_mla.py cutlass_moe.py
│   ├── sampling.py allreduce.py speculative.py
│   ├── top_k.py mamba.py memory.py kvcacheio.py
│   ├── expert_specialization.py spatial.py scalar_type.py
│   ├── sparse_flash_attn.py grammar.py load_utils.py
│   ├── quantization/       # GGUF 包装
│   └── testing/            # 测试辅助工具
├── tests/                  # pytest 测试
├── benchmark/              # triton.testing.do_bench(_cudagraph)
├── pyproject.toml          # scikit-build-core 主配置
├── pyproject_rocm.toml     # ROCm 变体
├── pyproject_musa.toml     # MUSA 变体
├── pyproject_cpu.toml      # CPU 变体
├── setup_musa.py setup_rocm.py
├── Makefile                # make build / make format
└── Dockerfile              # 构建环境
```

从源码到 Python 可用的对象，链路是：

```
.cu / .cpp 源码
      │  （csrc/<domain>/*.cu）
      ▼
C++ 函数声明（include/sgl_kernel_ops.h）
      │
      ▼  TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) { m.def; m.impl(torch::kCUDA, ...); }
      │  （csrc/*_extension.cc）
      ▼
编译为 .so（common_ops / flash_ops / spatial_ops / flashmla / deep_gemm_cpp）
      │
      ▼  importlib.util.spec_from_file_location(...).exec_module(...)
      │  （python/sgl_kernel/load_utils.py，按 SM 架构选 sm90/sm100 版本）
      ▼
torch.ops.sgl_kernel.<op>.default(...)   ←  torch.compile 友好
      │
      ▼  python/sgl_kernel/<domain>.py 薄包装：默认 out、dtype 分支、FlashInfer fallback
      ▼
sgl_kernel.<func>                         ←  给人调用的高层 API
      │
      ▼  from sgl_kernel import rmsnorm, fused_add_rmsnorm, ...
      ▼
sglang.srt.layers.* / models.* / speculative.* / constrained.*
```

---

## 2. 构建后端：scikit-build-core + CMake

`sgl-kernel/pyproject.toml`：

```1:9:sgl-kernel/pyproject.toml
[build-system]
requires = [
  "scikit-build-core>=0.10",
  "torch>=2.8.0",
  "wheel",
]
build-backend = "scikit_build_core.build"
```

```36:43:sgl-kernel/pyproject.toml
[tool.scikit-build]
cmake.build-type = "Release"
minimum-version = "build-system.requires"

wheel.py-api = "cp310"
wheel.license-files = []
wheel.packages = ["python/sgl_kernel"]
```

要点：

- `build-backend = "scikit_build_core.build"`：`pip install sglang-kernel` 或 `make build` 最终都会触发 `CMakeLists.txt`。
- `wheel.py-api = "cp310"`：所有 `.so` 走 CPython **ABI3**，Python 3.10/3.11/3.12/3.13 共用一个 wheel。这也是 `CMakeLists.txt` 里 `Python_add_library(... MODULE USE_SABI ${SKBUILD_SABI_VERSION} WITH_SOABI ...)` 的来源。
- `wheel.packages = ["python/sgl_kernel"]`：把纯 Python 子包装进 wheel。

---

## 3. 顶层 CMakeLists.txt 的关键决策

### 3.1 找 PyTorch

```42:43:sgl-kernel/CMakeLists.txt
find_package(Torch REQUIRED)
clear_cuda_arches(CMAKE_FLAG)
```

`find_package(Torch)` 由 PyTorch 安装时提供的 `TorchConfig.cmake` 注入：
- 头文件路径（`torch/all.h`, `torch/library.h`, `c10/...`, `ATen/...`）
- 链接库（`torch`, `torch_cpu`, `torch_cuda`, `c10`, `c10_cuda`）
- 宏（例如 `TORCH_API`、`C10_CUDA_KERNEL_LAUNCH_CHECK`）

这是整个"与 PyTorch 结合"的编译期入口——没有它，`TORCH_LIBRARY_FRAGMENT` 宏和 `at::Tensor` / `at::cuda::getCurrentCUDAStream()` 都用不起来。

### 3.2 `FetchContent` 拉第三方源码（锁 commit）

```47:107:sgl-kernel/CMakeLists.txt
FetchContent_Declare(
    repo-cutlass
    GIT_REPOSITORY https://github.com/NVIDIA/cutlass
    GIT_TAG        57e3cfb47a2d9e0d46eb6335c3dc411498efa198
    GIT_SHALLOW    OFF
)
FetchContent_Populate(repo-cutlass)

# DeepGEMM
FetchContent_Declare(repo-deepgemm ...)
# fmt
FetchContent_Declare(repo-fmt ...)
# Triton kernel
FetchContent_Declare(repo-triton ... GIT_TAG v3.5.1 ...)
# flashinfer
FetchContent_Declare(repo-flashinfer ...)
# flash-attention（用的是 sgl-project/sgl-attn 分支）
FetchContent_Declare(repo-flash-attention ...)
# mscclpp
FetchContent_Declare(repo-mscclpp ...)
```

每个第三方都锁到具体 commit。构建时会把 `${repo-xxx_SOURCE_DIR}/csrc/...` 里的源码直接编译进 `common_ops`，或把头文件加进 `target_include_directories`。也就是说 **FlashInfer / FA3 / CUTLASS 的实现都会被当成 `sgl-kernel` 自己的 translation unit 来编译**，不是运行时去链别人的 .so。

### 3.3 编译参数与 SM 架构

`SGL_KERNEL_CUDA_FLAGS`（节选）：

```133:169:sgl-kernel/CMakeLists.txt
set(SGL_KERNEL_CUDA_FLAGS
    "-DNDEBUG"
    "-DOPERATOR_NAMESPACE=sgl-kernel"
    "-O3"
    "-Xcompiler" "-fPIC"
    "-gencode=arch=compute_90,code=sm_90"
    "-std=c++17"
    "-DFLASHINFER_ENABLE_F16"
    "-DCUTE_USE_PACKED_TUPLE=1"
    "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1"
    "--expt-relaxed-constexpr"
    "--expt-extended-lambda"
    ...
)
```

按检测到的 CUDA 版本增量打开更多架构：

```206:250:sgl-kernel/CMakeLists.txt
if (ENABLE_BELOW_SM90)
    list(APPEND SGL_KERNEL_CUDA_FLAGS
        "-gencode=arch=compute_80,code=sm_80"
        "-gencode=arch=compute_89,code=sm_89"
    )
    ...
endif()

if ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.8" OR SGL_KERNEL_ENABLE_SM100A)
    list(APPEND SGL_KERNEL_CUDA_FLAGS
        "-gencode=arch=compute_100a,code=sm_100a"
        "-gencode=arch=compute_120a,code=sm_120a"
    )
    ...
endif()

if ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.4")
    set(SGL_KERNEL_ENABLE_FA3 ON)
    list(APPEND SGL_KERNEL_CUDA_FLAGS
        "-gencode=arch=compute_90a,code=sm_90a"
    )
endif()
```

一次编译出 **fat binary**，支持 SM80 → SM120 多架构；但是 `common_ops` 会做双份（见 3.4），因此运行时用户还要再选。

### 3.4 一份源码、两次编译：`sm90` vs `sm100+`

```340:365:sgl-kernel/CMakeLists.txt
# =========================== Common SM90 Build ============================= #
Python_add_library(common_ops_sm90_build MODULE USE_SABI ${SKBUILD_SABI_VERSION} WITH_SOABI ${SOURCES})
target_compile_options(common_ops_sm90_build PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${SGL_KERNEL_CUDA_FLAGS} -use_fast_math>
)
set_target_properties(common_ops_sm90_build PROPERTIES
    OUTPUT_NAME "common_ops"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/sm90"
)

# =========================== Common SM100+ Build =========================== #
Python_add_library(common_ops_sm100_build MODULE USE_SABI ${SKBUILD_SABI_VERSION} WITH_SOABI ${SOURCES})
target_compile_options(common_ops_sm100_build PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${SGL_KERNEL_CUDA_FLAGS}>          # 没有 -use_fast_math
)
set_target_properties(common_ops_sm100_build PROPERTIES
    OUTPUT_NAME "common_ops"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/sm100"
)
```

两次编译的差别只有一个：**SM90 打开 `-use_fast_math`，SM100+ 关闭**。

原因：Hopper（H100）上 `-use_fast_math` 带来的吞吐收益明显，精度回退可接受；Blackwell 等新卡精度模型有变化，需要用"精确数学"保证数值稳定。

产物布局：

```
install/
├── sgl_kernel/
│   ├── sm90/common_ops.abi3.so    ← fast_math，装给 H100/Hopper
│   ├── sm100/common_ops.abi3.so   ← 精确数学，装给 SM100/120/…
│   └── <Python 子模块>
```

运行时由 `python/sgl_kernel/load_utils.py::_load_architecture_specific_ops()` 根据 `torch.cuda.get_device_capability()` 选择加载哪一份（见 02 章）。

### 3.5 独立产物

除 `common_ops` 外，顶层 CMake 还产出：

| 产物 | 绑定入口 | 说明 |
| --- | --- | --- |
| `flash_ops.*.so` | `csrc/flash_extension.cc` | FA3；独立编译参数（WGMMA / PDL），`FLASHATTENTION_DISABLE_BACKWARD` 等宏 |
| `spatial_ops.*.so` | `csrc/spatial_extension.cc` | Green Context Stream（SM 分区） |
| `flashmla.*.so` | `csrc/flashmla_extension.cc`（`cmake/flashmla.cmake` 拉 FlashMLA 仓库） | MLA 解码 kernel |
| `deep_gemm_cpp.*.so` | `${repo-deepgemm}/csrc/python_api.cpp` | DeepGEMM 的 JIT 框架，链 `nvrtc`，运行时 JIT 编译 |
| `deep_gemm/` 目录 | `install(DIRECTORY ${repo-deepgemm}/deep_gemm/ DESTINATION deep_gemm)` | 把 DeepGEMM 的 Python 源码和 include 整体装到顶层 `deep_gemm` 包 |
| `triton_kernels/` 目录 | `install(DIRECTORY ${repo-triton}/python/triton_kernels/triton_kernels/ DESTINATION triton_kernels)` | 装 Triton 仓库里的 `triton_kernels` Python 包 |

注意：**`deep_gemm` 和 `triton_kernels` 不是 `sgl_kernel` 的子包**，它们是 `sgl-kernel` wheel 附带安装的**独立顶层包**，上层用 `import deep_gemm` / `import triton_kernels` 即可，不需要经过 `sgl_kernel` 命名空间。

### 3.6 C++ ABI 对齐

```367:381:sgl-kernel/CMakeLists.txt
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
    OUTPUT_VARIABLE TORCH_CXX11_ABI
    ...
)
if(TORCH_CXX11_ABI STREQUAL "0")
    ... "-D_GLIBCXX_USE_CXX11_ABI=0"
else()
    ... "-D_GLIBCXX_USE_CXX11_ABI=1"
endif()
```

必须和 PyTorch 安装时用的 ABI 对齐；否则一旦 `std::string` / `std::list` 穿过边界就会崩溃。这是 PyTorch 扩展开发常见坑。

---

## 4. 其它平台入口

`sgl-kernel` 面向 CUDA 的主体是上面的 `CMakeLists.txt`；另有：

- `setup_rocm.py` + `pyproject_rocm.toml` → 使用 `csrc/common_extension_rocm.cc`（库里很多 op 直接复用 `csrc/*` 的 HIP 可兼容路径，少数用 `#ifdef USE_ROCM` 分支）。
- `setup_musa.py` + `pyproject_musa.toml` → 使用 `csrc/common_extension_musa.cc`，仅导出摩尔线程可支持的算子子集。
- `pyproject_cpu.toml` + `csrc/cpu/torch_extension_cpu.cpp` → CPU 后端，x86 走 AMX/AVX-512，aarch64 走 `csrc/cpu/aarch64/`。

它们与 CUDA 主库**共用 `TORCH_LIBRARY_FRAGMENT(sgl_kernel, m)` 命名空间**，因此上层 `torch.ops.sgl_kernel.<op>` 在同一进程里即可根据 Tensor 的 `device` 自动 dispatch 到 CPU / ROCm 实现（前提是对应实现已注册 `torch::kCPU` 等 dispatch key）。

---

## 5. README 里给扩展开发者的"6 步流程"

`sgl-kernel/README.md` 给的标准新增 kernel 流程：

1. 在 `csrc/<domain>/` 写 `foo.cu`。
2. 在 `include/sgl_kernel_ops.h` 声明对外函数。
3. 在 `csrc/common_extension.cc` 写 `m.def` + `m.impl`。
4. 在 `CMakeLists.txt` 的 `SOURCES` 列表里加一行（字母序）。
5. 在 `python/sgl_kernel/<domain>.py` 暴露 Python 接口。
6. 在 `tests/` 和 `benchmark/` 加 pytest + benchmark。

04 章会把这套流程走一遍完整示例。
