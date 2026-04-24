# SGLang `sgl-kernel` 深度解析

本文档面向 `sgl-kernel/`，解释它如何把散落在 FlashInfer、CUTLASS、Flash-Attention、DeepGEMM、mscclpp、Triton 等上游仓库的手写 kernel，以及 SGLang 自研的 MoE / MLA / Speculative / 量化算子，统一封装成 `torch.ops.sgl_kernel.*` 可调用的 PyTorch C++ 扩展。

重点围绕两条主线：

1. **`sgl-kernel` 如何与 PyTorch 结合**（Schema 注册、Dispatcher、类型 shim、架构选 .so）。
2. **上层模块（`python/sglang/` 主体）如何与 `sgl-kernel` 结合**（三种调用路径、MultiPlatformOp 分发、与 FlashInfer 的 fallback 关系）。

## 阅读顺序

1. [01_architecture_overview.md](./01_architecture_overview.md) — 顶层目录 / 构建体系 / 多产物切分
2. [02_pytorch_binding.md](./02_pytorch_binding.md) — ★ 与 PyTorch 结合的底层机制
3. [03_upstream_integration.md](./03_upstream_integration.md) — ★ Python 门面层 + 上层 `sglang.srt` 接入方式
4. [04_end_to_end_and_howto.md](./04_end_to_end_and_howto.md) — 端到端调用链实战 + 新增 kernel 的扩展指南

## 一句话理解

`sgl-kernel` 不是"一堆 CUDA 代码的集合"，而是一个**以 PyTorch Custom Op 为共同接口**的 LLM 推理算子仓库：

- **底层**：`csrc/**/*.cu` + 若干 `FetchContent` 拉进来的上游仓库源码。
- **绑定层**：`csrc/*_extension.cc` 用 `TORCH_LIBRARY_FRAGMENT(sgl_kernel, m)` 把所有 kernel 注册进 PyTorch dispatcher，并用 `include/sgl_kernel_torch_shim.h` 解决"上游 C++ 原生类型 vs PyTorch 绑定类型"不匹配。
- **编译产物**：产出 `common_ops.*.so`（按 SM90 / SM100 分两份）、`flash_ops.*.so`、`spatial_ops.*.so`、`flashmla.*.so`、`deep_gemm_cpp.*.so`，装到 `sgl_kernel/sm90/`、`sgl_kernel/sm100/`、`sgl_kernel/` 等不同子目录。
- **Python 门面**：`python/sgl_kernel/__init__.py` 运行时按当前 GPU 架构加载对应 `.so`；子模块（`elementwise.py`, `gemm.py`, `moe.py`, `flash_attn.py` 等）是**薄包装**，做默认 out buffer、dtype 分支、FlashInfer/自研双路径 fallback、PDL 自动开启等事情。
- **对外 API**：`torch.ops.sgl_kernel.<op>.default(...)`（`torch.compile` 友好）与 `from sgl_kernel import <op>`（带默认值的人友好函数）两条等价通路。

## 顶层模块地图

| 路径 | 作用 |
| --- | --- |
| `csrc/` | 所有 CUDA / C++ kernel 源码（按功能域：`attention/`, `gemm/`, `moe/`, `elementwise/`, `allreduce/`, `speculative/`, `quantization/`, `mamba/`, `grammar/`, `kvcacheio/`, `expert_specialization/`, `spatial/`, `cpu/` …） |
| `csrc/common_extension.cc` | CUDA 主 `.so`（`common_ops`）的 `TORCH_LIBRARY_FRAGMENT` 注册入口 |
| `csrc/flash_extension.cc` | FA3 专用 `.so`（`flash_ops`）的注册入口 |
| `csrc/flashmla_extension.cc` | FlashMLA 专用 `.so` 的注册入口 |
| `csrc/spatial_extension.cc` | Green Context Stream `.so`（`spatial_ops`）的注册入口 |
| `csrc/common_extension_rocm.cc` / `common_extension_musa.cc` | ROCm / MUSA 平台变体 |
| `csrc/cpu/torch_extension_cpu.cpp` | CPU 算子注册（AMX / AVX-512 / aarch64） |
| `include/sgl_kernel_ops.h` | 所有 GPU kernel 的 C++ 声明（约 845 行） |
| `include/sgl_flash_kernel_ops.h` | FA3 接口声明 |
| `include/sgl_kernel_torch_shim.h` | `int↔int64_t`、`float↔double`、`optional<T>&` 等类型适配器 |
| `include/scalar_type.hpp`, `utils.h` | 公共工具 / `ScalarType` 包装 |
| `cmake/flashmla.cmake`, `cmake/utils.cmake` | 子构建脚本 |
| `CMakeLists.txt` | 顶层构建（scikit-build-core 驱动） |
| `python/sgl_kernel/` | Python 发行包入口（薄包装 + `.so` 加载） |
| `python/sgl_kernel/__init__.py` | 加载 `.so` + 重新导出所有 op |
| `python/sgl_kernel/load_utils.py` | 按 SM 架构选择 `sm90` / `sm100` 版本 |
| `python/sgl_kernel/debug_utils.py` | API 日志注入（`SGLANG_KERNEL_API_LOGLEVEL`） |
| `python/sgl_kernel/<domain>.py` | 逐模块门面：`elementwise`, `gemm`, `moe`, `attention`, `flash_attn`, `flash_mla`, `sampling`, `allreduce`, `speculative`, `quantization/`, … |
| `tests/` | pytest 单测（按子模块分） |
| `benchmark/` | 基于 `triton.testing.do_bench_cudagraph` 的性能测试 |
| `pyproject.toml` | scikit-build-core 后端 + ABI3 wheel 配置 |

## 建议抓主线的方式

如果只想抓主干，建议按以下顺序读代码：

`python/sgl_kernel/__init__.py` →
`python/sgl_kernel/load_utils.py` →
`csrc/common_extension.cc`（看 `m.def` 和 `m.impl` 的配对）→
挑一个算子（如 `rmsnorm`）沿着 `csrc/elementwise/fused_add_rms_norm_kernel.cu` 跟进 CUDA 实现 →
反向看 `python/sgl_kernel/elementwise.py` 怎么包装 →
最终看 `python/sglang/srt/layers/layernorm.py` 如何消费。

这条路径会同时触达"Python 门面 / torch.ops dispatcher / C++ 绑定 / CUDA kernel / 上层消费"五层。
