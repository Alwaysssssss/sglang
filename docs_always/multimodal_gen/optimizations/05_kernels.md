# 05. 算子 / Kernel 融合

> 源码位置：`python/sglang/multimodal_gen/runtime/layers/layernorm.py`、`runtime/layers/activation.py`、`runtime/layers/elementwise.py`、`runtime/layers/mlp.py`、`runtime/layers/linear.py`、`runtime/layers/custom_op.py`、`runtime/layers/rotary_embedding/*`、`runtime/layers/visual_embedding.py`、`runtime/layers/utils.py`、`csrc/render/*`。

扩散模型的算子分布与 LLM 不同：**时间步嵌入（Timesteps）、AdaLN 类 modulate、QK norm + RoPE、3D patch embed、RMSNorm / SiluAndMul 高频调用**。本篇整理这些算子在 `multimodal_gen` 中的实现分支（CUDA / ROCm / NPU / MUSA / XPU / MPS / 通用 native），以及自带的 C++/CUDA 扩展。

## 1. 分发抽象：`CustomOp`

文件 `runtime/layers/custom_op.py`。

- 所有融合 / 可替换算子都继承 `CustomOp`，子类实现 `forward_native`、`forward_cuda`（及 `forward_npu`、`forward_hip`、`forward_musa`、`forward_xpu`）；
- `dispatch_forward()`（约 72–84 行）按 `current_platform` 选 cuda / hip / npu / xpu / musa / native；
- `forward` 经 `@debug_kernel_api` 包装（约 28–31 行）；
- `@CustomOp.register(name)` 把算子登记到 `op_registry`（约 99–116 行）；
- 注释明确 `forward_native` 用于 `torch.compile` / XLA 测试（约 33–37 行）——**这点很关键：所有 `forward_native` 都应保持可 trace**。

辅助：`runtime/layers/utils.py::direct_register_custom_op` 向 `torch.library` 注册算子；NPU 分支使用 `PrivateUse1`（约 107–111 行），便于 `torch.compile` 的 fake/meta 接口（约 193–267 行）。`CustomOpWrapper` 支持延迟注册；注释说明**lazy registration 与 torch.compile 不兼容时需 `eager=True`**（约 151–158 行）。

## 2. LayerNorm / RMSNorm

文件 `runtime/layers/layernorm.py`。

### 2.1 RMSNorm 分支矩阵

| 平台 | 实现 |
|------|------|
| CUDA | `forward_cuda`：`float32` 且无 residual 时走 `forward_triton`（`rms_norm_fn`）；否则 `fused_add_rmsnorm`（`sgl_kernel`）或 `triton_one_pass_rms_norm`（hidden≤128）或 `rmsnorm`（`sgl_kernel`）（约 79–112 行）|
| ROCm | **`forward_hip` 显式退回 `forward_native`**：注释说明 ROCm 的 sgl-kernel 未暴露 rmsnorm custom ops（约 172–178 行）|
| NPU | `torch_npu.npu_rms_norm` / `npu_add_rms_norm`（约 160–170 行）|
| MUSA | `fused_add_rmsnorm` 或 `F.rms_norm`（约 191–217 行）|
| XPU | 与 CUDA 类似，用 `rmsnorm`/`fused_add_rmsnorm`（约 219–238 行）|
| CPU / 通用 | `forward_native` |

**TP 分片 RMSNorm**：`tensor_parallel_rms_norm`（约 849–860 行）用 `all_reduce` 聚合方差，与单卡 RMSNorm 不同。

### 2.2 LayerNorm

| 平台 | 实现 |
|------|------|
| CUDA | `forward_triton` → `norm_infer`（Triton）（约 285–301 行）|
| NPU | `forward_native` 带 `@torch.compile(..., disable=current_platform.is_npu())`（约 303–318 行）——NPU 禁用该子图编译 |
| MUSA | `forward_musa` → `F.layer_norm`（约 327–328 行）|
| 其它 | 继承 `forward_cuda` 或 native |

### 2.3 CuTeDSL 级联融合

文件内还提供了 3 个 CuTeDSL 实现（CUDA）：

- `_ScaleResidualNormScaleShift`
- `_NormScaleShift`
- `_NormTanhMulAdd`

位于约 359–640 行。这些融合算子在维度不满足时会在 CUDA 上 warn 并回落 `forward_native`；在 `forward_hip` / `forward_musa` / `forward_xpu` 上**永远 native**（约 428–441、537–550 行）。

### 2.4 QK Norm + RoPE 融合

`apply_qk_norm` / `apply_qk_norm_rope`（约 642–818 行）：

- CUDA 可走 `fused_inplace_qknorm`、`fused_inplace_qknorm_rope`（JIT）；
- 否则退回 RMSNorm + `apply_flashinfer_rope_qk_inplace`（见 §4）。

### 2.5 MLX 融合（MPS）

`README.md` 提到 `SGLANG_USE_MLX=1` 可启用 MLX 融合 Metal norm；具体挂接在 `runtime/platforms/mps.py` 里，非本文件路径（`layernorm.py` 内**没有** MLX 代码）。

## 3. 激活函数

文件 `runtime/layers/activation.py`。

- **`SiluAndMul`**（`@CustomOp.register("silu_and_mul")`）：
  - CUDA / HIP：`sgl_kernel.silu_and_mul`（约 19–20、42–47 行）；
  - `forward_native`：`F.silu(...) * ...`（约 49–52 行）；
  - NPU：`torch_npu.npu_swiglu`（约 54–56 行）；
  - MUSA：`nn.SwishGLU`（约 58–59 行）。
- **`GeluAndMul`** / **`NewGELU`** / **`QuickGELU`**：`forward_cuda` 直接调 `forward_native`（约 79–80、97–98、115–116 行）——**没有单独的 sgl-kernel GeGLU 融合**。
- **`get_act_fn`**（约 126–142 行）：映射到 `nn.GELU` / `NewGELU` / `nn.SiLU`；
- **`get_act_and_mul_fn`**（约 145–157 行）：返回 `GeluAndMul` / `SiluAndMul`。

### 3.1 MLP 层次

`runtime/layers/mlp.py`：

- `MLP`（约 26–68 行）：`ColumnParallelLinear` → `get_act_fn(act_type)` → `RowParallelLinear`。**分离的 Linear + 激活**，不是单 kernel 融合 MLP。
- `FeedForward`（约 97–116 行）：使用 diffusers 的 `GEGLU` / `SwiGLU` / `GELU` 等，同样不是 `sgl_kernel` 的 fused MLP。

### 3.2 真正的「融合 GELU MLP」

在 `runtime/models/dits/flux.py::_fused_gelu_mlp`（约 85–165 行）：配合 **Nunchaku / `_svdq_gemm_w4a4` 等 W4A4/NVFP4 量化路径**，与通用 MLP 不同；仅 Flux 系列走这条路径。

## 4. Rotary Embedding

目录 `runtime/layers/rotary_embedding/`。

| 文件 | 主要类 / 函数 | 作用 |
|------|---------------|------|
| `base.py` | `RotaryEmbedding`（`@CustomOp.register("rotary_embedding")`）、`LinearScalingRotaryEmbedding` | 纯 PyTorch 查表 + `_apply_rotary_emb`；**CUDA 路径等于 native**（约 61–62 行）|
| `mrope.py` | `NDRotaryEmbedding`、`OneDRotaryEmbedding`、`get_1d_rotary_pos_embed` | 多轴（t/h/w）位置；`rope_dim_list` 分轴拼 cos/sin；`forward_from_grid` 与 SP 协同做 shard（约 296–392 行）|
| `factory.py` | `get_rope`、`get_rotary_pos_embed` | 1D RoPE 缓存；构造 `NDRotaryEmbedding` 并在 grid 上生成 freqs（约 85–171 行）|
| `utils.py` | `_apply_rotary_emb`、`apply_flashinfer_rope_qk_inplace` | Neox 用 chunk；非 Neox 用 `sglang.jit_kernel.apply_rotary_embedding`（Triton）；CUDA 上 FlashInfer 就地 RoPE（约 33–62、94–116 行）|

**注意**：上述工厂路径**未集成 Liger**。

### 4.1 DiT 模型内的 AITer RoPE（不在 `rotary_embedding/` 包内）

`runtime/models/dits/wanvideo.py` 若 `_use_aiter`（`SGLANG_USE_AITER` + HIP），则使用 `aiter.ops.rope.rope_cached_2c_fwd_inplace`（例如约 556–574、802–820 行）；否则用 `_apply_rotary_emb`（Triton/PyTorch）。这属于模型内按需 override，与 `rotary_embedding/` 工厂并行。

## 5. Linear

文件 `runtime/layers/linear.py`。

- 核心类：`ColumnParallelLinear` / `RowParallelLinear` / `QKVParallelLinear` / `MergedColumnParallelLinear`；
- 量化接入：`LinearBase.__init__` 在 `quant_config` 非空时 `quant_config.get_quant_method(self, prefix)`（约 196–199 行）；
- `MergedColumnParallelLinear.weight_loader`（约 444–655 行）：把磁盘上已合并的 fused 权重（例如 gate+up）正确地切到对应 TP shard；属于**加载层面**的融合，非单 kernel。

## 6. Elementwise：`MulAdd`

文件 `runtime/layers/elementwise.py`。

- `MulAdd`（`@CustomOp.register("mul_add")` 等效）：`c + a * (k + b)`，支持 4D `gate`（多帧 modulate）与 3D broadcast（`forward_native` 约 17–30 行）；
- CUDA 路径调用 **`fuse_scale_shift_kernel`（Triton）**（约 32–35 行）；
- XPU 退回 native（约 37–40 行）。

用于 AdaLN / 门控形式的 scale-shift 融合，**不是通用 unary/binary 库**。

## 7. Visual Embedding

文件 `runtime/layers/visual_embedding.py`。

- `PatchEmbed`（约 83–111 行）：5D 输入走 **reshape + `F.linear`** 快路径，否则 `nn.Conv3d`；
- `Timesteps`（约 114–131 行）：CUDA 用 `sglang.jit_kernel.timestep_embedding.timestep_embedding_cuda`，否则 diffusers；
- `TimestepEmbedder`（约 173–215 行）：内部 `MLP` + `timestep_embedding`；
- `ModulateProjection`（约 248–274 行）：`ColumnParallelLinear` + `get_act_fn`。

## 8. `csrc/render`：渲染与 3D 算子

`multimodal_gen/csrc/render/` 提供 **Hunyuan3D 纹理管线**使用的 C++/CUDA 扩展。

### 8.1 `__init__.py`

`load_extension_with_recovery`（约 85–127 行）：JIT 编译扩展失败时清理缓存重试，被子包复用。

### 8.2 `hunyuan3d_rasterizer/`

- `rasterizer.h` 约 53–54 行声明 `rasterize_image_gpu(V, F, D, width, height, ...)` 等，含重心坐标辅助；
- `rasterizer_gpu.cu` + `rasterizer.cpp`：CUDA 光栅化实现；
- `__init__.py` 约 32–40 行 JIT 加载 `rasterizer.cpp` + `rasterizer_gpu.cu`，导出 `rasterize`（三角网格 → face index + 重心）、`interpolate`（顶点属性插值）（约 44–85 行）。

### 8.3 `mesh_processor/`

- `mesh_processor.cpp`：`meshVerticeInpaint_smooth`——基于纹理与 mask、UV 与拓扑对顶点颜色做 **inpaint / 平滑**（C++ + pybind）；
- `__init__.py` 约 39–58 行加载扩展，导出 `meshVerticeInpaint`。

> 这两个扩展服务于 Hunyuan3D 类 3D 生成管线，与 DiT 的 `layers/*` 无直接代码依赖。

## 9. 一张图：算子调用矩阵

```text
                    ┌────────────────────────────────┐
                    │  CustomOp.dispatch_forward     │
                    │  (按 current_platform 选分支)   │
                    └────────────────────────────────┘
                                    │
   ┌──────────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
   ▼                  ▼              ▼              ▼              ▼              ▼
forward_cuda    forward_hip     forward_npu    forward_xpu    forward_musa    forward_native
   │                  │              │              │              │              │
 sgl_kernel /       native       torch_npu      sgl_kernel    Triton /       PyTorch
  Triton /       (RMSNorm/      npu_swiglu    flash_attn    F.layer_norm    回退
  CuTeDSL         Elementwise)   npu_rms_norm                SwishGLU
  fused_*
```

## 10. 与 `torch.compile` 的交互

- 所有 `CustomOp` 子类都提供 `forward_native`，对 Dynamo 友好；
- `NPU LayerNorm.forward_native` 上 `@torch.compile(disable=npu)`，避免 NPU 编译栈问题；
- NVFP4 JIT 预热（`prewarm_nvfp4_jit_modules`）使用 `@torch.compiler.disable` 规避被包装（详见 [`06_torch_compile.md`](./06_torch_compile.md)）；
- `direct_register_custom_op` 为算子提供 fake/meta，支持 Inductor 下的 symbolic shape。

## 11. 调优建议

- **ROCm 上 RMSNorm**：由于 `forward_hip` 回 native，关键路径若瓶颈在 Norm，可考虑自己把 AITer RMSNorm 打在 `forward_hip`；
- **MPS**：开启 `SGLANG_USE_MLX=1` 使用 MLX 融合 Metal Norm；
- **FLUX 量化模型**：`_fused_gelu_mlp` 已覆盖主要 MLP，不需要再额外融合；
- **自定义模型接入**：新 DiT 若有 AdaLN-modulate pattern，可直接用 `MulAdd` 的 Triton 路径而不是手写 elementwise；
- **CuTeDSL 融合**：维度要满足（hidden 要与 Tile 对齐），否则 warn + native，要关注日志；
- **QK Norm + RoPE**：在 Hopper 以上推荐走 `fused_inplace_qknorm_rope` JIT，其次 `apply_flashinfer_rope_qk_inplace`。
