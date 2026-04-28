# 06. 编译优化（torch.compile + NVFP4 JIT + NPU torchair）

> 源码位置：`python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py`（关键函数 `_maybe_enable_torch_compile`、`_needs_nvfp4_jit_prewarm`）、`runtime/pipelines/diffusers_pipeline.py`（`_apply_torch_compile`）、`runtime/layers/custom_op.py`、`runtime/layers/utils.py`、`runtime/server_args.py`、`sglang/jit_kernel/nvfp4.py`、`sglang/srt/utils/common.py::get_compiler_backend`。

`multimodal_gen` 里 `torch.compile` 被嵌入到 **DenoisingStage 的生命周期**而不是加载期一次性编译。主要目标：

1. **在 cache-dit 已挂载之后**才 compile transformer（避免 cache-dit patch 被 Inductor 破坏）；
2. **NVFP4 JIT 预热**：避免 Dynamo 追踪时才去 build JIT 模块；
3. **启用 compute/comm overlap**：利用 Inductor 的 `reorder_for_compute_comm_overlap`；
4. **NPU 专用 backend**：torchair + `disable` 特定子图。

## 1. 总体触发点

文件：`runtime/pipelines_core/stages/denoising.py`。

- `DenoisingStage.__init__`（约 173–175 行）：对 `self.transformer`、（双塔时）`self.transformer_2` 各调用 `_maybe_enable_torch_compile`；
- 延迟加载 transformer 的路径（约 516–524 行）：加载完成后先 `_maybe_enable_cache_dit(...)`，再 `_maybe_enable_torch_compile(self.transformer)`。**顺序不能颠倒**。

`_maybe_enable_torch_compile(module)`（约 197–234 行）的核心：

```python
if not server_args.enable_torch_compile or not isinstance(module, nn.Module):
    return
if self._needs_nvfp4_jit_prewarm(module):
    prewarm_nvfp4_jit_modules()
if is_npu:
    compile_kwargs = dict(fullgraph=False, dynamic=False,
                          backend=get_compiler_backend())
else:
    torch._inductor.config.reorder_for_compute_comm_overlap = True
    compile_kwargs = dict(fullgraph=False, dynamic=None,
                          mode=os.environ.get("SGLANG_TORCH_COMPILE_MODE",
                                              "max-autotune-no-cudagraphs"))
module.compile(**compile_kwargs)
```

关键参数解读：

- `fullgraph=False`：允许有 graph break；因为 DiT 会调到 FlashAttention / Sage / 稀疏 attention 等未被 Inductor 完整支持的自定义算子；
- `dynamic=None`：让 PyTorch 自动决定 dynamic shape；
- `mode="max-autotune-no-cudagraphs"`：启 Inductor 最大 autotune，**但不启用 CUDA Graph**。

**编译对象是整个 transformer 模块**（`nn.Module.compile` 等价于 `torch.compile(self, ...)`），**不是逐 block**。这与 LLM 的 piecewise CUDA graph 思路不同。

## 2. NVFP4 JIT 预热

`_needs_nvfp4_jit_prewarm(module)`（约 236–244 行）：遍历子模块，如有 `ModelOptFp4LinearMethod` 则 True。

若为 True，在 compile **之前**调用：

```python
sglang.jit_kernel.nvfp4.prewarm_nvfp4_jit_modules()
```

该函数在 `jit_kernel/nvfp4.py` 上带 `@torch.compiler.disable`（约 50–60 行），**不会被 Dynamo 包装**。这样可以提前完成 NVFP4 算子的 JIT 编译，避免 `torch.compile` 追踪过程中碰到未就绪的 JIT 模块导致 recompilation / fallback。

## 3. Inductor 通信重叠

非 NPU 路径里在 compile 前显式：

```python
torch._inductor.config.reorder_for_compute_comm_overlap = True
```

这是 Inductor 的一个全局开关（对所有随后编译的图生效），会在调度器里把 compute 与 comm op 重排，提升 TP / SP 场景下的重叠率。开启时长视模型/图结构实际受益，通常在多卡长序列 DiT 上有 5–15% 提升。

## 4. CUDA Graph 与 piecewise

默认 `mode` 名带 `no-cudagraphs`：**Inductor 内 CUDA Graph 被禁用**。

`AttentionBackend.accept_output_buffer` 注释（`attention_backend.py` 约 22–25 行）提到 piecewise cudagraph，属于 attention 后端的设计允许；但**当前 DenoisingStage 路径未手写 `torch.cuda.graph` 捕获**。

`gpu_worker.py` 中**没有** `torch.compile` 或 cuda graph 逻辑，管理完全下放给 stage。

## 5. Ascend NPU 专属路径

- `get_compiler_backend()` 定义在 `sglang/srt/utils/common.py` 约 1981–2007 行，NPU 分支返回 torchair backend；
- NPU 下 `dynamic=False`：torchair 对动态 shape 支持较弱；
- `runtime/layers/layernorm.py` 的 `LayerNorm.forward_native` 使用 `@torch.compile(..., disable=current_platform.is_npu())`（约 303–304 行），**对 NPU 禁用 LayerNorm 子图编译**，因为 torchair 路径不稳定；
- `runtime/layers/utils.py::direct_register_custom_op` 在 NPU 上用 `PrivateUse1` 注册（约 107–111 行）。

## 6. Diffusers 后端的 compile 路径

文件 `runtime/pipelines/diffusers_pipeline.py::_apply_torch_compile`（约 472–475 行附近）：

- 顺序固定 **先 `_apply_cache_dit` 再 `_apply_torch_compile`**；
- 若 cache-dit 已启用，会尝试 `cache_dit.set_compile_configs()`（约 620–625 行）——这是 cache-dit 包为 compile 场景提供的兼容配置；
- 底层依然走 `torch.compile`，只是作用在 diffusers 的 pipeline 组件上。

## 7. 与量化的兼容性

- **FP8（ModelOpt / 通用）**：transformer 内 Linear 走 `quant_method.apply`，可被 Dynamo 正常 trace。主要 Inductor 能识别 `apply_fp8_linear` 这类子图；
- **NVFP4**：依赖 JIT 预热，否则 Dynamo 会在 trace 时触发 JIT 构建，导致 recompile；
- **Nunchaku SVDQuant**：其 kernel 是带 `@custom_op` 的 CUDA 扩展；是否完整兼容 compile 需按发行版验证，保守起见可 `torch.compiler.disable` 外层或不启用；
- **ModelSlim（昇腾）**：需与 torchair 共存，建议启用 compile 前先压测。

## 8. 与 attention backend 的兼容性

- FA / FA2 / FA3 / FA4：通过 `jit_kernel.flash_attention` 暴露的 ATen op 形式，`fullgraph=False` 能跨越；
- Sage / Sage3：外部包，多数是 ATen op 包装；
- 稀疏注意力（VSA / VMoBA / SVG2 / STA / SLA）：底层多为 Triton kernel + Python 调度，建议保持 `fullgraph=False`，否则会频繁 graph break；
- SDPA：完美兼容。

## 9. `ServerArgs` 相关字段

- `enable_torch_compile: bool = False`（`server_args.py` 约 201–202 行）；
- CLI `--enable-torch-compile`（约 910–913 行），doc 写明加速 DiT；
- **没有** `compile_mode` 字段：mode 由 **环境变量 `SGLANG_TORCH_COMPILE_MODE`** 控制。

## 10. 环境变量一览

| 变量 | 作用 |
|------|------|
| `SGLANG_TORCH_COMPILE_MODE` | Inductor mode，默认 `max-autotune-no-cudagraphs` |
| `SGLANG_USE_AITER` | ROCm 上 DiT 模型内切换到 AITer kernel |
| `SGLANG_USE_MLX` | MPS 上启用 MLX 融合 Metal Norm |
| `SGLANG_DIFFUSION_ATTENTION_BACKEND` | 强制指定 attention backend |

## 11. 推荐的 Compile 配置

| 场景 | `enable_torch_compile` | `SGLANG_TORCH_COMPILE_MODE` | 备注 |
|------|:---------------------:|------------------------------|------|
| Hopper / Ada + BF16 通用 DiT | ✔ | `max-autotune-no-cudagraphs` | 默认 |
| Blackwell + FLUX2 NVFP4 | ✔ | `max-autotune-no-cudagraphs` | 需 NVFP4 JIT 预热（自动触发）|
| 开发迭代 | ✔ | `reduce-overhead` | 减小编译时间 |
| 多卡 + 稀疏 attention | ✔ | `max-autotune-no-cudagraphs` | Inductor 重排利于 comm overlap |
| NPU | ✔ | — | 自动 torchair backend |
| 首次冷启动很慢不可接受 | ✘ | — | 纯 eager |

## 12. 排障清单

- **首次 request 很慢**：这是正常 compile 时间；可通过 warmup request 提前触发。`DenoisingStage` 已排除 `batch.is_warmup` 下 cache-dit，但 compile 会随第一次真实请求执行。
- **频繁 recompile**：Dynamo `shape` 触发 guard 失效；可通过 `dynamic=True` 或提供固定 shape warmup 解决。
- **`AttributeError: '_original_forward'`**：cache-dit 把 forward 包成 partial，若 compile 在 cache-dit 之前挂，会破坏拦截；确保顺序。
- **NPU 上段级 OOM**：torchair backend 对内存布局敏感，可减小 `sp_degree` 或减小 batch。
- **NVFP4 第一个 step 卡住**：没预热；检查 `_needs_nvfp4_jit_prewarm` 返回、`ModelOptFp4LinearMethod` 是否正确识别。
- **多卡 compute/comm 没重叠**：检查 `reorder_for_compute_comm_overlap` 是否为 True（NPU 默认不开）；部分稀疏 attention 会破坏 overlap。

## 13. 与 `CustomOp` 的交互细节

- 所有 `CustomOp.forward_native` 必须 **可 trace**（没有 `print` / host-side 条件跳转）；
- `register_custom_op` 为算子提供 fake/meta（`runtime/layers/utils.py` 约 193–267 行），让 Inductor 能推断输出 shape；
- `CustomOpWrapper` 注释写明 lazy 注册与 compile 不兼容时需显式 `eager=True`（约 151–158 行）；
- 部分算子（例如 `prewarm_nvfp4_jit_modules`）使用 `@torch.compiler.disable` 完全避开 Dynamo。

---

> 结合 [`03_cache.md`](./03_cache.md) 的 cache-dit 生命周期、[`02_quantization.md`](./02_quantization.md) 的 NVFP4 路径、[`01_parallelism.md`](./01_parallelism.md) 的 SP + TP + comm overlap 一并阅读，可以建立完整的「并行 + 缓存 + 编译」调优视图。
