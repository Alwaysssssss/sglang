# `python/sglang/srt/batch_invariant_ops` 模块分析

## 定位

`batch_invariant_ops` 提供面向确定性推理的 batch-invariant 算子替换。它在开启 deterministic inference 时替换部分 CUDA aten 实现，使 `mm/addmm/bmm/_log_softmax/mean.dim` 和 RMSNorm 尽量避免随 batch 组合变化而产生数值差异。

## 关键文件与对象

- `batch_invariant_ops.py`：唯一核心文件。
- `enable_batch_invariant_mode()`：注册自定义 CUDA aten impl，并 monkeypatch `torch.bmm`。
- `set_batch_invariant_mode()`：上下文式切换 batch-invariant 实现注册状态。
- `matmul_persistent` / `matmul`：优先走 DeepGEMM bf16 路径，否则使用 Triton persistent matmul。
- `bmm_batch_invariant`、`log_softmax`、`mean_dim`：替换 batch 相关基础算子。
- `rms_norm_batch_invariant`：被 `layers.layernorm.RMSNorm` 在确定性模式下调用。

## 运行流程

`ModelRunner` 在 `enable_deterministic_inference` 时调用 `enable_batch_invariant_mode()`。该函数通过 `torch.library` 注册 CUDA aten 覆盖实现，并把 `torch.bmm` 改到本模块的 batch-invariant 路径。模型执行期间，线性层、softmax、mean、RMSNorm 等会走新实现，从而减少 batch 排布对计算顺序的影响。

## 依赖关系

该模块依赖 Triton、DeepGEMM wrapper、`utils.common` 的平台判断和 `layers.layernorm`。它被 `model_executor/model_runner.py`、`layers/layernorm.py` 和部分 MoE Triton 路径引用。

## 设计要点和风险

- 这是进程级全局替换，影响所有 CUDA aten 调用；启用范围应尽量清晰。
- 支持 shape/dtype 有边界：DeepGEMM 路径要求 bf16 contiguous 和合适维度；`log_softmax` 只支持最后一维；`bmm` 只支持 3D。
- `set_batch_invariant_mode` 能恢复 torch library impl 状态，但 monkeypatched `torch.bmm` 的恢复语义需要谨慎确认。
- 确定性路径通常牺牲性能，尤其是替换 matmul 或大 vocab softmax 时。
