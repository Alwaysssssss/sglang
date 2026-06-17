# `python/sglang/srt/compilation` 模块分析

## 定位

`compilation` 提供 SRT 自定义 `torch.compile` 后端和 piecewise CUDA/NPU graph capture 框架。它把模型 forward 拆成可编译/可捕获子图，并为 attention、通信、linear attention 等特殊 op 提供 split 边界。

## 关键文件

- `compile.py`：`install_torch_compiled` 等入口，给 module forward 装编译 trampoline。
- `backend.py`：`SGLangBackend` 和 graph split/compile 主流程。
- `compiler_interface.py`：`IntermediateTensors`、compiler manager 和编译接口抽象。
- `compilation_config.py`：`CompilationConfig`、split op 注册等配置。
- `cuda_piecewise_backend.py`、`npu_piecewise_backend.py`：平台 piecewise backend。
- `piecewise_context_manager.py`：forward context 保存和 `get_forward_context()`。
- `inductor_pass.py`、`fx_utils.py`、`pass_manager.py`、`fix_functionalization.py`：FX/Inductor pass 工具。
- `weak_ref_tensor.py`：弱引用 tensor 支持。

## 运行流程

piecewise graph runner 进入 capture 模式后，`install_torch_compiled` 替换模型 forward。首次运行触发 Dynamo 捕获，`SGLangBackend` 根据注册的 `sglang.*` split op 切分 FX graph；普通子图交给 Inductor/Eager 编译，特殊 split op 保持为运行时边界。运行阶段设置 forward context，使 `RadixAttention` 等层可以访问当前 `ForwardBatch` 与 attention layer 列表，并在合适 capture size 下 warmup/capture/replay。

## 依赖关系

该模块被 `model_executor/piecewise_cuda_graph_runner.py` 使用；`distributed.parallel_state`、`layers.radix_attention`、`layers.radix_linear_attention` 和部分模型/通信函数会注册 split/custom op。底层依赖 PyTorch Dynamo/Inductor、Triton、CUDA graph 和 NPU 后端。

## 设计要点和风险

- 代码依赖 PyTorch 私有/半私有编译 API，PyTorch 升级可能破坏行为。
- forward context 是跨模块隐式依赖，attention、MoE、quantization、模型代码会读取它；线程/嵌套执行需谨慎。
- graph split 边界是性能和正确性的关键，漏注册会导致不可捕获逻辑进入 Inductor，过度 split 会降低优化空间。
- cache load/write 逻辑还有实现空洞时，要避免把“已缓存编译产物”当成已验证能力。
