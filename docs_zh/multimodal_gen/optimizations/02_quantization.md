# 02. 量化（Quantization）

> 源码位置：`python/sglang/multimodal_gen/runtime/layers/quantization/`、`runtime/layers/linear.py`、`runtime/loader/transformer_load_utils.py`、`runtime/utils/quantization_utils.py`、`configs/quantization/`、`tools/convert_hf_to_fp8.py`、`tools/build_modelopt_fp8_transformer.py`、`tools/build_modelopt_nvfp4_transformer.py`、`runtime/pipelines/flux_2_nvfp4.py`。

## 1. 概览与注册表

多模态生成子系统当前支持 **5 种主量化方案 + 1 种外挂方案**，通过 `QuantizationConfig` / `LinearMethodBase` 抽象对接；核心注册表位于 `runtime/layers/quantization/__init__.py`（约 18–31 行）：

```18:31:python/sglang/multimodal_gen/runtime/layers/quantization/__init__.py
QuantizationMethods = Literal[
    "fp8", "modelopt", "modelopt_fp8", "modelopt_fp4", "modelslim"
]
...
_CUSTOMIZED_METHOD_TO_QUANT_CONFIG = {
    "modelopt": ModelOptFp8DiffusionConfig,
    "modelopt_fp8": ModelOptFp8Config,
    "modelopt_fp4": ModelOptFp4Config,
    "modelslim": ModelSlimConfig,
    "fp8": Fp8Config,
}
```

`Nunchaku / SVDQuant`（`quant_method: "svdquant"`）**不走这个注册表**，而是通过 `ServerArgs.nunchaku_config` 独立注入，在加载路径中替换 Linear 类。

| 名称 | 类型 | 权重格式 | 激活格式 | 主要依赖 |
|------|------|----------|----------|----------|
| `fp8` | E4M3 per-tensor / per-channel / block | `float8_e4m3fn` + `weight_scale`（或 block 的 `weight_scale_inv`） | `static` / `dynamic` | `sglang.srt.layers.quantization.fp8_utils`、可选 Marlin |
| `modelopt` / `modelopt_fp8` | 静态 FP8 (per-tensor) | 同上 + 标量 `input_scale` | `static` | NVIDIA ModelOpt 导出 + `apply_fp8_linear` |
| `modelopt_fp4` | NVFP4 (E2M1) | `uint8` packed + `weight_scale` (group) + `alpha` | 运行时 `fp4_quantize` | FlashInfer / sgl_kernel |
| `modelslim` | W8A8 / W8A8_dyn / W4A4_dyn (昇腾) | 逐层按 `quant_model_description.json` | 随 scheme | `sglang.srt...ModelSlimW8A8Int8/W4A4Int4` |
| `svdquant` (Nunchaku) | W4A4 / NVFP4 + 低秩 LoRA 合并 | `qweight`、`wscales`、`smooth_*`、`proj_down/up` | CUDA kernel 内融合 | Nunchaku 包 |

## 2. 抽象基类与挂接机制

与 vLLM 风格一致：

- **`QuantizationConfig`**（`quantization/configs/base_config.py` 约 65–152 行）：定义 `get_name`、`from_config`、`get_quant_method(layer, prefix) -> QuantizeMethodBase`；
- **`QuantizeMethodBase`**（同文件约 19–50 行）：`create_weights`、`apply`、`process_weights_after_loading`；
- **`LinearMethodBase`**（`runtime/layers/linear.py` 约 91–126 行）：专用于 Linear。`UnquantizedLinearMethod` 为默认（未量化）。

`LinearBase.__init__` 在 `quant_config` 非空时调用 `quant_config.get_quant_method(self, prefix)`（`linear.py` 约 196–199 行）。**所有量化方案都只在 `isinstance(layer, LinearBase)` 时替换方法**，Embedding / Conv 等保持浮点（除非另有特化，如 Nunchaku 的 LoRA 路径）。

## 3. 逐方案解析

### 3.1 `fp8`（通用 HF FP8）

文件 `quantization/fp8.py`。

- `Fp8Config.get_quant_method`（约 144–152 行）：若 prefix 匹配 `ignored_layers` → `UnquantizedLinearMethod`；否则 `Fp8LinearMethod`。
- `Fp8LinearMethod.create_weights`（约 193–300 行）：根据 `activation_scheme` 与 `block_quant` 创建不同 parameter：`ModelWeightParameter`（权重）、`PerTensorScaleParameter` / `BlockQuantScaleParameter`。
- `Fp8LinearMethod.apply`（约 442–498 行）：
  - `use_marlin` → Marlin；
  - `block_quant` → `dispatch_w8a8_block_fp8_linear()`；
  - 否则 `apply_fp8_linear`（`sglang.srt.layers.quantization.fp8_utils`）。
- `get_min_capability = 80`（Ampere+）。
- 注意 `use_per_token_if_dynamic=False`（497 行），即默认 per-tensor，与 LLM 侧某些路径不同。
- 块量化（`weight_scale_inv`）激活强制 dynamic。

### 3.2 Diffusion 专用 `modelopt` （= `ModelOptFp8Config` in `modelopt_fp8.py`）

文件 `quantization/modelopt_fp8.py`。与 `modelopt_quant.py` 内同名类不同：前者服务于 diffusers 风格的**扁平 config**，后者服务于 ModelOpt 原生嵌套 config。

- `get_name() == "modelopt"`；`get_min_capability == 89`（Hopper 及以上）；
- `_is_layer_ignored(prefix)` 用 glob + 首段匹配（约 90–104 行）；
- `ModelOptFp8LinearMethod.process_weights_after_loading` 对 diffusion 简化为 **max scale + 转置**（约 172–189 行）。注释 173–176 行明确写出：**避免 LLM 路径在 CPU 加载阶段调 CUDA 导致的兼容问题**。

### 3.3 `modelopt_quant.py` 中的 FP8 / FP4

提供 ModelOpt 原生嵌套配置（`hf_quant_config.json`）的解析。

- `ModelOptQuantConfig`（约 77–136 行）：基类，管理 `exclude_modules` 正则、`packed_modules_mapping`。
- `ModelOptFp8Config`（约 138–192 行）：`get_name → "modelopt_fp8"`（155 行），`get_quant_method → ModelOptFp8LinearMethod`（191 行）。
- `ModelOptFp4Config`（约 195–311 行）：仅接受 `quant_algo == "NVFP4"`（291–294 行），`get_quant_method → ModelOptFp4LinearMethod`（310 行）。
- `override_quantization_method`（约 113–127 行）：用户给 `modelopt` / `modelopt_fp8` / `modelopt_fp4` 时，若与 checkpoint 的 `quant_algo` 冲突会**重新映射**到 `modelopt_fp8` 或 `modelopt_fp4`。

#### NVFP4 `apply`

`ModelOptFp4LinearMethod.apply`（约 568–627 行）：

1. `fp4_quantize(x, layer.input_scale_inv)` → `x_fp4, x_scale_interleaved`；
2. **FP4 GEMM**（来自 `current_platform.get_modelopt_fp4_gemm_op`，可能是 FlashInfer 或 sgl_kernel）；
3. 辅助 `pad_nvfp4_weight`、`pad_nvfp4_activation_for_cutlass`、`slice_nvfp4_output`（来自 `sglang.srt.layers.quantization.modelopt_quant`，29–32 行）；
4. 对齐 FlashInfer 的 `shuffle_matrix_a` 等（526–535 行）。

### 3.4 `modelslim`（昇腾 / W8A8 / W4A4）

文件 `quantization/modelslim.py`。

- 由 `quantization_utils.find_quant_modelslim_config`（约 99–107 行）在权重目录下读 `quant_model_description.json`，注入伪 `quant_method: "modelslim"`；
- `ModelSlimConfig.get_quant_method`（约 77–106 行）决定跳过 / 使用 scheme，具体 kernel 在 `ModelSlimLinearMethod.layer.scheme` 里，引用 `sglang.srt.layers.quantization.modelslim.schemes.ModelSlimW8A8Int8` / `ModelSlimW4A4Int4`。

### 3.5 Nunchaku (`svdquant`)

文件 `quantization/nunchaku_linear.py` + `configs/quantization/nunchaku.py`。

- `NunchakuConfig.get_name == "svdquant"`；`get_quant_method`（约 81–121 行）根据 `get_nunchaku_quant_rules()` 跳过或返回：
  - `NunchakuAWQLinearMethod`（W4A16）；
  - `NunchakuSVDQLinearMethod`（W4A4 或 NVFP4）；
- 精度 `nvfp4` 时 `group_size=16`，否则 `64`（约 35–38 行）；
- 核心 kernel：`svdq_gemm_w4a4_cuda`、`awq_gemv_w4a16_cuda`、`svdq_quantize_w4a4_act_fuse_lora_cuda`（约 14–21 行），**激活量化与 LoRA 低秩路径在同一 kernel 内融合**（172–196 行），是与通用 QDQ 量化最大的差异。

## 4. 配置解析链路

统一入口：`runtime/utils/quantization_utils.py`（函数 `get_quant_config`）。

1. **ModelSlim 检查**：若权重目录下存在 `quant_model_description.json`，构造字典交给 `ModelSlimConfig.from_config`；
2. **HF `quantization_config`**：读 `model_config["quantization_config"]`；ModelOpt 扁平配置会先经 `normalize_flat_modelopt_quant_config` 补齐 `quant_type`（约 19–36 行）；
3. **方法名解析**：`_resolve_quant_method_name`（约 70–88 行）：`quant_method == "modelopt"` 时按 `quant_algo` 映射到 `modelopt_fp8` 或 `modelopt_fp4`；
4. **嵌套 JSON**：`hf_quant_config.json` 由 `get_config_filenames()` 暴露并合并 `packed_modules_mapping`。

Transformer 侧加载管线：`resolve_transformer_quant_load_spec`（`transformer_load_utils.py` 约 356–402 行）优先级：

```
组件目录 config.json
  → 从 safetensors 推断 NVFP4（build_nvfp4_config_from_safetensors_list + _merge_modelopt_fp4_configs）
  → transformer_weights_path 目录 config
  → 逐个 safetensors metadata（get_quant_config_from_safetensors_metadata）
```

`_build_nvfp4_config_from_safetensors_files`（约 263–447 行）扫描 `_quantization_metadata`、`layers[].format == "nvfp4"` 或同时存在 `.weight + .weight_scale` 的模块名，推断 `group_size`、`exclude_modules`、`checkpoint_uses_packed_qkv`。

`handle_fp8_metadata_format`（约 193–200 行）把带 `layers`/`format_version` 的 diffusers FP8 元数据转成 `quant_method: "fp8"` + `activation_scheme: "dynamic"`。

加载后处理：`fsdp_load.py::maybe_load_fsdp_model`（约 197–209 行）对每个带 `quant_method` 的模块调用 `process_weights_after_loading`；随后 `model.post_load_weights()`。对于 checkpoint 是 FP8 但模块需要 BF16 的情况，`_maybe_dequantize_fp8`（约 82–108 行）按 `weight_scale` 反量化。

## 5. 激活量化位置

| 方案 | 激活量化位置 | 粒度 |
|------|--------------|------|
| `fp8`（非 block） | `apply_fp8_linear`（默认 `use_per_token_if_dynamic=False`，约 497 行）| per-tensor |
| `fp8`（block） | `w8a8_block_fp8_linear` | block 内按组 |
| `modelopt` / `modelopt_fp8` | `create_weights` 写入 per-tensor `input_scale`（约 161–170 行）| per-tensor 静态 |
| `modelopt_fp4` | `fp4_quantize(x, layer.input_scale_inv)`，produce `x_scale_interleaved` | group |
| Nunchaku SVDQ | `svdq_quantize_w4a4_act_fuse_lora_cuda` 内部（172–177 行）| block + LoRA 融合 |
| ModelSlim W8A8 / W4A4 | 由 scheme 决定 | 各异 |

## 6. 与 LoRA 共存

- **通用路径**：`runtime/layers/lora/linear.py` 的 `BaseLayerWithLoRA.forward`（约 77–102 行）若未合并，会在 base_layer 输出上再加 LoRA delta；`ColumnParallelLinearWithLoRA`（约 327–346 行）显式调用 `self.base_layer.quant_method.apply` 再加 LoRA。
- **合并路径**：`_merge_lora_into_data` 对 `base_layer.weight.data` 做 in-place 加法（约 248–253 行）。对 **FP8 / 整型打包权重**，这等价于直接改量化后权重，不总是正确；官方推荐的稳妥路径是**未合并**模式。
- **Nunchaku** 独立处理：LoRA 被融进激活量化 kernel（`lora_act_in` / `lora_up`）。

## 7. `ServerArgs` 开关

- **量化相关 CLI 主要是 Nunchaku**：`_adjust_quant_config` 备注 **当前仅处理 nunchaku**（约 332–346 行）。CLI 包括：
  - `--enable-svdquant`
  - `--transformer-weights-path`
  - `--quantization-precision`
  - `--quantization-rank`
  - `--quantization-act-unsigned`
  - 字段 `nunchaku_config`（默认 `NunchakuSVDQuantArgs()`，约 211–214 行），再由 `_adjust_quant_config` 转成 `NunchakuConfig | None`。
- **ModelOpt / FP8 / ModelSlim** 通常**不需要额外 CLI**：依赖 checkpoint 中的 `quantization_config` 或 safetensors metadata，由 `get_quant_config` / `_resolve_quant_config` 自动解析。

## 8. 预处理工具

| 工具 | 路径 | 作用 |
|------|------|------|
| HF → FP8（per-tensor / channel / block） | `tools/convert_hf_to_fp8.py` | 写 `weight_scale` 或 `weight_scale_inv`，并生成 `quantization_config` |
| ModelOpt → SGLang FP8 transformer | `tools/build_modelopt_fp8_transformer.py` | 重建 scale、写 `quantization_config` / `_quantization_metadata` |
| ModelOpt NVFP4 + 混合 BF16 | `tools/build_modelopt_nvfp4_transformer.py` | 更新 `quantization_config.ignore`、`swap_weight_nibbles` |

## 9. 适配器与特殊模型

`_Flux2Nvfp4FallbackAdapter`（`transformer_load_utils.py` 约 171–219 行）：当模型是 `Flux2Transformer2DModel` + `modelopt_fp4` + mixed 权重 + `tp_size>1` 时，**关闭 `dit_cpu_offload` / `text_encoder_cpu_offload`**，避免 TP all-gather 时和 CPU offload 冲突。

`_ModelOptFp8OffloadAdapter`（约 222–259 行）：使用 `modelopt_fp8` 时**强制关闭 `dit_cpu_offload`**，但保留 layerwise offload。

`pipelines/flux_2_nvfp4.py` 里的 `Flux2NvfpPipeline` 继承 `Flux2Pipeline`，从 `black-forest-labs/FLUX.2-dev` 取非 transformer 组件，transformer 走 `transformer_weights_path`（可自动选 `*-mixed.safetensors`，约 34–55 行）。`test/server/perf_baselines.json` 的 `flux_2_nvfp4_t2i` 就是针对它的性能门禁。

## 10. 加载流程简图

```text
ServerArgs (transformer_weights_path, nunchaku_config)
    │
    ▼
TransformerLoader.load_customized
    ├─ resolve_transformer_safetensors_to_load
    └─ resolve_transformer_quant_load_spec
          ├─ _resolve_quant_config
          │     ├─ get_quant_config → (Fp8Config / ModelOpt* / ModelSlim)
          │     ├─ build_nvfp4_config_from_safetensors_list
          │     └─ _merge_modelopt_fp4_configs
          └─ adapters: Flux2 NVFP4 / ModelOpt FP8 offload / Nunchaku
    │
    ▼
model_cls(hf_config, config, quant_config | nunchaku_config)
    │
    ▼
maybe_load_fsdp_model → load weights → quant_method.process_weights_after_loading
    │
    ▼
post_load_hooks (e.g. _patch_nunchaku_scales)
```

## 11. 跑通的组合

| 模型 | 方案 | 备注 |
|------|------|------|
| FLUX.2-dev | `modelopt_fp4`（NVFP4）+ BF16 mixed | 由 `Flux2NvfpPipeline` 装载 |
| FLUX.1 | `svdquant` / `modelopt_fp8` | keep_bf16 列表见 `build_modelopt_fp8_transformer.py` |
| Wan / Wan2.2 | `fp8` 动态 / `modelopt_fp8` | 与 cache-dit / layerwise offload 组合需注意冲突 |
| 通用 diffusers 权重 | `fp8`（由 `convert_hf_to_fp8` 产出） | 最通用、无后端依赖 |

## 12. 调优建议

- **Blackwell + FLUX2** 首选 NVFP4，结合 `enable_torch_compile` + NVFP4 JIT 预热（详见 [`06_torch_compile.md`](./06_torch_compile.md)）；
- **Hopper / L40S** 用 `modelopt_fp8` 或 `fp8`；
- **24GB 卡 + FLUX.1** 首选 Nunchaku SVDQuant（W4A4 + LoRA 合并）；
- **昇腾 NPU** 用 `modelslim` 导出的 W8A8；
- **QLoRA 场景** 建议不要 merge，保留未合并路径；
- **不要同时**启用 `modelopt_fp8` 和 `dit_cpu_offload`，会被 `_ModelOptFp8OffloadAdapter` 自动关闭但语义上是冲突的。
