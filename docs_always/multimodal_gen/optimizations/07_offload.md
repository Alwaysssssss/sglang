# 07. 显存 / Offload 策略

> 源码位置：`python/sglang/multimodal_gen/runtime/utils/layerwise_offload.py`、`runtime/managers/gpu_worker.py`、`runtime/loader/component_loaders/*`、`runtime/loader/fsdp_load.py`、`runtime/pipelines_core/stages/decoding.py`、`runtime/pipelines/diffusers_pipeline.py`、`runtime/server_args.py`。

扩散模型显存压力集中在：**DiT 权重 + 激活（占大头）**、**Text / Image Encoder（CLIP / T5 / Llama）**、**VAE（高分辨率时激活爆炸）**。`multimodal_gen` 在这三条线上分别做了 offload，并对 Wan / MOVA / Flux 等大型视频模型自动推断默认值。

## 1. Offload 类型总览

| 类型 | 含义 | 主要入口 |
|------|------|----------|
| **整模 DiT CPU offload** (`dit_cpu_offload`) | 通过 **FSDP `CPUOffloadPolicy`** 把 DiT 权重放在 CPU，推理时按策略拉上 GPU | `transformer_loader.maybe_load_fsdp_model(..., cpu_offload=..., pin_cpu_memory=...)`（约 124–138 行）|
| **逐层 / layerwise** (`dit_layerwise_offload`) | 按 block 把权重 consolidate 到 pinned CPU，用独立 CUDA stream prefetch；**与 `dit_cpu_offload` 互斥** | `runtime/utils/layerwise_offload.py` + `gpu_worker.configure_layerwise_offload` |
| **组件级 offload** | `text_encoder_cpu_offload`、`image_encoder_cpu_offload`、`vae_cpu_offload` 分别控制对应 loader/stage | 各 `component_loaders/*.py` |
| **Diffusers 后端全家桶** | 任一组件 offload 为真时 `pipe.enable_model_cpu_offload(...)` | `diffusers_pipeline.py` 约 452–465 行 |
| **VAE Tiling / Slicing** | 解码时分块 / 分片，降低 VAE 激活峰值 | `decoding.py::decode`、`diffusers_pipeline._apply_vae_optimizations` |

**说明**：没有独立的 "sequential offload" 标志。顺序性体现在 layerwise 的**按层 pre/post hook 驱动的 prefetch/release**，以及各 denoising stage 对 `dit_cpu_offload` 的条件分支（例如 `if not server_args.dit_cpu_offload` 触发某些 fuse 路径）。

## 2. Layerwise Offload 实现

### 2.1 核心类：`LayerwiseOffloadManager`

文件 `runtime/utils/layerwise_offload.py` 约 15–505 行。

**数据结构**：

- **Pinned CPU 存储**：按层、按 dtype 把连续参数拼成一块大 buffer：
  ```python
  cpu_buffer = torch.empty(..., pin_memory=pin_cpu_memory)
  ```
  （约 175–178 行）；非连续参数单独 `empty_strided(..., pin_memory=...)`（约 141–148 行）。
- **GPU 侧占位**：原参数 `weight.data` 指向共享的 `torch.empty((1,), device=gpu, dtype=...)`（`_get_shared_empty_tensor`，约 87–92、196 行）——**每个 dtype 共享一个 1-byte 占位**，保证参数在 CPU 上时 GPU 还是可以正确被 `torch.compile` / FSDP 访问到形状。

**专用 stream**：

```python
self.copy_stream = torch.get_device_module().Stream()
```

（约 50 行）；`prefetch_layer` 在 `copy_stream` 上 `copy_` CPU→GPU，再把参数 `.data` view 回映射好的 GPU 区域（约 244–278 行）。

### 2.2 Prefetch 深度与环形调度

`OffloadableDiTMixin.configure_layerwise_offload`（约 516–538 行）读取 `dit_offload_prefetch_size`：

- 整数 → 固定层数；
- `(0, 1)` 内小数 → `(num_layers-1) * ratio` 作为窗口大小。

**Forward hook 逻辑**（约 461–498 行）：

- **pre_hook**：对当前层 `i`：
  - 若存在 `_prefetch_events[i]` 则 `current_stream.wait_event(event)`；
  - 每 `prefetch_size` 层主动触发 `prefetch_layer(i+prefetch_size .. i+2*prefetch_size-1)`（对 `num_layers` 取模形成环）；
- **post_hook**：`release_layer(i)` 把该层 GPU `.data` 换回占位，CPU 上的原始数据保留。

**双缓冲语义**：代码没有显式 `double_buffer` 字段，但语义上是 **当前层在 compute stream 上跑，同时 copy stream 预取后续窗口内的层**，配合 `Event` 做跨流同步，典型的流水线重叠。

### 2.3 辅助：访问真实 CPU 权重

`OffloadableDiTMixin.iter_materialized_weights` / `update_cpu_weights`（约 572–433 行）绕过占位参数读真实 CPU 权重，供：

- Checksum 校验；
- LoRA refit；
- RL 权重热加载（`weights_updater.py`）等。

### 2.4 关键约束

- **与 `dit_cpu_offload` 互斥**：`_validate_offload`（`server_args.py` 约 1346–1361 行）在启用 layerwise 时若 `dit_cpu_offload is None` 自动置 False；
- **与 `use_fsdp_inference` 互斥**：layerwise 开启时会强制关闭 FSDP（约 1351–1355 行）；
- **与 cache-dit 互斥**：`SGLANG_CACHE_DIT_ENABLED` 与 `dit_layerwise_offload` 不能同时启用（约 1363–1368 行）；
- **MPS 强制关闭**：`_adjust_platform_specific`（约 658–693 行）在 MPS 平台上把 `dit_layerwise_offload = False`。

## 3. FSDP CPU Offload

`runtime/loader/fsdp_load.py`：`CPUOffloadPolicy(pin_memory=pin_cpu_memory)`（约 263 行附近）。

Text Encoder 组件也有类似逻辑：`text_encoder_loader.shard_model(..., pin_cpu_memory=server_args.pin_cpu_memory)`（约 330–337 行）。

## 4. Pinned Memory

`ServerArgs.pin_cpu_memory: bool = True`（`server_args.py` 约 195 行）；CLI `--pin-cpu-memory`（约 978–982 行）。默认开启；关闭的主要场景是系统限制 mlock 配额。

## 5. VAE Tiling / Slicing

### 5.1 原生 SGLang 解码

`runtime/pipelines_core/stages/decoding.py::decode`（约 148–152 行）：

```python
if server_args.pipeline_config.vae_tiling:
    self.vae.enable_tiling()
...
self.vae.decode(latents)
```

（**slicing 在此文件中未直接调用**，主要是给 Diffusers pipeline 用）。

解码完毕后：`offload_model` 若 `vae_cpu_offload=True` 则 `vae.to("cpu")`（约 182–183 行）。

### 5.2 Diffusers 后端

`diffusers_pipeline._apply_vae_optimizations`（约 479–513 行）：

```python
if vae_slicing: pipe.vae.enable_slicing()
if vae_tiling:  pipe.vae.enable_tiling()  # 或 pipeline 级 API
```

## 6. 自动默认值推断

### 6.1 `_adjust_offload`（非 CPU 平台）

`server_args.py` 约 368–404 行：

- **GPU 显存 < 30GB**：未指定的 `dit / text / image / vae` 四类 offload **全部默认 True**；
- **`task_type.is_image_gen()`**：`dit` / `text_encoder` 默认 True，`image_encoder` / `vae` 默认 False（图像任务不做 I2V 所以这两者没必要常驻 CPU）；
- **其它任务（视频）**：四类未指定时默认全 True。

### 6.2 `_adjust_platform_specific`

约 658–693 行：

- **MPS**：`dit_layerwise_offload = False`；
- **Wan / MOVA 且未开 cache-dit、且 `dit_layerwise_offload is None`**：
  - `enable_dit_layerwise_offload_for_wan_by_default()` 为真；
  - CUDA 平台；
  - **设备显存 < 130GB** → 自动置 True；
  - **设备显存 ≥ 130GB（如 H200）** → 强制 False（注释说明 H200 上 latency 反而回退）。

### 6.3 顺序依赖

`_adjust_parameters` 调用顺序（约 302–315 行）：**先 `_adjust_offload()`，后 `_adjust_platform_specific()`**。因此 platform specific 有权覆盖通用逻辑。

## 7. 多卡场景

- `GPUWorker` 每进程一个，layerwise 在每个 rank 上对**本进程的 DiT shard** 注册 hook；
- 一旦开启 FSDP 会与 layerwise 冲突，选一不选二；
- `_validate_offload` 日志在多卡显存充裕时会提示关闭 `dit_layerwise_offload`（约 1371–1373 行）；
- `CPUWorker`（`cpu_worker.py`）继承 `GPUWorker`，主要是 NUMA 绑核与 OpenMP（约 37–76 行），CPU 平台直接 `return`，不改 GPU offload 标志。

## 8. 参数一览（ServerArgs）

| 字段 | 默认 | CLI |
|------|------|-----|
| `dit_cpu_offload` | 自动（见 §6）| `--dit-cpu-offload` |
| `dit_layerwise_offload` | 自动（Wan/MOVA 非高显存平台默认 True）| `--dit-layerwise-offload` |
| `dit_offload_prefetch_size` | 默认 1 或比例 | `--dit-offload-prefetch-size` |
| `text_encoder_cpu_offload` | 自动 | `--text-encoder-cpu-offload` |
| `image_encoder_cpu_offload` | 自动 | `--image-encoder-cpu-offload` |
| `vae_cpu_offload` | 自动 | `--vae-cpu-offload` |
| `pin_cpu_memory` | True | `--pin-cpu-memory` |
| `vae_tiling` | 由 pipeline config 决定 | — |
| `vae_slicing` | 由 pipeline config 决定 | — |
| `use_fsdp_inference` | False | `--use-fsdp-inference` |
| `hsdp_replicate_dim` / `hsdp_shard_dim` | 自动 | — |

## 9. 调度简图

```text
request ─▶ EncodingStage
             │ text_encoder / image_encoder (可 cpu_offload=True, 用完 to("cpu"))
             ▼
           DenoisingStage
             │  if dit_layerwise_offload:
             │    pre_hook ─▶ prefetch_layer (copy stream) ─▶ compute ─▶ release
             │  elif dit_cpu_offload + FSDP:
             │    FSDP full_shard + CPUOffloadPolicy
             ▼
           DecodingStage
             │ if vae_tiling: vae.enable_tiling()
             │ vae.decode(latents)
             │ if vae_cpu_offload: vae.to("cpu")
             ▼
           save_outputs (RIFE / Real-ESRGAN)
```

## 10. 调优建议

- **24GB 卡跑 Wan2.1**：`dit_layerwise_offload=True` + `pin_cpu_memory=True` + `prefetch_size=2 或 0.2`；
- **48GB 卡跑 Wan2.2**：关 layerwise 开 `dit_cpu_offload + FSDP`；
- **80GB+ 多卡**：推荐直接 `use_fsdp_inference=True` 或全部关闭 offload；
- **H200 / B200**：offload 通常反而更慢，系统会自动关，请不要显式启用；
- **I2V 场景显存爆**：仅开 `text_encoder_cpu_offload` + `image_encoder_cpu_offload`，保留 VAE 常驻；
- **VAE OOM**：开 `vae_tiling=True`（原生）或 `vae_slicing=True`（Diffusers 后端）；
- **模型重载慢**：`pin_cpu_memory=True` + NVMe 直读会显著缩短权重加载时间；
- **cache-dit + offload 同时需求**：二选一，cache-dit 通常收益更大，选 cache-dit；
- **ROCm 平台**：AITer 的 RMSNorm 没有接入，整模 FSDP + `dit_cpu_offload` 是主要路径。

## 11. 引用清单

| 文件 | 符号 | 作用 |
|------|------|------|
| `runtime/utils/layerwise_offload.py` | `LayerwiseOffloadManager`、`OffloadableDiTMixin`、`iter_materialized_weights` | layerwise 实现 |
| `runtime/managers/gpu_worker.py` | `GPUWorker`（约 138–156 行调 `configure_layerwise_offload`）| pipeline build 后触发 |
| `runtime/loader/fsdp_load.py` | `maybe_load_fsdp_model`、`CPUOffloadPolicy` | FSDP CPU offload |
| `runtime/loader/component_loaders/*` | `text_encoder_loader`、`image_encoder_loader`、`vae_loader`、`transformer_loader` | 组件级 offload + pinned |
| `runtime/pipelines_core/stages/decoding.py` | `DecodingStage.decode` / `offload_model` | VAE tiling + 后 offload |
| `runtime/pipelines/diffusers_pipeline.py` | `_apply_vae_optimizations`、`enable_model_cpu_offload` | Diffusers 后端 |
| `runtime/server_args.py` | `_adjust_offload`、`_adjust_platform_specific`、`_validate_offload` | 默认值 + 校验 |
