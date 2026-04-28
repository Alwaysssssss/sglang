# Wan2.1 端到端运行全流程：从 CLI 到最底层算子

本文把 `sglang generate` 敲下去那一瞬间，一直走到 GPU 上某个 FlashAttention kernel 的整条路径拆开，面向「想跟读 Wan-Video/Wan2.1 在 `sglang.multimodal_gen` 里究竟发生了什么」的开发者。所有行号都基于当前仓库的实际代码，便于你在源码里一路追下去。

前置阅读：

- [01_architecture_overview.md](./01_architecture_overview.md)
- [03_runtime_execution.md](./03_runtime_execution.md)
- [04_pipeline_and_stage.md](./04_pipeline_and_stage.md)
- [09_case_study_wan2_1.md](./09_case_study_wan2_1.md)
- [wan2_1_guide.md](./wan2_1_guide.md)（用户使用手册）

---

## 0. 总览：一次请求经过的文件

```
用户命令 / HTTP
     │
     ▼
sglang.multimodal_gen.runtime.entrypoints.cli / http_server
     │ 组装 ServerArgs + SamplingParams
     ▼
DiffGenerator          runtime/entrypoints/diffusion_generator.py
     │ from_pretrained → from_server_args → launch_server
     ▼
launch_server          runtime/launch_server.py
     │ 拉起 N 个 run_scheduler_process 子进程
     ▼
Scheduler              runtime/managers/scheduler.py  (rank0 负责 ZMQ ROUTER)
     │ _handle_generation
     ▼
GPUWorker.execute_forward  runtime/managers/gpu_worker.py
     │ self.pipeline.forward(req, server_args)
     ▼
WanPipeline            runtime/pipelines/wan_pipeline.py
     │ 继承 ComposedPipelineBase，create_pipeline_stages() = add_standard_t2i_stages()
     ▼
7 个 Stage 串联         runtime/pipelines_core/stages/*
     │ InputValidation → TextEncoding → LatentPrep → TimestepPrep
     │ → Denoising → (VAE) Decoding
     ▼
具体算子               runtime/models/{dits,vaes,encoders,schedulers}/*
     │ + runtime/layers/{linear,layernorm,attention,rotary_embedding,...}
     ▼
底层 kernel            FlashAttention / FlashInfer RoPE / cuBLAS GEMM /
                      NCCL all-to-all / all-reduce / all-gather
```

和 Wan2.1 直接相关的"七个文件"入口：

| 类别 | 路径 |
| --- | --- |
| Pipeline 类 | `python/sglang/multimodal_gen/runtime/pipelines/wan_pipeline.py` |
| Pipeline 配置 | `python/sglang/multimodal_gen/configs/pipeline_configs/wan.py` |
| 采样参数 | `python/sglang/multimodal_gen/configs/sample/wan.py` |
| 模型注册 | `python/sglang/multimodal_gen/registry.py`（`_register_configs` 中的 Wan 系列条目） |
| DiT 模型 | `python/sglang/multimodal_gen/runtime/models/dits/wanvideo.py` |
| VAE 模型 | `python/sglang/multimodal_gen/runtime/models/vaes/wanvae.py` |
| 自定义 Scheduler | `python/sglang/multimodal_gen/runtime/models/schedulers/scheduling_flow_unipc_multistep.py` |

---

## 1. 启动阶段：进程拓扑怎么搭起来

### 1.1 两条入口

- `sglang generate` → `DiffGenerator.from_pretrained(...).generate(...)`，`local_mode=True`，跑完退出。
- `sglang serve` → 直接 `launch_server(..., launch_http_server=True)`，启动 FastAPI。

两条链路底层一样，仅由 `local_mode` 决定要不要起 HTTP。

### 1.2 `DiffGenerator` 做的事

文件：`runtime/entrypoints/diffusion_generator.py`

1. 把用户 kwargs（含 `--model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers`、`--num-gpus 4`、`--ulysses-degree 2` 等）规范化成 `ServerArgs`。
2. `ServerArgs` 里调 `registry.get_model_info()`：
   - 根据 HF id 或本地路径命中 `_MODEL_HF_PATH_TO_NAME`，拿到该变体对应的 `PipelineConfig` 类（例：`WanT2V480PConfig`）和 `SamplingParams` 类（`WanT2V_1_3B_SamplingParams`）。
   - 实例化 `pipeline_config`，里面已经挂好 `WanVideoConfig`（DiT）、`WanVAEConfig`、`T5Config`、`flow_shift=3.0`、`postprocess_text_funcs=(t5_postprocess_text,)` 等静态配置。
3. `from_server_args()` → `launch_server(server_args, launch_http_server=False)`。

### 1.3 `launch_server` 的进程拓扑

文件：`runtime/launch_server.py`

- 按 `server_args.num_gpus` 每张卡 `spawn` 一个子进程 `run_scheduler_process`。
- 子进程内部：
  1. 设 `CUDA_VISIBLE_DEVICES`、`RANK`、`WORLD_SIZE`、`MASTER_ADDR/PORT`。
  2. `init_process_group(backend="nccl")`，然后按 `tp_size × sp_size × cfg_parallel` 构建 TP/SP/CFG/DP 的 `ProcessGroup`（`runtime/distributed/parallel_state.py`）。
  3. 构造 `Scheduler`：rank 0 创建 ZMQ `ROUTER`；非 0 rank 作为 worker 通过 pipe 等 rank 0 派活。
  4. 构造 `GPUWorker`，在里面完成真正的「设备绑定 + 模型加载 + pipeline 建图」。

### 1.4 `GPUWorker.init_device_and_model`

文件：`runtime/managers/gpu_worker.py`

- `build_pipeline(server_args)` → 调 `pipelines_core.__init__.build_pipeline`。
- 逻辑：
  ```python
  pipeline_cls = _PIPELINE_REGISTRY["WanPipeline"]  # 来自 wan_pipeline.py 的 EntryClass
  self.pipeline = pipeline_cls(model_path, server_args)
  ```
- `WanPipeline.__init__`（`ComposedPipelineBase.__init__`）依次：
  1. 创建 executor（`ParallelExecutor`）。
  2. `load_modules()`：读 `model_index.json`，对 `text_encoder, tokenizer, vae, transformer, scheduler` 五个组件走 `PipelineComponentLoader.load_component`：
     - `text_encoder` + `tokenizer` → UMT5-XXL（`runtime/models/encoders/t5.py`）
     - `vae` → `runtime/models/vaes/wanvae.py::AutoencoderKLWan`（`load_encoder=False, load_decoder=True`，见 `WanT2V480PConfig.__post_init__`）
     - `transformer` → `runtime/models/dits/wanvideo.py::WanTransformer3DModel`，会按 `param_names_mapping` 把 Diffusers 权重键重映射成 SGLang 的 `to_q / to_k / to_v / to_out / ffn.*`
     - `scheduler` → 先占位，下一步再替换
  3. `initialize_pipeline()`：Wan 特有逻辑只有一行——把 Diffusers 默认 scheduler 换成阿里官方的 UniPC：
     ```python
     def initialize_pipeline(self, server_args: ServerArgs):
         # We use UniPCMScheduler from Wan2.1 official repo, not the one in diffusers.
         self.modules["scheduler"] = FlowUniPCMultistepScheduler(
             shift=server_args.pipeline_config.flow_shift
         )
     ```
  4. `create_pipeline_stages(server_args)` → `add_standard_t2i_stages()`，把 7 个 Stage 实例挂到 executor。

### 1.5 warmup

`Scheduler.prepare_server_warmup_reqs()` 会合成一个最小请求（T2V：空 prompt、最小分辨率、`num_inference_steps=1`）塞进队列，跑一遍保证所有 CUDA graph / FlashAttention / cuBLAS workspace、layerwise-offload 缓冲区都预分配。

---

## 2. 用户请求 → `Req` 对象

文件：`runtime/entrypoints/utils.py::prepare_request`

`DiffGenerator.generate(sampling_params_kwargs={...})` 把 prompt、`height/width/num_frames/guidance_scale/seed/enable_teacache/...` 包成 `SamplingParams`，再 `prepare_request` 包成 `Req`：

- 自动 `adjust_num_frames`（Wan I2V：`(num_frames-1) % vae_temporal_scale == 0`）。
- 从 `pipeline_config.task_type` 校验输入合法性（T2V 不允许传 image_path 等）。
- `Req` 是后续所有 Stage 共享的"状态载体"，每个 Stage 只增量读写字段（`prompt_embeds / latents / timesteps / image_embeds / raw_latent_shape / trajectory_latents ...`）。

`SyncSchedulerClient.forward(req)` → rank 0 的 `Scheduler._handle_generation(req)` → 透传给 `GPUWorker.execute_forward(req)`。

---

## 3. `GPUWorker.execute_forward → Pipeline.forward`

`execute_forward` 做 5 件事：

1. 重置峰值显存统计；
2. 记录性能/内存基线；
3. `req.log()` 打印本次请求信息；
4. `self.pipeline.forward(req, server_args)`；
5. 将 `Req` 或结果包装为 `OutputBatch`。

`pipeline.forward` 把 `req` 喂给 `ParallelExecutor.execute(stages, req)`，按声明的 `parallelism_type` 挨个跑 Stage，每个 Stage 自带 `verify_input / verify_output / profiling`，出错即抛，不会默默乱算。

---

## 4. Stage 1：`InputValidationStage`

文件：`runtime/pipelines_core/stages/input_validation.py`

- 生成 `torch.Generator(device).manual_seed(batch.seed)`。
- Wan I2V 特殊逻辑（`load_and_process_image`）：把 `input_reference` 读成 `PIL`，按 480P/720P 策略 resize、crop、转为 `[-1,1]` 的 tensor，准备后面 VAE encode。
- T2V 不走这段；主要作用是规范 `batch.height / batch.width / batch.num_frames / batch.generator`。

---

## 5. Stage 2：`TextEncodingStage`（UMT5-XXL）

文件：`runtime/pipelines_core/stages/text_encoding.py`

- Wan 的 `text_encoder_configs = (T5Config(),)`，只有一个编码器。
- `TextEncodingStage.forward` 对正/负 prompt 分别调 `encode_text`：
  1. `tokenizer.__call__` → `input_ids + attention_mask`，pad 到 `tokenizer_kwargs.max_length=226`（UMT5）。
  2. `text_encoder.forward(input_ids, attention_mask, output_hidden_states=True)`：这个 encoder 就是 HuggingFace `T5EncoderModel` 的 SGLang 封装（`runtime/models/encoders/t5.py`），跑 encoder-only 的 24 层（XXL ≈ 11B）。
  3. `postprocess_func = t5_postprocess_text`：按 attention_mask 截断 variable-length hidden，再右 pad 到 `text_len=512`，对齐 DiT 的 cross-attn key length。
- 结果：
  - `batch.prompt_embeds = [Tensor(B, 512, 4096)]`
  - `batch.prompt_attention_mask = [Tensor(B, 226)]`
  - 若 `do_classifier_free_guidance=True`（T2V 默认开），对 `negative_prompt` 再跑一次。

> 这个 Stage 是 `StageParallelismType.REPLICATED` —— 每个 rank 都完整跑，但 `--text-encoder-cpu-offload` 打开时 T5 常驻 CPU，用到才 H2D，结束后立刻 `.to("cpu")`，见 `encoders/base.py`。

---

## 6. Stage 3：`LatentPreparationStage`

文件：`runtime/pipelines_core/stages/latent_preparation.py`

- `pipeline_config.prepare_latent_shape(batch, bsz, num_frames)` 算 shape，T2V 1.3B 480×832×81 为例：
  - VAE 空间 stride = 8 → `H/W = 480/8 = 60, 832/8 = 104`
  - VAE 时间 stride = 4 → `F' = (81-1)//4 + 1 = 21`
  - Wan DiT `in_channels = 16` → 初始 latent shape = `(1, 16, 21, 60, 104)`
- `randn_tensor(shape, generator=batch.generator, device=cuda, dtype=bf16)` 生成 `x_T` 纯噪声。
- `latents *= scheduler.init_noise_sigma`（UniPC 的 `init_noise_sigma = 1`，实际不变）。
- 写入 `batch.latents` 与 `batch.raw_latent_shape`。

---

## 7. Stage 4：`TimestepPreparationStage`

文件：`runtime/pipelines_core/stages/timestep_preparation.py`

调 `FlowUniPCMultistepScheduler.set_timesteps(num_inference_steps=50, device=cuda, shift=3.0)`：

```python
# scheduling_flow_unipc_multistep.py  (set_timesteps 主体)
sigmas = np.linspace(self.sigma_max, self.sigma_min, num_inference_steps + 1).copy()[:-1]
sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
timesteps = sigmas * self.config.num_train_timesteps  # 1000
sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)
self.sigmas = torch.from_numpy(sigmas).to(device=device)
self.timesteps = torch.from_numpy(timesteps).to(device=device, dtype=torch.int64)
```

- 核心事实：Wan 的 "时间步" 本质是 flow-matching 的 **sigma 序列**，不是 DDPM 的纯整数 timestep；`shift=3.0/5.0/8.0` 的效果是把 sigma 压向高噪声端，让采样步更关注低信息区域。
- 产出：`batch.timesteps` 形如 `[1000, 991, 980, ..., 15]` 的 int64 Tensor（长 50）。

---

## 8. Stage 5：`DenoisingStage`（热路径，90%+ GPU 时间都在这）

文件：`runtime/pipelines_core/stages/denoising.py`

### 8.1 循环骨架

```python
with torch.autocast(device_type=..., dtype=bf16, enabled=True):
    with self.progress_bar(total=ctx.num_inference_steps) as progress_bar:
        for step_index, t_host in enumerate(timesteps_cpu):
            step = self._prepare_step_state(ctx, batch, ..., step_index, t_host, ...)
            self._run_denoising_step(ctx, step, batch, server_args)
```

`_run_denoising_step` 的 6 步：

```python
# 1. Prepare latent inputs in the model's compute dtype.
latent_model_input = ctx.latents.to(ctx.target_dtype)  # bf16
# 2. Expand the timestep to the shape expected by the current model.
timestep = self.expand_timestep_before_forward(...)
# 3. Apply scheduler-side input scaling before the model forward.
latent_model_input = self.scheduler.scale_model_input(latent_model_input, step.t_device)
# 4. Run the model prediction path, including CFG when enabled.
noise_pred = self._predict_noise_with_cfg(...)
# 5. Advance the scheduler state with the predicted noise.
ctx.latents = self.scheduler.step(
    model_output=noise_pred, timestep=step.t_device,
    sample=ctx.latents, ..., return_dict=False,
)[0]
# 6. Re-apply any model-specific latent constraints after the update.
ctx.latents = self.post_forward_for_ti2v_task(...)
```

### 8.2 CFG（classifier-free guidance）

`_predict_noise_with_cfg`：

- **串行 CFG**：先用 `batch.prompt_embeds` 跑一次 `transformer.forward` → `noise_pred_cond`；再用 `negative_prompt_embeds` 跑一次 → `noise_pred_uncond`；最后 `noise_pred = uncond + cfg * (cond - uncond)`。
- **CFG 并行**（`--enable-cfg-parallel`）：两路拆到不同 GPU，最后 `cfg_model_parallel_all_reduce` 合并，效率 ~2×。
- Wan 不使用 `cfg_normalization / guidance_rescale`，默认为 0。

### 8.3 `WanTransformer3DModel.forward` 每一层细节

文件：`runtime/models/dits/wanvideo.py::WanTransformer3DModel.forward`

输入（T2V 1.3B，单步）：

- `hidden_states`：`(1, 16, 21, 60, 104)` bf16
- `encoder_hidden_states`：`(1, 512, 4096)` bf16（T5 后处理）
- `timestep`：`(1,)` int64
- `encoder_hidden_states_image`：`None`（T2V）

**Step A — 3D Patch Embed**

```python
hidden_states = self.patch_embedding(hidden_states)  # nn.Conv3d(16 → 1536, kernel=(1,2,2), stride=(1,2,2))
hidden_states = hidden_states.flatten(2).transpose(1, 2)
# → (B, T'*H'*W', inner_dim) = (1, 21*30*52, 1536) = (1, 32760, 1536)
```

`runtime/layers/visual_embedding.py::PatchEmbed`，本质是一次 Conv3d 触发 cuDNN 的 `implicit_gemm`。

**Step B — 序列并行切分（若 `sp_size > 1`）**

```python
sp_rank = get_sp_group().rank_in_group
local_seq_len = hidden_states.shape[1] // self.sp_size
hidden_states = hidden_states[:, sp_rank, :, :]  # 每个 rank 持有 1/sp_size 的 token
freqs_cos, freqs_sin = self._compute_rope_for_sequence_shard(local_seq_len, sp_rank, ...)
```

同时 3D 位置编码按本 rank 的 token 区间重新生成 cos/sin。

**Step C — Timestep / Text 条件嵌入**（`WanTimeTextImageEmbedding`）

- `TimestepEmbedder(timestep)`：对 1000 个离散时间点做 sinusoidal embedding，再过两层 Linear+SiLU，得到 `temb (B, 1536)`。
- `time_modulation = ModulateProjection(dim, factor=6)`：产出 `timestep_proj (B, 6, 1536)`，每块用于 (shift/scale/gate) × 2（pre-attn 和 pre-ffn）。
- `text_embedder`：两层 MLP + gelu，把 T5 的 4096 → 1536。

**Step D — 30 个 `WanTransformerBlock` 串行**

每块核心在 `WanTransformerBlock.forward`。逐个 sub-op：

1. **LayerNormScaleShift**（融合 op，`runtime/layers/layernorm.py`）：一次 kernel 做 `LN(x) * (1+scale) + shift`，比 PyTorch 原生快 2–3×。
2. **QKV 投影**：`ColumnParallelLinear`（`runtime/layers/linear.py`），TP > 1 时 weight 按 `num_heads/tp_size` 切列，底层是 `torch.nn.functional.linear` → cuBLAS `gemm bf16`。
3. **RMSNorm**（`runtime/layers/layernorm.py::RMSNorm`）对 Q/K 各做一次；TP>1 且 `qk_norm == "rms_norm_across_heads"` 走 `tensor_parallel_rms_norm`，一次 `all_reduce(sum(x**2))` 统一跨卡 mean-square。
4. **3D RoPE**：
   ```python
   if _is_cuda and query.shape == key.shape:
       cos_sin_cache = torch.cat([cos, sin], dim=-1)
       query, key = apply_flashinfer_rope_qk_inplace(query, key, cos_sin_cache, is_neox=False)
   ```
   CUDA 下用 FlashInfer fused kernel 同时对 Q/K 做 inplace RoPE，省 6 次 elementwise。回退路径：AMD 走 `aiter.rope_cached_2c_fwd_inplace`，纯 PyTorch 走 `_apply_rotary_emb`。
5. **Self-Attention**：`USPAttention`（`runtime/layers/attention/layer.py`）。
   - TP=1 / SP=1：直接 `self.attn_impl.forward(q, k, v, ctx_attn_metadata)`；`attn_impl` 由 `get_attn_backend(head_size, dtype)` 选出，默认 FlashAttention2（H20/H100 下走 FA3，AMD 下走 `aiter` 或 SDPA）。
   - **SP > 1 完整路径**：
     ```python
     if sp_size > 1:
         q = _usp_input_all_to_all(q, head_dim=2)   # [B,S_local,H,D] → [B,S,H_local,D]
         k = _usp_input_all_to_all(k, head_dim=2)
         v = _usp_input_all_to_all(v, head_dim=2)
     if get_ring_parallel_world_size() > 1:
         out = ring_attn(q, k, v, attn_impl=self.attn_impl, is_causal=False, dropout_p=0.0)
     else:
         out = self.attn_impl.forward(q, k, v, ctx_attn_metadata)
     if sp_size > 1:
         out = _usp_output_all_to_all(out, head_dim=2)
     ```
     - 第一次 all-to-all：每 rank 持有全序列，但只算 `H / sp_size` 个头（Ulysses head-sharding）。
     - `ring_degree > 1`：同一个 head subgroup 内进一步用 Ring Attention 循环传递 K/V 块，通信-计算重叠。
     - `ring_degree == 1`：直接 FlashAttention2（`flash_attn_func`），对 `(B=1, S=32760, H=12, D=128)` 这种 shape，一次调用占整个 block 30–60% 时间。
     - 第二次 all-to-all：把 head 收回，序列重新切分。
   - **Cross-Attention** 用 `is_cross_attention=True → skip_sequence_parallel=True`：KV 是 T5 的 `(1,512,4096)`，每个 SP rank 都是完整拷贝，所以跳过所有 all-to-all，本地 Q-shard 对完整 KV 做 attention。
6. **RowParallelLinear**（`self.to_out`）：TP>1 时沿 input 维切行，输出后 `all_reduce(SUM)` 合回完整张量。
7. **ScaleResidualLayerNormScaleShift**：融合 kernel，一次搞定
   ```
   h_new = h + gate * attn_out
   out   = LN(h_new) * (1+scale) + shift
   ```
   返回 `(normed, h_new)`，下一步 cross-attn 用 `normed`，残差用 `h_new`。
8. **Cross-Attention**（`WanT2VCrossAttention.forward`）：Q 来自视觉 token（本 rank 的 shard），K/V 来自 T5 的 `(1,512,4096)`（skip_sequence_parallel）。同样是 `ColumnParallelLinear + RMSNorm + attn + RowParallelLinear + allreduce`，但没有 RoPE。
9. **FFN**（`runtime/layers/mlp.py::MLP`）：`fc1 (1536→8960) → GELU_tanh → fc2 (8960→1536)`，TP 友好，all_reduce 只在 fc2 后一次。
10. **MulAdd**（融合 `gate * ffn_out + residual`，`runtime/layers/elementwise.py`）。

30 个 block 过完之后：

**Step E — 输出 norm + 线性 + unpatchify**

```python
if sequence_shard_enabled:
    hidden_states = sequence_model_parallel_all_gather(hidden_states, dim=1)  # 收回 SP 切片
hidden_states = self.norm_out(hidden_states, shift, scale)   # 融合 LN
hidden_states, _ = self.proj_out(hidden_states)              # linear 1536 → 16*(1*2*2) = 64
hidden_states = hidden_states.reshape(B, F', H', W', 1, 2, 2, -1).permute(...).flatten(...)
# → (1, 16, 21, 60, 104) == 输入 latent shape，即模型预测的 "flow"
```

### 8.4 TeaCache / CacheDiT（可选跳步）

- **TeaCache**：`should_skip_forward_for_cached_states` 用 `previous_residual` 和当前 `timestep_proj` 做相似度判断，相似就复用上一步残差，跳过 30 个 block。
- **Cache-DiT**：`_maybe_enable_cache_dit` 挂钩 `enable_cache_on_transformer`，对 block 级残差做 TaylorSeer 预测 + SCM mask 策略，可与 TP/SP 共存。

### 8.5 Scheduler.step（UniPC multistep）

拿到 `noise_pred` 后进 `FlowUniPCMultistepScheduler.step`：

```python
sample = sample.to(model_output.device)
model_output_convert = self.convert_model_output(model_output, sample=sample)
if use_corrector:
    sample = self.multistep_uni_c_bh_update(
        this_model_output=model_output_convert,
        last_sample=self.last_sample,
        this_sample=sample,
        order=self.this_order,
    )
self.model_outputs[-1] = model_output_convert
prev_sample = self.multistep_uni_p_bh_update(
    model_output=model_output, sample=sample, order=self.this_order,
)
```

- `convert_model_output`：flow-matching → `x0_pred = x_t - σ_t * v_pred`（`predict_x0=True`）。
- `multistep_uni_c_bh_update`：用上一步的 `last_sample` 做 corrector（UniC-p）。
- `multistep_uni_p_bh_update`：用当前点 + 历史 `model_outputs` 解出 `rhos`，再解析式构造 `x_{t-1}`（UniP-p，order=2 时 Cramer 规则显式求解，避免 CPU sync）。
- `self._step_index += 1`，返回 `(prev_sample,)`。

50 步之后 `ctx.latents` 就是最终的干净 latent，形状仍是 `(1, 16, 21, 60, 104)`。

---

## 9. Stage 6：`DecodingStage`（VAE 解码）

文件：`runtime/pipelines_core/stages/decoding.py`、`runtime/models/vaes/wanvae.py`

### 9.1 `scale_and_shift`

```python
scaling_factor, shift_factor = pipeline_config.get_decode_scale_and_shift(...)
# Wan: scaling_factor = 1 / latents_std, shift_factor = latents_mean
latents = latents / scaling_factor + shift_factor
```

`latents_mean / latents_std` 是每通道 16 个数，硬编码在 `WanVAEArchConfig`。贴错维度会直接得到纯噪声，是 Wan 接入最高频的坑。

### 9.2 `AutoencoderKLWan.decode`

Wan 默认走 `use_feature_cache=True`，一帧一帧解码：

```python
def decode(self, z: torch.Tensor) -> torch.Tensor:
    if self.use_feature_cache:
        self.clear_cache()
        iter_ = z.shape[2]
        x = self.post_quant_conv(z)
        with forward_context(feat_cache_arg=self._feat_map, feat_idx_arg=self._conv_idx):
            for i in range(iter_):
                feat_idx.set(0)
                if i == 0:
                    first_chunk.set(True)
                    out = self.decoder(x[:, :, i : i + 1, :, :])
                else:
                    first_chunk.set(False)
                    out_ = self.decoder(x[:, :, i : i + 1, :, :])
                    out = torch.cat([out, out_], 2)
```

`WanDecoder3d.forward` 是一个典型的 3D VAE decoder：

- `conv_in` (CausalConv3d, z_dim=16 → 512)
- `mid_block`：`(ResBlock, Attn, ResBlock)`
- 4 个 `UpBlock`：若干 `WanResidualBlock`（RMSNorm + Conv3d + SiLU + Conv3d + 残差）和 `WanResample`（上采样 + Conv3d）
- `conv_out` (CausalConv3d → 3)

关键底层算子：

- `WanCausalConv3d`（`runtime/models/vaes/parallel/wan_common_utils.py`）：`nn.Conv3d` 外包 padding + feature-cache（用 `contextvars` 保存上一个 chunk 的最后 2 帧，作为下一 chunk 的 "history"）。cuDNN 选 `implicit_precomp_gemm` / `winograd`。
- `WanRMS_norm`：`F.rms_norm`（PyTorch ≥ 2.4）。
- `get_act_fn("silu")` = `F.silu`。
- `WanAttentionBlock`：空间 1×1 attention（`nn.Conv2d` 算 QKV，做一次 `scaled_dot_product`），只出现在 `mid_block`，和 DiT 里的 attention 无关。
- `temporal_compression_ratio=4`：21 帧 latent → 81 帧 RGB（causal 解码，`start_frame_idx = 3` 被丢弃）。

输出 `x ∈ [-1, 1]`，最后 `DecodingStage.decode` 做 `image = (x/2+0.5).clamp(0,1)`，回到 `[0, 1]`。

### 9.3 OutputBatch 构造

```python
output_batch = OutputBatch(
    output=frames,
    trajectory_timesteps=batch.trajectory_timesteps,
    ...
)
if not getattr(batch, "is_warmup", False):
    self.offload_model()  # vae_cpu_offload 时 VAE.to("cpu")
return output_batch
```

---

## 10. 输出与后处理

`GPUWorker.execute_forward` 从 Pipeline 拿到 `OutputBatch`：

1. 若 `save_output=True`：用 `torchvision.io.write_video` / `imageio` 写 mp4，文件名按 `output_file_name` / 时间戳生成。
2. 可选 **帧插值**（`--enable-frame-interpolation`）：加载 `RIFE-4.22`（`runtime/pipelines_core/stages/upsampling.py`）对每两帧插 1–3 帧。
3. 可选 **超分**（`--enable-upscaling`）：Real-ESRGAN。
4. `OutputBatch` 通过 ZMQ 送回主进程 → `DiffGenerator` → `GenerationResult(output_path=..., metrics={...})`。

HTTP 模式下，同一个 `OutputBatch` 由 `runtime/entrypoints/openai/video_api.py` 生成视频 id，落盘，`GET /v1/videos/{id}/content` 时流式返回。

---

## 11. Wan2.1 独有的几个"非通用"热点

1. **`flow_shift` 注入**：`WanPipeline.initialize_pipeline` 在 scheduler 实例化时显式把 `WanT2V480PConfig.flow_shift=3.0` 传进来；14B 是 5.0、Turbo 是 8.0。
2. **T5 `text_len = 512` 的硬 pad**：由 `configs/pipeline_configs/wan.py::t5_postprocess_text` 统一保证，上游 tokenizer `max_length=226`、下游 DiT `text_len=512`，两者靠这个后处理粘合。
3. **3D RoPE 专用实现**：`NDRotaryEmbedding` 按 `(T', H', W')` 三维生成位置（`rope_dim_list = [d-4·d/6, 2·d/6, 2·d/6]`），SP 切分后每 rank 用 `_compute_rope_for_sequence_shard` 只算自己那段，`@lru_cache(1)` 缓存——Wan 能跑 SP 的关键之一。
4. **VAE feature-cache**：`use_feature_cache=True` 让 VAE 以 1 帧 latent / 4 帧 RGB 的粒度逐块解码，`contextvars` 保存 `CACHE_T=2` 帧"拖尾"，14B 720P 81 帧显存差异可达 15 GiB。
5. **layerwise offload**（`OffloadableDiTMixin`，`runtime/utils/layerwise_offload.py`）：DiT 按层滚动 H2D 队列，`dit_layerwise_offload_prefetch` 控制重叠度，Wan 14B 单卡 24G 主要靠这个。
6. **TeaCache 系数**：`_wan_1_3b_coefficients / _wan_14b_coefficients` 写死在 `configs/sample/wan.py`，对 `timestep_proj` 的 L1 差作多项式拟合，决定该步能否跳。
7. **Wan2.2 boundary expert 切换**（`_handle_boundary_ratio + _select_and_manage_model`）：high-noise 阶段用 `transformer`、low-noise 阶段用 `transformer_2`，并同时做显存腾挪。1.3B T2V 不走这路。

---

## 12. "最底层算子"速查（profiler 里要找的 kernel）

| 阶段 | 上层调用 | 最终 GPU kernel |
| --- | --- | --- |
| Patch embed | `PatchEmbed` → `nn.Conv3d` | cuDNN `implicit_gemm` / `winograd` |
| QKV / FFN / proj_out | `ColumnParallelLinear` / `RowParallelLinear` | cuBLAS `ampere_bf16_gemm` |
| RMSNorm | `RMSNorm` | `F.rms_norm` fused CUDA kernel |
| LayerNorm(shift/scale) | `LayerNormScaleShift` | sglang 自家融合 kernel（`runtime/layers/layernorm.py`） |
| RoPE | `apply_flashinfer_rope_qk_inplace` | FlashInfer `apply_rope_pos_ids_cos_sin_cache` |
| Self/Cross Attention | `USPAttention → attn_impl.forward` | FlashAttention2/3 `flash_attn_func` |
| SP all-to-all | `_usp_input/output_all_to_all` | NCCL `ncclAllToAll` |
| TP all-reduce | `RowParallelLinear` 末尾 | NCCL `ncclAllReduce(SUM)` |
| Residual+Norm 融合 | `ScaleResidualLayerNormScaleShift` | 自定义 fused kernel |
| MulAdd | `MulAdd` | 单次 elementwise |
| Scheduler step | `multistep_uni_p/c_bh_update` | cuBLAS gemv + torch elementwise |
| VAE causal conv | `WanCausalConv3d` | cuDNN Conv3d |
| VAE mid attn | `attention_block_forward` | `F.scaled_dot_product_attention`（内部挑 FA） |

---

## 13. 跟读建议

想在源码里一路跟读，最实用的是下面这三个断点，覆盖了 Wan2.1 运行时 95% 的算力落点：

1. `_run_denoising_step`（`runtime/pipelines_core/stages/denoising.py`）：看一次单步输入/输出。
2. `WanTransformerBlock.forward`（`runtime/models/dits/wanvideo.py`）：看一个 block 的各子 op。
3. `USPAttention.forward`（`runtime/layers/attention/layer.py`）：看 SP/Ring 通信与 FlashAttention 的关系。

---

## 14. 关联文档

- 组件与执行框架：[03_runtime_execution.md](./03_runtime_execution.md)、[04_pipeline_and_stage.md](./04_pipeline_and_stage.md)
- 注册表与 PipelineConfig：[02_registry_and_config.md](./02_registry_and_config.md)
- Loader 与权重重映射：[05_loader_and_models.md](./05_loader_and_models.md)
- 三段式拆服务与优化：[07_disaggregation_and_optimization.md](./07_disaggregation_and_optimization.md)
- 新增 Wan 变体的 9 步实战：[09_case_study_wan2_1.md](./09_case_study_wan2_1.md)
- 用户使用手册：[wan2_1_guide.md](./wan2_1_guide.md)
