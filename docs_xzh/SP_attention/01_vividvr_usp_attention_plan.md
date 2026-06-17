# VividVR SP Ulysses Attention 等价改造方案

更新时间：`2026-06-17 UTC`

## 1. 背景与问题

### 1.1 当前 VividVR SP 的问题

当前 VividVR 双卡 SP（Sequence Parallelism）使用**朴素 shard-only 策略**：

```
shard_video_tokens → [30 层 CogVideoXBlock on local tokens] → gather_video_tokens
                       ↑ 无通信，self-attention 被截断
```

这导致 **self-attention 数学上不等价于单卡**。GPU 0 的 token 看不到 GPU 1 的 token，跨 GPU 边界的 attention 完全丢失。

证据：双卡 SP 所有结果 SSIM 收敛到 0.962-0.965，与 `native_sp_only`（纯局部控制，无全局补偿）一致。

### 1.2 参考方案：WAN 视频编辑的 USPAttention

WAN 视频编辑模型使用 `USPAttention`（Ulysses Sequence Parallel Attention），通过 **all-to-all 通信** 实现数学等价的分布式 attention：

```
SP 分片后: 每个 GPU 有 [B, S/2, H, D]  (一半 sequence, 全部 heads)

↓ all-to-all: scatter dim=head, gather dim=seq

           每个 GPU 有 [B, S, H/2, D]  (完整 sequence, 一半 heads)

↓ 各自做 attention（完整序列 × 一半 heads）

↓ all-to-all (逆操作)

           每个 GPU 有 [B, S/2, H, D]  (恢复原始分片)
```

**数学上完全等价于单卡 attention。** 通信代价：每层 2 次 all-to-all。

### 1.3 CogVideoX Joint Attention 的特殊性

CogVideoXBlock 做的是 **joint attention**——text tokens 和 video tokens 拼接后一起做 self-attention：

```python
# diffusers CogVideoXAttnProcessor2_0.__call__
hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
# [B, 226, d] + [B, 13500, d] → [B, 13726, d]
```

在当前朴素 SP 下，**text tokens 也无法 attend 到其他 GPU 的 video tokens**，进一步加剧了质量退化。

这恰好是 `USPAttention._forward_with_replicated_prefix` 设计要处理的场景：
- **Replicated prefix**: text tokens (226)，各 GPU 完全相同
- **Sharded suffix**: video tokens (13500 per GPU)，SP 分片

## 2. 修改目标

1. 让 CogVideoXBlock 的 joint self-attention 在 SP 模式下**数学等价于单卡**
2. 让 ControlNet 的 joint self-attention 同样等价
3. 保证单卡模式和关闭 SP 时的行为不变（向后兼容）
4. Connector 可后续简化（当前 v2 的 all-gather global control 可能不再必需）

## 3. 关键代码位置

| 文件 | 作用 |
|------|------|
| `cogvideox_attention_backend.py` | Attention processor 定义、backend 注册 |
| `cogvideox_vividvr.py` | Transformer (30 层) forward |
| `cogvideox_vividvr_controlnet.py` | ControlNet (6 层) forward |
| `cogvideox_vividvr_common.py` | SP shard/gather、Connector |
| `vividvr.py` (denoising stage) | 设置 forward context、attention backend |
| `layer.py` | `USPAttention` 定义 |
| `usp.py` | Ulysses all-to-all 通信实现 |

## 4. 详细修改步骤

### 4.1 Step 1：新增 `CogVideoXSPFlashAttnProcessor`

**文件**：`python/sglang/multimodal_gen/runtime/models/dits/cogvideox_attention_backend.py`

新增一个 SP 感知的 attention processor，在 SP 启用时使用 `USPAttention` + replicated prefix：

```python
class CogVideoXSPFlashAttnProcessor:
    """SP-aware CogVideoX joint-attention processor using Ulysses all-to-all.

    当 SP 启用时（sp_world_size > 1），将 text tokens 视为 replicated prefix，
    video tokens 视为 SP-sharded suffix，通过 USPAttention 实现数学等价的
    分布式 joint attention。

    当 SP 未启用时，退化为标准 FlashAttention 路径。
    """

    _attention_backend = "fa_sp"

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 1) 检查是否启用 SP
        sp_size = get_sp_world_size() if model_parallel_is_initialized() else 1
        if sp_size <= 1:
            # 单卡：退化为标准 FA processor
            return CogVideoXFlashAttnProcessor()(
                attn=attn,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                image_rotary_emb=image_rotary_emb,
            )

        # 2) SP 路径：text = replicated prefix, video = sharded suffix
        text_seq_length = encoder_hidden_states.size(1)
        batch_size = hidden_states.shape[0]

        # 复用 _prepare_cogvideox_qkv 做 projection + norm + RoPE
        # 它内部 concat text+video, 投影 QKV, 做 norm/RoPE
        _, query, key, value = _prepare_cogvideox_qkv(
            attn=attn,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )
        # query/key/value: [B, H, S_total, D] where S_total = text_seq + video_seq

        head_dim = query.shape[-1]
        num_heads = query.shape[1]

        # 3) 获取或创建 USPAttention 实例（带 replicated prefix 支持）
        usp_attn = _get_cogvideox_sp_usp_attention(
            num_heads=num_heads,
            head_size=head_dim,
        )

        # 4) USPAttention.forward with replicated_prefix=text_seq_length
        # 输入布局: [B, S_local, H, D]，text 在前面
        out = usp_attn(
            query.transpose(1, 2).contiguous(),   # [B, S_local, H, D]
            key.transpose(1, 2).contiguous(),
            value.transpose(1, 2).contiguous(),
            num_replicated_prefix=text_seq_length,
        )
        # 输出: [B, S_local, H, D]

        # 5) Output projection + split back
        out = out.reshape(batch_size, -1, num_heads * head_dim)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)

        encoder_hidden_states, hidden_states = out.split(
            [text_seq_length, out.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states
```

### 4.2 Step 2：新增 USPAttention 缓存工厂函数

**文件**：`python/sglang/multimodal_gen/runtime/models/dits/cogvideox_attention_backend.py`

```python
@lru_cache(maxsize=4)
def _get_cogvideox_sp_usp_attention(
    *,
    num_heads: int,
    head_size: int,
) -> USPAttention:
    """创建用于 CogVideoX SP joint attention 的 USPAttention 实例。

    使用 skip_sequence_parallel=False（默认），让 USPAttention
    执行完整的 Ulysses all-to-all 通信管线。
    """
    return USPAttention(
        num_heads=num_heads,
        head_size=head_size,
        softmax_scale=None,
        causal=False,
        supported_attention_backends={
            AttentionBackendEnum.FA,
            AttentionBackendEnum.FA2,
        },
        prefix=f"cogvideox_sp_attn_{num_heads}_{head_size}",
    )
```

### 4.3 Step 3：修改 `build_cogvideox_attention_processor` 和 `set_cogvideox_attention_backend`

**文件**：`python/sglang/multimodal_gen/runtime/models/dits/cogvideox_attention_backend.py`

```python
# 修改 build_cogvideox_attention_processor，新增 fa_sp backend
def build_cogvideox_attention_processor(backend: str) -> object:
    normalized_backend = normalize_cogvideox_attention_backend(backend)
    if normalized_backend == "native":
        return CogVideoXNativeAttnProcessor()
    if normalized_backend == "fa":
        return CogVideoXFlashAttnProcessor()
    if normalized_backend == "fa_sp":
        return CogVideoXSPFlashAttnProcessor()
    raise ValueError(...)

# 修改 normalize_cogvideox_attention_backend，新增 alias
alias_map = {
    ...
    "fa_sp": "fa_sp",
    "sp_fa": "fa_sp",
    "usp": "fa_sp",
}
```

### 4.4 Step 4：修改 Denoising Stage 自动选择 SP attention backend

**文件**：`python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py`

在 `VividVRDenoisingStage.prepare_denoising_state()` 中，当 SP 启用时自动选择 `fa_sp` backend：

```python
# 在 prepare_denoising_state 中，当前已有 sequence_shard_enabled 变量
# 修改 attention backend 选择逻辑：

attn_backend_str = server_args.pipeline_config.dit_attention_backend or "fa"
if sequence_shard_enabled and sp_group is not None and int(sp_group.world_size) > 1:
    attn_backend_str = "fa_sp"  # SP 模式强制使用 SP 感知 processor

# 然后调用 set_cogvideox_attention_backend
self.transformer.set_attention_backend(attn_backend_str)
self.controlnet.set_attention_backend(attn_backend_str)
```

### 4.5 Step 5：确保 SP shard/gather 与 USP 协调

**当前状态**：`shard_vividvr_video_tokens` 已经正确处理了：
- ✅ 只 shard video tokens，text tokens 保持 replicated
- ✅ RoPE 正确按 rank 切片
- ✅ Transformer 入口的分离：`encoder_hidden_states` (text) 与 `hidden_states` (video)

**需要确认**：shard 后 video tokens 的 seq_len = 13500，text tokens seq_len = 226。`USPAttention._forward_with_replicated_prefix` 期望：
- `q/k/v` 维度为 `[B, S_local, H, D]`，其中 `S_local = text(226) + video(13500) = 13726`
- `num_replicated_prefix = 226`（text tokens 数）

这与当前 CogVideoXBlock 传入 attention processor 的数据流完全吻合——无需修改 shard/gather 逻辑。

### 4.6 Step 6：Connector 简化（可选，后续迭代）

USP 等价 attention 部署后，可进行以下评估：

1. **对比实验**：`fa_sp` + Connector v2 vs `fa_sp` + Connector v1 vs `fa_sp` + Connector native_sp_only
2. 如果 USP 已保证 self-attention 等价，Connector v2 的 all-gather global control 可能冗余
3. 可以降级为 Connector v1 或 native_sp_only 以节省通信

**本方案不包含此 step**，作为后续迭代方向。

## 5. 通信与性能评估

### 5.1 通信量分析

以双卡 A100 (B=1, H=30, S_video=13500, S_text=226, D=64) 为例：

**单层通信量**：
- All-to-All input: `B × S_local × H × D × fp16 = 1 × 13726 × 30 × 64 × 2 = 52.7 MB`
- All-to-All output: 同上，52.7 MB
- 每层总计: ~105 MB

**全模型通信量**：
- 每层 2 次 all-to-all（input + output），30 层 Transformer + 6 层 ControlNet
- 36 层 × 105 MB = ~3.8 GB 总通信量
- A100 NVLink 600 GB/s → ~6.3 ms 纯通信时间

### 5.2 与其他方案对比

| 方案 | Self-Attention 等价 | 通信量/层 | 预期质量 |
|------|-------------------|-----------|----------|
| 当前 VividVR (shard-only) | ❌ | 0 | SSIM ~0.964 |
| Ulysses USP (本方案) | ✅ | 2×All-to-All | SSIM ~0.984 (预期) |
| Ring Attention | ✅ | 环形传递 K/V | SSIM ~0.984 |
| Ulysses + Ring | ✅ | 混合 | SSIM ~0.984 |

### 5.3 通信与计算重叠机会

`USPAttention` 内部使用 `all_to_all_single`，在 PyTorch 2.6+ 支持 AsyncCollectiveTensor，通信可与后续计算重叠：

```python
# usp.py - all-to-all 使用 pytorch functional collectives
x = ft_c.all_to_all_single(x, ...)  # 返回 AsyncCollectiveTensor
x = _maybe_wait(x)                   # 仅在需要数据时等待
```

后续优化可考虑在 all-to-all 等待之前插入 FFN 或其他无关计算。

## 6. 测试验证计划

### 6.1 单卡等价验证

```
# 单卡 fa_sp backend 应退化为 fa backend
sglang serve --model-path ... --num-gpus 1
# 验证: SSIM = 0.9845（与当前单卡一致）
```

### 6.2 双卡 SP 等价验证

```
# 双卡 SP + fa_sp backend
sglang serve --model-path ... --num-gpus 2
# 验证: SSIM 应恢复到 0.9846（与 6/11 GOOD baseline 一致）
```

### 6.3 数值等价验证（第一 timestep）

```
# 单卡 fa vs 双卡 fa_sp: 第一层 attention output 误差应 < 1e-3
# 验证方式: dump Q/K/V/attention_out 对比 norm 差异
```

### 6.4 回归测试

| 测试场景 | 预期结果 |
|----------|----------|
| 单卡 fa_sp | SSIM = 0.9845（退化为 fa） |
| 双卡 fa_sp (v2) | SSIM ≈ 0.9846（恢复 GOOD） |
| 双卡 fa_sp (native_sp_only) | SSIM（待测，预计优于 0.9628） |
| 双卡 fa_sp + compile=False | 验证与 compile 无关 |
| 关闭 SP (sp_degree=1) | SSIM = 0.9845（不变） |

## 7. 文件修改清单

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| `cogvideox_attention_backend.py` | **主要修改** | 新增 `CogVideoXSPFlashAttnProcessor`、`_get_cogvideox_sp_usp_attention`、backend 注册 |
| `vividvr.py` (denoising stage) | **轻量修改** | SP 启用时自动选择 `fa_sp` backend |
| `cogvideox_vividvr_common.py` | **无需修改** | SP shard/gather 逻辑不变 |
| `cogvideox_vividvr.py` | **无需修改** | Transformer forward 不变 |
| `cogvideox_vividvr_controlnet.py` | **无需修改** | ControlNet forward 不变 |

## 8. 风险与注意事项

1. **USPAttention 兼容性**：USPAttention 要求 q/k/v dtype 为 fp16/bf16。CogVideoX 默认使用 fp16，满足条件。

2. **QK Norm + RoPE 融合**：当前 `_prepare_cogvideox_qkv` 支持 QK norm fusion 和 RoPE fusion。USP 模式下 QKV 已经过 norm 和 RoPE 处理，USPAttention 内部不再重复处理，不会冲突。

3. **Attention Scale**：CogVideoX 使用 `softmax_scale=None`（即 `head_size**-0.5`），USPAttention 默认相同，一致。

4. **All-to-All 与 Ring Attention 的组合**：当前 `ulysses_degree=2, ring_degree=1`（双卡时默认），仅使用 Ulysses，不使用 Ring。如果将来扩展到 >2 GPU 且启用 Ring，需要确保兼容。

5. **ControlNet 的 attention backend 设置**：当前 `set_cogvideox_attention_backend` 对 ControlNet 和 Transformer 都安装相同的 processor。Step 4 中对两者都设置 `fa_sp`。

6. **向后兼容**：`fa_sp` 是新增 backend，不改变现有 `fa`/`native` 行为。单卡时 `fa_sp` 自动退化为标准 FA 路径。

## 9. 参考代码路径

| 组件 | 文件路径 |
|------|----------|
| USPAttention | `python/sglang/multimodal_gen/runtime/layers/attention/layer.py` |
| Ulysses all-to-all | `python/sglang/multimodal_gen/runtime/layers/usp.py` |
| WAN 模型 (参考实现) | `python/sglang/multimodal_gen/runtime/models/dits/wanvideo.py` |
| CogVideoX attention backend | `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_attention_backend.py` |
| VividVR Transformer | `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr.py` |
| VividVR ControlNet | `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr_controlnet.py` |
| VividVR SP 工具函数 | `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr_common.py` |
| VividVR Denoising Stage | `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py` |
| SP 并行状态 | `python/sglang/multimodal_gen/runtime/distributed/parallel_state.py` |
