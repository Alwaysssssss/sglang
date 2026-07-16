# VividVR CogVideoX VAE 空间 Tile 并行设计

日期：2026-07-16
状态：实现与验收已完成（实验开关，默认关闭）

## 1. 目标

在严格保持当前 CogVideoX VAE tiled decode 数值语义、VividVR 长短视频编排和后处理语义不变的前提下，复用现有 sequence parallel（SP）进程组并行解码空间 tiles，降低 VAE `Decode/Trim` 耗时。

第一阶段只改变空间 tile 的执行位置，不改变 tile plan、单 tile temporal decode、overlap blend、crop、trim、stitch、AdaIN 或 reference color fix。

## 2. 已确认的约束

- 只实现 CogVideoX VAE 的空间 tile 并行，不修改其他 VAE。
- 复用 `get_sp_group()`，所有 collective 只允许发生在 SP subgroup 内，禁止使用 WORLD group。
- `SP=2` 时由两个 SP rank 分担 tiles；`SP=4` 时由四个 SP rank 分担 tiles。
- `CFG=2 × SP=2` 时，每个 SP subgroup 独立执行二路 tile 并行，禁止跨 CFG group 通信。
- 每个 rank 最终都获得完整 decoded clip，后续 replicated stages 保持不变。
- 不做 temporal parallel，不增加专用 VAE rank，不引入 leader-only 后处理。
- `vae_sp` 保持实验开关，本阶段不修改 Phase E 正式默认配置。
- 本设计不复用 `runtime/models/vaes/common.py` 中现有的通用并行 tiled VAE 路径；该路径的 tensor 布局、collective group 和 CogVideoX temporal cache 语义都不适合作为本次直接基础。

## 3. 方案选择

第一阶段采用“各 rank 解码部分 tiles，SP subgroup all-gather 全部结果，各 rank 按原顺序独立 merge”的方案。

没有选择 leader gather/merge/broadcast，原因是第一阶段优先降低语义风险：所有 rank 都复用同一套原始 row-major merge 规则，没有 leader 分支，也不会使后续 replicated stage 依赖广播结果。leader merge 可以在第一阶段完成并获得 profile 后作为后续优化讨论。

## 4. 代码边界

核心实现放在：

```text
python/sglang/multimodal_gen/runtime/models/vaes/cogvideox.py
```

保持继承调用结构：

```text
diffusers decode
  -> diffusers _decode
    -> CogVideoX tiled_decode override
       -> serial: super().tiled_decode(...)
       -> parallel: _parallel_spatial_tiled_decode(...)
```

推荐拆分以下内部职责：

```text
_build_spatial_tile_plan
_assign_tiles
_decode_one_spatial_tile
_all_gather_decoded_tiles
_merge_spatial_tiles
```

VividVR pipeline 在 VAE 初始化后显式配置 `vae_sp_requested`。decode stage 继续负责调用 VAE，并收集 VAE 返回的并行统计；不在 stage 内重新实现 tile 算法。

## 5. Tile plan 与任务分配

tile plan 必须完全沿用当前 diffusers `AutoencoderKLCogVideoX.tiled_decode` 的计算规则，包括：

- latent tile 高宽；
- overlap stride；
- sample-space blend extent；
- row limit；
- row-major 遍历顺序。

每个 tile 获得稳定的 `global_tile_index`。任务采用 round-robin 分配：

```text
owner_rank = global_tile_index % sp_world_size
```

这样能保证每个 tile 只被解码一次，并使各 rank 的 tile 数量差不超过 1。边界 tile 尺寸可以不同，不能假设所有 decoded tile payload 等长。

## 6. 单 Tile Decode 语义

每个空间 tile 内必须完整复用原始 temporal decode loop：

- temporal batch 大小不变；
- conv cache 在一个空间 tile 的 temporal batches 之间延续；
- 开始下一个空间 tile 时重新初始化 conv cache；
- scaling、dtype、device 和输出布局不变。

禁止把 temporal batches 分配给不同 rank，也禁止跨空间 tile 共享 conv cache。

## 7. 通信协议

采用固定 metadata 加变长 payload 的两阶段 tensor collective，不使用 `all_gather_object`。

### 7.1 输入一致性预检

进入本地 tile decode 前，在 SP subgroup 内 all-gather 固定长度的输入描述符，至少检查：

- latent shape；
- dtype；
- tile/overlap 参数；
- tile 总数；
- SP world size。

任一 rank 描述符不一致时，所有 rank 在本地 decode 和变长 collective 开始前一致失败。

### 7.2 Metadata

每个 rank 固定提供：

```text
slots_per_rank = ceil(total_tiles / sp_world_size)
```

每个 slot 的 GPU tensor metadata 至少包含：

```text
global_tile_index
batch
channels
frames
height
width
numel
```

空 slot 使用 `global_tile_index = -1`。metadata tensor 在所有 rank 上 shape 相同，因此可以直接 all-gather。

### 7.3 Payload

每个 rank 将本地 decoded tiles flatten 后连续拼接。metadata gather 完成后，所有 rank 都能计算最大的 rank-local payload 长度；本地 payload 补零到该长度，再执行第二次 all-gather。

接收方根据 source rank、slot metadata 和 rank-local offset 恢复每个 tile。恢复时优先使用 gathered buffer 的 view，避免无必要的完整二次复制。

该协议必须覆盖：

- tile 数少于 rank 数；
- tile 数不能整除 world size；
- 边界 tile shape 不同；
- 某个 rank 没有实际 tile。

## 8. Merge 语义

每个 rank 在获得完整 tile 集合后，按 `global_tile_index` 恢复原始 row-major 顺序，并严格执行当前 diffusers 逻辑：

1. 对当前 tile 执行 `blend_v`；
2. 再执行 `blend_h`；
3. 按原始 row limits crop；
4. 先拼接一行，再拼接所有行。

当前 blend 会原地修改当前 tile，因此不能交换 `blend_v` 和 `blend_h`，不能改变遍历顺序，也不能用数学上看似等价的向量化重写替代第一阶段实现。

## 9. 开关与 Fallback 契约

| 条件 | 行为 | 状态原因 |
| --- | --- | --- |
| `vae_sp=False` | 调用继承的串行路径 | `not_requested` |
| `vae_sp=True`，SP size 为 1 | 调用继承的串行路径 | `sp_world_size_one` |
| `vae_sp=True`，输入未触发 tiling | 调用原始非 tiled decode | `input_below_tiling_threshold` |
| `vae_sp=True`，SP size 大于 1 且触发 tiling | 空间 tile 并行 | `effective` |
| `vae_sp=True`，但 tiling 关闭 | 启动阶段配置错误 | 不允许静默降级 |
| `vae_sp=True`，但 SP group 未正确初始化 | 明确报错 | 禁止退化到其他 group |

正常 fallback 只允许发生在未请求、SP size 为 1 或输入未触发 tiling 三种情形。

某个 clip 一旦进入并行路径，不捕获执行异常后改跑串行。部分 rank 已经进入 collective 时切换路径会造成 collective 次序不一致或死锁。预检后发生的 CUDA OOM、NCCL 错误或 tile decode 错误直接作为 worker 失败向上抛出。

## 10. 可观测性

每次 VAE decode 至少记录：

```text
vae_sp_requested
vae_sp_effective
vae_sp_fallback_reason
vae_sp_world_size
vae_sp_group_type
vae_total_tiles
vae_local_tiles_per_rank
vae_tile_decode_seconds
vae_tile_gather_seconds
vae_tile_merge_seconds
vae_decode_seconds
```

长视频 decode stage 收集各 temporal clip 的记录，并对 tile 数和耗时求和，同时保留每个 clip 的 effective/fallback 状态。任一 clip 执行失败仍按整个请求失败处理。

内部阶段计时使用 CUDA events，在整个 decode 结束后统一同步并读取，避免为了统计在 decode、gather 和 merge 之间额外插入同步点。正式总耗时继续沿用现有 `total_runtime_seconds`、`model_inference_runtime_seconds` 和同步 Stage profiling 口径。

## 11. 测试设计

### 11.1 无真实推理的测试

- tile plan 与当前 diffusers 坐标、stride、blend extent 和 crop 一致；
- round-robin 分配覆盖 tile 少于 rank、等于 rank、不能整除 world size；
- metadata/payload 能恢复不同 shape 的边界 tile 和空 slot；
- 非对称输入下，parallel merge 与原始 `blend_v -> blend_h -> crop` 逐元素一致；
- `vae_sp=False`、SP1、不触发 tiling 均调用原始路径；
- tiling 关闭但请求 `vae_sp` 时明确失败；
- 输入描述符不一致时，各 rank 在预检阶段一致失败。

### 11.2 分布式与真实 VAE 对齐

- SP2 覆盖 tile 可整除和不可整除场景；
- `CFG=2 × SP=2` 使用两组不同标记输入，验证 subgroup 隔离和 collective 次序；
- SP4 只运行固定 latent 的轻量 distributed decode，验证四 rank 分配、collective 和完整结果恢复，不运行纯 SP4 的完整 `130f / 20 step` benchmark；
- 使用同一固定 latent 比较串行、SP2、SP4 的单 tile、merge 后 tensor 和后处理帧。

数值目标是 bitwise equal。先运行两次串行 decode 判断底层 kernel 自身是否确定：串行自比较完全一致时，并行也必须 `torch.equal`；如果串行本身存在不确定性，并行误差不得超过串行重复运行的误差包络，且必须记录最大误差和平均误差，不能直接放宽固定阈值。

## 12. 正式性能与质量验收

不重复运行单卡 Phase C、独立 Phase D/E 基线、双卡 SDPA 配置，也不重跑 VAE 并行关闭的 R99/R100。串行 fallback 与基础语义由单元测试和固定 latent 对齐测试保护。

只新增以下两个正式实验组：

| 实验组 | 唯一新增变量 | 历史对照 |
| --- | --- | --- |
| `R99 + vae_sp` | 在 R99 全量已实现加速配置上设置 `vae_sp=True` | R99 |
| `R100 + vae_sp` | 在 R100 全量已实现加速配置上设置 `vae_sp=True` | R100 |

R99 固定为双卡 `SP=2 + FA-SP + torch.compile + modulation/residual fusion`。R100 固定为四卡 `CFG=2 × SP=2 + FA-SP + torch.compile + modulation/residual fusion`。R100 验证的是两个隔离的 SP2 subgroup，不代表 SP4 性能实验。

历史对照值为：

| 指标 | R99 | R100 |
| --- | ---: | ---: |
| 总耗时 | 551.119 s | 370.881 s |
| 模型推理耗时 | 544.321 s | 365.067 s |
| Denoise | 380.176 s | 195.652 s |
| Decode/Trim | 100.274 s | 101.786 s |

新实验必须保持原 benchmark 的模型、输入、caption、seed、20 steps、FA-SP、compile warmup、fusion、服务生命周期和质量比较方式不变。除实现本身所需的指标字段外，唯一运行时变量是 `vae_sp=True`。

正式结果必须报告：

- Decode/Trim 及其相对历史控制组的增量加速比；
- tile decode、gather、merge；
- model inference、总耗时、GPU·秒；
- 各 rank 峰值显存；
- 现有视频质量指标与人工检查结果；
- effective topology、`vae_sp_requested/effective` 和 subgroup 信息。

通过条件是 Decode/Trim 获得可复现的正收益，model inference 和端到端总耗时不回归，质量与 VividVR 前后处理语义保持。若正确性通过但没有端到端收益，`vae_sp` 只能保留为实验性 opt-in，不能替换正式默认配置。

所有正式推理必须在 tmux 中执行，日志、指标 JSON 和结果视频分别保存到仓库规定的统一验收目录。

## 13. 主要风险与防护

| 风险 | 防护 |
| --- | --- |
| temporal conv cache 生命周期变化 | 单 tile 内复用完整 temporal loop，跨 tile 重置，并增加行为测试 |
| blend 顺序改变导致接缝 | 锁定 row-major、`blend_v -> blend_h -> crop`，使用非对称测试数据 |
| CFG group 间串数据 | 只使用 `get_sp_group()`，增加 `CFG=2 × SP=2` 标记隔离测试 |
| 边界 tile shape 不同 | 固定 metadata + padded variable payload，不假设等长 tensor |
| rank 在 collective 前分叉 | collective 前执行固定 descriptor 预检，固定 collective 次序 |
| gather staging 增加显存 | 恢复 tile 时优先使用 gathered buffer view，正式记录单 rank 峰值显存 |
| 开关看似打开但未生效 | 同时记录 requested、effective 和 fallback reason |

## 14. 后续优化边界

只有第一阶段正确性和正式 profile 完成后，才讨论：

- leader gather/merge 后 broadcast；
- 分批 gather 与流式 merge；
- 通信和下一个 tile decode 重叠；
- 更复杂的基于 tile 计算量的负载均衡。

这些优化都不属于第一阶段实现，不应提前增加主路径复杂度。

## 15. 实施与验收补记

2026-07-16 已按本设计完成实现、真实 NCCL 正确性验证和两条正式 FlowCut 服务 benchmark。核心实现提交为 `1ca30dd7b`、`14a1610c3`、`cf8be47ae`、`2e4cff479`、`da687d21d`、`6e6438b90`、`22dc14157`、`8f3e2ff9b`、`8701b6197`。

实施中确认 VividVR decode 边界上的 SP rank latent 可能不同，且实际 tensor 可能是非 contiguous view。因此按第 4.3 节既定防护，在并行 decode 前增加了 SP subgroup root latent broadcast，并显式生成 contiguous buffer 以满足 NCCL 要求。真实 non-contiguous 固定 latent 验证覆盖 SP2、SP4、CFG2×SP2，串行与并行输出均 bitwise equal。

正式 R99/R100 treatment 的 Decode/Trim 分别从 `100.274310 s` 降至 `58.937737 s`、从 `101.785656 s` 降至 `60.178877 s`；端到端 speedup 分别为 `1.078657×`、`1.087615×`。质量 JSON 保留原始硬门禁状态，微小差异依据用户明确决定做人工豁免，最终验收通过。

完整命令、指标、质量、服务证据与风险记录见：

- `docs_xzh/distribute/vividvr_vae_spatial_tile_parallel_acceptance_20260716.md`

`vae_sp` 继续保持默认关闭的实验性 opt-in；Phase E 正式默认配置、服务请求契约和 `AGENTS.md` 均未改变。
