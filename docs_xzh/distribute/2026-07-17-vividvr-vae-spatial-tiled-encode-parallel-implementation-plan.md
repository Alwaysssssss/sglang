# Vivid-VR VAE Spatial Tiled Encode 并行实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法跟踪进度。本任务明确禁止使用子代理，并且经用户确认直接在当前分支实施，不创建 worktree。

**目标：** 为 Vivid-VR 的 CogVideoX VAE tiled encode 增加独立的 SP-subgroup 空间 tile 并行，在保持串行 moments 与 sampled control latents 逐位一致的前提下，降低 `VividVRLongClipPreparationStage` 耗时，并完成 SP2、CFG2xSP2、SP4 三组 Treatment-only 正式验收。

**架构：** encode 与现有 decode 并行共享底层定长 tensor metadata/payload collective，但保留独立的 sample-space tile plan、本地 encoder worker、merge 和统计类型。每个 SP subgroup 按 `global_tile_index % sp_world_size` 分配 tiles，subgroup 内 gather 完整 posterior moments tiles 后，每个 rank 按 Diffusers 原 row-major 顺序复制执行 `blend_v -> blend_h -> crop -> concat`，随后由原 `DiagonalGaussianDistribution` 和原 generator 路径采样。配置、pipeline 和 stage 只负责独立开关与逐 clip 统计，不改变 Phase C/D/E、服务请求契约或默认配置。

**技术栈：** Python 3.10、PyTorch CUDA/NCCL distributed、Diffusers 0.37.0 `AutoencoderKLCogVideoX`、SGLang multimodal runtime、pytest、torchrun、tmux。

---

## 0. 执行合同

- 设计合同固定为 `docs_xzh/distribute/2026-07-17-vividvr-vae-spatial-tiled-encode-parallel-design.md`；实现发现冲突时先停在当前任务检查点，不自行扩大设计。
- 直接使用当前分支 `sglang_Vivid`；不创建 worktree，不回退或覆盖用户已有修改。
- 当前工作区已知存在用户修改：`docs_xzh/distribute/2026-07-16-vividvr-vae-spatial-tile-parallel-implementation-plan.md`。每次提交都逐项写出本任务文件路径，禁止 `git add .`，也禁止暂存该 7 月 16 日文档。
- 全程不使用子代理。执行方式固定为当前会话使用 `$executing-plans`，按任务顺序实施和审查。
- Python 固定为 `/home/zhiheng/sglang/.venv/bin/python`，`PYTHONPATH=python`。
- 所有真实 GPU 验证和推理必须通过命名明确的 tmux session 启动，并把日志写入 `Vivid_Acceptance/logs/`。
- `--vae-sp` 保持 decode-only；新开关为 `--vae-encode-sp`，默认 `False`。四种 encode/decode 组合必须彼此独立。
- 只使用 `get_sp_group()`；算法中禁止 WORLD collective 和 `all_gather_object`。验证脚本可在算法完成后用 WORLD 收集小型 JSON 报告，但不能传输模型 tensor。
- 并行 collective 开始后任何异常都必须传播；禁止捕获异常后改跑串行。
- 正确性硬门槛是 serial 与 parallel 的完整 moments 和等价 generator sampled latents 均满足 `torch.equal`。SSIM 不能豁免该门槛。
- 正式 Control 只读，不重新推理。每条 Treatment 使用其各自历史目录，runner 在前后校验 Control 的 SHA-256 和 `mtime_ns`。
- 三条正式 Treatment 串行运行；每条 compile 方案只执行现有的 1-step warmup 和一次 20-step formal。

## 1. 文件结构

### 创建

- `python/sglang/multimodal_gen/test/unit/test_cogvideox_vae_spatial_tile_encode_parallel.py`：encode tile plan、worker、merge、transport、dispatch、fallback、统计和 bitwise helper 的集中单元测试。
- `python/sglang/multimodal_gen/tools/run_vividvr_vae_spatial_encode_validation.py`：加载真实 CogVideoX VAE，在 SP2、SP4、CFG2xSP2 下比较 serial/parallel moments 与 sampled latents，并写标准验收 JSON。
- `docs_xzh/distribute/vividvr_vae_spatial_tiled_encode_parallel_acceptance_20260717.md`：记录正确性和 Treatment-only 正式验收的命令、Control 指纹、产物、门槛及结论。

### 修改

- `python/sglang/multimodal_gen/configs/pipeline_configs/base.py`：新增 `vae_encode_sp` 字段、CLI 和 tiling 前置校验。
- `python/sglang/multimodal_gen/configs/pipeline_configs/vividvr.py`：固定 Vivid-VR 默认 `vae_encode_sp=False`。
- `python/sglang/multimodal_gen/runtime/models/vaes/cogvideox.py`：实现 encode 专属 plan/worker/merge/stats/dispatch，并把 tensor transport 抽成 encode/decode 可共用的 helper。
- `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py`：配置 encode 并行；兼容长视频路径聚合 encode stats。
- `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py`：Condition stage 收集每次 encode stats，单 clip 与 modular long-clip 路径写入 debug。
- `python/sglang/multimodal_gen/test/unit/test_server_args.py`：保护新 CLI 的默认值和独立解析。
- `python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_attention_backend.py`：保护 VAE encode 并行 wiring 和缺失 native interface 时 fail-fast。
- `python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_offload.py`：保护 Condition stage 采集 encode stats 且不破坏 VAE CPU offload。
- `python/sglang/multimodal_gen/test/unit/test_stage_d_vividvr_temporal_orchestration.py`：保护两 clip encode stats 聚合。
- `python/sglang/multimodal_gen/tools/run_vividvr_acceleration_benchmark.py`：新增三个不进入默认矩阵的 encode Treatments、Control 严格校验/防改快照、有效配置校验和派生门槛。
- `python/sglang/multimodal_gen/test/unit/test_vividvr_acceleration_benchmark.py`：保护 treatment registry、命令、历史 Control、派生指标和 runner 防改行为。
- `docs_xzh/run_command/vividvr_default_run_and_serve_commands.md`：增加实验性 `--vae-encode-sp` 说明，同时明确正式默认命令不变。
- `docs_xzh/docs_analysis/acceleration_benchmark_results_20260716.md`：追加三条 Treatment 的 Long Clip Preparation、model/total、Denoise、Decode/Trim 和相对历史 Control 加速比。

### 保持不变

- 不修改 `--vae-sp` 的含义或已有 decode stats 字段。
- 不修改默认 `single_gpu_fa_compile`、`dual_gpu_fa_eager_compile`、`dual_gpu_sdpa_eager_compile`。
- 不修改 clip split、caption、denoise、decode trim、color fix、stitch 或外部 FlowCut 请求字段。
- 不修改三个历史 Control JSON；不把验收产物提交进代码 commit，除非这些路径当前受 Git 跟踪且仓库惯例明确要求。

## 2. 固定内部合同

实现期间统一使用以下名称，避免后续任务出现同义接口：

```python
@dataclass(frozen=True)
class CogVideoXSpatialEncodeTile:
    global_index: int
    row_index: int
    column_index: int
    sample_top: int
    sample_left: int


@dataclass(frozen=True)
class CogVideoXSpatialEncodeTilePlan:
    tiles: tuple[CogVideoXSpatialEncodeTile, ...]
    num_rows: int
    num_columns: int
    overlap_height: int
    overlap_width: int
    blend_extent_height: int
    blend_extent_width: int
    row_limit_height: int
    row_limit_width: int


@dataclass(frozen=True)
class CogVideoXVaeSpatialEncodeStats:
    requested: bool
    effective: bool
    fallback_reason: str
    world_size: int
    group_type: str
    total_tiles: int
    local_tiles_per_rank: tuple[int, ...]
    tile_compute_seconds: float
    tile_gather_seconds: float
    tile_merge_seconds: float
    encode_seconds: float
```

VAE 方法固定为：

```python
configure_spatial_tile_encode_parallel(requested: bool) -> None
get_last_spatial_encode_stats() -> CogVideoXVaeSpatialEncodeStats
_set_serial_encode_stats(reason: str, *, world_size: int, encode_seconds: float) -> None
_serial_tiled_encode_with_stats(x, *, reason: str, world_size: int) -> torch.Tensor
_parallel_spatial_tiled_encode(x, *, sp_group) -> torch.Tensor
```

debug 字段固定为：

```text
vae_encode_sp_requested
vae_encode_sp_effective
vae_encode_sp_fallback_reason
vae_encode_sp_world_size
vae_encode_sp_group_type
vae_encode_total_tiles
vae_encode_local_tiles_per_rank
vae_encode_tile_compute_seconds
vae_encode_tile_gather_seconds
vae_encode_tile_merge_seconds
vae_encode_seconds
vae_encode_sp_clips
```

合法 fallback 仅为 `not_requested`、`sp_world_size_one`、`input_below_tiling_threshold`；有效并行的 reason 固定为 `effective`。

---

### 任务 1：新增独立配置、CLI 与 pipeline wiring

**文件：**
- 修改：`python/sglang/multimodal_gen/configs/pipeline_configs/base.py:181-187,500-512,703-707`
- 修改：`python/sglang/multimodal_gen/configs/pipeline_configs/vividvr.py:53-58`
- 修改：`python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py:345-353,988-992`
- 测试：`python/sglang/multimodal_gen/test/unit/test_server_args.py`
- 测试：`python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_attention_backend.py:137-154`

- [ ] **步骤 1：先写 CLI 与 wiring 失败测试**

在 `test_server_args.py` 增加解析测试，确认默认关闭且只打开 encode 不会打开 decode：

```python
def test_vividvr_vae_encode_sp_cli_is_independent():
    parser = FlexibleArgumentParser()
    VividVRPipelineConfig.add_cli_args(parser)

    defaults = parser.parse_args([])
    encode_only = parser.parse_args(["--vae-encode-sp"])

    assert defaults.vae_encode_sp is False
    assert encode_only.vae_encode_sp is True
    assert encode_only.vae_sp is False
```

在 `test_stage_e_vividvr_attention_backend.py` 导入并测试新 helper：

```python
def test_vividvr_pipeline_forwards_vae_encode_sp_request_to_native_vae(self):
    vae = SimpleNamespace(
        configure_spatial_tile_encode_parallel=unittest.mock.Mock()
    )
    _configure_vividvr_vae_spatial_tile_encode_parallel(vae, True)
    vae.configure_spatial_tile_encode_parallel.assert_called_once_with(
        requested=True
    )

def test_vividvr_pipeline_rejects_vae_encode_sp_without_native_interface(self):
    with self.assertRaisesRegex(TypeError, "native CogVideoX VAE runtime"):
        _configure_vividvr_vae_spatial_tile_encode_parallel(object(), True)
```

- [ ] **步骤 2：运行失败测试**

```bash
PYTHONPATH=python .venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_server_args.py \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_attention_backend.py \
  -q
```

预期：新测试因 `vae_encode_sp` 字段、CLI 或 helper 尚不存在而失败；已有测试仍通过到新断言位置。

- [ ] **步骤 3：实现配置和独立 helper**

在 `PipelineConfig` 的 `vae_sp` 后增加：

```python
vae_encode_sp: bool = False
```

在 CLI 中增加：

```python
parser.add_argument(
    f"--{prefix_with_dot}vae-encode-sp",
    action=StoreBoolean,
    dest=f"{prefix_with_dot.replace('-', '_')}vae_encode_sp",
    default=PipelineConfig.vae_encode_sp,
    help="Parallelize CogVideoX VAE tiled encode across the SP subgroup.",
)
```

在配置校验中增加：

```python
if self.vae_encode_sp and not self.vae_tiling:
    raise ValueError(
        "Currently enabling vae_encode_sp requires enabling vae_tiling, "
        "please set --vae-tiling to True."
    )
```

`VividVRPipelineConfig` 显式固定 `vae_encode_sp: bool = False`。pipeline helper 使用与 decode helper 相同的 fail-fast 风格：

```python
def _configure_vividvr_vae_spatial_tile_encode_parallel(
    vae: object, requested: bool
) -> None:
    configure = getattr(vae, "configure_spatial_tile_encode_parallel", None)
    if configure is None:
        if requested:
            raise TypeError(
                "VividVR vae_encode_sp requires the native CogVideoX VAE runtime"
            )
        return
    configure(requested=requested)
```

在 VAE 加载后的 decode 配置旁调用：

```python
_configure_vividvr_vae_spatial_tile_encode_parallel(
    vae, bool(server_args.pipeline_config.vae_encode_sp)
)
```

- [ ] **步骤 4：运行测试并检查 help**

```bash
PYTHONPATH=python .venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_server_args.py \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_attention_backend.py \
  -q
PYTHONPATH=python .venv/bin/sglang serve --help 2>&1 \
  | rg -- '--vae-encode-sp'
```

预期：pytest 全部通过；help 恰好显示新开关，且旧 `--vae-sp` 仍存在。

- [ ] **步骤 5：提交任务 1**

```bash
git add \
  python/sglang/multimodal_gen/configs/pipeline_configs/base.py \
  python/sglang/multimodal_gen/configs/pipeline_configs/vividvr.py \
  python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py \
  python/sglang/multimodal_gen/test/unit/test_server_args.py \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_attention_backend.py
git commit -m "feat(vividvr): wire VAE tiled encode parallel config"
```

---

### 任务 2：实现 encode tile plan、worker、merge 与共享 tensor transport

**文件：**
- 修改：`python/sglang/multimodal_gen/runtime/models/vaes/cogvideox.py:13-376`
- 创建：`python/sglang/multimodal_gen/test/unit/test_cogvideox_vae_spatial_tile_encode_parallel.py`
- 回归测试：`python/sglang/multimodal_gen/test/unit/test_cogvideox_vae_spatial_tile_parallel.py`
- 回归测试：`test/srt/multimodal_gen/test_cogvideox_vae_spatial_tile_parallel_distributed.py`

- [ ] **步骤 1：写 plan 与 ownership 失败测试**

测试 720x960 使用 VAE 的 240x360 sample tiles 和 overlap 后得到 4x4 row-major plan，并验证 SP2/SP4 数量：

```python
def test_encode_plan_matches_formal_720x960_geometry():
    plan = _build_spatial_encode_tile_plan(
        sample_height=720,
        sample_width=960,
        tile_sample_min_height=240,
        tile_sample_min_width=360,
        tile_latent_min_height=30,
        tile_latent_min_width=45,
        tile_overlap_factor_height=1 / 6,
        tile_overlap_factor_width=1 / 5,
    )
    assert (plan.num_rows, plan.num_columns) == (4, 4)
    assert [(t.global_index, t.sample_top, t.sample_left) for t in plan.tiles] == [
        (index, top, left)
        for index, (top, left) in enumerate(
            ([(top, left) for top in (0, 200, 400, 600) for left in (0, 288, 576, 864)])
        )
    ]
    assert (plan.blend_extent_height, plan.blend_extent_width) == (5, 9)
    assert (plan.row_limit_height, plan.row_limit_width) == (25, 36)

@pytest.mark.parametrize("world_size,expected", [(2, [8, 8]), (4, [4, 4, 4, 4])])
def test_encode_tiles_use_round_robin_ownership(world_size, expected):
    plan = make_formal_encode_plan()
    counts = [
        len(_assign_spatial_tiles(plan.tiles, rank, world_size))
        for rank in range(world_size)
    ]
    assert counts == expected
```

- [ ] **步骤 2：写 worker 与 merge 失败测试**

使用 toy encoder 记录每个 temporal slice 和 `conv_cache`：

```python
def test_encode_one_tile_preserves_temporal_cache_and_quant_conv():
    vae = ToyEncodeVae(frame_batch_size=4)
    x = torch.arange(1 * 3 * 9 * 6 * 8, dtype=torch.float32).reshape(1, 3, 9, 6, 8)
    tile = CogVideoXSpatialEncodeTile(0, 0, 0, 0, 0)

    encoded = _encode_one_spatial_tile(vae, x, tile)

    assert vae.encoder_ranges == [(0, 5), (5, 9)]
    assert vae.encoder_input_caches == [None, "cache-0"]
    assert vae.quant_conv_calls == 2
    assert encoded.shape[2] == 9
```

merge 测试用会记录调用的 `blend_v`/`blend_h`，并断言先竖直、后水平、最后 crop 与 row-major concat：

```python
def test_encode_merge_matches_diffusers_order_and_crop():
    vae = RecordingBlendVae()
    plan = make_two_by_two_encode_plan()
    tiles = {index: tagged_tile(index) for index in range(4)}

    merged = _merge_spatial_encode_tiles(vae, plan, tiles)

    assert vae.calls == [
        ("h", 0, 1),
        ("v", 0, 2),
        ("v", 1, 3),
        ("h", 2, 3),
    ]
    assert merged.shape[-2:] == (
        plan.num_rows * plan.row_limit_height,
        plan.num_columns * plan.row_limit_width,
    )
```

- [ ] **步骤 3：运行失败测试**

```bash
PYTHONPATH=python .venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_cogvideox_vae_spatial_tile_encode_parallel.py \
  -q
```

预期：collection 因 encode 类型和 helper 尚不存在而失败。

- [ ] **步骤 4：实现 encode 专属纯函数**

plan 必须复制 Diffusers encode 几何，而不是复用 decode 几何：

```python
def _build_spatial_encode_tile_plan(
    *,
    sample_height: int,
    sample_width: int,
    tile_sample_min_height: int,
    tile_sample_min_width: int,
    tile_latent_min_height: int,
    tile_latent_min_width: int,
    tile_overlap_factor_height: float,
    tile_overlap_factor_width: float,
) -> CogVideoXSpatialEncodeTilePlan:
    overlap_height = int(tile_sample_min_height * (1 - tile_overlap_factor_height))
    overlap_width = int(tile_sample_min_width * (1 - tile_overlap_factor_width))
    if overlap_height <= 0 or overlap_width <= 0:
        raise ValueError("CogVideoX VAE encode tile overlap stride must be positive")
    blend_extent_height = int(tile_latent_min_height * tile_overlap_factor_height)
    blend_extent_width = int(tile_latent_min_width * tile_overlap_factor_width)
    coordinates = [
        (row, column, top, left)
        for row, top in enumerate(range(0, sample_height, overlap_height))
        for column, left in enumerate(range(0, sample_width, overlap_width))
    ]
    return CogVideoXSpatialEncodeTilePlan(
        tiles=tuple(
            CogVideoXSpatialEncodeTile(index, row, column, top, left)
            for index, (row, column, top, left) in enumerate(coordinates)
        ),
        num_rows=len(range(0, sample_height, overlap_height)),
        num_columns=len(range(0, sample_width, overlap_width)),
        overlap_height=overlap_height,
        overlap_width=overlap_width,
        blend_extent_height=blend_extent_height,
        blend_extent_width=blend_extent_width,
        row_limit_height=tile_latent_min_height - blend_extent_height,
        row_limit_width=tile_latent_min_width - blend_extent_width,
    )
```

worker 完整保留 temporal loop，每个空间 tile 从 `conv_cache=None` 开始，并在每个 temporal chunk 的 encoder 后立即应用 `quant_conv`：

```python
def _encode_one_spatial_tile(vae, x, tile):
    frame_batch_size = vae.num_sample_frames_batch_size
    num_frames = x.shape[2]
    num_batches = max(num_frames // frame_batch_size, 1)
    conv_cache = None
    temporal_parts = []
    for batch_index in range(num_batches):
        remaining_frames = num_frames % frame_batch_size
        start_frame = frame_batch_size * batch_index + (
            0 if batch_index == 0 else remaining_frames
        )
        end_frame = frame_batch_size * (batch_index + 1) + remaining_frames
        encoded = x[
            :, :, start_frame:end_frame,
            tile.sample_top : tile.sample_top + vae.tile_sample_min_height,
            tile.sample_left : tile.sample_left + vae.tile_sample_min_width,
        ]
        encoded, conv_cache = vae.encoder(encoded, conv_cache=conv_cache)
        if vae.quant_conv is not None:
            encoded = vae.quant_conv(encoded)
        temporal_parts.append(encoded)
    return torch.cat(temporal_parts, dim=2)
```

merge body与 Diffusers `tiled_encode` 同序，且从未 blend 的 `rows` 取上方和左侧 tile。

- [ ] **步骤 5：把 transport 提升为共享 helper，保留 decode 兼容 wrapper**

不要删除 `_all_gather_decoded_tiles`，避免破坏已有 unit/NCCL 测试。新增：

```python
def _all_gather_spatial_tiles(
    sp_group, local_tiles, total_tiles, *, payload_dtype, payload_device
):
    slots_per_rank = (total_tiles + sp_group.world_size - 1) // sp_group.world_size
    metadata, payload = _pack_local_tiles(
        local_tiles,
        slots_per_rank,
        payload_dtype=payload_dtype,
        payload_device=payload_device,
    )
    gathered_metadata = sp_group.all_gather(metadata, dim=0).reshape(
        sp_group.world_size, slots_per_rank, 7
    )
    rank_numels = gathered_metadata[:, :, 6].clamp_min(0).sum(dim=1)
    max_rank_numel = int(rank_numels.max().item())
    padded_payload = torch.zeros(
        max_rank_numel, dtype=payload.dtype, device=payload.device
    )
    padded_payload[: payload.numel()].copy_(payload)
    gathered_payload = sp_group.all_gather(padded_payload, dim=0).reshape(
        sp_group.world_size, max_rank_numel
    )
    return _unpack_gathered_tiles(
        gathered_metadata, gathered_payload, total_tiles=total_tiles
    )


def _all_gather_decoded_tiles(
    sp_group, local_tiles, total_tiles, *, payload_dtype, payload_device
):
    return _all_gather_spatial_tiles(
        sp_group,
        local_tiles,
        total_tiles,
        payload_dtype=payload_dtype,
        payload_device=payload_device,
    )
```

上述函数体必须由当前 `_all_gather_decoded_tiles` 原样提升，不允许改变 `_pack_local_tiles`、`_unpack_gathered_tiles` 或 collective 顺序。增加测试确认 wrapper 和共享 helper 对同一 fake gather 返回完全一致的 keys、shape、dtype 和 tensor bytes。

- [ ] **步骤 6：运行 encode 与 decode 回归测试**

```bash
PYTHONPATH=python .venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_cogvideox_vae_spatial_tile_encode_parallel.py \
  python/sglang/multimodal_gen/test/unit/test_cogvideox_vae_spatial_tile_parallel.py \
  -q
```

预期：全部通过；encode 4x4/edge tile/worker/merge/transport 测试通过，decode 测试数量与行为不减少。

- [ ] **步骤 7：提交任务 2**

```bash
git add \
  python/sglang/multimodal_gen/runtime/models/vaes/cogvideox.py \
  python/sglang/multimodal_gen/test/unit/test_cogvideox_vae_spatial_tile_encode_parallel.py
git commit -m "feat(vividvr): add VAE tiled encode primitives"
```

---

### 任务 3：实现 VAE encode dispatch、canonicalization、统计与失败策略

**文件：**
- 修改：`python/sglang/multimodal_gen/runtime/models/vaes/cogvideox.py:197-251,379-597`
- 测试：`python/sglang/multimodal_gen/test/unit/test_cogvideox_vae_spatial_tile_encode_parallel.py`

- [ ] **步骤 1：写 descriptor、canonicalization 和错误测试**

覆盖 shape/dtype/tiling/world size/subgroup ranks descriptor、非连续输入、rank-divergent root broadcast、duplicate/missing tile 和不支持 dtype：

```python
def test_encode_descriptor_rejects_rank_mismatch():
    group = DescriptorMismatchGroup(world_size=2)
    x = torch.zeros(1, 3, 9, 241, 361)
    with pytest.raises(RuntimeError, match="encode input descriptor mismatch"):
        _validate_spatial_encode_descriptor(group, x, make_encode_plan())

def test_encode_input_is_contiguous_and_canonicalized_from_subgroup_root():
    root = torch.arange(24).reshape(1, 3, 2, 2, 2).transpose(-1, -2)
    local = torch.full_like(root, -1)
    group = BroadcastFakeGroup(root.contiguous())
    canonical = _canonicalize_spatial_encode_input(group, local)
    assert canonical.is_contiguous()
    assert torch.equal(canonical, root.contiguous())
```

- [ ] **步骤 2：写 dispatch、fallback、post-collective fatal 和 stats 失败测试**

```python
def test_encode_parallel_startup_requires_tiling_and_sp_group(monkeypatch):
    vae = make_toy_runtime(use_tiling=False)
    with pytest.raises(ValueError, match="vae_encode_sp requires VAE tiling"):
        vae.configure_spatial_tile_encode_parallel(True)

def test_encode_parallel_does_not_retry_serial_after_collective_failure(monkeypatch):
    vae = make_toy_runtime(use_tiling=True, world_size=2)
    monkeypatch.setattr(cogvideox, "_all_gather_spatial_tiles", fail_collective)
    serial_calls = 0
    original_serial = DiffusersAutoencoderKLCogVideoX.tiled_encode

    def record_serial_call(self, x):
        nonlocal serial_calls
        serial_calls += 1
        return original_serial(self, x)

    monkeypatch.setattr(
        DiffusersAutoencoderKLCogVideoX, "tiled_encode", record_serial_call
    )
    with pytest.raises(RuntimeError, match="collective failed"):
        vae.tiled_encode(make_tiled_input())
    assert serial_calls == 0

@pytest.mark.parametrize(
    "requested,world_size,tiled,reason",
    [
        (False, 1, True, "not_requested"),
        (True, 1, True, "sp_world_size_one"),
        (True, 2, False, "input_below_tiling_threshold"),
    ],
)
def test_encode_serial_fallback_reasons(requested, world_size, tiled, reason):
    vae = make_toy_runtime(use_tiling=True, world_size=world_size)
    vae.configure_spatial_tile_encode_parallel(requested)
    x = (
        make_tiled_input()
        if tiled
        else torch.zeros(1, 3, 5, 240, 360, dtype=torch.float32)
    )

    vae._encode(x)

    stats = vae.get_last_spatial_encode_stats()
    assert stats.effective is False
    assert stats.fallback_reason == reason
    assert stats.total_tiles == 0
    assert stats.tile_compute_seconds >= 0.0
    assert stats.tile_gather_seconds >= 0.0
    assert stats.tile_merge_seconds >= 0.0
    assert stats.encode_seconds >= 0.0
```

`make_toy_runtime` 必须给请求路径注入 `FakeGroup(world_size)`；world size 2 的 below-threshold 输入固定为 `[1,3,5,240,360]`。

- [ ] **步骤 3：运行失败测试**

```bash
PYTHONPATH=python .venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_cogvideox_vae_spatial_tile_encode_parallel.py \
  -q
```

预期：因 encode stats、配置方法、descriptor 和 parallel dispatch 尚不存在而失败。

- [ ] **步骤 4：实现 encode stats 与配置方法**

`serial_default()` 和 `to_debug_dict()` 必须完整输出第 2 节固定字段。构造函数新增：

```python
self._vae_encode_sp_requested = False
self._vae_encode_sp_group = None
self._last_spatial_encode_stats = CogVideoXVaeSpatialEncodeStats.serial_default()
```

配置方法必须在请求时校验 tiling 和 `get_sp_group()`，并把 AssertionError 转为带 `vae_encode_sp` 的 RuntimeError；关闭时不要求分布式初始化。

- [ ] **步骤 5：实现 `_encode` 与 `tiled_encode` dispatch**

```python
def _encode(self, x: torch.Tensor) -> torch.Tensor:
    tiled = self.use_tiling and (
        x.shape[-1] > self.tile_sample_min_width
        or x.shape[-2] > self.tile_sample_min_height
    )
    if tiled:
        return self.tiled_encode(x)
    reason = (
        "input_below_tiling_threshold"
        if self._vae_encode_sp_requested
        else "not_requested"
    )
    return self._serial_encode_with_stats(
        x, reason=reason,
        world_size=self._vae_encode_sp_group.world_size
        if self._vae_encode_sp_group else 1,
    )

def tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
    if not self._vae_encode_sp_requested:
        return self._serial_tiled_encode_with_stats(
            x, reason="not_requested", world_size=1
        )
    sp_group = self._vae_encode_sp_group
    if sp_group is None:
        raise RuntimeError("CogVideoX VAE encode SP group is unavailable")
    if sp_group.world_size == 1:
        return self._serial_tiled_encode_with_stats(
            x, reason="sp_world_size_one", world_size=1
        )
    return self._parallel_spatial_tiled_encode(x, sp_group=sp_group)
```

串行 helper 用 CUDA events 包住 `super()._encode(x)` 或 `super().tiled_encode(x)`，不可从 override 递归回自己。

- [ ] **步骤 6：实现 parallel dataflow**

执行顺序固定为：build plan → tensor descriptor all-gather → contiguous clone + subgroup-root broadcast → round-robin local encode → `_all_gather_spatial_tiles` → validate complete sorted tiles → replicated merge → merged shape 校验 → 写 stats。descriptor 必须包含：输入 5D shape、dtype code、tile 参数、tile count、world size、`sp_group.ranks`。

统计计时边界固定为：

```text
tile_compute_seconds = canonicalization 完成后到本地 tiles 完成
tile_gather_seconds  = gather 开始到完整 tiles 恢复
tile_merge_seconds   = merge 开始到 merged moments 完成
encode_seconds       = parallel 方法入口到 merged moments 完成
```

使用与 decode 相同的 CUDA event 方式；不得在 timing 之外额外调用 `torch.cuda.synchronize()` 改变热点路径。

- [ ] **步骤 7：运行完整 VAE 单元回归**

```bash
PYTHONPATH=python .venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_cogvideox_vae_spatial_tile_encode_parallel.py \
  python/sglang/multimodal_gen/test/unit/test_cogvideox_vae_spatial_tile_parallel.py \
  -q
```

预期：全部通过；特别确认 fatal 测试证明 collective 后没有 serial retry。

- [ ] **步骤 8：提交任务 3**

```bash
git add \
  python/sglang/multimodal_gen/runtime/models/vaes/cogvideox.py \
  python/sglang/multimodal_gen/test/unit/test_cogvideox_vae_spatial_tile_encode_parallel.py
git commit -m "feat(vividvr): parallelize VAE tiled encode"
```

---

### 任务 4：传播单 clip 与长视频逐 clip encode 统计

**文件：**
- 修改：`python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py:166-218,287-311,483-625,1595-1745`
- 修改：`python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py:78-80,1220-1470`
- 测试：`python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_offload.py:250-320`
- 测试：`python/sglang/multimodal_gen/test/unit/test_stage_d_vividvr_temporal_orchestration.py:491-581`

- [ ] **步骤 1：写 Condition stage 失败测试**

fake VAE 的每次 `encode()` 后返回不同 stats，断言 `prepare_condition_inputs()` 返回 clip-level stats，且 CPU offload 仍发生在 stats 读取之后：

```python
def test_condition_stage_exposes_vae_encode_stats_before_cpu_offload(self):
    vae = EncodeStatsVae()
    stage = make_condition_stage(vae)
    prepared = stage.prepare_condition_inputs(
        make_batch(), make_server_args(vae_cpu_offload=True)
    )
    self.assertTrue(prepared["vae_encode_stats"]["vae_encode_sp_effective"])
    self.assertEqual(vae.to_calls[-1], "cpu")
```

- [ ] **步骤 2：写两 clip 聚合失败测试**

在 temporal orchestration 测试中让两个 clip 分别返回 16 tiles、`[8,8]`，断言：

```python
self.assertEqual(debug["vae_encode_total_tiles"], 32)
self.assertEqual(debug["vae_encode_local_tiles_per_rank"], [16, 16])
self.assertEqual(len(debug["vae_encode_sp_clips"]), 2)
self.assertAlmostEqual(debug["vae_encode_seconds"], 12.0)
self.assertTrue(debug["vae_encode_sp_requested"])
self.assertTrue(debug["vae_encode_sp_effective"])
```

另加 topology mismatch 测试：一个 clip world size 2、另一个 4 时抛出 `RuntimeError("VAE encode SP clip topology mismatch")`。

- [ ] **步骤 3：运行失败测试**

```bash
PYTHONPATH=python .venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_offload.py \
  python/sglang/multimodal_gen/test/unit/test_stage_d_vividvr_temporal_orchestration.py \
  -q
```

预期：新 encode stats 断言失败，既有 decode 与 temporal 测试继续通过。

- [ ] **步骤 4：实现 slice-level 收集和 clip-level 聚合**

把 `_encode_control_latents` 的 list comprehension 改为等序显式循环；每次 `vae.encode` 后读取 `get_last_spatial_encode_stats().to_debug_dict()`。返回 `(control_latents, slice_stats)`，generator 调用顺序保持完全不变。`prepare_condition_inputs` 使用新 `_aggregate_vae_spatial_encode_stats(slice_stats)` 得到单 temporal clip 统计并放入 `prepared["vae_encode_stats"]`。

新增 aggregator 的字段计算与 decode 对称：requested/effective 用 `all`，reason 单一时原值否则 `mixed`，world/group/本地列表宽度必须一致，tile 数和四个 duration 求和，并附原始 `vae_encode_sp_clips`。对空列表返回 `{}`。

- [ ] **步骤 5：接入三条调用路径**

1. `VividVRConditionEncodingStage.forward`：将 `prepared["vae_encode_stats"]` 更新到 `vividvr_debug`，并保留单 clip 的 `vae_encode_sp_clips`。
2. `VividVRLongClipPreparationStage.forward`：循环中收集每个 `prepared_condition["vae_encode_stats"]`，循环结束用 aggregator 更新 debug。
3. `VividVRPipeline` 保留的兼容长视频循环：执行同样收集与聚合，避免该路径丢失 benchmark 字段。

不要在 latent preparation、denoise 或 decode stage 再读取 encode stats，防止最后一次调用覆盖 clip 边界。

- [ ] **步骤 6：运行 stage 与 pipeline 回归**

```bash
PYTHONPATH=python .venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_offload.py \
  python/sglang/multimodal_gen/test/unit/test_stage_d_vividvr_temporal_orchestration.py \
  python/sglang/multimodal_gen/test/unit/test_stage_c_vividvr_single_clip.py \
  -q
```

预期：全部通过，decode stats、clip trim/stitch 和 Phase C prompt/latent 断言不变。

- [ ] **步骤 7：提交任务 4**

```bash
git add \
  python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py \
  python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_offload.py \
  python/sglang/multimodal_gen/test/unit/test_stage_d_vividvr_temporal_orchestration.py
git commit -m "feat(vividvr): report VAE tiled encode SP stats"
```

---

### 任务 5：新增真实 VAE/NCCL bitwise 验证工具

**文件：**
- 创建：`python/sglang/multimodal_gen/tools/run_vividvr_vae_spatial_encode_validation.py`
- 修改：`python/sglang/multimodal_gen/test/unit/test_cogvideox_vae_spatial_tile_encode_parallel.py`
- 参考但不修改：`python/sglang/multimodal_gen/tools/run_vividvr_vae_spatial_decode_validation.py`

- [ ] **步骤 1：先写纯比较 helper 失败测试**

```python
def test_encode_validation_requires_exact_moments_and_sampled_latents():
    moments = torch.arange(8, dtype=torch.bfloat16)
    latents = torch.arange(4, dtype=torch.bfloat16)
    assert compare_serial_and_parallel_encode(
        moments, moments.clone(), latents, latents.clone()
    )["passed"]
    changed = moments.clone()
    changed[0] += 1
    result = compare_serial_and_parallel_encode(
        moments, changed, latents, latents.clone()
    )
    assert result["moments_exact"] is False
    assert result["passed"] is False
```

- [ ] **步骤 2：运行失败测试**

```bash
PYTHONPATH=python .venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_cogvideox_vae_spatial_tile_encode_parallel.py \
  -q
```

预期：验证 module/helper 尚不存在导致失败。

- [ ] **步骤 3：实现验证 CLI 和 exact comparison**

CLI 固定支持：

```text
--model-path（默认 /home/zhiheng/ckpts/CogVideoX1.5-5B）
--topology {sp2,sp4,cfg2_sp2}
--seed（默认 42）
--sample-frames（默认 17）
--sample-height（默认 720）
--sample-width（默认 960）
--output-json（必填）
```

`compare_serial_and_parallel_encode` 只以两个 `torch.equal` 决定 pass，同时记录 shape、max/mean abs error 便于诊断：

```python
def compare_serial_and_parallel_encode(
    serial_moments, parallel_moments, serial_latents, parallel_latents
):
    moments_exact = (
        serial_moments.shape == parallel_moments.shape
        and torch.equal(serial_moments, parallel_moments)
    )
    sampled_latents_exact = (
        serial_latents.shape == parallel_latents.shape
        and torch.equal(serial_latents, parallel_latents)
    )
    return {
        "moments_exact": bool(moments_exact),
        "sampled_latents_exact": bool(sampled_latents_exact),
        "passed": bool(moments_exact and sampled_latents_exact),
        "serial_moments_shape": list(serial_moments.shape),
        "parallel_moments_shape": list(parallel_moments.shape),
        "serial_latents_shape": list(serial_latents.shape),
        "parallel_latents_shape": list(parallel_latents.shape),
        "moments_max_abs_error": float(
            (serial_moments.float() - parallel_moments.float()).abs().max().item()
        )
        if serial_moments.shape == parallel_moments.shape
        else None,
        "moments_mean_abs_error": float(
            (serial_moments.float() - parallel_moments.float()).abs().mean().item()
        )
        if serial_moments.shape == parallel_moments.shape
        else None,
        "sampled_latents_max_abs_error": float(
            (serial_latents.float() - parallel_latents.float()).abs().max().item()
        )
        if serial_latents.shape == parallel_latents.shape
        else None,
        "sampled_latents_mean_abs_error": float(
            (serial_latents.float() - parallel_latents.float()).abs().mean().item()
        )
        if serial_latents.shape == parallel_latents.shape
        else None,
    }
```

- [ ] **步骤 4：实现真实模型流程**

复用 decode validation 的 topology 初始化和 `PipelineComponentLoader`。每个 CFG subgroup 使用 `seed + cfg_group_index` 生成独立 input；通过先创建 `[W,H]` 再 transpose 得到非连续 `[H,W]` 输入。流程固定：

1. 配置 encode SP 关闭，执行 `vae.encode(x)`，保存 `latent_dist.parameters`。
2. 创建 `torch.Generator(device).manual_seed(seed + 1000 + cfg_group_index)`，采样 serial latents。
3. 配置 encode SP 打开，重建相同 generator state，执行 parallel encode 和 sampling。
4. 创建 rank-divergent input，验证 subgroup-root canonicalization 的结果仍与 root serial reference exact。
5. 算法 tensor 全部只通过 SP subgroup；完成后才用 `dist.all_gather_object` 汇集每 rank 的小型报告。

rank 0 JSON 至少包含标准 `total_runtime_seconds`、`model_inference_runtime_seconds`，并包含 topology、subgroup ranks、input/moments/latents shape、noncontiguous 标记、每 rank exact booleans、rank-divergent exact booleans、encode stats、peak memory 和 `overall_pass`。

- [ ] **步骤 5：运行纯 Python 测试和 CLI help**

```bash
PYTHONPATH=python .venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_cogvideox_vae_spatial_tile_encode_parallel.py \
  -q
PYTHONPATH=python .venv/bin/python \
  python/sglang/multimodal_gen/tools/run_vividvr_vae_spatial_encode_validation.py \
  --help
```

预期：单元测试通过；help 显示三个 topology 和固定尺寸参数。

- [ ] **步骤 6：提交任务 5**

```bash
git add \
  python/sglang/multimodal_gen/tools/run_vividvr_vae_spatial_encode_validation.py \
  python/sglang/multimodal_gen/test/unit/test_cogvideox_vae_spatial_tile_encode_parallel.py
git commit -m "test(vividvr): add VAE encode SP bitwise validation"
```

---

### 任务 6：扩展 benchmark Treatment registry、命令与有效配置校验

**文件：**
- 修改：`python/sglang/multimodal_gen/tools/run_vividvr_acceleration_benchmark.py:58-155,250-295,440-570,2140-2190`
- 修改：`python/sglang/multimodal_gen/test/unit/test_vividvr_acceleration_benchmark.py:90-170,290-440`

- [ ] **步骤 1：写 registry 和命令失败测试**

新增三个 ID：`R99_VAE_ENCODE_SP`、`R100_VAE_ENCODE_SP`、`R101_VAE_ENCODE_SP4`。测试断言它们只存在于 `ALL_SCHEMES`，不进入默认 `SCHEMES`/`run-all`；每条 treatment command 相比对应历史 control command 只多 `--vae-encode-sp`，且同时保留 `--vae-sp`。

```python
@pytest.mark.parametrize(
    "treatment_id,control_id",
    [
        ("R99_VAE_ENCODE_SP", "R99_VAE_SP"),
        ("R100_VAE_ENCODE_SP", "R100_VAE_SP"),
        ("R101_VAE_ENCODE_SP4", "R101_VAE_SP4"),
    ],
)
def test_vae_encode_sp_treatment_adds_only_encode_flag(
    treatment_id, control_id, tmp_path
):
    treatment = build_service_command(ALL_SCHEMES[treatment_id], make_config(tmp_path))
    control = build_service_command(ALL_SCHEMES[control_id], make_config(tmp_path))
    assert treatment == control + ["--vae-encode-sp"]
```

- [ ] **步骤 2：写 effective debug 失败测试**

扩展 `make_perf_fixture(vae_encode_sp=True)` 生成固定字段，测试 effective=true、world size 与 scheme 相同、本地 counts 总和等于 total、四个 timing 非负；将 effective 改为 false/fallback 后必须抛 `BenchmarkDataError`。

- [ ] **步骤 3：运行失败测试**

```bash
PYTHONPATH=python .venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_vividvr_acceleration_benchmark.py \
  -q
```

预期：新 scheme、field 和 service flag 尚不存在导致失败。

- [ ] **步骤 4：实现 scheme registry**

`Scheme` 和 `_scheme` 新增 `vae_encode_sp: bool = False`。新 registry：

```python
VAE_ENCODE_SP_TREATMENTS = {
    "R99_VAE_ENCODE_SP": _scheme(
        "R99_VAE_ENCODE_SP", "R99 VAE decode+encode spatial tile parallel",
        gpu_count=2, parallel_mode="sp", sp_degree=2,
        compile_enabled=True, modulation_fusion=True,
        vae_sp=True, vae_encode_sp=True, controls=("R99_VAE_SP",),
    ),
    "R100_VAE_ENCODE_SP": _scheme(
        "R100_VAE_ENCODE_SP", "R100 VAE decode+encode spatial tile parallel",
        gpu_count=4, parallel_mode="cfg_sp", sp_degree=2,
        compile_enabled=True, modulation_fusion=True,
        vae_sp=True, vae_encode_sp=True, controls=("R100_VAE_SP",),
    ),
    "R101_VAE_ENCODE_SP4": _scheme(
        "R101_VAE_ENCODE_SP4", "SP4 VAE decode+encode spatial tile parallel",
        gpu_count=4, parallel_mode="sp", sp_degree=4,
        compile_enabled=True, modulation_fusion=True,
        vae_sp=True, vae_encode_sp=True, controls=("R101_VAE_SP4",),
    ),
}
ALL_SCHEMES = {**SCHEMES, **VAE_SP_TREATMENTS, **VAE_ENCODE_SP_TREATMENTS}
```

`build_service_command` 在现有 `--vae-sp` 后独立追加 `--vae-encode-sp`。

- [ ] **步骤 5：实现 encode effective config 校验**

在 `validate_effective_config` 中独立校验第 2 节全部 encode 字段。`vae_encode_sp_clips` 必须是非空 list、长度等于 `debug["num_clips"]`，且每个 clip effective。validated runtime 原样保留 aggregate counts/timings 和 clips，便于正式 record 审计。

- [ ] **步骤 6：运行 benchmark 单元测试**

```bash
PYTHONPATH=python .venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_vividvr_acceleration_benchmark.py \
  -q
```

预期：全部通过；默认 `run-all` scheme 数量不增加。

- [ ] **步骤 7：提交任务 6**

```bash
git add \
  python/sglang/multimodal_gen/tools/run_vividvr_acceleration_benchmark.py \
  python/sglang/multimodal_gen/test/unit/test_vividvr_acceleration_benchmark.py
git commit -m "feat(vividvr): register VAE encode SP benchmarks"
```

---

### 任务 7：实现历史 Control 严格校验、防改快照与性能门槛

**文件：**
- 修改：`python/sglang/multimodal_gen/tools/run_vividvr_acceleration_benchmark.py:630-900,1810-2010`
- 修改：`python/sglang/multimodal_gen/test/unit/test_vividvr_acceleration_benchmark.py:520-620`

- [ ] **步骤 1：写历史 Control 身份失败测试**

fixture 必须包含完整 scheme、inputs、runtime 和三个 stage。分别篡改 `scheme_id`、topology、seed、caption path、decode `vae_sp_effective`、encode SP=true、Long Clip/Denoise/Decode stage 缺失，逐项断言 `load_historical_controls` 拒绝。

合法记录需允许 `status="quality_failed"`，因为三个历史 Control 已按用户确认可用。

- [ ] **步骤 2：写 SHA/mtime 防改失败测试**

```python
def test_historical_control_snapshot_detects_content_or_mtime_change(tmp_path):
    control_dir = write_complete_encode_control(tmp_path)
    controls = load_historical_controls(
        control_dir, ALL_SCHEMES["R99_VAE_ENCODE_SP"]
    )
    path = control_dir / "records/R99_VAE_SP_formal.json"
    path.write_text(path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(BenchmarkDataError, match="historical control changed"):
        verify_historical_controls_unchanged(controls)
```

另一个测试只调用 `os.utime` 改 `mtime_ns`，内容 hash 不变也必须失败。

- [ ] **步骤 3：写派生性能门槛失败测试**

参数化三种 topology，断言：

```python
assert derived["long_clip_preparation_speedup"] == control_prep / treatment_prep
assert derived["long_clip_preparation_gate"] is expected_prep_gate
assert derived["model_inference_improved"] is expected_model_gate
assert derived["denoise_regression_ratio"] == treatment_denoise / control_denoise - 1
assert derived["decode_trim_regression_ratio"] == treatment_decode / control_decode - 1
assert derived["performance_gates_passed"] is expected_all
```

边界值必须覆盖 SP2/CFG2xSP2 的 1.5x、SP4 的 2.5x，以及 Denoise/Decode 恰好 3% 通过、超过 3% 失败。

- [ ] **步骤 4：运行失败测试**

```bash
PYTHONPATH=python .venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_vividvr_acceleration_benchmark.py \
  -q
```

预期：snapshot、严格身份校验和 encode 派生函数不存在导致失败。

- [ ] **步骤 5：实现 Control snapshot 和完整身份校验**

新增 `_file_sha256(path)`，读取时把下面字段加入内存 copy，不写回 Control：

```python
copied["_control_record_snapshot"] = {
    "path": str(path.resolve()),
    "sha256": _file_sha256(path),
    "mtime_ns": path.stat().st_mtime_ns,
}
```

`verify_historical_controls_unchanged` 重新计算两项并逐文件对比。encode treatment 的 Control 校验固定检查：

- Control scheme ID 等于 treatment 唯一 control ID；
- gpu_count/backend/parallel_mode/sp_degree/compile/fusion 与 `ALL_SCHEMES[control_id]` 相等；
- `vae_sp is True`，`vae_encode_sp` 缺失或 false；
- input/caption/reference 是 `BenchmarkConfig` 的规范 resolved path；
- frames=130、temporal=121、steps=20、seed=42、guidance=6、restoration=-1、upscale=1、dtype=bfloat16；
- runtime effective backend/topology/SP/CFG/compile/fusion/decode VAE SP 与 Control scheme 一致；
- total/model、`VividVRLongClipPreparationStage`、`VividVRMultiClipDenoisingStage`、`VividVRMultiClipDecodeTrimStage` 全为正数；
- quality 字段存在，但 `status` 和 `pass_compare` 不作为拒绝质量失败历史记录的条件。

- [ ] **步骤 6：实现 encode 派生门槛**

新增 `compute_vae_encode_sp_derived_metrics`，Control GPU 数从 `ALL_SCHEMES[control_id].gpu_count` 获取。threshold 为：

```python
required_prep_speedup = 2.5 if scheme.sp_degree == 4 else 1.5
model_inference_improved = treatment_model < control_model
denoise_regression_ratio = treatment_denoise / control_denoise - 1.0
decode_trim_regression_ratio = treatment_decode / control_decode - 1.0
performance_gates_passed = all(
    (
        control_prep / treatment_prep >= required_prep_speedup,
        model_inference_improved,
        denoise_regression_ratio <= 0.03,
        decode_trim_regression_ratio <= 0.03,
    )
)
```

同时保留 total/model speedup、control/treatment GPU-seconds、质量 delta、Control path/hash/mtime 和各 gate 的布尔值。

- [ ] **步骤 7：把 snapshot 校验接入 runner 生命周期**

每个 treatment 在 `start_scheme` 前加载并校验 Control；在 scheme 的 `finally` 中先停止本 scheme owned sessions，再调用 `verify_historical_controls_unchanged`。无论 warmup、formal、quality compare 或 service cleanup 是否成功，Control 都要复核。正式 record 成功或 `quality_failed` 时调用 encode 派生函数；旧 decode-only treatment 继续调用原函数。

- [ ] **步骤 8：运行 benchmark 测试**

```bash
PYTHONPATH=python .venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_vividvr_acceleration_benchmark.py \
  -q
```

预期：全部通过，包括 content change、mtime-only change、quality_failed Control 和所有 gate 边界。

- [ ] **步骤 9：提交任务 7**

```bash
git add \
  python/sglang/multimodal_gen/tools/run_vividvr_acceleration_benchmark.py \
  python/sglang/multimodal_gen/test/unit/test_vividvr_acceleration_benchmark.py
git commit -m "test(vividvr): harden VAE encode SP acceptance controls"
```

---

### 任务 8：完成静态、单元和已有 decode distributed 回归

**文件：**
- 检查：本计划涉及的全部 Python 文件

- [ ] **步骤 1：运行格式和 lint**

先检查主环境是否已有 `pre_commit`：

```bash
.venv/bin/python -c 'import pre_commit; print(pre_commit.__file__)'
```

若该检查失败，按仓库环境规则把工具安装进同一个 `.venv`，然后重新执行检查；不要切换解释器：

```bash
uv pip install --python /home/zhiheng/sglang/.venv/bin/python pre-commit
```

随后运行：

```bash
.venv/bin/python -m pre_commit run --files \
  python/sglang/multimodal_gen/configs/pipeline_configs/base.py \
  python/sglang/multimodal_gen/configs/pipeline_configs/vividvr.py \
  python/sglang/multimodal_gen/runtime/models/vaes/cogvideox.py \
  python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py \
  python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py \
  python/sglang/multimodal_gen/tools/run_vividvr_vae_spatial_encode_validation.py \
  python/sglang/multimodal_gen/tools/run_vividvr_acceleration_benchmark.py \
  python/sglang/multimodal_gen/test/unit/test_cogvideox_vae_spatial_tile_encode_parallel.py \
  python/sglang/multimodal_gen/test/unit/test_server_args.py \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_attention_backend.py \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_offload.py \
  python/sglang/multimodal_gen/test/unit/test_stage_d_vividvr_temporal_orchestration.py \
  python/sglang/multimodal_gen/test/unit/test_vividvr_acceleration_benchmark.py
```

预期：所有 hooks passed；若 formatter 改文件，只 add 本任务路径并追加 `style(vividvr): format VAE encode SP changes` commit。

- [ ] **步骤 2：运行聚焦单元回归**

```bash
PYTHONPATH=python .venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_cogvideox_vae_spatial_tile_encode_parallel.py \
  python/sglang/multimodal_gen/test/unit/test_cogvideox_vae_spatial_tile_parallel.py \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_attention_backend.py \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_offload.py \
  python/sglang/multimodal_gen/test/unit/test_stage_c_vividvr_single_clip.py \
  python/sglang/multimodal_gen/test/unit/test_stage_d_vividvr_temporal_orchestration.py \
  python/sglang/multimodal_gen/test/unit/test_vividvr_acceleration_benchmark.py \
  python/sglang/multimodal_gen/test/unit/test_server_args.py \
  -q
```

预期：全部通过，无 skip 增量。

- [ ] **步骤 3：在 tmux 运行已有 decode NCCL 回归**

```bash
tmux new-session -d -s vividvr_vae_decode_transport_regression \
  'cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/logs && \
   CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=python \
   .venv/bin/python -m pytest \
   test/srt/multimodal_gen/test_cogvideox_vae_spatial_tile_parallel_distributed.py \
   -q 2>&1 | tee Vivid_Acceptance/logs/vae_decode_transport_regression_20260717.log'
```

只读查看：

```bash
tmux attach -r -t vividvr_vae_decode_transport_regression
```

预期：distributed decode transport 测试 PASS，证明共享 helper 没有改变 decode。

---

### 任务 9：运行三种 topology 的真实 bitwise 正确性硬门槛

**产物：**
- `Vivid_Acceptance/indicator/vae_encode_sp_sp2_seed42_${STAMP}.json`
- `Vivid_Acceptance/indicator/vae_encode_sp_sp4_seed42_${STAMP}.json`
- `Vivid_Acceptance/indicator/vae_encode_sp_cfg2_sp2_seed42_${STAMP}.json`
- `Vivid_Acceptance/logs/vae_encode_sp_*_${STAMP}.log`

- [ ] **步骤 1：确认 GPU 0-3 空闲并记录代码状态**

```bash
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
git status --short
git rev-parse HEAD
```

预期：0-3 没有不属于本验收的 compute 进程；工作区只含本任务明确文件和用户原有文档修改。

- [ ] **步骤 2：在 tmux 启动 SP2 验证**

```bash
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
tmux new-session -d -s vividvr_vae_encode_sp2 \
  "cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/indicator Vivid_Acceptance/logs && \
   CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=python .venv/bin/torchrun \
   --standalone --nproc-per-node=2 \
   python/sglang/multimodal_gen/tools/run_vividvr_vae_spatial_encode_validation.py \
   --topology sp2 --seed 42 --sample-frames 17 --sample-height 720 --sample-width 960 \
   --output-json Vivid_Acceptance/indicator/vae_encode_sp_sp2_seed42_${STAMP}.json \
   2>&1 | tee Vivid_Acceptance/logs/vae_encode_sp_sp2_${STAMP}.log"
```

只读查看：`tmux attach -r -t vividvr_vae_encode_sp2`。

- [ ] **步骤 3：检查 SP2 硬门槛**

```bash
.venv/bin/python - <<'PY'
import glob, json
p = sorted(glob.glob('Vivid_Acceptance/indicator/vae_encode_sp_sp2_seed42_*.json'))[-1]
d = json.load(open(p))
assert d['overall_pass'] is True
assert all(r['moments_exact'] for r in d['ranks'])
assert all(r['sampled_latents_exact'] for r in d['ranks'])
assert all(r['rank_divergent_input_comparison']['passed'] for r in d['ranks'])
assert d['noncontiguous_inputs_exercised'] is True
print(p)
PY
```

预期：无 assertion，输出唯一验收 JSON 路径。失败时停止，不进入性能验收，按 `$systematic-debugging` 定位。

- [ ] **步骤 4：串行启动并检查 SP4**

使用同一命令模板，session 改为 `vividvr_vae_encode_sp4`，`CUDA_VISIBLE_DEVICES=0,1,2,3`，`--nproc-per-node=4 --topology sp4`，文件前缀改为 `vae_encode_sp_sp4_seed42_`。检查脚本除 glob 前缀外与 SP2 相同，并额外断言每 rank `vae_encode_local_tiles_per_rank == [4,4,4,4]`（单 temporal clip 4x4 grid）。

- [ ] **步骤 5：串行启动并检查 CFG2xSP2**

使用同一命令模板，session 改为 `vividvr_vae_encode_cfg2_sp2`，四卡、`--nproc-per-node=4 --topology cfg2_sp2`，文件前缀改为 `vae_encode_sp_cfg2_sp2_seed42_`。检查 `overall_pass`、所有 exact 字段，并断言 subgroup 集合为 `[[0,1],[2,3]]`、两个 subgroup seed/output marker 不同、每个 subgroup 内一致。

- [ ] **步骤 6：记录三份 JSON 的 SHA-256**

```bash
sha256sum \
  $(ls -1t Vivid_Acceptance/indicator/vae_encode_sp_sp2_seed42_*.json | head -1) \
  $(ls -1t Vivid_Acceptance/indicator/vae_encode_sp_sp4_seed42_*.json | head -1) \
  $(ls -1t Vivid_Acceptance/indicator/vae_encode_sp_cfg2_sp2_seed42_*.json | head -1)
```

将路径和 hash 保存到任务 11 的 acceptance 文档。三个 `overall_pass` 全为 true 才能继续。

---

### 任务 10：运行三条 Treatment-only 正式推理与性能门槛

**只读 Controls：**

| Treatment | Control JSON | Control Long Clip | Treatment 最大 Long Clip | Denoise 最大值 | Decode/Trim 最大值 |
|---|---|---:|---:|---:|---:|
| `R99_VAE_ENCODE_SP` | `Vivid_Acceptance/acceleration_benchmark/vividvr_vae_sp_r99_canonicalized_v2_20260716/records/R99_VAE_SP_formal.json` | 60.905336 s | 40.603557 s | 392.607188 s | 60.705869 s |
| `R100_VAE_ENCODE_SP` | `Vivid_Acceptance/acceleration_benchmark/vividvr_vae_sp_r100_canonicalized_20260716/records/R100_VAE_SP_formal.json` | 73.079098 s | 48.719399 s | 204.713278 s | 61.984243 s |
| `R101_VAE_ENCODE_SP4` | `Vivid_Acceptance/acceleration_benchmark/vividvr_vae_sp4_fusion_20260717/records/R101_VAE_SP4_formal.json` | 64.434853 s | 25.773941 s | 209.053944 s | 30.896123 s |

- [ ] **步骤 1：记录三个 Control 的执行前指纹和 mtime**

```bash
for f in \
  Vivid_Acceptance/acceleration_benchmark/vividvr_vae_sp_r99_canonicalized_v2_20260716/records/R99_VAE_SP_formal.json \
  Vivid_Acceptance/acceleration_benchmark/vividvr_vae_sp_r100_canonicalized_20260716/records/R100_VAE_SP_formal.json \
  Vivid_Acceptance/acceleration_benchmark/vividvr_vae_sp4_fusion_20260717/records/R101_VAE_SP4_formal.json; do
  sha256sum "$f"
  stat -c '%n %Y %y' "$f"
done | tee Vivid_Acceptance/logs/vae_encode_sp_control_fingerprints_before_20260717.log
```

- [ ] **步骤 2：dry-run 三条命令**

```bash
PYTHONPATH=python .venv/bin/python \
  python/sglang/multimodal_gen/tools/run_vividvr_acceleration_benchmark.py \
  dry-run --scheme R99_VAE_ENCODE_SP \
  --control-batch-dir /home/zhiheng/sglang/Vivid_Acceptance/acceleration_benchmark/vividvr_vae_sp_r99_canonicalized_v2_20260716 \
  --gpu-ids 0,1
PYTHONPATH=python .venv/bin/python \
  python/sglang/multimodal_gen/tools/run_vividvr_acceleration_benchmark.py \
  dry-run --scheme R100_VAE_ENCODE_SP \
  --control-batch-dir /home/zhiheng/sglang/Vivid_Acceptance/acceleration_benchmark/vividvr_vae_sp_r100_canonicalized_20260716 \
  --gpu-ids 0,1,2,3
PYTHONPATH=python .venv/bin/python \
  python/sglang/multimodal_gen/tools/run_vividvr_acceleration_benchmark.py \
  dry-run --scheme R101_VAE_ENCODE_SP4 \
  --control-batch-dir /home/zhiheng/sglang/Vivid_Acceptance/acceleration_benchmark/vividvr_vae_sp4_fusion_20260717 \
  --gpu-ids 0,1,2,3
```

检查每份 report 中只有一个 Treatment、不会启动 Control、service command 同时含 `--vae-sp --vae-encode-sp`，topology 与表一致。

- [ ] **步骤 3：启动 R99 Treatment**

```bash
PYTHONPATH=python .venv/bin/python \
  python/sglang/multimodal_gen/tools/run_vividvr_acceleration_benchmark.py \
  run-one --scheme R99_VAE_ENCODE_SP \
  --batch-id vividvr_vae_encode_sp_r99_20260717 \
  --control-batch-dir /home/zhiheng/sglang/Vivid_Acceptance/acceleration_benchmark/vividvr_vae_sp_r99_canonicalized_v2_20260716 \
  --gpu-ids 0,1
```

该入口会创建 `vividvr_accel_batch_vividvr_vae_encode_sp_r99_20260717` tmux session。只读查看：

```bash
tmux attach -r -t vividvr_accel_batch_vividvr_vae_encode_sp_r99_20260717
```

预期：只执行一次 1-step warmup 和一次 20-step formal；Control 服务从未启动。

- [ ] **步骤 4：检查 R99 record 和 gate**

```bash
.venv/bin/python - <<'PY'
import json
p='Vivid_Acceptance/acceleration_benchmark/vividvr_vae_encode_sp_r99_20260717/records/R99_VAE_ENCODE_SP_formal.json'
d=json.load(open(p))
assert d['status'] in {'succeeded','quality_failed'}
assert d['runtime']['vae_sp_effective'] is True
assert d['runtime']['vae_encode_sp_effective'] is True
assert len(d['runtime']['vae_encode_sp_clips']) == 2
assert d['derived']['long_clip_preparation_speedup'] >= 1.5
assert d['derived']['model_inference_improved'] is True
assert d['derived']['denoise_regression_ratio'] <= 0.03
assert d['derived']['decode_trim_regression_ratio'] <= 0.03
assert d['derived']['performance_gates_passed'] is True
print(p)
PY
```

- [ ] **步骤 5：串行运行并检查 R100**

确认 R99 session 已退出、GPU 无遗留进程。使用任务 10 步骤 3 命令，替换为：scheme `R100_VAE_ENCODE_SP`、batch `vividvr_vae_encode_sp_r100_20260717`、R100 control parent、GPU `0,1,2,3`。attach session 与 batch 同名。检查脚本替换 record 路径和 scheme，并保持 1.5x/3% 门槛。

- [ ] **步骤 6：串行运行并检查 R101 SP4**

确认 R100 session 已退出、GPU 无遗留进程。替换为：scheme `R101_VAE_ENCODE_SP4`、batch `vividvr_vae_encode_sp4_r101_20260717`、SP4 control parent、GPU `0,1,2,3`。检查 Long Clip speedup `>=2.5`，encode world size 4、本地累计 tile counts `[8,8,8,8]`（两个 temporal clips），其余门槛相同。

- [ ] **步骤 7：核对 Control 完全未改**

重新运行步骤 1 命令输出到 `vae_encode_sp_control_fingerprints_after_20260717.log`，然后：

```bash
diff -u \
  Vivid_Acceptance/logs/vae_encode_sp_control_fingerprints_before_20260717.log \
  Vivid_Acceptance/logs/vae_encode_sp_control_fingerprints_after_20260717.log
```

预期：无 diff；三个 Treatment record 内的 Control snapshot 也与该输出一致。

- [ ] **步骤 8：计算并人工交叉检查派生值**

用独立 Python 脚本从每对 JSON 重算 `control/treatment` Long Clip、model、total speedup，以及 Denoise/Decode regression；与 record `derived` 的差绝对值必须小于 `1e-9`。记录 Treatment 总用时、模型用时、Long Clip、Denoise、Decode/Trim、GPU-seconds、质量指标和全部 gate。

任一性能 gate 失败时，验收结论为未通过；保留产物，使用 `$systematic-debugging` 分析，不修改阈值、不重跑 Control。

---

### 任务 11：更新验收与运行文档

**文件：**
- 创建：`docs_xzh/distribute/vividvr_vae_spatial_tiled_encode_parallel_acceptance_20260717.md`
- 修改：`docs_xzh/run_command/vividvr_default_run_and_serve_commands.md`
- 修改：`docs_xzh/docs_analysis/acceleration_benchmark_results_20260716.md`

- [ ] **步骤 1：编写 acceptance 文档**

文档必须包含：实现 commit 列表；三份 bitwise JSON 路径/hash 和所有 exact 结果；三个 Control path/hash/mtime 前后对比；三条 Treatment 的 batch/session/record/video/perf/compare 路径；完整指标表；四项性能 gate；质量状态说明；默认配置与服务契约未变化；回滚方式为移除 `--vae-encode-sp`。

- [ ] **步骤 2：更新命令文档**

在实验性加速章节说明：

```text
--vae-sp         仅控制 tiled decode
--vae-encode-sp  仅控制 tiled encode，要求 VAE tiling 和已初始化 SP subgroup
```

给出 SP2、CFG2xSP2、SP4 的两 flag 示例，但不把 encode flag 加进任何默认正式命令。

- [ ] **步骤 3：更新 benchmark 分析表**

追加三条行并明确 Control 是历史 decode-only record，至少列：GPU/topology、Long Clip Preparation、相对 Control stage speedup、model/total time 与 speedup、Denoise、Decode/Trim、GPU-seconds、SSIM mean/min/failed ratio、bitwise gate、performance gate。

- [ ] **步骤 4：校验文档数字来自 JSON**

用 `.venv/bin/python` 读取六个正式 JSON 和三份 correctness JSON，打印 Markdown 行；逐项与文档比对，禁止手工抄写后不复核。

- [ ] **步骤 5：提交文档**

```bash
git add \
  docs_xzh/distribute/vividvr_vae_spatial_tiled_encode_parallel_acceptance_20260717.md \
  docs_xzh/run_command/vividvr_default_run_and_serve_commands.md \
  docs_xzh/docs_analysis/acceleration_benchmark_results_20260716.md
git commit -m "docs(vividvr): record VAE encode SP acceptance"
```

`AGENTS.md` 不需要修改，因为三个正式默认配置、服务契约和标准 Phase C 命令均未改变；若实际实施改变了其中任一项，则此结论失效，必须在该任务内同步更新 `AGENTS.md` 并解释偏差。

---

### 任务 12：最终验证、变更审计与推送

**文件：**
- 审计：本计划全部变更

- [ ] **步骤 1：使用 verification-before-completion 技能**

在宣称完成前完整读取并执行 `$verification-before-completion`；该技能导致新增验证动作时，在执行记录中明确说明。

- [ ] **步骤 2：重跑最终聚焦回归**

```bash
PYTHONPATH=python .venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_cogvideox_vae_spatial_tile_encode_parallel.py \
  python/sglang/multimodal_gen/test/unit/test_cogvideox_vae_spatial_tile_parallel.py \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_attention_backend.py \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_offload.py \
  python/sglang/multimodal_gen/test/unit/test_stage_c_vividvr_single_clip.py \
  python/sglang/multimodal_gen/test/unit/test_stage_d_vividvr_temporal_orchestration.py \
  python/sglang/multimodal_gen/test/unit/test_vividvr_acceleration_benchmark.py \
  python/sglang/multimodal_gen/test/unit/test_server_args.py \
  -q
```

预期：全部通过。

- [ ] **步骤 3：审计 diff 和提交范围**

```bash
git status --short
git diff --check
git log --oneline --decorate -12
git diff --stat cb7b7afb1..HEAD
git diff --name-only cb7b7afb1..HEAD
```

预期：没有 whitespace error；没有历史 Control JSON 变更；用户原有 `2026-07-16` plan 修改未混入任何 commit；实现只覆盖第 1 节列出的文件。

- [ ] **步骤 4：核对全部硬门槛证据**

确认三份 correctness JSON `overall_pass=true`；三份 formal record `performance_gates_passed=true`；Control before/after fingerprint 无差异；每个 formal record 同时有 `total_runtime_seconds` 和 `model_inference_runtime_seconds`；输出视频、perf、compare、callback 和 service log 路径存在。

- [ ] **步骤 5：提交遗漏的仅限本任务修正**

若最终验证产生必要修正，先重跑直接相关测试，再用显式路径提交：

```bash
git status --short
# 从第 1 节文件清单逐项选择确有必要的修正文件；禁止使用目录或通配符。
git add python/sglang/multimodal_gen/runtime/models/vaes/cogvideox.py
git commit -m "fix(vividvr): finalize VAE encode SP acceptance"
```

若必要修正不在上例的 VAE 文件，必须把该 `git add` 行替换成第 1 节中对应文件的完整路径；若没有遗漏修正，则跳过本步骤，不创建空 commit。

- [ ] **步骤 6：推送当前分支**

```bash
git branch --show-current
git push origin sglang_Vivid
```

预期：当前分支确为 `sglang_Vivid`，push 成功。最终汇报列出：代码/文档文件、每个 commit、单元/NCCL/真实 bitwise/三条正式 Treatment 结果、验收是否通过、Control 未改证据和剩余风险。

## 3. 需求到任务的覆盖映射

| 设计要求 | 实施任务 |
|---|---|
| 独立开关、默认关闭、decode 含义不变 | 任务 1、6、11 |
| sample-space plan、4x4、round-robin | 任务 2 |
| temporal chunk/conv cache/quant conv 原语义 | 任务 2 |
| tensor-only subgroup gather、CFG 隔离 | 任务 2、3、5、9 |
| replicated row-major exact merge | 任务 2、3 |
| 不改变 posterior sampling | 任务 3、5 |
| 三种合法 fallback、collective 后 fatal | 任务 3 |
| encode 独立统计与逐 clip 聚合 | 任务 3、4、6 |
| moments 和 sampled latents bitwise 硬门槛 | 任务 5、9、12 |
| 只跑三个 Treatment、不重跑 Control | 任务 6、7、10 |
| Control 身份、hash、mtime 防改 | 任务 7、10、12 |
| Long Clip 1.5x/1.5x/2.5x 与 3% 门槛 | 任务 7、10 |
| Phase C/D/E、decode 和默认服务回归 | 任务 1、4、8、11、12 |
| tmux、标准验收字段和产物 | 任务 8、9、10、11 |

## 4. 停止条件

以下任一条件出现时，不得继续声称验收通过：

- 任一 topology 的 moments 或 sampled latents 不是 `torch.equal`；
- CFG2xSP2 出现跨 subgroup tensor/marker 混合；
- collective 开始后发生 silent serial fallback；
- 任一历史 Control 的 SHA-256 或 mtime 改变；
- runner 启动了 Control 服务、Control warmup 或 Control inference；
- 任一 Treatment 未达到对应 Long Clip speedup、model improvement、Denoise/Decode 3% gate；
- 新 flag 改变现有默认命令或 Phase C/D/E 行为；
- 重型验证未在 tmux 中执行或缺少日志/JSON 证据。
