# VividVR CogVideoX VAE 空间 Tile 并行实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 在不改变 VividVR 已验收的 Phase C/D/E 语义和正式默认配置的前提下，让 CogVideoX VAE 在现有 SP subgroup 内并行解码空间 tiles，并用 `R99 + vae_sp`、`R100 + vae_sp` 两个正式 treatment 验证质量与端到端收益。

**架构：** 保留 Diffusers 0.37.0 的 CogVideoX temporal decode、conv cache、tile overlap、`blend_v -> blend_h -> crop` 和 row-major merge 语义，只把空间 tile 按 `global_tile_index % sp_world_size` 分配给当前 `get_sp_group()` 的 ranks。各 rank 用固定 metadata 和 padded tensor payload 做两阶段 subgroup all-gather，随后都恢复完整 tile 集并独立执行原顺序 merge；VividVR stage 只传播开关和统计，不承载 tile 算法。正式验收继续由 benchmark runner 记录结构化结果，但服务生命周期必须是 `docs_xzh/run_command/mock_test.md` 的程序化复刻，不允许绕过 FlowCut 服务直接调用 pipeline。

**技术栈：** Python 3.10、PyTorch CUDA/NCCL distributed、Diffusers 0.37.0 `AutoencoderKLCogVideoX`、SGLang multimodal runtime、pytest、torchrun、tmux。

---

## 实施边界与验收口径

- 只修改 CogVideoX VAE；不接入 `runtime/models/vaes/common.py` 的通用 parallel tiled VAE，也不改变其他 VAE。
- 所有 collective 必须通过 `get_sp_group()` 返回的 coordinator 执行；禁止使用 WORLD group。
- `CFG=2 × SP=2` 下 `[rank0, rank1]` 与 `[rank2, rank3]` 两个 SP subgroup 独立通信，不能交换 tiles。
- `vae_sp=False`、SP world size 为 1、输入未触发 tiling 时继续使用继承的串行路径。
- `vae_sp=True` 且 tiling 关闭或 SP group 未初始化时明确失败；并行 decode 开始后不捕获异常改跑串行。
- `vae_sp` 默认值保持 `False`，不修改 `single_gpu_fa_compile`、`dual_gpu_fa_eager_compile` 或任何服务请求契约。
- 开发期只执行固定 latent 的 SP2、SP4、CFG2×SP2 正确性验证。
- 正式 `130f / 20 step` 只运行 `R99_VAE_SP` 和 `R100_VAE_SP`，直接读取历史 R99/R100 JSON 作为控制组；不重跑单卡、Phase C/D/E、SDPA、SP4 全量或关闭 `vae_sp` 的 R99/R100。
- 正式历史控制目录固定为 `Vivid_Acceptance/acceleration_benchmark/vividvr_accel_full_warmup1_20260716`。
- 正式服务验收以 `docs_xzh/run_command/mock_test.md` 的第 1、3、4、5.3、6、9 节为合同：必须经过 Moto S3、callback receiver、固定 caption sidecar mock、`sglang serve`、FlowCut POST/轮询、对象下载、compare 和有所有权边界的清理。
- 正式权重只从 `/home/zhiheng/ckpts` 加载：基础模型固定为 `/home/zhiheng/ckpts/CogVideoX1.5-5B`，VividVR 组件固定为 `/home/zhiheng/ckpts/Vivid-VR`；禁止回退到原版 `/home/zhiheng/Vivid-VR` 的运行时代码或从其他目录隐式加载同名权重。

历史控制值必须直接从原 JSON 读取；下表仅用于执行前交叉检查：

| 指标 | R99 | R100 |
| --- | ---: | ---: |
| record status | `succeeded` | `quality_failed` |
| 总耗时 | 551.119174 s | 370.881417 s |
| 模型推理耗时 | 544.320553 s | 365.066904 s |
| Denoise | 380.175751 s | 195.651912 s |
| Decode/Trim | 100.274310 s | 101.785656 s |
| SSIM mean | 0.984667384 | 0.984619328 |
| SSIM min | 0.980502773 | 0.978691849 |
| failed frame ratio | 0 | 0.015384615 |

R100 历史记录因 2/130 帧未过原阈值而标记为 `quality_failed`，但它仍是本实验唯一合法的四卡性能控制。控制读取不能把 `status == "succeeded"` 或 `quality.pass_compare is True` 当成前置条件；新 treatment 的质量判定必须同时报告原 compare 结果和相对各自历史控制是否回归。

## 正式服务验收合同

`run_vividvr_acceleration_benchmark.py run-one` 仍是正式执行入口，因为它负责配置指纹、warmup/formal 分离、显存采样、历史控制读取和标准 JSON 落盘；但 runner 的 `TmuxBenchmarkLifecycle` 与 `FlowCutRequestExecutor` 必须逐项保持下列 `mock_test.md` 服务语义，不能把它理解为另一条直接推理入口：

1. 在独立且带 batch ownership 的 tmux sessions 中启动 `moto_server`、callback receiver 和固定 caption sidecar mock；健康检查通过后才能启动主服务。
2. Moto 固定监听 `127.0.0.1:4566`，bucket 为 `flowcut`；callback 固定监听 `127.0.0.1:39090`；caption mock 固定监听 `127.0.0.1:31200`，读取已验收 sidecar：

   ```text
   /home/zhiheng/sglang/Vivid_Acceptance/captions/service_sidecars/quad-test-video-long-960x720-130f-run2-20260708T060202Z.txt
   ```

   正式性能验收禁止启动真实 CogVLM2 caption 模型，避免额外 GPU 占用和 caption 随机性；请求中不传 `prompt_file_path` 或 `caption_file_path`，由 caption bridge 按 HTTP 契约生成本次任务的 sidecar。
3. 主服务固定使用 `/home/zhiheng/sglang/.venv/bin/sglang serve`，并显式包含：

   ```text
   --model-path /home/zhiheng/ckpts/CogVideoX1.5-5B
   --component-paths.vividvr /home/zhiheng/ckpts/Vivid-VR
   --model-id VividVR
   --pipeline-class-name CogVideoXVividVRControlNetPipeline
   --attention-backend fa
   --enable-torch-compile
   --enable-cogvideox-modulation-fusion
   --cogvideox-modulation-fusion-targets transformer,controlnet
   --vividvr-caption-bridge
   --vividvr-caption-sidecar-url http://127.0.0.1:31200
   --vae-sp
   ```

4. 两条 treatment 的主服务拓扑固定如下，除 `--vae-sp` 外必须复用各自历史 control 的完整 service command 和环境：

   | treatment | GPU | topology | 必需参数 |
   | --- | ---: | --- | --- |
   | `R99_VAE_SP` | 2 | 纯 `SP=2` | `--sp-degree 2 --ulysses-degree 2 --vividvr-parallel-mode sp` |
   | `R100_VAE_SP` | 4 | `CFG=2 × SP=2` | `--sp-degree 2 --ulysses-degree 2 --enable-cfg-parallel --vividvr-parallel-mode cfg_sp` |

   四卡 treatment 不是 `SP=4`；VAE collective 只能发生在两个独立的 SP subgroup 内。
5. 每个 compile treatment 在同一个已启动服务上先提交一次 `1 step` 完整 FlowCut warmup，再提交一次 `20 step` formal；formal 请求固定为 `130f`、seed 42、temporal process frames 121，并包含必填 `callbackUrl`、`outputObjectKey`、`perf_dump_path` 与 `minioConfig`。
6. runner 必须轮询 `/v1/videos/repairs/flowcut/{taskId}/progress` 到 completed，再读取 detail URL、从 Moto S3 下载结果、运行逐帧 compare，并保留 request payload、callback JSONL、service log、perf JSON、compare JSON 和下载视频。只有这些服务证据完整时，正式 record 才有效。
7. R99 与 R100 串行执行。每条结束或失败时只清理由当前 batch ownership marker 创建的 tmux sessions；清理顺序为主服务、caption、callback、Moto，且不能 kill 用户已有的同名外部服务。

## 文件结构

### 创建

- `python/sglang/multimodal_gen/test/unit/test_cogvideox_vae_spatial_tile_parallel.py`：tile plan、分配、单 tile temporal cache、merge、transport、fallback 和统计的 CPU 单元测试。
- `test/srt/multimodal_gen/test_cogvideox_vae_spatial_tile_parallel_distributed.py`：SP2、SP4、CFG2×SP2 的真实 NCCL subgroup transport 测试。
- `python/sglang/multimodal_gen/tools/run_vividvr_vae_spatial_decode_validation.py`：加载真实 CogVideoX VAE，生成固定 latent，比较串行重复结果与并行 decoded tensor，并写标准指标 JSON。
- `docs_xzh/distribute/vividvr_vae_spatial_tile_parallel_acceptance_20260716.md`：记录轻量正确性和两条正式 treatment 的命令、产物、质量与性能结论。

### 修改

- `python/sglang/multimodal_gen/runtime/models/vaes/cogvideox.py`：实现 tile plan、round-robin、本地 tile decode、descriptor 预检、tensor gather、merge、fallback 和 CUDA event 统计。
- `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py`：VAE 加载后配置 `vae_sp`，并将 requested 值纳入 runtime acceleration debug。
- `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py`：收集单 clip/多 clip VAE 统计并写入 `vividvr_debug`。
- `python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_offload.py`：保护 decode stage 的统计采集与 CPU offload 行为。
- `python/sglang/multimodal_gen/test/unit/test_stage_d_vividvr_temporal_orchestration.py`：保护长视频逐 clip 聚合字段和既有 trim/stitch 语义。
- `python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_attention_backend.py`：保护 pipeline 初始化时 VAE 并行配置与 fail-fast。
- `python/sglang/multimodal_gen/tools/run_vividvr_acceleration_benchmark.py`：增加两个不进入默认 `run-all` 的 VAE SP treatment、历史控制读取和有效配置校验。
- `python/sglang/multimodal_gen/test/unit/test_vividvr_acceleration_benchmark.py`：保护 treatment 命令、`mock_test.md` 服务合同、`/home/zhiheng/ckpts` 权重默认值、默认矩阵隔离、历史控制派生指标和 debug 校验。
- `docs_xzh/hand_over/2026-07-16-vividvr-vae-spatial-tile-parallel-design.md`：实现完成后追加实际结果与偏差；不改已确认设计结论。
- `docs_xzh/hand_over/vividvr_vae_tile_parallel_next_handover_20260716.md`：实现完成后更新状态、提交、产物和下一步。

## 固定内部合同

核心类型和方法名在所有任务中固定如下，实施时不要另起同义接口：

```python
@dataclass(frozen=True)
class CogVideoXSpatialTile:
    global_index: int
    row_index: int
    column_index: int
    latent_top: int
    latent_left: int


@dataclass(frozen=True)
class CogVideoXSpatialTilePlan:
    tiles: tuple[CogVideoXSpatialTile, ...]
    num_rows: int
    num_columns: int
    overlap_height: int
    overlap_width: int
    blend_extent_height: int
    blend_extent_width: int
    row_limit_height: int
    row_limit_width: int


@dataclass(frozen=True)
class CogVideoXVaeSpatialDecodeStats:
    requested: bool
    effective: bool
    fallback_reason: str
    world_size: int
    group_type: str
    total_tiles: int
    local_tiles_per_rank: tuple[int, ...]
    tile_decode_seconds: float
    tile_gather_seconds: float
    tile_merge_seconds: float
    decode_seconds: float

    @classmethod
    def serial_default(cls) -> "CogVideoXVaeSpatialDecodeStats":
        return cls(
            requested=False,
            effective=False,
            fallback_reason="not_requested",
            world_size=1,
            group_type="none",
            total_tiles=0,
            local_tiles_per_rank=(0,),
            tile_decode_seconds=0.0,
            tile_gather_seconds=0.0,
            tile_merge_seconds=0.0,
            decode_seconds=0.0,
        )

    def to_debug_dict(self) -> dict[str, object]:
        return {
            "vae_sp_requested": self.requested,
            "vae_sp_effective": self.effective,
            "vae_sp_fallback_reason": self.fallback_reason,
            "vae_sp_world_size": self.world_size,
            "vae_sp_group_type": self.group_type,
            "vae_total_tiles": self.total_tiles,
            "vae_local_tiles_per_rank": list(self.local_tiles_per_rank),
            "vae_tile_decode_seconds": self.tile_decode_seconds,
            "vae_tile_gather_seconds": self.tile_gather_seconds,
            "vae_tile_merge_seconds": self.tile_merge_seconds,
            "vae_decode_seconds": self.decode_seconds,
        }
```

VAE 公共接线接口固定为：

```python
vae.configure_spatial_tile_parallel(requested: bool) -> None
vae.get_last_spatial_decode_stats() -> CogVideoXVaeSpatialDecodeStats
```

debug 字段固定为：

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
vae_sp_clips
```

长视频顶层数值字段是各 clip 的求和；`vae_sp_effective` 只有全部 clip effective 时才为 `True`，`vae_sp_fallback_reason` 在所有 clip 原因相同时取该原因，否则取 `mixed`，每个 clip 的原始记录保存在 `vae_sp_clips`。

## 任务 1：锁定 tile plan 与 round-robin 分配

**文件：**

- 创建：`python/sglang/multimodal_gen/test/unit/test_cogvideox_vae_spatial_tile_parallel.py`
- 修改：`python/sglang/multimodal_gen/runtime/models/vaes/cogvideox.py:1-54`

- [ ] **步骤 1：编写 tile plan 与分配失败测试**

```python
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.multimodal_gen.runtime.models.vaes import cogvideox

from sglang.multimodal_gen.runtime.models.vaes.cogvideox import (
    AutoencoderKLCogVideoX,
    CogVideoXSpatialTile,
    DiffusersAutoencoderKLCogVideoX,
    _assign_spatial_tiles,
    _build_spatial_tile_plan,
)


def test_tile_plan_matches_diffusers_row_major_geometry():
    plan = _build_spatial_tile_plan(
        latent_height=65,
        latent_width=97,
        tile_latent_min_height=30,
        tile_latent_min_width=45,
        tile_sample_min_height=240,
        tile_sample_min_width=360,
        tile_overlap_factor_height=1 / 6,
        tile_overlap_factor_width=1 / 5,
    )
    assert plan.overlap_height == 25
    assert plan.overlap_width == 36
    assert plan.blend_extent_height == 40
    assert plan.blend_extent_width == 72
    assert plan.row_limit_height == 200
    assert plan.row_limit_width == 288
    assert [(tile.latent_top, tile.latent_left) for tile in plan.tiles] == [
        (top, left)
        for top in range(0, 65, 25)
        for left in range(0, 97, 36)
    ]
    assert [tile.global_index for tile in plan.tiles] == list(range(9))


@pytest.mark.parametrize(
    ("total_tiles", "world_size", "expected"),
    [
        (2, 4, ((0,), (1,), (), ())),
        (4, 4, ((0,), (1,), (2,), (3,))),
        (7, 3, ((0, 3, 6), (1, 4), (2, 5))),
    ],
)
def test_round_robin_assignment_is_complete_and_balanced(
    total_tiles, world_size, expected
):
    tiles = tuple(
        CogVideoXSpatialTile(index, 0, index, 0, index)
        for index in range(total_tiles)
    )
    actual = tuple(
        tuple(
            tile.global_index
            for tile in _assign_spatial_tiles(tiles, rank, world_size)
        )
        for rank in range(world_size)
    )
    assert actual == expected
    assert sorted(index for rank_tiles in actual for index in rank_tiles) == list(
        range(total_tiles)
    )
```

- [ ] **步骤 2：运行测试并确认接口不存在**

运行：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_cogvideox_vae_spatial_tile_parallel.py \
  -k 'tile_plan or round_robin' -q
```

预期：collection FAIL，错误包含 `cannot import name '_build_spatial_tile_plan'`。

- [ ] **步骤 3：实现不可变 tile plan 与分配纯函数**

在 `cogvideox.py` 中加入本计划“固定内部合同”的两个 dataclass，并实现：

```python
def _build_spatial_tile_plan(
    *,
    latent_height: int,
    latent_width: int,
    tile_latent_min_height: int,
    tile_latent_min_width: int,
    tile_sample_min_height: int,
    tile_sample_min_width: int,
    tile_overlap_factor_height: float,
    tile_overlap_factor_width: float,
) -> CogVideoXSpatialTilePlan:
    overlap_height = int(
        tile_latent_min_height * (1 - tile_overlap_factor_height)
    )
    overlap_width = int(
        tile_latent_min_width * (1 - tile_overlap_factor_width)
    )
    if overlap_height <= 0 or overlap_width <= 0:
        raise ValueError("CogVideoX VAE tile overlap stride must be positive")
    blend_extent_height = int(
        tile_sample_min_height * tile_overlap_factor_height
    )
    blend_extent_width = int(tile_sample_min_width * tile_overlap_factor_width)
    coordinates = [
        (row_index, column_index, top, left)
        for row_index, top in enumerate(range(0, latent_height, overlap_height))
        for column_index, left in enumerate(range(0, latent_width, overlap_width))
    ]
    num_rows = len(range(0, latent_height, overlap_height))
    num_columns = len(range(0, latent_width, overlap_width))
    return CogVideoXSpatialTilePlan(
        tiles=tuple(
            CogVideoXSpatialTile(index, row, column, top, left)
            for index, (row, column, top, left) in enumerate(coordinates)
        ),
        num_rows=num_rows,
        num_columns=num_columns,
        overlap_height=overlap_height,
        overlap_width=overlap_width,
        blend_extent_height=blend_extent_height,
        blend_extent_width=blend_extent_width,
        row_limit_height=tile_sample_min_height - blend_extent_height,
        row_limit_width=tile_sample_min_width - blend_extent_width,
    )


def _assign_spatial_tiles(
    tiles: tuple[CogVideoXSpatialTile, ...], rank: int, world_size: int
) -> tuple[CogVideoXSpatialTile, ...]:
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError(f"invalid SP rank/world size: {rank}/{world_size}")
    return tuple(
        tile for tile in tiles if tile.global_index % world_size == rank
    )
```

实际 tiled decode 固定调用 `_assign_spatial_tiles(plan.tiles, rank, world_size)`，返回对象保留完整 row/column/latent 坐标。

- [ ] **步骤 4：运行定向测试**

运行步骤 2 的命令。预期：5 个参数化 case 全部 PASS。

- [ ] **步骤 5：提交 tile plan**

```bash
git add \
  python/sglang/multimodal_gen/runtime/models/vaes/cogvideox.py \
  python/sglang/multimodal_gen/test/unit/test_cogvideox_vae_spatial_tile_parallel.py
git commit -m "feat(vividvr): define CogVideoX VAE spatial tile plan"
```

## 任务 2：锁定单 tile temporal cache 与原顺序 merge

**文件：**

- 修改：`python/sglang/multimodal_gen/runtime/models/vaes/cogvideox.py`
- 测试：`python/sglang/multimodal_gen/test/unit/test_cogvideox_vae_spatial_tile_parallel.py`

- [ ] **步骤 1：增加 temporal cache 生命周期失败测试**

```python
import torch

from sglang.multimodal_gen.runtime.models.vaes.cogvideox import (
    CogVideoXSpatialTile,
    _decode_one_spatial_tile,
)


class RecordingDecoder:
    def __init__(self):
        self.received_cache = []

    def __call__(self, tensor, *, conv_cache):
        self.received_cache.append(conv_cache)
        next_cache = f"cache-{len(self.received_cache)}"
        return tensor + 10, next_cache


def test_one_spatial_tile_keeps_cache_only_across_its_temporal_batches():
    decoder = RecordingDecoder()
    vae = SimpleNamespace(
        num_latent_frames_batch_size=2,
        tile_latent_min_height=3,
        tile_latent_min_width=4,
        post_quant_conv=None,
        decoder=decoder,
    )
    z = torch.arange(1 * 1 * 5 * 5 * 6).reshape(1, 1, 5, 5, 6).float()
    tile = CogVideoXSpatialTile(0, 0, 0, 1, 1)

    first = _decode_one_spatial_tile(vae, z, tile)
    second = _decode_one_spatial_tile(vae, z, tile)

    assert first.shape == (1, 1, 5, 3, 4)
    assert torch.equal(first, second)
    assert decoder.received_cache[:2] == [None, "cache-1"]
    assert decoder.received_cache[2:] == [None, "cache-3"]
```

- [ ] **步骤 2：增加非对称 merge 顺序失败测试**

```python
from sglang.multimodal_gen.runtime.models.vaes.cogvideox import (
    _merge_spatial_tiles,
)


def test_merge_is_row_major_vertical_then_horizontal_then_crop():
    calls = []

    class BlendVAE:
        @staticmethod
        def blend_v(above, current, extent):
            calls.append(("v", int(above.flatten()[0]), int(current.flatten()[0]), extent))
            current.add_(100)
            return current

        @staticmethod
        def blend_h(left, current, extent):
            calls.append(("h", int(left.flatten()[0]), int(current.flatten()[0]), extent))
            current.add_(1000)
            return current

    plan = make_two_by_two_plan(row_limit_height=1, row_limit_width=1)
    tiles = {
        index: torch.full((1, 1, 1, 2, 2), float(index + 1))
        for index in range(4)
    }
    actual = _merge_spatial_tiles(BlendVAE(), plan, tiles)

    assert calls == [
        ("h", 1, 2, plan.blend_extent_width),
        ("v", 1, 3, plan.blend_extent_height),
        ("v", 1002, 4, plan.blend_extent_height),
        ("h", 103, 104, plan.blend_extent_width),
    ]
    assert actual.shape == (1, 1, 1, 2, 2)
```

`make_two_by_two_plan` 在测试文件中返回四个 row-major tile 的真实 `CogVideoXSpatialTilePlan`，所有 blend extent 设为 1。

- [ ] **步骤 3：确认两个内部函数尚不存在**

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_cogvideox_vae_spatial_tile_parallel.py \
  -k 'one_spatial_tile or merge_is_row_major' -q
```

预期：FAIL，错误包含 `_decode_one_spatial_tile` 或 `_merge_spatial_tiles` 无法导入。

- [ ] **步骤 4：镜像 Diffusers temporal loop**

```python
def _decode_one_spatial_tile(vae, z, tile: CogVideoXSpatialTile) -> torch.Tensor:
    frame_batch_size = vae.num_latent_frames_batch_size
    num_frames = z.shape[2]
    num_batches = max(num_frames // frame_batch_size, 1)
    conv_cache = None
    temporal_parts = []
    for batch_index in range(num_batches):
        remaining_frames = num_frames % frame_batch_size
        start_frame = frame_batch_size * batch_index + (
            0 if batch_index == 0 else remaining_frames
        )
        end_frame = frame_batch_size * (batch_index + 1) + remaining_frames
        decoded = z[
            :,
            :,
            start_frame:end_frame,
            tile.latent_top : tile.latent_top + vae.tile_latent_min_height,
            tile.latent_left : tile.latent_left + vae.tile_latent_min_width,
        ]
        if vae.post_quant_conv is not None:
            decoded = vae.post_quant_conv(decoded)
        decoded, conv_cache = vae.decoder(decoded, conv_cache=conv_cache)
        temporal_parts.append(decoded)
    return torch.cat(temporal_parts, dim=2)
```

- [ ] **步骤 5：实现严格 row-major merge**

```python
def _merge_spatial_tiles(vae, plan, decoded_tiles):
    rows = [
        [decoded_tiles[row * plan.num_columns + column] for column in range(plan.num_columns)]
        for row in range(plan.num_rows)
    ]
    result_rows = []
    for row_index, row in enumerate(rows):
        result_row = []
        for column_index, tile in enumerate(row):
            if row_index > 0:
                tile = vae.blend_v(
                    rows[row_index - 1][column_index],
                    tile,
                    plan.blend_extent_height,
                )
            if column_index > 0:
                tile = vae.blend_h(
                    row[column_index - 1], tile, plan.blend_extent_width
                )
            result_row.append(
                tile[
                    :, :, :, : plan.row_limit_height, : plan.row_limit_width
                ]
            )
        result_rows.append(torch.cat(result_row, dim=4))
    return torch.cat(result_rows, dim=3)
```

- [ ] **步骤 6：运行定向与已有 VAE toy 测试**

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_cogvideox_vae_spatial_tile_parallel.py \
  python/sglang/multimodal_gen/test/unit/test_stage_b_vividvr_components.py \
  -q
```

预期：全部 PASS，toy encode/decode shape 不变。

- [ ] **步骤 7：提交 tile decode 与 merge**

```bash
git add \
  python/sglang/multimodal_gen/runtime/models/vaes/cogvideox.py \
  python/sglang/multimodal_gen/test/unit/test_cogvideox_vae_spatial_tile_parallel.py
git commit -m "feat(vividvr): preserve CogVideoX tiled decode semantics"
```

## 任务 3：实现 descriptor 预检与变长 tensor all-gather

**文件：**

- 修改：`python/sglang/multimodal_gen/runtime/models/vaes/cogvideox.py`
- 测试：`python/sglang/multimodal_gen/test/unit/test_cogvideox_vae_spatial_tile_parallel.py`

- [ ] **步骤 1：增加 metadata/payload round-trip 失败测试**

```python
def test_unpack_tiles_handles_boundary_shapes_and_empty_rank():
    rank_tiles = (
        {
            0: torch.arange(8, dtype=torch.float32).reshape(1, 1, 1, 2, 4),
            3: torch.arange(6, dtype=torch.float32).reshape(1, 1, 1, 2, 3),
        },
        {
            1: torch.arange(4, dtype=torch.float32).reshape(1, 1, 1, 1, 4),
            2: torch.arange(3, dtype=torch.float32).reshape(1, 1, 1, 1, 3),
        },
        {},
    )
    gathered_metadata, gathered_payload = simulate_fixed_tensor_gather(
        rank_tiles, total_tiles=4
    )
    recovered = _unpack_gathered_tiles(
        gathered_metadata, gathered_payload, total_tiles=4
    )

    assert tuple(recovered) == (0, 1, 2, 3)
    assert recovered[3].shape == (1, 1, 1, 2, 3)


@pytest.mark.parametrize(
    ("metadata_payload", "match"),
    [
        (simulate_duplicate_index_gather(), "duplicate.*global tile index"),
        (simulate_missing_index_gather(), "missing.*global tile index"),
    ],
)
def test_unpack_tiles_rejects_duplicate_or_missing_global_index(
    metadata_payload, match
):
    metadata, payload = metadata_payload
    with pytest.raises(RuntimeError, match=match):
        _unpack_gathered_tiles(metadata, payload, total_tiles=2)
```

测试 helper 只用 `torch.stack` 模拟 coordinator 返回值，不启动进程。

- [ ] **步骤 2：增加 descriptor 不一致失败测试**

```python
def test_descriptor_preflight_rejects_rank_mismatch_before_payload_gather():
    group = FakeGroup(
        world_size=2,
        rank_in_group=0,
        gathered_descriptors=torch.tensor(
            [[1, 16, 5, 90, 120, 2, 30, 45, 25, 36, 2, 9, 2],
             [1, 16, 5, 90, 121, 2, 30, 45, 25, 36, 2, 9, 2]],
            dtype=torch.int64,
        ),
    )
    with pytest.raises(RuntimeError, match="SP input descriptor mismatch"):
        _validate_spatial_decode_descriptor(
            group, torch.zeros(1, 16, 5, 90, 120), make_plan_3x3()
        )
    assert group.all_gather_calls == 1
```

- [ ] **步骤 3：运行测试并确认 transport 接口不存在**

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_cogvideox_vae_spatial_tile_parallel.py \
  -k 'unpack_tiles or descriptor_preflight' -q
```

预期：FAIL，缺少 `_unpack_gathered_tiles` 和 `_validate_spatial_decode_descriptor`。

- [ ] **步骤 4：实现固定 descriptor**

descriptor 使用 `torch.int64` CUDA tensor，字段顺序固定为：

```text
B,C,T,H,W,dtype_code,
tile_latent_min_height,tile_latent_min_width,
overlap_height,overlap_width,
num_latent_frames_batch_size,total_tiles,sp_world_size
```

dtype code 固定为 `float16=1`、`bfloat16=2`、`float32=3`；其他 dtype 明确报错。实现核心比较：

```python
def _validate_spatial_decode_descriptor(sp_group, z, plan) -> None:
    local = _build_spatial_decode_descriptor(z, plan, sp_group.world_size)
    gathered = sp_group.all_gather(local.unsqueeze(0), dim=0)
    reference = gathered[0]
    mismatch_ranks = [
        rank
        for rank in range(sp_group.world_size)
        if not torch.equal(gathered[rank], reference)
    ]
    if mismatch_ranks:
        raise RuntimeError(
            "CogVideoX VAE SP input descriptor mismatch on ranks "
            f"{mismatch_ranks}"
        )
```

- [ ] **步骤 5：实现固定 metadata 和 padded payload**

metadata 每 slot 七列：`global_index,batch,channels,frames,height,width,numel`。空 slot 全零后把 `global_index` 设为 `-1`。

```python
def _all_gather_decoded_tiles(
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
```

生产调用固定传入首个本地 decoded tile 的 dtype/device；若当前 rank 没有 tile，则传入输入 latent 的 dtype/device，并断言所有非空 tile 与之相同。这样 `total_tiles < sp_world_size` 时空 rank 也能创建合法的零长度 payload。`_unpack_gathered_tiles` 必须逐 source rank、逐 slot 累加该 rank 的 payload offset，用 `narrow(offset, numel).view(shape)` 恢复 view，并验证 `numel == batch*channels*frames*height*width`、index 唯一且最终集合严格等于 `range(total_tiles)`。

- [ ] **步骤 6：运行完整纯逻辑测试**

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_cogvideox_vae_spatial_tile_parallel.py -q
```

预期：全部 PASS，且测试中搜索不到 `all_gather_object`。

- [ ] **步骤 7：提交 transport**

```bash
git add \
  python/sglang/multimodal_gen/runtime/models/vaes/cogvideox.py \
  python/sglang/multimodal_gen/test/unit/test_cogvideox_vae_spatial_tile_parallel.py
git commit -m "feat(vividvr): gather CogVideoX VAE tiles inside SP groups"
```

## 任务 4：接入 VAE dispatch、fallback 与 CUDA event 统计

**文件：**

- 修改：`python/sglang/multimodal_gen/runtime/models/vaes/cogvideox.py`
- 修改：`python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py:965-979`
- 测试：`python/sglang/multimodal_gen/test/unit/test_cogvideox_vae_spatial_tile_parallel.py`
- 测试：`python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_attention_backend.py`

- [ ] **步骤 1：增加 activation/fallback 失败测试**

```python
@pytest.mark.parametrize(
    ("requested", "world_size", "is_tiled", "expected_reason"),
    [
        (False, 2, True, "not_requested"),
        (True, 1, True, "sp_world_size_one"),
        (True, 2, False, "input_below_tiling_threshold"),
    ],
)
def test_decode_dispatch_uses_only_declared_fallbacks(
    requested, world_size, is_tiled, expected_reason, monkeypatch
):
    vae = make_toy_runtime_vae(use_tiling=True)
    monkeypatch.setattr(cogvideox, "get_sp_group", lambda: FakeGroup(world_size))
    vae.configure_spatial_tile_parallel(requested=requested)
    z = make_latent(trigger_tiling=is_tiled)

    vae._decode(z)

    stats = vae.get_last_spatial_decode_stats()
    assert stats.effective is False
    assert stats.fallback_reason == expected_reason


def test_requested_parallel_rejects_disabled_tiling():
    vae = make_toy_runtime_vae(use_tiling=False)
    with pytest.raises(ValueError, match="vae_sp requires VAE tiling"):
        vae.configure_spatial_tile_parallel(requested=True)


def test_requested_parallel_rejects_uninitialized_sp_group(monkeypatch):
    vae = make_toy_runtime_vae(use_tiling=True)
    monkeypatch.setattr(
        cogvideox,
        "get_sp_group",
        lambda: (_ for _ in ()).throw(AssertionError("not initialized")),
    )
    with pytest.raises(RuntimeError, match="SP group is not initialized"):
        vae.configure_spatial_tile_parallel(requested=True)
```

- [ ] **步骤 2：增加 parallel integration 与 no-retry 失败测试**

```python
def test_parallel_tiled_decode_decodes_only_owned_tiles_and_merges_all(monkeypatch):
    vae = make_toy_runtime_vae(use_tiling=True)
    group = ScriptedTwoRankGroup(rank_in_group=0)
    monkeypatch.setattr(cogvideox, "get_sp_group", lambda: group)
    vae.configure_spatial_tile_parallel(requested=True)

    actual = vae.tiled_decode(make_three_tile_latent()).sample

    assert vae.decoder.decoded_tile_indices == [0, 2]
    assert actual.shape == serial_expected_shape()
    assert vae.get_last_spatial_decode_stats().effective is True


def test_parallel_failure_is_not_retried_serially(monkeypatch):
    vae = make_toy_runtime_vae(use_tiling=True)
    monkeypatch.setattr(cogvideox, "get_sp_group", lambda: FakeGroup(2))
    monkeypatch.setattr(
        cogvideox,
        "_all_gather_decoded_tiles",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("NCCL failed")),
    )
    serial = Mock()
    monkeypatch.setattr(DiffusersAutoencoderKLCogVideoX, "tiled_decode", serial)
    vae.configure_spatial_tile_parallel(requested=True)

    with pytest.raises(RuntimeError, match="NCCL failed"):
        vae.tiled_decode(make_three_tile_latent())
    serial.assert_not_called()
```

- [ ] **步骤 3：运行测试并确认 dispatch 尚未接入**

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_cogvideox_vae_spatial_tile_parallel.py \
  -k 'dispatch or requested_parallel or parallel_tiled or not_retried' -q
```

预期：FAIL，缺少 `configure_spatial_tile_parallel` 或统计接口。

- [ ] **步骤 4：初始化 runtime 状态并实现显式配置**

在 VAE `__init__` 末尾加入：

```python
self._vae_sp_requested = False
self._vae_sp_group = None
self._last_spatial_decode_stats = CogVideoXVaeSpatialDecodeStats.serial_default()
```

实现：

```python
def configure_spatial_tile_parallel(self, requested: bool) -> None:
    requested = bool(requested)
    if requested and not self.use_tiling:
        raise ValueError("CogVideoX vae_sp requires VAE tiling to be enabled")
    group = None
    if requested:
        try:
            group = get_sp_group()
        except AssertionError as error:
            raise RuntimeError(
                "CogVideoX vae_sp requested but SP group is not initialized"
            ) from error
    self._vae_sp_requested = requested
    self._vae_sp_group = group
```

- [ ] **步骤 5：覆盖 `_decode` 并完整计时非 tiled 串行路径**

```python
def _decode(self, z: torch.Tensor, return_dict: bool = True):
    tiled = self.use_tiling and (
        z.shape[-1] > self.tile_latent_min_width
        or z.shape[-2] > self.tile_latent_min_height
    )
    if tiled:
        return super()._decode(z, return_dict=return_dict)
    reason = (
        "input_below_tiling_threshold"
        if self._vae_sp_requested
        else "not_requested"
    )
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    decoded = super()._decode(z, return_dict=return_dict)
    end.record()
    end.synchronize()
    self._set_serial_decode_stats(
        reason,
        world_size=(self._vae_sp_group.world_size if self._vae_sp_group else 1),
        decode_seconds=start.elapsed_time(end) / 1000.0,
    )
    return decoded
```

这里不能复制非 tiled temporal loop；继续调用 Diffusers `_decode`，避免改变继承路径。正式 runtime 的 latent 必须在 CUDA；CPU 单元测试通过 monkeypatch `torch.cuda.Event` 的 fake event 验证字段，不为生产路径增加 CPU timer 分支。

- [ ] **步骤 6：覆盖 `tiled_decode` 并实现 parallel 主路径**

```python
def tiled_decode(self, z: torch.Tensor, return_dict: bool = True):
    if not self._vae_sp_requested:
        return self._serial_tiled_decode_with_stats(
            z, reason="not_requested", world_size=1, return_dict=return_dict
        )
    sp_group = self._vae_sp_group
    if sp_group is None:
        raise RuntimeError("CogVideoX VAE SP group is unavailable")
    if sp_group.world_size == 1:
        return self._serial_tiled_decode_with_stats(
            z,
            reason="sp_world_size_one",
            world_size=1,
            return_dict=return_dict,
        )
    return self._parallel_spatial_tiled_decode(
        z, sp_group=sp_group, return_dict=return_dict
    )
```

`_serial_tiled_decode_with_stats` 用一对 CUDA event 包住 `super().tiled_decode(...)`，只同步 end event，并把三个并行子阶段写成 `0.0`。`_parallel_spatial_tiled_decode` 固定按以下次序执行：build plan → descriptor all-gather/compare → decode 当前 rank 的 round-robin tiles → metadata gather → payload gather → row-major merge → `DecoderOutput`/tuple。任何一步异常直接抛出。

- [ ] **步骤 7：实现单次同步的 CUDA event 统计**

在 parallel 主路径依次 `record()`：`decode_start`、`tile_start`、`gather_start`、`merge_start`、`decode_end`。只调用一次 `decode_end.synchronize()`，随后读取：

```python
tile_decode_seconds = tile_start.elapsed_time(gather_start) / 1000.0
tile_gather_seconds = gather_start.elapsed_time(merge_start) / 1000.0
tile_merge_seconds = merge_start.elapsed_time(decode_end) / 1000.0
decode_seconds = decode_start.elapsed_time(decode_end) / 1000.0
```

用 metadata 计算 `local_tiles_per_rank`，不要额外 collective。serial fallback 同样由 CUDA event 记录 `decode_seconds`，三个并行子阶段置为 `0.0`。`get_last_spatial_decode_stats()` 只返回 `_last_spatial_decode_stats` 的不可变 dataclass；每次 decode（包括 `not_requested`）都必须覆盖旧值，禁止泄漏上一 clip 的统计。

- [ ] **步骤 8：pipeline 初始化后配置 VAE**

在 `vividvr_pipeline.py` 第一次 `vae = self.get_module("vae")` 后、创建 `VideoProcessor` 前加入：

```python
configure_vae_sp = getattr(vae, "configure_spatial_tile_parallel", None)
if configure_vae_sp is None and server_args.pipeline_config.vae_sp:
    raise TypeError("VividVR vae_sp requires the native CogVideoX VAE runtime")
if configure_vae_sp is not None:
    configure_vae_sp(requested=bool(server_args.pipeline_config.vae_sp))
```

在 pipeline 测试中用 mock VAE 断言 requested `True/False` 原样传递，并断言请求开启但 VAE 无接口时 fail-fast。

- [ ] **步骤 9：运行 VAE 与 pipeline 定向测试**

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_cogvideox_vae_spatial_tile_parallel.py \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_attention_backend.py \
  -q
```

预期：全部 PASS。

- [ ] **步骤 10：提交 VAE runtime 接入**

```bash
git add \
  python/sglang/multimodal_gen/runtime/models/vaes/cogvideox.py \
  python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py \
  python/sglang/multimodal_gen/test/unit/test_cogvideox_vae_spatial_tile_parallel.py \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_attention_backend.py
git commit -m "feat(vividvr): enable CogVideoX VAE tile parallel decode"
```

## 任务 5：接入单 clip 与长视频可观测性

**文件：**

- 修改：`python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py:1374-1419`
- 修改：`python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py:1777-1822`
- 修改：`python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py` 的 `_build_runtime_acceleration_debug`
- 测试：`python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_offload.py`
- 测试：`python/sglang/multimodal_gen/test/unit/test_stage_d_vividvr_temporal_orchestration.py`

- [ ] **步骤 1：扩展 dummy VAE 并增加单 clip 失败测试**

```python
class _DummyDecodeVAE(torch.nn.Module):
    # 保留现有初始化与 decode
    def get_last_spatial_decode_stats(self):
        return SimpleNamespace(
            to_debug_dict=lambda: {
                "vae_sp_requested": True,
                "vae_sp_effective": True,
                "vae_sp_fallback_reason": "effective",
                "vae_sp_world_size": 2,
                "vae_sp_group_type": "sp",
                "vae_total_tiles": 9,
                "vae_local_tiles_per_rank": [5, 4],
                "vae_tile_decode_seconds": 1.2,
                "vae_tile_gather_seconds": 0.2,
                "vae_tile_merge_seconds": 0.1,
                "vae_decode_seconds": 1.5,
            }
        )


def test_decode_stage_exposes_last_vae_spatial_stats(self):
    stage = VividVRDecodingStage(vae=_DummyDecodeVAE())
    stage.decode_latents(torch.zeros(1, 3, 16, 4, 4), 0, make_server_args())
    assert stage.last_vae_decode_stats["vae_sp_effective"] is True
    assert stage.last_vae_decode_stats["vae_local_tiles_per_rank"] == [5, 4]
```

- [ ] **步骤 2：增加多 clip 聚合失败测试**

```python
def test_multi_clip_decode_aggregates_parallel_stats_without_changing_trim():
    stage, batch = make_two_clip_decode_stage_and_batch(
        clip_stats=[
            make_vae_stats(total_tiles=9, local=[5, 4], decode=1.5),
            make_vae_stats(total_tiles=6, local=[3, 3], decode=1.0),
        ]
    )
    stage.forward(batch, make_server_args())
    debug = batch.extra["vividvr_debug"]

    assert debug["vae_sp_requested"] is True
    assert debug["vae_sp_effective"] is True
    assert debug["vae_total_tiles"] == 15
    assert debug["vae_local_tiles_per_rank"] == [8, 7]
    assert debug["vae_decode_seconds"] == pytest.approx(2.5)
    assert len(debug["vae_sp_clips"]) == 2
    assert len(batch.extra["vividvr_long_video_runtime"]["trimmed_clips"]) == 2
```

- [ ] **步骤 3：运行测试并确认统计未传播**

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_offload.py \
  python/sglang/multimodal_gen/test/unit/test_stage_d_vividvr_temporal_orchestration.py \
  -k 'vae_spatial or aggregates_parallel_stats' -q
```

预期：FAIL，`last_vae_decode_stats` 或 debug 字段不存在。

- [ ] **步骤 4：decode stage 保存最近一次不可变快照**

在 `VividVRDecodingStage.__init__` 初始化空字典，并在 `self.vae.decode(...)` 返回后、CPU offload 前复制：

```python
stats_getter = getattr(self.vae, "get_last_spatial_decode_stats", None)
self.last_vae_decode_stats = (
    dict(stats_getter().to_debug_dict()) if stats_getter is not None else {}
)
```

单 clip `forward` 用 `debug.update(self.last_vae_decode_stats)`，并设置 `debug["vae_sp_clips"] = [dict(...)]`。现有 `vae_tiling_enabled` 字段继续保留。

- [ ] **步骤 5：实现长视频聚合纯函数**

```python
def _aggregate_vae_spatial_decode_stats(
    clip_stats: list[dict[str, object]],
) -> dict[str, object]:
    if not clip_stats:
        return {}
    topologies = {
        (
            int(item["vae_sp_world_size"]),
            str(item["vae_sp_group_type"]),
            len(item["vae_local_tiles_per_rank"]),
        )
        for item in clip_stats
    }
    if len(topologies) != 1:
        raise RuntimeError(f"VAE SP clip topology mismatch: {sorted(topologies)}")
    reasons = {str(item["vae_sp_fallback_reason"]) for item in clip_stats}
    local_width = max(len(item["vae_local_tiles_per_rank"]) for item in clip_stats)
    local_totals = [0] * local_width
    for item in clip_stats:
        for rank, count in enumerate(item["vae_local_tiles_per_rank"]):
            local_totals[rank] += int(count)
    return {
        "vae_sp_requested": all(bool(item["vae_sp_requested"]) for item in clip_stats),
        "vae_sp_effective": all(bool(item["vae_sp_effective"]) for item in clip_stats),
        "vae_sp_fallback_reason": next(iter(reasons)) if len(reasons) == 1 else "mixed",
        "vae_sp_world_size": int(clip_stats[0]["vae_sp_world_size"]),
        "vae_sp_group_type": str(clip_stats[0]["vae_sp_group_type"]),
        "vae_total_tiles": sum(int(item["vae_total_tiles"]) for item in clip_stats),
        "vae_local_tiles_per_rank": local_totals,
        "vae_tile_decode_seconds": sum(float(item["vae_tile_decode_seconds"]) for item in clip_stats),
        "vae_tile_gather_seconds": sum(float(item["vae_tile_gather_seconds"]) for item in clip_stats),
        "vae_tile_merge_seconds": sum(float(item["vae_tile_merge_seconds"]) for item in clip_stats),
        "vae_decode_seconds": sum(float(item["vae_decode_seconds"]) for item in clip_stats),
        "vae_sp_clips": [dict(item) for item in clip_stats],
    }
```

多 clip loop 每次 `decode_latents` 后立即复制 `decoding_stage.last_vae_decode_stats`，loop 结束后调用该函数并更新 debug。

- [ ] **步骤 6：运行两个完整 stage 测试文件**

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_offload.py \
  python/sglang/multimodal_gen/test/unit/test_stage_d_vividvr_temporal_orchestration.py \
  -q
```

预期：全部 PASS，既有 `vae_tiling_enabled`、trim、stitch 断言不变。

- [ ] **步骤 7：提交可观测性**

```bash
git add \
  python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py \
  python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_offload.py \
  python/sglang/multimodal_gen/test/unit/test_stage_d_vividvr_temporal_orchestration.py
git commit -m "feat(vividvr): report VAE tile parallel decode metrics"
```

## 任务 6：增加两条隔离的正式 benchmark treatment

**文件：**

- 修改：`python/sglang/multimodal_gen/tools/run_vividvr_acceleration_benchmark.py:61-2501`
- 测试：`python/sglang/multimodal_gen/test/unit/test_vividvr_acceleration_benchmark.py`

- [ ] **步骤 1：增加 treatment registry、服务合同和命令失败测试**

```python
from dataclasses import replace
from pathlib import Path

from sglang.multimodal_gen.tools.run_vividvr_acceleration_benchmark import (
    ALL_SCHEMES,
    BenchmarkConfig,
    RunRole,
    SCHEMES,
    VAE_SP_TREATMENTS,
    build_request_payload,
    build_service_command,
)


def test_vae_sp_treatments_do_not_expand_default_run_all_matrix():
    assert list(VAE_SP_TREATMENTS) == ["R99_VAE_SP", "R100_VAE_SP"]
    assert list(SCHEMES)[-2:] == ["R99", "R100"]
    assert "R99_VAE_SP" not in SCHEMES
    assert ALL_SCHEMES["R99_VAE_SP"].controls == ("R99",)
    assert ALL_SCHEMES["R100_VAE_SP"].controls == ("R100",)


@pytest.mark.parametrize("scheme_id", ["R99_VAE_SP", "R100_VAE_SP"])
def test_vae_sp_treatment_adds_only_vae_sp_to_control_command(tmp_path, scheme_id):
    treatment = ALL_SCHEMES[scheme_id]
    control = SCHEMES[treatment.controls[0]]
    treatment_command = build_service_command(treatment, make_config(tmp_path))
    control_command = build_service_command(control, make_config(tmp_path))
    assert treatment_command == control_command + ["--vae-sp"]


def test_vae_sp_formal_defaults_follow_mock_test_service_contract():
    config = BenchmarkConfig()
    assert config.model_path == Path("/home/zhiheng/ckpts/CogVideoX1.5-5B")
    assert config.vividvr_path == Path("/home/zhiheng/ckpts/Vivid-VR")
    assert config.service_port == 31221
    assert config.caption_port == 31200
    assert config.callback_port == 39090
    assert config.s3_port == 4566
    assert config.s3_bucket == "flowcut"

    r99 = build_service_command(ALL_SCHEMES["R99_VAE_SP"], config)
    assert r99[r99.index("--model-path") + 1] == str(config.model_path)
    assert r99[r99.index("--component-paths.vividvr") + 1] == str(
        config.vividvr_path
    )
    assert "--vividvr-caption-bridge" in r99
    assert "--vae-sp" in r99


def test_vae_sp_formal_request_keeps_flowcut_contract(tmp_path):
    config = replace(BenchmarkConfig(), output_root=tmp_path)
    payload = build_request_payload(
        config,
        role=RunRole.FORMAL,
        task_id="r99-vae-sp-formal",
        callback_url="http://127.0.0.1:39090/tasks/r99/callback",
        output_path=tmp_path / "service-output.mp4",
        perf_path=tmp_path / "perf.json",
    )
    assert payload["num_inference_steps"] == 20
    assert payload["seed"] == 42
    assert payload["num_temporal_process_frames"] == 121
    assert payload["callbackUrl"].startswith("http://127.0.0.1:39090/")
    assert payload["minioConfig"]["endpoint"] == "127.0.0.1:4566"
    assert "caption_file_path" not in payload
    assert "prompt_file_path" not in payload
```

- [ ] **步骤 2：增加 effective debug 失败测试**

```python
def test_validate_effective_config_requires_effective_vae_sp_for_treatment():
    perf = make_perf_fixture(modulation_fusion=True, vae_sp=True)
    validated = validate_effective_config(ALL_SCHEMES["R99_VAE_SP"], perf)
    assert validated["vae_sp_effective"] is True
    assert validated["vae_sp_world_size"] == 2


def test_validate_effective_config_rejects_vae_sp_silent_fallback():
    perf = make_perf_fixture(modulation_fusion=True, vae_sp=True)
    perf["meta"]["vividvr_debug"]["vae_sp_effective"] = False
    perf["meta"]["vividvr_debug"]["vae_sp_fallback_reason"] = "sp_world_size_one"
    with pytest.raises(BenchmarkDataError, match="VAE SP expected effective"):
        validate_effective_config(ALL_SCHEMES["R99_VAE_SP"], perf)
```

fixture 的 `vae_sp=True` 分支必须补齐 total tiles、local counts、四个时间字段和 `group_type="sp"`。

- [ ] **步骤 3：增加历史控制读取与派生指标失败测试**

```python
def test_load_historical_control_and_compute_vae_sp_speedups(tmp_path):
    control_dir = tmp_path / "control"
    write_formal_record(
        control_dir / "records/R99_formal.json",
        total=551.119,
        model=544.321,
        decode_trim=100.274,
        quality_passed=True,
    )
    controls = load_historical_controls(
        control_dir, ALL_SCHEMES["R99_VAE_SP"]
    )
    treatment = formal_record_with_stage(total=500.0, model=493.0, decode_trim=50.0)
    derived = compute_vae_sp_derived_metrics(
        ALL_SCHEMES["R99_VAE_SP"], treatment, controls["R99"]
    )
    assert derived["control_scheme_id"] == "R99"
    assert derived["decode_trim_speedup"] == pytest.approx(100.274 / 50.0)
    assert derived["model_inference_speedup"] == pytest.approx(544.321 / 493.0)
    assert derived["total_runtime_speedup"] == pytest.approx(551.119 / 500.0)


def test_load_historical_r100_accepts_recorded_quality_failed_control(tmp_path):
    control_dir = tmp_path / "control"
    write_formal_record(
        control_dir / "records/R100_formal.json",
        scheme_id="R100",
        status="quality_failed",
        total=370.881,
        model=365.067,
        decode_trim=101.786,
        quality_passed=False,
        ssim_mean=0.9846193275671117,
        ssim_min=0.978691848628344,
        failed_frame_ratio=2 / 130,
    )
    controls = load_historical_controls(
        control_dir, ALL_SCHEMES["R100_VAE_SP"]
    )
    assert controls["R100"]["status"] == "quality_failed"
```

- [ ] **步骤 4：运行新 benchmark 测试并确认失败**

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_vividvr_acceleration_benchmark.py \
  -k 'vae_sp or historical_control' -q
```

预期：FAIL，缺少 treatment registry、Scheme 字段或历史控制函数。

- [ ] **步骤 5：扩展 Scheme，但保持默认矩阵不变**

给 `Scheme` 和 `_scheme` 增加 `vae_sp: bool = False`。保持现有 `SCHEMES` 原样，另建：

```python
VAE_SP_TREATMENTS = {
    "R99_VAE_SP": _scheme(
        "R99_VAE_SP",
        "R99 + CogVideoX VAE spatial tile parallel",
        gpu_count=2,
        parallel_mode="sp",
        sp_degree=2,
        compile_enabled=True,
        modulation_fusion=True,
        vae_sp=True,
        controls=("R99",),
    ),
    "R100_VAE_SP": _scheme(
        "R100_VAE_SP",
        "R100 + CogVideoX VAE spatial tile parallel",
        gpu_count=4,
        parallel_mode="cfg_sp",
        sp_degree=2,
        compile_enabled=True,
        modulation_fusion=True,
        vae_sp=True,
        controls=("R100",),
    ),
}
ALL_SCHEMES = {**SCHEMES, **VAE_SP_TREATMENTS}
```

`run-all` 和无 `--scheme` 的 `dry-run` / 内部 batch 继续返回 `list(SCHEMES.values())`；`run-one` 与 `_run-batch --scheme` 的 choices 改为 `ALL_SCHEMES`。给 `dry-run` 增加可选 `--scheme`（choices 也是 `ALL_SCHEMES`），这样可以只打印 treatment 的预检而不启动服务。`_selected_schemes` 在 scheme ID 存在时查 `ALL_SCHEMES[scheme_id]`，否则保持默认 `list(SCHEMES.values())`。

- [ ] **步骤 6：只为 treatment 添加服务参数并校验 debug**

`build_service_command` 末尾加入：

```python
if scheme.vae_sp:
    command.append("--vae-sp")
```

`BenchmarkConfig` 的正式默认值保持与 `mock_test.md` 一致：`model_path=/home/zhiheng/ckpts/CogVideoX1.5-5B`、`vividvr_path=/home/zhiheng/ckpts/Vivid-VR`、service/caption/callback/S3 ports 为 `31221/31200/39090/4566`，bucket 为 `flowcut`。`TmuxBenchmarkLifecycle` 必须继续依次启动 Moto、callback、固定 caption mock 和 scheme service；`FlowCutRequestExecutor` 必须继续通过 `/v1/videos/repairs/flowcut` 提交、轮询、下载并 compare。VAE SP treatment 不增加任何直接调用 pipeline 的旁路。

`validate_effective_config` 只在 `scheme.vae_sp` 为真时要求：requested/effective 都为真、fallback reason 为 `effective`、group type 为 `sp`、world size 等于 `sp_degree`、local count 长度等于 SP world size、local count 求和等于 total tiles、所有时间字段为非负数。现有 R0-R100 不强制包含新字段，以便历史记录仍可读取。

- [ ] **步骤 7：增加 `--control-batch-dir` 并读取历史 JSON**

给 `BenchmarkConfig` 增加 `control_batch_dir: Path | None = None`，CLI 增加同名 Path 参数。运行 VAE SP treatment 时要求该目录存在，并从：

```text
<control_batch_dir>/records/<control_scheme_id>_formal.json
```

读取控制记录。验证 `run_role == "formal"`、`scheme.scheme_id` 与声明 control 一致、timings/quality 字段完整，且 status 只能是 `succeeded` 或 `quality_failed`；不要拒绝历史 R100 的 `quality_failed`。`_config_from_args` 对 `None` 保持 `None`，只对非空路径执行 `expanduser().resolve()`；`_config_cli_arguments` 也只在非空时传递该参数。

`compute_vae_sp_derived_metrics` 写入 treatment formal JSON 的 `derived`，至少包含三个 speedup、control/treatment GPU·秒、控制文件路径、控制 batch ID、`control_quality_passed`，以及 treatment 相对 control 的 SSIM mean/min delta 和 failed-frame-ratio delta。增加纯函数 `quality_not_worse_than_control(treatment, control)`：要求 SSIM mean/min 不低于 control 超过 `1e-6`，failed frame ratio 不高于 control；它不篡改 benchmark 原有 `pass_compare` 或 record status。

在 `BenchmarkRunner.run` 写 formal 派生指标的位置显式分支：当 `scheme.vae_sp` 且 treatment status 属于 `{"succeeded", "quality_failed"}` 时，加载声明的历史 control 并调用 `compute_vae_sp_derived_metrics`；其他 scheme 继续只在 `status == "succeeded"` 时调用原 `compute_derived_metrics`。这样 R100 treatment 即使复现历史 `quality_failed`，仍会产出性能和相对质量指标，且 R0-R100 原逻辑完全不变。

- [ ] **步骤 8：运行完整 benchmark unit 文件**

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_vividvr_acceleration_benchmark.py -q
```

预期：全部 PASS；原 `test_scheme_registry_has_fixed_order_and_capabilities` 无需改变期望列表。

- [ ] **步骤 9：提交 benchmark treatment**

```bash
git add \
  python/sglang/multimodal_gen/tools/run_vividvr_acceleration_benchmark.py \
  python/sglang/multimodal_gen/test/unit/test_vividvr_acceleration_benchmark.py
git commit -m "feat(vividvr): add VAE SP benchmark treatments"
```

## 任务 7：实现 NCCL subgroup 与真实 VAE 固定 latent 验证

**文件：**

- 创建：`test/srt/multimodal_gen/test_cogvideox_vae_spatial_tile_parallel_distributed.py`
- 创建：`python/sglang/multimodal_gen/tools/run_vividvr_vae_spatial_decode_validation.py`

- [ ] **步骤 1：编写可直接 torchrun 的 subgroup transport 测试**

入口接收 `--topology sp2|sp4|cfg2_sp2`，复用正式 runtime 的 group 初始化，不在测试里另造 coordinator：

```python
def initialize_topology(topology: str, rank: int, local_rank: int, world_size: int):
    cfg_degree, sp_degree = {
        "sp2": (1, 2),
        "sp4": (1, 4),
        "cfg2_sp2": (2, 2),
    }[topology]
    assert world_size == cfg_degree * sp_degree
    torch.cuda.set_device(local_rank)
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        backend="nccl",
        device_id=torch.device("cuda", local_rank),
    )
    initialize_model_parallel(
        classifier_free_guidance_degree=cfg_degree,
        sequence_parallel_degree=sp_degree,
        ulysses_degree=sp_degree,
        ring_degree=1,
    )
    return get_sp_group(), cfg_degree, sp_degree
```

每个 subgroup 使用不同 marker：`(rank // sp_degree) * 10000`。构造 7 个不同 shape 的 synthetic decoded tiles，按 round-robin 只在 owner rank 保留，调用 `_all_gather_decoded_tiles(get_sp_group(), ...)` 后断言本 subgroup 恢复 `0..6`，且 tensor 首值包含本 subgroup marker。最后 `dist.barrier()`，rank 0 打印 PASS；`finally` 中依次调用 `destroy_model_parallel()` 和 `dist.destroy_process_group()`。

```text
PASS: CogVideoX VAE tile transport topology=<topology> subgroup isolation verified
```

- [ ] **步骤 2：运行 SP2 synthetic transport**

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/torchrun \
  --standalone --nproc_per_node=2 \
  test/srt/multimodal_gen/test_cogvideox_vae_spatial_tile_parallel_distributed.py \
  --topology sp2
```

预期：退出码 0，输出上述 PASS 行；测试和生产 helper 都只从 `get_sp_group()` 取得 collective coordinator，不直接传 WORLD group。

- [ ] **步骤 3：实现真实 VAE validation tool 参数与加载**

参数固定为：

```python
parser.add_argument("--model-path", type=Path, default=Path("/home/zhiheng/ckpts/CogVideoX1.5-5B"))
parser.add_argument("--topology", choices=("sp2", "sp4", "cfg2_sp2"), required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--latent-frames", type=int, default=5)
parser.add_argument("--latent-height", type=int, default=65)
parser.add_argument("--latent-width", type=int, default=97)
parser.add_argument("--output-json", type=Path, required=True)
```

工具必须：

1. 调用与上面完全相同的 `init_distributed_environment(...)`、`initialize_model_parallel(...)` 初始化 NCCL 和正式 topology；
2. 用 `PipelineComponentLoader.load_component("vae", model_path / "vae", "diffusers", server_args)` 加载 native SGLang CogVideoX VAE；
3. 每个 CFG subgroup 用 `seed + cfg_group_index` 生成 `[1, C, 5, 65, 97]` BF16 fixed latent；
4. `requested=False` 连续解码两次得到 `serial_a/serial_b`；
5. `requested=True` 解码一次得到 `parallel`；
6. 比较 shape、`torch.equal`、max/mean absolute error；
7. all-gather 各 rank 的 JSON 数值并由 global rank 0 原子写入输出文件。

数值判定实现为：

```python
serial_repeat = (serial_a.float() - serial_b.float()).abs()
parallel_error = (serial_a.float() - parallel.float()).abs()
deterministic = torch.equal(serial_a, serial_b)
passed = (
    torch.equal(serial_a, parallel)
    if deterministic
    else (
        parallel_error.max() <= serial_repeat.max()
        and parallel_error.mean() <= serial_repeat.mean()
    )
)
```

输出 JSON 至少包含 topology、SP subgroup ranks、latent shape/dtype/seed、serial deterministic、serial repeat max/mean、parallel max/mean、每 rank stats、每 rank pass、overall pass、`total_runtime_seconds` 和 `model_inference_runtime_seconds`。两个时间字段覆盖范围要在 JSON 中注明：前者是进程启动至结果落盘前，后者是三次 `vae.decode(...)` 的 CUDA-event 总和。

- [ ] **步骤 4：为 tool 的纯比较函数增加 CPU 测试**

把比较提取为 `compare_serial_and_parallel_decode(serial_a, serial_b, parallel)`，在 VAE unit 文件测试 deterministic exact pass、deterministic mismatch fail、nondeterministic envelope pass/fail 四种情况。

- [ ] **步骤 5：运行纯测试与 import smoke**

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_cogvideox_vae_spatial_tile_parallel.py -q
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python \
  python/sglang/multimodal_gen/tools/run_vividvr_vae_spatial_decode_validation.py --help
```

预期：pytest 全部 PASS，help 退出码 0 并列出三种 topology。

- [ ] **步骤 6：提交 distributed validation 工具**

```bash
git add \
  test/srt/multimodal_gen/test_cogvideox_vae_spatial_tile_parallel_distributed.py \
  python/sglang/multimodal_gen/tools/run_vividvr_vae_spatial_decode_validation.py \
  python/sglang/multimodal_gen/test/unit/test_cogvideox_vae_spatial_tile_parallel.py
git commit -m "test(vividvr): validate VAE tile parallel subgroups"
```

## 任务 8：执行轻量正确性验收

**文件：**

- 产出：`Vivid_Acceptance/indicator/vividvr_vae_sp_fixed_latent_*.json`
- 产出：`Vivid_Acceptance/logs/vividvr_vae_sp_fixed_latent_*.log`

三个 topology 必须串行执行：前一个 tmux 已退出、日志完整且 GPU 无残留进程后，才启动下一个。

- [ ] **步骤 1：确认 GPU 空闲与工作区提交完整**

```bash
git status --short --branch
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
```

预期：任务相关代码已提交；所选 GPU 没有推理进程。若存在进程，停止验收并先确认归属，不终止未知进程。

- [ ] **步骤 2：在 tmux 中运行 SP2 fixed latent**

```bash
tmux new-session -d -s vividvr_vae_sp2_fixed \
  'cd /home/zhiheng/sglang && export PYTHONPATH=python && \
   /home/zhiheng/sglang/.venv/bin/torchrun --standalone --nproc_per_node=2 \
   python/sglang/multimodal_gen/tools/run_vividvr_vae_spatial_decode_validation.py \
   --topology sp2 \
   --output-json Vivid_Acceptance/indicator/vividvr_vae_sp_fixed_latent_sp2_20260716.json \
   2>&1 | tee Vivid_Acceptance/logs/vividvr_vae_sp_fixed_latent_sp2_20260716.log'
```

只读查看：`tmux attach -r -t vividvr_vae_sp2_fixed`。预期 JSON `overall_pass=true`，所有 rank `vae_sp_world_size=2`、`vae_sp_effective=true`。

- [ ] **步骤 3：在 tmux 中运行 SP4 fixed latent**

```bash
tmux new-session -d -s vividvr_vae_sp4_fixed \
  'cd /home/zhiheng/sglang && export PYTHONPATH=python && \
   /home/zhiheng/sglang/.venv/bin/torchrun --standalone --nproc_per_node=4 \
   python/sglang/multimodal_gen/tools/run_vividvr_vae_spatial_decode_validation.py \
   --topology sp4 \
   --output-json Vivid_Acceptance/indicator/vividvr_vae_sp_fixed_latent_sp4_20260716.json \
   2>&1 | tee Vivid_Acceptance/logs/vividvr_vae_sp_fixed_latent_sp4_20260716.log'
```

只读查看：`tmux attach -r -t vividvr_vae_sp4_fixed`。预期 JSON `overall_pass=true`，world size 为 4，local tile 数差不超过 1。该步骤不是 `130f / 20 step` 性能实验。

- [ ] **步骤 4：在 tmux 中运行 CFG2×SP2 fixed latent**

```bash
tmux new-session -d -s vividvr_vae_cfg2sp2_fixed \
  'cd /home/zhiheng/sglang && export PYTHONPATH=python && \
   /home/zhiheng/sglang/.venv/bin/torchrun --standalone --nproc_per_node=4 \
   python/sglang/multimodal_gen/tools/run_vividvr_vae_spatial_decode_validation.py \
   --topology cfg2_sp2 \
   --output-json Vivid_Acceptance/indicator/vividvr_vae_sp_fixed_latent_cfg2_sp2_20260716.json \
   2>&1 | tee Vivid_Acceptance/logs/vividvr_vae_sp_fixed_latent_cfg2_sp2_20260716.log'
```

只读查看：`tmux attach -r -t vividvr_vae_cfg2sp2_fixed`。预期两个 subgroup 各自 world size 为 2，两个 seed 的结果不混合，所有 rank pass。

- [ ] **步骤 5：检查三个 JSON 的共同门槛**

```bash
/home/zhiheng/sglang/.venv/bin/python - <<'PY'
import json
from pathlib import Path

paths = sorted(Path("Vivid_Acceptance/indicator").glob(
    "vividvr_vae_sp_fixed_latent_*_20260716.json"
))
assert len(paths) == 3, paths
for path in paths:
    payload = json.loads(path.read_text())
    assert payload["overall_pass"] is True, path
    assert all(item["vae_sp_effective"] for item in payload["ranks"]), path
print("PASS:", *(str(path) for path in paths), sep="\n")
PY
```

预期：打印三个文件并退出 0。

## 任务 9：执行且仅执行 R99/R100 两条正式 treatment

**文件：**

- 读取：`Vivid_Acceptance/acceleration_benchmark/vividvr_accel_full_warmup1_20260716/records/R99_formal.json`
- 读取：`Vivid_Acceptance/acceleration_benchmark/vividvr_accel_full_warmup1_20260716/records/R100_formal.json`
- 读取：`docs_xzh/run_command/mock_test.md`
- 读取权重：`/home/zhiheng/ckpts/CogVideoX1.5-5B`
- 读取权重：`/home/zhiheng/ckpts/Vivid-VR`
- 产出：`Vivid_Acceptance/acceleration_benchmark/vividvr_vae_sp_r99_20260716`
- 产出：`Vivid_Acceptance/acceleration_benchmark/vividvr_vae_sp_r100_20260716`

- [ ] **步骤 1：预检权重、服务依赖和两个 treatment 的唯一变量**

```bash
test -d /home/zhiheng/ckpts/CogVideoX1.5-5B
test -d /home/zhiheng/ckpts/CogVideoX1.5-5B/vae
test -d /home/zhiheng/ckpts/Vivid-VR
test -d /home/zhiheng/ckpts/Vivid-VR/controlnet
test -f /home/zhiheng/sglang/Vivid_Acceptance/captions/service_sidecars/quad-test-video-long-960x720-130f-run2-20260708T060202Z.txt
test -f /home/zhiheng/input/test_video_long_960x720_130f.mp4
stat -c '%n %Y' \
  Vivid_Acceptance/acceleration_benchmark/vividvr_accel_full_warmup1_20260716/records/R99_formal.json \
  Vivid_Acceptance/acceleration_benchmark/vividvr_accel_full_warmup1_20260716/records/R100_formal.json
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python \
  python/sglang/multimodal_gen/tools/run_vividvr_acceleration_benchmark.py \
  dry-run \
  --control-batch-dir Vivid_Acceptance/acceleration_benchmark/vividvr_accel_full_warmup1_20260716
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python \
  python/sglang/multimodal_gen/tools/run_vividvr_acceleration_benchmark.py \
  dry-run --scheme R99_VAE_SP \
  --model-path /home/zhiheng/ckpts/CogVideoX1.5-5B \
  --vividvr-path /home/zhiheng/ckpts/Vivid-VR \
  --control-batch-dir Vivid_Acceptance/acceleration_benchmark/vividvr_accel_full_warmup1_20260716
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python \
  python/sglang/multimodal_gen/tools/run_vividvr_acceleration_benchmark.py \
  dry-run --scheme R100_VAE_SP \
  --model-path /home/zhiheng/ckpts/CogVideoX1.5-5B \
  --vividvr-path /home/zhiheng/ckpts/Vivid-VR \
  --control-batch-dir Vivid_Acceptance/acceleration_benchmark/vividvr_accel_full_warmup1_20260716
```

任一 `test` 失败都停止，不从其他目录猜测或复制权重。保存两行 mtime 输出。检查第一次 dry-run 的默认矩阵仍止于 R100，不包含 treatment。后两次分别只输出一个 treatment；逐项比较 service command 与历史 JSON 中 `reproducibility.service_command`，差异必须只有末尾 `--vae-sp`。两者都必须显示 `/home/zhiheng/ckpts` 下的两条权重路径、compile 的一次 1-step warmup 和 20-step formal 请求，dry-run 全程不得启动服务。

- [ ] **步骤 2：按 `mock_test.md` 生命周期启动 R99_VAE_SP 的服务验收**

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python \
  python/sglang/multimodal_gen/tools/run_vividvr_acceleration_benchmark.py \
  run-one --scheme R99_VAE_SP \
  --batch-id vividvr_vae_sp_r99_20260716 \
  --control-batch-dir Vivid_Acceptance/acceleration_benchmark/vividvr_accel_full_warmup1_20260716 \
  --model-path /home/zhiheng/ckpts/CogVideoX1.5-5B \
  --vividvr-path /home/zhiheng/ckpts/Vivid-VR \
  --gpu-ids 0,1
```

该 runner 自己创建 batch tmux，禁止再套一层 tmux。确认启动 JSON 中 session 为 `vividvr_accel_batch_vividvr_vae_sp_r99_20260716`；只读查看：`tmux attach -r -t vividvr_accel_batch_vividvr_vae_sp_r99_20260716`。batch 内部必须按顺序出现带同一 ownership token 的 Moto、callback、caption 和 `R99_VAE_SP_service` sessions，三个 HTTP health check 全部成功后才允许提交 FlowCut warmup。正式请求只能在同一服务完成 1-step warmup 后提交。预期 formal `succeeded`、质量 compare passed、runtime 中 `vae_sp_effective=true`，历史 R99 文件 mtime 不变。

- [ ] **步骤 3：核对 R99 服务证据并清理，再启动 R100_VAE_SP**

R99 batch 退出后先检查下列文件均存在且非空：

```bash
test -s Vivid_Acceptance/acceleration_benchmark/vividvr_vae_sp_r99_20260716/logs/moto.log
test -s Vivid_Acceptance/acceleration_benchmark/vividvr_vae_sp_r99_20260716/logs/callbacks.jsonl
test -s Vivid_Acceptance/acceleration_benchmark/vividvr_vae_sp_r99_20260716/logs/caption.log
test -s Vivid_Acceptance/acceleration_benchmark/vividvr_vae_sp_r99_20260716/logs/R99_VAE_SP_service.log
test -s Vivid_Acceptance/acceleration_benchmark/vividvr_vae_sp_r99_20260716/records/R99_VAE_SP_formal.json
```

确认 callback JSONL 含 `accepted`、`input_ready`、`caption_ready`、`denoising`、`uploading_result` 和最终 `succeeded`，formal record 的 `artifacts.result_video`、`perf_json`、`compare_json` 均存在；确认当前 batch 创建的全部 tmux sessions 已清理且 GPU 无残留推理进程，再执行：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python \
  python/sglang/multimodal_gen/tools/run_vividvr_acceleration_benchmark.py \
  run-one --scheme R100_VAE_SP \
  --batch-id vividvr_vae_sp_r100_20260716 \
  --control-batch-dir Vivid_Acceptance/acceleration_benchmark/vividvr_accel_full_warmup1_20260716 \
  --model-path /home/zhiheng/ckpts/CogVideoX1.5-5B \
  --vividvr-path /home/zhiheng/ckpts/Vivid-VR \
  --gpu-ids 0,1,2,3
```

只读查看：`tmux attach -r -t vividvr_accel_batch_vividvr_vae_sp_r100_20260716`。内部必须复用同一套 `mock_test.md` 依赖服务，但主服务为四卡 `cfg_sp`；预期 CFG parallel 开启、SP world size 为 2、每个 clip effective，R100 历史文件 mtime 不变。由于历史 R100 自身为 `quality_failed`，新记录允许按原 compare 规则保持同一 status，但必须满足下面的相对质量门槛。

- [ ] **步骤 4：核对 R100 服务证据和两条 record 的可复现配置**

对 R100 重复 R99 的 Moto/callback/caption/service log、callback 阶段、下载视频、perf JSON 和 compare JSON 检查。然后从两条 formal record 的 `reproducibility` 校验：

- `model_path == /home/zhiheng/ckpts/CogVideoX1.5-5B`；
- `vividvr_path == /home/zhiheng/ckpts/Vivid-VR`；
- service command 包含 caption bridge、FA、compile、modulation fusion 和 `--vae-sp`；
- R99 为 2 GPU、`sp`、SP degree 2，R100 为 4 GPU、`cfg_sp`、CFG enabled、SP degree 2；
- formal request 为 20 steps、seed 42、121 temporal process frames，并包含 callback、Moto S3 object key 和 perf dump path；
- caption/prompt path 均未直接写进 FlowCut request；
- 两个历史 control 文件的 mtime 与步骤 1 完全一致。

- [ ] **步骤 5：检查正式结果门槛**

两条 treatment 共同检查：

```text
runtime.vae_sp_effective == true
derived.quality_not_worse_than_control == true
derived.decode_trim_speedup > 1.0
derived.model_inference_speedup >= 1.0
derived.total_runtime_speedup >= 1.0
```

相对质量门槛固定为：SSIM mean/min 不低于各自 control 超过 `1e-6`，failed-frame ratio 不高于 control。R99 另要求 `quality.pass_compare == true`；R100 必须原样报告 `pass_compare`，但不把历史控制自身未达到的绝对阈值误作 VAE SP 回归。若 R100 treatment 达到原 compare 阈值则如实记为 `succeeded`，否则在相对质量未回归时可记为 `quality_failed`。

同时记录 tile decode/gather/merge、总 model inference、总 runtime、GPU·秒和每 rank 峰值显存。若正确性通过但任一端到端 speedup 小于 1，保留实验开关但不晋升默认配置。

- [ ] **步骤 6：人工检查两条结果视频**

重点查看空间 tile 接缝、闪烁、颜色漂移、temporal clip stitch 边界、首三帧删除和末尾 crop。人工结论以 `pass/fail + 观察说明` 写入 acceptance 文档，不用肉眼判断替代 JSON 数值门槛。

## 任务 10：完整回归、文档收口与阶段提交

**文件：**

- 创建：`docs_xzh/distribute/vividvr_vae_spatial_tile_parallel_acceptance_20260716.md`
- 修改：`docs_xzh/hand_over/2026-07-16-vividvr-vae-spatial-tile-parallel-design.md`
- 修改：`docs_xzh/hand_over/vividvr_vae_tile_parallel_next_handover_20260716.md`

- [ ] **步骤 1：运行本改动覆盖的完整轻量测试集**

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_cogvideox_vae_spatial_tile_parallel.py \
  python/sglang/multimodal_gen/test/unit/test_stage_b_vividvr_components.py \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_offload.py \
  python/sglang/multimodal_gen/test/unit/test_stage_d_vividvr_temporal_orchestration.py \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_attention_backend.py \
  python/sglang/multimodal_gen/test/unit/test_vividvr_acceleration_benchmark.py \
  -q
```

预期：全部 PASS。该测试集合保护默认关闭、串行 fallback、已有长视频编排和 benchmark 默认矩阵，不额外运行单卡真实推理。

- [ ] **步骤 2：静态检查 collective 边界与禁用实现**

```bash
rg -n "all_gather_object|dist\.group\.WORLD|group=WORLD" \
  python/sglang/multimodal_gen/runtime/models/vaes/cogvideox.py
rg -n "get_sp_group|\.all_gather\(" \
  python/sglang/multimodal_gen/runtime/models/vaes/cogvideox.py
```

预期：第一条无输出；第二条显示所有 descriptor、metadata、payload collective 都由 SP coordinator 发起。

- [ ] **步骤 3：编写 acceptance 文档**

文档必须列出：

- 实际 commit 列表与 diffusers 版本；
- SP2/SP4/CFG2×SP2 fixed latent JSON 和日志路径；
- R99/R100 历史控制文件及其四个基线值；
- 两条 treatment 的总耗时、model inference、Decode/Trim、tile decode/gather/merge、GPU·秒、显存、质量指标；
- 人工检查结论；
- 是否满足默认晋升条件。第一阶段默认决定仍为 `vae_sp=False`，除非用户另行批准修改正式默认。

- [ ] **步骤 4：更新两份 handover 状态**

把“尚未实施”更新为实际完成状态，写入代码提交、验收产物、已知风险和下一轮入口。若实现与设计没有偏差，明确写“实现遵循已确认设计”；若存在偏差，逐项说明原因和证据，不重写原设计历史。

- [ ] **步骤 5：执行完成前验证**

使用 `verification-before-completion` 技能，重新运行步骤 1、步骤 2，并检查：

```bash
git status --short
git diff --check
git diff --stat HEAD~1..HEAD
```

预期：测试和静态检查通过，`git diff --check` 无输出，仅存在本阶段文档收口改动。

- [ ] **步骤 6：提交文档收口**

```bash
git add \
  docs_xzh/distribute/vividvr_vae_spatial_tile_parallel_acceptance_20260716.md \
  docs_xzh/hand_over/2026-07-16-vividvr-vae-spatial-tile-parallel-design.md \
  docs_xzh/hand_over/vividvr_vae_tile_parallel_next_handover_20260716.md
git commit -m "docs(vividvr): close VAE tile parallel acceptance"
```

- [ ] **步骤 7：检查提交范围并推送当前分支**

```bash
git log --oneline --decorate -10
git status --short --branch
git push origin sglang_Vivid
```

预期：所有阶段提交只含任务相关文件，工作区干净，远端 `sglang_Vivid` 更新成功。

## 计划执行前检查

执行者开始任务 1 前必须：

1. 阅读 `docs_xzh/hand_over/2026-07-16-vividvr-vae-spatial-tile-parallel-design.md`；
2. 阅读 `docs_xzh/hand_over/vividvr_vae_tile_parallel_next_handover_20260716.md`；
3. 使用 `using-git-worktrees` 建立隔离工作区；
4. 确认 `.venv` 中 Diffusers 版本仍为 0.37.0，并重新查看实际 `tiled_decode` 源码；
5. 检查 `git status`，不带入用户无关改动。

本计划不授权修改正式默认配置、服务契约或运行其他正式 benchmark。任何超出两条 treatment 的真实推理都必须先取得用户明确同意。
