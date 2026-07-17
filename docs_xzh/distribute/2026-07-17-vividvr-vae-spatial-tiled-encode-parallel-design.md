# Vivid-VR VAE Spatial Tiled Encode Parallel Design

**Date:** 2026-07-17

**Status:** Approved design

**Scope:** Parallelize CogVideoX VAE tiled encode inside the existing Vivid-VR SP subgroup to reduce `VividVRLongClipPreparationStage` latency.

## 1. Background

The accepted Vivid-VR long-video path prepares each temporal clip serially. During
`VividVRLongClipPreparationStage`, `prepare_condition_inputs` calls
`vae.encode(...)` to produce the control latent distribution, then samples control
latents with the request generator.

For the formal 720x960, 130-frame workload:

- the request is split into two temporal clips;
- CogVideoX VAE tiling uses 240x360 sample-space tiles with the existing overlap
  factors;
- each temporal clip produces a 4x4 spatial tile grid;
- the request therefore performs about 32 heavyweight encode-tile jobs in total;
- the temporal clips remain serial, and every tile retains the VAE encoder's
  existing serial temporal-chunk and causal `conv_cache` behavior.

Existing VAE spatial parallelism covers tiled decode only. It has already shown
that tile computation dominates communication and that SP-subgroup tensor
collectives can preserve deterministic row-major reconstruction. Encode should
reuse that proven communication substrate without changing decode semantics.

Current historical controls report approximately 60-73 seconds for Long Clip
Preparation, which makes tiled encode the dominant optimization target in that
stage.

## 2. Goals and acceptance criteria

### 2.1 Functional goals

1. Add an independent `--vae-encode-sp` switch for CogVideoX tiled encode.
2. Distribute spatial encode tiles only across the existing SP subgroup.
3. Preserve the original Diffusers tile traversal, temporal encoding, overlap
   blending, cropping, posterior construction, and generator sampling semantics.
4. Support SP2, pure SP4, and CFG2xSP2. In CFG2xSP2, the two CFG groups must remain
   isolated and each group must perform its own SP2 encode.
5. Expose encode-specific runtime statistics without changing existing decode
   statistics.

### 2.2 Correctness hard gate

For a fixed input and seed, the parallel path must be bitwise equal to the serial
path for both:

- the full posterior moments tensor returned by tiled encode;
- the sampled control latents produced from that posterior with an equivalent
  generator state.

The check uses `torch.equal`, not a tolerance. Final video SSIM cannot waive a
moments or sampled-latents mismatch.

### 2.3 Performance gates

Using the accepted historical decode-only records as read-only controls:

- SP2 Long Clip Preparation speedup must be at least 1.5x;
- CFG2xSP2 Long Clip Preparation speedup must be at least 1.5x;
- pure SP4 Long Clip Preparation speedup must be at least 2.5x;
- every treatment must reduce `model_inference_runtime_seconds`;
- Denoise and Decode/Trim must not regress by more than 3% relative to their
  matching control;
- treatment records must retain total runtime, model inference runtime, stage
  timings, memory, quality, and reproducibility fields.

The stage speedup is defined as:

```text
historical_control_long_clip_preparation_seconds
------------------------------------------------
treatment_long_clip_preparation_seconds
```

## 3. Considered designs

### 3.1 Option A: spatial tiles inside the SP subgroup (selected)

Each SP rank encodes a deterministic subset of spatial tiles. Encoded posterior
moments are gathered within the subgroup, reconstructed in global row-major tile
order, and merged identically on every rank. Posterior sampling then follows the
existing path on every rank.

This option matches the accepted tiled-decode architecture, offers 16 independent
spatial jobs per temporal clip, naturally supports SP2 and SP4, and does not alter
temporal clip orchestration.

### 3.2 Option B: temporal-clip parallelism (rejected)

The formal request contains only two temporal clips, so SP4 would leave half the
ranks idle. It would also require redistributing clip state, generator state, and
control latents before timestep-level orchestration. The utilization and semantic
risk are worse than spatial tiling.

### 3.3 Option C: temporal/model parallelism inside each encode tile (rejected)

CogVideoX VAE encode carries causal temporal state through `conv_cache`. Splitting
that loop would require new halo/state exchange and a much larger equivalence
proof. It is unnecessary while 16 spatial tiles per clip already provide enough
parallel work.

## 4. Public configuration and compatibility

Add a new pipeline configuration field and CLI argument:

```text
vae_encode_sp: bool = False
--vae-encode-sp
```

The controls remain independent:

| `--vae-sp` | `--vae-encode-sp` | Encode | Decode |
|---|---|---|---|
| off | off | serial | serial |
| on | off | serial | spatial parallel |
| off | on | spatial parallel | serial |
| on | on | spatial parallel | spatial parallel |

`--vae-sp` keeps its current decode-only meaning. Both flags default to false, so
no existing default configuration or service request contract changes.

The Vivid-VR pipeline configures encode through a separate native VAE method,
conceptually:

```text
configure_spatial_tile_encode_parallel(requested: bool)
```

This avoids silently broadening the existing decode configuration method.

## 5. Architecture

### 5.1 Ownership

The long-clip stage call path remains unchanged. All encode-tile parallel
algorithm details live in the native CogVideoX VAE runtime. Pipeline wiring only
passes the configuration, and the stage only aggregates per-clip statistics.

The encode and decode implementations may share a small tensor-collective helper,
but they must retain separate:

- tile descriptor and plan types;
- sample-to-latent versus latent-to-sample coordinate rules;
- local tile workers;
- merge functions;
- statistics.

### 5.2 Activation rules

The VAE overrides the tiled-encode entry point while preserving its public return
contract.

- If encode SP is not requested, call the existing serial implementation.
- If the SP world size is one, use the serial implementation and report the
  explicit fallback.
- If the input does not cross the tiling threshold, keep the existing non-tiled
  encode path and report the explicit fallback.
- If tiled encode and encode SP are active with SP world size greater than one,
  use the parallel path.

No fallback to serial is allowed after a distributed operation has begun.

## 6. Parallel data flow

For every temporal clip, the following sequence runs collectively within its SP
subgroup.

### 6.1 Canonicalize and validate input

1. Build a tensor descriptor containing shape, dtype, device type, tiling
   parameters, expected tile count, SP world size, and subgroup identity.
2. Tensor-all-gather descriptors and require every rank to agree.
3. Make the tensor contiguous and broadcast the subgroup-root control-video
   tensor so each SP rank starts from the same canonical bytes.
4. Do not communicate through the global world group and do not use
   `all_gather_object`.

In CFG2xSP2, each SP subgroup has its own root and collectives. No encode payload
crosses the CFG boundary.

### 6.2 Build and assign encode tiles

Construct the exact Diffusers sample-space tile ranges and row-major global tile
indices. Assign tiles by:

```text
owner_rank = global_tile_index % sp_world_size
```

For the formal 4x4 grid this produces:

- SP2: eight tiles per rank per temporal clip;
- SP4: four tiles per rank per temporal clip.

The plan stores enough sample-space and latent-space coordinates to validate every
returned tile before merge.

### 6.3 Encode local tiles

Each owner rank runs the unchanged per-tile algorithm:

1. reset `conv_cache` for the spatial tile;
2. iterate the same temporal chunks in the same order;
3. call the existing VAE encoder;
4. carry causal `conv_cache` only between temporal chunks of that same tile;
5. concatenate temporal outputs identically;
6. apply the existing `quant_conv` to produce posterior moments.

No temporal chunk is distributed, reordered, cached across spatial tiles, or
shared between requests.

### 6.4 Gather encoded moments

Gather fixed-width tensor metadata followed by tensor payloads within the SP
subgroup. The transport must support variable edge-tile shapes without Python
object collectives.

After gathering, every rank:

- validates owner, global index, shape, dtype, payload length, and uniqueness;
- rejects missing, duplicate, or out-of-range tiles;
- reconstructs the complete tile list sorted by global row-major index.

### 6.5 Replicated deterministic merge

Every rank performs the same merge in the exact serial order:

1. blend the current tile with the tile above;
2. blend it with the tile to the left;
3. crop to the existing latent row/column limits;
4. concatenate tiles within a row;
5. concatenate rows.

The operations, operands, slicing boundaries, dtype, and device remain identical
to serial Diffusers tiled encode. Replicated merge avoids a second redistribution
before downstream denoising.

### 6.6 Preserve posterior sampling

Parallel tiled encode returns only the complete merged moments tensor. Existing
Diffusers code constructs `DiagonalGaussianDistribution`, and the unchanged
`retrieve_latents(..., generator)` path samples the control latents.

The distributed algorithm must not sample per tile or advance the request
generator. Correctness validation creates equivalent generator states for serial
and parallel sampling and compares both moments and final sampled latents with
`torch.equal`.

## 7. Error handling and fallback policy

### 7.1 Startup errors

Fail before inference when encode SP is requested but:

- VAE tiling is disabled;
- the native CogVideoX VAE lacks the encode-parallel interface;
- distributed execution requires an SP subgroup that is not initialized.

### 7.2 Legal serial fallbacks

Only these pre-collective cases may fall back:

- `not_requested`;
- `sp_world_size_one`;
- `input_below_tiling_threshold`.

### 7.3 Fatal distributed errors

Once parallel tiled encode starts, descriptor mismatch, plan mismatch, unexpected
dtype/device, invalid tile metadata, missing payload, duplicate tile, collective
failure, or merged-shape mismatch must propagate as an error on the request. A
rank must never silently switch to serial because that can deadlock its peers.

## 8. Runtime statistics

Encode exposes a namespace separate from tiled decode:

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

The long-clip stage records one stats object per temporal clip and aggregates
counts and durations across clips. `vae_encode_sp_effective` is true only when all
tiled clips expected to use the parallel path did so. Per-rank tile counts remain
visible so uneven or missing work cannot be hidden by an aggregate.

The benchmark report includes Long Clip Preparation and its encode compute,
gather, and merge subphases, together with model time, total time, GPU-seconds,
peak memory, and quality.

## 9. Verification strategy

### 9.1 Unit tests

CPU/fake-collective tests cover:

- the 720x960 4x4 plan and SP2/SP4 round-robin ownership;
- odd and edge spatial sizes with variable encoded tile shapes;
- preservation of per-tile temporal chunk order and `conv_cache` reset/carry;
- placement of `quant_conv` before tile gathering;
- exact row-major vertical-then-horizontal blend and crop order;
- metadata/payload reconstruction and rejection of missing or duplicate tiles;
- CFG subgroup isolation;
- independent CLI/config wiring for `vae_sp` and `vae_encode_sp`;
- legal fallback reasons and fatal post-collective errors;
- per-clip and long-clip statistics aggregation;
- historical-control validation and derived speedup calculations.

Existing Phase C/D/E and tiled-decode tests remain regression coverage.

### 9.2 Real VAE and NCCL correctness tests

An encode-specific validation tool runs the real CogVideoX VAE with fixed inputs
and seeds for:

- SP2 on GPUs 0-1;
- pure SP4 on GPUs 0-3;
- CFG2xSP2 on GPUs 0-3.

For each topology it compares serial and parallel full posterior moments and
sampled control latents with `torch.equal`. It also covers non-contiguous inputs,
rank-divergent input tensors that must be canonicalized from the subgroup root,
tile ownership counts, CFG subgroup markers, and collective metadata.

These long-running validations run in named `tmux` sessions with logs and
artifacts under `Vivid_Acceptance`.

### 9.3 Treatment-only formal inference

Formal acceptance does not rerun Control inference. It runs exactly three new
Treatments with both `--vae-sp` and `--vae-encode-sp`:

| Treatment | Topology | Read-only historical Control |
|---|---|---|
| R99 encode SP | SP2 | `Vivid_Acceptance/acceleration_benchmark/vividvr_vae_sp_r99_canonicalized_v2_20260716/records/R99_VAE_SP_formal.json` |
| R100 encode SP | CFG2xSP2 | `Vivid_Acceptance/acceleration_benchmark/vividvr_vae_sp_r100_canonicalized_20260716/records/R100_VAE_SP_formal.json` |
| R101 encode SP4 | pure SP4 | `Vivid_Acceptance/acceleration_benchmark/vividvr_vae_sp4_fusion_20260717/records/R101_VAE_SP4_formal.json` |

Before running a Treatment, the runner validates that its Control exists and has
the expected scheme ID, topology, backend, input video, caption/reference source,
130 frames, 20 inference steps, seed 42, effective decode `vae_sp`, absent/false
encode SP, and complete timing fields.

The runner records each Control's hash and modification time before acceptance and
verifies both are unchanged afterward. A historical quality status does not
invalidate its use as a timing control when its configuration and timing record
are complete; treatment quality is still evaluated and reported independently.

Each compile-enabled Treatment receives the existing single 1-step warmup and one
formal 20-step request. No Control service, warmup, or inference is launched.
Every long-running command uses a dedicated `tmux` session.

## 10. Expected implementation surface

The implementation is expected to touch only the minimal relevant areas:

- `python/sglang/multimodal_gen/configs/pipeline_configs/base.py`
- `python/sglang/multimodal_gen/configs/pipeline_configs/vividvr.py`
- `python/sglang/multimodal_gen/runtime/models/vaes/cogvideox.py`
- `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py`
- `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py`
- `python/sglang/multimodal_gen/tools/run_vividvr_acceleration_benchmark.py`
- a new encode-specific validation tool under
  `python/sglang/multimodal_gen/tools/`
- focused unit tests under `python/sglang/multimodal_gen/test/unit/`
- acceptance and run documentation affected by the new experimental flag.

The exact implementation plan will name each function and test after this design
is reviewed.

## 11. Non-goals and rollback

This work does not:

- change the existing `--vae-sp` decode-only behavior;
- change any production default or external service request field;
- change clip splitting, captioning, conditioning semantics, denoising, decoding,
  trimming, color correction, or stitching;
- parallelize temporal clips or temporal chunks;
- communicate encode tensors through the global world group or across CFG groups;
- rerun, rewrite, or overwrite the R99/R100/R101 historical Control artifacts;
- unify encode and decode planners when their coordinate semantics differ.

The runtime rollback is simply to omit `--vae-encode-sp`. Encode then returns to
the current serial path, while `--vae-sp` may independently continue to accelerate
tiled decode.
