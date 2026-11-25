"""Benchmark-only geometry, stage residency, and per-rank measurements.

No server changes are made unless this directory is explicitly on PYTHONPATH and
VIDEOEDIT_SP_BENCH_DIR is set. Transfers and instrumentation count toward request
time; latent hashing is separately reported and happens outside stage timing.
"""
import functools
import hashlib
import json
import os
from pathlib import Path
import time


def full_frame_geometry(clips, height, width):
    # Do not alter the original mask pixels or the frame/window policy.
    if (width, height) != (1920, 1080):
        raise ValueError(f"Expected 1920x1080 source, got {width}x{height}")
    for clip in clips:
        for chunk in clip.chunks:
            chunk.bbox = (0, 0, width, height)
            chunk.crop_h, chunk.crop_w = height, width
            chunk.aligned_h, chunk.aligned_w = 1088, 1920


def install_geometry(module):
    if os.environ.get("VIDEOEDIT_SP_BENCH_GEOMETRY") != "full":
        return

    def assign(clips, boxes, height, width, **kwargs):
        full_frame_geometry(clips, height, width)

    module.assign_chunk_bboxes = assign


def emit(kind, batch, **fields):
    import torch.distributed as dist

    rank = dist.get_rank() if dist.is_initialized() else 0
    path = Path(os.environ["VIDEOEDIT_SP_BENCH_DIR"]) / f"rank{rank}.jsonl"
    with path.open("a") as handle:
        handle.write(json.dumps(dict(kind=kind, rank=rank,
                                    output=batch.output_file_path(), **fields)) + "\n")


def tensor_hash(tensor):
    import torch

    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def install_stages(module):
    import torch

    resident = os.environ.get("VIDEOEDIT_SP_BENCH_RESIDENT") == "1"
    names = [name for name in vars(module) if name.startswith("VideoEdit") and name.endswith("Stage")]
    for name in names:
        cls = getattr(module, name)
        original = cls.forward

        def wrap(original, name):
            @functools.wraps(original)
            def forward(self, batch, args):
                params = batch.sampling_params
                denoise = name == "VideoEditDenoisingStage"
                vae_stage = name in ("VideoEditConditionEncodingStage", "VideoEditDecodingStage")
                decisions = []
                hook_handles = []
                block_calls = [0]
                decision_original = None
                transformer_forward_original = None
                offload_managers = []
                transfer_start = (0, 0)
                if denoise:
                    offload_managers = getattr(self.transformer, "layerwise_offload_managers", [])
                    transfer_start = (
                        sum(getattr(m, "h2d_weight_bytes", 0) for m in offload_managers),
                        sum(getattr(m, "h2d_layer_loads", 0) for m in offload_managers),
                    )
                    if not batch.enable_teacache or batch.teacache_params.teacache_thresh != 0.3:
                        raise ValueError("Benchmark requires TeaCache threshold 0.3")
                    decision_original = self.transformer.should_skip_forward_for_cached_states

                    def decision(**kwargs):
                        skip = decision_original(**kwargs)
                        decisions.append({"step": params.runtime_current_step,
                                          "negative": bool(batch.is_cfg_negative), "skip": bool(skip)})
                        return skip

                    self.transformer.should_skip_forward_for_cached_states = decision

                    def count_block(*unused):
                        block_calls[0] += 1

                    hook_handles.append(self.transformer.blocks[0].register_forward_pre_hook(count_block))
                    # Capture one warmed full conditional forward, only for an
                    # explicitly named diagnostic request. Never profile normal
                    # business requests or mix these timings into benchmarks.
                    output_name = Path(batch.output_file_path()).stem
                    if output_name.startswith("trace-crop-"):
                        import torch.distributed as dist

                        transformer_forward_original = self.transformer.forward
                        trace_captured = [False]

                        def trace_forward(*args_, **kwargs_):
                            if (not trace_captured[0] and params.runtime_current_step == 1
                                    and not batch.is_cfg_negative):
                                trace_captured[0] = True
                                with torch.profiler.profile(activities=[
                                    torch.profiler.ProfilerActivity.CPU,
                                    torch.profiler.ProfilerActivity.CUDA,
                                ]) as profiler:
                                    with torch.profiler.record_function("videoedit_full_cond_step1"):
                                        result = transformer_forward_original(*args_, **kwargs_)
                                rank = dist.get_rank() if dist.is_initialized() else 0
                                trace_dir = Path(os.environ["VIDEOEDIT_SP_BENCH_DIR"]) / "traces"
                                trace_dir.mkdir(exist_ok=True)
                                profiler.export_chrome_trace(str(trace_dir / f"{output_name}-rank{rank}.json"))
                                (trace_dir / f"{output_name}-rank{rank}-operators.txt").write_text(
                                    profiler.key_averages().table(sort_by="self_device_time_total", row_limit=40))
                                return result
                            return transformer_forward_original(*args_, **kwargs_)

                        self.transformer.forward = trace_forward
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                status = "ok"
                denoise_start = None
                denoise_seconds = None
                try:
                    if resident and denoise:
                        if args.dit_layerwise_offload or args.use_fsdp_inference:
                            raise ValueError("Stage residency requires full unsharded DiT")
                        pipeline = self.pipeline()
                        for model in pipeline.modules.values():
                            if model is not self.transformer and isinstance(model, torch.nn.Module):
                                model.to("cpu")
                        torch.cuda.empty_cache()
                        self.transformer.to(module.get_local_torch_device())
                        if not all(p.device.type == "cuda" for p in self.transformer.parameters()):
                            raise RuntimeError("DiT was not fully resident")
                    if denoise:
                        torch.cuda.synchronize()
                        denoise_start = time.perf_counter()
                    result = original(self, batch, args)
                    if denoise:
                        torch.cuda.synchronize()
                        denoise_seconds = time.perf_counter() - denoise_start
                    return result
                except BaseException as exc:
                    status = f"{type(exc).__name__}: {exc}"
                    raise
                finally:
                    # Move full weights out before the next model stage/request.
                    try:
                        if resident and denoise:
                            self.transformer.to("cpu")
                            torch.cuda.empty_cache()
                        if resident and vae_stage:
                            self.vae.to("cpu")
                            torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        seconds = time.perf_counter() - start
                        fields = dict(stage=name, window=params.runtime_window_index,
                                      seconds=seconds, status=status,
                                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                      peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                                      model_frames=params.runtime_num_frames,
                                      model_height=params.runtime_height, model_width=params.runtime_width)
                        if name in ("VideoEditTextEncodingStage", "VideoEditImageEncodingStage"):
                            encoder = getattr(self, "text_encoder", None) if name == "VideoEditTextEncodingStage" else getattr(self, "image_encoder", None)
                            fields["encoder_cpu_master"] = os.environ.get("SGLANG_VIDEOEDIT_ENCODER_CPU_MASTER") == "1"
                            if encoder is not None:
                                fields["encoder_devices_after_stage"] = sorted({
                                    str(t.device) for t in list(encoder.parameters()) + list(encoder.buffers())
                                })
                        if name == "VideoEditConditioningEncodingStage":
                            fields.update(
                                text_seconds=getattr(self, "last_text_seconds", None),
                                clip_seconds=getattr(self, "last_clip_seconds", None),
                                positive_encode_calls=getattr(
                                    self.text_stage, "positive_encode_calls", None
                                ),
                                negative_encode_calls=getattr(
                                    self.text_stage, "negative_encode_calls", None
                                ),
                                cached_negative_text=getattr(
                                    self.text_stage, "cached_negative_text", None
                                ),
                            )
                        if vae_stage:
                            fields.update(
                                vae_sp=bool(getattr(args.pipeline_config, "vae_sp", False)),
                                vae_sp_mode=os.environ.get(
                                    "SGLANG_VIDEOEDIT_VAE_SP_MODE", "both"
                                ),
                                vae_encode_degree=int(
                                    os.environ.get(
                                        "SGLANG_VIDEOEDIT_VAE_ENCODE_DEGREE",
                                        str(args.sp_degree),
                                    )
                                ),
                                vae_decode_degree=int(
                                    os.environ.get(
                                        "SGLANG_VIDEOEDIT_VAE_DECODE_DEGREE",
                                        str(args.sp_degree),
                                    )
                                ),
                            )
                        if denoise:
                            fields.update(decisions=decisions, full_forwards=block_calls[0],
                                          latent_shape=list(params.runtime_latents.shape),
                                          denoise_seconds=denoise_seconds,
                                          overlap_prefetch=os.environ.get("SGLANG_VIDEOEDIT_OVERLAP_PREFETCH") == "1",
                                          fused_qkv=os.environ.get("SGLANG_VIDEOEDIT_FUSED_QKV") == "1")
                            fields["residency_plan"] = getattr(self, "residency_plan", None)
                            fields["resident_layers"] = int(os.environ.get("SGLANG_VIDEOEDIT_RESIDENT_LAYERS", "0"))
                            fields["usp_chunks_requested"] = int(os.environ.get("SGLANG_VIDEOEDIT_USP_CHUNKS", "1"))
                            fields["effective_resident_layers"] = [getattr(m, "resident_layers", 0) for m in offload_managers]
                            fields["effective_prefetch_layers"] = [getattr(m, "prefetch_size", 0) for m in offload_managers]
                            fields["shared_cpu_weight_bytes"] = sum(getattr(m, "shared_cpu_weight_bytes", 0) for m in offload_managers)
                            # PSS accounts shared pages proportionally; sum across ranks.
                            # Capture after the stage timer, not in the DiT interval.
                            try:
                                host_memory = {}
                                for line in Path("/proc/self/smaps_rollup").read_text().splitlines():
                                    parts = line.split()
                                    if parts[0].rstrip(":") in ("Rss", "Pss", "Private_Clean", "Private_Dirty", "Shared_Clean", "Shared_Dirty"):
                                        host_memory[parts[0].rstrip(":")] = int(parts[1]) * 1024
                                fields["host_memory_bytes"] = host_memory
                                shared_directory = os.environ.get("SGLANG_VIDEOEDIT_SHARED_CPU_DIR")
                                if shared_directory:
                                    shared_pages = {}
                                    active = False
                                    for line in Path("/proc/self/smaps").read_text().splitlines():
                                        parts = line.split()
                                        if "-" in parts[0]:
                                            active = shared_directory in line and ".bin" in line
                                        elif active and parts[0].rstrip(":") in ("Rss", "Pss", "Private_Clean", "Private_Dirty", "Shared_Clean", "Shared_Dirty", "Locked"):
                                            key = parts[0].rstrip(":")
                                            shared_pages[key] = shared_pages.get(key, 0) + int(parts[1]) * 1024
                                    fields["shared_master_mapping_bytes"] = shared_pages
                                    files = list(Path(shared_directory).glob("*.bin"))
                                    fields["shared_store_file_bytes"] = sum(p.stat().st_size for p in files)
                                    fields["shared_store_files"] = len(files)
                            except OSError:
                                pass
                            fields["pinned_master_bytes"] = sum(getattr(m, "pinned_weight_bytes", 0) for m in offload_managers)
                            pools = [m.staging_pool for m in offload_managers if getattr(m, "staging_pool", None) is not None]
                            fields["pinned_staging_bytes"] = sum(p.pinned_bytes for p in pools)
                            fields["pageable_fallback_copies"] = sum(p.fallback_copies for p in pools)
                            fields["offload_pool_slots_after_stage"] = [
                                len(getattr(m, "_retired_gpu_buffers", [])) for m in offload_managers
                            ]
                            fields["release_after_denoise"] = os.environ.get("SGLANG_VIDEOEDIT_RELEASE_AFTER_DENOISE") == "1"
                            fields["gpu_layers_after_stage"] = [len(m._gpu_layers) for m in offload_managers]
                            fields["h2d_weight_bytes"] = sum(getattr(m, "h2d_weight_bytes", 0) for m in offload_managers) - transfer_start[0]
                            fields["h2d_layer_loads"] = sum(getattr(m, "h2d_layer_loads", 0) for m in offload_managers) - transfer_start[1]
                        emit("stage", batch, **fields)
                        if denoise and status == "ok":
                            hash_start = time.perf_counter()
                            digest = tensor_hash(params.runtime_latents)
                            emit("latent", batch, window=params.runtime_window_index,
                                 sha256=digest, seconds=time.perf_counter() - hash_start)
                    finally:
                        if transformer_forward_original is not None:
                            self.transformer.forward = transformer_forward_original
                        if decision_original is not None:
                            self.transformer.should_skip_forward_for_cached_states = decision_original
                        for handle in hook_handles:
                            handle.remove()
            return forward

        cls.forward = wrap(original, name)


def install_pipeline(module):
    import torch

    original = module.WanVideoEditPipeline.forward

    @functools.wraps(original)
    def forward(self, batch, args):
        torch.cuda.synchronize()
        start = time.perf_counter()
        status = "ok"
        try:
            return original(self, batch, args)
        except BaseException as exc:
            status = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            torch.cuda.synchronize()
            emit("request", batch, seconds=time.perf_counter() - start, status=status)

    module.WanVideoEditPipeline.forward = forward
