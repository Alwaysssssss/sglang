import os
import re
from itertools import chain
from typing import Any, Dict, List, Set, Tuple

import torch

from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.videoedit_memory import (
    PinnedStagingPool,
    env_nonnegative_int,
    select_weight_budget,
)

logger = init_logger(__name__)


def _empty_cpu_tensor(
    shape: tuple[int, ...], dtype: torch.dtype, pin_memory: bool
) -> torch.Tensor:
    try:
        return torch.empty(shape, dtype=dtype, pin_memory=pin_memory)
    except (RuntimeError, TypeError):
        if not pin_memory:
            raise
        return torch.empty(shape, dtype=dtype)


def _empty_strided_cpu_tensor(
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    dtype: torch.dtype,
    pin_memory: bool,
) -> torch.Tensor:
    try:
        return torch.empty_strided(
            shape, stride, dtype=dtype, device="cpu", pin_memory=pin_memory
        )
    except (RuntimeError, TypeError):
        if not pin_memory:
            raise
        return torch.empty_strided(shape, stride, dtype=dtype, device="cpu")


def _empty_strided_device_tensor(
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    return torch.empty_strided(shape, stride, dtype=dtype, device=device)


# Adapted from skywork AI Infra diffusion optimize
class LayerwiseOffloadManager:
    """A lightweight layerwise CPU offload manager.

    This utility offloads per-layer parameters/buffers from GPU to CPU, and
    supports async H2D prefetch using a dedicated CUDA stream.

    Typical usage:
    - Construct the manager with the target model and the list-like module
      attribute that represents transformer blocks (e.g. ``blocks``).
    - Call :meth:`initialize` once to offload weights and prefetch layer 0.
    - During forward, call :meth:`prefetch_layer` for the next layer and
      :meth:`release_layer` for the finished layer.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        layers_attr_str: str,
        num_layers: int,
        enabled: bool,
        pin_cpu_memory: bool = True,
        prefetch_size: int = 1,
    ) -> None:
        self.model = model
        self.layers_attr_str = layers_attr_str
        self.num_layers = num_layers
        self.pin_cpu_memory = pin_cpu_memory
        self.prefetch_size = min(max(1, prefetch_size), self.num_layers)
        self.prefetch_cap = self.prefetch_size
        # Opt-in while validated against the VideoEdit workload. Removing the
        # copy->compute stream dependency requires explicit allocation lifetimes.
        self.overlap_prefetch = (
            os.environ.get("SGLANG_VIDEOEDIT_OVERLAP_PREFETCH") == "1"
            and current_platform.device_type == "cuda"
        )
        self.auto_residency = (
            os.environ.get("SGLANG_VIDEOEDIT_AUTO_RESIDENCY") == "1"
            and current_platform.device_type == "cuda"
        )
        self.explicit_overlap_prefetch = self.overlap_prefetch
        self.overlap_prefetch = self.overlap_prefetch or self.auto_residency
        # A bounded read-only weight cache; release_all still releases every
        # layer. Default zero preserves the existing offload policy.
        self.resident_layers = int(os.environ.get("SGLANG_VIDEOEDIT_RESIDENT_LAYERS", "0"))
        self.release_after_denoise = (
            os.environ.get("SGLANG_VIDEOEDIT_RELEASE_AFTER_DENOISE") == "1"
        )
        if not 0 <= self.resident_layers <= self.num_layers:
            raise ValueError(
                "SGLANG_VIDEOEDIT_RESIDENT_LAYERS must be within the layer count"
            )
        self.enabled = bool(enabled and torch.get_device_module().is_available())
        if not self.enabled:
            return
        self.device = torch.device(
            current_platform.device_type, torch.get_device_module().current_device()
        )
        self.copy_stream = torch.get_device_module().Stream()
        staging_mb = env_nonnegative_int("SGLANG_VIDEOEDIT_OFFLOAD_PINNED_MB")
        self.staging_pool = (
            PinnedStagingPool(staging_mb * 1024**2)
            if staging_mb and pin_cpu_memory and current_platform.device_type == "cuda"
            else None
        )

        self.shared_cpu_dir = os.environ.get("SGLANG_VIDEOEDIT_SHARED_CPU_DIR", "")
        if self.shared_cpu_dir and self.staging_pool is not None:
            raise ValueError("Shared CPU weights and bounded offload staging are alternative policies")
        if self.shared_cpu_dir and current_platform.device_type != "cuda":
            raise ValueError("Shared CPU offload currently requires CUDA")
        self._shared_cpu_keys: set[tuple[int, torch.dtype]] = set()
        self.shared_cpu_weight_bytes = 0

        self._layer_name_re = re.compile(
            rf"(^|\.){re.escape(layers_attr_str)}\.(\d+)(\.|$)"
        )

        # layer_idx -> {dtype: consolidated_pinned_cpu_tensor}
        # stores the consolidated weight from a same layer, of same dtype
        self._consolidated_cpu_weights: Dict[int, Dict[torch.dtype, torch.Tensor]] = {}
        # layer_idx -> {name: layout-preserving CPU tensor}
        # stores tensors whose stride must be preserved (e.g. FP8 column-major weights)
        self._strided_cpu_weights: Dict[int, Dict[str, torch.Tensor]] = {}
        # layer_idx -> {name: {dtype, offset, numel, shape, stride, storage}}
        # stores the offset and numel of each weight in flat storage, or the
        # layout metadata for stride-preserving storage.
        self._weight_metadata: Dict[int, Dict[str, Dict[str, Any]]] = {}
        # layer indices that are already in gpu
        self._gpu_layers: Set[int] = set()
        self.h2d_weight_bytes = 0
        self.h2d_layer_loads = 0
        self._gpu_buffers: Dict[int, Dict[torch.dtype, torch.Tensor]] = {}
        self._strided_gpu_buffers: Dict[int, Dict[str, torch.Tensor]] = {}
        # Recycle flat staging buffers only after the previous consumer finishes.
        # A rolling lookahead bounds uniform-layer storage to prefetch_size + 1
        # slots, plus resident layers, even when Python runs ahead of the GPU.
        self._retired_gpu_buffers: list[tuple[dict, Any]] = []
        # layer_idx -> torch.get_device_module().Event for fine-grained sync, to make sure the weight is resident in pre-hook
        self._prefetch_events: Dict[int, torch.get_device_module().Event] = {}

        self._named_parameters: Dict[str, torch.nn.Parameter] = {}
        self._named_buffers: Dict[str, torch.Tensor] = {}
        self._offload_placeholders: Dict[torch.dtype, torch.Tensor] = {}
        # Store forward hooks for removal
        self._forward_hooks: List[Any] = []

        self._initialize()

    def _match_layer_idx(self, name: str) -> int | None:
        m = self._layer_name_re.search(name)
        if not m:
            return None
        try:
            return int(m.group(2))
        except Exception:
            return None

    def _get_shared_empty_tensor(self, dtype: torch.dtype) -> torch.Tensor:
        placeholder = self._offload_placeholders.get(dtype)
        if placeholder is None:
            placeholder = torch.empty((1,), device=self.device, dtype=dtype)
            self._offload_placeholders[dtype] = placeholder
        return placeholder

    @torch.compiler.disable
    def _initialize(self) -> None:
        if not self.enabled:
            return

        self._named_parameters = dict(self.model.named_parameters())
        self._named_buffers = dict(self.model.named_buffers())

        # 1. collect and group tensors by layer and dtype
        layer_groups: Dict[int, Dict[torch.dtype, List[Tuple[str, torch.Tensor]]]] = {}
        all_tensors = chain(self._named_parameters.items(), self._named_buffers.items())
        for name, tensor in all_tensors:
            layer_idx = self._match_layer_idx(name)
            if layer_idx is None or layer_idx >= self.num_layers:
                continue
            layer_groups.setdefault(layer_idx, {}).setdefault(tensor.dtype, []).append(
                (name, tensor)
            )

        # 2. concat and offload (in pinned memory)
        for layer_idx, dtype_to_params in layer_groups.items():
            self._consolidated_cpu_weights[layer_idx] = {}
            self._strided_cpu_weights[layer_idx] = {}
            self._weight_metadata[layer_idx] = {}

            for dtype, weights in dtype_to_params.items():
                flat_weights = [(name, t) for name, t in weights if t.is_contiguous()]
                strided_weights = [
                    (name, t) for name, t in weights if not t.is_contiguous()
                ]

                cpu_buffer = None
                if flat_weights:
                    total_numel = sum(t.numel() for _, t in flat_weights)
                    # create concatenated CPU buffer (in pinned memory)
                    cpu_buffer = _empty_cpu_tensor(
                        (total_numel,), dtype=dtype,
                        pin_memory=self.pin_cpu_memory and self.staging_pool is None and not self.shared_cpu_dir,
                    )

                # offload weights to the buffer
                current_offset = 0
                for name, weight in flat_weights:
                    assert cpu_buffer is not None
                    numel = weight.numel()
                    cpu_buffer[current_offset : current_offset + numel].copy_(
                        weight.flatten()
                    )
                    self._weight_metadata[layer_idx][name] = {
                        "dtype": dtype,
                        "offset": current_offset,
                        "numel": numel,
                        "shape": tuple(weight.shape),
                        "stride": tuple(weight.stride()),
                        "storage": "flat",
                    }

                    weight.data = self._get_shared_empty_tensor(dtype)

                    current_offset += numel

                if cpu_buffer is not None:
                    if self.shared_cpu_dir:
                        from sglang.multimodal_gen.runtime.utils.shared_cpu_weights import share_cpu_tensor

                        cpu_buffer = share_cpu_tensor(
                            cpu_buffer, self.shared_cpu_dir,
                            pin_memory=self.pin_cpu_memory, device=self.device,
                        )
                        self._shared_cpu_keys.add((layer_idx, dtype))
                        self.shared_cpu_weight_bytes += cpu_buffer.numel() * cpu_buffer.element_size()
                    self._consolidated_cpu_weights[layer_idx][dtype] = cpu_buffer

                for name, weight in strided_weights:
                    shape = tuple(weight.shape)
                    stride = tuple(weight.stride())
                    cpu_weight = _empty_strided_cpu_tensor(
                        shape, stride, dtype=dtype,
                        pin_memory=self.pin_cpu_memory and self.staging_pool is None,
                    )
                    cpu_weight.copy_(weight)
                    self._strided_cpu_weights[layer_idx][name] = cpu_weight
                    self._weight_metadata[layer_idx][name] = {
                        "dtype": dtype,
                        "offset": None,
                        "numel": weight.numel(),
                        "shape": shape,
                        "stride": stride,
                        "storage": "strided",
                    }

                    weight.data = self._get_shared_empty_tensor(dtype)

        self.cpu_weight_bytes = sum(
            t.untyped_storage().nbytes()
            for group in (self._consolidated_cpu_weights, self._strided_cpu_weights)
            for tensors in group.values() for t in tensors.values()
        )
        self.pinned_weight_bytes = sum(
            t.untyped_storage().nbytes()
            for group in (self._consolidated_cpu_weights, self._strided_cpu_weights)
            for tensors in group.values() for t in tensors.values() if t.is_pinned()
        )
        if self.pin_cpu_memory and self.staging_pool is None and self.pinned_weight_bytes < self.cpu_weight_bytes:
            logger.warning("Some offload masters are pageable: pinned=%d total=%d bytes",
                           self.pinned_weight_bytes, self.cpu_weight_bytes)
        # prefetch the first layer for warm-up
        self.prepare_for_next_req(non_blocking=False)

        self.register_forward_hooks()
        logger.info(
            f"LayerwiseOffloadManager initialized with num prefetched layer: {self.prefetch_size}, total num layers: {self.num_layers}, overlap_prefetch={self.overlap_prefetch}, resident_layers={self.resident_layers}"
        )

    def prepare_for_next_req(self, non_blocking=True):
        """
        Prepare for the next round of denoising loop with prefetching the necessary layers
        """
        for i in range(self.prefetch_size):
            self.prefetch_layer(i, non_blocking=non_blocking)
        if not non_blocking and self.copy_stream is not None:
            torch.get_device_module().current_stream().wait_stream(self.copy_stream)

    def get_target_with_name(self, name: str) -> torch.Tensor:
        """get the target model weight/buffer to be replaced"""
        if name in self._named_parameters:
            target = self._named_parameters[name]
        else:
            target = self._named_buffers[name]
        return target

    @torch.compiler.disable
    def prefetch_layer(self, layer_idx: int, non_blocking: bool = True) -> None:
        """
        idempotent
        """
        if not self.enabled or self.device is None or self.copy_stream is None:
            return
        if layer_idx < 0 or layer_idx >= self.num_layers:
            return
        if layer_idx in self._gpu_layers:
            return
        if (
            layer_idx not in self._consolidated_cpu_weights
            and layer_idx not in self._strided_cpu_weights
        ):
            return
        if not self.overlap_prefetch:
            self.copy_stream.wait_stream(torch.get_device_module().current_stream())

        # create gpu buffer and load from CPU buffer
        gpu_buffers: Dict[torch.dtype, torch.Tensor] = {}
        strided_gpu_weights: Dict[str, torch.Tensor] = {}
        if self.overlap_prefetch and self._retired_gpu_buffers:
            gpu_buffers, consumed = self._retired_gpu_buffers.pop(0)
            self.copy_stream.wait_event(consumed)
        with torch.get_device_module().stream(self.copy_stream):
            cpu_buffers = self._consolidated_cpu_weights.get(layer_idx, {})
            gpu_buffers = {
                dtype: tensor
                for dtype, tensor in gpu_buffers.items()
                if dtype in cpu_buffers and tensor.shape == cpu_buffers[dtype].shape
            }
            for dtype, cpu_buffer in self._consolidated_cpu_weights.get(
                layer_idx, {}
            ).items():
                gpu_buffer = gpu_buffers.get(dtype)
                if gpu_buffer is None:
                    gpu_buffer = torch.empty(
                        cpu_buffer.shape, dtype=dtype, device=self.device
                    )
                if self.staging_pool is not None:
                    self.staging_pool.copy_to(gpu_buffer, cpu_buffer, self.copy_stream)
                else:
                    gpu_buffer.copy_(cpu_buffer, non_blocking=non_blocking)
                gpu_buffers[dtype] = gpu_buffer
            for name, cpu_weight in self._strided_cpu_weights.get(
                layer_idx, {}
            ).items():
                meta = self._weight_metadata[layer_idx][name]
                gpu_weight = _empty_strided_device_tensor(
                    meta["shape"], meta["stride"], meta["dtype"], self.device
                )
                if self.staging_pool is not None:
                    self.staging_pool.copy_to(gpu_weight, cpu_weight, self.copy_stream)
                else:
                    gpu_weight.copy_(cpu_weight, non_blocking=non_blocking)
                strided_gpu_weights[name] = gpu_weight

        # record the prefetch event of this layer
        event = torch.get_device_module().Event()
        event.record(self.copy_stream)
        self._prefetch_events[layer_idx] = event
        if self.staging_pool is not None and not non_blocking:
            # load_all_layers can remove the consuming hooks; preserve its
            # blocking-copy contract even though staging submits async DMA.
            event.synchronize()

        # restore model's weights by their metadata using gpu buffer
        for name, meta in self._weight_metadata[layer_idx].items():
            target = self.get_target_with_name(name)
            if meta["storage"] == "flat":
                dtype = meta["dtype"]
                gpu_buffer = gpu_buffers[dtype]
                # map the parameter's data to the correct slice of the GPU buffer
                target.data = gpu_buffer[
                    meta["offset"] : meta["offset"] + meta["numel"]
                ].view(meta["shape"])
            else:
                target.data = strided_gpu_weights[name]

        self._gpu_layers.add(layer_idx)
        self.h2d_layer_loads += 1
        self.h2d_weight_bytes += sum(
            tensor.numel() * tensor.element_size()
            for tensor in chain(gpu_buffers.values(), strided_gpu_weights.values())
        )
        if self.overlap_prefetch or self.resident_layers:
            self._gpu_buffers[layer_idx] = gpu_buffers
            self._strided_gpu_buffers[layer_idx] = strided_gpu_weights

    @torch.compiler.disable
    def release_layer(self, layer_idx: int, *, force: bool = False) -> None:
        """
        lightweight release layer weights
        Basically set the reference count to the gpu weight tensor to zero. The weights on cpu is untouched
        """
        if not self.enabled or self.device is None:
            return
        if layer_idx < self.resident_layers and not force:
            return

        # clear prefetch event, since it's useless and needs to be reset
        self._prefetch_events.pop(layer_idx, None)

        if layer_idx not in self._gpu_layers:
            return

        for name, meta in self._weight_metadata.get(layer_idx, {}).items():
            target = self.get_target_with_name(name)
            # Wraparound prefetch will reload the layer when it is needed again
            target.data = self._get_shared_empty_tensor(meta["dtype"])

        self._gpu_layers.discard(layer_idx)
        # record_stream in the consuming pre-hook protects these allocations
        # until all queued kernels on that stream have finished using them.
        buffers = self._gpu_buffers.pop(layer_idx, None)
        if self.overlap_prefetch and buffers and not force:
            consumed = torch.get_device_module().Event()
            consumed.record(torch.get_device_module().current_stream())
            self._retired_gpu_buffers.append((buffers, consumed))
        self._strided_gpu_buffers.pop(layer_idx, None)

    @torch.compiler.disable
    def release_all(self) -> None:
        if not self.enabled or self.device is None:
            return
        if self.copy_stream is not None:
            torch.get_device_module().current_stream().wait_stream(self.copy_stream)

        for layer_idx in list(self._gpu_layers):
            self.release_layer(layer_idx, force=True)
        self._retired_gpu_buffers.clear()

    def layer_bytes(self) -> list[int]:
        """Device bytes each layer occupies once it is resident."""
        return [sum(
            t.untyped_storage().nbytes()
            for group in (self._consolidated_cpu_weights, self._strided_cpu_weights)
            for t in group.get(index, {}).values()
        ) for index in range(self.num_layers)]

    def configure_memory_budget(self, budget_bytes: int) -> None:
        """Reconfigure between denoising stages; excludes activation memory."""
        if not (self.overlap_prefetch or self.auto_residency):
            raise ValueError("adaptive residency requires overlap prefetch")
        sizes = self.layer_bytes()
        # Small inputs retain the tested full-cache batch schedule. Partial
        # residency uses one-layer rolling lookahead, leaving more room for
        # cached weights and avoiding oversized transient prefetch groups.
        cap = 1 if self.auto_residency else self.prefetch_cap
        resident, prefetch = select_weight_budget(sizes, budget_bytes, cap)
        self.release_all()
        if self.auto_residency:
            self.overlap_prefetch = self.explicit_overlap_prefetch or resident < self.num_layers
            if resident == self.num_layers:
                prefetch = self.prefetch_cap
        remaining = sorted(sizes[resident:], reverse=True)
        self.weight_capacity_bytes = sum(sizes[:resident]) + sum(remaining[:prefetch + 1])
        self.resident_layers, self.prefetch_size = resident, prefetch
        logger.info("Offload budget=%d bytes: resident=%d prefetch=%d",
                    budget_bytes, resident, prefetch)

    @torch.compiler.disable
    def load_all_layers(self) -> None:
        """Load all layers from CPU to GPU."""
        if not self.enabled or self.device is None:
            return
        if self.copy_stream is not None:
            torch.get_device_module().current_stream().wait_stream(self.copy_stream)

        for layer_idx in range(self.num_layers):
            if layer_idx not in self._gpu_layers:
                self.prefetch_layer(layer_idx, non_blocking=False)

    def _detach_shared_cpu_layer(self, layer_idx: int) -> None:
        """Detach before refit/LoRA writes; other ranks retain immutable originals."""
        keys = [key for key in self._shared_cpu_keys if key[0] == layer_idx]
        if not keys:
            return
        self.copy_stream.synchronize()
        for key in keys:
            _, dtype = key
            original = self._consolidated_cpu_weights[layer_idx][dtype]
            private = _empty_cpu_tensor(tuple(original.shape), dtype, self.pin_cpu_memory)
            private.copy_(original)
            self._consolidated_cpu_weights[layer_idx][dtype] = private
            size = original.numel() * original.element_size()
            self.shared_cpu_weight_bytes -= size
            self.pinned_weight_bytes += size * (int(private.is_pinned()) - int(original.is_pinned()))
            self._shared_cpu_keys.remove(key)

    @torch.compiler.disable
    def sync_layer_to_cpu(self, layer_idx: int) -> None:
        """Sync a layer's weights from GPU back to CPU."""
        if not self.enabled or layer_idx not in self._gpu_layers:
            return
        if (
            layer_idx not in self._consolidated_cpu_weights
            and layer_idx not in self._strided_cpu_weights
        ):
            return

        if self.copy_stream is not None:
            torch.get_device_module().current_stream().wait_stream(self.copy_stream)

        self._detach_shared_cpu_layer(layer_idx)

        # Collect current GPU weights and write back to CPU buffer
        for name, meta in self._weight_metadata.get(layer_idx, {}).items():
            target = self.get_target_with_name(name)
            if meta["storage"] == "flat":
                gpu_weight = target.data.flatten().cpu()
                dtype = meta["dtype"]
                cpu_buffer = self._consolidated_cpu_weights[layer_idx][dtype]
                offset = meta["offset"]
                numel = meta["numel"]
                cpu_buffer[offset : offset + numel].copy_(gpu_weight)
            else:
                self._strided_cpu_weights[layer_idx][name].copy_(
                    target.data.detach().cpu()
                )

    @torch.compiler.disable
    def sync_all_layers_to_cpu(self) -> None:
        """Sync all loaded layers' weights from GPU back to CPU."""
        if not self.enabled or self.device is None:
            return
        if self.copy_stream is not None:
            torch.get_device_module().current_stream().wait_stream(self.copy_stream)

        for layer_idx in list(self._gpu_layers):
            self.sync_layer_to_cpu(layer_idx)

    @torch.compiler.disable
    def update_cpu_weights(
        self, weight_dict: Dict[str, torch.Tensor]
    ) -> Set[str] | None:
        """Update consolidated CPU buffers with new weights.

        When layerwise offload (--dit-layerwise-offload) is enabled, the
        offload manager replaces GPU parameters with small torch.empty((1,))
        placeholders while real weights live in consolidated pinned CPU
        buffers.

        The refit process writes new weights directly into the CPU buffers,
        bypassing the placeholders.  For any layer that happens to be resident
        on the GPU at update time, the live GPU tensor is also updated.

        Args:
            weight_dict: Mapping of parameter name to new weight tensor.

        Returns:
            Set of parameter names that were successfully updated.

        Raises:
            ValueError: If a weight's shape does not match the recorded
                metadata (i.e., the real shape, not the placeholder shape).
        """
        if not self.enabled:
            return None

        # CPU masters may still be DMA sources for an outstanding prefetch.
        # A stream wait alone cannot protect a host-side write to pinned memory.
        if self.copy_stream is not None:
            self.copy_stream.synchronize()
        updated_names: Set[str] = set()
        for name, loaded_weight in weight_dict.items():
            layer_idx = self._match_layer_idx(name)
            if layer_idx is None:
                continue
            meta_layer = self._weight_metadata.get(layer_idx)
            if meta_layer is None or name not in meta_layer:
                continue

            meta = meta_layer[name]
            if tuple(meta["shape"]) != tuple(loaded_weight.shape):
                raise ValueError(
                    f"Shape mismatch for {name}: "
                    f"expected={tuple(meta['shape'])}, "
                    f"loaded={tuple(loaded_weight.shape)}"
                )

            self._detach_shared_cpu_layer(layer_idx)
            dtype = meta["dtype"]
            if meta["storage"] == "flat":
                offset = meta["offset"]
                numel = meta["numel"]
                cpu_buffer = self._consolidated_cpu_weights[layer_idx][dtype]
                cpu_buffer[offset : offset + numel].copy_(
                    loaded_weight.to(dtype=dtype).flatten()
                )
            else:
                self._strided_cpu_weights[layer_idx][name].copy_(
                    loaded_weight.to(dtype=dtype)
                )

            # If this layer is currently on GPU, update the live parameter.
            if layer_idx in self._gpu_layers:
                target = self.get_target_with_name(name)
                target.data.copy_(loaded_weight.to(dtype=target.dtype))

            updated_names.add(name)

        return updated_names

    def iter_cpu_weights(self):
        """Yield (name, tensor) pairs from consolidated CPU buffers.

        This reconstructs the original weight tensors (with correct shapes)
        from the flat CPU buffers using stored metadata.  Unlike
        model.named_parameters(), which returns (1,) placeholders
        when offload is enabled, this method returns the real weights and
        can be used for checksum computation.
        """
        for layer_idx in sorted(self._weight_metadata):
            for name, meta in self._weight_metadata[layer_idx].items():
                if meta["storage"] == "flat":
                    dtype = meta["dtype"]
                    offset = meta["offset"]
                    numel = meta["numel"]
                    shape = meta["shape"]
                    cpu_buffer = self._consolidated_cpu_weights[layer_idx][dtype]
                    yield name, cpu_buffer[offset : offset + numel].reshape(shape)
                else:
                    yield name, self._strided_cpu_weights[layer_idx][name]

    def register_forward_hooks(self) -> None:
        if not self.enabled:
            return

        layers = getattr(self.model, self.layers_attr_str)

        def make_pre_hook(i):
            def hook(module, input):
                # wait only for the current layer if it's being prefetched
                if i == 0:
                    self.prepare_for_next_req(non_blocking=self.overlap_prefetch)
                if self.overlap_prefetch:
                    # Demand loading also makes a new forward safe after release_all.
                    self.prefetch_layer(i, non_blocking=True)
                if i in self._prefetch_events:
                    torch.get_device_module().current_stream().wait_event(
                        self._prefetch_events[i]
                    )
                if self.overlap_prefetch or self.resident_layers:
                    consumer = torch.get_device_module().current_stream()
                    for weight in chain(
                        self._gpu_buffers.get(i, {}).values(),
                        self._strided_gpu_buffers.get(i, {}).values(),
                    ):
                        weight.record_stream(consumer)

                if self.overlap_prefetch:
                    # Keep a rolling window of upcoming nonresident layers. Do not
                    # wrap: TeaCache may skip the next forward entirely.
                    if self.prefetch_size == 0:
                        return
                    scheduled = 0
                    for j in range(i + 1, self.num_layers):
                        if j < self.resident_layers and j in self._gpu_layers:
                            continue
                        self.prefetch_layer(j, non_blocking=True)
                        scheduled += 1
                        if scheduled == self.prefetch_size:
                            break
                # Preserve the legacy batch schedule when overlap is disabled.
                elif i % self.prefetch_size == 0:
                    scheduled = 0
                    for offset in range(self.num_layers):
                        layer_to_prefetch = (
                            i + self.prefetch_size + offset
                        ) % self.num_layers
                        # Reuse cached layers without spending the lookahead
                        # budget on them. During the cold pass they still load.
                        if (
                            layer_to_prefetch < self.resident_layers
                            and layer_to_prefetch in self._gpu_layers
                        ):
                            continue
                        self.prefetch_layer(layer_to_prefetch, non_blocking=True)
                        scheduled += 1
                        if scheduled == self.prefetch_size:
                            break

            return hook

        def make_post_hook(i):
            def hook(module, input, output):
                # previous, we wait here, until the copy stream for next layer is finished,
                # now with any prefetch_size, only wait for the copy stream, when the copy stream is for the next layer
                self.release_layer(i)

            return hook

        # register prefetch & release hooks for each layer
        self._forward_hooks.clear()
        for i, layer in enumerate(layers):
            pre_hook_handle = layer.register_forward_pre_hook(make_pre_hook(i))
            post_hook_handle = layer.register_forward_hook(make_post_hook(i))
            self._forward_hooks.extend([pre_hook_handle, post_hook_handle])

    def remove_forward_hooks(self) -> None:
        """Remove all registered forward hooks."""
        for hook_handle in self._forward_hooks:
            hook_handle.remove()
        self._forward_hooks.clear()


class OffloadableDiTMixin:
    """
    A mixin that registers forward hooks for a DiT to enable layerwise offload
    """

    # the list of names of a DiT's layers/blocks
    layer_names: List[str]
    layerwise_offload_managers: list[LayerwiseOffloadManager] = []

    def configure_layerwise_offload(self, server_args: ServerArgs):
        self.layerwise_offload_managers = []
        for layer_name in self.layer_names:
            # a manager per layer-list
            module_list = getattr(self, layer_name, None)
            if module_list is None or not isinstance(module_list, torch.nn.ModuleList):
                continue

            num_layers = len(module_list)
            if server_args.dit_offload_prefetch_size < 1.0:
                prefetch_size = 1 + int(
                    round(server_args.dit_offload_prefetch_size * (num_layers - 1))
                )
            else:
                prefetch_size = int(server_args.dit_offload_prefetch_size)

            manager = LayerwiseOffloadManager(
                model=self,
                layers_attr_str=layer_name,
                num_layers=num_layers,
                enabled=True,
                pin_cpu_memory=server_args.pin_cpu_memory,
                prefetch_size=prefetch_size,
            )
            self.layerwise_offload_managers.append(manager)

        logger.info(
            f"Enabled layerwise offload for {self.__class__.__name__} on modules: {self.layer_names}"
        )

    def prepare_for_next_req(self):
        if self.layerwise_offload_managers is None:
            return
        for manager in self.layerwise_offload_managers:
            manager.prepare_for_next_req(non_blocking=True)

    def disable_offload(self) -> None:
        """Disable layerwise offload: load all layers to GPU and remove hooks."""
        if self.layerwise_offload_managers is None:
            return
        for manager in self.layerwise_offload_managers:
            if manager.enabled:
                manager.remove_forward_hooks()
                manager.load_all_layers()

    def enable_offload(self) -> None:
        """Re-enable layerwise offload: sync weights to CPU, release layers, and restore hooks."""
        if self.layerwise_offload_managers is None:
            return
        for manager in self.layerwise_offload_managers:
            if manager.enabled:
                manager.sync_all_layers_to_cpu()
                manager.release_all()
                manager.register_forward_hooks()


def iter_materialized_weights(module: torch.nn.Module):
    """Yield (name, tensor) pairs with materialized weights, even under offload.

    When layerwise offload is active, module.named_parameters() returns
    (1,) placeholders for offloaded layers.  This function reads the
    actual data from the offload manager's CPU buffers and chains it with
    the non-offloaded parameters.
    """
    offload_managers: list = []
    if isinstance(module, OffloadableDiTMixin) and module.layerwise_offload_managers:
        offload_managers = [m for m in module.layerwise_offload_managers if m.enabled]

    if not offload_managers:
        yield from module.named_parameters()
        return

    # Collect offloaded names and their real tensors from CPU buffers.
    offloaded_names: set[str] = set()
    for manager in offload_managers:
        for name, tensor in manager.iter_cpu_weights():
            offloaded_names.add(name)
            yield name, tensor

    # Yield non-offloaded parameters (e.g. final norms, embeddings).
    for name, param in module.named_parameters():
        if name not in offloaded_names:
            yield name, param
