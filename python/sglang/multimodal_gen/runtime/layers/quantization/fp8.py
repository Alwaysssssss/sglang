from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_tensor_model_parallel_world_size,
)
from sglang.multimodal_gen.runtime.layers.linear import (
    LinearMethodBase,
    UnquantizedLinearMethod,
    maybe_log_linear_runtime_shape,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.multimodal_gen.runtime.models.parameter import (
    BlockQuantScaleParameter,
    ChannelQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.common import (
    cpu_has_amx_support,
    get_bool_env_var,
    use_intel_amx_backend,
)
from sglang.srt.layers.amx_utils import _amx_process_weight_after_loading
from sglang.srt.layers.quantization.fp8_kernel import (
    fp8_dtype,
    fp8_max,
    fp8_min,
    is_fp8_fnuz,
    per_token_group_quant_fp8,
)
from sglang.srt.layers.quantization.fp8_utils import (
    FP8_GEMM_BACKENDS,
    apply_fp8_linear,
    can_auto_enable_marlin_fp8,
    cutlass_fp8_supported,
    dispatch_w8a8_block_fp8_linear,
    input_to_float8,
    normalize_e4m3fn_to_e4m3fnuz,
    requant_weight_ue8m0_inplace,
    resolve_fp8_linear_gemm_backend,
)
from sglang.srt.layers.quantization.marlin_utils_fp8 import (
    apply_fp8_marlin_linear,
    prepare_fp8_layer_for_marlin,
)
from sglang.srt.layers.quantization.utils import (
    convert_to_channelwise,
    is_layer_skipped,
    requantize_with_max_scale,
)

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.w4afp8 import W4AFp8Config

_is_hip = current_platform.is_hip()
_is_cuda = current_platform.is_cuda()
_is_npu = current_platform.is_npu()
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = current_platform.is_cpu()
_is_fp8_fnuz = is_fp8_fnuz()
_use_hip_int4 = get_bool_env_var("SGLANG_INT4_WEIGHT") and _is_hip
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if _use_aiter or _use_hip_int4:
    pass


ACTIVATION_SCHEMES = ["static", "dynamic"]
WEIGHT_SCALE_GRANULARITIES = ["tensor", "channel"]
FAKE_W8A8_DEQUANT_ENV = "SGLANG_DIFFUSION_FAKE_FP8_W8A8_DEQUANT"

logger = logging.getLogger(__name__)


def _fake_quant_dequant_fp8_per_token(x: torch.Tensor) -> torch.Tensor:
    x_2d = x.reshape(-1, x.shape[-1])
    x_fp32 = x_2d.float()
    scale = x_fp32.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / fp8_max
    q_x = torch.clamp(x_fp32 / scale, fp8_min, fp8_max).to(fp8_dtype)
    return (q_x.float() * scale).to(x.dtype).reshape_as(x)


def _fake_quant_fp8_weight_per_channel(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight_fp32 = weight.float()
    scale = weight_fp32.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / fp8_max
    qweight = torch.clamp(weight_fp32 / scale, fp8_min, fp8_max).to(fp8_dtype)
    return qweight, scale.t().contiguous()


class Fp8Config(QuantizationConfig):
    """Config class for FP8."""

    def __init__(
        self,
        is_checkpoint_fp8_serialized: bool = False,
        activation_scheme: str = "dynamic",
        ignored_layers: Optional[List[str]] = None,
        weight_block_size: List[int] = None,
        gemm_backend: str = "auto",
        weight_scale_granularity: str | None = None,
    ) -> None:
        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized
        if is_checkpoint_fp8_serialized:
            logger.info("Detected fp8 checkpoint.")
        if activation_scheme not in ACTIVATION_SCHEMES:
            raise ValueError(f"Unsupported activation scheme {activation_scheme}")
        self.activation_scheme = activation_scheme
        if gemm_backend not in FP8_GEMM_BACKENDS:
            raise ValueError(
                f"Unsupported FP8 GEMM backend {gemm_backend!r}; "
                f"expected one of {FP8_GEMM_BACKENDS}."
            )
        self.gemm_backend = gemm_backend
        if weight_scale_granularity is None:
            weight_scale_granularity = (
                "tensor" if is_checkpoint_fp8_serialized else "channel"
            )
        if weight_scale_granularity not in WEIGHT_SCALE_GRANULARITIES:
            raise ValueError(
                "Unsupported FP8 weight scale granularity "
                f"{weight_scale_granularity!r}; expected one of "
                f"{WEIGHT_SCALE_GRANULARITIES}."
            )
        if weight_block_size is not None and weight_scale_granularity != "tensor":
            raise ValueError(
                "weight_scale_granularity applies only to non-block FP8 weights"
            )
        self.weight_scale_granularity = weight_scale_granularity
        self.ignored_layers = ignored_layers or []
        if weight_block_size is not None:
            if not is_checkpoint_fp8_serialized:
                raise ValueError(
                    f"The block-wise quantization only supports fp8-serialized checkpoint for now."
                )
            if len(weight_block_size) != 2:
                raise ValueError(
                    f"The quantization block size of weight must have 2 dimensions, but got {len(weight_block_size)} dimensions."
                )
            if activation_scheme != "dynamic":
                raise ValueError(
                    f"The block-wise quantization only supports dynamic activation scheme for now, but got {activation_scheme} activation scheme."
                )
        self.weight_block_size = weight_block_size

    @classmethod
    def get_name(cls) -> str:
        return "fp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> Fp8Config:
        quant_method = cls.get_from_keys(config, ["quant_method"])
        is_checkpoint_fp8_serialized = "fp8" in quant_method
        activation_scheme = cls.get_from_keys(config, ["activation_scheme"])
        ignored_layers = cls.get_from_keys_or(
            config, ["ignored_layers", "modules_to_not_convert"], None
        )
        if ignored_layers:
            # hacking ministral
            ignored_layers = [layer.replace("model.", "") for layer in ignored_layers]
        weight_block_size = cls.get_from_keys_or(config, ["weight_block_size"], None)
        gemm_backend = cls.get_from_keys_or(config, ["gemm_backend"], "auto")
        weight_scale_granularity = cls.get_from_keys_or(
            config, ["weight_scale_granularity"], None
        )
        return cls(
            is_checkpoint_fp8_serialized=is_checkpoint_fp8_serialized,
            activation_scheme=activation_scheme,
            ignored_layers=ignored_layers,
            weight_block_size=weight_block_size,
            gemm_backend=gemm_backend,
            weight_scale_granularity=weight_scale_granularity,
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        from sglang.multimodal_gen.runtime.layers.linear import LinearBase

        if isinstance(layer, LinearBase):
            if is_layer_skipped(prefix, self.ignored_layers):
                return UnquantizedLinearMethod()
            return Fp8LinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class Fp8LinearMethod(LinearMethodBase):
    """Linear method for FP8.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.

    Also supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.

    Limitations:
    1. Only support per-tensor quantization due to torch._scaled_mm support.
    2. Only support float8_e4m3fn data type due to the limitation of
       torch._scaled_mm (https://github.com/pytorch/pytorch/blob/2e48b39603411a41c5025efbe52f89560b827825/aten/src/ATen/native/cuda/Blas.cpp#L854-L856)

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: Union[Fp8Config, W4AFp8Config]):
        self.quant_config = quant_config
        self.gemm_backend = getattr(quant_config, "gemm_backend", "auto")
        self.cutlass_fp8_supported = cutlass_fp8_supported()

        # For GPUs that lack FP8 hardware support, we can leverage the Marlin
        # kernel for fast weight-only FP8 quantization
        self.use_marlin = False
        self.use_dequant_fallback = False
        if _is_cuda:
            force_marlin = get_bool_env_var("SGLANG_FORCE_FP8_MARLIN")
            auto_enable = can_auto_enable_marlin_fp8()
            self.use_marlin = force_marlin or auto_enable
            if auto_enable and not force_marlin:
                # Some sm80 environments cannot JIT the FP8 Marlin kernel used
                # by diffusion runtime dynamic quantization. Keep FP8 storage and
                # use an unfused dequantized matmul fallback unless Marlin is
                # explicitly forced.
                self.use_marlin = False
                self.use_dequant_fallback = True

        self.block_quant = self.quant_config.weight_block_size is not None

        self.w8a8_block_fp8_linear = dispatch_w8a8_block_fp8_linear()

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        tp_size = get_tensor_model_parallel_world_size()
        if self.block_quant:
            block_n, block_k = (
                self.quant_config.weight_block_size[0],
                self.quant_config.weight_block_size[1],
            )
            # Required by row parallel
            if tp_size > 1 and input_size // input_size_per_partition == tp_size:
                if input_size_per_partition % block_k != 0:
                    raise ValueError(
                        f"Weight input_size_per_partition = "
                        f"{input_size_per_partition} is not divisible by "
                        f"weight quantization block_k = {block_k}."
                    )
            # Required by column parallel or enabling merged weights
            if (
                tp_size > 1 and output_size // output_size_per_partition == tp_size
            ) or len(output_partition_sizes) > 1:
                for output_partition_size in output_partition_sizes:
                    if output_partition_size % block_n != 0:
                        raise ValueError(
                            f"Weight output_partition_size = "
                            f"{output_partition_size} is not divisible by "
                            f"weight quantization block_n = {block_n}."
                        )

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        # WEIGHT
        weight_dtype = (
            torch.float8_e4m3fn
            if self.quant_config.is_checkpoint_fp8_serialized
            else params_dtype
        )

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition, input_size_per_partition, dtype=weight_dtype
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # If checkpoint is serialized fp8, load them.
        # Otherwise, wait until process_weights_after_loading.
        if self.quant_config.is_checkpoint_fp8_serialized:
            # WEIGHT SCALE
            if self.block_quant:
                if hasattr(self.quant_config, "activation_scheme"):
                    assert self.quant_config.activation_scheme == "dynamic"
                elif hasattr(self.quant_config, "linear_activation_scheme"):
                    assert self.quant_config.linear_activation_scheme == "dynamic"
                scale = BlockQuantScaleParameter(
                    data=torch.empty(
                        (output_size_per_partition + block_n - 1) // block_n,
                        (input_size_per_partition + block_k - 1) // block_k,
                        dtype=torch.float32,
                    ),
                    input_dim=1,
                    output_dim=0,
                    weight_loader=weight_loader,
                )
                scale.format_ue8m0 = False
                scale[:] = torch.finfo(torch.float32).min
                layer.register_parameter("weight_scale_inv", scale)
            elif self.quant_config.weight_scale_granularity == "channel":
                scale = ChannelQuantScaleParameter(
                    data=torch.empty(
                        output_size_per_partition,
                        dtype=torch.float32,
                    ),
                    output_dim=0,
                    weight_loader=weight_loader,
                )
                scale[:] = torch.finfo(torch.float32).min
                layer.register_parameter("weight_scale", scale)
            else:
                scale = PerTensorScaleParameter(
                    data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                    weight_loader=weight_loader,
                )
                scale[:] = torch.finfo(torch.float32).min
                layer.register_parameter("weight_scale", scale)

            # INPUT ACTIVATION SCALE
            if (
                hasattr(self.quant_config, "activation_scheme")
                and self.quant_config.activation_scheme == "static"
            ) or (
                hasattr(self.quant_config, "linear_activation_scheme")
                and self.quant_config.linear_activation_scheme == "static"
            ):
                scale = PerTensorScaleParameter(
                    data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                    weight_loader=weight_loader,
                )

                scale[:] = torch.finfo(torch.float32).min
                layer.register_parameter("input_scale", scale)
            else:
                layer.register_parameter("input_scale", None)

    def process_weights_after_loading(self, layer: Module) -> None:
        if self.block_quant:
            # If ROCm, normalize the weights and scales to e4m3fnuz
            if _is_fp8_fnuz:
                # activation_scheme: dynamic
                weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                    weight=layer.weight,
                    weight_scale=layer.weight_scale_inv,
                    input_scale=None,
                )
                layer.input_scale = None
            elif _is_cpu:
                assert (
                    _is_cpu_amx_available
                ), "Fp8LinearMethod on CPU requires that CPU has AMX support"
                _amx_process_weight_after_loading(layer, ["weight"])
                layer.weight_scale_inv = torch.nn.Parameter(
                    layer.weight_scale_inv.data, requires_grad=False
                )
                return
            else:
                # For fp8 linear weights run with deepgemm, the weights and scales need be requantized to ue8m0
                from sglang.srt.layers.quantization.fp8_utils import (
                    deepgemm_w8a8_block_fp8_linear_with_fallback,
                )
                from sglang.srt.model_loader.utils import (
                    should_deepgemm_weight_requant_ue8m0,
                )

                if (
                    should_deepgemm_weight_requant_ue8m0(
                        weight_block_size=getattr(
                            self.quant_config, "weight_block_size", None
                        ),
                    )
                    and (
                        self.w8a8_block_fp8_linear
                        is deepgemm_w8a8_block_fp8_linear_with_fallback
                    )
                    and (not layer.weight_scale_inv.format_ue8m0)
                ):
                    requant_weight_ue8m0_inplace(
                        layer.weight,
                        layer.weight_scale_inv,
                        self.quant_config.weight_block_size,
                    )
                    layer.weight_scale_inv.format_ue8m0 = True
                weight, weight_scale = layer.weight.data, layer.weight_scale_inv.data

            layer.weight.data = weight.data
            layer.weight_scale_inv.data = weight_scale.data
        else:
            layer.weight = Parameter(layer.weight.data, requires_grad=False)

            # If checkpoint not serialized fp8, quantize the weights.
            if not self.quant_config.is_checkpoint_fp8_serialized:
                if get_bool_env_var(FAKE_W8A8_DEQUANT_ENV):
                    qweight, weight_scale = _fake_quant_fp8_weight_per_channel(
                        layer.weight
                    )
                elif self.cutlass_fp8_supported:
                    # apply per-channel quantization default as
                    # cutlass sgl-kernel supports per-channel scale
                    qweight, weight_scale = per_token_group_quant_fp8(
                        layer.weight, layer.weight.shape[-1]
                    )
                    weight_scale = weight_scale.t().contiguous()
                else:
                    # per-tensor quantization
                    # On sm80 Marlin fallback, Triton per-channel FP8 quantization
                    # can compile to an unsupported fp8e4nv dtype. Tensor-wise
                    # conversion keeps runtime FP8 dynamic loading usable there.
                    qweight, weight_scale = input_to_float8(layer.weight)

                # Update the layer with the new values.
                layer.weight = Parameter(qweight.t(), requires_grad=False)
                layer.weight_scale = Parameter(weight_scale, requires_grad=False)
                layer.input_scale = None

            # If checkpoint is fp8, handle that there are N scales for N
            # shards in a fused module
            else:
                layer.weight_scale = Parameter(
                    layer.weight_scale.data, requires_grad=False
                )
                if (
                    hasattr(self.quant_config, "activation_scheme")
                    and self.quant_config.activation_scheme == "static"
                ) or (
                    hasattr(self.quant_config, "linear_activation_scheme")
                    and self.quant_config.linear_activation_scheme == "static"
                ):
                    layer.input_scale = Parameter(
                        layer.input_scale.data, requires_grad=False
                    )

                if self.quant_config.weight_scale_granularity == "channel":
                    weight = layer.weight
                    weight_scale = layer.weight_scale
                    expected_scales = weight.shape[0]
                    if weight_scale.numel() != expected_scales:
                        raise ValueError(
                            "Serialized per-channel FP8 scale count does not match "
                            f"the output dimension: {weight_scale.numel()} != "
                            f"{expected_scales}"
                        )
                    if not torch.isfinite(weight_scale).all() or not torch.all(
                        weight_scale > 0
                    ):
                        raise ValueError(
                            "Serialized per-channel FP8 scales must be finite and "
                            "strictly positive"
                        )
                    weight_scale = weight_scale.reshape(1, -1).contiguous()
                # cutlass sgl-kernel and marlin only support per-channel scale
                elif self.cutlass_fp8_supported or self.use_marlin:
                    weight = layer.weight
                    weight_scale = convert_to_channelwise(
                        layer.weight_scale, layer.logical_widths
                    )
                else:
                    # Dequant -> Quant with max scale so we can run per tensor.
                    weight = layer.weight
                    weight_scale = layer.weight_scale
                    # If ROCm, normalize the weights and scales to e4m3fnuz
                    if _is_fp8_fnuz:
                        weight, weight_scale, input_scale = (
                            normalize_e4m3fn_to_e4m3fnuz(
                                weight=weight,
                                weight_scale=weight_scale,
                                input_scale=layer.input_scale,
                            )
                        )
                        if input_scale is not None:
                            layer.input_scale = Parameter(
                                input_scale, requires_grad=False
                            )

                    weight_scale, weight = requantize_with_max_scale(
                        weight=weight,
                        weight_scale=weight_scale,
                        logical_widths=layer.logical_widths,
                    )

                # Update layer with new values.
                layer.weight = Parameter(weight.t(), requires_grad=False)
                layer.weight_scale = Parameter(weight_scale, requires_grad=False)
                if (
                    hasattr(self.quant_config, "activation_scheme")
                    and self.quant_config.activation_scheme == "static"
                ) or (
                    hasattr(self.quant_config, "linear_activation_scheme")
                    and self.quant_config.linear_activation_scheme == "static"
                ):
                    layer.input_scale = Parameter(
                        layer.input_scale.max(), requires_grad=False
                    )

        if self.use_marlin:
            if self.block_quant:
                layer.weight_block_size = self.quant_config.weight_block_size
            prepare_fp8_layer_for_marlin(layer, not self.block_quant)
            # Activations not quantized for marlin.
            del layer.input_scale

    def runtime_kernel_name(
        self, layer: torch.nn.Module, x: Optional[torch.Tensor] = None
    ) -> str:
        if self.use_dequant_fallback and not self.block_quant:
            return "bf16_matmul_dequant_fallback"
        if self.use_marlin:
            return "fp8_marlin_weight_only"
        if self.block_quant:
            backend = getattr(
                self.w8a8_block_fp8_linear,
                "__name__",
                type(self.w8a8_block_fp8_linear).__name__,
            )
            return f"dynamic_block_fp8+{backend}"

        activation = (
            "dynamic_per_token_fp8"
            if getattr(layer, "input_scale", None) is None
            else "static_fp8"
        )
        weight = layer.weight
        input_num_tokens = None
        if isinstance(x, torch.Tensor):
            input_num_tokens = x.numel() // x.shape[-1]
        backend = resolve_fp8_linear_gemm_backend(
            requested_backend=self.gemm_backend,
            cutlass_fp8_supported=self.cutlass_fp8_supported,
            weight_shape=tuple(weight.shape),
            weight_scale_numel=layer.weight_scale.numel(),
            input_num_tokens=input_num_tokens,
        )
        gemm = {
            "sgl_cutlass": "sgl_kernel.fp8_scaled_mm",
            "triton": "triton_scaled_mm",
            "non_cutlass": "apply_fp8_linear.non_cutlass",
            "hybrid_shape_dependent": (
                "hybrid(sgl_kernel.fp8_scaled_mm|triton_scaled_mm)"
            ),
        }[backend]
        return f"{activation}+{gemm}"

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        maybe_log_linear_runtime_shape(
            layer,
            x,
            method=type(self).__name__,
            kernel=self.runtime_kernel_name(layer, x),
        )
        if self.use_dequant_fallback and not self.block_quant:
            if get_bool_env_var(FAKE_W8A8_DEQUANT_ENV):
                x = _fake_quant_dequant_fp8_per_token(x)
            weight = layer.weight.float() * layer.weight_scale.float()
            output = torch.matmul(x, weight.to(x.dtype))
            if bias is not None:
                output = output + bias
            return output

        if self.use_marlin:
            return apply_fp8_marlin_linear(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                workspace=layer.workspace,
                size_n=layer.output_size_per_partition,
                size_k=layer.input_size_per_partition,
                bias=bias,
            )

        if self.block_quant:
            if use_intel_amx_backend(layer):
                return torch.ops.sgl_kernel.fp8_scaled_mm_cpu(
                    x,
                    layer.weight,
                    layer.weight_scale_inv,
                    self.quant_config.weight_block_size,
                    bias,
                    x.dtype,
                    True,  # is_vnni
                )

            if isinstance(x, tuple):
                return self.w8a8_block_fp8_linear(
                    input=x[0],
                    weight=layer.weight,
                    block_size=self.quant_config.weight_block_size,
                    weight_scale=layer.weight_scale_inv,
                    input_scale=x[1],
                    bias=bias,
                )

            return self.w8a8_block_fp8_linear(
                input=x,
                weight=layer.weight,
                block_size=self.quant_config.weight_block_size,
                weight_scale=layer.weight_scale_inv,
                input_scale=None,
                bias=bias,
            )

        return apply_fp8_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            input_scale=layer.input_scale,
            bias=bias,
            cutlass_fp8_supported=self.cutlass_fp8_supported,
            use_per_token_if_dynamic=False,
            gemm_backend=self.gemm_backend,
        )
