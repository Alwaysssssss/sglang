# SPDX-License-Identifier: Apache-2.0
"""Isolated round-four experiments; apply only to a frozen VSR instance."""

from types import MethodType

import torch
from diffusers.models.autoencoders.autoencoder_kl_wan import DecoderOutput, unpatchify
from sglang.multimodal_gen.runtime.vsr.compile import _implicit_spatial_padding


def decode_cat_once(self, z, return_dict=True):
    if self.use_tiling:
        raise ValueError("Experiment requires VSR outer tiling")
    self.clear_cache()
    x = self.post_quant_conv(z)
    outputs = []
    for i in range(z.shape[2]):
        self._conv_idx = [0]
        outputs.append(
            self.decoder(
                x[:, :, i : i + 1],
                feat_cache=self._feat_map,
                feat_idx=self._conv_idx,
                first_chunk=i == 0,
            )
        )
    out = torch.cat(outputs, 2)
    if self.config.patch_size is not None:
        out = unpatchify(out, patch_size=self.config.patch_size)
    out = out.clamp(-1, 1)
    self.clear_cache()
    return DecoderOutput(sample=out) if return_dict else (out,)


def memoize_fixed(module):
    original = module.forward
    cache = {}

    def forward(*args, **kwargs):
        # VSR fixes timestep/text and weights; shapes separate batch variants.
        key = tuple(
            (tuple(x.shape), x.dtype, x.device)
            for x in args
            if isinstance(x, torch.Tensor)
        )
        if key not in cache:
            cache[key] = original(*args, **kwargs)
        return cache[key]

    module.forward = forward


def fixed_condition(model, collapse=False, exact=False):
    memoize_fixed(model.dit.condition_embedder)
    for block in model.dit.blocks:
        attention = block.attn2
        if collapse:
            original = attention.forward
            cache = {}

            def cross(
                hidden_states,
                encoder_hidden_states=None,
                attention_mask=None,
                rotary_emb=None,
                _original=original,
                _cache=cache,
                **kwargs,
            ):
                assert encoder_hidden_states.shape[1] == 1 and attention_mask is None
                key = (
                    tuple(hidden_states.shape) if exact else hidden_states.shape[0],
                    hidden_states.dtype,
                    hidden_states.device,
                )
                if key not in _cache:
                    _cache[key] = _original(
                        hidden_states if exact else hidden_states[:, :1],
                        encoder_hidden_states,
                        attention_mask,
                        rotary_emb,
                        **kwargs,
                    )
                return (
                    _cache[key]
                    if exact
                    else _cache[key].expand(-1, hidden_states.shape[1], -1)
                )

            attention.forward = cross
        else:
            memoize_fixed(attention.to_k)
            memoize_fixed(attention.to_v)


def apply(model, mode):
    vae = model.vae.vae
    if mode in {"cat_once", "outer_compile"}:
        vae._decode = MethodType(decode_cat_once, vae)
    if mode in {"implicit_pad", "combined"}:
        from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d

        for module in vae.decoder.modules():
            if isinstance(module, WanCausalConv3d):
                module.forward = MethodType(_implicit_spatial_padding, module)
    if mode == "outer_compile":
        vae.decoder.forward = vae.decoder.forward._torchdynamo_orig_callable
        vae._decode = torch.compile(vae._decode, fullgraph=True, dynamic=False)
    if mode in {"condition", "cross_constant", "combined"}:
        fixed_condition(model, collapse=mode in {"cross_constant", "combined"})
    if mode == "graph":
        return "graph"


def capture(model, window):
    static_input = window.to(model.device, dtype=model.dtype)
    with torch.no_grad():
        for _ in range(3):
            model.restore_window(static_input)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output = model.restore_window(static_input)
    shape = tuple(static_input.shape)

    def replay(window):
        assert tuple(window.shape) == shape
        static_input.copy_(window)
        graph.replay()
        return static_output.clone()

    model._restore_window = replay
