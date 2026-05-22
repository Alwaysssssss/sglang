# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor

from sglang.multimodal_gen.runtime.models.schedulers.base import BaseScheduler


def _make_linear_beta_schedule(
    num_train_timesteps: int,
    linear_start: float,
    linear_end: float,
) -> np.ndarray:
    roots = torch.linspace(
        linear_start**0.5,
        linear_end**0.5,
        num_train_timesteps,
        dtype=torch.float64,
    )
    return (roots**2).cpu().numpy()


def _generate_roughly_equally_spaced_steps(
    num_substeps: int,
    max_step: int,
) -> np.ndarray:
    return np.linspace(max_step - 1, 0, num_substeps, endpoint=False).astype(int)[::-1]


def _build_zero_snr_alphas_cumprod_sqrt(
    *,
    num_inference_steps: int,
    num_train_timesteps: int,
    linear_start: float,
    linear_end: float,
    shift_scale: float,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    betas = _make_linear_beta_schedule(
        num_train_timesteps=num_train_timesteps,
        linear_start=linear_start,
        linear_end=linear_end,
    )
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_cumprod = alphas_cumprod / (
        shift_scale + (1.0 - shift_scale) * alphas_cumprod
    )

    if num_inference_steps < num_train_timesteps:
        timesteps = _generate_roughly_equally_spaced_steps(
            num_substeps=num_inference_steps,
            max_step=num_train_timesteps,
        )
        alphas_cumprod = alphas_cumprod[timesteps]
    elif num_inference_steps == num_train_timesteps:
        timesteps = np.arange(num_train_timesteps, dtype=np.int64)
    else:
        raise ValueError(
            "num_inference_steps must be <= num_train_timesteps for "
            "StarVPSDEDPMPP2MScheduler"
        )

    alphas_cumprod = torch.tensor(
        alphas_cumprod,
        dtype=torch.float32,
        device=device,
    )
    alphas_cumprod_sqrt = alphas_cumprod.sqrt()
    alpha_0 = alphas_cumprod_sqrt[0].clone()
    alpha_t = alphas_cumprod_sqrt[-1].clone()
    alphas_cumprod_sqrt = alphas_cumprod_sqrt - alpha_t
    alphas_cumprod_sqrt = alphas_cumprod_sqrt * (
        alpha_0 / (alpha_0 - alpha_t)
    )
    # Match STAR sampler order: highest-noise step first.
    alphas_cumprod_sqrt = torch.flip(alphas_cumprod_sqrt, dims=(0,))
    timesteps = torch.tensor(timesteps[::-1].copy(), dtype=torch.int64, device=device)
    return alphas_cumprod_sqrt, timesteps


@dataclass
class StarVPSDEDPMPP2MSchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor


class StarVPSDEDPMPP2MScheduler(SchedulerMixin, ConfigMixin, BaseScheduler):
    """Thin STAR-native scheduler adapter for VPSDE DPM++ 2M sampling."""

    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        num_steps: int = 50,
        linear_start: float = 0.00085,
        linear_end: float = 0.012,
        shift_scale: float = 1.0,
        scale: float = 6.0,
        exp: float = 5.0,
        verbose: bool = False,
        **kwargs,
    ):
        del kwargs
        self.num_train_timesteps = num_train_timesteps
        self.default_num_inference_steps = num_steps
        self.linear_start = linear_start
        self.linear_end = linear_end
        self.shift_scale = shift_scale
        self.cfg_scale = scale
        self.cfg_exp = exp
        self.verbose = verbose

        self.timesteps = torch.empty(0, dtype=torch.int64)
        self.alphas_cumprod_sqrt = torch.empty(0, dtype=torch.float32)
        self.sigmas = torch.empty(0, dtype=torch.float32)
        self.init_noise_sigma = 1.0

        self._num_inference_steps: int | None = None
        self._step_index: int | None = None
        self._old_denoised: torch.Tensor | None = None

        BaseScheduler.__init__(self)

    def set_shift(self, shift: float) -> None:
        self.shift_scale = float(shift)
        if self._num_inference_steps is not None:
            self.set_timesteps(self._num_inference_steps, device=self.timesteps.device)

    def _reset_step_state(self) -> None:
        self._step_index = None
        self._old_denoised = None

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        device: torch.device | str | None = None,
        timesteps: list[int] | torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        del kwargs
        device = device or "cpu"
        self._reset_step_state()

        if timesteps is not None:
            if isinstance(timesteps, torch.Tensor):
                self.timesteps = timesteps.to(device=device, dtype=torch.int64)
            else:
                self.timesteps = torch.tensor(
                    list(timesteps),
                    dtype=torch.int64,
                    device=device,
                )
            if self.timesteps.ndim != 1:
                raise ValueError("timesteps must be a 1D tensor or list")
            self._num_inference_steps = int(self.timesteps.numel())
            full_alphas, _ = _build_zero_snr_alphas_cumprod_sqrt(
                num_inference_steps=self.num_train_timesteps,
                num_train_timesteps=self.num_train_timesteps,
                linear_start=self.linear_start,
                linear_end=self.linear_end,
                shift_scale=self.shift_scale,
                device=device,
            )
            alpha_by_timestep = torch.flip(full_alphas, dims=(0,))
            selected = alpha_by_timestep.index_select(0, self.timesteps)
            self.alphas_cumprod_sqrt = selected
        else:
            if num_inference_steps is None:
                num_inference_steps = self.default_num_inference_steps
            self._num_inference_steps = int(num_inference_steps)
            self.alphas_cumprod_sqrt, self.timesteps = _build_zero_snr_alphas_cumprod_sqrt(
                num_inference_steps=self._num_inference_steps,
                num_train_timesteps=self.num_train_timesteps,
                linear_start=self.linear_start,
                linear_end=self.linear_end,
                shift_scale=self.shift_scale,
                device=device,
            )

        self.alphas_cumprod_sqrt = torch.cat(
            [
                self.alphas_cumprod_sqrt,
                torch.ones(1, dtype=self.alphas_cumprod_sqrt.dtype, device=device),
            ],
            dim=0,
        )
        self.sigmas = torch.sqrt(
            torch.clamp(1.0 - self.alphas_cumprod_sqrt.square(), min=0.0)
        )

    def scale_model_input(
        self, sample: torch.Tensor, timestep: int | None = None
    ) -> torch.Tensor:
        del timestep
        return sample

    def _init_step_index(self, timestep: torch.Tensor | int) -> None:
        if isinstance(timestep, torch.Tensor):
            timestep_value = int(timestep.flatten()[0].item())
        else:
            timestep_value = int(timestep)
        matches = (self.timesteps == timestep_value).nonzero(as_tuple=False)
        if matches.numel() == 0:
            raise ValueError(f"Unknown STAR scheduler timestep: {timestep_value}")
        self._step_index = int(matches[0].item())

    def _epsilon_to_denoised(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor,
        current_alpha_cumprod_sqrt: torch.Tensor,
    ) -> torch.Tensor:
        sample_f = sample.to(torch.float32)
        model_output_f = model_output.to(torch.float32)
        current_alpha = current_alpha_cumprod_sqrt.to(
            device=sample.device,
            dtype=torch.float32,
        )
        current_sigma = torch.sqrt(
            torch.clamp(1.0 - current_alpha.square(), min=0.0)
        )
        return current_alpha * sample_f - current_sigma * model_output_f

    @staticmethod
    def _get_variables(
        alpha_cumprod_sqrt: torch.Tensor,
        next_alpha_cumprod_sqrt: torch.Tensor,
        previous_alpha_cumprod_sqrt: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        alpha_cumprod = alpha_cumprod_sqrt.square()
        next_alpha_cumprod = next_alpha_cumprod_sqrt.square()
        lamb = torch.log(torch.sqrt(alpha_cumprod / (1.0 - alpha_cumprod)))
        lamb_next = torch.log(
            torch.sqrt(next_alpha_cumprod / (1.0 - next_alpha_cumprod))
        )
        h = lamb_next - lamb

        if previous_alpha_cumprod_sqrt is None:
            return h, None

        previous_alpha_cumprod = previous_alpha_cumprod_sqrt.square()
        lamb_previous = torch.log(
            torch.sqrt(previous_alpha_cumprod / (1.0 - previous_alpha_cumprod))
        )
        h_last = lamb - lamb_previous
        r = h_last / h
        return h, r

    @staticmethod
    def _append_dims(value: torch.Tensor, ndim: int) -> torch.Tensor:
        while value.ndim < ndim:
            value = value.unsqueeze(-1)
        return value

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int | torch.Tensor,
        sample: torch.Tensor,
        generator: torch.Generator | list[torch.Generator] | None = None,
        return_dict: bool = True,
        **kwargs,
    ) -> StarVPSDEDPMPP2MSchedulerOutput | tuple[torch.FloatTensor, ...]:
        del kwargs
        if self._step_index is None:
            self._init_step_index(timestep)
        assert self._step_index is not None

        step_index = self._step_index
        current_alpha = self.alphas_cumprod_sqrt[step_index].to(sample.device)
        next_alpha = self.alphas_cumprod_sqrt[step_index + 1].to(sample.device)
        previous_alpha = (
            None
            if step_index == 0
            else self.alphas_cumprod_sqrt[step_index - 1].to(sample.device)
        )

        denoised = self._epsilon_to_denoised(
            model_output=model_output,
            sample=sample,
            current_alpha_cumprod_sqrt=current_alpha,
        )

        if step_index == self.timesteps.numel() - 1:
            prev_sample = denoised
        else:
            h, r = self._get_variables(current_alpha, next_alpha, previous_alpha)
            mult1 = (
                torch.sqrt((1.0 - next_alpha.square()) / (1.0 - current_alpha.square()))
                * torch.exp(-h)
            )
            mult2 = torch.expm1(-2.0 * h) * next_alpha
            mult_noise = torch.sqrt(1.0 - next_alpha.square()) * torch.sqrt(
                torch.clamp(1.0 - torch.exp(-2.0 * h), min=0.0)
            )

            sample_f = sample.to(torch.float32)
            noise = randn_tensor(
                sample.shape,
                generator=generator,
                device=sample.device,
                dtype=torch.float32,
            )
            mult1 = self._append_dims(mult1, sample.ndim)
            mult2 = self._append_dims(mult2, sample.ndim)
            mult_noise = self._append_dims(mult_noise, sample.ndim)
            x_standard = mult1 * sample_f - mult2 * denoised + mult_noise * noise

            if self._old_denoised is None or r is None or torch.sum(next_alpha) < 1e-14:
                prev_sample = x_standard
            else:
                mult3 = self._append_dims(1.0 + 1.0 / (2.0 * r), sample.ndim)
                mult4 = self._append_dims(1.0 / (2.0 * r), sample.ndim)
                denoised_d = mult3 * denoised - mult4 * self._old_denoised
                prev_sample = mult1 * sample_f - mult2 * denoised_d + mult_noise * noise

        self._old_denoised = denoised
        self._step_index += 1
        prev_sample = prev_sample.to(model_output.dtype)

        if not return_dict:
            return (prev_sample,)
        return StarVPSDEDPMPP2MSchedulerOutput(prev_sample=prev_sample)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        if timesteps.ndim == 0:
            timesteps = timesteps.unsqueeze(0)
        alpha_lookup = torch.flip(self.alphas_cumprod_sqrt[:-1], dims=(0,))
        alpha = alpha_lookup.index_select(0, timesteps.to(torch.long))
        alpha = self._append_dims(alpha.to(original_samples.device), original_samples.ndim)
        sigma = torch.sqrt(torch.clamp(1.0 - alpha.square(), min=0.0))
        return alpha * original_samples + sigma * noise


EntryClass = StarVPSDEDPMPP2MScheduler
