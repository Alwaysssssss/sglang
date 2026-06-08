# SPDX-License-Identifier: Apache-2.0
from typing import Literal

import numpy as np
import torch
from diffusers.schedulers.scheduling_dpm_cogvideox import (
    CogVideoXDPMScheduler as DiffusersCogVideoXDPMScheduler,
)
from diffusers.schedulers.scheduling_dpm_cogvideox import DDIMSchedulerOutput
from diffusers.schedulers.scheduling_dpm_cogvideox import betas_for_alpha_bar
from diffusers.schedulers.scheduling_dpm_cogvideox import rescale_zero_terminal_snr
from sglang.multimodal_gen.runtime.utils.common import (
    randn_tensor_with_generator_device,
)


class CogVideoXDPMScheduler(DiffusersCogVideoXDPMScheduler):
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.0120,
        beta_schedule: Literal["linear", "scaled_linear", "squaredcos_cap_v2"] = "scaled_linear",
        trained_betas: np.ndarray | list[float] | None = None,
        clip_sample: bool = True,
        set_alpha_to_one: bool = True,
        steps_offset: int = 0,
        prediction_type: Literal["epsilon", "sample", "v_prediction"] = "epsilon",
        clip_sample_range: float = 1.0,
        sample_max_value: float = 1.0,
        timestep_spacing: Literal["leading", "linspace", "trailing"] = "leading",
        rescale_betas_zero_snr: bool = False,
        snr_shift_scale: float = 3.0,
        **_: object,
    ) -> None:
        super().__init__(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            trained_betas=trained_betas,
            clip_sample=clip_sample,
            set_alpha_to_one=set_alpha_to_one,
            steps_offset=steps_offset,
            prediction_type=prediction_type,
            clip_sample_range=clip_sample_range,
            sample_max_value=sample_max_value,
            timestep_spacing=timestep_spacing,
            rescale_betas_zero_snr=rescale_betas_zero_snr,
            snr_shift_scale=snr_shift_scale,
        )
        self.num_train_timesteps = self.config.num_train_timesteps
        self.snr_shift_scale = self.config.snr_shift_scale

    def _recompute_alphas_cumprod(self, shift: float | None = None) -> None:
        shift = self.config.snr_shift_scale if shift is None else shift
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod = self.alphas_cumprod / (
            shift + (1 - shift) * self.alphas_cumprod
        )
        if self.config.rescale_betas_zero_snr:
            self.alphas_cumprod = rescale_zero_terminal_snr(self.alphas_cumprod)
        self.final_alpha_cumprod = (
            torch.tensor(1.0)
            if self.config.set_alpha_to_one
            else self.alphas_cumprod[0]
        )

    def set_shift(self, shift: float) -> None:
        self.snr_shift_scale = shift
        self._recompute_alphas_cumprod(shift)

    def step(
        self,
        model_output: torch.Tensor,
        old_pred_original_sample: torch.Tensor | None,
        timestep: int,
        timestep_back: int | None,
        sample: torch.Tensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator: torch.Generator | None = None,
        variance_noise: torch.Tensor | None = None,
        return_dict: bool = False,
        restoration_guidance_scale: float = -1.0,
        restoration_ori_latent: torch.Tensor | None = None,
    ) -> DDIMSchedulerOutput | tuple:
        del eta, use_clipped_model_output, variance_noise

        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps

        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.final_alpha_cumprod
        )
        alpha_prod_t_back = (
            self.alphas_cumprod[timestep_back] if timestep_back is not None else None
        )

        beta_prod_t = 1 - alpha_prod_t

        if self.config.prediction_type == "epsilon":
            pred_original_sample = (
                sample - beta_prod_t**0.5 * model_output
            ) / alpha_prod_t**0.5
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = alpha_prod_t**0.5 * sample - beta_prod_t**0.5 * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or `v_prediction`"
            )

        if restoration_guidance_scale > 0 and restoration_ori_latent is not None:
            timestep_value = int(timestep.item()) if isinstance(timestep, torch.Tensor) else int(timestep)
            restoration_direction = restoration_ori_latent - pred_original_sample
            restoration_strength = (
                float(timestep_value) / len(self.alphas)
            ) ** restoration_guidance_scale
            pred_original_sample = (
                pred_original_sample + restoration_strength * restoration_direction
            )

        h, r, lamb, lamb_next = self.get_variables(
            alpha_prod_t, alpha_prod_t_prev, alpha_prod_t_back
        )
        del lamb, lamb_next
        mult = list(self.get_mult(h, r, alpha_prod_t, alpha_prod_t_prev, alpha_prod_t_back))
        mult_noise = (1 - alpha_prod_t_prev) ** 0.5 * (1 - (-2 * h).exp()) ** 0.5

        noise = randn_tensor_with_generator_device(
            sample.shape,
            generator=generator,
            device=sample.device,
            dtype=sample.dtype,
        )
        prev_sample = mult[0] * sample - mult[1] * pred_original_sample + mult_noise * noise

        if old_pred_original_sample is None or prev_timestep < 0:
            if not return_dict:
                return prev_sample, pred_original_sample
            return DDIMSchedulerOutput(
                prev_sample=prev_sample,
                pred_original_sample=pred_original_sample,
            )

        denoised_d = mult[2] * pred_original_sample - mult[3] * old_pred_original_sample
        noise = randn_tensor_with_generator_device(
            sample.shape,
            generator=generator,
            device=sample.device,
            dtype=sample.dtype,
        )
        prev_sample = mult[0] * sample - mult[1] * denoised_d + mult_noise * noise

        if not return_dict:
            return prev_sample, pred_original_sample

        return DDIMSchedulerOutput(
            prev_sample=prev_sample,
            pred_original_sample=pred_original_sample,
        )


class CogVideoXDDIMScheduler(CogVideoXDPMScheduler):
    pass


EntryClass = [CogVideoXDPMScheduler, CogVideoXDDIMScheduler]
