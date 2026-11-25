"""Prospective SD 1.x DDIM path from an explicit native latent and CLIP pair.

This detached experiment uses the archived project DDIM call order. It owns no
base loading, fitted adapter, gate calibration, or image observer.
"""

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite, sqrt

import torch
from diffusers import DDIMScheduler
from particle_sliders.adaptive_steering import ActivationGate, activation_projection


@dataclass(frozen=True)
class Decision:
    scale: float
    score: float | None = None
    probability: float | None = None
    fired: bool | None = None


class ActivationStrength:
    """A fresh shared gate and an explicitly calibrated native signal per request."""

    def __init__(self, *, method, parameter, direction, signal, seed=None, clabs=None):
        if method not in {"cfp", "clabs"} or not callable(signal):
            raise ValueError("Expected a calibrated CFP or CLABS native activation readout")
        probe = torch.as_tensor(direction, dtype=torch.float32).detach().clone()
        if (probe.ndim != 1 or not torch.isfinite(probe).all()
                or not torch.isclose(probe.norm(), probe.new_tensor(1.0))):
            raise ValueError("Expected a finite unit direction in the calibrated activation space")
        self.method = method
        self.parameter = parameter
        self.direction = probe
        self.signal = signal
        self.seed = seed
        self.clabs = clabs
        self.new_gate()

    def new_gate(self):
        return ActivationGate(self.method, self.parameter, seed=self.seed, clabs=self.clabs)

    def begin(self, nominal):
        gate = self.new_gate()

        def resolve(index, timestep, model_input):
            residual = self.signal(index, timestep, model_input)
            if residual.ndim != 1 or residual.shape != self.direction.shape:
                raise ValueError("Native activation and the fitted direction have different shapes")
            score = float(activation_projection(residual, self.direction.to(residual.device), strength=nominal))
            choice = gate.step(score)
            return Decision(nominal if choice.fired else 0.0, score, choice.probability, choice.fired)

        return resolve


def normalized_sigma_coordinates(scheduler):
    """Derive the eta-zero epsilon-DDIM clock in x/sqrt(alpha_bar)."""
    if type(scheduler) is not DDIMScheduler or scheduler.num_inference_steps is None:
        raise ValueError("Expected an initialized native DDIM scheduler")
    if scheduler.config.prediction_type != "epsilon" or scheduler.config.clip_sample or scheduler.config.thresholding:
        raise ValueError("Normalized dose clock requires epsilon DDIM without clipping or thresholding")
    timesteps = tuple(int(value) for value in scheduler.timesteps)
    stride = scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    if len(timesteps) != scheduler.num_inference_steps or any(
            previous - stride != current for previous, current in zip(timesteps, timesteps[1:])):
        raise ValueError("Every native DDIM predecessor must equal the next registered timestep")
    previous = timesteps[-1] - stride
    if previous >= 0:
        terminal_alpha = scheduler.alphas_cumprod[previous]
    else:
        terminal_alpha = scheduler.final_alpha_cumprod
    alphas = tuple(float(scheduler.alphas_cumprod[index]) for index in timesteps) + (float(terminal_alpha),)
    coordinates = tuple(sqrt((1.0 - value) / value) for value in alphas)
    if not all(isfinite(value) for value in coordinates) or any(
            current <= following for current, following in zip(coordinates, coordinates[1:])):
        raise ValueError("Native DDIM clock must be finite and strictly decreasing")
    return timesteps, coordinates


@torch.no_grad()
def render_ddim(unet, vae, scheduler, *, initial_noise, negative, positive, steps, guidance,
                strength=0.0, adapter=None, activation=None):
    """Evaluate one paired CFG UNet per interval, followed by native VAE decode.

    ``adapter.scaled(float)`` owns the attached RAW branches for both CFG rows.
    Trace values are requested scales. Qualification must separately verify the
    scale actually applied by the exact saved adapter to its native branches.
    """
    if type(scheduler) is not DDIMScheduler:
        raise ValueError("This reviewed denoising clock is DDIM")
    if type(steps) is not int or steps < 1 or not isfinite(guidance) or guidance < 0:
        raise ValueError("Expected explicit positive steps and finite nonnegative CFG")
    if not isfinite(strength) or not 0.0 <= strength <= 1.0 or (strength and adapter is None):
        raise ValueError("Expected unipolar strength and an explicit RAW adapter for nonzero scale")
    if activation is not None and not isinstance(activation, ActivationStrength):
        raise TypeError("Activation plans must use the shared per-request gate bridge")
    weight = next(unet.parameters())
    if (initial_noise.ndim != 4 or initial_noise.shape[0] != 1
            or initial_noise.shape[1] != unet.config.in_channels
            or positive.ndim != 3 or positive.shape[0] != 1 or positive.shape != negative.shape
            or positive.shape[-1] != unet.config.cross_attention_dim):
        raise ValueError("Expected one native latent and one exact matched negative/positive CLIP pair")
    scheduler = deepcopy(scheduler)
    scheduler.set_timesteps(steps, device=weight.device)
    timesteps, coordinates = normalized_sigma_coordinates(scheduler)
    latents = initial_noise.to(weight) * scheduler.init_noise_sigma
    embeds = torch.cat((negative, positive)).to(weight)
    resolve = activation.begin(strength) if strength and activation is not None else None
    trace = []
    for index, timestep in enumerate(scheduler.timesteps):
        # This is the paired native input before either UNet branch receives an edit.
        model_input = scheduler.scale_model_input(torch.cat((latents, latents)), timestep)
        decision = resolve(index, timestep, model_input.detach().clone()) if resolve else Decision(float(strength))
        scope = adapter.scaled(decision.scale) if adapter is not None else nullcontext()
        with scope:
            epsilon = unet(model_input, timestep, encoder_hidden_states=embeds).sample
        unconditional, conditional = epsilon.chunk(2)
        prediction = unconditional + guidance * (conditional - unconditional)
        latents = scheduler.step(prediction, timestep, latents, eta=0).prev_sample
        trace.append(decision)
    decoded = vae.decode(latents.to(next(vae.parameters())) / vae.config.scaling_factor).sample
    if not torch.isfinite(decoded).all():
        raise ValueError("Nonfinite SD VAE decode")
    pixels = ((decoded / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).float() * 255).round().to(torch.uint8)
    return pixels.cpu(), latents, dict(native_timesteps=timesteps, normalized_sigma_coordinates=coordinates,
                                      decisions=tuple(trace))
