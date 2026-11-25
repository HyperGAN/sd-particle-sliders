"""Experimental stock SD1.5 DDIM replay of independently qualified A-PSR probes."""

from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass, replace
from hashlib import sha256
from inspect import getsourcefile
import json
from math import isfinite
from pathlib import Path
from types import SimpleNamespace

import torch
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from diffusers.models.downsampling import Downsample2D
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.transformers.transformer_2d import Transformer2DModel
from diffusers.models.unets.unet_2d_blocks import CrossAttnDownBlock2D
from diffusers.models.upsampling import Upsample2D

from particle_sliders.prompt_steering import PromptSteering
from particle_sliders.strength_schedule import match_binary32_interval_dose

from sd_native_ddim import normalized_sigma_coordinates, render_ddim


RAW_SCHEMA = "sd1p5.exact_all_layer_A_PSR_raw_same_first_progress.v1"
PARAMETERS = {"direction.weight", "probe.weight", "probe.bias"}
NATIVE_BLOCKS = (ResnetBlock2D, Transformer2DModel, Downsample2D, Upsample2D)
# Exact publisher beta settings with the archived eta-zero DDIM constructor.
STOCK_DDIM_CONFIG = {
    "num_train_timesteps": 1000, "beta_start": 0.00085, "beta_end": 0.012,
    "beta_schedule": "scaled_linear", "trained_betas": None,
    "clip_sample": False, "set_alpha_to_one": True, "steps_offset": 0,
    "prediction_type": "epsilon", "thresholding": False,
    "timestep_spacing": "leading", "rescale_betas_zero_snr": False,
}
SOURCE_PINS = (
    (UNet2DConditionModel, "0976bd0b650b2d6889f78fd3c8776ba365bd1f193dec18f72bdafdbadf91fe50"),
    (CrossAttnDownBlock2D, "545d158c9d98fb8971c501303117fd3e32d27ae28506b89005c65548404f7437"),
    (DDIMScheduler, "a61e4d99402d1186d2aec1bec8b0f336108b06d72c07e7eb2eb837a44a9b196c"),
    (AutoencoderKL, "091d72ea9ae2f45f465e70f05142e04850d1153052dca520291ceb707bda8c98"),
    (PromptSteering, "67ee03066e255fe105735820b0edba53757d943b93239b1279de707d23dbae59"),
    (match_binary32_interval_dose, "4ffd07fa3793bb0d96e9e207dd45e5516a422497def938a7db7fd083ecf600f4"),
)
CLOCK_PATH = Path(__file__).with_name("SD15-APSR-clock.json")


@dataclass(frozen=True)
class ReplayArm:
    pixels: torch.Tensor
    coefficients: dict[str, torch.Tensor] | None
    final_latent_witness: dict


@dataclass(frozen=True)
class ReplayComparison:
    off: ReplayArm
    adaptive: ReplayArm
    constant: ReplayArm
    timesteps: tuple[int, ...]
    coordinates: tuple[float, ...]
    lattices: dict[str, tuple[int, int]]
    constant_fields: dict[str, torch.Tensor]
    rounding: dict[str, dict]


def _check_owner(owner):
    if not callable(owner) or owner() is not True:
        raise PermissionError("The independently admitted replay owner is required")


def _stock_clock():
    data = CLOCK_PATH.read_bytes()
    if sha256(data).hexdigest() != "14970ea7194bf1fb4502f629d99d43da63adaa1a3261695483666039c2670424":
        raise ValueError("The registered native SD1.5 topology or DDIM clock changed")
    clock = json.loads(data)
    return tuple(clock["layers"]), tuple(clock["timesteps"]), tuple(clock["coordinates"])


def _require_stock_models(unet, vae):
    if type(unet) is not UNet2DConditionModel or type(vae) is not AutoencoderKL:
        raise TypeError("Require the stock SD1.5 UNet and VAE classes")


def _require_stock_scheduler(scheduler, timesteps, coordinates, device):
    if type(scheduler) is not DDIMScheduler:
        raise TypeError("Require the stock DDIM scheduler")
    if any(scheduler.config[name] != value for name, value in STOCK_DDIM_CONFIG.items()):
        raise ValueError("The stock DDIM scheduler configuration differs")
    reference = DDIMScheduler(**STOCK_DDIM_CONFIG)
    if scheduler.init_noise_sigma != reference.init_noise_sigma:
        raise ValueError("The stock DDIM initial noise scale differs")
    for name in ("alphas_cumprod", "final_alpha_cumprod"):
        actual, expected = getattr(scheduler, name), getattr(reference, name)
        if actual.dtype != expected.dtype or not torch.equal(actual.cpu(), expected):
            raise ValueError(f"The stock DDIM coefficients differ at {name}")
    planned = deepcopy(scheduler)
    planned.set_timesteps(len(timesteps), device=device)
    if normalized_sigma_coordinates(planned) != (timesteps, coordinates):
        raise ValueError("The stock DDIM timestep or dose clock differs")


def _native_blocks(unet):
    return {
        name: module
        for name, module in unet.named_modules()
        if name.startswith(("down_blocks.", "mid_block.", "up_blocks."))
        and isinstance(module, NATIVE_BLOCKS)
    }


def _probes_from_raw(packet, *, qualified_updates, layers, device):
    if (not isinstance(packet, dict) or packet.get("schema") != RAW_SCHEMA
            or packet.get("weights") != "raw" or "ema_decay" not in packet
            or packet["ema_decay"] is not None or type(qualified_updates) is not int
            or qualified_updates < 1
            or type(packet.get("completed_optimizer_updates")) is not int
            or packet["completed_optimizer_updates"] != qualified_updates):
        raise ValueError("Require the independently qualified RAW A-PSR checkpoint with no EMA")
    states = packet.get("raw_probes")
    if not isinstance(states, dict) or tuple(states) != layers:
        raise ValueError("The RAW checkpoint must contain every probe in registered layer order")
    probes = {}
    for name, state in states.items():
        if (not isinstance(state, dict) or state.keys() != PARAMETERS
                or any(not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32
                       or not torch.isfinite(tensor).all() for tensor in state.values())):
            raise ValueError(f"Invalid finite FP32 RAW probe state at {name}")
        shape = state["probe.weight"].shape
        if (len(shape) != 2 or shape[0] != 1 or not shape[1]
                or state["direction.weight"].shape != shape or state["probe.bias"].shape != (1,)):
            raise ValueError(f"Invalid RAW probe geometry at {name}")
        probes[name] = _materialize_probe(state, shape[1], device)
    return probes


def _materialize_probe(state, width, device):
    # Construct on meta so replay loads the frozen tensors without consuming the RNG.
    probe = PromptSteering(width, device="meta", dtype=torch.float32)
    probe.load_state_dict(
        {key: tensor.detach().to(device=device, copy=True) for key, tensor in state.items()},
        assign=True,
    )
    return probe.eval().requires_grad_(False)


def tensor_witness(value):
    data = value.detach().cpu().contiguous()
    return dict(shape=list(data.shape), dtype=str(data.dtype),
                sha256=sha256(data.view(torch.uint8).numpy().tobytes()).hexdigest())


def _state(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple):
        return output[0]
    return output.sample


def _replace_state(output, edited):
    if isinstance(output, torch.Tensor):
        return edited
    if isinstance(output, tuple):
        return (edited, *output[1:])
    return replace(output, sample=edited)


def _match_field(trace, lattice, coordinates):
    positions = lattice[0] * lattice[1]
    if (trace.dtype != torch.float32 or trace.device.type != "cpu"
            or trace.shape != (len(coordinates) - 1, positions)):
        raise ValueError("Expected complete native binary32 coefficient lattice")
    matches = [match_binary32_interval_dose(values, coordinates) for values in trace.T.tolist()]
    constants = [match.constant_strength for match in matches]
    errors = [abs(match.constant_dose - match.adaptive_dose) for match in matches]
    field = torch.tensor(constants, dtype=torch.float32).reshape(lattice)
    if field.flatten().tolist() != constants:
        raise ValueError("The shared fixed field changed at the binary32 coefficient boundary")
    return field, {
        "positions": positions,
        "exact_dose_positions": errors.count(0),
        "max_absolute_coefficient_dose_rounding_error": str(max(errors)),
    }


class _ReplayArm:
    def __init__(self, unet, vae, layers, timesteps, modules, owner, probes=None, alpha=None,
                 lattices=None, fixed_fields=None):
        self.unet = unet
        self.vae = vae
        self.config = unet.config
        self.parameters = unet.parameters
        self.layers = layers
        self.timesteps = timesteps
        self.modules = modules
        self.owner = owner
        self.probes = probes
        self.alpha = alpha
        self.lattices = lattices
        self.fixed_fields = fixed_fields
        self.visits = []
        self.fields = None if probes is None else {name: [] for name in layers}
        self.decoded = False

    def __call__(self, model_input, timestep, *, encoder_hidden_states):
        _check_owner(self.owner)
        index = len(self.visits)
        if index >= len(self.timesteps) or int(timestep) != self.timesteps[index]:
            raise ValueError("A native DDIM position was omitted, repeated, or reordered")
        observed = {}
        with ExitStack() as stack:
            if self.probes is not None:
                for name, module in self.modules.items():
                    stack.callback(module.register_forward_hook(self._edit(name, observed)).remove)
            sample = self.unet(model_input, timestep, encoder_hidden_states=encoder_hidden_states).sample
        if (self.probes is not None and tuple(observed) != self.layers
                or sample.shape != model_input.shape or sample.dtype != torch.float16
                or not torch.isfinite(sample).all()):
            raise ValueError("Native stock forward did not return all required boundaries and finite FP16 epsilon")
        for name, field in observed.items():
            self.fields[name].append(field[0].to(device="cpu", copy=True))
        self.visits.append(int(timestep))
        _check_owner(self.owner)
        return SimpleNamespace(sample=sample)

    def _edit(self, name, observed):
        def edit(_module, _args, output):
            index = len(observed)
            if index >= len(self.layers) or self.layers[index] != name:
                raise ValueError("A native residual was omitted, repeated, or reordered")
            state = _state(output)
            probe = self.probes[name]
            if (state.ndim != 4 or state.shape[:2] != (2, probe.probe.in_features)
                    or state.dtype != torch.float16 or not all(state.shape[-2:])):
                raise ValueError(f"The paired FP16 NCHW residual changed at {name}")
            lattice = tuple(state.shape[-2:])
            if self.lattices.setdefault(name, lattice) != lattice:
                raise ValueError(f"The native spatial lattice changed at {name}")
            conditional = state[1:2].flatten(start_dim=2).transpose(1, 2)
            if self.fixed_fields is None:
                edited, predicted = probe(conditional, self.alpha)
                field = predicted.new_tensor(self.alpha) * predicted
            else:
                field = self.fixed_fields[name].reshape(1, -1)
                scaled = field[..., None]
                delta = scaled * probe.direction.weight[0]
                edited = torch.where(scaled == 0, conditional, (conditional.float() + delta).to(state.dtype))
            if (field.dtype != torch.float32 or field.shape != conditional.shape[:2]
                    or not torch.isfinite(field).all() or (field < 0).any()
                    or not torch.isfinite(edited).all()):
                raise ValueError(f"Invalid realized FP32 coefficient or edited residual at {name}")
            observed[name] = field
            spatial = edited.transpose(1, 2).reshape(state[1:2].shape)
            return _replace_state(output, torch.cat((state[:1], spatial), dim=0))
        return edit

    def decode(self, *args, **kwargs):
        _check_owner(self.owner)
        if self.decoded or tuple(self.visits) != self.timesteps:
            raise ValueError("Complete the registered stock UNet clock before decoding")
        decoded = self.vae.decode(*args, **kwargs)
        self.decoded = True
        _check_owner(self.owner)
        return decoded

    def render(self, scheduler, inputs, coordinates):
        vae = SimpleNamespace(config=self.vae.config, parameters=self.vae.parameters, decode=self.decode)
        device_type = next(self.unet.parameters()).device.type
        with torch.autocast(device_type=device_type, enabled=False):
            pixels, latent, trace = render_ddim(self, vae, scheduler, steps=len(self.timesteps), **inputs)
        if (not self.decoded or trace["native_timesteps"] != self.timesteps
                or trace["normalized_sigma_coordinates"] != coordinates):
            raise ValueError("The finished render did not execute its exact registered clock")
        fields = None if self.fields is None else {
            name: torch.stack(rows) for name, rows in self.fields.items()
        }
        return ReplayArm(pixels, fields, tensor_witness(latent))


def _verify_stock_runtime(unet, vae, scheduler):
    _require_stock_models(unet, vae)
    for source, expected in SOURCE_PINS:
        if sha256(Path(getsourcefile(source)).read_bytes()).hexdigest() != expected:
            raise ValueError(f"The pinned replay runtime source changed for {source.__name__}")
    layers, timesteps, coordinates = _stock_clock()
    device = next(unet.parameters()).device
    for model, dtype in ((unet, torch.float16), (vae, torch.float32)):
        if model.training or any(weight.requires_grad or weight.dtype != dtype or weight.device != device
                                 for weight in model.parameters()):
            raise ValueError("Require the frozen same-device FP16 UNet and FP32 stock VAE")
    _require_stock_scheduler(scheduler, timesteps, coordinates, device)
    modules = _native_blocks(unet)
    if modules.keys() != set(layers):
        raise ValueError("The loaded UNet does not expose all 44 registered native boundaries")
    return layers, timesteps, coordinates, modules, device


def _replay_inputs(initial_noise, negative, positive, guidance):
    if any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all()
           for value in (initial_noise, negative, positive)):
        raise ValueError("Require finite replay inputs")
    return dict(initial_noise=initial_noise.detach().clone(), negative=negative.detach().clone(),
                positive=positive.detach().clone(), guidance=guidance)


def _render_controls(unet, vae, scheduler, layers, timesteps, coordinates, modules, owner,
                     inputs, probes, alpha):
    lattices = {}
    adaptive = _ReplayArm(unet, vae, layers, timesteps, modules, owner, probes, alpha,
                          lattices).render(scheduler, inputs, coordinates)
    fields, rounding = {}, {}
    for name in layers:
        _check_owner(owner)
        fields[name], rounding[name] = _match_field(adaptive.coefficients[name], lattices[name], coordinates)
    device = next(unet.parameters()).device
    on_device = {name: field.to(device=device, copy=True) for name, field in fields.items()}
    constant = _ReplayArm(unet, vae, layers, timesteps, modules, owner,
                          probes=probes, lattices=lattices, fixed_fields=on_device).render(
                              scheduler, inputs, coordinates)
    for name, field in fields.items():
        applied = constant.coefficients[name].view(torch.int32)
        expected = field.reshape(1, -1).view(torch.int32).expand_as(applied)
        if not torch.equal(applied, expected):
            raise ValueError(f"Fixed coefficient bits changed during the native replay at {name}")
    _check_owner(owner)
    return adaptive, constant, lattices, fields, rounding


@torch.no_grad()
def render_apsr_comparison(unet, vae, scheduler, *, raw_packet, qualified_updates, alpha,
                           initial_noise, negative, positive, guidance, owner):
    """Render OFF, adaptive, and same-RAW fixed control for one admitted case.

    The caller supplies a separately qualified RAW packet and the stock assets
    and inputs. The reported coefficients are the values used before
    each edit is cast to the native residual dtype.
    """
    _check_owner(owner)
    if type(alpha) not in (int, float) or not isfinite(alpha) or alpha < 0:
        raise ValueError("Require an explicit finite nonnegative A-PSR strength")
    if not torch.isfinite(torch.tensor(alpha, dtype=torch.float32)):
        raise ValueError("A-PSR strength must be finite in the native coefficient dtype")
    layers, timesteps, coordinates, modules, device = _verify_stock_runtime(unet, vae, scheduler)
    probes = _probes_from_raw(raw_packet, qualified_updates=qualified_updates, layers=layers,
                              device=device)
    inputs = _replay_inputs(initial_noise, negative, positive, guidance)
    off = _ReplayArm(unet, vae, layers, timesteps, modules, owner).render(scheduler, inputs, coordinates)
    adaptive, constant, lattices, fields, rounding = _render_controls(
        unet, vae, scheduler, layers, timesteps, coordinates, modules, owner, inputs, probes, alpha)
    return ReplayComparison(off, adaptive, constant, timesteps, coordinates, lattices, fields, rounding)


@torch.no_grad()
def verify_apsr_zero_technical(unet, vae, scheduler, *, widths, initial_noise, negative, positive,
                              guidance, reference, owner):
    """Run the two new zero-edit arms against the saved stock OFF digests."""
    _check_owner(owner)
    layers, timesteps, coordinates, modules, device = _verify_stock_runtime(unet, vae, scheduler)
    inputs = _replay_inputs(initial_noise, negative, positive, guidance)
    if (tuple(widths) != layers
            or any(type(width) is not int or width < 1 for width in widths.values())):
        raise ValueError("Require all 44 verified original layer widths in order")
    for input_name, receipt_name in (("initial_noise", "initial_native_latent"),
                                     ("negative", "unconditional_CLIP"), ("positive", "positive_CLIP")):
        if tensor_witness(inputs[input_name]) != reference[receipt_name]:
            raise ValueError(f"Original independently verified stock input changed at {input_name}")
    probes = {}
    for name in layers:
        _check_owner(owner)
        width = widths[name]
        # A signal and direction are present, so dropping the zero scale would
        # change the activation. These fixture tensors have no fit lineage.
        direction = torch.zeros((1, width), dtype=torch.float32)
        direction[0, 0] = 1
        state = {"probe.weight": torch.zeros((1, width), dtype=torch.float32),
                 "probe.bias": torch.ones((1,), dtype=torch.float32),
                 "direction.weight": direction}
        probes[name] = _materialize_probe(state, width, device)
    adaptive, constant, _, _, rounding = _render_controls(
        unet, vae, scheduler, layers, timesteps, coordinates, modules, owner, inputs, probes, 0)
    for arm in (adaptive, constant):
        if (tensor_witness(arm.pixels)["sha256"] != reference["pixel_sha256"]
                or arm.final_latent_witness != reference["last_native_latent"]
                or any(torch.count_nonzero(field) for field in arm.coefficients.values())):
            raise ValueError("The zero replay differs from the archived stock OFF witnesses")
    if any(audit["positions"] != audit["exact_dose_positions"] for audit in rounding.values()):
        raise ValueError("The zero coefficient dose must agree exactly")
    _check_owner(owner)
    return dict(native_UNet_forwards=2 * len(timesteps), native_VAE_calls=2,
                stock_OFF_forwards=0, equal_pixel_digest=True, equal_final_latent_digest=True,
                zero_coefficients=True, native_boundaries=len(layers), ema_decay=None)
