"""Stable Diffusion bindings only. The shared core owns all GAN/lifecycle math."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
from importlib.metadata import version
import inspect
import json
import math
from pathlib import Path
import re

import torch
from torch import nn

from particle_sliders.fibo_hndl import HNDLFieldCritic, HNDLParticleProjection, verified_builder
from particle_sliders.inference import InferenceBackend, InferenceBinding, native_identity
from particle_sliders.lifecycle import Binding, SessionConfig, _restore_all
from particle_sliders.pg094 import ConditionalPG094
from particle_sliders.pg09 import PG_SOURCE_PROFILE, validate_pg09_domain
from particle_sliders.pg09_backends import ResponseRow
from particle_sliders.pg09_images import ImageResponseAdapter, image_condition_features
from particle_sliders.response import ResponseBackend, ResponseCapabilities

NAME = 'sd_shared_game'
COHORT = 'sd-native-field-pg094-hg-hndl010-v1'
PROTOCOL = COHORT + ':unipolar-positive-raw-prediction'
TARGET = re.compile(r'^.+\.attn2\.(to_q|to_k|to_v|to_out\.0)$')


def _retry_cleanup(actions):
    """Retire only completed actions, including during exception unwinding."""
    pending = list(actions)
    def close():
        def attempt(action):
            # A nested best-effort cleanup can return False when its failures
            # were added to an active original exception instead of raised.
            if action() is not False:
                pending.remove(action)
        _restore_all([lambda action=action: attempt(action) for action in tuple(pending)])
        return not pending
    return close


def _restore_buffer(buffer, value):
    with torch.no_grad():
        if (buffer.shape, buffer.dtype, buffer.device) == (value.shape, value.dtype, value.device):
            buffer.copy_(value)
        else:
            buffer.data = value.clone()


def _borrowed_restorations(model):
    """Separate native restoration leaves so failures cannot retire other leaves."""
    actions = []
    for module in model.modules():
        actions.append(lambda module=module, mode=module.training: setattr(module, 'training', mode))
        for key in ('gradient_checkpointing', '_gradient_checkpointing_func'):
            if hasattr(module, key):
                actions.append(lambda module=module, key=key, value=getattr(module, key):
                               setattr(module, key, value))
            else:
                def remove_added(module=module, key=key):
                    if hasattr(module, key):
                        delattr(module, key)
                actions.append(remove_added)
        registered, nonpersistent = dict(module._buffers), set(module._non_persistent_buffers_set)
        for buffer in registered.values():
            if buffer is not None:
                actions.append(lambda buffer=buffer, value=buffer.detach().clone(): _restore_buffer(buffer, value))
        def restore_registration(module=module, registered=registered, nonpersistent=nonpersistent):
            module._buffers.clear()
            module._buffers.update(registered)
            module._non_persistent_buffers_set.clear()
            module._non_persistent_buffers_set.update(nonpersistent)
        actions.append(restore_registration)
    for parameter in model.parameters():
        actions.append(lambda parameter=parameter, flag=parameter.requires_grad: parameter.requires_grad_(flag))
        actions.append(lambda parameter=parameter, gradient=parameter.grad: setattr(parameter, 'grad', gradient))
    return actions


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime():
    """Check the exact modern source closure before modifying a borrowed model."""
    if version('particle-sliders-core') != '0.12.0' or version('diffusers') != '0.35.1':
        raise ValueError('install the isolated SD core0.12/diffusers0.35.1 environment')
    import diffusers
    root = Path(diffusers.__file__).parent
    expected = json.loads(Path(__file__).with_name('native_sources.json').read_text())
    actual = {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*.py'))}
    if actual != expected['diffusers']:
        raise ValueError('Diffusers source closure differs from the SD cohort')
    import particle_sliders
    core_root = Path(particle_sliders.__file__).parent
    actual_core = {str(p.relative_to(core_root)): sha(p) for p in sorted(core_root.rglob('*'))
                   if p.is_file() and p.suffix in ('.py', '.json', '.hndl')}
    if actual_core != expected['particle_sliders']:
        raise ValueError('shared core source closure differs from the SD cohort')
    builder, identity = verified_builder()
    validate_pg09_domain((1, 4, 8, 8), response_domain='continuous_field')
    return builder, identity


def targets(model):
    from diffusers import UNet2DConditionModel
    if type(model) is not UNet2DConditionModel:
        raise ValueError('SD requires an actual UNet2DConditionModel')
    cfg = model.config
    if (cfg.in_channels != 4 or cfg.out_channels != 4 or cfg.addition_embed_type is not None
            or cfg.class_embed_type is not None or cfg.encoder_hid_dim is not None
            or not isinstance(cfg.cross_attention_dim, int)):
        raise ValueError('unsupported SD UNet architecture; SDXL/SD3 require their own binding')
    if any(getattr(m, 'gradient_checkpointing', False) for m in model.modules()):
        raise ValueError('this SD cohort requires checkpointing disabled before binding')
    if any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('this qualified SD cohort supports CPU float32 only')
    selected = {name: m for name, m in model.named_modules() if TARGET.fullmatch(name)}
    if not selected or any(type(m) is not nn.Linear or 'forward' in m.__dict__
                           or m._forward_hooks or m._forward_pre_hooks for m in selected.values()):
        raise ValueError('SD cross-attention leaves must be plain unhooked Linear projections')
    if len({id(m) for m in selected.values()}) != len(selected):
        raise ValueError('aliased SD targets')
    return selected


def model_identity(model, scheduler, response_shape=(1, 4, 8, 8)):
    """Bind actual frozen weights/config/sources, never a caller's model label."""
    from diffusers import DDIMScheduler, DDPMScheduler
    targets(model)
    if type(scheduler) not in (DDIMScheduler, DDPMScheduler):
        raise ValueError('only native DDIM/DDPM raw discrete prediction is qualified')
    if scheduler.config.prediction_type not in ('epsilon', 'v_prediction'):
        raise ValueError('declare epsilon or v_prediction')
    def tensor_id(value):
        value = value.detach().cpu().contiguous()
        return dict(shape=list(value.shape), dtype=str(value.dtype),
                    sha256=hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())
    # All native module implementations participate, including attention processors.
    classes = {type(m) for m in model.modules()} | {type(scheduler)}
    classes.update(type(m.processor) for m in model.modules() if hasattr(m, 'processor'))
    def config_identity(value):
        # Diffusers creates this bookkeeping list from a set. Its order is not
        # model semantics and varies with Python's per-process hash seed.
        return {key: sorted(item) if key == '_use_default_values' else item
                for key, item in value.items()}
    return dict(protocol=PROTOCOL, response_shape=list(response_shape),
                config=config_identity(model.config), scheduler=config_identity(scheduler.config),
                scheduler_class=type(scheduler).__name__,
                weights={name: tensor_id(value) for name, value in model.state_dict().items()},
                sources={cls.__module__ + '.' + cls.__qualname__: sha(inspect.getsourcefile(cls))
                         for cls in classes})


class SDParticleAdapter(nn.Module):
    """Cross-attention rank8/alpha8 routed branches with one shared128x4 bank."""
    def __init__(self, selected, builder, identity, seed):
        super().__init__()
        self.scale = 0.
        self.target_names = tuple(selected)
        self.identity = deepcopy(identity)
        self._attached = []
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.particles = nn.Parameter(torch.randn(128, 4))
            self.adapters = nn.ModuleDict({name.replace('.', '-'): HNDLParticleProjection(
                name, module, self.particles, 8, 8, builder=builder, identity=identity, attach=False)
                for name, module in selected.items()})
        self.names = tuple(name for name, _ in self.named_parameters())

    def attach(self, selected):
        try:
            for name, module in selected.items():
                owned = self.adapters[name.replace('.', '-')].forward
                self._attached.append((module, owned))
                module.forward = owned
        except BaseException:
            self.close()
            raise

    def close(self):
        def detach(record):
            module, owned = record
            # A replacement belongs to a later owner; never delete its forward.
            if module.__dict__.get('forward') is owned:
                delattr(module, 'forward')
            self._attached.remove(record)
        _restore_all([lambda record=record: detach(record) for record in tuple(reversed(self._attached))])
        return not self._attached

    @contextmanager
    def scaled(self, scale):
        if not math.isfinite(scale) or not 0 <= scale <= 1:
            raise ValueError('SD unipolar strength must be in [0,1]')
        previous = self.scale
        multipliers = [m.multiplier for m in self.adapters.values()]
        try:
            self.scale = float(scale)
            for module in self.adapters.values():
                module.multiplier = self.scale
            yield self
        finally:
            self.scale = previous
            for module, value in zip(self.adapters.values(), multipliers):
                module.multiplier = value

    def extra_repr(self):
        return json.dumps(dict(protocol=PROTOCOL, targets=self.target_names, runtime=self.identity), sort_keys=True)


class SDResponse(ImageResponseAdapter):
    def metadata(self):
        result = super().metadata()
        result.update(protocol=PROTOCOL, particle_bank='actual shared128x4 routed bank',
                      response_coordinates='native [1,4,H,W] scheduler-declared epsilon or v_prediction',
                      target='frozen positive raw prediction; neutral-caption student; unipolar strength',
                      frozen_control_rows=list(getattr(self, 'frozen_control_rows', ())),
                      guidance_contract='zero-guidance rows are frozen negative controls; at least one nonzero-guidance training row',
                      trajectory_contract='raw single-timestep response only',
                      targets=list(self.network.target_names))
        return result


def validate_row(row, shape, cross_width):
    if not isinstance(row, ResponseRow) or tuple(row.latent.shape) != tuple(shape):
        raise ValueError('row requires the declared native [1,4,H,W] latent')
    if row.schedule is not None or row.timestep is None or row.timestep.numel() != 1:
        raise ValueError('SD requires one raw discrete timestep; trajectories are unsupported')
    t = float(row.timestep)
    if not math.isfinite(t) or t != int(t) or not 0 <= t < 1000:
        raise ValueError('SD discrete timestep must be an integer in [0,999]')
    if row.scale != 1 or row.strength != 1 or not math.isfinite(row.guidance) or row.guidance < 0:
        raise ValueError('SD rows require unipolar scale/strength1 and finite nonnegative CFG')
    for condition in (row.neutral, row.positive, row.negative):
        if not isinstance(condition, dict) or set(condition) != {'embeds'}:
            raise ValueError('SD conditioning requires frozen embeds only')
        value = condition['embeds']
        if (not isinstance(value, torch.Tensor) or value.ndim != 3 or value.shape[0] != 1
                or value.shape[1] < 1 or value.shape[2] != cross_width or value.requires_grad
                or value.device.type != 'cpu' or value.dtype != torch.float32 or not torch.isfinite(value).all()):
            raise ValueError('SD embeddings must be finite frozen float32 [1,tokens,cross_width]')
    if row.latent.device.type != 'cpu' or row.latent.dtype != torch.float32 or not torch.isfinite(row.latent).all():
        raise ValueError('SD latent must be finite CPU float32')


def attach(model, scheduler, *, expected=None, seed=7, response_shape=(1, 4, 8, 8)):
    builder, identity = runtime()
    validate(dict(response_shape=list(response_shape)))
    actual = model_identity(model, scheduler, response_shape)
    if expected is not None and json.loads(json.dumps(actual, sort_keys=True)) != expected:
        raise ValueError('SD frozen model/scheduler identity differs before attachment')
    selected = targets(model)
    network = SDParticleAdapter(selected, builder, identity, seed)
    # Resolve network.close at execution time so an incomplete nested cleanup
    # remains pending, even if it preserved an active original exception.
    close = _retry_cleanup([lambda: network.close(), *_borrowed_restorations(model)])
    try:
        model.eval().requires_grad_(False)
        network.attach(selected)
        response = SDResponse('sd', model, network, scheduler=scheduler, model_identity=actual)
        return response, actual, close
    except BaseException:
        _restore_all([close])
        raise


def capabilities(options):
    shape = tuple(options.get('response_shape', (1, 4, 8, 8)))
    return ResponseCapabilities('routed_particle_gan', 'image', 'SD raw native prediction',
        'continuous_field', shape, '0.9.4', PG_SOURCE_PROFILE)


def validate(options):
    allowed = {'response_shape', 'steps', 'batch', 'seed', 'preset', 'ema_decay', 'base_precision',
               'fixture', 'model_path', 'scheduler_path', 'rows_path', 'pg_source_profile', 'penalty_input_domain'}
    if set(options) - allowed or options.get('base_precision', 'float32') != 'float32':
        raise ValueError('unsupported SD option/precision')
    shape = tuple(options.get('response_shape', (1, 4, 8, 8)))
    if len(shape) != 4 or shape[:2] != (1, 4) or any(type(v) is not int or v < 1 for v in shape):
        raise ValueError('SD response must be native [1,4,H,W]')
    if options.get('preset', 'bispf_optimized') not in ('bispf', 'bispf_optimized'):
        raise ValueError('select a shared public PG preset')
    if options.get('fixture', False) is not False and options['fixture'] is not True:
        raise ValueError('fixture must be boolean')
    runtime()


def fixture_native():
    """Actual tiny Diffusers UNet/scheduler, random weights; never a pretrained claim."""
    from diffusers import UNet2DConditionModel, DDIMScheduler
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19)
        model = UNet2DConditionModel(sample_size=8, in_channels=4, out_channels=4,
            down_block_types=('CrossAttnDownBlock2D',), up_block_types=('CrossAttnUpBlock2D',),
            block_out_channels=(8,), layers_per_block=1, cross_attention_dim=8,
            attention_head_dim=4, norm_num_groups=4)
    return model, DDIMScheduler(prediction_type='epsilon')


def load_native(options):
    if options.get('fixture', False):
        if options.get('model_path') or options.get('scheduler_path'):
            raise ValueError('fixture and staged-model paths cannot be combined')
        return fixture_native()
    from diffusers import UNet2DConditionModel, DDIMScheduler
    # Only explicit existing local directories; Hub downloads are never a fallback.
    folder = Path(options['model_path']).resolve(strict=True)
    scheduler_folder = Path(options['scheduler_path']).resolve(strict=True)
    if not folder.is_dir() or not scheduler_folder.is_dir():
        raise ValueError('staged UNet and scheduler directories required')
    return (UNet2DConditionModel.from_pretrained(str(folder), local_files_only=True,
                use_safetensors=True, torch_dtype=torch.float32),
            DDIMScheduler.from_pretrained(str(scheduler_folder), local_files_only=True))


def fixture_rows(shape=(1, 4, 8, 8)):
    generator = torch.Generator().manual_seed(29)
    return [ResponseRow({'embeds': torch.randn(1, 3, 8, generator=generator)},
        {'embeds': torch.randn(1, 3, 8, generator=generator)},
        {'embeds': torch.zeros(1, 3, 8)}, torch.randn(shape, generator=generator),
        timestep=torch.tensor(50 + index * 200), guidance=1.) for index in range(2)]


def prepare(options, *, model=None, scheduler=None, rows=None):
    if model is None and scheduler is None:
        model, scheduler = load_native(options)
    elif model is None or scheduler is None:
        raise ValueError('stage both model and scheduler')
    if rows is None:
        if options.get('fixture', False):
            rows = fixture_rows(tuple(options.get('response_shape', (1, 4, 8, 8))))
        else:
            packet = torch.load(options['rows_path'], map_location='cpu', weights_only=True)
            if not isinstance(packet, list):
                raise ValueError('rows file must contain a list of ResponseRow field dictionaries')
            rows = [ResponseRow(**value) for value in packet]
    rows = list(rows)
    if len(rows) < 2:
        raise ValueError('at least two frozen SD rows required')
    for row in rows:
        validate_row(row, options.get('response_shape', (1, 4, 8, 8)), model.config.cross_attention_dim)
    if not any(row.guidance > 0 for row in rows):
        raise ValueError('SD game requires at least one nonzero-guidance training row; guidance0 is a frozen negative control')
    response, identity, close = attach(model, scheduler, seed=options.get('seed', 7),
                                       response_shape=options.get('response_shape', (1, 4, 8, 8)))
    try:
        response.frozen_control_rows = tuple(index for index, row in enumerate(rows) if row.guidance == 0)
        prepared = [response.prepare(row) for row in rows]
        conditions = image_condition_features(prepared)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(options.get('seed', 7) + 1)
            critic = HNDLFieldCritic(4, conditions.shape[1], prepared[0].target.shape[2:], width=4)
        game = ConditionalPG094(response, prepared, critic, conditions=conditions,
            particles=response.network.particles, response_domain='continuous_field',
            total_steps=options.get('steps', 4), batch_size=options.get('batch', 2),
            seed=options.get('seed', 7), preset=options.get('preset', 'bispf_optimized'))
        native_config = dict(response_shape=list(prepared[0].target.shape), fixture=options.get('fixture', False))
        spec = INFERENCE.specification(response, model_identity=identity, protocol=PROTOCOL,
                                       configuration=native_config)
        return Binding(game, close, inference=spec)
    except BaseException:
        _restore_all([close])
        raise


def inference_validate(options, spec):
    if set(options) - {'model_path', 'scheduler_path', 'base_precision'}:
        raise ValueError('native inference accepts only staged local model options')
    if spec.identity['protocol'] != PROTOCOL:
        raise ValueError('SD inference protocol differs')
    if spec.configuration['response_shape'] != spec.identity['model']['response_shape']:
        raise ValueError('SD inference native shape differs from its trained identity')
    validate(spec.configuration)


def inference_prepare(options, spec):
    model, scheduler = load_native(dict(options, fixture=spec.configuration['fixture']))
    response, identity, close = attach(model, scheduler, expected=spec.identity['model'],
                                       response_shape=spec.configuration['response_shape'])
    try:
        def predict(*, row, strength, seed):
            validate_row(row, spec.configuration['response_shape'], model.config.cross_attention_dim)
            return response._response(row, scale=strength).detach().clone()
        return InferenceBinding(response.network,
            native_identity(response, model_identity=identity, protocol=PROTOCOL), close, response=predict)
    except BaseException:
        _restore_all([close])
        raise


def config(**options):
    return SessionConfig(NAME, COHORT, **options)


class SDBackend(ResponseBackend):
    def open(self, config, **kwargs):
        if config.cohort != COHORT:
            raise ValueError('SD shared game requires its distinct exact cohort')
        return super().open(config, **kwargs)


BACKEND = SDBackend(NAME, capabilities({}), prepare, validate, configure_capabilities=capabilities)
INFERENCE = InferenceBackend(NAME, capabilities({}), inference_prepare, inference_validate,
                             configure_capabilities=capabilities)
