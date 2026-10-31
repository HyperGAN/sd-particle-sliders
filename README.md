# SD Particle Sliders — opt-in shared game

This standalone repository retains the distribution `conceptmod-sd-shared-game`,
Python namespace `sd_slider_game`, and the cohort
`sd-native-field-pg094-hg-hndl010-v1`. The historical
`conceptmod/textsliders/train_lora.py` MSE/prompt-pair trainer stays in
HyperGAN/particle-sliders, with its objective, configuration, environment and
weight files unchanged. Its existing opt-in dispatch can use this installed
distribution. That trainer is deliberately not copied here. SDXL, SD3 and other
image-family trainers are outside this repository. See EXTRACTION.md and
PROVENANCE.json for the exact source allowlist and original blob mapping.

The new game uses the shared `ConditionalPG094`, `ResponseBackend`, session,
EMA, resume and native-inference interfaces. It selects an existing public
`bispf_optimized` or `bispf` recipe. It contains no copied loss, penalty,
optimizer or update law. A frozen positive-caption UNet prediction is the
teacher, and the adapted neutral-caption prediction is the student. This is a
new unipolar GAN objective, **not a migration of `prompt_pair.loss`**.

Native scheduler-declared epsilon or v-prediction stays `[B,4,H,W]` at the game
and penalty boundary. DDIM/DDPM input scaling is applied using an independent
scheduler copy. One raw integer timestep per row is supported. No sampler,
VAE, image rendering, trajectories or prompt encoder is claimed. CFG uses the
shared adapter's frozen unconditional branch and an adapted conditional branch.
Training rows supply frozen native text embeddings, latents and timesteps.
Training requires at least one row with nonzero guidance. At guidance0, the
shared symmetric CFG formula makes both teacher and student the same frozen
negative response: the graph is connected but adapter gradients are zero.
Mixed datasets may retain these rows as explicitly labeled frozen controls in
the game metadata. Native guidance0 inference remains supported; this is an
SD training-dataset admission rule, not a cross-model guidance prohibition.

Only plain native SD cross-attention `attn2.to_q/to_k/to_v/to_out.0` Linear
projections are targeted. The graph is the shared actual HyperGAN-built HNDL
rank8/alpha8 routed architecture with one trainable128x4 particle bank. Its
FieldCritic is also an actual shared HNDL graph. No PEFT LoRA, historical
conceptmod LoRA, merged weights, existing forward hooks, SDXL added conditioning
or active gradient checkpointing may be present on the staged UNet.

## Isolated installation and CPU fixture

Use a **new** Python3.11–3.14 virtual environment. The historical repository
requirements are a separate environment and must not be combined with this one.

```sh
python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.6.0+cpu'
python -m pip install '.[test]'
python -m pytest tests -q
OMP_NUM_THREADS=1 sd-shared-game --fixture --output /tmp/sd-game-fixture
```

The fixture instantiates a tiny **actual Diffusers UNet**, with random CPU
weights, and no downloads. It qualifies native API/geometry, graph execution,
derivatives and lifecycle software only. Full SD1/SD2 weights, pretrained
quality, GPU, mixed precision and production training are **unqualified**.
This cohort intentionally accepts CPU float32 only.

Core source is the immutable compatibility candidate `7a8a321d04323d49319d0b08e6093a0ddce36b27`
from [draft148](https://github.com/HyperGAN/particle-sliders/pull/148).
It selects released ParticleGAN0.9.4, HNDL0.10.0 and actual HyperGAN develop
`5f368a3770cf5bdae0448995c9f76e29af8049f7`. Diffusers0.35.1 has an exhaustive
source manifest in this wheel; both shared dependency closures and native
sources are checked before attachment. **Final core lifecycle admission is
pending the central owner's reviewed corrections.** This focused draft does
not assert that draft148 is admitted or ready to merge. Its source lock includes
the published lifecycle fixes; the older4abc candidate is refused.

## Local staged model/data API

```python
from sd_slider_game.adapter import BACKEND, config
from particle_sliders.inference import export_inference
# unet is an already-staged diffusers.UNet2DConditionModel on CPU float32;
# scheduler is a staged DDIMScheduler or DDPMScheduler.
# rows is a sequence of shared particle_sliders.pg09_backends.ResponseRow:
# neutral/positive/negative={'embeds': float32_tensor[1,tokens,cross_dim]},
# latent=float32_tensor[1,4,H,W], timestep=scalar_integer_tensor,
# guidance=1.0 (or explicit nonnegative CFG), scale=strength=1.0.
with BACKEND.open(config(response_shape=[1,4,64,64], steps=100, batch=2,
                         ema_decay=.99), model=unet, scheduler=scheduler,
                  rows=rows) as session:
    session.train()
    session.checkpoint('training.sd-pg094.pt')
    export_inference(session, 'adapter.sd-pg094.ema.pt', selection='ema')
```

For CLI operation use `--config options.json` instead of `--fixture`.
Options include `model_path` (local safetensors UNet directory),
`scheduler_path` (local DDIM scheduler directory), `rows_path` (weights-only
Torch list of ResponseRow field dictionaries), `response_shape`, `steps`,
`batch`, `seed`, `preset`, `ema_decay`. All paths must already exist.
Loading uses `local_files_only=True`; no Hub ID or remote fallback is allowed.
The actual staged weight/config/source digests form the base identity. These
content checks do not authenticate an upstream model name or license receipt.

Cleanup attempts every borrowed mode, gradient object/flag, checkpoint attribute,
buffer value/registration and owned projection forward. Successful actions are
retired individually; failed actions remain retryable after the cause is fixed.
An active original exception is preserved with restoration-failure notes. A
completed close does not replay snapshots, and detachment leaves a later owner's
forward replacement or hooks alone. Base tensors remain frozen during the session. Adapter
strength0 executes the original native projections exactly; intermediate and
strength1 use the same trained branches. Strength is constrained to `[0,1]`.

At the unchanged provisional core7a8 pin, the shared session marks itself closed
before invoking its product callback. If that callback fails, explicitly retry
`session.binding.close()` after addressing the failure; repeated
`session.close()` does not re-enter it. This shared-wrapper limitation is
reported for central correction and is not hidden by the product fix.

## Checkpoints and native-only inference

Shared training packets include the exact cohort, source/model/shape identity,
recipe, optimizer, RNG, row and EMA states. Old MSE weights and other cohorts
are rejected; they are never relabeled. Raw and EMA exports use distinct names
and shared native-inference-v2 packets, containing only the adapter state plus
native identity and origin receipt. They do not embed rows, teachers, critics
or optimizer state.

```python
from particle_sliders.inference import open_inference
# The training session is closed and training rows may have been removed.
with open_inference('adapter.sd-pg094.ema.pt', operation='response',
                    options={'model_path': '/local/unet',
                             'scheduler_path': '/local/scheduler'}) as native:
    prediction = native.response(row=fresh_row, strength=.5)
```

The inference factory loads only the staged base/scheduler and adapter. Changed
base weights, source, scheduler, native geometry and protocol are refused. It
does not create a game, critic, teacher cache or training rows. The complete
native configuration must match; native library configuration fields can also
bind staging paths. Reproducing full training requires the
original fixed-row data and exact dependency environment.

This source export does not change model code or artifact identities. All runtime
Python sources, tests and the native source lock retain their original bytes.
Only root packaging/license metadata and repository documentation are adapted.
The source-only implementation remains subject to actual-diff Claude review,
core qualification and root merge decisions; no native admission follows from
repository separation.
