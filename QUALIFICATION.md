# CPU software qualification, 2026-10-31

Original umbrella repository base: `3c25f785d14073e6f627c0fa9d2297a041c79913`.
The isolated opt-in package uses core candidate
`7a8a321d04323d49319d0b08e6093a0ddce36b27` from draft148, released
ParticleGAN0.9.4, actual HyperGAN develop
`5f368a3770cf5bdae0448995c9f76e29af8049f7`, HNDL0.10.0 and Diffusers0.35.1.
Central post-fix review/admission and root's Claude review remain required
before landing. No merge, package release or native training is included.

Python3.12.15/Linux, Torch2.6.0+cpu: **33 installed-wheel tests pass**, and the
same33 tests pass after reinstalling the corrected sdist in the second
environment originally provisioned with public CPU dependency resolution. The
dependency pins are unchanged. Both environments
pass `pip check`. The package checks all49 shared core Python/JSON/HNDL files
and the exhaustive Diffusers Python source closure before attaching adapters.
Actual installed product source files were also compared byte-for-byte with
the checkout. No source `PYTHONPATH` or editable install is used by these tests.

The tests execute actual tiny randomly initialized Diffusers UNets, the actual
verified HyperGAN-built `hndl.torch.Network` projections/critic, and the shared
released-PG game/session. They cover:

- DDIM/DDPM raw epsilon and v-prediction, native spatial geometry and CFG.
- Matched routed weights through the full native UNet: outputs and input,
  embedding, particle-bank and every projection/router/bridge parameter VJP.
- Matched native-field critic outputs and input/parameter VJPs; actual public
  BISPF penalty values and critic-weight second derivatives.
- Both public presets: actual generator/particle updates, optimizer ownership,
  full versus split exact resume, EMA, strengths0/.5/1 and observation isolation.
- Exact raw/EMA native-only output parity after training data deletion; a new
  Python process with a different hash seed; no game/teacher/critic construction.
- Rejection of old MSE packets, changed cohorts, sources, base weights,
  scheduler/shape, nonfinite state, unsupported precision and checkpointing.
  Invalid restore packets do not reach live load hooks or alter owned state.
- Teacher, critic, game, inference-spec and update failure cleanup; restoration
  of borrowed modes, original gradient objects/flags and native forwards;
  nested strength scopes; cleanup continues after another action fails and
  preserves the original exception.

The bounded lifecycle correction reproduces three regressions against the
original installed994a98d package. The corrected product keeps a per-action
pending journal instead of marking cleanup complete before restoration.
Fail-before-restoration coverage now includes hook deletion, mode, gradient
flag/object, buffer value/registration and checkpoint restoration. A later
retry completes only pending actions; completed actions do not overwrite new
caller state. A later owner's native forward or hooks remain intact, including
after repeated successful close. Shared best-effort/original-exception behavior
is retained; no shared helper is modified.

Four paired old/new fixed-seed cases (both public presets, epsilon/v-prediction,
CFG3) match bit-exactly for initial128x4 bank/adapter, teacher and neutral fields,
updated bank/adapter/critic, both optimizer states, EMA, metrics and student
outputs at strengths0/.5/1. Prediction units, native geometry, teacher/student
CFG, bank and objective are unchanged. Corrected implementation source hashes
remain part of artifact identity; old994a98d packets retain their original
source environment and are not relabeled.

The unchanged core7a8 shared session marks itself closed before calling its
product callback. Its repeated`session.close()` does not retry that callback;
the tested explicit retry seam is`session.binding.close()`. This wrapper limit
is reported centrally. No core repin or native admission is claimed here.

The focused SD guidance0 characterization confirms that teacher and student
equal the same frozen negative field and that the graph remains connected
while every adapter gradient is zero. Two regressions fail before correction:
all-zero-guidance datasets were admitted, and mixed controls were unlabeled.
The corrected factory refuses an entirely zero-guidance training dataset
before attachment. Mixed zero/nonzero datasets label zero-guidance rows as
frozen controls; actual nonzero-row projection updates still occur. Native
guidance0 inference and strength-zero neutral identity remain supported. This
validation changes no nonzero-guidance equations or other model families.

The historical trainer is byte-identical to its base after removing the nine
new opt-in dispatch lines. Original trainer SHA256:
`591177cc32c98d152ed586917670223c4f3c782281b1d9d427cefff5f81a3628`.
No historical requirements, objectives, defaults or checkpoint writers change.

Reproduction from this standalone repository root, in a new environment:

```sh
python -m pip install --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu 'torch==2.6.0+cpu' 'build' 'setuptools>=84.2.1'
python -m build --no-isolation --outdir /tmp/sd-dist .
python -m pip install '/tmp/sd-dist/conceptmod_sd_shared_game-0.1.0-py3-none-any.whl[test]'
OMP_NUM_THREADS=1 python -m pytest tests -q
python -m pip check
OMP_NUM_THREADS=1 sd-shared-game --fixture --output /tmp/sd-fixture
# In another new environment, resolve/install the sdist and repeat the tests:
python -m pip install --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu 'torch==2.6.0+cpu' '/tmp/sd-dist/conceptmod_sd_shared_game-0.1.0.tar.gz[test]'
```

The dispatcher smoke completed4 shared updates and emitted distinct training,
raw and EMA packets with native response shape `[1,4,8,8]` and the declared new
cohort. The subprocess regression also confirms native-only loading without
reconstructing training data or its factory.

These are bounded **CPU API fixtures**, not SD1/SD2 pretrained qualification,
quality evidence, GPU tests or a production training run. No model weights were
downloaded. The modern cohort accepts CPU float32 only; it exposes numerical
raw prediction responses and does not render images or implement a sampler.
No wide-vector or domain-admission guard is changed. There is no new shared API
gap for this spatial SD binding. The repository has no hosted unit-test workflow;
local test receipts are not hosted CI.

## Standalone extraction

The versioned results above were recorded for the original source candidate.
The repository-root relocation is verified separately using an installed wheel
and sdist from a neutral directory, with no umbrella or export source checkout
on the import path. Runtime/test/source-lock files are byte-identical to the
original source; root documentation and license packaging metadata are adapted.
EXTRACTION.md and PROVENANCE.json record the allowlist, license and identity
boundaries. Historical embedded trainers remain in the umbrella unchanged.
