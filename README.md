# SD Particle Sliders

**Memory (calculated):** retired current scaffold: **0 GPU /2 GiB RAM**. Historical SD1.5 512² sampling reference only: **8 GiB VRAM /12 GiB RAM**; no current native trainer. See [MEMORY.md](MEMORY.md) for the complete budgets and assumptions.





Retired Stable Diffusion integration scaffold. There is no native trainer,
adapter package, model release or deployed application on the default branch.

Mikkel retired the SD1.5 extraction in favor of H3/MiniMax on
[PR 1](https://github.com/HyperGAN/sd-particle-sliders/pull/1#issuecomment-6549702979).
The closed implementation branch and its artifacts retain their original source
and dependency identities. The separate unified-training-command branch also
remains independent of this scaffold.

## Current shared-core bootstrap

For shared API inspection, use a new **Python 3.11 or newer** environment with
the immutable core source in [requirements-core.txt](requirements-core.txt):

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.6.0+cpu'
python -m pip install -r requirements-core.txt
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/check_core.py
python -m pip check
```

The reference is `particle-sliders-core==0.15.2` from
[`b1f780decc516808cfd3019d5e5b674457161f86`](https://github.com/HyperGAN/particle-sliders/commit/b1f780decc516808cfd3019d5e5b674457161f86),
which requires official `particlegan==0.11.1` and Torch `>=2.6,<3`. The CPU
check verifies the pinned core install, uses the core's exhaustive ParticleGAN
source manifest for `release-0.11.1`, and imports the shared engine, CLI, and
CFP, CLABS and PSR APIs. [Compatibility contract](https://github.com/HyperGAN/particle-sliders/blob/b1f780decc516808cfd3019d5e5b674457161f86/packages/particle-sliders-core/COMPATIBILITY.md)

## Experimental SD1.5 replay

The opt-in research entry compares stock OFF, adaptive A-PSR and a time-constant
control at matched native coefficient dose. It takes caller-supplied stock DDIM
inputs, qualified RAW probes and an explicit strength. A guarded zero-edit
comparison matched archived stock outputs. Held-out effect measurements for a
fitted nonzero schedule are pending. [Replay requirements and usage](docs/sd15-replay.md)
