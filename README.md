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

The reference is `particle-sliders-core==0.15.1` from
[`c13890f8e2a1ecf1dc8eb9f2a37f9187cb9fea78`](https://github.com/HyperGAN/particle-sliders/commit/c13890f8e2a1ecf1dc8eb9f2a37f9187cb9fea78),
which requires official `particlegan==0.11.0` and Torch `>=2.6,<3`. The core checks
the exhaustive Python source manifest for the `release-0.11.0` profile. The check
script verifies that profile and imports the shared engine and CLI on CPU. It
downloads no model weights and performs no native SD training or qualification.
See the pinned core's
[compatibility contract](https://github.com/HyperGAN/particle-sliders/blob/c13890f8e2a1ecf1dc8eb9f2a37f9187cb9fea78/packages/particle-sliders-core/COMPATIBILITY.md).

This is current dependency/reference upkeep for the scaffold. It does not revive
the retired implementation, admit its old cohort against the new core, convert
historical MSE/SD3 weights, or train or publish sliders. A future native
integration would need its own implementation, source pins and qualification.
