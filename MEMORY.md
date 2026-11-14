# Memory requirements

| Operation / profile | GPU count | Calculated VRAM per GPU (GiB) | Calculated host RAM (GiB) | Input limits / precision |
| --- | ---: | --- | ---: | --- |
| Current retired scaffold / core check | 0 | 0 | 2 | No model; no native training or sampling |
| Historical SD1.5 sampling planning reference | 1 | 8 | 12 | B1;512²;CLIP77;FP16;SDPA;no offload;not implemented on main |


These are **calculated planning budgets, not measured peaks or hard runtime ceilings**.
They apply only to the bounded profiles in the table, for one process/slider at a
time. GiB means 2^30 bytes. Add concurrent jobs' resident allocations; do not sum
mutually exclusive loading, conditioning, training and decoding phases as if all
their models were resident together. Allow the stated host RAM independently of
VRAM. Larger batches, longer inputs, extra references, FP32 bases, different
attention kernels or allocator fragmentation can require more. There is no finite
universal maximum for arbitrary inputs. These calculations start no model jobs.

## Current repository

This is a retired scaffold. [README](README.md), [core check](scripts/check_core.py)
and [requirements](requirements-core.txt) do not implement native training,
model loading, sampling or an adapter. Current main therefore needs **no GPU**;
the2 GiB host allowance covers interpreter/package imports and metadata checks.
It does not reserve or allocate model weights.

## Explicitly historical reference

For comparison only, the second row is a calculated SD1.5 **sampling** profile,
not a restored feature or a benchmark:
`stable-diffusion-v1-5/stable-diffusion-v1-5@451f4fe16113bff5a5d2269ed5ad43b0592e9a14`.
The pinned FP16 serialized files upper-bound tensor residency:

| Frozen component | Bytes | GiB |
| --- | ---: | ---: |
| UNet | 1,719,125,304 | 1.601 |
| CLIP text encoder | 246,144,864 | 0.229 |
| VAE | 167,335,342 | 0.156 |

Frozen sum=1.986 GiB. Add2.5 GiB for the512² UNet skip/attention
working tensors (64² latent, channels320/640/1280,77 text tokens),1 GiB VAE
decode and2.5 GiB CUDA/kernel/allocator headroom: round up to8 GiB. Host12 allows
up to4 GiB source FP32 weights while loading/converting, a second component copy,
tokenizer/Python and image/decode buffers. No KV cache persists across diffusion
steps. Efficient SDPA is assumed; this is not a budget for arbitrary resolutions.

There is **no current training profile to size**: choosing an optimizer, adapter
rank, target modules or checkpointing here would invent a trainer. Do not treat
the sampling reference as a training requirement or an invitation to rerun a
retired model. A future revival must add its actual trainable-state and activation
ledger. The generic PG discussion below applies only if such a separately declared
trainer uses that recipe; no optimizer is allocated by this retired repository.


## Optimizer and estimate policy

Training figures freeze pretrained weights, use RAW current adapters and explicitly
disable weight EMA. They include no full-base gradients or EMA shadow model.
The PG 0.9.5 `bispf_optimized` recipe uses zero-momentum DualNorm, not Adam's two
full-size moment tensors. Its direction blend still retains protected gradients,
clones originals/displacements, and converts up to two protected normals to
FP64. For a conservative planning allowance, the tables reserve **80 bytes per
trainable scalar in total**: 8 for FP32 weights+gradients, 16 for two FP64 normals,
and 56 for protected FP32 gradients, FP32/FP64 direction/projection/polar,
rollback and serialization temporaries. This is an explicit allowance, not a claim
that every temporary is simultaneously allocated or that it is an exact peak.
Small scalar optimizer statistics remain enabled. Recipe changes need a new ledger.
See released [PG 0.9.5 recipes](https://github.com/255BITS/ParticleGAN/blob/v0.9.5/particlegan/recipes.py),
[DualNorm](https://github.com/255BITS/ParticleGAN/blob/v0.9.5/particlegan/optim/dualnorm.py)
and [direction blend](https://github.com/255BITS/ParticleGAN/blob/v0.9.5/particlegan/optim/direction_blend.py).

Activation entries below are architecture-based envelopes. Fused/memory-efficient
SDPA is assumed for the pretrained model; a math attention fallback that materializes
all score matrices can exceed them. No model benchmark or runtime measurement was
performed for this documentation. Machine-readable inputs, arithmetic and all table
rows are in [docs/memory/inputs.json](docs/memory/inputs.json).

## Explicit rounding ledger

| Operation | Calculated GPU subtotal (GiB) → planning budget | Host subtotal (GiB) → planning budget |
| --- | --- | --- |
| Current retired scaffold / core check | 0.000 → 0 | 2.000 → 2 |
| Historical SD1.5 sampling planning reference | 7.986 → 8 | 10.060 → 12 |

The detailed host terms and per-device arithmetic are included in the machine-readable inputs.
