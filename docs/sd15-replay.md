# SD1.5 adaptive DDIM replay experiment

This research entry evaluates one independently admitted Stable Diffusion 1.5
case with the stock OFF, adaptive A-PSR and same-RAW constant arms. It returns
three finished images, the native DDIM clock, the effective coefficient traces
and the constant control's dose rounding. The core's [Prompt Steering Replacement
implementation][core-psr] computes the adaptive edit at each of the 44 native
UNet boundaries. The [shared interval dose matcher][core-dose] supplies the
constant field on the same spatial lattice and in the same edit direction.

## Runtime and inputs

Install [the pinned experimental requirements](../requirements-sd15-replay.txt)
in a Python 3.11 environment with an appropriate Torch build. The reviewed
runtime used Torch 2.11.0+cu128. Add `scripts/` to the Python import path.

The caller supplies the original frozen stock FP16 UNet and FP32 VAE, the stock
eta-zero epsilon DDIM scheduler, one starting latent and the exact ordered
negative and positive CLIP conditioning. The replay checks the archived
[44-boundary topology and 28-interval clock](../scripts/SD15-APSR-clock.json)
against the actual runtime, including the stock epsilon prediction, disabled
clipping and thresholding, unit initial noise scale and scheduler coefficients.
Keep the backbone, numerical settings, guard and
independent case and checkpoint custody bound in the caller.

Use the independently qualified RAW packet with schema
`sd1p5.exact_all_layer_A_PSR_raw_same_first_progress.v1`. Its complete ordered
`raw_probes` must contain one FP32 `probe.weight`, `probe.bias` and
`direction.weight` per boundary. The caller supplies the verified completed
update count and explicit registered strength. Replay requires
`weights="raw"` and `ema_decay=None`. Checkpoint origin, completed fit status
and case eligibility belong to the independent admission record.

```python
from sd_apsr_replay import render_apsr_comparison

result = render_apsr_comparison(
    unet,
    vae,
    scheduler,
    raw_packet=qualified_raw_packet,
    qualified_updates=verified_update_count,
    alpha=registered_strength,
    initial_noise=initial_noise,
    negative=negative_clip,
    positive=positive_clip,
    guidance=registered_cfg,
    owner=verify_admitted_owner,
)
```

`verify_admitted_owner` returns exactly `True` while the admitted replay
owns the runtime. The function checks ownership before and after each native
prediction and decode, and while resolving the constant fields. It uses one
paired native UNet prediction per interval per arm and decodes each arm once.
Only the conditional activation row is edited. The resolved binary32 field is
held constant in time independently at each spatial position and layer. Exact
matches and residual rounding at native binary32 precision are reported for
each layer.

## Verification

Run the synthetic integration suite with CUDA hidden.

```sh
CUDA_VISIBLE_DEVICES= PYTHONPATH=scripts python -m unittest discover -s tests -v
```

The suite uses the pinned stock DDIM scheduler with synthetic UNet and VAE
fixtures over all 44 boundaries. It checks stock/zero parity, causal edits,
native binary32 control arithmetic, input refusals and owner loss.

For an independently admitted technical case, `verify_apsr_zero_technical`
exercises the two new zero-edit arms against prior stock input, latent and pixel
witnesses. It uses deterministic fixture probes and returns only parity
metadata. On one guarded GPU case from the fit split, both zero-edit arms
matched the archived stock final latent and RGB digests across all 28 DDIM
steps. Verification reused the archived stock witnesses. A fitted nonzero
comparison and independent held-out effect measurement are pending.

[core-psr]: https://github.com/HyperGAN/particle-sliders/blob/c4b3655f43ee9b6e9780bb65766dee0d603cb749/packages/particle-sliders-core/src/particle_sliders/prompt_steering.py
[core-dose]: https://github.com/HyperGAN/particle-sliders/blob/c4b3655f43ee9b6e9780bb65766dee0d603cb749/packages/particle-sliders-core/src/particle_sliders/strength_schedule.py
