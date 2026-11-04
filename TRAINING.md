# One training command

```sh
particle-slider train --config training.example.json
```

The project wrapper remains available: `./train --config training.example.json`.

`./train` forwards to the common `particle-slider train` command with this
project's explicit `training_profile.json`. The frontend can be selected through
`PARTICLE_SLIDER_COMMAND`; it does not install or replace product dependencies.
`./train --config training.example.json --plan` inspects the request without
allocating a GPU. Execution and `--check` return the concrete readiness reason
before loading native model code.

Current readiness: **blocked**. The checked-out default source is a review scaffold and contains no executable Stable Diffusion training factory.

The example contains only the intended update budget and is not a claim that
this product can train. Existing commands, algorithms and dependency locks are
unchanged. A real state-preserving training factory must be integrated before
this profile can become executable; no random fixture or unrelated model is
substituted. Training, when supported, will not automatically search LRs,
render samples or publish artifacts.
