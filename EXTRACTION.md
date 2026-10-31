# Standalone extraction contract

Source: HyperGAN/particle-sliders at a4de751c82d8123203953117eec80e952291aea7,
product root conceptmod/sd-shared-game/. Proposed destination:
HyperGAN/sd-particle-sliders. This is a clean source export, not a history graft,
mirror, repository transfer or retry of the held umbrella publication.

PROVENANCE.json is the explicit source allowlist and records each original
repository/commit/path/blob and its exported root path and SHA256. All runtime
Python files, tests and native_sources.json are byte-identical. Package/import
names, version, console/backend entrypoints, cohort, bank, model logic, generic
particle_sliders.fibo_hndl paths and the exact provisional core7a8 dependency
are unchanged. No shared APIs, objective, guard or HNDL optimization is changed.

The umbrella MIT LICENSE and its full Rohit Gandikota copyright notice are
preserved verbatim at LICENSE and declared in wheel/sdist metadata. Root README,
qualification commands, artifact exclusions and manifest are adapted to the new
repository layout. These documentation/package-metadata bytes and the new clean
Git history necessarily differ; model-code and artifact-source identity do not.
Old artifacts still need their exact code/dependency/runtime/native identities;
separation does not authorize relabeling or migration. CPU software checks do
not qualify native model quality, GPU use or the provisional shared core.

No historical model-specific files outside the product root are copied. In
particular conceptmod/textsliders/train_lora.py remains in the umbrella. The
modern package has no import of that trainer or local conceptmod utilities.
It uses installed Diffusers, Torch and particle-sliders-core only; shared
HNDL graphs/manifests come from that exact installed core dependency.

The export excludes private evidence, absolute environment paths, credentials,
logs, recordings, weights, run outputs, environments, caches and unrelated model
families. Source qualification prose is retained; execution logs are not copied.

The clean main branch contains only a minimal README scaffold. Implementation
belongs on extract/sd-a4de751 and a source-only draft PR for Claude review.
No GitHub write, release, deployment, workflow/permission change, merge or GPU
work is implied by the local preparation itself. Root owns destination approval,
actual-source review, final qualification and merge decisions. Originals remain
intact until a verified replacement is accepted.
