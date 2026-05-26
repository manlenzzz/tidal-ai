# Roadmap

TIDAL is intended to become a broad efficient-AI toolkit, not a single-paper reproduction repo. The roadmap keeps the first release grounded while leaving room for inference, serving, post-training, and RL work.

## Near Term

- Stabilize the manifest, source provenance, and method-entry format.
- Add CPU-testable utilities for rank allocation, pruning decisions, precision policies, and allocation export formats.
- Add smoke checks for the three integrated upstream repositories.
- Record minimal reproducible examples for each seed method.

## Mid Term

- Provide reusable APIs for compression search and adapter policy export.
- Add benchmark harnesses for LoRA serving and adapter-heavy workloads.
- Build experiment templates that route GPU work through the workspace Mint Ray policy.
- Publish verified reproduction reports as method evidence accumulates.

## Long Term

- Add efficient inference components, including runtime benchmarks and kernel-oriented integrations.
- Add post-training pipelines for SFT, preference optimization, and evaluation.
- Add efficient RL workflows when the project has stable training and serving primitives.
- Grow the method zoo beyond the initial six papers while keeping provenance and evidence requirements strict.
