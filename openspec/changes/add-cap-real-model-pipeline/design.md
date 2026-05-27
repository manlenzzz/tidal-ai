## Context

CAP Phase A exposed algorithm primitives. Phase B needs to connect those primitives to model workflows: target layer selection, calibration loss, compression execution, and logging. The repo already has shared model-support selectors and an optional HF smoke example; CAP should reuse those instead of adding method-specific path logic.

## Goals / Non-Goals

**Goals:**
- Provide a clear CAP pipeline API for real models.
- Support pruned/wrapped module names through existing `name_filter` and `target_roles` composition.
- Allow calibration-driven global search using user-provided Torch batches.
- Return structured metadata without changing packed-layer behavior.

**Non-Goals:**
- Downloading models in default tests.
- Implementing lm-eval, Wikitext, or full paper benchmark scripts in this change.
- Adding custom sparse CUDA kernels.

## Decisions

- Keep the pipeline in `torch.py` because it needs model patching and Torch loss evaluation. The NumPy core remains independent.
- Use a callable `loss_fn(model, batch)` for calibration so callers can support causal LM labels, classification, or custom objectives.
- The evaluator will temporarily patch selected linear modules with `CAPPackedLinear` instances, run loss under `torch.no_grad()`, then restore original modules.
- `run_cap_compression` will return a dataclass with the compressed model, selected target metadata, CAP search result, and packed-layer summary.

## Risks / Trade-offs

- Temporary patching can be expensive during search -> acceptable for CPU tests and faithful calibration hooks; future work can cache packed modules.
- Loss functions differ across model families -> make the evaluator caller-supplied and provide a small causal-LM helper only in examples.
- Deepcopy of real LLMs is expensive -> preserve `inplace` support and avoid hidden extra copies beyond existing behavior.
