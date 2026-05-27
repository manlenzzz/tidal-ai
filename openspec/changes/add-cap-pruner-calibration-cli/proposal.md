## Why

CAP now has a Torch pipeline that can compress Hugging Face-style models, but users still need a reproducible entry point for pruned-model continuation experiments. The next step is to connect pruner/WANDA module outputs and calibration text batches so the public toolkit can run method-level smoke experiments against real HF models.

## What Changes

- Add a reusable CAP experiment helper for loading pruner target names from text, JSON, JSONL, or Torch state files.
- Add calibration utilities that turn local text or JSONL records into causal-LM batches with labels.
- Add a CPU-oriented example CLI that runs CAP with optional pruner filters and calibration loss, then writes a structured JSON summary.
- Update documentation and tests for the new experiment path.
- No breaking API changes.

## Capabilities

### New Capabilities

- `cap-pruner-calibration-cli`: CAP real-model experiment entry point with pruner target filtering and calibration batches.

### Modified Capabilities

None.

## Impact

- Affected code: `tidal/methods/global_rank_sparsity/`, `examples/global_rank_sparsity/`, and tests.
- Optional runtime dependencies remain under the existing `torch` extra: PyTorch and Transformers.
- The CLI defaults to CPU-safe smoke settings and uses explicit cache/output paths supplied by the user.
