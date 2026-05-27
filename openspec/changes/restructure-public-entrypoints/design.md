## Architecture

The repository keeps `tidal.methods.*` as method-owned implementation packages. A new user-facing layer is added above them:

- `tidal.targets`: model target selection and pruner/WANDA/LLM-Pruner name loading.
- `tidal.data`: calibration text loading and causal-LM batch construction.
- `tidal.reports`: JSON-friendly summary/result helpers.
- `tidal.workflows`: task-first APIs that compose methods, targets, data, and reports.
- `tidal.cli`: formal command line interface.

This separates reusable infrastructure from paper-specific method code without moving existing method APIs.

## First Workflow

The first workflow is CAP compression: `tidal.workflows.compression.cap_compress`. It accepts either an in-memory model or a Hugging Face model id. For this change, tests focus on the in-memory path so the workflow can be verified without network or model downloads. The HF path is available for real use and mirrors the current example script.

The returned `CompressionResult` exposes `model`, `summary`, `method_result`, and `save(path)`. The output directory contains `summary.json`, leaving room for future `config.yaml`, `layers.jsonl`, and artifacts.

## CLI

`tidal compress cap` is the formal CLI. It supports HF model loading, pruner targets, calibration data, budget/search settings, and an output path. The existing example script can delegate to this CLI module later, but this change keeps it intact to minimize risk.

## Compatibility

`tidal.methods.global_rank_sparsity.experiments` remains importable and re-exports the shared target/data/report helpers, so current examples and tests continue to work. Existing method and legacy imports are unchanged.

## Testing

Tests cover shared modules, the in-memory CAP workflow, result saving, CLI help/import behavior, package console-script metadata, and preservation of existing method imports.
