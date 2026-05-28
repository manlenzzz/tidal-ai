# Design

## Overview
Add QPruner as the second task-first compression workflow. The method-level code in `tidal.methods.qpruner` remains the paper implementation. The workflow layer becomes responsible for loading models/tokenizers, selecting pruned modules, constructing calibration batches, choosing the QPruner execution path, and emitting a stable summary.

## API Shape
`qpruner_compress` mirrors `cap_compress` where possible:

- model inputs: `model` or `model_id`; optional `tokenizer` for in-memory text calibration.
- target inputs: `pruner_targets`, `target_roles`, `name_filter`.
- calibration inputs: `calibration_batches`, `calibration_data`, tokenizer/text field/max samples/max length/batch size.
- QPruner inputs: `importances`, `candidate_bits`, `max_memory_bits`, `max_average_bits`, `objective`, `refine_trials`, `prediction_fn`, `bins`, `seed`, `inplace`.

If `importances` are provided, the workflow builds and applies a quantization plan directly. If importances are absent, calibration batches/data are required and the workflow delegates to `run_qpruner_mixed_precision`. This keeps CPU tests deterministic while preserving the paper-style MI path for real use.

## Reporting
`tidal.reports.summarize_qpruner_run` computes `memory_bits` via `config_memory_bits` and `average_bits` from total selected layer parameters. Summaries should be concise but enough for scripts and README examples. `CompressionResult.method_result` becomes method-agnostic so CAP and QPruner can share the same container.

## CLI
`tidal compress qpruner` gets a sibling parser to CAP. It intentionally focuses on Hugging Face/text calibration execution, because direct in-memory models are a Python API use case. Candidate bitwidths are parsed from comma-separated integers. The CLI forwards cache and trust options consistently with CAP.

## Tests
Tests stay CPU-only and offline:

- Python workflow with a tiny in-memory Torch model and explicit importances.
- Summary saving checks JSON fields and bitwidth decisions.
- CLI help checks that expected user-facing options are present.
- Existing public entrypoint tests guard CAP compatibility.

No GPU or model download is required for this change.
