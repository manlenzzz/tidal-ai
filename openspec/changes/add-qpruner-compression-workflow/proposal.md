# Add QPruner Compression Workflow

## Why
TIDAL already has a method-level QPruner implementation, but users should not need to wire model loading, pruner target filtering, calibration data, reporting, and CLI execution by hand. QPruner should be available as a task-first compression workflow alongside CAP so the toolkit reads as a usable efficient-model system rather than a collection of paper folders.

## What Changes
- Add a public `qpruner_compress` workflow under `tidal.workflows.compression`.
- Reuse existing target loading, calibration data, and report helpers.
- Support two entry paths: direct importance scores for deterministic CPU usage and calibration batches/data for MI collection.
- Add `tidal compress qpruner` CLI options for Hugging Face models, pruner targets, calibration text, bit candidates, memory budgets, and output summary writing.
- Update README examples so QPruner is shown as a normal public workflow, not only a method-level API.

## Capabilities
- qpruner-compression-workflow

## Impact
- Public API expands from CAP-only compression to CAP plus QPruner.
- `CompressionResult` must become method-agnostic enough to hold QPruner results.
- Tests should stay CPU-only and avoid network downloads.
