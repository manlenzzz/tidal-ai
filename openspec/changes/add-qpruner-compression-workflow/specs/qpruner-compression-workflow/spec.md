# QPruner Compression Workflow Spec

## ADDED Requirements

### Requirement: Public QPruner workflow
The system SHALL expose a task-first `qpruner_compress` function from `tidal.workflows.compression` and `tidal.workflows`.

#### Scenario: Compress in-memory Torch model from importance scores
- **WHEN** a caller passes an in-memory Torch model, per-layer importances, candidate bitwidths, and a memory budget
- **THEN** the workflow builds a mixed-precision QPruner plan
- **AND** rewrites selected linear modules into quantized modules
- **AND** returns a `CompressionResult` containing the quantized model, method result, and summary.

#### Scenario: Preserve source model unless requested
- **WHEN** `inplace=False`
- **THEN** the returned model is a copied quantized model
- **AND** the caller's original model remains unchanged.

### Requirement: Target and calibration integration
The system SHALL reuse shared target and calibration helpers for QPruner compression.

#### Scenario: Filter by exported pruner targets
- **WHEN** `pruner_targets` is supplied as a file path or sequence
- **THEN** the workflow builds a module name filter from those exported target names
- **AND** applies it together with any caller-provided name filter.

#### Scenario: Collect MI from calibration batches
- **WHEN** calibration batches are supplied instead of importances
- **THEN** the workflow collects layer-output/prediction mutual information
- **AND** uses those scores for mixed-precision allocation.

#### Scenario: Load text calibration data for Hugging Face models
- **WHEN** a caller supplies `model_id` and text calibration data
- **THEN** the workflow loads the causal LM and tokenizer with the same cache, revision, local-files, and trust-remote-code controls used by CAP
- **AND** builds CPU calibration batches before running QPruner.

### Requirement: QPruner run summaries
The system SHALL write QPruner summary JSON using the shared report layer.

#### Scenario: Save summary
- **WHEN** a QPruner `CompressionResult` is saved
- **THEN** `summary.json` includes method `qpruner`, model id, target roles, selected bitwidths, layer sizes, importances, total memory bits, average bits, and optional calibration/pruner metadata.

### Requirement: QPruner CLI
The system SHALL expose QPruner through the `tidal compress qpruner` CLI.

#### Scenario: Inspect CLI help
- **WHEN** a user runs `tidal compress qpruner --help`
- **THEN** the help includes model id, calibration data, pruner targets, candidate bitwidths, memory budgets, and output options.

#### Scenario: CLI writes summary
- **WHEN** the CLI is run with valid inputs and `--output`
- **THEN** it executes the QPruner workflow
- **AND** writes the same `summary.json` artifact as the Python API.
