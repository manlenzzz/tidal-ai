# cap-pruner-calibration-cli Specification

## Requirements

### Requirement: Pruner target inputs

The toolkit SHALL provide a helper that loads pruner target names from plain text, JSON, JSONL, and Torch state files.

#### Scenario: Load plain text names

- **WHEN** a file contains one module or tensor name per non-empty line
- **THEN** the helper returns those names in file order without comments or blank lines

#### Scenario: Load JSON names

- **WHEN** a JSON file contains a list of strings or a mapping whose keys are tensor names
- **THEN** the helper returns the names needed to build a pruned module filter

#### Scenario: Load Torch state names

- **WHEN** a Torch checkpoint or state dict file is provided
- **THEN** the helper returns state dict keys without materializing model-specific behavior

### Requirement: Text calibration batches

The toolkit SHALL provide CPU calibration utilities for causal language model loss.

#### Scenario: Build labels from text

- **WHEN** calibration texts are tokenized
- **THEN** each batch includes `input_ids`, `attention_mask`, and `labels` tensors on the requested device

#### Scenario: Reject empty calibration

- **WHEN** no usable calibration texts are provided
- **THEN** the helper raises `ValueError`

### Requirement: CAP experiment CLI

The toolkit SHALL provide a Hugging Face CAP experiment CLI that accepts optional pruner targets and optional calibration text.

#### Scenario: Run with pruner targets

- **WHEN** the CLI receives a pruner target file
- **THEN** it composes the pruner-derived filter with the configured target roles before CAP compression

#### Scenario: Emit summary JSON

- **WHEN** the CLI completes a run
- **THEN** it emits JSON containing model id, target count, compressed layer count, parameter count, selected candidate count, and layer summaries
