# public-entrypoints Specification

## Requirements

### Requirement: Shared target and data helpers

TIDAL SHALL expose reusable target loading and calibration helpers outside individual method packages.

#### Scenario: Pruner targets are loaded through public target API

- **WHEN** user code imports `load_pruner_target_names` from `tidal.targets`
- **THEN** it can load the same target formats supported by CAP experiments

#### Scenario: Calibration batches are built through public data API

- **WHEN** user code imports `build_text_calibration_batches` from `tidal.data`
- **THEN** it receives causal-LM batches with `input_ids`, `attention_mask`, and `labels`

### Requirement: Task-first CAP workflow

TIDAL SHALL provide a public compression workflow for CAP.

#### Scenario: Compress an in-memory model

- **WHEN** user code calls `cap_compress(model=module, budget=N)`
- **THEN** it returns an object containing the compressed model, summary dictionary, and underlying CAP method result

#### Scenario: Save workflow summary

- **WHEN** user code calls `result.save(path)`
- **THEN** TIDAL writes `summary.json` to that directory

### Requirement: Formal CLI

TIDAL SHALL define a console script named `tidal`.

#### Scenario: Show CAP compression help

- **WHEN** a user runs `tidal compress cap --help`
- **THEN** the CLI displays options for model id, pruner targets, calibration data, budget, and output path

### Requirement: Backward compatibility

Existing method and experiment imports SHALL continue to work after the public-entrypoint restructuring.
