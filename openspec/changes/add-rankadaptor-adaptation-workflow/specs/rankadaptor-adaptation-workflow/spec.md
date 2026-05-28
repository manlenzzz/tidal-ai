# RankAdaptor Adaptation Workflow Spec

## ADDED Requirements

### Requirement: Public RankAdaptor workflow
The system SHALL expose a task-first `rankadaptor_adapt` function from `tidal.workflows.adaptation` and `tidal.workflows`.

#### Scenario: Adapt an in-memory Torch model
- **WHEN** a caller passes an in-memory Torch model, per-module sensitivities, rank bounds, and an adapter budget
- **THEN** the workflow collects linear module profiles
- **AND** runs RankAdaptor rank allocation
- **AND** builds a PEFT LoRA config with rank and alpha patterns
- **AND** returns an adaptation result containing the PEFT model, LoRA config, search result, and summary.

#### Scenario: Preserve source model unless requested
- **WHEN** `inplace=False`
- **THEN** the PEFT model is built from a copied model
- **AND** the caller's original model remains without LoRA adapters.

### Requirement: Pruner target integration
The system SHALL support the same pruner target formats used by compression workflows.

#### Scenario: Filter by exported pruner targets
- **WHEN** `pruner_targets` is supplied as a file path or sequence
- **THEN** the workflow builds a canonical module-name filter
- **AND** rank allocation only profiles selected linear modules.

### Requirement: RankAdaptor run summaries
The system SHALL write RankAdaptor summary JSON using the shared report layer.

#### Scenario: Save summary
- **WHEN** a RankAdaptor result is saved
- **THEN** `summary.json` includes method `rankadaptor`, model id, target roles, rank config, profile count, adapter cost, score, search type, and optional pruner target metadata.

### Requirement: RankAdaptor CLI
The system SHALL expose RankAdaptor through the `tidal adapt rankadaptor` CLI.

#### Scenario: Inspect CLI help
- **WHEN** a user runs `tidal adapt rankadaptor --help`
- **THEN** the help includes model id, budget, sensitivities, rank bounds, pruner targets, and output options.
