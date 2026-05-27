## ADDED Requirements

### Requirement: CAP Target Collection
The system SHALL collect CAP target linear modules from Torch/HF models using shared name and role filters.

#### Scenario: Modern target roles exclude unsafe modules
- **WHEN** target roles are `modern`
- **THEN** attention, MLP, and MoE expert linears are collected while `lm_head`, routers, and embedding projections are excluded

#### Scenario: Pruner name filters compose with role filters
- **WHEN** a pruner/WANDA name filter is supplied with target roles
- **THEN** only modules accepted by both filters are collected

### Requirement: Calibration Evaluator
The system SHALL build a CAP evaluator that scores candidate compressions by temporarily patching model layers and computing average calibration loss.

#### Scenario: Calibration evaluator restores original modules
- **WHEN** the evaluator finishes scoring compressions
- **THEN** the model's original modules are restored

#### Scenario: Calibration evaluator computes finite average loss
- **WHEN** calibration batches and a loss function are supplied
- **THEN** the evaluator returns a finite float average across batches

### Requirement: CAP Pipeline Runner
The system SHALL provide a runner that executes global CAP compression and returns structured metadata.

#### Scenario: Runner returns compressed model and metadata
- **WHEN** CAP pipeline compression succeeds
- **THEN** the result contains the compressed model, target summaries, CAP global result, and packed layer summaries

#### Scenario: Runner supports optional calibration
- **WHEN** calibration batches and a loss function are supplied
- **THEN** global CAP search uses calibration loss instead of reconstruction-only loss
