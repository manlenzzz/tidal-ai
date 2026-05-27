## ADDED Requirements

### Requirement: Stage 1 Candidate Pool
The system SHALL expose CAP Stage 1 as a candidate-pool primitive that decomposes each matrix with RPCA and emits low-rank and sparse candidates with explicit costs.

#### Scenario: Low-rank candidates have matrix-factor costs
- **WHEN** a matrix with shape `(m, n)` produces a retained singular direction
- **THEN** that candidate cost is `m + n` parameters

#### Scenario: Sparse candidates have scalar costs
- **WHEN** a nonzero sparse residual entry is emitted as a candidate
- **THEN** that candidate cost is `1` parameter

### Requirement: Budgeted Selection
The system SHALL provide deterministic candidate selection that respects a hard total parameter budget.

#### Scenario: Selection respects total budget
- **WHEN** candidates are selected under budget `K`
- **THEN** the sum of selected candidate costs is less than or equal to `K`

#### Scenario: Deterministic tie handling
- **WHEN** candidate scores tie
- **THEN** selection order is deterministic by layer name, candidate kind, and local index

### Requirement: CAP Search Compatibility
The system SHALL preserve existing single-matrix, multi-matrix, and Torch compression APIs while routing them through the explicit CAP core.

#### Scenario: Existing APIs return budget-valid compressions
- **WHEN** existing CAP optimization APIs are called
- **THEN** returned compressions satisfy their requested parameter budgets and existing result fields remain available

#### Scenario: Global search exposes selected candidate metadata
- **WHEN** global CAP search returns
- **THEN** the result includes selected candidate metadata suitable for logging and future calibration pipeline integration
