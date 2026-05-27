## ADDED Requirements

### Requirement: Optional Hugging Face Smoke Script
The system SHALL provide an opt-in CPU smoke script that loads a user-selected Hugging Face causal LM and reports TIDAL-selected linear modules.

#### Scenario: Help runs without downloading a model
- **WHEN** the smoke script is invoked with `--help`
- **THEN** it exits successfully without importing Hugging Face model code or downloading weights

#### Scenario: Smoke run reports selected modules
- **WHEN** the smoke script successfully loads a model
- **THEN** it reports selected module counts by role and a bounded sample of selected module names

### Requirement: Smoke Script Avoids Default Test Downloads
The system SHALL keep Hugging Face model downloads outside the default unit-test suite.

#### Scenario: Default tests stay offline
- **WHEN** the repository unit tests are run
- **THEN** they validate smoke-script availability without downloading model weights
