## 1. Tests

- [x] 1.1 Add failing tests for CAP target collection with modern and pruner-composed filters.
- [x] 1.2 Add failing tests for calibration evaluator loss computation and module restoration.
- [x] 1.3 Add failing tests for the CAP pipeline runner returning compressed model metadata.

## 2. Torch Pipeline

- [x] 2.1 Add CAP target and pipeline result dataclasses.
- [x] 2.2 Implement CAP target collection using shared selector composition.
- [x] 2.3 Implement temporary module patching and calibration evaluator.
- [x] 2.4 Implement `run_cap_compression` orchestration on top of global CAP search and packing.

## 3. Examples and Docs

- [x] 3.1 Add optional CPU HF CAP smoke example.
- [x] 3.2 Update CAP README with pipeline usage.

## 4. Verification

- [x] 4.1 Run targeted CAP pipeline tests and full unit tests.
- [x] 4.2 Run compile and diff hygiene checks.
