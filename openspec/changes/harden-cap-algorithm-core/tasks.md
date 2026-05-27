## 1. Tests

- [x] 1.1 Add failing tests for public CAP candidate-pool metadata and candidate costs.
- [x] 1.2 Add failing tests for deterministic budgeted global selection and tie handling.
- [x] 1.3 Add failing compatibility tests that existing CAP APIs expose selected metadata while staying budget-valid.

## 2. Algorithm Core

- [x] 2.1 Add public candidate and candidate-pool dataclasses.
- [x] 2.2 Implement Stage 1 candidate-pool construction from RPCA results.
- [x] 2.3 Implement deterministic budgeted candidate selection.
- [x] 2.4 Route existing single-matrix and global CAP APIs through the explicit core without breaking callers.

## 3. Documentation

- [x] 3.1 Update CAP README with the Stage 1/Stage 2 public API shape.

## 4. Verification

- [x] 4.1 Run targeted CAP tests and full unit tests.
- [x] 4.2 Run compile and diff hygiene checks.
