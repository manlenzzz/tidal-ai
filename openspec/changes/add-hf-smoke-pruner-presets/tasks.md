## 1. Tests

- [x] 1.1 Add failing tests for pruning-output path canonicalization and filters.
- [x] 1.2 Add a failing offline test that the HF smoke script help path runs without downloads.

## 2. Core Implementation

- [x] 2.1 Implement canonical module-name and tensor-key normalization in `tidal.model_support`.
- [x] 2.2 Implement pruning-output filter construction using canonical names and existing role filters.

## 3. Smoke Script and Docs

- [x] 3.1 Add an optional CPU Hugging Face smoke script with lazy model imports.
- [x] 3.2 Document smoke usage and pruning-output adapter usage in the public README.

## 4. Verification

- [x] 4.1 Run targeted and full test verification.
- [x] 4.2 Run compile and diff hygiene checks.
