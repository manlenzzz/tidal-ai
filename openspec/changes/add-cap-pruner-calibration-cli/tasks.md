## 1. Tests

- [x] 1.1 Add failing tests for pruner target name loading from text, JSON, JSONL, and Torch state files.
- [x] 1.2 Add failing tests for calibration text loading, batch construction, and causal-LM loss.
- [x] 1.3 Add failing tests for CAP run summary and CLI help surface.

## 2. Experiment Helpers

- [x] 2.1 Implement pruner target loading helpers.
- [x] 2.2 Implement text calibration loading and batching helpers.
- [x] 2.3 Implement causal-LM loss and CAP summary helpers.

## 3. CLI And Docs

- [x] 3.1 Add `examples/global_rank_sparsity/hf_cap_experiment.py`.
- [x] 3.2 Update CAP README with pruner/calibration experiment usage.

## 4. Verification

- [x] 4.1 Run targeted tests for CAP experiment helpers and CLI.
- [x] 4.2 Run full unit tests, compile, and diff hygiene checks.
