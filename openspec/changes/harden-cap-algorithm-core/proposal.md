## Why

CAP is the most algorithmically complex method in TIDAL and should be represented as a faithful two-stage compression core rather than a monolithic reference routine. The next step is to expose stable, testable primitives for decomposition, candidate construction, budgeted selection, and policy search before adding heavier real-model experiment pipelines.

## What Changes

- Split CAP algorithm concerns into public primitives for Stage 1 decomposition/candidate generation and Stage 2 budgeted allocation.
- Add explicit candidate metadata for low-rank singular directions and sparse residual entries, including layer name, local index, value, cost, and role.
- Add deterministic global selection under a hard parameter budget as the reproducible export path after probabilistic search.
- Keep existing `compress_global_rank_sparsity`, `optimize_global_rank_sparsity`, `optimize_global_rank_sparsity_for_matrices`, and Torch APIs compatible.
- Add focused tests that validate decomposition reconstruction, candidate costs, global budget constraints, and deterministic behavior.

## Capabilities

### New Capabilities
- `cap-algorithm-core`: Stable CAP Stage 1/Stage 2 primitives and faithful budgeted selection semantics.

### Modified Capabilities

## Impact

- Affected code: `tidal/methods/global_rank_sparsity/core.py`, package exports, CAP tests, and CAP method documentation.
- No GPU work is required for this phase. Real HF/pruned-model pipeline support remains a follow-up phase.
