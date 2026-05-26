## 1. Test Coverage

- [x] 1.1 Add synthetic Hugging Face-style model fixtures for Qwen/Llama-style blocks, fused QKV blocks, and MoE blocks.
- [x] 1.2 Add failing tests for linear module role classification and safe default target selection.
- [x] 1.3 Add failing integration tests for RankAdaptor, QPruner, and Global Rank-Sparsity using the shared selector.

## 2. Shared Model Support

- [x] 2.1 Add `tidal.model_support` with linear module metadata, role classification, selector presets, and predicate composition.
- [x] 2.2 Export stable public APIs for listing target modules and building name filters.

## 3. Method Integration

- [x] 3.1 Wire RankAdaptor `collect_linear_profiles` to the shared selector while preserving `name_filter` behavior.
- [x] 3.2 Wire QPruner Torch helpers to the shared selector while preserving `name_filter` behavior.
- [x] 3.3 Wire Global Rank-Sparsity Torch helpers to the shared selector while preserving `name_filter` behavior.

## 4. Documentation and Verification

- [x] 4.1 Document modern model targeting in README and relevant method docs.
- [x] 4.2 Run targeted and full verification, then commit and push the change.
