## Why

TIDAL's model selectors need to work on real Hugging Face module names and on names exported by common pruning pipelines, not only hand-built toy modules. This change adds an offline-tested compatibility layer and an opt-in smoke script for downloading a tiny HF model when network access is available.

## What Changes

- Add canonical module-name normalization for wrapper prefixes commonly introduced by PEFT, torch compile, and pruning pipelines.
- Add tensor-key to module-name normalization for WANDA-style and checkpoint-style parameter keys.
- Add pruning-output name filters that can be built from LLM-Pruner/WANDA module lists, mask dictionaries, or state dictionaries.
- Add an optional HF smoke-test example that loads a tiny causal LM on CPU and reports selected TIDAL target modules.
- Document how to run the smoke test without making network downloads part of the default unit-test suite.

## Capabilities

### New Capabilities
- `real-model-adapters`: Normalize and filter module names from Hugging Face, LLM-Pruner, and WANDA outputs.
- `hf-smoke-test`: Provide an optional CPU smoke script for a tiny Hugging Face model.

### Modified Capabilities

## Impact

- Affected code: `tidal/model_support.py`, tests, docs, and examples.
- Public API additions are additive; existing `target_roles="modern"` behavior remains supported.
- No new mandatory runtime dependency is introduced beyond existing optional HF/torch usage for the smoke script.
