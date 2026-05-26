## Why

TIDAL methods currently rely on raw `torch.nn.Linear` traversal plus optional manual filters. That is not enough for modern Hugging Face causal LMs whose projection names vary across Qwen, Llama, Gemma, Phi, DeepSeek, Mistral, Mixtral, MoE, and fused-projection families.

A shared model-support layer lets RankAdaptor, QPruner, and Global Rank-Sparsity target the right model submodules by default while avoiding fragile per-model hard-coding.

## What Changes

- Add a shared model-inspection capability that discovers linear modules and assigns stable roles such as attention, MLP, MoE expert, MoE router, output head, embedding projection, and other.
- Add reusable target-selection APIs that include trainable/compressible projection layers and exclude unsafe defaults such as `lm_head` and router/gate projections unless requested.
- Wire RankAdaptor, QPruner, and Global Rank-Sparsity Torch integrations to accept a model family/preset or module-role selector instead of requiring custom `name_filter` lambdas.
- Add synthetic HF-style tests for modern model naming patterns, including Qwen/Llama-style attention-MLP blocks, fused QKV projections, and DeepSeek/Mixtral-style MoE experts and routers.
- Update README/method docs with the supported-model usage pattern.

## Capabilities

### New Capabilities

- `modern-hf-model-support`: Discover, classify, and select linear modules in modern Hugging Face-style large language models for TIDAL methods.

### Modified Capabilities

- None.

## Impact

- Affected code: new shared model-support module, RankAdaptor profile collection, QPruner Torch planning/quantization, Global Rank-Sparsity Torch compression, examples/docs, and tests.
- Public API impact: additive. Existing `name_filter` arguments continue to work.
- Dependencies: no new required runtime dependency beyond existing optional Torch/Transformers/PEFT extras.
- Systems impact: CPU-only unit tests with synthetic models; no GPU jobs and no model-weight downloads in this change.
