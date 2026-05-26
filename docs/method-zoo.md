# Method Zoo

The TIDAL method zoo tracks algorithms, code, and reproduction status across efficient AI systems. The initial wave is organized around compression, adaptation, and serving; future waves can add inference kernels, post-training, and RL components without changing the project identity.

## Compression

| Method | Problem | Status | Entry |
| --- | --- | --- | --- |
| QPruner | Structured pruning through probabilistic decision quantization | Paper-derived implementation plan | [`qpruner`](../reproductions/qpruner/README.md) |
| Global Rank/Sparsity Optimization | Jointly allocate global rank and sparsity budgets for LLM compression | Paper-derived implementation plan | [`global-rank-sparsity`](../reproductions/global-rank-sparsity/README.md) |

## Adaptation and Fine-Tuning

| Method | Problem | Status | Entry |
| --- | --- | --- | --- |
| RankAdaptor | Allocate LoRA ranks for fine-tuning pruned LLMs using a performance model | Paper-derived implementation plan | [`rankadaptor`](../reproductions/rankadaptor/README.md) |
| QR-Adaptor | Align mixed-precision fine-tuning with linguistic hierarchies | External source integrated | [`qr-adaptor`](../reproductions/qr-adaptor/README.md) |
| AutoQRA | Jointly optimize mixed-precision quantization and low-rank adapters | External source integrated | [`autoqra`](../reproductions/autoqra/README.md) |

## Serving

| Method | Problem | Status | Entry |
| --- | --- | --- | --- |
| Dynamic Operator Optimization | Optimize operators for efficient multi-tenant LoRA model serving | External source integrated | [`dynamic-operator-optimization`](../reproductions/dynamic-operator-optimization/README.md) |

## Status Labels

| Label | Meaning |
| --- | --- |
| External source integrated | Upstream code is tracked under `external/`; local smoke and GPU evidence are still method-specific. |
| Paper-derived implementation plan | No public code was found or integrated yet; implementation will be derived from the paper and validated incrementally. |
| Reproduced | Reserved for entries with recorded prerequisites, commands, outputs, target metrics, and verification evidence. |
