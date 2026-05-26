# AutoQRA

TIDAL method entry for joint mixed-precision quantization and low-rank adapter optimization.

## Method Paper

`AutoQRA: Joint Optimization of Mixed-Precision Quantization and Low-rank Adapters for Efficient LLM Fine-Tuning`

## TIDAL Status

External source integrated as a Git submodule; non-GPU smoke checks and search/fine-tuning evidence are pending.

## Sources

- Paper: https://arxiv.org/abs/2602.22268
- PDF: `../../papers/pdf/autoqra.pdf`
- Text: `../../papers/text/autoqra.txt`
- Upstream code: https://github.com/harrysyz99/autoqra
- Local code path: `../../external/autoqra`
- Recorded revision: `66a5de5f2372695b2a3a6bed4e4c4b647d47f1ca`

## Integration Plan

1. Inspect CLI, Python API, configs, docs, examples, and tests.
2. Record search-space definitions and adapter export formats.
3. Run CPU-only tests or command-help checks that do not load large models.
4. Route search, fine-tuning, and evaluation jobs through Mint Ray.

## Verification Criteria

- External repo revision and license are recorded.
- Non-GPU package checks pass.
- Search/fine-tuning results are only claimed after recorded GPU execution.

## Current Evidence

- Local PDF and extracted text are staged outside Git.
- External source is integrated as a Git submodule.
