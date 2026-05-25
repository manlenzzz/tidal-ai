# AutoQRA

Paper: `AutoQRA: Joint Optimization of Mixed-Precision Quantization and Low-rank Adapters for Efficient LLM Fine-Tuning`

Status: external source integrated; non-GPU smoke checks pending.

## Sources

- Paper: https://arxiv.org/abs/2602.22268
- PDF: `../../papers/pdf/autoqra.pdf`
- Text: `../../papers/text/autoqra.txt`
- Upstream code: https://github.com/harrysyz99/autoqra
- Local code path: `../../external/autoqra`

## Integration Plan

1. Record the upstream revision and license.
2. Inspect CLI, Python API, configs, docs, examples, and tests.
3. Run CPU-only tests or command help checks that do not load large models.
4. Route search, fine-tuning, and evaluation jobs through Mint Ray.

## Verification Criteria

- External repo revision is recorded.
- Non-GPU package checks pass.
- Search/fine-tuning results are only claimed after recorded GPU execution.

## Current Evidence

- Local PDF and text are staged.
- External source is integrated as a Git submodule; non-GPU smoke checks are pending.
