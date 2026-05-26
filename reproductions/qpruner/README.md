# QPruner

TIDAL method entry for structured LLM pruning with probabilistic decision quantization.

## Method Paper

`QPruner: Probabilistic Decision Quantization for Structured Pruning in Large Language Models`

## TIDAL Status

Paper-derived implementation plan. No full reproduction is claimed yet.

## Sources

- Paper: https://arxiv.org/abs/2412.11629
- PDF: `../../papers/pdf/qpruner.pdf`
- Text: `../../papers/text/qpruner.txt`

## Planned Toolkit Components

1. Probabilistic decision quantization utilities.
2. Structured pruning unit definitions for transformer blocks.
3. CPU-testable scoring and mask construction logic.
4. Export path for model-level pruning plans.

## Verification Criteria

- Pruning decision utilities pass deterministic CPU tests.
- Structured sparsity masks match expected shapes for toy transformer metadata.
- GPU evaluation results are recorded with command, artifacts, and cleanup evidence.

## Current Evidence

- Local PDF and extracted text are staged outside Git.
- Paper-derived implementation plan is recorded.
