# QPruner

Paper: `Qpruner: Probabilistic decision quantization for structured pruning in large language models`

Status: plan drafted, paper-derived implementation.

## Sources

- Paper: https://arxiv.org/abs/2412.11629
- PDF: `../../papers/pdf/qpruner.pdf`
- Text: `../../papers/text/qpruner.txt`

## Implementation Plan

1. Extract the probabilistic decision quantization formulation and structured pruning units from the paper.
2. Implement CPU-testable scoring and quantized pruning-decision utilities.
3. Add model-shape adapters for common transformer blocks without running model pruning locally.
4. Schedule the first pruning/evaluation run through Mint Ray after data/model prerequisites are recorded.

## Verification Criteria

- Pruning decision utilities pass deterministic CPU tests.
- Structured sparsity masks match expected shapes for toy transformer metadata.
- GPU evaluation results are recorded with command, artifacts, and cleanup evidence.

## Current Evidence

- Local PDF and text are staged.
- No full reproduction is claimed yet.
