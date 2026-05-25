# Global Rank/Sparsity Optimization

Paper: `Large Language Model Compression with Global Rank and Sparsity Optimization`

Status: plan drafted, paper-derived implementation.

## Sources

- Paper: https://arxiv.org/abs/2505.03801
- OpenReview: https://openreview.net/forum?id=ZaPmQ0NHs4
- PDF: `../../papers/pdf/global-rank-sparsity.pdf`
- Text: `../../papers/text/global-rank-sparsity.txt`

## Implementation Plan

1. Extract the global optimization objective, constraints, and coupling between rank and sparsity.
2. Implement a toy optimizer for layer-level rank/sparsity budgets using synthetic metadata.
3. Add export formats for pruning masks and adapter ranks.
4. Schedule model-level compression and evaluation through Mint Ray after prerequisites are recorded.

## Verification Criteria

- Toy optimizer produces feasible allocations under fixed global budgets.
- Exported rank/sparsity plans are deterministic and inspectable.
- Full reproduction requires recorded model, dataset, commands, artifacts, and cleanup evidence.

## Current Evidence

- Local PDF and text are staged.
- No full reproduction is claimed yet.
