# Global Rank/Sparsity Optimization

TIDAL method entry for global LLM compression with coupled rank and sparsity budgets.

## Method Paper

`Large Language Model Compression with Global Rank and Sparsity Optimization`

## TIDAL Status

Paper-derived implementation plan. No full reproduction is claimed yet.

## Sources

- Paper: https://arxiv.org/abs/2505.03801
- OpenReview: https://openreview.net/forum?id=ZaPmQ0NHs4
- PDF: `../../papers/pdf/global-rank-sparsity.pdf`
- Text: `../../papers/text/global-rank-sparsity.txt`

## Planned Toolkit Components

1. Global optimization objective and constraint parser.
2. Toy optimizer for layer-level rank and sparsity budgets.
3. Export formats for pruning masks and adapter ranks.
4. CPU tests for feasibility, determinism, and allocation accounting.

## Verification Criteria

- Toy optimizer produces feasible allocations under fixed global budgets.
- Exported rank/sparsity plans are deterministic and inspectable.
- Full reproduction requires recorded model, dataset, commands, artifacts, and cleanup evidence.

## Current Evidence

- Local PDF and extracted text are staged outside Git.
- Paper-derived implementation plan is recorded.
