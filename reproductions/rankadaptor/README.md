# RankAdaptor

Paper: `RankAdaptor: Hierarchical Rank Allocation for Efficient Fine-Tuning Pruned LLMs via Performance Model`

Status: plan drafted, paper-derived implementation.

## Sources

- Paper: https://arxiv.org/abs/2406.15734
- PDF: `../../papers/pdf/rankadaptor.pdf`
- Text: `../../papers/text/rankadaptor.txt`

## Implementation Plan

1. Extract the paper's performance model inputs, pruning assumptions, and hierarchical rank allocation objective.
2. Implement a rank-allocation simulator that accepts layer/module metadata and a target adapter budget.
3. Add LoRA rank assignment export compatible with common PEFT-style configuration.
4. Reproduce the smallest reported setting first, using CPU-only unit tests for allocation logic before any GPU fine-tuning.

## Verification Criteria

- Manifest and paper text are available locally.
- Rank allocation outputs are deterministic for fixed model metadata and budget.
- GPU fine-tuning results are recorded only after Mint Ray execution with cleanup evidence.

## Current Evidence

- Local PDF and text are staged.
- No full reproduction is claimed yet.
