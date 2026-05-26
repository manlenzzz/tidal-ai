# RankAdaptor

TIDAL method entry for hierarchical LoRA rank allocation when fine-tuning pruned LLMs.

## Method Paper

`RankAdaptor: Hierarchical Rank Allocation for Efficient Fine-Tuning Pruned LLMs via Performance Model`

## TIDAL Status

Paper-derived implementation plan. No full reproduction is claimed yet.

## Sources

- Paper: https://arxiv.org/abs/2406.15734
- PDF: `../../papers/pdf/rankadaptor.pdf`
- Text: `../../papers/text/rankadaptor.txt`

## Planned Toolkit Components

1. Performance-model input parser for layer and module metadata.
2. Hierarchical rank allocation solver for adapter budgets.
3. PEFT-style export for LoRA rank assignments.
4. CPU unit tests for deterministic allocation behavior.

## Verification Criteria

- Manifest and paper text are available locally.
- Rank allocation outputs are deterministic for fixed metadata and budget.
- GPU fine-tuning results are recorded only after Mint Ray execution with cleanup evidence.

## Current Evidence

- Local PDF and extracted text are staged outside Git.
- Paper-derived implementation plan is recorded.
