## Context

The CAP paper describes a two-stage method: RPCA decomposes each weight matrix into low-rank and sparse components, then a global probabilistic pruning stage allocates a parameter budget over retained singular directions and sparse entries. The current code implements those ideas but keeps candidate construction and selection mostly private, making it hard to test or use as a toolkit building block.

## Goals / Non-Goals

**Goals:**
- Make Stage 1 and Stage 2 explicit and testable.
- Represent CAP candidates with enough metadata for debugging, logging, and future HF pipeline integration.
- Enforce hard parameter budgets in deterministic selection and final exported compressions.
- Preserve current public APIs.

**Non-Goals:**
- Running LLaMA/Qwen experiments in this change.
- Implementing sparse CUDA kernels or benchmark claims.
- Adding dataset calibration loaders; evaluator hooks remain accepted but Phase B will own real data pipelines.

## Decisions

- Keep NumPy as the algorithm-core dependency for CPU determinism and simple tests. Torch packing stays in `torch.py`.
- Add public dataclasses rather than exposing raw candidate arrays. Internal vectorized arrays can still be used for search efficiency.
- Use deterministic selection by descending probability or score with cost-aware budget checks for export. Probabilistic search may sample over budget during learning, but the returned final compression must satisfy the budget.
- Track layer names in global candidate metadata so future calibration logs can explain which layers received rank and sparse budget.

## Risks / Trade-offs

- More public primitives increase API surface -> keep them small and method-specific.
- Candidate dataclasses add conversion overhead -> keep search internals array-based and expose metadata only at boundaries.
- RPCA on real LLM matrices is expensive -> this phase improves correctness boundaries, while Phase B will add chunking/caching/pipeline controls.
