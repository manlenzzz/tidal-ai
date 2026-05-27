## Context

`tidal/model_support.py` already classifies modern HF linear module names by role and excludes unsafe targets such as heads and routers by default. Real pipelines introduce extra wrappers and tensor suffixes: PEFT may expose `base_model.model.*`, torch compile may expose `_orig_mod.*`, and pruning outputs often use parameter keys such as `*.weight`, `*.weight_mask`, or quantized tensor suffixes.

## Goals / Non-Goals

**Goals:**
- Normalize model, parameter, mask, and checkpoint keys into comparable module paths.
- Keep existing role-based target selection as the final safety layer.
- Provide an optional real-model smoke script that is useful to contributors but not part of default offline tests.

**Non-Goals:**
- Run full LLM-Pruner/WANDA experiments in CI.
- Download HF weights in the default unit-test suite.
- Claim paper reproduction metrics from the smoke path.

## Decisions

- Add canonicalization helpers in `tidal.model_support` instead of each method implementing path cleanup independently. This keeps RankAdaptor, QPruner, CAP, and future methods aligned.
- Treat LLM-Pruner/WANDA compatibility as name adaptation plus role filtering. Their outputs generally preserve HF module structure, so a predicate built from exported names is enough for method-level selection.
- Keep the smoke script under `examples/model_support/` with lazy imports. `--help` must work in minimal environments and must not trigger downloads.

## Risks / Trade-offs

- New pruning tools may emit suffixes not covered initially -> expose the suffix list in one module and add tests as new formats appear.
- Canonicalizing too aggressively could collapse distinct names -> strip only common wrapper prefixes and tensor suffixes, while preserving the model's structural path.
- A tiny HF smoke model may move or require network access -> make `--model-id` configurable and document cache usage rather than relying on CI downloads.
