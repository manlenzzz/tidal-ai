## Context

The existing TIDAL Torch integrations traverse every `torch.nn.Linear` and optionally accept a caller-supplied `name_filter`. That is adequate for toy examples but brittle for modern Hugging Face causal LMs, where linear modules include attention projections, MLP projections, MoE experts, routers, output heads, embedding projections, and model-family-specific fused projections.

This change adds a shared model-support layer that classifies linear modules by role and exposes a reusable selector. The selector is then used by RankAdaptor, QPruner, and Global Rank-Sparsity without downloading real model weights or running GPU jobs.

## Goals / Non-Goals

**Goals:**

- Provide a stable role taxonomy for modern Hugging Face-style linear modules.
- Support common naming patterns used by Qwen/Qwen2/Qwen3, Llama-family models, Gemma, Phi, Mistral/Mixtral, DeepSeek-style MoE blocks, and fused QKV modules.
- Exclude risky modules such as `lm_head`, routers, and embedding projections by default.
- Keep existing `name_filter` APIs working.
- Verify behavior with synthetic CPU-only modules that mimic current HF naming structures.

**Non-Goals:**

- Downloading or benchmarking real model weights.
- Adding GPU training/evaluation pipelines.
- Re-implementing upstream model-specific Transformers internals.
- Guaranteeing semantic correctness for arbitrary custom model names beyond the documented fallback behavior.

## Decisions

1. **Use name-pattern classification instead of importing Transformers model classes.**
   Rationale: TIDAL should work on modules after pruning, wrapping, or custom loading. Named-module paths are the stable interface all Torch models expose. Alternative considered: class-based dispatch by `config.model_type`; rejected because wrappers and PEFT/pruning transforms can obscure model classes.

2. **Expose role-based selectors rather than one hard-coded target list.**
   Rationale: RankAdaptor, QPruner, and CAP all need similar but not identical targeting. A selector with include/exclude role sets is explicit and testable. Alternative considered: only provide `default_name_filter`; rejected because users will need safe overrides for router/output-head experiments.

3. **Treat MoE router/gate modules conservatively.**
   Rationale: router layers affect token routing and are not equivalent to expert feed-forward matrices. Default method targets include experts but exclude routers. Callers can opt in by role.

4. **Keep compatibility by composing with existing `name_filter`.**
   Rationale: Existing examples and user code must keep working. New selector arguments will be additive; if a caller passes both role selection and `name_filter`, both predicates must pass.

5. **Use synthetic model tests.**
   Rationale: The target behavior is module discovery and routing, not model quality. Synthetic modules avoid network, GPU, and licensing issues while still covering modern naming patterns.

## Risks / Trade-offs

- **Risk: a new architecture uses unexpected names.** -> Mitigation: unknown linear modules are classified as `other`, and users can include `other` or provide `name_filter`.
- **Risk: a model uses `gate_proj` for both MLP and router concepts.** -> Mitigation: path context controls classification: MLP/expert paths classify as MLP/expert, router paths classify as router.
- **Risk: default exclusions surprise advanced users.** -> Mitigation: expose explicit include/exclude role configuration and document defaults.
- **Risk: selectors become too broad.** -> Mitigation: tests assert exclusion of `lm_head`, embedding projections, and routers by default.

## Migration Plan

- Add the shared support layer and tests.
- Wire methods through additive optional selector parameters while preserving `name_filter`.
- Update README and method docs with usage examples.
- If regressions occur, callers can continue using existing `name_filter` APIs because no existing signature behavior is removed.

## Open Questions

- Real-model smoke tests should be added later using small local or CI-safe model fixtures when the project decides which model families to certify.
