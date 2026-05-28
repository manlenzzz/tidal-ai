# Design

## Overview
Add a workflow layer for RankAdaptor without changing the method implementation. The method package continues to own profile collection, rank search, and PEFT config construction. The workflow coordinates user-facing concerns: model loading, pruner-target filtering, sensitivity loading, PEFT application, and summary writing.

## API Shape
`rankadaptor_adapt` accepts:

- model inputs: `model` or `model_id`; HF cache/revision/local/trust options for model loading.
- target inputs: `pruner_targets`, `target_roles`, `name_filter`.
- RankAdaptor inputs: `sensitivities`, `budget`, `min_rank`, `max_rank`, `rank_step`, optional `performance_model`, `alpha_multiplier`, `seed`.
- PEFT controls: `apply_peft=True`, `inplace=False`, and extra PEFT kwargs.

The default path uses `search_rank_allocation`, which is deterministic and CPU-friendly for tests. A later extension can expose the online incremental evaluator path once we wire real task evaluation scripts.

## Data Flow
1. Resolve the target model.
2. Merge exported pruner targets with any caller name filter.
3. Collect linear profiles with LoRA cost per rank.
4. Run rank allocation under the adapter budget.
5. Build a PEFT `LoraConfig`.
6. Optionally apply PEFT to a copied model.
7. Return an `AdaptationResult` with model/config/search result/summary.

## Reporting
`tidal.reports.summarize_rankadaptor_run` records rank config, adapter cost, score, profile metadata, target roles, and pruner target count. `AdaptationResult.save` uses the same `summary.json` convention as compression workflows.

## CLI
`tidal adapt rankadaptor` focuses on HF model execution and summary generation. Sensitivities are loaded from JSON/JSONL/plain-text mappings so users can pass outputs from pruning or probing scripts without Python glue.

## Tests
Tests stay CPU-only and offline:

- In-memory tiny Torch model produces a PEFT model and summary.
- Pruner target file restricts rank allocation to a selected module.
- CLI help exposes the public options.
- Existing RankAdaptor method tests continue to cover lower-level search details.
