# Global Rank And Sparsity

Paper: **Large Language Model Compression with Global Rank and Sparsity Optimization**

This package contains the local TIDAL implementation of the CAP-style two-stage compression workflow:

1. decompose each target weight matrix into low-rank and sparse components with RPCA solved by ADMM;
2. build a candidate pool of retained singular directions and sparse entries;
3. learn Bernoulli retention probabilities with a global budget-aware policy-gradient loop;
4. deterministically select candidates under one total budget;
5. rewrite `torch.nn.Linear` modules as packed low-rank plus sparse layers.

Primary APIs:

- `robust_pca`
- `build_cap_candidate_pool`
- `select_cap_candidates`
- `optimize_global_rank_sparsity_for_matrices`
- `compress_global_rank_sparsity`
- `apply_global_cap_compression`
- `CAPPackedLinear`

Stage 1 and Stage 2 can be inspected directly for debugging and method research:

```python
from tidal.methods.global_rank_sparsity import build_cap_candidate_pool, select_cap_candidates

pool = build_cap_candidate_pool({"model.layers.0.mlp.down_proj": weight}, max_iter=200)
selected = select_cap_candidates(pool.candidates, budget=global_parameter_budget)
```

Each candidate records its layer name, component kind (`rank` or `sparse`), local index, score value, and parameter cost. Low-rank singular directions cost `rows + cols`; sparse residual entries cost `1`. The global optimizer returns `selected_candidates` so allocation decisions can be logged before Torch packing.

The Torch pipeline can collect real model targets, optionally score candidates on calibration batches, and return structured metadata:

```python
from tidal.methods.global_rank_sparsity.torch import run_cap_compression

result = run_cap_compression(
    model,
    total_budget=global_parameter_budget,
    target_roles="modern",
    calibration_batches=calibration_batches,
    loss_fn=lambda model, batch: model(**batch).loss,
)
compressed_model = result.compressed_model
print(result.global_result.parameter_count, len(result.global_result.selected_candidates))
```

`name_filter` can be built from LLM-Pruner/WANDA outputs with `build_pruned_module_name_filter` and composed with `target_roles="modern"`. Calibration is optional; without it, CAP uses reconstruction loss in the NumPy core.

Modern Hugging Face-style models can be targeted without hand-written layer filters:

```python
compressed_model = apply_global_cap_compression(
    model,
    total_budget=global_parameter_budget,
    target_roles="modern",
)
```

The `modern` selector includes attention, MLP, and MoE expert linear modules while excluding `lm_head`, embedding projections, and MoE routers by default. Existing `name_filter` callables remain supported and are composed with the role selector when both are provided.

For pruned-model continuation experiments, use the HF CAP experiment CLI with a WANDA or LLM-Pruner-style target file and optional local calibration text:

```bash
python examples/global_rank_sparsity/hf_cap_experiment.py \
  --model-id hf-internal-testing/tiny-random-LlamaForCausalLM \
  --pruner-targets pruned_targets.txt \
  --calibration-data calibration.txt \
  --total-budget 256 \
  --summary-json output/cap-summary.json
```

`--pruner-targets` accepts plain text, JSON, JSONL, or Torch state files. `--calibration-data` accepts plain text lines or JSONL records and uses causal-LM loss as the CAP evaluator. Omit calibration data to use the reconstruction-loss evaluator.

Examples:

- `examples/global_rank_sparsity/cap_policy_search.py`
- `examples/global_rank_sparsity/torch_compress.py`
- `examples/global_rank_sparsity/hf_cap_smoke.py`
- `examples/global_rank_sparsity/hf_cap_experiment.py`

