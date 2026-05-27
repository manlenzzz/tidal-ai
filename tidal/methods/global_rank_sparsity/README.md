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

Modern Hugging Face-style models can be targeted without hand-written layer filters:

```python
compressed_model = apply_global_cap_compression(
    model,
    total_budget=global_parameter_budget,
    target_roles="modern",
)
```

The `modern` selector includes attention, MLP, and MoE expert linear modules while excluding `lm_head`, embedding projections, and MoE routers by default. Existing `name_filter` callables remain supported and are composed with the role selector when both are provided.

Examples:

- `examples/global_rank_sparsity/cap_policy_search.py`
- `examples/global_rank_sparsity/torch_compress.py`

