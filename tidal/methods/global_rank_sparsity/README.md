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
- `optimize_global_rank_sparsity_for_matrices`
- `compress_global_rank_sparsity`
- `apply_global_cap_compression`
- `CAPPackedLinear`

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

