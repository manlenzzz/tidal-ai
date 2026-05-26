# QPruner

Paper: **Qpruner: Probabilistic decision quantization for structured pruning in large language models**

This package contains the local TIDAL implementation of the QPruner mixed-precision path:

1. run representative calibration data through the pruned model;
2. record each target layer output `X` and the model prediction `Y`;
3. compute mutual information `I(X; Y)` for each layer;
4. initialize a memory-constrained mixed-precision bitwidth plan;
5. optionally refine the plan with Gaussian-process Bayesian optimization;
6. rewrite selected `torch.nn.Linear` modules with a small symmetric n-bit backend.

Primary APIs:

- `layer_mutual_information`
- `allocate_bitwidths`
- `bayesian_refine_bitwidths`
- `collect_linear_mutual_information`
- `build_quantization_plan`
- `apply_mixed_precision_quantization`
- `run_qpruner_mixed_precision`

Modern Hugging Face-style models can be targeted without hand-written layer filters:

```python
run = run_qpruner_mixed_precision(
    pruned_model,
    calibration_batches,
    target_roles="modern",
    max_average_bits=4.0,
)
```

The `modern` selector includes attention, MLP, and MoE expert linear modules while excluding `lm_head`, embedding projections, and MoE routers by default. Existing `name_filter` callables remain supported and are composed with the role selector when both are provided.

Examples:

- `examples/qpruner/mi_bo_quantization.py`
- `examples/qpruner/torch_quantize.py`

