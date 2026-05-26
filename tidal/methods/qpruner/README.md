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

Examples:

- `examples/qpruner/mi_bo_quantization.py`
- `examples/qpruner/torch_quantize.py`

