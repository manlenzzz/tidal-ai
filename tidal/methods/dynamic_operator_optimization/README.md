# Dynamic Operator Optimization

Paper: **Dynamic operator optimization for efficient multi-tenant LoRA model serving**

The original upstream implementation is integrated under `external/Dop`. This package provides a lightweight CPU reference for segmented LoRA SGMV semantics so tests and downstream users have a stable correctness oracle before using the CUDA kernels.

Primary APIs:

- `Segment`
- `sgmv`
- `lora_sgmv`

Example:

- `examples/dynamic_operator_optimization/sgmv_reference.py`

