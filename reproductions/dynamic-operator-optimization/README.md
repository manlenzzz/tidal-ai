# Dynamic Operator Optimization

TIDAL method entry for efficient multi-tenant LoRA model serving.

## Method Paper

`Dynamic Operator Optimization for Efficient Multi-Tenant LoRA Model Serving`

## TIDAL Status

External source integrated as a Git submodule; non-GPU smoke checks and serving benchmarks are pending.

## Sources

- Paper: https://ojs.aaai.org/index.php/AAAI/article/view/34453
- PDF: `../../papers/pdf/dynamic-operator-optimization.pdf`
- Text: `../../papers/text/dynamic-operator-optimization.txt`
- Upstream code: https://github.com/harrysyz99/Dop
- Local code path: `../../external/Dop`
- Recorded revision: `59e50ae0894b1043e275828a157354befd7e67a3`

## Integration Plan

1. Inspect build requirements without compiling CUDA locally.
2. Record available CLIs, configs, examples, and benchmarks.
3. Run CPU-only metadata or import checks when available.
4. Move CUDA builds, serving benchmarks, and multi-tenant LoRA experiments to Mint Ray or an allowed worker.

## Verification Criteria

- External repo revision and license are recorded.
- Non-GPU source inspection succeeds.
- CUDA/operator benchmarks are only claimed after cluster execution and result capture.

## Current Evidence

- Local PDF and extracted text are staged outside Git.
- External source is integrated as a Git submodule.
