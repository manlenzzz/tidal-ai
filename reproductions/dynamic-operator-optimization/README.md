# Dynamic Operator Optimization

Paper: `Dynamic operator optimization for efficient multi-tenant LoRA model serving`

Status: external source integrated; non-GPU smoke checks pending.

## Sources

- Paper: https://ojs.aaai.org/index.php/AAAI/article/view/34453
- PDF: `../../papers/pdf/dynamic-operator-optimization.pdf`
- Text: `../../papers/text/dynamic-operator-optimization.txt`
- Upstream code: https://github.com/harrysyz99/Dop
- Local code path: `../../external/Dop`

## Integration Plan

1. Record the upstream revision and license.
2. Inspect build requirements without compiling CUDA locally.
3. Run CPU-only metadata or Python import checks when available.
4. Move CUDA build, serving benchmarks, and multi-tenant LoRA experiments to Mint Ray or an allowed worker.

## Verification Criteria

- External repo revision is recorded.
- Non-GPU source inspection succeeds.
- CUDA/operator benchmarks are only claimed after cluster execution and result capture.

## Current Evidence

- Local PDF and text are staged.
- External source is integrated as a Git submodule; non-GPU smoke checks are pending.
