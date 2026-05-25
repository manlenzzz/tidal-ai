# QR-Adaptor

Paper: `Balancing fidelity and plasticity: Aligning mixed-precision fine-tuning with linguistic hierarchies`

Status: external source integrated; non-GPU smoke checks pending.

## Sources

- Paper: https://arxiv.org/abs/2505.03802
- PDF: `../../papers/pdf/qr-adaptor.pdf`
- Text: `../../papers/text/qr-adaptor.txt`
- Upstream code: https://github.com/harrysyz99/qr_adapter
- Local code path: `../../external/qr_adapter`

## Integration Plan

1. Record the upstream revision and license.
2. Inspect package layout, requirements, configs, and examples.
3. Run CPU-only validation or tests that do not load large models.
4. Route mixed-precision fine-tuning experiments through Mint Ray.

## Verification Criteria

- External repo revision is recorded.
- Config and sensitivity assets are discoverable.
- Fine-tuning results are only claimed after GPU execution evidence exists.

## Current Evidence

- Local PDF and text are staged.
- External source is integrated as a Git submodule; non-GPU smoke checks are pending.
