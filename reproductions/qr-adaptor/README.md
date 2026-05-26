# QR-Adaptor

TIDAL method entry for mixed-precision fine-tuning aligned with linguistic hierarchies.

## Method Paper

`Balancing Fidelity and Plasticity: Aligning Mixed-Precision Fine-Tuning with Linguistic Hierarchies`

## TIDAL Status

External source integrated as a Git submodule; non-GPU smoke checks and fine-tuning evidence are pending.

## Sources

- Paper: https://arxiv.org/abs/2505.03802
- PDF: `../../papers/pdf/qr-adaptor.pdf`
- Text: `../../papers/text/qr-adaptor.txt`
- Upstream code: https://github.com/harrysyz99/qr_adapter
- Local code path: `../../external/qr_adapter`
- Recorded revision: `e1b89fdfa3040d7cb6a6ad300ad1c63e7f0f197f`

## Integration Plan

1. Inspect package layout, requirements, configs, and examples.
2. Record precision-policy assets and sensitivity metadata if present.
3. Run CPU-only validation that does not load large models.
4. Route mixed-precision fine-tuning experiments through Mint Ray.

## Verification Criteria

- External repo revision and license are recorded.
- Config and sensitivity assets are discoverable.
- Fine-tuning results are only claimed after GPU execution evidence exists.

## Current Evidence

- Local PDF and extracted text are staged outside Git.
- External source is integrated as a Git submodule.
