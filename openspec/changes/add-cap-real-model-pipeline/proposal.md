## Why

After hardening CAP's algorithm core, the next step is to make CAP usable on real Hugging Face and pruned-model workflows without requiring every experiment script to rebuild target collection, calibration scoring, and compression orchestration. This phase keeps the default tests CPU/offline while establishing the pipeline hooks needed for real LLM-Pruner/WANDA and HF runs.

## What Changes

- Add CAP target collection for Torch/HF modules using the shared modern/pruner-compatible selector.
- Add a calibration evaluator that scores candidate CAP compressions by temporarily patching model weights and measuring loss over user-provided batches.
- Add a run wrapper that compresses selected targets, attaches the CAP global result, and returns structured metadata.
- Add a CPU-only example showing how to run CAP on a tiny HF causal LM when network/cache is available.
- Keep model downloads and GPU execution outside the default unit-test suite.

## Capabilities

### New Capabilities
- `cap-real-model-pipeline`: CAP target collection, calibration scoring, and compression orchestration for real Torch/HF models.

### Modified Capabilities

## Impact

- Affected code: `tidal/methods/global_rank_sparsity/torch.py`, examples, tests, and CAP docs.
- No mandatory new dependency. The optional HF example uses existing `transformers` optional dependency.
- GPU policy remains unchanged; this phase does not run GPU jobs.
