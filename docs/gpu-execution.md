# GPU Execution Policy

TIDAL separates CPU-testable toolkit logic from GPU-required training, inference, serving, and evaluation. All GPU-required reproduction work must follow the root workspace policy.

## Local Commands

Local execution is allowed for:

- manifest validation
- source inspection
- text extraction
- CPU-only smoke checks
- documentation updates

## Mint Ray Commands

Use Mint Ray or an allowed worker for:

- CUDA builds or CUDA kernel tests
- model training or fine-tuning
- large-model inference or generation
- batch scoring or evaluation
- Ray jobs requesting `num_gpus > 0`

Use generic public metadata only. Do not include paper titles, dataset names, unpublished method names, claims, or customer names in Ray job names, namespaces, task names, temp prefixes, or remote output directories.

## Cleanup Evidence

Every Mint Ray run must record:

- namespace
- launch command
- finish state
- cleanup command
- actor audit result

After a run finishes, verify no owned detached actors remain alive:

```bash
ray list actors --filter "ray_namespace=<ns>" --filter "state=ALIVE"
```

If owned zombie actors remain, clean them through the Ray Jobs API on the cluster head, scoped only to owned `tinker_<user>_*` namespaces.
