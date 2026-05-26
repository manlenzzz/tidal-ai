# Contributing

TIDAL contributions should make the toolkit more reusable, reproducible, or easier to evaluate. Keep changes scoped and record enough evidence for another researcher or engineer to understand the status of a method.

## Add a Method

1. Add or update the method entry under `reproductions/<method-id>/README.md`.
2. Add source metadata to `papers/sources.yaml` when the method is paper-backed.
3. Add external code under `external/` as a submodule when upstream code exists and the license permits it.
4. Update [`method-zoo.md`](method-zoo.md) and [`reproduction-matrix.md`](reproduction-matrix.md).
5. Add CPU-only tests for parsing, allocation, search, export, or bookkeeping logic before running GPU jobs.

## Evidence Expectations

A reproduction claim should include:

- exact source revision or paper version
- environment and dependency notes
- command lines or launcher configuration
- datasets, model checkpoints, and artifact locations
- output metrics and comparison targets
- GPU execution and cleanup evidence when applicable

## Local Development

```bash
/opt/venv/bin/python -m tidal.manifest papers/sources.yaml
/opt/venv/bin/python -m pytest tests/test_manifest_validation.py
```

Do not commit large model files, datasets, checkpoints, extracted PDFs, or experiment outputs. Keep large artifacts under the durable workspace paths required by the root `AGENTS.md` policy.
