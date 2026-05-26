# Contributing

TIDAL contributions should add working code, tests, examples, or maintained integrations.

## Add a Module

1. Add the implementation under `tidal/` with a focused public API.
2. Add a small runnable example under `examples/`.
3. Add CPU tests under `tests/` for the public API.
4. Update the README module table if the API is user-facing.
5. Add external repositories as submodules under `external/` when the upstream license permits it.

## Development Checks

```bash
/opt/venv/bin/python -m tidal.manifest papers/sources.yaml
/opt/venv/bin/python -m pytest
```

Do not commit large model weights, datasets, checkpoints, extracted PDFs, or experiment outputs.
