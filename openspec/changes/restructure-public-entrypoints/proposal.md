## Why

The current repository is method-first, which is useful for paper reproduction but hard for external users who want a clear compression or fine-tuning workflow. TIDAL needs stable task-first entrypoints while preserving method packages for research and compatibility.

## What Changes

- Add shared public modules for target selection, calibration data, and run summaries.
- Add a task-first compression workflow API for CAP.
- Add a formal `tidal` CLI entrypoint with `tidal compress cap`.
- Keep existing `tidal.methods.*`, examples, and legacy re-export modules working.
- Update README to show task-first usage before method internals.
- No breaking changes.

## Capabilities

### New Capabilities

- `public-entrypoints`: Task-first Python and CLI entrypoints for model compression workflows.

### Modified Capabilities

None.

## Impact

- Affected code: `tidal/targets`, `tidal/data`, `tidal/reports`, `tidal/workflows`, `tidal/cli`, README, and tests.
- Existing CAP experiment helpers become compatibility wrappers over shared modules.
- Packaging gains a console script entrypoint.
