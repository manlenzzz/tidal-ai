## 1. Tests

- [x] 1.1 Add failing tests for `tidal.targets`, `tidal.data`, and `tidal.reports` public helpers.
- [x] 1.2 Add failing tests for `tidal.workflows.compression.cap_compress` and result saving.
- [x] 1.3 Add failing tests for `tidal compress cap --help` and console script metadata.
- [x] 1.4 Add compatibility tests for existing CAP experiment helper imports.

## 2. Shared Modules

- [x] 2.1 Add `tidal.targets` public helper package.
- [x] 2.2 Add `tidal.data` public helper package.
- [x] 2.3 Add `tidal.reports` public helper package.
- [x] 2.4 Re-export shared helpers from CAP experiment compatibility module.

## 3. Workflow And CLI

- [x] 3.1 Add `tidal.workflows.compression.cap_compress` and `CompressionResult`.
- [x] 3.2 Add `tidal.cli` with `compress cap` command.
- [x] 3.3 Add `tidal` console script to packaging metadata.

## 4. Docs

- [x] 4.1 Update README with task-first quickstart and architecture map.

## 5. Verification

- [x] 5.1 Run targeted public-entrypoint tests.
- [x] 5.2 Run full tests, compile, and diff hygiene checks.
