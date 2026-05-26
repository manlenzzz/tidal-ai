# TIDAL

<p align="center">
  <strong>Efficient LLM Training, Compression, and Serving Toolkit</strong>
</p>

<p align="center">
  <a href="https://github.com/manlenzzz/tidal-ai"><img alt="Project" src="https://img.shields.io/badge/project-TIDAL-0f766e"></a>
  <a href="docs/citation.md"><img alt="Methods" src="https://img.shields.io/badge/methods-6-2563eb"></a>
  <a href="https://manlenzzz.github.io/tidal-ai/"><img alt="Homepage" src="https://img.shields.io/badge/homepage-online-7c3aed"></a>
</p>

TIDAL is an open-source toolkit for efficient large-model systems. It provides practical components for pruning, quantization, low-rank adaptation, and multi-tenant LoRA serving, with research methods exposed as reusable Python APIs, examples, tests, and integrations with upstream repositories.

## Features

- RankAdaptor-style performance-model rank search for LoRA fine-tuning on pruned LLMs.
- QPruner-style mutual-information initialization plus budgeted mixed-precision refinement.
- CAP-style robust PCA and global rank/sparsity policy search for compression.
- CPU reference implementation for segmented gather matrix-vector LoRA serving operators.
- Integrated upstream codebases for Dynamic Operator Optimization, QR-Adaptor, and AutoQRA.
- A clean API surface for adding more efficient inference, fine-tuning, serving, post-training, and RL components.

## Installation

```bash
git clone --recurse-submodules https://github.com/manlenzzz/tidal-ai.git
cd tidal-ai
python -m pip install -e .
```

For local development without packaging metadata, run commands from the repository root:

```bash
/opt/venv/bin/python -m pytest
```

## Quickstart

Validate the method/source manifest:

```bash
/opt/venv/bin/python -m tidal.manifest methods/sources.yaml
```

Run CPU examples for the local method implementations:

```bash
/opt/venv/bin/python examples/rankadaptor_allocate.py
/opt/venv/bin/python examples/qpruner_allocate.py
/opt/venv/bin/python examples/cap_optimize.py
```

## Method Modules

| Module | Description |
| --- | --- |
| `tidal.rankadaptor` | Log-rank performance surrogate fitting, coordinate rank search, and PEFT rank-pattern export. |
| `tidal.qpruner` | Mutual-information scoring, feasible bitwidth enumeration, and GP expected-improvement refinement. |
| `tidal.cap` | RPCA decomposition, greedy compression, and Bernoulli policy search for global rank/sparse budgets. |
| `tidal.sgmv` | CPU reference implementation of segmented LoRA SGMV shrink/expand operators. |

## Integrated Repositories

| Method | Upstream | Local path |
| --- | --- | --- |
| Dynamic Operator Optimization | `harrysyz99/Dop` | `external/Dop` |
| QR-Adaptor | `harrysyz99/qr_adapter` | `external/qr_adapter` |
| AutoQRA | `harrysyz99/autoqra` | `external/autoqra` |

## Team

TIDAL is developed by our team: Changhai Zhou, Yuhua Zhou, and Shiyang Zhang. The repository is structured to host team-maintained efficient-LLM components behind consistent APIs and examples.

## Citation

See [`docs/citation.md`](docs/citation.md) for toolkit and method citation guidance.
