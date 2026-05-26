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

- Rank allocation for LoRA fine-tuning on pruned LLMs.
- Mixed-precision quantization utilities for structured pruning pipelines.
- Global rank and sparsity allocation for low-rank plus sparse compression.
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

Run a CPU example:

```bash
/opt/venv/bin/python examples/rankadaptor_allocate.py
```

## Method Modules

| Module | Description |
| --- | --- |
| `tidal.rankadaptor` | Hierarchical LoRA rank allocation under a budget. |
| `tidal.qpruner` | Mutual-information driven mixed-precision bit allocation and quantization helpers. |
| `tidal.cap` | RPCA decomposition and global rank/sparse budget allocation. |
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
