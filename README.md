# TIDAL

<p align="center">
  <strong>Efficient LLM Training, Compression, and Serving Toolkit</strong>
</p>

<p align="center">
  <a href="https://github.com/manlenzzz/tidal-ai"><img alt="Project" src="https://img.shields.io/badge/project-TIDAL-0f766e"></a>
  <a href="docs/citation.md"><img alt="Methods" src="https://img.shields.io/badge/methods-6-2563eb"></a>
  <a href="https://manlenzzz.github.io/tidal-ai/"><img alt="Homepage" src="https://img.shields.io/badge/homepage-online-7c3aed"></a>
</p>

TIDAL is an open-source toolkit for efficient large-model systems. It provides reusable implementations and integrations for pruning, quantization, low-rank adaptation, compression, and multi-tenant LoRA serving. The codebase is designed as a research and engineering base that can grow into efficient inference, fine-tuning, serving, post-training, and RL components.

## Features

- RankAdaptor-style performance-model rank search with direct PEFT `LoraConfig` export.
- QPruner-style mutual-information initialization, budgeted mixed-precision search, and PyTorch `nn.Linear` replacement.
- CAP-style robust PCA, global rank/sparsity policy search, and PyTorch low-rank+sparse packed linear layers.
- CPU reference implementation for segmented gather matrix-vector LoRA serving operators.
- Integrated upstream codebases for Dynamic Operator Optimization, QR-Adaptor, and AutoQRA.
- Tests and examples that exercise both algorithm kernels and Torch/PEFT integration paths.

## Installation

```bash
git clone --recurse-submodules https://github.com/manlenzzz/tidal-ai.git
cd tidal-ai
python -m pip install -e .
```

Install the Torch/PEFT integration extra when using the model-rewrite examples:

```bash
python -m pip install -e ".[torch]"
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

Run the Torch/PEFT reproduction entry points:

```bash
/opt/venv/bin/python examples/torch_rankadaptor_peft.py
/opt/venv/bin/python examples/torch_qpruner_quantize.py
/opt/venv/bin/python examples/torch_cap_compress.py
```

## Torch/PEFT APIs

RankAdaptor can profile `torch.nn.Linear` modules and create a PEFT config with per-layer ranks:

```python
from peft import get_peft_model
from tidal.rankadaptor import build_lora_config, collect_linear_profiles, search_rank_allocation

profiles = collect_linear_profiles(model, sensitivities=sensitivity, min_rank=1, max_rank=64)
allocation = search_rank_allocation(profiles, budget=adapter_budget)
peft_model = get_peft_model(model, build_lora_config(allocation.config))
```

QPruner can allocate mixed bitwidths and replace selected linear layers with a small symmetric n-bit backend:

```python
from tidal.qpruner_torch import apply_mixed_precision_quantization, build_quantization_plan

plan = build_quantization_plan(model, layer_importance, candidate_bits=(2, 4, 8), max_average_bits=4.0)
quantized_model = apply_mixed_precision_quantization(model, plan)
```

CAP can replace linear layers with low-rank factors plus sparse residuals under per-layer budgets:

```python
from tidal.cap_torch import apply_cap_compression

compressed_model = apply_cap_compression(model, {"layers.0.mlp.down_proj": 4096})
```

## Method Modules

| Module | Description |
| --- | --- |
| `tidal.rankadaptor` | Log-rank performance surrogate fitting, coordinate rank search, Torch linear profiling, and PEFT config export. |
| `tidal.qpruner` | Mutual-information scoring, feasible bitwidth enumeration, and GP expected-improvement refinement. |
| `tidal.qpruner_torch` | PyTorch model rewrite backend for QPruner mixed-precision decisions. |
| `tidal.cap` | RPCA decomposition, greedy compression, and Bernoulli policy search for global rank/sparse budgets. |
| `tidal.cap_torch` | PyTorch packed linear backend for CAP low-rank+sparse compression. |
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
