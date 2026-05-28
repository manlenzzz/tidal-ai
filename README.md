# TIDAL

<p align="center">
  <strong>Efficient large-model training, compression, and serving toolkit</strong>
</p>

<p align="center">
  <a href="https://github.com/manlenzzz/tidal-ai"><img alt="Project" src="https://img.shields.io/badge/project-TIDAL-0f766e"></a>
  <a href="https://manlenzzz.github.io/tidal-ai/"><img alt="Homepage" src="https://img.shields.io/badge/homepage-online-2563eb"></a>
  <a href="docs/citation.md"><img alt="Methods" src="https://img.shields.io/badge/methods-6-b45309"></a>
</p>

TIDAL is a task-first toolkit for efficient large-model systems, with method-first internals for paper reproduction. Users get stable workflows for compression, adaptation, and serving; researchers can still inspect each method implementation directly.

The codebase is maintained by our team: Changhai Zhou, Yuhua Zhou, and Shiyang Zhang.

## Public Entry Points

Use `tidal.workflows` or the `tidal` CLI for normal experiments. Method packages remain available for lower-level research code.

```python
from tidal.workflows.compression import cap_compress, qpruner_compress

cap_run = cap_compress(
    model_id="hf-internal-testing/tiny-random-LlamaForCausalLM",
    pruner_targets="pruned_targets.txt",
    calibration_data="calibration.txt",
    budget=256,
    local_files_only=True,
)
cap_run.save("runs/cap-smoke")

qpruner_run = qpruner_compress(
    model_id="hf-internal-testing/tiny-random-LlamaForCausalLM",
    pruner_targets="pruned_targets.txt",
    calibration_data="calibration.txt",
    candidate_bits=(2, 4, 8),
    max_average_bits=4.0,
    local_files_only=True,
)
qpruner_run.save("runs/qpruner-smoke")
```

```bash
tidal compress cap \
  --model-id hf-internal-testing/tiny-random-LlamaForCausalLM \
  --pruner-targets pruned_targets.txt \
  --calibration-data calibration.txt \
  --budget 256 \
  --output runs/cap-smoke

tidal compress qpruner \
  --model-id hf-internal-testing/tiny-random-LlamaForCausalLM \
  --pruner-targets pruned_targets.txt \
  --calibration-data calibration.txt \
  --candidate-bits 2,4,8 \
  --max-average-bits 4.0 \
  --output runs/qpruner-smoke
```

Shared infrastructure is organized by user need:

| Layer | Package | Purpose |
| --- | --- | --- |
| Workflows | `tidal.workflows` | Task-first APIs such as CAP and QPruner compression |
| CLI | `tidal.cli` | Command-line workflows such as `tidal compress cap` and `tidal compress qpruner` |
| Targets | `tidal.targets` | HF module roles, pruner/WANDA/LLM-Pruner target loading |
| Data | `tidal.data` | Calibration text loading and causal-LM batches |
| Reports | `tidal.reports` | Summary JSON and run artifact helpers |
| Methods | `tidal.methods.*` | Paper-level algorithm implementations |

## Method Layout

Each method owns its implementation, Torch integration, examples, and method notes:

| Method | Package | Examples | Status |
| --- | --- | --- | --- |
| RankAdaptor | `tidal.methods.rankadaptor` | `examples/rankadaptor/` | Local implementation |
| QPruner | `tidal.methods.qpruner` | `examples/qpruner/` | Local implementation |
| Global Rank and Sparsity / CAP | `tidal.methods.global_rank_sparsity` | `examples/global_rank_sparsity/` | Local implementation |
| Dynamic Operator Optimization | `tidal.methods.dynamic_operator_optimization` | `external/Dop` | Upstream integration plus CPU reference |
| QR-Adaptor | `tidal.methods.qr_adaptor` | `external/qr_adapter` | Upstream integration |
| AutoQRA | `tidal.methods.autoqra` | `external/autoqra` | Upstream integration |

Compatibility imports such as `tidal.rankadaptor`, `tidal.qpruner_torch`, and `tidal.cap_torch` are kept as thin re-export layers. New workflow code should import from `tidal.workflows`; new method research code should import from `tidal.methods.*`.

## Installation

```bash
git clone --recurse-submodules https://github.com/manlenzzz/tidal-ai.git
cd tidal-ai
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[torch]"
```

For algorithm-only usage without Torch/PEFT integrations:

```bash
python -m pip install -e .
```

## Quickstart

Validate the method/source manifest:

```bash
python -m tidal.manifest methods/sources.yaml
```

Run method-scoped examples:

```bash
python examples/rankadaptor/basic_rank_search.py
python examples/rankadaptor/torch_peft_search.py
python examples/qpruner/mi_bo_quantization.py
python examples/qpruner/torch_quantize.py
python examples/global_rank_sparsity/cap_policy_search.py
python examples/global_rank_sparsity/torch_compress.py
python examples/dynamic_operator_optimization/sgmv_reference.py
```

## Modern Model Targeting

TIDAL includes a shared Hugging Face-style model selector for recent causal LMs with Qwen/Llama-style attention blocks, fused QKV projections, Gemma/Phi/Mistral-style MLPs, and DeepSeek/Mixtral-style MoE experts. Use `target_roles="modern"` to include attention, MLP, and MoE expert linear modules while excluding `lm_head`, embedding projections, and MoE routers by default.

```python
from tidal.model_support import ModuleRole, list_linear_modules
from tidal.methods.rankadaptor import collect_linear_profiles
from tidal.methods.qpruner.torch import collect_linear_layer_sizes
from tidal.methods.global_rank_sparsity.torch import apply_global_cap_compression

targets = list_linear_modules(model, target_roles="modern")
profiles = collect_linear_profiles(model, target_roles="modern")
layer_sizes = collect_linear_layer_sizes(model, target_roles="modern")
compressed = apply_global_cap_compression(model, total_budget=budget, target_roles="modern")
```

Advanced users can pass explicit roles such as `{ModuleRole.ATTENTION, ModuleRole.MLP}` or keep using existing `name_filter` callables. When both are supplied, both predicates must match.

For LLM-Pruner/WANDA-style outputs, build a selector from exported module names, mask dictionaries, or state-dict tensor keys. TIDAL canonicalizes common wrapper prefixes such as `base_model.model.*`, `_orig_mod.*`, and parameter suffixes such as `.weight`, `.weight_mask`, `.qweight`, and `.scales` before applying the same role safety filter.

```python
from tidal.model_support import build_pruned_module_name_filter, list_linear_modules

name_filter = build_pruned_module_name_filter(wanda_masks_or_state_dict, target_roles="wanda")
targets = list_linear_modules(pruned_model, name_filter=name_filter)
```

To smoke-test the selector against a real tiny Hugging Face model on CPU, run the optional example. It is intentionally outside the default test suite because it may download model files.

```bash
TIDAL_HF_CACHE=/vePFS-Mindverse/user/intern/zhouch/.hf_cache \
  python examples/model_support/hf_smoke.py \
  --model-id hf-internal-testing/tiny-random-LlamaForCausalLM
```

## RankAdaptor

RankAdaptor searches hierarchical LoRA ranks for recovering a pruned model. The local implementation follows the paper workflow: candidate rank configurations, five-layer MLP performance model, online incremental task evaluation with prediction-error convergence, and PEFT export.

```python
from peft import get_peft_model
from tidal.methods.rankadaptor import (
    build_lora_config,
    collect_linear_profiles,
    online_incremental_rank_search,
)

profiles = collect_linear_profiles(model, sensitivities=sensitivity, min_rank=1, max_rank=64)
search = online_incremental_rank_search(
    profiles,
    budget=adapter_budget,
    evaluate_config=finetune_and_eval,
)
peft_model = get_peft_model(model, build_lora_config(search.best_config))
```

## QPruner

QPruner applies mixed-precision quantization after pruning. The local implementation collects calibration activations, computes `I(X; Y)` between layer outputs and model predictions, initializes a memory-constrained bitwidth plan, and can refine it with GP Bayesian optimization.

```python
from tidal.workflows.compression import qpruner_compress

run = qpruner_compress(
    model=pruned_model,
    calibration_batches=calibration_batches,
    candidate_bits=(2, 4, 8),
    max_average_bits=4.0,
    objective=finetune_and_eval_bitwidths,
    refine_trials=8,
)
quantized_model = run.model
```

For method-level experiments that already manage calibration and targets, use `tidal.methods.qpruner.torch.run_qpruner_mixed_precision` directly.

## Global Rank And Sparsity

The global rank/sparsity implementation follows the CAP two-stage design: RPCA decomposition with ADMM, global Bernoulli retention probabilities over singular directions and sparse entries, deterministic budget selection, and Torch packed Linear replacement.

```python
from tidal.methods.global_rank_sparsity.torch import apply_global_cap_compression

compressed_model = apply_global_cap_compression(
    model,
    total_budget=global_parameter_budget,
    evaluator=calibration_loss,
    policy_steps=80,
)
```

## Upstream Integrations

The team-maintained upstream repositories are kept intact under `external/` and connected through method packages:

| Method | Upstream | Local path |
| --- | --- | --- |
| Dynamic Operator Optimization | `harrysyz99/Dop` | `external/Dop` |
| QR-Adaptor | `harrysyz99/qr_adapter` | `external/qr_adapter` |
| AutoQRA | `harrysyz99/autoqra` | `external/autoqra` |

## Citation

### Toolkit

```bibtex
@misc{tidal2026,
  title = {TIDAL: Efficient Large-Model Systems Toolkit},
  author = {Zhou, Changhai and Zhou, Yuhua and Zhang, Shiyang},
  year = {2026},
  url = {https://github.com/manlenzzz/tidal-ai}
}
```

### Method Papers

#### RankAdaptor

```bibtex
@misc{zhou2024rankadaptor,
  title = {RankAdaptor: Hierarchical Rank Allocation for Efficient Fine-Tuning Pruned LLMs via Performance Model},
  author = {Zhou, Changhai and Han, Shijie and Yang, Lining and Zhou, Yuhua and Cheng, Xu and Wang, Yibin and Li, Hongguang},
  year = {2024},
  eprint = {2406.15734},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  doi = {10.48550/arXiv.2406.15734},
  url = {https://arxiv.org/abs/2406.15734}
}
```

#### QPruner

```bibtex
@misc{zhou2024qpruner,
  title = {QPruner: Probabilistic Decision Quantization for Structured Pruning in Large Language Models},
  author = {Zhou, Changhai and Zhou, Yuhua and Han, Shijie and Qiao, Qian and Li, Hongguang},
  year = {2024},
  eprint = {2412.11629},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  doi = {10.48550/arXiv.2412.11629},
  url = {https://arxiv.org/abs/2412.11629}
}
```

#### Dynamic Operator Optimization

```bibtex
@article{zhou2025dynamicoperator,
  title = {Dynamic Operator Optimization for Efficient Multi-Tenant LoRA Model Serving},
  author = {Zhou, Changhai and Zhou, Yuhua and Zhang, Shiyang and Wang, Yibin and Liu, Zekai},
  journal = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume = {39},
  number = {21},
  pages = {22910--22918},
  year = {2025},
  doi = {10.1609/aaai.v39i21.34453},
  url = {https://ojs.aaai.org/index.php/AAAI/article/view/34453}
}
```

#### QR-Adaptor

```bibtex
@misc{zhou2025qradaptor,
  title = {Balancing Fidelity and Plasticity: Aligning Mixed-Precision Fine-Tuning with Linguistic Hierarchies},
  author = {Zhou, Changhai and Zhang, Shiyang and Zhou, Yuhua and Qiao, Qian and Gao, Jun and Weng, Shichao and Zhang, Weizhong and Jin, Cheng},
  year = {2025},
  eprint = {2505.03802},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  doi = {10.48550/arXiv.2505.03802},
  url = {https://arxiv.org/abs/2505.03802}
}
```

#### AutoQRA

```bibtex
@misc{zhou2026autoqra,
  title = {AutoQRA: Joint Optimization of Mixed-Precision Quantization and Low-rank Adapters for Efficient LLM Fine-Tuning},
  author = {Zhou, Changhai and Zhang, Shiyang and Zhou, Yuhua and Qiao, Qian and Gao, Jun and Jin, Cheng and Qin, Kaizhou and Zhang, Weizhong},
  year = {2026},
  eprint = {2602.22268},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  doi = {10.48550/arXiv.2602.22268},
  url = {https://arxiv.org/abs/2602.22268}
}
```

#### Global Rank and Sparsity Optimization

```bibtex
@inproceedings{zhou2026globalranksparsity,
  title = {Large Language Model Compression with Global Rank and Sparsity Optimization},
  author = {Zhou, Changhai and Qiao, Qian and Zhou, Yuhua and Wu, Yuxin and Weng, Shichao and Zhang, Weizhong and Jin, Cheng},
  booktitle = {International Conference on Learning Representations},
  year = {2026},
  url = {https://openreview.net/forum?id=ZaPmQ0NHs4}
}
```


## Development

```bash
python -m pytest -q
python -m compileall -q tidal examples tests
```

See `docs/citation.md` for citation guidance.
