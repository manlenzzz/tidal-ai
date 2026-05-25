# First-Author LLM Toolkit

This repository collects local paper artifacts, source provenance, and reproducibility plans for six first-author LLM efficiency papers.

## Papers

- RankAdaptor: Hierarchical Rank Allocation for Efficient Fine-Tuning Pruned LLMs via Performance Model
- QPruner: Probabilistic Decision Quantization for Structured Pruning in Large Language Models
- Dynamic Operator Optimization for Efficient Multi-Tenant LoRA Model Serving
- Balancing Fidelity and Plasticity: Aligning Mixed-Precision Fine-Tuning with Linguistic Hierarchies
- AutoQRA: Joint Optimization of Mixed-Precision Quantization and Low-rank Adapters for Efficient LLM Fine-Tuning
- Large Language Model Compression with Global Rank and Sparsity Optimization

## Layout

- `papers/sources.yaml`: source-of-truth manifest for paper URLs, local artifact paths, checksums, and code status.
- `papers/pdf/` and `papers/text/`: local PDF/text artifacts. These are ignored by Git by default.
- `external/`: upstream code repositories for papers with released code.
- `reproductions/`: per-paper reproduction plans and status.
- `toolkit/`: shared local helpers.
- `docs/`: reproduction matrix and GPU execution policy.
- `experiments/`: local experiment entry points and notes; large outputs are ignored.

## Current Status

The corpus manifest and paper artifacts have been staged locally. The open-source repositories are integrated under `external/` when available. Full scientific reproduction is not claimed until each per-paper entry records the required commands, data/model prerequisites, results, and verification evidence.

## Local Checks

```bash
/opt/venv/bin/python -m toolkit.manifest papers/sources.yaml
/opt/venv/bin/python -m pytest tests/test_manifest_validation.py
```

## GPU Policy

Do not run GPU-required jobs locally. Training, fine-tuning, CUDA kernel work, large-model inference, and batch scoring must use Mint Ray or an allowed worker from `/vePFS-Mindverse/user/intern/zhouch/config/mint_ray.yaml`. Public job names and namespaces must stay generic, such as `zhouch+train` or `zhouch+eval`.
