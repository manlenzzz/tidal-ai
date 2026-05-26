# TIDAL

<p align="center">
  <strong>Toolkit for Inference, Deployment, Adaptation, and Learning</strong><br>
  Open infrastructure for efficient AI systems, from compression and fine-tuning to serving, post-training, and future RL workflows.
</p>

<p align="center">
  <a href="https://github.com/manlenzzz/tidal-ai"><img alt="Project" src="https://img.shields.io/badge/project-TIDAL-0f766e"></a>
  <a href="docs/method-zoo.md"><img alt="Methods" src="https://img.shields.io/badge/methods-6%20seed%20papers-2563eb"></a>
  <a href="docs/roadmap.md"><img alt="Scope" src="https://img.shields.io/badge/scope-compression%20%7C%20adaptation%20%7C%20serving%20%7C%20post--training-7c3aed"></a>
</p>

TIDAL is an open-source toolkit for efficient large-model systems. The project starts from six papers by Changhai Zhou and is organized as a reusable research and engineering base rather than a private paper archive. The long-term goal is to collect practical components for efficient inference, serving, fine-tuning, compression, post-training, and reinforcement learning.

## What TIDAL Covers

| Area | Current focus | Direction |
| --- | --- | --- |
| Compression | Pruning, quantization, rank/sparsity optimization | Search spaces, export formats, reproducible compression recipes |
| Adaptation | LoRA rank allocation, mixed precision, quantization-aware adapters | Adapter allocation, precision policies, PEFT-compatible configs |
| Serving | Multi-tenant LoRA operator optimization | Benchmarks and deployment paths for adapter-heavy serving |
| Post-training | Manifest, experiment, and GPU execution discipline | SFT and preference-optimization pipelines |
| Reinforcement learning | Planned | Efficient RL training and evaluation workflows |

## Seed Method Wave

| Method | Axis | Source status | TIDAL entry |
| --- | --- | --- | --- |
| RankAdaptor | Efficient fine-tuning of pruned LLMs | Paper-derived implementation plan | [`reproductions/rankadaptor`](reproductions/rankadaptor/README.md) |
| QPruner | Structured pruning with probabilistic decision quantization | Paper-derived implementation plan | [`reproductions/qpruner`](reproductions/qpruner/README.md) |
| Dynamic Operator Optimization | Multi-tenant LoRA serving | Upstream code integrated as submodule | [`reproductions/dynamic-operator-optimization`](reproductions/dynamic-operator-optimization/README.md) |
| QR-Adaptor | Mixed-precision fine-tuning aligned with linguistic hierarchies | Upstream code integrated as submodule | [`reproductions/qr-adaptor`](reproductions/qr-adaptor/README.md) |
| AutoQRA | Joint mixed-precision quantization and low-rank adaptation | Upstream code integrated as submodule | [`reproductions/autoqra`](reproductions/autoqra/README.md) |
| Global Rank/Sparsity Optimization | Joint compression search | Paper-derived implementation plan | [`reproductions/global-rank-sparsity`](reproductions/global-rank-sparsity/README.md) |

The external codebases are tracked under `external/` as Git submodules when available. Full scientific reproduction is claimed only after commands, prerequisites, outputs, comparison targets, and verification evidence are recorded for a method.

## Repository Layout

```text
papers/sources.yaml        Paper and artifact manifest
external/                  Integrated upstream code repositories
reproductions/             Method entries, reproduction plans, and evidence
tidal/                     Shared TIDAL helpers
docs/                      Method zoo, roadmap, policies, citations
site/                      Static project homepage
experiments/               Local experiment entry points; large outputs ignored
```

Local PDFs and extracted text live under `papers/pdf/` and `papers/text/`, but they are ignored by Git by default to keep the public repository lightweight.

## Quickstart

Clone with submodules:

```bash
git clone --recurse-submodules https://github.com/manlenzzz/tidal-ai.git
cd tidal-ai
```

Validate the method manifest:

```bash
/opt/venv/bin/python -m tidal.manifest papers/sources.yaml
```

Run the current CPU-only checks:

```bash
/opt/venv/bin/python -m pytest tests/test_manifest_validation.py
```

## Documentation

- [`docs/index.md`](docs/index.md): documentation entry point
- [`docs/method-zoo.md`](docs/method-zoo.md): current method taxonomy and integration status
- [`docs/reproduction-matrix.md`](docs/reproduction-matrix.md): reproduction readiness matrix
- [`docs/roadmap.md`](docs/roadmap.md): project scope and staged roadmap
- [`docs/gpu-execution.md`](docs/gpu-execution.md): GPU execution and cleanup policy
- [`docs/contributing.md`](docs/contributing.md): contribution workflow
- [`docs/citation.md`](docs/citation.md): citation guidance

## GPU and Artifact Policy

Do not run GPU-required jobs locally in this workspace. Training, fine-tuning, CUDA kernel work, large-model inference, and batch scoring must use Mint Ray or an allowed worker from `/vePFS-Mindverse/user/intern/zhouch/config/mint_ray.yaml`. Public job names and namespaces must stay generic.

Large model weights, datasets, checkpoints, and experiment outputs should stay outside Git and under the durable GPFS workspace paths described in the root `AGENTS.md` policy.

## Maintainer

TIDAL is maintained by Changhai Zhou. The project is designed to grow beyond the initial paper wave into a broader toolkit for efficient AI research and deployment.
