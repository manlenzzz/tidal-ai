# TIDAL Documentation

TIDAL stands for Toolkit for Inference, Deployment, Adaptation, and Learning. This documentation organizes the project as an open-source toolkit for efficient AI systems, with the initial method wave coming from six papers by Changhai Zhou.

## Start Here

| Page | Purpose |
| --- | --- |
| [`method-zoo.md`](method-zoo.md) | Method taxonomy, integration status, and source mapping |
| [`reproduction-matrix.md`](reproduction-matrix.md) | Current reproduction readiness for each seed method |
| [`roadmap.md`](roadmap.md) | Project direction across inference, serving, fine-tuning, post-training, and RL |
| [`gpu-execution.md`](gpu-execution.md) | Required execution policy for GPU work in this workspace |
| [`contributing.md`](contributing.md) | How to add methods, wrappers, tests, and evidence |
| [`citation.md`](citation.md) | How to cite TIDAL and the seed papers |

## Toolkit Principles

- Keep methods reusable beyond a single paper artifact.
- Separate CPU-testable logic from GPU-required training and evaluation.
- Record provenance for papers, code, commands, artifacts, and results.
- Prefer submodules or documented upstream links for collaborator-maintained codebases.
- Avoid claiming reproduction until evidence is captured in the method entry.
