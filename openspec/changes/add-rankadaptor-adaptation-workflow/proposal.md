# Add RankAdaptor Adaptation Workflow

## Why
TIDAL has a method-level RankAdaptor implementation, but users still need to manually collect target modules, load pruner outputs, run rank allocation, build PEFT LoRA configs, and save run metadata. The toolkit should expose RankAdaptor as a reusable adaptation workflow for recovering pruned models with rank-adaptive LoRA.

## What Changes
- Add a public `rankadaptor_adapt` workflow under `tidal.workflows.adaptation`.
- Reuse shared target loading so LLM-Pruner/WANDA exported module names select the same modules as compression workflows.
- Build module profiles, run RankAdaptor rank allocation, build PEFT `LoraConfig`, and optionally apply it with `get_peft_model`.
- Add RankAdaptor summary reporting with rank config, adapter budget/cost, score, and target metadata.
- Add `tidal adapt rankadaptor` CLI options for Hugging Face models, sensitivities, rank bounds, budgets, pruner targets, and output summaries.
- Update README public entrypoints to show adaptation as a first-class workflow.

## Capabilities
- rankadaptor-adaptation-workflow

## Impact
- Public workflows expand from compression-only to compression plus adaptation.
- CLI gains an `adapt` command group.
- Tests stay CPU-only and use tiny in-memory Torch modules where possible.
