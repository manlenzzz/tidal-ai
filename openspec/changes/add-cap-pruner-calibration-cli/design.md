## Architecture

The change adds a small experiment layer under `tidal.methods.global_rank_sparsity.experiments`. This module owns file parsing, calibration batch construction, loss computation, summary serialization, and orchestration around the existing CAP Torch API. The existing algorithm code remains responsible for RPCA, global candidate selection, module packing, and temporary evaluator patching.

The example script `examples/global_rank_sparsity/hf_cap_experiment.py` is a thin CLI wrapper. It loads a Hugging Face causal LM and tokenizer, builds an optional pruner-derived `name_filter`, builds optional calibration batches, calls `run_cap_compression`, and writes or prints a JSON summary.

## Components

- `load_pruner_target_names(path)`: reads module names from plain text, JSON lists/dicts, JSONL records, or Torch checkpoint/state files. Tensor suffix and wrapper canonicalization are still handled by `build_pruned_module_name_filter`.
- `build_text_calibration_batches(tokenizer, texts, ...)`: tokenizes local text into CPU tensors with `input_ids`, `attention_mask`, and `labels`.
- `load_calibration_texts(path, text_field="text")`: reads plain text or JSONL calibration records.
- `causal_lm_loss(model, batch)`: returns model loss for evaluator use.
- `summarize_cap_run(result, model_id, ...)`: emits stable JSON-friendly metadata.

## Data Flow

The CLI resolves cache and output paths, then loads model/tokenizer lazily. If a pruner target file is provided, it creates a filter with `build_pruned_module_name_filter(names, target_roles=args.target_roles)`. If calibration data is provided, it builds a bounded list of short batches and passes `calibration_batches` plus `causal_lm_loss` to `run_cap_compression`. The final summary includes target counts, selected budget, selected candidates, per-layer summaries, and the paths/settings used for the run.

## Error Handling

The helpers raise `ValueError` for empty calibration data, unreadable name formats, calibration files with no usable text records, and empty tokenized batches. The CLI lets these errors surface with a clear traceback during research use rather than hiding them behind broad exception handling.

## Testing

Tests cover pruner-name loading formats, calibration batch construction with a fake tokenizer, causal-LM loss on a tiny model, summary fields, and CLI help. A local HF tiny model smoke remains optional and is not required for the normal unit test suite.
