#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a CPU Hugging Face CAP experiment with optional pruner targets and text calibration."
    )
    parser.add_argument("--model-id", default="hf-internal-testing/tiny-random-LlamaForCausalLM")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--target-roles", default="modern")
    parser.add_argument("--pruner-targets", default=None, help="Text, JSON, JSONL, or Torch state file with pruned target names.")
    parser.add_argument("--calibration-data", default=None, help="Plain text or JSONL file for calibration loss batches.")
    parser.add_argument("--calibration-text-field", default="text")
    parser.add_argument("--calibration-max-samples", type=int, default=8)
    parser.add_argument("--calibration-max-length", type=int, default=128)
    parser.add_argument("--calibration-batch-size", type=int, default=1)
    parser.add_argument("--summary-json", default=None, help="Optional path to write the run summary JSON.")
    parser.add_argument("--total-budget", type=int, default=256)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--policy-steps", type=int, default=4)
    parser.add_argument("--samples-per-step", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args(argv)


def resolve_cache_dir(cache_dir: str | None) -> str:
    if cache_dir:
        return cache_dir
    return os.environ.get("TIDAL_HF_CACHE") or os.environ.get("HF_HOME") or str(Path.cwd() / ".hf_cache")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from tidal.model_support import build_pruned_module_name_filter
    from tidal.methods.global_rank_sparsity.experiments import (
        build_text_calibration_batches,
        causal_lm_loss,
        load_calibration_texts,
        load_pruner_target_names,
        summarize_cap_run,
    )
    from tidal.methods.global_rank_sparsity.torch import run_cap_compression

    cache_dir = resolve_cache_dir(args.cache_dir)
    load_kwargs: dict[str, object] = {
        "cache_dir": cache_dir,
        "local_files_only": args.local_files_only,
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": "auto",
    }
    if args.revision:
        load_kwargs["revision"] = args.revision

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **load_kwargs)
    model.eval()

    pruner_names: list[str] | None = None
    name_filter = None
    if args.pruner_targets:
        pruner_names = load_pruner_target_names(args.pruner_targets)
        name_filter = build_pruned_module_name_filter(pruner_names, target_roles=args.target_roles)

    calibration_batches = None
    if args.calibration_data:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_id,
            cache_dir=cache_dir,
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code,
        )
        texts = load_calibration_texts(args.calibration_data, text_field=args.calibration_text_field)
        texts = texts[: max(0, args.calibration_max_samples)]
        calibration_batches = build_text_calibration_batches(
            tokenizer,
            texts,
            max_length=args.calibration_max_length,
            batch_size=args.calibration_batch_size,
            device="cpu",
        )

    result = run_cap_compression(
        model,
        total_budget=args.total_budget,
        name_filter=name_filter,
        target_roles=args.target_roles,
        calibration_batches=calibration_batches,
        loss_fn=causal_lm_loss if calibration_batches is not None else None,
        max_iter=args.max_iter,
        policy_steps=args.policy_steps,
        samples_per_step=args.samples_per_step,
        seed=args.seed,
    )
    summary = summarize_cap_run(
        result,
        model_id=args.model_id,
        target_roles=args.target_roles,
        pruner_target_count=len(pruner_names) if pruner_names is not None else None,
        calibration_batches=len(calibration_batches) if calibration_batches is not None else None,
    )
    payload = json.dumps(summary, indent=2, sort_keys=True)
    if args.summary_json:
        output_path = Path(args.summary_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
