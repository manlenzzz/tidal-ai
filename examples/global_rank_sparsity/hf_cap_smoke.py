#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a CPU CAP smoke compression on a tiny Hugging Face causal LM.")
    parser.add_argument("--model-id", default="hf-internal-testing/tiny-random-LlamaForCausalLM")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--target-roles", default="modern")
    parser.add_argument("--total-budget", type=int, default=256)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--policy-steps", type=int, default=4)
    parser.add_argument("--samples-per-step", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args(argv)


def resolve_cache_dir(cache_dir: str | None) -> str:
    if cache_dir:
        return cache_dir
    return os.environ.get("TIDAL_HF_CACHE") or os.environ.get("HF_HOME") or str(Path.cwd() / ".hf_cache")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from transformers import AutoModelForCausalLM

    from tidal.methods.global_rank_sparsity.torch import collect_cap_targets, run_cap_compression

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        cache_dir=resolve_cache_dir(args.cache_dir),
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        torch_dtype="auto",
    )
    targets = collect_cap_targets(model, target_roles=args.target_roles)
    result = run_cap_compression(
        model,
        total_budget=args.total_budget,
        target_roles=args.target_roles,
        max_iter=args.max_iter,
        policy_steps=args.policy_steps,
        samples_per_step=args.samples_per_step,
        seed=args.seed,
    )
    payload = {
        "model_id": args.model_id,
        "target_count": len(targets),
        "compressed_layers": len(result.layer_summaries),
        "parameter_count": result.global_result.parameter_count,
        "selected_candidates": len(result.global_result.selected_candidates),
        "layers": [summary.__dict__ for summary in result.layer_summaries[:16]],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
