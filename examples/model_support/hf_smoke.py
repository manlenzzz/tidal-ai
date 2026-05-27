#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a tiny Hugging Face causal LM on CPU and report TIDAL-selected linear modules."
    )
    parser.add_argument(
        "--model-id",
        default="hf-internal-testing/tiny-random-LlamaForCausalLM",
        help="Hugging Face model id or local model path.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Cache directory. Defaults to TIDAL_HF_CACHE, HF_HOME, then ./.hf_cache.",
    )
    parser.add_argument(
        "--target-roles",
        default="modern",
        help="TIDAL target role preset or explicit role name.",
    )
    parser.add_argument(
        "--max-modules",
        type=int,
        default=24,
        help="Maximum selected modules to include in the JSON sample.",
    )
    parser.add_argument("--revision", default=None, help="Optional Hugging Face revision.")
    parser.add_argument("--local-files-only", action="store_true", help="Do not download files from the Hub.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to transformers.")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32", "none"),
        default="auto",
        help="Torch dtype for from_pretrained; use none to omit torch_dtype.",
    )
    return parser.parse_args(argv)


def resolve_cache_dir(cache_dir: str | None) -> str:
    if cache_dir:
        return cache_dir
    return os.environ.get("TIDAL_HF_CACHE") or os.environ.get("HF_HOME") or str(Path.cwd() / ".hf_cache")


def dtype_argument(dtype: str) -> object:
    if dtype == "none":
        return None
    if dtype == "auto":
        return "auto"
    import torch

    return getattr(torch, dtype)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from transformers import AutoModelForCausalLM

    from tidal.model_support import list_linear_modules

    cache_dir = resolve_cache_dir(args.cache_dir)
    load_kwargs: dict[str, object] = {
        "cache_dir": cache_dir,
        "local_files_only": args.local_files_only,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.revision:
        load_kwargs["revision"] = args.revision
    dtype = dtype_argument(args.dtype)
    if dtype is not None:
        load_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **load_kwargs)
    model.eval()

    modules = list_linear_modules(model, target_roles=args.target_roles)
    role_counts = Counter(info.role.value for info in modules)
    sample = [
        {
            "name": info.name,
            "canonical_name": info.canonical_name,
            "role": info.role.value,
            "in_features": info.in_features,
            "out_features": info.out_features,
        }
        for info in modules[: max(0, args.max_modules)]
    ]
    result = {
        "model_id": args.model_id,
        "cache_dir": cache_dir,
        "target_roles": args.target_roles,
        "total_selected": len(modules),
        "roles": dict(sorted(role_counts.items())),
        "modules": sample,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
