from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tidal.workflows.compression import cap_compress


def resolve_cache_dir(cache_dir: str | None) -> str | None:
    if cache_dir:
        return cache_dir
    return os.environ.get("TIDAL_HF_CACHE") or os.environ.get("HF_HOME")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tidal", description="Task-first CLI for TIDAL efficient large-model workflows.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    compress = subparsers.add_parser("compress", help="Compression workflows.")
    compress_subparsers = compress.add_subparsers(dest="method", required=True)

    cap = compress_subparsers.add_parser("cap", help="Run CAP global rank/sparsity compression.")
    cap.add_argument("--model-id", required=True, help="Hugging Face model id or local model path.")
    cap.add_argument("--cache-dir", default=None)
    cap.add_argument("--target-roles", default="modern")
    cap.add_argument("--pruner-targets", default=None, help="Text, JSON, JSONL, or Torch state file with pruned target names.")
    cap.add_argument("--calibration-data", default=None, help="Plain text or JSONL file for calibration loss batches.")
    cap.add_argument("--calibration-text-field", default="text")
    cap.add_argument("--calibration-max-samples", type=int, default=8)
    cap.add_argument("--calibration-max-length", type=int, default=128)
    cap.add_argument("--calibration-batch-size", type=int, default=1)
    cap.add_argument("--budget", type=int, required=True, help="Total CAP parameter budget.")
    cap.add_argument("--max-iter", type=int, default=20)
    cap.add_argument("--policy-steps", type=int, default=4)
    cap.add_argument("--samples-per-step", type=int, default=2)
    cap.add_argument("--seed", type=int, default=0)
    cap.add_argument("--revision", default=None)
    cap.add_argument("--local-files-only", action="store_true")
    cap.add_argument("--trust-remote-code", action="store_true")
    cap.add_argument("--output", default=None, help="Optional run directory. Writes summary.json when supplied.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "compress" and args.method == "cap":
        result = cap_compress(
            model_id=args.model_id,
            cache_dir=resolve_cache_dir(args.cache_dir),
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code,
            revision=args.revision,
            budget=args.budget,
            pruner_targets=args.pruner_targets,
            calibration_data=args.calibration_data,
            calibration_text_field=args.calibration_text_field,
            calibration_max_samples=args.calibration_max_samples,
            calibration_max_length=args.calibration_max_length,
            calibration_batch_size=args.calibration_batch_size,
            target_roles=args.target_roles,
            max_iter=args.max_iter,
            policy_steps=args.policy_steps,
            samples_per_step=args.samples_per_step,
            seed=args.seed,
        )
        if args.output:
            result.save(Path(args.output))
        print(json.dumps(result.summary, indent=2, sort_keys=True))
        return 0
    parser.error("unsupported command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
