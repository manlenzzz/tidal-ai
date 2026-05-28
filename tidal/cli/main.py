from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tidal.workflows.adaptation import rankadaptor_adapt
from tidal.workflows.compression import cap_compress, qpruner_compress


def resolve_cache_dir(cache_dir: str | None) -> str | None:
    if cache_dir:
        return cache_dir
    return os.environ.get("TIDAL_HF_CACHE") or os.environ.get("HF_HOME")


def parse_int_list(value: str) -> tuple[int, ...]:
    bits = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not bits:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return bits


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

    qpruner = compress_subparsers.add_parser("qpruner", help="Run QPruner mixed-precision compression.")
    qpruner.add_argument("--model-id", required=True, help="Hugging Face model id or local model path.")
    qpruner.add_argument("--cache-dir", default=None)
    qpruner.add_argument("--target-roles", default="modern")
    qpruner.add_argument("--pruner-targets", default=None, help="Text, JSON, JSONL, or Torch state file with pruned target names.")
    qpruner.add_argument("--calibration-data", default=None, help="Plain text or JSONL file for MI calibration batches.")
    qpruner.add_argument("--calibration-text-field", default="text")
    qpruner.add_argument("--calibration-max-samples", type=int, default=8)
    qpruner.add_argument("--calibration-max-length", type=int, default=128)
    qpruner.add_argument("--calibration-batch-size", type=int, default=1)
    qpruner.add_argument("--candidate-bits", type=parse_int_list, default=(2, 4, 8), help="Comma-separated bitwidth candidates, e.g. 2,4,8.")
    qpruner.add_argument("--max-memory-bits", type=int, default=None, help="Global mixed-precision memory budget in bits.")
    qpruner.add_argument("--max-average-bits", type=float, default=None, help="Average bitwidth budget across selected weights.")
    qpruner.add_argument("--refine-trials", type=int, default=0, help="Optional GP refinement trials when an objective is supplied by Python API.")
    qpruner.add_argument("--bins", type=int, default=16, help="Discretization bins for mutual information.")
    qpruner.add_argument("--seed", type=int, default=0)
    qpruner.add_argument("--revision", default=None)
    qpruner.add_argument("--local-files-only", action="store_true")
    qpruner.add_argument("--trust-remote-code", action="store_true")
    qpruner.add_argument("--output", default=None, help="Optional run directory. Writes summary.json when supplied.")

    adapt = subparsers.add_parser("adapt", help="Adaptation and fine-tuning preparation workflows.")
    adapt_subparsers = adapt.add_subparsers(dest="method", required=True)

    rankadaptor = adapt_subparsers.add_parser("rankadaptor", help="Run RankAdaptor LoRA rank allocation.")
    rankadaptor.add_argument("--model-id", required=True, help="Hugging Face model id or local model path.")
    rankadaptor.add_argument("--cache-dir", default=None)
    rankadaptor.add_argument("--target-roles", default="modern")
    rankadaptor.add_argument("--pruner-targets", default=None, help="Text, JSON, JSONL, or Torch state file with pruned target names.")
    rankadaptor.add_argument("--sensitivities", default=None, help="JSON, JSONL, or text file mapping module names to sensitivity scores.")
    rankadaptor.add_argument("--budget", type=int, required=True, help="Total LoRA adapter parameter budget.")
    rankadaptor.add_argument("--min-rank", type=int, default=1)
    rankadaptor.add_argument("--max-rank", type=int, default=64)
    rankadaptor.add_argument("--rank-step", type=int, default=1)
    rankadaptor.add_argument("--alpha-multiplier", type=int, default=2)
    rankadaptor.add_argument("--max-steps", type=int, default=None)
    rankadaptor.add_argument("--min-gain", type=float, default=0.0)
    rankadaptor.add_argument("--seed", type=int, default=0)
    rankadaptor.add_argument("--revision", default=None)
    rankadaptor.add_argument("--local-files-only", action="store_true")
    rankadaptor.add_argument("--trust-remote-code", action="store_true")
    rankadaptor.add_argument("--no-apply-peft", action="store_true", help="Only emit the selected PEFT config summary; do not wrap the model.")
    rankadaptor.add_argument("--output", default=None, help="Optional run directory. Writes summary.json when supplied.")
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
    if args.command == "compress" and args.method == "qpruner":
        result = qpruner_compress(
            model_id=args.model_id,
            cache_dir=resolve_cache_dir(args.cache_dir),
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code,
            revision=args.revision,
            pruner_targets=args.pruner_targets,
            calibration_data=args.calibration_data,
            calibration_text_field=args.calibration_text_field,
            calibration_max_samples=args.calibration_max_samples,
            calibration_max_length=args.calibration_max_length,
            calibration_batch_size=args.calibration_batch_size,
            target_roles=args.target_roles,
            candidate_bits=args.candidate_bits,
            max_memory_bits=args.max_memory_bits,
            max_average_bits=args.max_average_bits,
            refine_trials=args.refine_trials,
            bins=args.bins,
            seed=args.seed,
        )
        if args.output:
            result.save(Path(args.output))
        print(json.dumps(result.summary, indent=2, sort_keys=True))
        return 0
    if args.command == "adapt" and args.method == "rankadaptor":
        result = rankadaptor_adapt(
            model_id=args.model_id,
            cache_dir=resolve_cache_dir(args.cache_dir),
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code,
            revision=args.revision,
            pruner_targets=args.pruner_targets,
            sensitivities=args.sensitivities,
            budget=args.budget,
            min_rank=args.min_rank,
            max_rank=args.max_rank,
            rank_step=args.rank_step,
            target_roles=args.target_roles,
            max_steps=args.max_steps,
            min_gain=args.min_gain,
            alpha_multiplier=args.alpha_multiplier,
            apply_peft=not args.no_apply_peft,
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
