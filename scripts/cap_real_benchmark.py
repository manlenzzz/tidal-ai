"""Real-model CAP benchmark on a Hugging Face causal LM (e.g. Qwen3-4B).

Measures, comparing original vs CAP-compressed:
  1. Model quality    : perplexity on a fixed text batch (lower = better)
  2. Stage 1 RPCA time: wall-clock of run_cap_compression (numpy vs torch backend)
  3. Compression ratio: targeted-parameter reduction
  4. Inference        : forward latency of the compressed model

Run with --rpca-backend torch --device cuda on a GPU node to exercise the GPU
RPCA path. Designed to run via the Mint Ray launcher on A800.
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "16")

import torch

from tidal.device import resolve_device, resolve_dtype
from tidal.methods.global_rank_sparsity.torch import CAPPackedLinear, collect_cap_targets, run_cap_compression


CALIB_TEXT = [
    "The history of large language models is one of rapid scaling and capability gains.",
    "Robust principal component analysis decomposes a matrix into low-rank and sparse parts.",
    "Compression methods trade a small amount of accuracy for large reductions in model size.",
    "Attention mechanisms let transformers weigh the relevance of different input tokens.",
]


@torch.no_grad()
def perplexity(model, batch) -> float:
    out = model(**batch)
    return float(torch.exp(out.loss).item())


@torch.no_grad()
def forward_latency_ms(model, batch, *, iters, warmup, device) -> float:
    fwd = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}
    for _ in range(warmup):
        model(**fwd)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        model(**fwd)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000.0


def targeted_params(model, names) -> int:
    by_name = dict(model.named_modules())
    total = 0
    for name in names:
        m = by_name[name]
        total += int(m.parameter_count) if isinstance(m, CAPPackedLinear) else int(m.weight.numel())
    return total


def main() -> int:
    p = argparse.ArgumentParser(description="Real-model CAP benchmark")
    p.add_argument("--model-id", required=True)
    p.add_argument("--cache-dir", default=os.environ.get("HF_HOME"))
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default=None)
    p.add_argument("--rpca-backend", default="torch", choices=["numpy", "torch"])
    p.add_argument("--budget", type=int, required=True, help="Total CAP parameter budget across targeted layers.")
    p.add_argument("--max-layers", type=int, default=0, help="If >0, only compress the first N targeted layers (scoping).")
    p.add_argument("--max-iter", type=int, default=50)
    p.add_argument("--policy-steps", type=int, default=3)
    p.add_argument("--samples-per-step", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--summary-json", default=None)
    args = p.parse_args()

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    torch_dtype = dtype if isinstance(dtype, torch.dtype) else torch.bfloat16

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_id, cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, cache_dir=args.cache_dir, torch_dtype=torch_dtype
    ).to(device)
    model.eval()

    enc = tok(CALIB_TEXT, return_tensors="pt", padding="max_length", truncation=True, max_length=args.seq_len)
    ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    eval_batch = {"input_ids": ids, "attention_mask": attn, "labels": ids.clone()}

    report = {
        "model_id": args.model_id,
        "device": str(device),
        "dtype": str(torch_dtype),
        "rpca_backend": args.rpca_backend,
        "model_total_params": int(sum(p_.numel() for p_ in model.parameters())),
    }

    # Scope which layers to compress (full model can be many minutes of Stage 1).
    name_filter = None
    if args.max_layers > 0:
        all_targets = collect_cap_targets(model, target_roles="modern")
        keep = {t.name for t in all_targets[: args.max_layers]}
        name_filter = lambda n, keep=keep: n in keep
        report["scoped_layers"] = len(keep)

    report["baseline_perplexity"] = perplexity(model, eval_batch)
    report["baseline_latency_ms"] = forward_latency_ms(
        model, eval_batch, iters=args.iters, warmup=args.warmup, device=device
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    result = run_cap_compression(
        model,
        total_budget=args.budget,
        inplace=False,
        target_roles="modern",
        name_filter=name_filter,
        max_iter=args.max_iter,
        policy_steps=args.policy_steps,
        samples_per_step=args.samples_per_step,
        seed=args.seed,
        rpca_backend=args.rpca_backend,
        rpca_device=device,
    )
    report["compression_time_s"] = round(time.perf_counter() - t0, 2)

    compressed = result.compressed_model
    names = [t.name for t in result.targets]
    orig = targeted_params(model, names)
    comp = targeted_params(compressed, names)
    report["targeted_layers"] = len(names)
    report["targeted_params_original"] = orig
    report["targeted_params_compressed"] = comp
    report["targeted_compression_ratio"] = round(orig / comp, 3) if comp else None
    report["targeted_param_reduction_pct"] = round(100.0 * (1 - comp / orig), 2) if orig else None

    report["compressed_perplexity"] = perplexity(compressed, eval_batch)
    report["perplexity_delta"] = report["compressed_perplexity"] - report["baseline_perplexity"]
    report["compressed_latency_ms"] = forward_latency_ms(
        compressed, eval_batch, iters=args.iters, warmup=args.warmup, device=device
    )
    if device.type == "cuda":
        report["peak_mem_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)

    print("CAP_REAL_BENCH " + json.dumps(report))
    if args.summary_json:
        with open(args.summary_json, "w") as f:
            json.dump(report, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
