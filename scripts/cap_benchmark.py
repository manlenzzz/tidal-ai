"""CAP performance benchmark — quality, compression-time, inference latency, compression ratio.

Self-contained, builds a small Llama locally (no large download). Compares the
original model against the CAP-compressed model on four axes:

  1. Model quality   : perplexity on a fixed eval batch (lower = better)
  2. Compression time: wall-clock of run_cap_compression
  3. Inference       : forward latency + parameter count of targeted layers
  4. Compression rate: targeted-parameter reduction and per-layer rank/sparsity

Runs on whatever device is passed (cpu / cuda / npu). Compression math stays on CPU/numpy
by design; only the model forward runs on the device.
"""
from __future__ import annotations

import argparse
import json
import os
import time

# On many-core hosts torch otherwise spawns one thread per core, and the
# sync overhead dominates these small matmuls (1000x slower). Cap before import.
os.environ.setdefault("OMP_NUM_THREADS", "8")

import torch

torch.set_num_threads(min(8, os.cpu_count() or 8))

from tidal.device import resolve_device, resolve_dtype
from tidal.methods.global_rank_sparsity.torch import CAPPackedLinear, run_cap_compression


def build_model(seed: int):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(cfg)
    model.eval()
    return model, cfg


@torch.no_grad()
def perplexity(model, batch) -> float:
    out = model(**batch)
    return float(torch.exp(out.loss).item())


@torch.no_grad()
def forward_latency_ms(model, batch, *, iters: int, warmup: int, device) -> float:
    for _ in range(warmup):
        model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    synchronize_device(device)
    start = time.perf_counter()
    for _ in range(iters):
        model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    synchronize_device(device)
    return (time.perf_counter() - start) / iters * 1000.0


def synchronize_device(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "npu":
        torch.npu.synchronize()


def reset_peak_memory_stats(device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    elif device.type == "npu" and hasattr(torch.npu, "reset_peak_memory_stats"):
        torch.npu.reset_peak_memory_stats()


def peak_memory_mb(device) -> float | None:
    if device.type == "cuda":
        return round(torch.cuda.max_memory_allocated() / 1024**2, 1)
    if device.type == "npu" and hasattr(torch.npu, "max_memory_allocated"):
        return round(torch.npu.max_memory_allocated() / 1024**2, 1)
    return None


def tokens_per_second(*, batch_size: int, seq_len: int, latency_ms: float) -> float | None:
    if latency_ms <= 0:
        return None
    return round((batch_size * seq_len) / (latency_ms / 1000.0), 3)


def targeted_param_count(model, names) -> int:
    total = 0
    by_name = dict(model.named_modules())
    for name in names:
        module = by_name[name]
        if isinstance(module, CAPPackedLinear):
            total += int(module.parameter_count)
        else:
            total += int(module.weight.numel())
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description="CAP performance benchmark")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--budget", type=int, default=8_000)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--policy-steps", type=int, default=2)
    parser.add_argument("--samples-per-step", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--summary-json", default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    torch_dtype = dtype if isinstance(dtype, torch.dtype) else torch.float32

    model, cfg = build_model(args.seed)
    model = model.to(device=device, dtype=torch_dtype)

    torch.manual_seed(args.seed)
    ids = torch.randint(0, cfg.vocab_size, (args.batch_size, args.seq_len), device=device)
    eval_batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}

    report: dict[str, object] = {
        "device": str(device),
        "dtype": str(torch_dtype),
        "model_total_params": int(sum(p.numel() for p in model.parameters())),
    }

    # Baseline quality + latency on the targeted (modern) linear layers.
    report["baseline_perplexity"] = perplexity(model, eval_batch)
    report["baseline_latency_ms"] = forward_latency_ms(
        model, eval_batch, iters=args.iters, warmup=args.warmup, device=device
    )
    report["baseline_tokens_per_s"] = tokens_per_second(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        latency_ms=float(report["baseline_latency_ms"]),
    )

    # Compression (math on CPU/numpy regardless of device).
    reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    result = run_cap_compression(
        model,
        total_budget=args.budget,
        inplace=False,
        target_roles="modern",
        max_iter=args.max_iter,
        policy_steps=args.policy_steps,
        samples_per_step=args.samples_per_step,
        seed=args.seed,
    )
    report["compression_time_s"] = time.perf_counter() - t0

    compressed = result.compressed_model
    target_names = [t.name for t in result.targets]
    orig_targeted = targeted_param_count(model, target_names)
    comp_targeted = targeted_param_count(compressed, target_names)
    report["targeted_layers"] = len(target_names)
    report["targeted_params_original"] = orig_targeted
    report["targeted_params_compressed"] = comp_targeted
    report["targeted_compression_ratio"] = (
        round(orig_targeted / comp_targeted, 3) if comp_targeted else None
    )
    report["targeted_param_reduction_pct"] = (
        round(100.0 * (1 - comp_targeted / orig_targeted), 2) if orig_targeted else None
    )

    report["compressed_perplexity"] = perplexity(compressed, eval_batch)
    report["perplexity_delta"] = report["compressed_perplexity"] - report["baseline_perplexity"]
    report["compressed_latency_ms"] = forward_latency_ms(
        compressed, eval_batch, iters=args.iters, warmup=args.warmup, device=device
    )
    report["compressed_tokens_per_s"] = tokens_per_second(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        latency_ms=float(report["compressed_latency_ms"]),
    )
    report["latency_speedup"] = (
        round(report["baseline_latency_ms"] / report["compressed_latency_ms"], 3)
        if report["compressed_latency_ms"]
        else None
    )

    peak_mem = peak_memory_mb(device)
    if peak_mem is not None:
        report["peak_mem_mb"] = peak_mem

    report["layers"] = [
        {"name": l.name, "rank": l.rank, "sparse_entries": l.sparse_entries, "params": l.parameter_count}
        for l in result.layer_summaries[:8]
    ]

    print("CAP_BENCH " + json.dumps(report))
    if args.summary_json:
        with open(args.summary_json, "w") as f:
            json.dump(report, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
