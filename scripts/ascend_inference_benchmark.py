#!/usr/bin/env python
"""Benchmark torch_npu/vLLM inference for locally available small models."""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from ascend_inference_probe import default_roots, find_model_path
from generation_padding import ensure_left_padding_for_generate
from tidal.device import resolve_device, resolve_dtype


DEFAULT_PROMPTS = (
    "Explain low-rank model compression in one concise paragraph.",
    "Summarize why Ascend NPU inference benchmarking needs synchronization.",
)


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "npu":
        torch.npu.synchronize()


def reset_peak_memory_stats(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    elif device.type == "npu" and hasattr(torch.npu, "reset_peak_memory_stats"):
        torch.npu.reset_peak_memory_stats()


def peak_memory_mb(device: torch.device) -> float | None:
    if device.type == "cuda":
        return round(torch.cuda.max_memory_allocated() / 1024**2, 1)
    if device.type == "npu" and hasattr(torch.npu, "max_memory_allocated"):
        return round(torch.npu.max_memory_allocated() / 1024**2, 1)
    return None


def tokens_per_second(*, generated_tokens: int, latency_ms: float) -> float | None:
    if latency_ms <= 0:
        return None
    return round(generated_tokens / (latency_ms / 1000.0), 3)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def generation_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    allowed = {"input_ids", "attention_mask", "position_ids"}
    return {key: value.to(device) for key, value in batch.items() if key in allowed}


@torch.no_grad()
def run_torch_generate_benchmark(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: list[str],
    device: torch.device,
    max_new_tokens: int,
    iters: int,
    warmup: int,
) -> dict[str, Any]:
    tokenizer_padding_side = ensure_left_padding_for_generate(tokenizer)
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    )
    batch = generation_batch(encoded, device)
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }
    for _ in range(warmup):
        model.generate(**batch, **generation_kwargs)
    synchronize_device(device)
    start = time.perf_counter()
    output = None
    for _ in range(iters):
        output = model.generate(**batch, **generation_kwargs)
    synchronize_device(device)
    latency_ms = (time.perf_counter() - start) / iters * 1000.0
    prompt_tokens = int(batch["input_ids"].numel())
    generated_tokens = len(prompts) * max_new_tokens
    if isinstance(output, torch.Tensor):
        generated_tokens = max(0, int(output.numel() - prompt_tokens))
    return {
        "backend": "torch",
        "status": "PASS",
        "batch_size": len(prompts),
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "latency_ms": latency_ms,
        "tokens_per_s": tokens_per_second(generated_tokens=generated_tokens, latency_ms=latency_ms),
        "tokenizer_padding_side": tokenizer_padding_side,
    }


def count_vllm_output_tokens(outputs: Any) -> int:
    total = 0
    for request_output in outputs:
        for output in getattr(request_output, "outputs", []) or []:
            token_ids = getattr(output, "token_ids", None)
            if token_ids is not None:
                total += len(token_ids)
                continue
            text = getattr(output, "text", "")
            total += len(str(text).split())
    return total


def run_vllm_generate_benchmark(
    *,
    llm: Any,
    sampling_params: Any,
    prompts: list[str],
    iters: int,
    warmup: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        llm.generate(prompts, sampling_params)
    start = time.perf_counter()
    outputs = None
    for _ in range(iters):
        outputs = llm.generate(prompts, sampling_params)
    latency_ms = (time.perf_counter() - start) / iters * 1000.0
    generated_tokens = count_vllm_output_tokens(outputs or [])
    return {
        "backend": "vllm",
        "status": "PASS",
        "batch_size": len(prompts),
        "generated_tokens": generated_tokens,
        "latency_ms": latency_ms,
        "tokens_per_s": tokens_per_second(generated_tokens=generated_tokens, latency_ms=latency_ms),
    }


def build_missing_model_report(*, model_id: str, roots: list[Path], backend: str) -> dict[str, Any]:
    return {
        "status": "MODEL_MISSING",
        "model_id": model_id,
        "model_path": None,
        "backend": backend,
        "download_attempted": False,
        "searched_roots": [str(root) for root in roots],
    }


def markdown_report(report: dict[str, Any]) -> str:
    rows = [
        ("Status", report.get("status")),
        ("Model", report.get("model_id")),
        ("Model path", report.get("model_path")),
        ("Backend", report.get("backend")),
        ("Device", report.get("device")),
        ("Dtype", report.get("dtype")),
        ("Batch size", report.get("batch_size")),
        ("Prompt tokens", report.get("prompt_tokens")),
        ("Generated tokens", report.get("generated_tokens")),
        ("Latency ms", report.get("latency_ms")),
        ("Tokens/s", report.get("tokens_per_s")),
        ("Peak MB", report.get("peak_mem_mb")),
    ]
    lines = [
        "# Ascend Inference Benchmark",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in rows:
        if value is not None:
            lines.append(f"| {key} | {value} |")
    if report.get("status") == "MODEL_MISSING":
        lines.extend(
            [
                "",
                "The benchmark did not attempt a download. Place the model under one scanned root, then rerun this exact script.",
            ]
        )
    if report.get("error"):
        lines.extend(["", f"Error: `{report.get('error_type')}: {report.get('error')}`"])
    return "\n".join(lines) + "\n"


def write_report(report: dict[str, Any], demo_root: Path, *, run_label: str) -> None:
    artifacts = demo_root / "artifacts"
    reports = demo_root / "reports"
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    suffix = f"_{run_label}" if run_label else ""
    (artifacts / f"ascend_inference{suffix}.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    report_suffix = f"-{run_label}" if run_label else ""
    (reports / f"ascend-inference{report_suffix}.md").write_text(markdown_report(report))


def run_torch_backend(
    *,
    model_path: Path,
    model_id: str,
    device: torch.device,
    dtype: torch.dtype,
    prompts: list[str],
    max_new_tokens: int,
    iters: int,
    warmup: int,
) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        local_files_only=True,
    ).to(device)
    model.eval()
    reset_peak_memory_stats(device)
    result = run_torch_generate_benchmark(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        device=device,
        max_new_tokens=max_new_tokens,
        iters=iters,
        warmup=warmup,
    )
    result.update(
        {
            "model_id": model_id,
            "model_path": str(model_path),
            "device": str(device),
            "dtype": str(dtype),
            "peak_mem_mb": peak_memory_mb(device),
        }
    )
    return result


def run_vllm_backend(
    *,
    model_path: Path,
    model_id: str,
    dtype: str,
    prompts: list[str],
    max_new_tokens: int,
    iters: int,
    warmup: int,
) -> dict[str, Any]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(model_path),
        tokenizer=str(model_path),
        dtype=dtype,
        trust_remote_code=True,
        max_model_len=512,
    )
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,
    )
    result = run_vllm_generate_benchmark(
        llm=llm,
        sampling_params=sampling_params,
        prompts=prompts,
        iters=iters,
        warmup=warmup,
    )
    result.update(
        {
            "model_id": model_id,
            "model_path": str(model_path),
            "dtype": dtype,
        }
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Ascend inference benchmark")
    parser.add_argument("--demo-root", default="/mnt/nvme/622/tidal-demo")
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--root", action="append", dest="roots")
    parser.add_argument("--backend", choices=["torch", "vllm"], default="torch")
    parser.add_argument("--device", default="npu")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--run-label", default="qwen3_torch")
    args = parser.parse_args()

    roots = [Path(root) for root in args.roots] if args.roots else default_roots()
    model_path = Path(args.model_path) if args.model_path else find_model_path(args.model_id, roots)[0]
    if model_path is None:
        report = build_missing_model_report(model_id=args.model_id, roots=roots, backend=args.backend)
        write_report(report, Path(args.demo_root), run_label=args.run_label)
        print("ASCEND_INFERENCE_BENCH " + json.dumps(report, sort_keys=True))
        return 2

    try:
        if args.backend == "torch":
            device = resolve_device(args.device)
            dtype = resolve_dtype(args.dtype, device)
            torch_dtype = dtype if isinstance(dtype, torch.dtype) else torch.float16
            report = run_torch_backend(
                model_path=model_path,
                model_id=args.model_id,
                device=device,
                dtype=torch_dtype,
                prompts=list(DEFAULT_PROMPTS),
                max_new_tokens=args.max_new_tokens,
                iters=args.iters,
                warmup=args.warmup,
            )
        else:
            report = run_vllm_backend(
                model_path=model_path,
                model_id=args.model_id,
                dtype=args.dtype,
                prompts=list(DEFAULT_PROMPTS),
                max_new_tokens=args.max_new_tokens,
                iters=args.iters,
                warmup=args.warmup,
            )
    except Exception as exc:
        report = {
            "status": "BACKEND_ERROR",
            "model_id": args.model_id,
            "model_path": str(model_path),
            "backend": args.backend,
            "error_type": type(exc).__name__,
            "error": str(exc)[:1000],
        }
    report.update({"platform": platform.platform(), "download_attempted": False})
    write_report(report, Path(args.demo_root), run_label=args.run_label)
    print("ASCEND_INFERENCE_BENCH " + json.dumps(report, sort_keys=True))
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
