#!/usr/bin/env python
"""Benchmark Transformers torch_generate on CAP/QPruner serving exports."""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from ascend_inference_benchmark import (  # noqa: E402
    DEFAULT_PROMPTS,
    generation_batch,
    peak_memory_mb,
    reset_peak_memory_stats,
    synchronize_device,
    tokens_per_second,
)
from tidal.device import resolve_device, resolve_dtype  # noqa: E402


METHODS = ("baseline", "cap", "qpruner")


def export_path(demo_root: Path, export_run_label: str, method: str) -> Path:
    return demo_root / "serving_exports" / export_run_label / method


def fmt(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def load_status(path: Path, *, dtype: torch.dtype) -> str:
    try:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        AutoConfig.from_pretrained(str(path), local_files_only=True)
        AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        AutoModelForCausalLM.from_pretrained(str(path), torch_dtype=dtype, local_files_only=True)
        return "PASS"
    except Exception:
        return "FAIL"


@torch.no_grad()
def benchmark_export(
    *,
    path: Path,
    method: str,
    device: torch.device,
    dtype: torch.dtype,
    prompts: Sequence[str],
    max_new_tokens: int,
    iters: int,
    warmup: int,
) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not path.exists():
        return {
            "status": "MISSING",
            "exists": False,
            "path": str(path),
            "transformers_load": "MISSING",
        }

    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(path),
        torch_dtype=dtype,
        local_files_only=True,
    ).to(device)
    model.eval()
    encoded = tokenizer(
        list(prompts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    )
    batch = generation_batch(encoded, device)
    generation_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False}
    reset_peak_memory_stats(device)
    for _ in range(warmup):
        model.generate(**batch, **generation_kwargs)
    synchronize_device(device)
    start = time.perf_counter()
    output = None
    for _ in range(iters):
        output = model.generate(**batch, **generation_kwargs)
    synchronize_device(device)
    latency_ms = (time.perf_counter() - start) / max(1, iters) * 1000.0
    prompt_tokens = int(batch["input_ids"].numel())
    generated_tokens = len(prompts) * max_new_tokens
    if isinstance(output, torch.Tensor):
        generated_tokens = max(0, int(output.numel() - prompt_tokens))
    return {
        "status": "PASS",
        "method": method,
        "exists": True,
        "path": str(path),
        "transformers_load": "PASS",
        "batch_size": len(prompts),
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "latency_ms": latency_ms,
        "tokens_per_s": tokens_per_second(generated_tokens=generated_tokens, latency_ms=latency_ms),
        "peak_mem_mb": peak_memory_mb(device),
    }


def missing_report(*, demo_root: Path, export_run_label: str, run_label: str, device: str, dtype: str) -> dict[str, Any]:
    exports = {
        method: {
            "status": "MISSING",
            "exists": False,
            "path": str(export_path(demo_root, export_run_label, method)),
            "transformers_load": "MISSING",
        }
        for method in METHODS
    }
    return {
        "status": "MODEL_MISSING",
        "run_label": run_label,
        "export_run_label": export_run_label,
        "backend": "torch_generate",
        "device": device,
        "dtype": dtype,
        "exports": exports,
        "summary": {
            "best_method": None,
            "best_tokens_per_s": None,
            "cap_vs_baseline_speedup": None,
            "qpruner_vs_baseline_speedup": None,
            "cap_vs_qpruner_speedup": None,
        },
        "download_attempted": False,
        "platform": platform.platform(),
    }


def _speedup(numerator: Any, denominator: Any) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return round(float(numerator) / float(denominator), 3)


def summarize(exports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    passing = {
        method: payload
        for method, payload in exports.items()
        if payload.get("status") == "PASS" and payload.get("tokens_per_s") is not None
    }
    if not passing:
        return {
            "best_method": None,
            "best_tokens_per_s": None,
            "cap_vs_baseline_speedup": None,
            "qpruner_vs_baseline_speedup": None,
            "cap_vs_qpruner_speedup": None,
        }
    best_method, best_payload = max(passing.items(), key=lambda item: float(item[1]["tokens_per_s"]))
    baseline_tps = exports.get("baseline", {}).get("tokens_per_s")
    cap_tps = exports.get("cap", {}).get("tokens_per_s")
    qpruner_tps = exports.get("qpruner", {}).get("tokens_per_s")
    return {
        "best_method": best_method,
        "best_tokens_per_s": best_payload.get("tokens_per_s"),
        "cap_vs_baseline_speedup": _speedup(cap_tps, baseline_tps),
        "qpruner_vs_baseline_speedup": _speedup(qpruner_tps, baseline_tps),
        "cap_vs_qpruner_speedup": _speedup(cap_tps, qpruner_tps),
    }


def run_benchmark(
    *,
    demo_root: Path,
    export_run_label: str,
    run_label: str,
    device_name: str,
    dtype_name: str,
    max_new_tokens: int,
    iters: int,
    warmup: int,
) -> dict[str, Any]:
    paths = {method: export_path(demo_root, export_run_label, method) for method in METHODS}
    if not any(path.exists() for path in paths.values()):
        return missing_report(
            demo_root=demo_root,
            export_run_label=export_run_label,
            run_label=run_label,
            device=device_name,
            dtype=dtype_name,
        )

    device = resolve_device(device_name)
    dtype = resolve_dtype(dtype_name, device)
    torch_dtype = dtype if isinstance(dtype, torch.dtype) else torch.float32
    exports = {
        method: benchmark_export(
            path=path,
            method=method,
            device=device,
            dtype=torch_dtype,
            prompts=DEFAULT_PROMPTS,
            max_new_tokens=max_new_tokens,
            iters=iters,
            warmup=warmup,
        )
        for method, path in paths.items()
    }
    status = "PASS" if all(payload.get("status") == "PASS" for payload in exports.values()) else "FAIL"
    return {
        "status": status,
        "run_label": run_label,
        "export_run_label": export_run_label,
        "backend": "torch_generate",
        "device": str(device),
        "dtype": str(torch_dtype),
        "max_new_tokens": max_new_tokens,
        "iters": iters,
        "warmup": warmup,
        "exports": exports,
        "summary": summarize(exports),
        "download_attempted": False,
        "platform": platform.platform(),
    }


def row(method: str, payload: dict[str, Any]) -> str:
    return "| {method} | {status} | {exists} | {load} | {latency} | {tps} | {peak} |".format(
        method=method,
        status=payload.get("status", "missing"),
        exists=payload.get("exists", "missing"),
        load=payload.get("transformers_load", "missing"),
        latency=payload.get("latency_ms", "missing"),
        tps=payload.get("tokens_per_s", "missing"),
        peak=payload.get("peak_mem_mb", "missing"),
    )


def markdown_report(report: dict[str, Any]) -> str:
    exports = report.get("exports", {})
    summary = report.get("summary", {})
    lines = [
        "# Torch Serving Fallback Benchmark",
        "",
        "Transformers torch_generate benchmark for dense HF serving exports. This is the runnable fallback when vLLM-Ascend is blocked by container/API compatibility.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Status | {report.get('status')} |",
        f"| Backend | {report.get('backend')} |",
        f"| Export run label | {report.get('export_run_label')} |",
        f"| Device | {report.get('device')} |",
        f"| Dtype | {report.get('dtype')} |",
        f"| best_method | {summary.get('best_method')} |",
        f"| best_tokens_per_s | {summary.get('best_tokens_per_s')} |",
        f"| cap_vs_baseline_speedup | {summary.get('cap_vs_baseline_speedup')} |",
        f"| qpruner_vs_baseline_speedup | {summary.get('qpruner_vs_baseline_speedup')} |",
        f"| cap_vs_qpruner_speedup | {summary.get('cap_vs_qpruner_speedup')} |",
        "",
        "| Method | Status | Exists | Transformers load | Latency ms | Tokens/s | Peak MB |",
        "|---|---:|---:|---:|---:|---:|---:|",
        row("Baseline", exports.get("baseline", {})),
        row("CAP", exports.get("cap", {})),
        row("QPruner", exports.get("qpruner", {})),
    ]
    return "\n".join(lines) + "\n"


def write_outputs(report: dict[str, Any], demo_root: Path, run_label: str) -> None:
    artifacts = demo_root / "artifacts"
    reports = demo_root / "reports"
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    (artifacts / f"torch_serving_fallback_{run_label}.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    (reports / f"torch-serving-fallback-{run_label}.md").write_text(markdown_report(report))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark torch_generate fallback on serving exports")
    parser.add_argument("--demo-root", default="/mnt/nvme/622/tidal-demo")
    parser.add_argument("--export-run-label", default="tiny_qwen3_serving_export_npu")
    parser.add_argument("--run-label", default="tiny_qwen3_serving_fallback_npu")
    parser.add_argument("--device", default="npu")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_benchmark(
        demo_root=Path(args.demo_root),
        export_run_label=args.export_run_label,
        run_label=args.run_label,
        device_name=args.device,
        dtype_name=args.dtype,
        max_new_tokens=args.max_new_tokens,
        iters=args.iters,
        warmup=args.warmup,
    )
    write_outputs(report, Path(args.demo_root), args.run_label)
    print("TORCH_SERVING_FALLBACK " + json.dumps(report, sort_keys=True))
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
