#!/usr/bin/env python
"""Export TinyQwen CAP/QPruner compressed models as serving-compatible HF directories."""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from tiny_qwen_compression_benchmark import (  # noqa: E402
    load_model,
    load_tokenizer,
    qpruner_average_bits,
    qwen_projection_filter,
    target_linear_names,
    targeted_parameter_count,
)
from tidal.device import resolve_device, resolve_dtype  # noqa: E402
from tidal.workflows.compression import cap_compress, qpruner_compress  # noqa: E402
from tidal.workflows.export import export_compressed_linears_to_dense  # noqa: E402


def dense_linear_count(model: nn.Module) -> int:
    return sum(isinstance(module, nn.Linear) for module in model.modules())


def export_dir(demo_root: Path, run_label: str, method: str) -> Path:
    return demo_root / "serving_exports" / run_label / method


def save_hf_export(*, model: nn.Module, tokenizer: Any, output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))


def load_check(output_dir: Path, *, dtype: torch.dtype) -> str:
    try:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            str(output_dir),
            local_files_only=True,
            torch_dtype=dtype,
        )
        model.eval()
        return "PASS"
    except Exception:
        return "FAIL"


def baseline_export(
    *,
    model: nn.Module,
    tokenizer: Any,
    baseline_target_names: list[str],
    baseline_targeted_params: int,
    demo_root: Path,
    run_label: str,
) -> dict[str, Any]:
    output_dir = export_dir(demo_root, run_label, "baseline")
    save_hf_export(model=model.cpu(), tokenizer=tokenizer, output_dir=output_dir)
    return {
        "status": "PASS",
        "path": str(output_dir),
        "targeted_layers": len(baseline_target_names),
        "targeted_params_original": baseline_targeted_params,
        "dense_linears": dense_linear_count(model),
        "load_check": load_check(output_dir, dtype=torch.float32),
    }


def cap_export(
    *,
    model: nn.Module,
    model_id: str,
    tokenizer: Any,
    baseline_targeted_params: int,
    demo_root: Path,
    run_label: str,
    device: torch.device,
    dtype: torch.dtype,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.perf_counter()
    result = cap_compress(
        model=model,
        model_id=model_id,
        budget=args.cap_budget,
        target_roles=None,
        name_filter=qwen_projection_filter,
        device=device,
        dtype=dtype,
        max_iter=args.cap_max_iter,
        policy_steps=args.cap_policy_steps,
        samples_per_step=args.cap_samples_per_step,
        seed=args.seed,
        rpca_backend=args.cap_rpca_backend,
    )
    target_names = [target.name for target in result.method_result.targets]
    compressed_params = targeted_parameter_count(result.model, target_names)
    before_dense = dense_linear_count(result.model)
    dense_model = export_compressed_linears_to_dense(result.model, inplace=False, dtype=dtype, device=device)
    exported_dense_linears = dense_linear_count(dense_model) - before_dense
    output_dir = export_dir(demo_root, run_label, "cap")
    save_hf_export(model=dense_model.cpu(), tokenizer=tokenizer, output_dir=output_dir)
    return {
        "status": "PASS",
        "path": str(output_dir),
        "targeted_layers": len(target_names),
        "targeted_compression_ratio": round(baseline_targeted_params / compressed_params, 3)
        if compressed_params
        else None,
        "exported_dense_linears": exported_dense_linears,
        "compression_time_s": time.perf_counter() - started,
        "load_check": load_check(output_dir, dtype=torch.float32),
    }


def qpruner_export(
    *,
    model: nn.Module,
    model_id: str,
    tokenizer: Any,
    baseline_target_names: list[str],
    baseline_targeted_params: int,
    demo_root: Path,
    run_label: str,
    device: torch.device,
    dtype: torch.dtype,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.perf_counter()
    result = qpruner_compress(
        model=model,
        model_id=model_id,
        importances={name: 1.0 for name in baseline_target_names},
        candidate_bits=(2, 4, 8),
        max_average_bits=args.qpruner_average_bits,
        target_roles=None,
        name_filter=qwen_projection_filter,
        device=device,
        dtype=dtype,
        seed=args.seed,
    )
    before_dense = dense_linear_count(result.model)
    dense_model = export_compressed_linears_to_dense(result.model, inplace=False, dtype=dtype, device=device)
    exported_dense_linears = dense_linear_count(dense_model) - before_dense
    output_dir = export_dir(demo_root, run_label, "qpruner")
    save_hf_export(model=dense_model.cpu(), tokenizer=tokenizer, output_dir=output_dir)
    return {
        "status": "PASS",
        "path": str(output_dir),
        "targeted_layers": len(baseline_target_names),
        "targeted_params_original": baseline_targeted_params,
        "average_bits": qpruner_average_bits(result.model, baseline_target_names),
        "exported_dense_linears": exported_dense_linears,
        "compression_time_s": time.perf_counter() - started,
        "load_check": load_check(output_dir, dtype=torch.float32),
    }


def compression_metric(method: str, payload: dict[str, Any]) -> str:
    if method == "Baseline":
        return "1.0x"
    if method == "CAP":
        value = payload.get("targeted_compression_ratio")
        return f"{value}x" if value is not None else "missing"
    value = payload.get("average_bits")
    return f"{value} avg bits" if value is not None else "missing"


def markdown_report(report: dict[str, Any]) -> str:
    exports = report.get("exports", {})
    baseline = exports.get("baseline", {})
    cap = exports.get("cap", {})
    qpruner = exports.get("qpruner", {})
    rows = [
        ("Baseline", baseline),
        ("CAP", cap),
        ("QPruner", qpruner),
    ]
    lines = [
        "# TinyQwen Compressed Serving Export",
        "",
        "Serving-compatible dense export for vLLM-Ascend or other HF-loading runtimes. This preserves the compressed model approximation but does not preserve compressed storage in the exported directories.",
        "",
        "| Method | Status | Targeted layers | Compression | Dense linears | Load check |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, payload in rows:
        dense_linears = payload.get("exported_dense_linears", payload.get("dense_linears", "missing"))
        lines.append(
            "| {method} | {status} | {layers} | {compression} | {dense} | {load} |".format(
                method=method,
                status=payload.get("status", "missing"),
                layers=payload.get("targeted_layers", "missing"),
                compression=compression_metric(method, payload),
                dense=dense_linears,
                load=payload.get("load_check", "missing"),
            )
        )
    lines.extend(
        [
            "",
            "## Output Paths",
            "",
            f"- CAP: `{cap.get('path', 'missing')}`",
            f"- QPruner: `{qpruner.get('path', 'missing')}`",
            f"- Baseline: `{baseline.get('path', 'missing')}`",
            "",
            "## Metadata",
            "",
            f"- Status: `{report.get('status')}`",
            f"- Model: `{report.get('model_id')}`",
            f"- Device: `{report.get('device')}`",
            f"- Dtype: `{report.get('dtype')}`",
            f"- Note: `{report.get('storage_note')}`",
        ]
    )
    if report.get("error"):
        lines.extend(["", f"Error: `{report.get('error_type')}: {report.get('error')}`"])
    return "\n".join(lines) + "\n"


def write_outputs(report: dict[str, Any], demo_root: Path, run_label: str) -> None:
    artifacts = demo_root / "artifacts"
    reports = demo_root / "reports"
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    (artifacts / f"tiny_qwen_serving_export_{run_label}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    (reports / f"tiny-qwen-serving-export-{run_label}.md").write_text(markdown_report(report))


def run_export(args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    torch_dtype = dtype if isinstance(dtype, torch.dtype) else torch.float32
    demo_root = Path(args.demo_root)
    model_path = Path(args.model_path)
    tokenizer = load_tokenizer(model_path)
    baseline_model = load_model(model_path, device, torch_dtype)
    baseline_target_names = target_linear_names(baseline_model)
    baseline_targeted_params = targeted_parameter_count(baseline_model, baseline_target_names)
    baseline = baseline_export(
        model=load_model(model_path, device, torch_dtype),
        tokenizer=tokenizer,
        baseline_target_names=baseline_target_names,
        baseline_targeted_params=baseline_targeted_params,
        demo_root=demo_root,
        run_label=args.run_label,
    )

    cap = cap_export(
        model=load_model(model_path, device, torch_dtype),
        model_id=args.model_id,
        tokenizer=tokenizer,
        baseline_targeted_params=baseline_targeted_params,
        demo_root=demo_root,
        run_label=args.run_label,
        device=device,
        dtype=torch_dtype,
        args=args,
    )
    qpruner = qpruner_export(
        model=load_model(model_path, device, torch_dtype),
        model_id=args.model_id,
        tokenizer=tokenizer,
        baseline_target_names=baseline_target_names,
        baseline_targeted_params=baseline_targeted_params,
        demo_root=demo_root,
        run_label=args.run_label,
        device=device,
        dtype=torch_dtype,
        args=args,
    )
    return {
        "status": "PASS",
        "model_id": args.model_id,
        "model_path": str(model_path),
        "device": str(device),
        "dtype": str(torch_dtype),
        "download_attempted": False,
        "storage_note": "Serving dense export preserves approximate weights but does not preserve compressed storage.",
        "exports": {"baseline": baseline, "cap": cap, "qpruner": qpruner},
        "platform": platform.platform(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Export compressed TinyQwen models for serving runtimes")
    parser.add_argument("--demo-root", default="/mnt/nvme/622/tidal-demo")
    parser.add_argument("--model-id", default="TinyQwen3-Offline")
    parser.add_argument("--model-path", default="/mnt/nvme/622/models/TinyQwen3-Offline")
    parser.add_argument("--device", default="npu")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--cap-budget", type=int, default=8192)
    parser.add_argument("--cap-max-iter", type=int, default=4)
    parser.add_argument("--cap-policy-steps", type=int, default=1)
    parser.add_argument("--cap-samples-per-step", type=int, default=1)
    parser.add_argument("--cap-rpca-backend", default="numpy")
    parser.add_argument("--qpruner-average-bits", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-label", default="tiny_qwen3_serving_export_npu")
    args = parser.parse_args()

    try:
        report = run_export(args)
    except Exception as exc:
        report = {
            "status": "FAIL",
            "model_id": args.model_id,
            "model_path": args.model_path,
            "device": args.device,
            "download_attempted": False,
            "error_type": type(exc).__name__,
            "error": str(exc)[:1000],
            "platform": platform.platform(),
        }
    write_outputs(report, Path(args.demo_root), args.run_label)
    print("TINY_QWEN_SERVING_EXPORT " + json.dumps(report, sort_keys=True))
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
