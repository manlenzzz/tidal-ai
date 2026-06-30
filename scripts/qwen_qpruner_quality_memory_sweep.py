#!/usr/bin/env python
"""QPruner-only Qwen quality-memory sweep for demo frontier artifacts."""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from ascend_inference_benchmark import peak_memory_mb, reset_peak_memory_stats, synchronize_device
from qwen_compression_generate_benchmark import select_target_names
from qwen_compression_quality_benchmark import (
    build_eval_batch,
    evaluate_quality_model,
    fmt,
    markdown_report,
    qpruner_quality_result,
)
from tiny_qwen_compression_benchmark import load_model, load_tokenizer, targeted_parameter_count
from tidal.device import resolve_device, resolve_dtype


DENSE_WEIGHT_BITS = 16.0


def parse_float_csv(value: str) -> list[float]:
    items = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("provide at least one float value")
    return items


def parse_int_csv(value: str) -> list[int]:
    items = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("provide at least one integer value")
    return items


def fmt_pct(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, float):
        return f"{value:.3f}%"
    return f"{value}%"


def fmt_bits_word(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "missing-bit"
    if number.is_integer():
        return f"{int(number)}-bit"
    return f"{number:.3f}-bit"


def memory_reduction_pct(point: dict[str, Any]) -> float | None:
    qpruner = point.get("qpruner", {}) if isinstance(point.get("qpruner"), dict) else {}
    average_bits = qpruner.get("average_bits")
    if average_bits is not None:
        try:
            return round((1.0 - float(average_bits) / DENSE_WEIGHT_BITS) * 100.0, 3)
        except (TypeError, ValueError, ZeroDivisionError):
            return None
    memory_bits = qpruner.get("memory_bits")
    baseline_params = point.get("baseline", {}).get("targeted_params")
    if memory_bits is None or baseline_params is None:
        return None
    try:
        dense_bits = float(baseline_params) * DENSE_WEIGHT_BITS
        return round((1.0 - float(memory_bits) / dense_bits) * 100.0, 3)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def bitwidth_mix(point: dict[str, Any]) -> str:
    qpruner = point.get("qpruner", {}) if isinstance(point.get("qpruner"), dict) else {}
    bitwidths = qpruner.get("bitwidths") if isinstance(qpruner.get("bitwidths"), dict) else {}
    counts: Counter[int] = Counter()
    order: list[int] = []
    for value in bitwidths.values():
        bit = int(value)
        if bit not in counts:
            order.append(bit)
        counts[bit] += 1
    if not counts:
        return "missing"
    return ", ".join(f"{bit}-bit:{counts[bit]}" for bit in order)


def passing_points(report: dict[str, Any]) -> list[dict[str, Any]]:
    points = report.get("points", [])
    if not isinstance(points, list):
        return []
    return [
        point
        for point in points
        if isinstance(point, dict)
        and point.get("status") == "PASS"
        and point.get("qpruner", {}).get("status", "PASS") == "PASS"
    ]


def best_memory_quality_point(report: dict[str, Any], *, max_loss_delta: float) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for point in passing_points(report):
        qpruner = point.get("qpruner", {})
        loss_delta = qpruner.get("loss_delta")
        reduction = memory_reduction_pct(point)
        if loss_delta is None or reduction is None:
            continue
        if float(loss_delta) <= max_loss_delta:
            candidates.append(point)
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda point: (
            memory_reduction_pct(point) or -1.0,
            -(float(point.get("qpruner", {}).get("loss_delta") or 0.0)),
        ),
    )


def frontier_note(report: dict[str, Any], *, max_loss_delta: float) -> str:
    points = passing_points(report)
    if not points:
        return "No passing QPruner sweep points were produced."
    best = best_memory_quality_point(report, max_loss_delta=max_loss_delta)
    notes: list[str] = []
    if best is None:
        notes.append(f"No QPruner point met loss delta <= {max_loss_delta:.3f}.")
    else:
        qpruner = best.get("qpruner", {})
        notes.append(
            "Best memory-quality point under loss delta <= {threshold} is {label}: "
            "{reduction} targeted memory reduction, loss delta {delta}, average bits {bits}.".format(
                threshold=fmt(max_loss_delta),
                label=best.get("run_label", "unknown"),
                reduction=fmt_pct(memory_reduction_pct(best)),
                delta=fmt(qpruner.get("loss_delta")),
                bits=fmt(qpruner.get("average_bits")),
            )
        )
    lowest_bits = min(
        points,
        key=lambda point: float(point.get("qpruner", {}).get("average_bits") or DENSE_WEIGHT_BITS),
    )
    qpruner = lowest_bits.get("qpruner", {})
    if qpruner.get("loss_delta") is not None and float(qpruner["loss_delta"]) > max_loss_delta:
        notes.append(
            "{bits} saves the most memory but exceeds the quality threshold with loss delta {delta}.".format(
                bits=fmt_bits_word(qpruner.get("average_bits")),
                delta=fmt(qpruner.get("loss_delta")),
            )
        )
    notes.append(
        "This QPruner-only runner reuses one baseline evaluation and skips CAP/WANDA for faster Ascend sweeps."
    )
    return " ".join(notes)


def aggregate_markdown(report: dict[str, Any], *, max_loss_delta: float = 0.5) -> str:
    lines = [
        "# Qwen QPruner Quality-Memory Sweep",
        "",
        "| Label | Status | Target layers | Requested bits | Actual bits | Target memory reduction | Loss delta | Tokens/s | Bitwidth mix |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for point in sorted(
        report.get("points", []),
        key=lambda item: (
            int(item.get("target_layer_limit") or 0),
            float(item.get("qpruner_average_bits_request") or 0.0),
            str(item.get("run_label") or ""),
        ),
    ):
        qpruner = point.get("qpruner", {}) if isinstance(point.get("qpruner"), dict) else {}
        lines.append(
            "| {label} | {status} | {limit} / {total} | {requested} | {actual} | {reduction} | {delta} | {tokens} | {mix} |".format(
                label=point.get("run_label", "unknown"),
                status=point.get("status", "missing"),
                limit=point.get("target_layer_limit", "missing"),
                total=point.get("targeted_layers_total", "missing"),
                requested=fmt(point.get("qpruner_average_bits_request")),
                actual=fmt(qpruner.get("average_bits")),
                reduction=fmt_pct(memory_reduction_pct(point)),
                delta=fmt(qpruner.get("loss_delta")),
                tokens=fmt(qpruner.get("tokens_per_s")),
                mix=bitwidth_mix(point),
            )
        )
    lines.extend(
        [
            "",
            "## Frontier Read",
            "",
            frontier_note(report, max_loss_delta=max_loss_delta),
            "",
            "## Metadata",
            "",
            f"- Model: `{report.get('model_id')}`",
            f"- Model path: `{report.get('model_path')}`",
            f"- Device: `{report.get('device')}`",
            f"- Dtype: `{report.get('dtype')}`",
            f"- Target layer pattern: `{report.get('target_layer_pattern')}`",
            f"- Target layer limits: `{report.get('target_layer_limits')}`",
            f"- QPruner requested bits: `{report.get('qpruner_average_bits')}`",
            f"- Peak MB: `{fmt(report.get('peak_mem_mb'))}`",
        ]
    )
    return "\n".join(lines) + "\n"


def aggregate_csv(report: dict[str, Any]) -> str:
    from io import StringIO

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "label",
            "status",
            "target_layer_limit",
            "targeted_layers_total",
            "requested_bits",
            "actual_bits",
            "target_memory_reduction_pct",
            "loss_delta",
            "tokens_per_s",
            "bitwidth_mix",
        ]
    )
    for point in sorted(
        report.get("points", []),
        key=lambda item: (
            int(item.get("target_layer_limit") or 0),
            float(item.get("qpruner_average_bits_request") or 0.0),
            str(item.get("run_label") or ""),
        ),
    ):
        qpruner = point.get("qpruner", {}) if isinstance(point.get("qpruner"), dict) else {}
        writer.writerow(
            [
                point.get("run_label", "unknown"),
                point.get("status", "missing"),
                point.get("target_layer_limit", "missing"),
                point.get("targeted_layers_total", "missing"),
                point.get("qpruner_average_bits_request", "missing"),
                qpruner.get("average_bits", "missing"),
                memory_reduction_pct(point),
                qpruner.get("loss_delta", "missing"),
                qpruner.get("tokens_per_s", "missing"),
                bitwidth_mix(point),
            ]
        )
    return output.getvalue()


def write_point_outputs(point: dict[str, Any], demo_root: Path) -> None:
    artifacts = demo_root / "artifacts"
    reports = demo_root / "reports"
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    run_label = point["run_label"]
    (artifacts / f"qwen_compression_quality_{run_label}.json").write_text(
        json.dumps(point, indent=2, sort_keys=True)
    )
    (reports / f"qwen-compression-quality-{run_label}.md").write_text(markdown_report(point))


def write_aggregate_outputs(report: dict[str, Any], demo_root: Path, run_label: str, *, max_loss_delta: float) -> None:
    artifacts = demo_root / "artifacts"
    reports = demo_root / "reports"
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    (artifacts / f"qwen_qpruner_quality_memory_sweep_{run_label}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    text = aggregate_markdown(report, max_loss_delta=max_loss_delta)
    csv_text = aggregate_csv(report)
    (reports / f"qwen-qpruner-quality-memory-sweep-{run_label}.md").write_text(text)
    (reports / f"qwen-qpruner-quality-memory-sweep-{run_label}.csv").write_text(csv_text)
    (reports / "qwen-qpruner-quality-memory-sweep.md").write_text(text)
    (reports / "qwen-qpruner-quality-memory-sweep.csv").write_text(csv_text)


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    torch_dtype = dtype if isinstance(dtype, torch.dtype) else torch.float32
    model_path = Path(args.model_path)
    tokenizer = load_tokenizer(model_path)
    batch = build_eval_batch(tokenizer=tokenizer, max_length=args.max_length, device=device)

    reset_peak_memory_stats(device)
    baseline_model = load_model(model_path, device, torch_dtype)
    baseline_metrics = evaluate_quality_model(
        baseline_model,
        batch,
        iters=args.iters,
        warmup=args.warmup,
        device=device,
    )

    points: list[dict[str, Any]] = []
    for layer_limit in args.target_layer_limits:
        selected_target_names, total_targeted_layers = select_target_names(
            baseline_model,
            layer_limit,
            args.target_layer_pattern,
        )
        if not selected_target_names:
            raise ValueError("model does not contain matching Qwen projection layers")
        baseline_targeted_params = targeted_parameter_count(baseline_model, selected_target_names)
        baseline = deepcopy(baseline_metrics)
        baseline["targeted_layers"] = len(selected_target_names)
        baseline["targeted_params"] = baseline_targeted_params
        for average_bits in args.qpruner_average_bits:
            run_label = f"{args.run_label}_layers{layer_limit}_bits{int(average_bits) if average_bits.is_integer() else average_bits}"
            point_args = argparse.Namespace(**vars(args))
            point_args.qpruner_average_bits = average_bits
            qpruner = qpruner_quality_result(
                model=load_model(model_path, device, torch_dtype),
                model_id=args.model_id,
                baseline_loss=float(baseline["loss"]),
                baseline_latency_ms=float(baseline["latency_ms"]),
                baseline_targeted_params=baseline_targeted_params,
                target_names=selected_target_names,
                batch=batch,
                device=device,
                dtype=torch_dtype,
                args=point_args,
            )
            point = {
                "status": "PASS",
                "backend": "torch_forward",
                "model_id": args.model_id,
                "model_path": str(model_path),
                "device": str(device),
                "dtype": str(torch_dtype),
                "download_attempted": False,
                "inference_cache_enabled": bool(args.enable_inference_cache),
                "inplace_compression": bool(args.inplace_compression),
                "target_layer_pattern": args.target_layer_pattern,
                "target_layer_limit": layer_limit,
                "targeted_layers_total": total_targeted_layers,
                "targeted_layer_names_sample": selected_target_names[:8],
                "max_length": args.max_length,
                "qpruner_average_bits_request": average_bits,
                "run_label": run_label,
                "baseline": baseline,
                "cap": {"status": "SKIPPED", "reason": "qpruner-only sweep"},
                "wanda": {"status": "SKIPPED", "reason": "qpruner-only sweep"},
                "qpruner": qpruner,
                "platform": platform.platform(),
            }
            write_point_outputs(point, Path(args.demo_root))
            points.append(point)
    synchronize_device(device)
    return {
        "status": "PASS",
        "backend": "torch_forward",
        "model_id": args.model_id,
        "model_path": str(model_path),
        "device": str(device),
        "dtype": str(torch_dtype),
        "target_layer_pattern": args.target_layer_pattern,
        "target_layer_limits": args.target_layer_limits,
        "qpruner_average_bits": args.qpruner_average_bits,
        "max_length": args.max_length,
        "points": points,
        "peak_mem_mb": peak_memory_mb(device),
        "platform": platform.platform(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QPruner-only Qwen quality-memory sweep")
    parser.add_argument("--demo-root", default="/mnt/nvme/622/tidal-demo")
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--model-path", default="/mnt/nvme/622/models/Qwen3-0.6B")
    parser.add_argument("--device", default="npu")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--target-layer-limits", type=parse_int_csv, default=[4])
    parser.add_argument("--target-layer-pattern", default="self_attn\\.(q_proj|k_proj)$")
    parser.add_argument("--qpruner-average-bits", type=parse_float_csv, default=[4.0, 6.0, 8.0])
    parser.add_argument("--enable-inference-cache", action="store_true")
    parser.add_argument("--inplace-compression", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-loss-delta", type=float, default=0.5)
    parser.add_argument("--run-label", default="qwen3_06b_qpruner_quality_memory_sweep_npu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = run_sweep(args)
    except Exception as exc:
        report = {
            "status": "FAIL",
            "backend": "torch_forward",
            "model_id": args.model_id,
            "model_path": args.model_path,
            "device": args.device,
            "target_layer_limits": args.target_layer_limits,
            "qpruner_average_bits": args.qpruner_average_bits,
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
            "platform": platform.platform(),
            "points": [],
        }
    write_aggregate_outputs(report, Path(args.demo_root), args.run_label, max_loss_delta=args.max_loss_delta)
    print("QWEN_QPRUNER_QUALITY_MEMORY_SWEEP " + json.dumps(report, sort_keys=True))
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
