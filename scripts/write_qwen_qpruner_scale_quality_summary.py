#!/usr/bin/env python
"""Write a memory-first QPruner scale/quality summary from existing sweep artifacts."""
from __future__ import annotations

import argparse
import csv
import json
import math
from io import StringIO
from pathlib import Path
from typing import Any, Sequence


DENSE_WEIGHT_BITS = 16.0
DEFAULT_ARTIFACT_GLOB = "qwen_compression_quality_qwen3_06b_qpruner_sweep*bits*.json"
SUMMARY_JSON = "qwen_qpruner_scale_quality_summary.json"
SUMMARY_MARKDOWN = "qwen-qpruner-scale-quality-summary.md"
SUMMARY_CSV = "qwen-qpruner-scale-quality-summary.csv"


def fmt(value: Any, suffix: str = "") -> str:
    if value is None:
        return "missing"
    if isinstance(value, float):
        return f"{value:.3f}{suffix}"
    return f"{value}{suffix}"


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_exp(value: Any) -> float | None:
    number = _safe_float(value)
    if number is None or number > 700:
        return None
    return round(math.exp(number), 3)


def _round_pct(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 3)


def normalize_label(report: dict[str, Any], path: Path) -> str:
    label = str(report.get("run_label") or path.stem)
    prefix = "qwen_compression_quality_"
    if label.startswith(prefix):
        label = label[len(prefix) :]
    return label


def read_point(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text())
    report["_path"] = str(path)
    report["_label"] = normalize_label(report, path)
    return report


def _target_layer_limit(report: dict[str, Any]) -> int | None:
    qpruner = report.get("qpruner", {}) if isinstance(report.get("qpruner"), dict) else {}
    baseline = report.get("baseline", {}) if isinstance(report.get("baseline"), dict) else {}
    return _safe_int(qpruner.get("targeted_layers") or baseline.get("targeted_layers") or report.get("target_layer_limit"))


def _requested_target_layer_limit(report: dict[str, Any]) -> int | None:
    return _safe_int(report.get("target_layer_limit"))


def _targeted_layers_total(report: dict[str, Any]) -> int | None:
    return _safe_int(report.get("targeted_layers_total"))


def coverage_pct(report: dict[str, Any]) -> float | None:
    limit = _target_layer_limit(report)
    total = _targeted_layers_total(report)
    if not limit or not total:
        return None
    return round(float(limit) / float(total) * 100.0, 3)


def qpruner_memory_reduction_pct(report: dict[str, Any], *, dense_weight_bits: float = DENSE_WEIGHT_BITS) -> float | None:
    qpruner = report.get("qpruner", {}) if isinstance(report.get("qpruner"), dict) else {}
    average_bits = _safe_float(qpruner.get("average_bits"))
    if average_bits is not None:
        return _round_pct((1.0 - average_bits / dense_weight_bits) * 100.0)
    memory_bits = _safe_float(qpruner.get("memory_bits"))
    targeted_params = _safe_float(report.get("baseline", {}).get("targeted_params"))
    if memory_bits is None or targeted_params is None:
        return None
    dense_bits = targeted_params * dense_weight_bits
    if dense_bits <= 0:
        return None
    return _round_pct((1.0 - memory_bits / dense_bits) * 100.0)


def bitwidth_mix(report: dict[str, Any]) -> str:
    qpruner = report.get("qpruner", {}) if isinstance(report.get("qpruner"), dict) else {}
    bitwidths = qpruner.get("bitwidths") if isinstance(qpruner.get("bitwidths"), dict) else {}
    counts: dict[int, int] = {}
    for value in bitwidths.values():
        bit = _safe_int(value)
        if bit is None:
            continue
        counts[bit] = counts.get(bit, 0) + 1
    if not counts:
        return "missing"
    return ", ".join(f"{bit}-bit:{counts[bit]}" for bit in sorted(counts))


def normalize_point(report: dict[str, Any]) -> dict[str, Any]:
    baseline = report.get("baseline", {}) if isinstance(report.get("baseline"), dict) else {}
    qpruner = report.get("qpruner", {}) if isinstance(report.get("qpruner"), dict) else {}
    baseline_loss = _safe_float(baseline.get("loss"))
    loss_delta = _safe_float(qpruner.get("loss_delta"))
    qpruner_loss = _safe_float(qpruner.get("loss"))
    if qpruner_loss is None and baseline_loss is not None and loss_delta is not None:
        qpruner_loss = baseline_loss + loss_delta
    baseline_ppl = _safe_exp(baseline_loss)
    qpruner_ppl = _safe_exp(qpruner_loss)
    ppl_delta_pct = None
    if baseline_ppl not in (None, 0) and qpruner_ppl is not None:
        ppl_delta_pct = round((qpruner_ppl / baseline_ppl - 1.0) * 100.0, 3)
    return {
        "run_label": report.get("_label", "unknown"),
        "status": report.get("status", "missing"),
        "model_id": report.get("model_id"),
        "model_path": report.get("model_path"),
        "device": report.get("device"),
        "dtype": report.get("dtype"),
        "requested_target_layer_limit": _requested_target_layer_limit(report),
        "target_layer_limit": _target_layer_limit(report),
        "targeted_layers_total": _targeted_layers_total(report),
        "coverage_pct": coverage_pct(report),
        "average_bits": _safe_float(qpruner.get("average_bits")),
        "memory_reduction_pct": qpruner_memory_reduction_pct(report),
        "baseline_loss": baseline_loss,
        "qpruner_loss": qpruner_loss,
        "loss_delta": loss_delta,
        "baseline_ppl": baseline_ppl,
        "qpruner_ppl": qpruner_ppl,
        "ppl_delta_pct": ppl_delta_pct,
        "tokens_per_s": _safe_float(qpruner.get("tokens_per_s")),
        "compression_time_s": _safe_float(qpruner.get("compression_time_s")),
        "peak_mem_mb": _safe_float(report.get("peak_mem_mb")),
        "bitwidth_mix": bitwidth_mix(report),
        "artifact": report.get("_path"),
        "qpruner_status": qpruner.get("status", "PASS"),
    }


def _is_passing(point: dict[str, Any]) -> bool:
    return point.get("status") == "PASS" and point.get("qpruner_status") == "PASS"


def best_scale_quality_point(points: list[dict[str, Any]], *, max_loss_delta: float) -> dict[str, Any] | None:
    candidates = [
        point
        for point in points
        if _is_passing(point)
        and point.get("loss_delta") is not None
        and float(point["loss_delta"]) <= max_loss_delta
        and point.get("coverage_pct") is not None
        and point.get("memory_reduction_pct") is not None
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda point: (
            float(point.get("coverage_pct") or -1.0),
            float(point.get("memory_reduction_pct") or -1.0),
            -(float(point.get("loss_delta") or 0.0)),
            float(point.get("tokens_per_s") or 0.0),
        ),
    )


def _sorted_points(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        points,
        key=lambda point: (
            int(point.get("target_layer_limit") or 0),
            float(point.get("average_bits") or 0.0),
            str(point.get("run_label") or ""),
        ),
    )


def build_summary(paths: Sequence[Path], *, max_loss_delta: float = 0.5) -> dict[str, Any]:
    reports = [read_point(path) for path in sorted(paths)]
    points = _sorted_points([normalize_point(report) for report in reports])
    passing = [point for point in points if _is_passing(point)]
    max_layer_limit = max((point.get("target_layer_limit") or 0 for point in passing), default=0) or None
    total = max((point.get("targeted_layers_total") or 0 for point in passing), default=0) or None
    max_coverage = None
    if max_layer_limit and total:
        max_coverage = round(float(max_layer_limit) / float(total) * 100.0, 3)
    best = best_scale_quality_point(points, max_loss_delta=max_loss_delta)
    if best:
        readout = (
            "QPruner scale-quality sweep reaches {limit}/{total} target layers ({coverage} coverage); "
            "best under loss delta <= {threshold} is {label} with {memory} targeted memory reduction "
            "and loss-derived approximate PPL {baseline_ppl} -> {qpruner_ppl} ({ppl_delta})."
        ).format(
            limit=best.get("target_layer_limit"),
            total=best.get("targeted_layers_total"),
            coverage=fmt(best.get("coverage_pct"), "%"),
            threshold=fmt(max_loss_delta),
            label=best.get("run_label"),
            memory=fmt(best.get("memory_reduction_pct"), "%"),
            baseline_ppl=fmt(best.get("baseline_ppl")),
            qpruner_ppl=fmt(best.get("qpruner_ppl")),
            ppl_delta=fmt(best.get("ppl_delta_pct"), "%"),
        )
    elif passing:
        readout = f"QPruner scale-quality sweep has passing points, but none met loss delta <= {max_loss_delta:.3f}."
    else:
        readout = "QPruner scale-quality sweep has no passing points."
    return {
        "status": "PASS" if passing else "MISSING",
        "max_loss_delta": max_loss_delta,
        "point_count": len(points),
        "passing_point_count": len(passing),
        "max_target_layer_limit": max_layer_limit,
        "targeted_layers_total": total,
        "max_coverage_pct": max_coverage,
        "best_under_loss_delta": best,
        "points": points,
        "readout": readout,
        "inputs": [str(path) for path in sorted(paths)],
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Qwen QPruner Scale-Quality Summary",
        "",
        report.get("readout", "missing"),
        "",
        "This table reports targeted memory reduction against 16-bit dense targeted weights. PPL is loss-derived approximate PPL from the held-out forward-loss artifact, not an external benchmark score.",
        "",
        "| Label | Status | Target layers | Coverage | QPruner bits | Target memory reduction | Loss delta | Baseline PPL | QPruner PPL | PPL delta | Tokens/s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for point in report.get("points", []):
        lines.append(
            "| {label} | {status} | {limit} / {total} | {coverage} | {bits} | {memory} | {delta} | {baseline_ppl} | {qpruner_ppl} | {ppl_delta} | {tokens} |".format(
                label=point.get("run_label", "unknown"),
                status=point.get("status", "missing"),
                limit=point.get("target_layer_limit", "missing"),
                total=point.get("targeted_layers_total", "missing"),
                coverage=fmt(point.get("coverage_pct"), "%"),
                bits=fmt(point.get("average_bits")),
                memory=fmt(point.get("memory_reduction_pct"), "%"),
                delta=fmt(point.get("loss_delta")),
                baseline_ppl=fmt(point.get("baseline_ppl")),
                qpruner_ppl=fmt(point.get("qpruner_ppl")),
                ppl_delta=fmt(point.get("ppl_delta_pct"), "%"),
                tokens=fmt(point.get("tokens_per_s")),
            )
        )
    best = report.get("best_under_loss_delta")
    lines.extend(["", "## Best Scale-Quality Point", ""])
    if isinstance(best, dict):
        lines.append(
            "Best scale-quality point under loss delta <= {threshold} is {label}: {limit}/{total} target layers, {coverage} coverage, {memory} targeted memory reduction, loss-derived approximate PPL {baseline_ppl} -> {qpruner_ppl}.".format(
                threshold=fmt(report.get("max_loss_delta")),
                label=best.get("run_label", "unknown"),
                limit=best.get("target_layer_limit", "missing"),
                total=best.get("targeted_layers_total", "missing"),
                coverage=fmt(best.get("coverage_pct"), "%"),
                memory=fmt(best.get("memory_reduction_pct"), "%"),
                baseline_ppl=fmt(best.get("baseline_ppl")),
                qpruner_ppl=fmt(best.get("qpruner_ppl")),
            )
        )
    else:
        lines.append(f"No passing point met loss delta <= {fmt(report.get('max_loss_delta'))}.")
    lines.extend(["", "## Artifact Inputs", ""])
    lines.extend(f"- `{path}`" for path in report.get("inputs", []))
    return "\n".join(lines) + "\n"


def csv_text(report: dict[str, Any]) -> str:
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "run_label",
            "status",
            "target_layer_limit",
            "targeted_layers_total",
            "coverage_pct",
            "average_bits",
            "memory_reduction_pct",
            "loss_delta",
            "baseline_ppl",
            "qpruner_ppl",
            "ppl_delta_pct",
            "tokens_per_s",
            "artifact",
        ]
    )
    for point in report.get("points", []):
        writer.writerow(
            [
                point.get("run_label"),
                point.get("status"),
                point.get("target_layer_limit"),
                point.get("targeted_layers_total"),
                point.get("coverage_pct"),
                point.get("average_bits"),
                point.get("memory_reduction_pct"),
                point.get("loss_delta"),
                point.get("baseline_ppl"),
                point.get("qpruner_ppl"),
                point.get("ppl_delta_pct"),
                point.get("tokens_per_s"),
                point.get("artifact"),
            ]
        )
    return output.getvalue()


def write_outputs(report: dict[str, Any], demo_root: Path) -> None:
    artifacts = demo_root / "artifacts"
    reports = demo_root / "reports"
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    (artifacts / SUMMARY_JSON).write_text(json.dumps(report, indent=2, sort_keys=True))
    (reports / SUMMARY_MARKDOWN).write_text(markdown(report))
    (reports / SUMMARY_CSV).write_text(csv_text(report))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write Qwen QPruner scale-quality summary")
    parser.add_argument("--demo-root", type=Path, default=Path("/mnt/nvme/622/tidal-demo"))
    parser.add_argument("--artifact-glob", default=DEFAULT_ARTIFACT_GLOB)
    parser.add_argument("--max-loss-delta", type=float, default=0.5)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    artifact_dir = args.demo_root / "artifacts"
    paths = sorted(artifact_dir.glob(args.artifact_glob))
    report = build_summary(paths, max_loss_delta=args.max_loss_delta)
    write_outputs(report, args.demo_root)
    print("QWEN_QPRUNER_SCALE_QUALITY_SUMMARY " + json.dumps(report, sort_keys=True))
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
