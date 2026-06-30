#!/usr/bin/env python
"""Summarize Ascend inference acceleration evidence into JSON/CSV/Markdown/SVG."""
from __future__ import annotations

import argparse
import csv
import html
import io
import json
import sys
from pathlib import Path
from typing import Any, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from tidal.reports.baselines import (
    engineering_baseline_note,
    engineering_baseline_summary,
    paper_baseline_alignment,
    paper_baseline_note,
)
from tidal.reports.npu_monitor_selection import best_npu_utilization_monitor
from tidal.reports.qwen_grouped_replay_selection import best_multicard_qwen_qpruner_grouped_replay


METHODS = ("baseline", "cap", "qpruner")
METHOD_LABELS = {"baseline": "baseline", "cap": "CAP", "qpruner": "QPruner"}
TRACKS = (
    {
        "id": "torch_fallback",
        "name": "torch fallback",
        "artifact": "torch_serving_fallback_tiny_qwen3_serving_fallback_npu.json",
        "source": "exports",
        "report": "reports/torch-serving-fallback-tiny_qwen3_serving_fallback_npu.md",
        "note": "torch.generate serving fallback for exported baseline/CAP/QPruner models",
    },
    {
        "id": "compressed_native",
        "name": "compressed-native",
        "artifact": "compressed_native_torch_serving_tiny_qwen3_native_serving_npu.json",
        "artifact_glob": "compressed_native_torch_serving_tiny_qwen3_native_serving*.json",
        "source": "top_level",
        "report": "reports/compressed-native-torch-serving-tiny_qwen3_native_serving_npu.md",
        "note": "torch.generate path that keeps CAPPackedLinear/QuantizedLinear modules live",
    },
    {
        "id": "vllm_metadata_shim",
        "name": "vLLM metadata-shim",
        "artifact": "vllm_serving_benchmark_tiny_qwen3_serving_vllm_metadata_shim_npu.json",
        "artifact_glob": "vllm_serving_benchmark_tiny_qwen3_serving_vllm_metadata_shim_npu*.json",
        "source": "exports",
        "report": "reports/vllm-serving-benchmark-tiny_qwen3_serving_vllm_metadata_shim_npu.md",
        "note": "vLLM-Ascend generate path with project-local metadata shim",
    },
)
SUMMARY_ARTIFACT = "inference_acceleration_summary.json"
SUMMARY_MARKDOWN = "inference-acceleration-summary.md"
SUMMARY_CSV = "inference-acceleration-summary.csv"
SUMMARY_SVG = "inference-acceleration-summary.svg"


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def select_artifact(artifacts: Path, track: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    fallback = str(track["artifact"])
    pattern = track.get("artifact_glob")
    if not pattern:
        return fallback, read_json(artifacts / fallback)
    candidates: list[tuple[tuple[int, int, int, float], str, dict[str, Any]]] = []
    seen: set[str] = set()
    paths = list(artifacts.glob(str(pattern)))
    paths.append(artifacts / fallback)
    for path in paths:
        if path.name in seen:
            continue
        seen.add(path.name)
        payload = read_json(path)
        if payload is None:
            continue
        status_score = 1 if payload.get("status") == "PASS" else 0
        try:
            max_new_tokens = int(payload.get("max_new_tokens") or 0)
        except (TypeError, ValueError):
            max_new_tokens = 0
        qpruner_cache_mode = payload.get("qpruner_cache_mode")
        code_cache_score = {
            "shape-aware-code": 3,
            "scaled-code-matmul": 2,
            "code": 1,
        }.get(str(qpruner_cache_mode), 0)
        if track.get("id") == "compressed_native":
            score = (status_score, code_cache_score, max_new_tokens, path.stat().st_mtime)
        else:
            score = (status_score, max_new_tokens, code_cache_score, path.stat().st_mtime)
        candidates.append((score, path.name, payload))
    if not candidates:
        return fallback, None
    _, name, payload = max(candidates, key=lambda item: item[0])
    return name, payload


def fmt(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rounded(value: Any, digits: int = 3) -> float | None:
    number = _number(value)
    if number is None:
        return None
    return round(number, digits)


def _method_payload(report: dict[str, Any] | None, source: str, method: str) -> dict[str, Any]:
    if not report:
        return {}
    if source == "exports":
        exports = report.get("exports", {})
        return exports.get(method, {}) if isinstance(exports.get(method), dict) else {}
    return report.get(method, {}) if isinstance(report.get(method), dict) else {}


def _speedup(tokens_per_s: Any, baseline_tokens_per_s: Any) -> float | None:
    numerator = _number(tokens_per_s)
    denominator = _number(baseline_tokens_per_s)
    if numerator is None or denominator in (None, 0.0):
        return None
    return round(numerator / denominator, 3)


def _native_memory_reduction(report: dict[str, Any], method: str, payload: dict[str, Any]) -> float | None:
    if method == "baseline":
        return 0.0
    if payload.get("targeted_param_reduction_pct") is not None:
        return _rounded(payload.get("targeted_param_reduction_pct"))
    memory = report.get("memory_reference", {}) if isinstance(report.get("memory_reference"), dict) else {}
    return _rounded(memory.get(f"{method}_targeted_param_reduction_pct"))


def _native_storage_reduction(report: dict[str, Any], method: str, payload: dict[str, Any]) -> float | None:
    if method == "baseline":
        return 0.0
    if payload.get("targeted_storage_reduction_pct") is not None:
        return _rounded(payload.get("targeted_storage_reduction_pct"))
    memory = report.get("memory_reference", {}) if isinstance(report.get("memory_reference"), dict) else {}
    return _rounded(memory.get(f"{method}_targeted_storage_reduction_pct"))


def _storage_bytes(report: dict[str, Any], method: str, payload: dict[str, Any]) -> int | None:
    if payload.get("targeted_storage_bytes") is not None:
        try:
            return int(payload["targeted_storage_bytes"])
        except (TypeError, ValueError):
            return None
    memory = report.get("memory_reference", {}) if isinstance(report.get("memory_reference"), dict) else {}
    value = memory.get(f"{method}_targeted_storage_bytes")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _memory_fields(track_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    source = payload.get("memory_measurement_source")
    peak = _rounded(payload.get("peak_mem_mb"), 1)
    if track_id == "vllm_metadata_shim" and source is None and peak == 0.0:
        return {
            "peak_mem_mb": None,
            "memory_measurement_source": "unavailable",
            "npu_smi_process_mem_mb": None,
            "torch_peak_mem_mb": _rounded(payload.get("torch_peak_mem_mb"), 1),
        }
    return {
        "peak_mem_mb": peak,
        "memory_measurement_source": source,
        "npu_smi_process_mem_mb": _rounded(payload.get("npu_smi_process_mem_mb"), 1),
        "torch_peak_mem_mb": _rounded(payload.get("torch_peak_mem_mb"), 1),
    }


def build_rows(demo_root: Path) -> list[dict[str, Any]]:
    artifacts = demo_root / "artifacts"
    rows: list[dict[str, Any]] = []
    for track in TRACKS:
        artifact_name, report = select_artifact(artifacts, track)
        baseline = _method_payload(report, str(track["source"]), "baseline")
        baseline_tokens = baseline.get("tokens_per_s")
        for method in METHODS:
            payload = _method_payload(report, str(track["source"]), method)
            memory_fields = _memory_fields(str(track["id"]), payload)
            row = {
                "track_id": track["id"],
                "track": track["name"],
                "method": method,
                "method_label": METHOD_LABELS[method],
                "artifact": f"artifacts/{artifact_name}",
                "report": track["report"],
                "track_status": report.get("status") if report else "MISSING",
                "status": payload.get("status", "MISSING"),
                "backend": (report or {}).get("backend"),
                "max_new_tokens": (report or {}).get("max_new_tokens"),
                "tokens_per_s": _rounded(payload.get("tokens_per_s")),
                "latency_ms": _rounded(payload.get("latency_ms")),
                **memory_fields,
                "speedup_vs_baseline": _speedup(payload.get("tokens_per_s"), baseline_tokens),
                "memory_reduction_pct": None,
                "storage_reduction_pct": None,
                "targeted_storage_bytes": None,
                "packed_layers": payload.get("packed_layers"),
                "quantized_layers": payload.get("quantized_layers"),
                "runtime_storage_format": payload.get("runtime_storage_format"),
                "runtime_strategy": payload.get("runtime_strategy"),
                "exported_dense_linears": payload.get("exported_dense_linears"),
                "average_bits": _rounded(payload.get("average_bits")),
                "qpruner_cache_mode": (report or {}).get("qpruner_cache_mode"),
                "code_cache_storage_reduction_pct": _rounded(payload.get("code_cache_storage_reduction_pct")),
                "serving_dense_export": (report or {}).get("serving_dense_export"),
                "note": track["note"],
            }
            if track["id"] == "compressed_native" and report:
                row["memory_reduction_pct"] = _native_memory_reduction(report, method, payload)
                row["storage_reduction_pct"] = _native_storage_reduction(report, method, payload)
                row["targeted_storage_bytes"] = _storage_bytes(report, method, payload)
            rows.append(row)
    return rows


def _best_throughput(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    passing = [row for row in rows if row.get("tokens_per_s") is not None and row.get("status") == "PASS"]
    if not passing:
        return None
    best = max(passing, key=lambda row: float(row["tokens_per_s"]))
    return {
        "track_id": best["track_id"],
        "track": best["track"],
        "method": best["method"],
        "method_label": best["method_label"],
        "tokens_per_s": best["tokens_per_s"],
        "latency_ms": best["latency_ms"],
        "peak_mem_mb": best["peak_mem_mb"],
        "max_new_tokens": best.get("max_new_tokens"),
    }


def _best_native_memory(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if row.get("track_id") == "compressed_native"
        and row.get("method") != "baseline"
        and row.get("memory_reduction_pct") is not None
    ]
    if not candidates:
        return None
    best = max(candidates, key=lambda row: float(row["memory_reduction_pct"]))
    return {
        "method": best["method"],
        "method_label": best["method_label"],
        "memory_reduction_pct": best["memory_reduction_pct"],
        "storage_reduction_pct": best["storage_reduction_pct"],
        "tokens_per_s": best["tokens_per_s"],
        "max_new_tokens": best.get("max_new_tokens"),
    }


def _artifact_payload_name(payload: dict[str, Any] | None) -> str | None:
    if not payload:
        return None
    name = payload.get("_artifact_name")
    return str(name) if name else None


def _grouped_replay_summary(demo_root: Path) -> dict[str, Any] | None:
    artifacts = demo_root / "artifacts"
    replay = best_multicard_qwen_qpruner_grouped_replay(artifacts)
    if not replay:
        return None
    monitor = best_npu_utilization_monitor(artifacts)
    aggregate = replay.get("aggregate", {}) if isinstance(replay.get("aggregate"), dict) else {}
    replay_name = _artifact_payload_name(replay)
    monitor_name = _artifact_payload_name(monitor)
    return {
        "status": replay.get("status"),
        "artifact": f"artifacts/{replay_name}" if replay_name else None,
        "run_label": replay.get("run_label"),
        "world_size": replay.get("world_size"),
        "pass_count": aggregate.get("pass_count"),
        "memory_native_worker_count": aggregate.get("memory_native_worker_count"),
        "code_cache_storage_reduction_pct": _rounded(aggregate.get("min_code_cache_storage_reduction_pct")),
        "best_grouped_speedup": _rounded(aggregate.get("best_grouped_speedup")),
        "mean_grouped_speedup": _rounded(aggregate.get("mean_grouped_speedup")),
        "released_packed_code_bytes_total": _rounded(aggregate.get("released_packed_code_bytes_total")),
        "live_compressed_payload_storage_bytes_total": _rounded(
            aggregate.get(
                "live_compressed_payload_storage_bytes_total",
                aggregate.get("live_compressed_payload_bytes_total"),
            )
        ),
        "monitor": {
            "status": (monitor or {}).get("status"),
            "artifact": f"artifacts/{monitor_name}" if monitor_name else None,
            "max_active_process_cards": (monitor or {}).get("max_active_process_cards"),
            "samples": (monitor or {}).get("samples"),
            "samples_with_8_active_process_cards": (monitor or {}).get("samples_with_8_active_process_cards"),
            "samples_with_8_nonzero_aicore": (monitor or {}).get("samples_with_8_nonzero_aicore"),
            "max_aicore_by_card": (monitor or {}).get("max_aicore_by_card"),
        },
    }


def _first_diagnosis(report: dict[str, Any] | None) -> dict[str, Any] | None:
    diagnoses = report.get("diagnoses", []) if report else []
    for item in diagnoses:
        if isinstance(item, dict):
            return item
    return None


def build_summary(demo_root: Path) -> dict[str, Any]:
    rows = build_rows(demo_root)
    grouped_replay = _grouped_replay_summary(demo_root)
    bottleneck_report = read_json(demo_root / "artifacts" / "inference_bottleneck_report.json")
    diagnosis = _first_diagnosis(bottleneck_report)
    missing_tracks = sorted({row["track"] for row in rows if row.get("track_status") == "MISSING"})
    status = "MISSING" if len(missing_tracks) == len(TRACKS) else "ACTIONABLE" if diagnosis else "PASS"
    return {
        "status": status,
        "demo_root": str(demo_root),
        "engineering_baseline": engineering_baseline_summary(),
        "paper_baseline_alignment": paper_baseline_alignment(),
        "rows": rows,
        "summary": {
            "best_throughput": _best_throughput(rows),
            "best_native_memory": _best_native_memory(rows),
            "grouped_replay": grouped_replay,
            "missing_tracks": missing_tracks,
        },
        "vllm_bottleneck": diagnosis
        or {
            "id": None,
            "title": "none",
            "evidence": "No inference bottleneck diagnosis artifact was available.",
        },
        "next_optimization_target": (bottleneck_report or {}).get("next_optimization_target"),
        "evidence": [
            "artifacts/inference_acceleration_summary.json",
            "reports/inference-acceleration-summary.md",
            "reports/inference-acceleration-summary.csv",
            "reports/inference-acceleration-summary.svg",
            "artifacts/inference_bottleneck_report.json",
            *sorted({str(row["artifact"]) for row in rows}),
            *(
                item
                for item in [
                    (grouped_replay or {}).get("artifact"),
                    ((grouped_replay or {}).get("monitor") or {}).get("artifact"),
                ]
                if item
            ),
        ],
    }


def _markdown_table(rows: Sequence[dict[str, Any]]) -> list[str]:
    lines = [
        "| Track | Method | Status | Max new tokens | Tokens/s | Latency ms | Peak MB | Peak source | Speedup | Memory reduction | Storage reduction | Code-cache storage | Runtime storage | Runtime strategy | Dense linears |",
        "|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|---|---:|",
    ]
    for row in rows:
        lines.append(
            "| {track} | {method} | {status} | {max_new_tokens} | {tokens} | {latency} | {peak} | {source} | {speedup} | {memory} | {storage} | {code_storage} | {runtime_storage} | {runtime_strategy} | {dense} |".format(
                track=row["track"],
                method=row["method_label"],
                status=row["status"],
                max_new_tokens=fmt(row.get("max_new_tokens")),
                tokens=fmt(row.get("tokens_per_s")),
                latency=fmt(row.get("latency_ms")),
                peak=fmt(row.get("peak_mem_mb")),
                source=fmt(row.get("memory_measurement_source")),
                speedup=fmt(row.get("speedup_vs_baseline")),
                memory=fmt(row.get("memory_reduction_pct"),) + ("%" if row.get("memory_reduction_pct") is not None else ""),
                storage=fmt(row.get("storage_reduction_pct")) + ("%" if row.get("storage_reduction_pct") is not None else ""),
                code_storage=fmt(row.get("code_cache_storage_reduction_pct")) + ("%" if row.get("code_cache_storage_reduction_pct") is not None else ""),
                runtime_storage=fmt(row.get("runtime_storage_format")),
                runtime_strategy=fmt(row.get("runtime_strategy")),
                dense=fmt(row.get("exported_dense_linears")),
            )
        )
    return lines


def markdown(summary: dict[str, Any]) -> str:
    best = summary.get("summary", {}).get("best_throughput")
    best_memory = summary.get("summary", {}).get("best_native_memory")
    grouped = summary.get("summary", {}).get("grouped_replay")
    bottleneck = summary.get("vllm_bottleneck", {}) if isinstance(summary.get("vllm_bottleneck"), dict) else {}
    rows = summary.get("rows", [])
    native_rows = [row for row in rows if row.get("track_id") == "compressed_native"]
    native_report = next((row for row in native_rows if row.get("serving_dense_export") is not None), {})
    lines = [
        "# Inference Acceleration Summary",
        "",
        "![Inference acceleration summary](inference-acceleration-summary.svg)",
        "",
        "This report separates three demo reads: torch fallback throughput, compressed-native memory preservation, and the vLLM-Ascend bottleneck still targeted for kernel/runtime work.",
        "",
        "## Read For Video",
        "",
        f"- {engineering_baseline_note()}",
        f"- {paper_baseline_note()}",
    ]
    if best:
        lines.append(
            "- Best throughput point: `{track} / {method}` at {tokens} tokens/s, latency {latency} ms, peak {peak} MB, max_new_tokens {tokens_out}.".format(
                track=best["track"],
                method=best["method_label"],
                tokens=fmt(best["tokens_per_s"]),
                latency=fmt(best["latency_ms"]),
                peak=fmt(best["peak_mem_mb"]),
                tokens_out=fmt(best.get("max_new_tokens")),
            )
        )
    if best_memory:
        lines.append(
            "- compressed-native preserves quantized/pruned modules: best memory point `{method}` saves {memory}% targeted memory and {storage}% targeted storage with dense export `{dense}`.".format(
                method=best_memory["method_label"],
                memory=fmt(best_memory.get("memory_reduction_pct")),
                storage=fmt(best_memory.get("storage_reduction_pct")),
                dense=native_report.get("serving_dense_export", "missing"),
            )
        )
        native_qpruner = next(
            (
                row
                for row in native_rows
                if row.get("method") == "qpruner" and row.get("qpruner_cache_mode") == "code"
            ),
            None,
        )
        if native_qpruner:
            lines.append(
                "- QPruner code-cache keeps packed storage live: runtime `{strategy}`, code-cache storage {storage}% reduction.".format(
                    strategy=native_qpruner.get("runtime_strategy"),
                    storage=fmt(native_qpruner.get("code_cache_storage_reduction_pct")),
                )
            )
    if isinstance(grouped, dict) and grouped:
        monitor = grouped.get("monitor", {}) if isinstance(grouped.get("monitor"), dict) else {}
        lines.append(
            "- 8-card grouped replay: {status}, memory-native workers {memory}/{passes}, code-cache storage {storage}% reduction, best grouped speedup {speedup}x; monitor {aicore_samples} samples with all 8 AICore nonzero; released packed-code bytes {released}, live compressed payload bytes {payload}.".format(
                status=grouped.get("status", "missing"),
                memory=fmt(grouped.get("memory_native_worker_count")),
                passes=fmt(grouped.get("pass_count")),
                storage=fmt(grouped.get("code_cache_storage_reduction_pct")),
                speedup=fmt(grouped.get("best_grouped_speedup")),
                aicore_samples=fmt(monitor.get("samples_with_8_nonzero_aicore")),
                released=fmt(grouped.get("released_packed_code_bytes_total")),
                payload=fmt(grouped.get("live_compressed_payload_storage_bytes_total")),
            )
        )
    lines.append(
        "- vLLM bottleneck: {title}. Next target: {target}.".format(
            title=bottleneck.get("title", "missing"),
            target=(summary.get("next_optimization_target") or {}).get("title", "missing")
            if isinstance(summary.get("next_optimization_target"), dict)
            else "missing",
        )
    )
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            *_markdown_table(rows),
            "",
            "## Evidence",
            "",
            "- `reports/inference-acceleration-summary.svg`",
            "- `reports/inference-acceleration-summary.csv`",
            "- `artifacts/inference_acceleration_summary.json`",
            "- `artifacts/inference_bottleneck_report.json`",
            *[f"- `{item}`" for item in summary.get("evidence", []) if str(item).startswith("artifacts/")],
        ]
    )
    return "\n".join(lines) + "\n"


def csv_text(summary: dict[str, Any]) -> str:
    fieldnames = [
        "track",
        "method",
        "status",
        "tokens_per_s",
        "latency_ms",
        "peak_mem_mb",
        "memory_measurement_source",
        "npu_smi_process_mem_mb",
        "torch_peak_mem_mb",
        "max_new_tokens",
        "speedup_vs_baseline",
        "memory_reduction_pct",
        "storage_reduction_pct",
        "targeted_storage_bytes",
        "packed_layers",
        "quantized_layers",
        "runtime_storage_format",
        "runtime_strategy",
        "qpruner_cache_mode",
        "code_cache_storage_reduction_pct",
        "exported_dense_linears",
        "artifact",
    ]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for row in summary.get("rows", []):
        writer.writerow({field: row.get(field) for field in fieldnames})
    return output.getvalue()


def _bar_width(value: float | None, max_value: float, width: float) -> float:
    if value is None or max_value <= 0:
        return 0.0
    return max(1.0, (value / max_value) * width)


def svg(summary: dict[str, Any]) -> str:
    rows = [row for row in summary.get("rows", []) if row.get("tokens_per_s") is not None]
    width = 1180
    height = 780
    left = 270
    bar_w = 650
    top = 108
    row_h = 54
    colors = {"baseline": "#4b5563", "cap": "#2563eb", "qpruner": "#dc2626"}
    max_tps = max((float(row["tokens_per_s"]) for row in rows), default=1.0)
    best = summary.get("summary", {}).get("best_throughput")
    best_label = "missing"
    if best:
        best_label = f"{best['track']} / {best['method_label']}"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Inference acceleration throughput latency memory summary">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#111827} .muted{fill:#6b7280} .axis{stroke:#111827;stroke-width:1.4} .grid{stroke:#e5e7eb;stroke-width:1} .card{fill:#f9fafb;stroke:#e5e7eb}",
        "</style>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        '<text x="64" y="48" font-size="29" font-weight="700">Inference acceleration: throughput, latency, memory</text>',
        '<text x="64" y="76" font-size="14" class="muted">Ascend 910B demo evidence across torch fallback, compressed-native torch_npu, and vLLM metadata-shim</text>',
        f'<text x="{left}" y="{top - 22}" font-size="13" class="muted">Tokens/s</text>',
        f'<line class="axis" x1="{left}" y1="{top - 8}" x2="{left + bar_w}" y2="{top - 8}"/>',
    ]
    for tick in range(6):
        x = left + (tick / 5) * bar_w
        value = (tick / 5) * max_tps
        parts.extend(
            [
                f'<line class="grid" x1="{x:.1f}" y1="{top - 8}" x2="{x:.1f}" y2="{top + row_h * max(len(rows), 1)}"/>',
                f'<text x="{x:.1f}" y="{top - 16}" font-size="11" text-anchor="middle" class="muted">{value:.0f}</text>',
            ]
        )
    for index, row in enumerate(rows):
        y = top + index * row_h
        color = colors.get(str(row.get("method")), "#6b7280")
        tps = _number(row.get("tokens_per_s"))
        bw = _bar_width(tps, max_tps, bar_w)
        label = f"{row['track']} / {row['method_label']}"
        parts.extend(
            [
                f'<text x="64" y="{y + 22}" font-size="14" font-weight="700">{html.escape(label)}</text>',
                f'<text x="64" y="{y + 41}" font-size="12" class="muted">lat {fmt(row.get("latency_ms"))} ms - Peak MB {fmt(row.get("peak_mem_mb"))} - speedup {fmt(row.get("speedup_vs_baseline"))}x</text>',
                f'<rect x="{left}" y="{y}" width="{bw:.1f}" height="28" rx="3" fill="{color}"><title>{html.escape(label)}: {fmt(row.get("tokens_per_s"))} tokens/s, latency {fmt(row.get("latency_ms"))} ms, Peak MB {fmt(row.get("peak_mem_mb"))}</title></rect>',
                f'<text x="{left + bw + 8:.1f}" y="{y + 19}" font-size="13">{fmt(row.get("tokens_per_s"))}</text>',
            ]
        )
        if row.get("track_id") == "compressed_native" and row.get("method") != "baseline":
            parts.append(
                f'<text x="{left + bar_w + 32}" y="{y + 19}" font-size="12" class="muted">memory {fmt(row.get("memory_reduction_pct"))}% / storage {fmt(row.get("storage_reduction_pct"))}%</text>'
            )
    legend_x = 930
    legend_y = 130
    parts.extend(
        [
            f'<rect class="card" x="{legend_x}" y="{legend_y}" width="210" height="210" rx="4"/>',
            f'<text x="{legend_x + 18}" y="{legend_y + 32}" font-size="15" font-weight="700">Legend</text>',
        ]
    )
    for index, method in enumerate(METHODS):
        y = legend_y + 62 + index * 30
        parts.extend(
            [
                f'<rect x="{legend_x + 18}" y="{y - 13}" width="22" height="14" fill="{colors[method]}"/>',
                f'<text x="{legend_x + 50}" y="{y}" font-size="13">{METHOD_LABELS[method]}</text>',
            ]
        )
    bottleneck = summary.get("vllm_bottleneck", {}) if isinstance(summary.get("vllm_bottleneck"), dict) else {}
    parts.extend(
        [
            f'<text x="{legend_x + 18}" y="{legend_y + 164}" font-size="13" font-weight="700">Best: {html.escape(best_label)}</text>',
            f'<text x="{legend_x + 18}" y="{legend_y + 188}" font-size="12" class="muted">vLLM bottleneck tracked separately</text>',
            f'<text x="64" y="{height - 48}" font-size="13" class="muted">Peak MB labels are runtime allocator peaks; compressed-native memory/storage percentages use targeted compressed-module accounting.</text>',
            f'<text x="64" y="{height - 24}" font-size="13" class="muted">baseline = engineering reference, not paper baseline; paper baselines are tracked separately.</text>',
            f'<!-- {html.escape(str(bottleneck.get("title", "missing")))} -->',
        ]
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def write_outputs(summary: dict[str, Any], demo_root: Path) -> None:
    artifacts = demo_root / "artifacts"
    reports = demo_root / "reports"
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    (artifacts / SUMMARY_ARTIFACT).write_text(json.dumps(summary, indent=2, sort_keys=True))
    (reports / SUMMARY_MARKDOWN).write_text(markdown(summary))
    (reports / SUMMARY_CSV).write_text(csv_text(summary))
    (reports / SUMMARY_SVG).write_text(svg(summary))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize inference acceleration demo evidence")
    parser.add_argument("--demo-root", type=Path, default=Path("/mnt/nvme/622/tidal-demo"))
    args = parser.parse_args(argv)
    summary = build_summary(args.demo_root)
    write_outputs(summary, args.demo_root)
    print(
        "INFERENCE_ACCELERATION_SUMMARY {json} {markdown} {csv} {svg}".format(
            json=args.demo_root / "artifacts" / SUMMARY_ARTIFACT,
            markdown=args.demo_root / "reports" / SUMMARY_MARKDOWN,
            csv=args.demo_root / "reports" / SUMMARY_CSV,
            svg=args.demo_root / "reports" / SUMMARY_SVG,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
