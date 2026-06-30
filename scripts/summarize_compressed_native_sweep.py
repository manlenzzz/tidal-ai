#!/usr/bin/env python
"""Summarize compressed-native torch serving cache and decode-length sweeps."""
from __future__ import annotations

import argparse
import csv
import html
import io
import json
from pathlib import Path
from typing import Any, Sequence


OUT_JSON = "compressed_native_sweep_summary.json"
OUT_MD = "compressed-native-cache-long-decode-sweep.md"
OUT_CSV = "compressed-native-cache-long-decode-sweep.csv"
OUT_SVG = "compressed-native-cache-long-decode-sweep.svg"
METHODS = ("baseline", "cap", "qpruner")
METHOD_LABELS = {"baseline": "baseline", "cap": "CAP", "qpruner": "QPruner"}
MEMORY_PRESERVING_QPRUNER_STRATEGIES = {
    "dequantize_per_forward",
    "int8_code_cache_dequantize_on_device",
    "scaled_int8_code_matmul",
    "shape_aware_int8_code_cache",
    "shape_aware_mixed_int8_code_cache",
}


def fmt(value: Any, suffix: str = "") -> str:
    if value is None:
        return "missing"
    if isinstance(value, float):
        return f"{value:.3f}{suffix}"
    return f"{value}{suffix}"


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round(value: Any, digits: int = 3) -> float | None:
    number = _number(value)
    if number is None:
        return None
    return round(number, digits)


def load_artifact(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    payload["_path"] = str(path)
    label = str(payload.get("run_label") or path.stem)
    prefix = "compressed_native_torch_serving_"
    if label.startswith(prefix):
        label = label[len(prefix) :]
    payload["_label"] = label
    return payload


def method_speedup(payload: dict[str, Any], method: str) -> float | None:
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
    key = f"{method}_vs_baseline_speedup"
    if summary.get(key) is not None:
        return _round(summary[key])
    baseline = payload.get("baseline", {}) if isinstance(payload.get("baseline"), dict) else {}
    method_payload = payload.get(method, {}) if isinstance(payload.get(method), dict) else {}
    baseline_latency = _number(baseline.get("latency_ms"))
    method_latency = _number(method_payload.get("latency_ms"))
    if baseline_latency is not None and method_latency not in (None, 0.0):
        return round(baseline_latency / method_latency, 3)
    baseline_tps = _number(baseline.get("tokens_per_s"))
    method_tps = _number(method_payload.get("tokens_per_s"))
    if baseline_tps not in (None, 0.0) and method_tps is not None:
        return round(method_tps / baseline_tps, 3)
    return None


def row_from_report(report: dict[str, Any]) -> dict[str, Any]:
    memory = report.get("memory_reference", {}) if isinstance(report.get("memory_reference"), dict) else {}
    baseline = report.get("baseline", {}) if isinstance(report.get("baseline"), dict) else {}
    cap = report.get("cap", {}) if isinstance(report.get("cap"), dict) else {}
    qpruner = report.get("qpruner", {}) if isinstance(report.get("qpruner"), dict) else {}
    qpruner_shape_plan = qpruner.get("shape_aware_plan", {}) if isinstance(qpruner.get("shape_aware_plan"), dict) else {}
    return {
        "label": report.get("_label", "unknown"),
        "path": report.get("_path"),
        "status": report.get("status", "missing"),
        "backend": report.get("backend"),
        "max_new_tokens": int(report.get("max_new_tokens") or 0),
        "inference_cache_enabled": bool(report.get("inference_cache_enabled")),
        "serving_dense_export": report.get("serving_dense_export"),
        "baseline_tokens_per_s": _round(baseline.get("tokens_per_s")),
        "cap_tokens_per_s": _round(cap.get("tokens_per_s")),
        "qpruner_tokens_per_s": _round(qpruner.get("tokens_per_s")),
        "baseline_latency_ms": _round(baseline.get("latency_ms")),
        "cap_latency_ms": _round(cap.get("latency_ms")),
        "qpruner_latency_ms": _round(qpruner.get("latency_ms")),
        "baseline_peak_mem_mb": _round(baseline.get("peak_mem_mb")),
        "cap_peak_mem_mb": _round(cap.get("peak_mem_mb")),
        "qpruner_peak_mem_mb": _round(qpruner.get("peak_mem_mb")),
        "cap_vs_baseline_speedup": method_speedup(report, "cap"),
        "qpruner_vs_baseline_speedup": method_speedup(report, "qpruner"),
        "cap_cache_modules": cap.get("cache_modules"),
        "qpruner_cache_modules": qpruner.get("cache_modules"),
        "cap_runtime_storage_format": cap.get("runtime_storage_format"),
        "cap_runtime_strategy": cap.get("runtime_strategy"),
        "cap_dense_sparse_buffers": cap.get("dense_sparse_buffers"),
        "cap_sparse_entries": cap.get("sparse_entries"),
        "qpruner_cache_mode": report.get("qpruner_cache_mode") or qpruner.get("cache_mode"),
        "qpruner_runtime_storage_format": qpruner.get("runtime_storage_format"),
        "qpruner_runtime_strategy": qpruner.get("runtime_strategy"),
        "qpruner_cached_dense_weight_modules": qpruner.get("cached_dense_weight_modules"),
        "qpruner_cached_dense_weight_bytes": qpruner.get("cached_dense_weight_bytes"),
        "qpruner_cached_code_modules": qpruner.get("cached_code_modules"),
        "qpruner_cached_code_bytes": qpruner.get("cached_code_bytes"),
        "qpruner_scaled_code_matmul_modules": qpruner.get("scaled_code_matmul_modules"),
        "qpruner_shape_aware_cache_modules": qpruner.get("shape_aware_cache_modules"),
        "qpruner_shape_policy_source": qpruner_shape_plan.get("shape_policy_source"),
        "qpruner_shape_policy_match_count": qpruner_shape_plan.get("shape_policy_match_count"),
        "qpruner_shape_policy_shape_count": qpruner_shape_plan.get("shape_policy_shape_count"),
        "qpruner_shape_strategy_source_counts": qpruner_shape_plan.get("strategy_source_counts"),
        "cap_storage_reduction_pct": _round(memory.get("cap_targeted_storage_reduction_pct")),
        "qpruner_storage_reduction_pct": _round(memory.get("qpruner_targeted_storage_reduction_pct")),
        "qpruner_code_cache_storage_reduction_pct": _round(
            qpruner.get("code_cache_storage_reduction_pct")
            if qpruner.get("code_cache_storage_reduction_pct") is not None
            else memory.get("qpruner_code_cache_storage_reduction_pct")
        ),
        "qpruner_code_cache_storage_bytes": qpruner.get("code_cache_storage_bytes")
        if qpruner.get("code_cache_storage_bytes") is not None
        else memory.get("qpruner_code_cache_storage_bytes"),
        "peak_mem_mb": max(
            [
                value
                for value in (
                    _number(baseline.get("peak_mem_mb")),
                    _number(cap.get("peak_mem_mb")),
                    _number(qpruner.get("peak_mem_mb")),
                )
                if value is not None
            ],
            default=None,
        ),
    }


def _sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (row["max_new_tokens"], row["inference_cache_enabled"], row["label"]))


def _best(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for row in rows:
        for method in ("cap", "qpruner"):
            speedup = row.get(f"{method}_vs_baseline_speedup")
            if speedup is None:
                continue
            candidates.append(
                {
                    "label": row["label"],
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "max_new_tokens": row["max_new_tokens"],
                    "inference_cache_enabled": row["inference_cache_enabled"],
                    "cap_vs_baseline_speedup": row.get("cap_vs_baseline_speedup"),
                    "qpruner_vs_baseline_speedup": row.get("qpruner_vs_baseline_speedup"),
                    "speedup": speedup,
                    "storage_reduction_pct": row.get(f"{method}_storage_reduction_pct"),
                }
            )
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            float(item["speedup"]),
            float(item.get("storage_reduction_pct") or 0.0),
            int(item["max_new_tokens"]),
        ),
    )


def _matching(rows: Sequence[dict[str, Any]], *, max_new_tokens: int, cache: bool) -> dict[str, Any] | None:
    matches = [
        row
        for row in rows
        if int(row.get("max_new_tokens") or 0) == int(max_new_tokens)
        and bool(row.get("inference_cache_enabled")) is cache
    ]
    return matches[0] if matches else None


def _memory_preserving_qpruner_best(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for row in rows:
        speedup = row.get("qpruner_vs_baseline_speedup")
        runtime_strategy = row.get("qpruner_runtime_strategy")
        cached_dense_modules = row.get("qpruner_cached_dense_weight_modules")
        if speedup is None or runtime_strategy not in MEMORY_PRESERVING_QPRUNER_STRATEGIES:
            continue
        if cached_dense_modules not in (None, 0):
            continue
        candidates.append(
            {
                "label": row["label"],
                "path": row.get("path"),
                "max_new_tokens": row["max_new_tokens"],
                "qpruner_cache_mode": row.get("qpruner_cache_mode") or "none",
                "qpruner_runtime_strategy": runtime_strategy,
                "qpruner_runtime_storage_format": row.get("qpruner_runtime_storage_format"),
                "qpruner_tokens_per_s": row.get("qpruner_tokens_per_s"),
                "qpruner_vs_baseline_speedup": speedup,
                "qpruner_storage_reduction_pct": row.get("qpruner_storage_reduction_pct"),
                "qpruner_code_cache_storage_reduction_pct": row.get(
                    "qpruner_code_cache_storage_reduction_pct"
                ),
                "qpruner_cached_dense_weight_modules": cached_dense_modules or 0,
                "qpruner_cached_dense_weight_bytes": row.get("qpruner_cached_dense_weight_bytes"),
                "qpruner_cached_code_modules": row.get("qpruner_cached_code_modules"),
                "qpruner_cached_code_bytes": row.get("qpruner_cached_code_bytes"),
                "qpruner_scaled_code_matmul_modules": row.get("qpruner_scaled_code_matmul_modules"),
                "qpruner_shape_aware_cache_modules": row.get("qpruner_shape_aware_cache_modules"),
                "qpruner_shape_policy_source": row.get("qpruner_shape_policy_source"),
                "qpruner_shape_policy_match_count": row.get("qpruner_shape_policy_match_count") or 0,
                "qpruner_shape_policy_shape_count": row.get("qpruner_shape_policy_shape_count") or 0,
                "qpruner_shape_strategy_source_counts": row.get("qpruner_shape_strategy_source_counts") or {},
            }
        )
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            1 if int(item.get("qpruner_shape_policy_match_count") or 0) > 0 else 0,
            float(item.get("qpruner_vs_baseline_speedup") or 0.0),
            float(
                item.get("qpruner_code_cache_storage_reduction_pct")
                if item.get("qpruner_code_cache_storage_reduction_pct") is not None
                else item.get("qpruner_storage_reduction_pct") or 0.0
            ),
            int(item.get("max_new_tokens") or 0),
        ),
    )


def _cache_effect(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    token_values = sorted({int(row["max_new_tokens"]) for row in rows})
    for tokens in token_values:
        off = _matching(rows, max_new_tokens=tokens, cache=False)
        on = _matching(rows, max_new_tokens=tokens, cache=True)
        if off and on:
            return {
                "max_new_tokens": tokens,
                "cap_speedup_delta": _delta(on.get("cap_vs_baseline_speedup"), off.get("cap_vs_baseline_speedup")),
                "qpruner_speedup_delta": _delta(
                    on.get("qpruner_vs_baseline_speedup"), off.get("qpruner_vs_baseline_speedup")
                ),
            }
    return {"max_new_tokens": None, "cap_speedup_delta": None, "qpruner_speedup_delta": None}


def _long_decode_effect(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    cached = [row for row in rows if row.get("inference_cache_enabled")]
    if len(cached) < 2:
        return {"from_max_new_tokens": None, "to_max_new_tokens": None, "cap_speedup_delta": None, "qpruner_speedup_delta": None}
    ordered = sorted(cached, key=lambda row: int(row["max_new_tokens"]))
    short = ordered[0]
    long = ordered[-1]
    return {
        "from_max_new_tokens": short["max_new_tokens"],
        "to_max_new_tokens": long["max_new_tokens"],
        "cap_speedup_delta": _delta(long.get("cap_vs_baseline_speedup"), short.get("cap_vs_baseline_speedup")),
        "qpruner_speedup_delta": _delta(
            long.get("qpruner_vs_baseline_speedup"), short.get("qpruner_vs_baseline_speedup")
        ),
    }


def _ratio(new_value: Any, old_value: Any) -> float | None:
    new_number = _number(new_value)
    old_number = _number(old_value)
    if new_number is None or old_number in (None, 0.0):
        return None
    return round(new_number / old_number, 3)


def _delta(new_value: Any, old_value: Any) -> float | None:
    new_number = _number(new_value)
    old_number = _number(old_value)
    if new_number is None or old_number is None:
        return None
    return round(new_number - old_number, 3)


def _signed_fmt(value: Any) -> str:
    number = _number(value)
    if number is None:
        return "missing"
    sign = "+" if number >= 0 else ""
    return f"{sign}{number:.3f}"


def _runtime_diagnosis(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    memory_preserving_best = _memory_preserving_qpruner_best(rows)
    cached = [row for row in rows if row.get("inference_cache_enabled")]
    if len(cached) >= 2:
        ordered_cached = sorted(cached, key=lambda row: int(row["max_new_tokens"]))
        short_cached = ordered_cached[0]
        long_cached = ordered_cached[-1]
    else:
        short_cached = None
        long_cached = None
    short_uncached = None
    if short_cached is not None:
        short_uncached = _matching(
            rows,
            max_new_tokens=int(short_cached.get("max_new_tokens") or 0),
            cache=False,
        )
    baseline_scaling = (
        _ratio(long_cached.get("baseline_tokens_per_s"), short_cached.get("baseline_tokens_per_s"))
        if short_cached and long_cached
        else None
    )
    cap_scaling = (
        _ratio(long_cached.get("cap_tokens_per_s"), short_cached.get("cap_tokens_per_s"))
        if short_cached and long_cached
        else None
    )
    qpruner_scaling = (
        _ratio(long_cached.get("qpruner_tokens_per_s"), short_cached.get("qpruner_tokens_per_s"))
        if short_cached and long_cached
        else None
    )
    qpruner_cache_peak_delta = None
    qpruner_cache_vs_baseline_peak_delta = None
    # Use method-specific peak values when present; older artifacts only have row peak.
    if short_cached and short_uncached:
        qpruner_cache_peak_delta = _delta(
            short_cached.get("qpruner_peak_mem_mb") or short_cached.get("peak_mem_mb"),
            short_uncached.get("qpruner_peak_mem_mb") or short_uncached.get("peak_mem_mb"),
        )
        qpruner_cache_vs_baseline_peak_delta = _delta(
            short_cached.get("qpruner_peak_mem_mb") or short_cached.get("peak_mem_mb"),
            short_uncached.get("baseline_peak_mem_mb") or short_uncached.get("peak_mem_mb"),
        )
    cache_delta = _cache_effect(rows).get("qpruner_speedup_delta")
    short_tokens = short_cached.get("max_new_tokens") if short_cached else None
    cap_coordinate_uncached = next(
        (
            row
            for row in rows
            if not row.get("inference_cache_enabled")
            and row.get("cap_runtime_strategy") == "coordinate_sparse_residual"
        ),
        None,
    )
    cap_cached = next(
        (
            row
            for row in sorted(rows, key=lambda item: int(item.get("max_new_tokens") or 0), reverse=True)
            if row.get("inference_cache_enabled") and row.get("cap_runtime_strategy") == "dense_weight_cache"
        ),
        None,
    )
    cap_cache_speedup_delta = (
        _delta(cap_cached.get("cap_vs_baseline_speedup"), cap_coordinate_uncached.get("cap_vs_baseline_speedup"))
        if cap_cached and cap_coordinate_uncached and cap_cached.get("max_new_tokens") == cap_coordinate_uncached.get("max_new_tokens")
        else _cache_effect(rows).get("cap_speedup_delta")
    )
    cap_dense_sparse_buffers = (
        cap_coordinate_uncached.get("cap_dense_sparse_buffers")
        if cap_coordinate_uncached
        else cap_cached.get("cap_dense_sparse_buffers") if cap_cached else None
    )
    return {
        "short_cached_label": short_cached.get("label") if short_cached else None,
        "long_cached_label": long_cached.get("label") if long_cached else None,
        "baseline_decode_scaling": baseline_scaling,
        "cap_decode_scaling": cap_scaling,
        "qpruner_decode_scaling": qpruner_scaling,
        "cap_scaling_gap_vs_baseline": _delta(cap_scaling, baseline_scaling),
        "qpruner_scaling_gap_vs_baseline": _delta(qpruner_scaling, baseline_scaling),
        "qpruner_cache_peak_mem_delta_mb": qpruner_cache_peak_delta,
        "qpruner_cache_vs_baseline_peak_mem_delta_mb": qpruner_cache_vs_baseline_peak_delta,
        "cap_coordinate_uncached_label": cap_coordinate_uncached.get("label") if cap_coordinate_uncached else None,
        "cap_coordinate_uncached_speedup": cap_coordinate_uncached.get("cap_vs_baseline_speedup")
        if cap_coordinate_uncached
        else None,
        "cap_coordinate_uncached_peak_mem_mb": cap_coordinate_uncached.get("cap_peak_mem_mb")
        if cap_coordinate_uncached
        else None,
        "cap_coordinate_runtime_strategy": cap_coordinate_uncached.get("cap_runtime_strategy")
        if cap_coordinate_uncached
        else None,
        "cap_cached_runtime_strategy": cap_cached.get("cap_runtime_strategy") if cap_cached else None,
        "cap_dense_sparse_buffers": cap_dense_sparse_buffers,
        "cap_runtime_tradeoff": _cap_runtime_tradeoff_text(
            dense_sparse_buffers=cap_dense_sparse_buffers,
            coordinate_speedup=cap_coordinate_uncached.get("cap_vs_baseline_speedup") if cap_coordinate_uncached else None,
            coordinate_peak_mem=cap_coordinate_uncached.get("cap_peak_mem_mb") if cap_coordinate_uncached else None,
            cache_speedup_delta=cap_cache_speedup_delta,
            cached_strategy=cap_cached.get("cap_runtime_strategy") if cap_cached else None,
        ),
        "cache_tradeoff": _cache_tradeoff_text(
            short_tokens=short_tokens,
            qpruner_cache_peak_delta=qpruner_cache_peak_delta,
            qpruner_cache_vs_baseline_peak_delta=qpruner_cache_vs_baseline_peak_delta,
            cache_speedup_delta=cache_delta,
        ),
        "memory_preserving_qpruner_tradeoff": _memory_preserving_qpruner_tradeoff_text(
            memory_preserving_best
        ),
        "next_action": "Keep compressed weights live, then fuse packed/quantized decode kernels on NPU for long generation.",
    }


def _cap_runtime_tradeoff_text(
    *,
    dense_sparse_buffers: Any,
    coordinate_speedup: Any,
    coordinate_peak_mem: Any,
    cache_speedup_delta: Any,
    cached_strategy: Any,
) -> str:
    if dense_sparse_buffers is None or coordinate_speedup is None:
        return "CAP coordinate sparse runtime tradeoff is missing; rerun compressed-native cache on/off points."
    return (
        "CAP stores sparse residuals as coordinates with {buffers} dense sparse buffers; "
        "uncached coordinate path reaches {speedup}x baseline at {peak} MB peak, "
        "while cache switches runtime strategy to {strategy} for {delta}x speedup delta."
    ).format(
        buffers=dense_sparse_buffers,
        speedup=fmt(coordinate_speedup),
        peak=fmt(coordinate_peak_mem),
        strategy=cached_strategy or "missing",
        delta=_signed_fmt(cache_speedup_delta),
    )


def _cache_tradeoff_text(
    *,
    short_tokens: Any,
    qpruner_cache_peak_delta: Any,
    qpruner_cache_vs_baseline_peak_delta: Any,
    cache_speedup_delta: Any,
) -> str:
    if qpruner_cache_peak_delta is None or cache_speedup_delta is None:
        return "QPruner cache memory/speed tradeoff is missing; rerun cache on/off points."
    return (
        "QPruner cache adds {mem} MB over uncached QPruner peak at max_new_tokens={tokens} "
        "for {delta}x speedup delta; +{baseline_mem} MB vs uncached baseline peak."
    ).format(
        mem=fmt(qpruner_cache_peak_delta),
        tokens=short_tokens,
        delta=fmt(cache_speedup_delta),
        baseline_mem=fmt(qpruner_cache_vs_baseline_peak_delta),
    )


def _memory_preserving_qpruner_tradeoff_text(best: dict[str, Any] | None) -> str:
    if not best:
        return "Memory-preserving QPruner runtime tradeoff is missing; rerun code-cache or scaled-code cache modes."
    return (
        "Best memory-preserving QPruner runtime {label} uses {strategy} "
        "with cache mode {mode}: {speedup}x baseline while retaining {storage}% code-cache storage reduction "
        "and {dense_modules} dense cached weight modules{policy_text}."
    ).format(
        label=best.get("label", "missing"),
        strategy=best.get("qpruner_runtime_strategy", "missing"),
        mode=best.get("qpruner_cache_mode", "missing"),
        speedup=fmt(best.get("qpruner_vs_baseline_speedup")),
        storage=fmt(
            best.get("qpruner_code_cache_storage_reduction_pct")
            if best.get("qpruner_code_cache_storage_reduction_pct") is not None
            else best.get("qpruner_storage_reduction_pct")
        ),
        dense_modules=fmt(best.get("qpruner_cached_dense_weight_modules")),
        policy_text=(
            f"; packed-decode shape policy matched {int(best.get('qpruner_shape_policy_match_count') or 0)} modules"
            if int(best.get("qpruner_shape_policy_match_count") or 0) > 0
            else ""
        ),
    )


def readout(best: dict[str, Any] | None) -> str:
    if not best:
        return "No passing compressed-native sweep point was available."
    if float(best.get("speedup") or 0.0) >= 1.0 and best.get("inference_cache_enabled") and int(best.get("max_new_tokens") or 0) > 1:
        return (
            "Cached long-decode compressed-native path is latency-positive: "
            "{method} reaches {speedup}x baseline at max_new_tokens={tokens} while retaining {storage}% targeted storage reduction."
        ).format(
            method=best["method_label"],
            speedup=fmt(best["speedup"]),
            tokens=best["max_new_tokens"],
            storage=fmt(best.get("storage_reduction_pct")),
        )
    if float(best.get("speedup") or 0.0) >= 1.0 and best.get("inference_cache_enabled"):
        return (
            "Cached compressed-native path is latency-positive at max_new_tokens={tokens}: "
            "{method} reaches {speedup}x baseline while retaining {storage}% targeted storage reduction; "
            "long-decode remains the next optimization target."
        ).format(
            method=best["method_label"],
            speedup=fmt(best["speedup"]),
            tokens=best["max_new_tokens"],
            storage=fmt(best.get("storage_reduction_pct")),
        )
    if float(best.get("speedup") or 0.0) >= 1.0:
        return (
            "At least one compressed-native path is latency-positive: {method} reaches {speedup}x baseline."
        ).format(method=best["method_label"], speedup=fmt(best["speedup"]))
    return (
        "Compressed-native storage savings are preserved, but this sweep is still latency-negative; "
        "next work should target kernel/runtime overhead."
    )


def build_summary(paths: Sequence[Path], *, demo_root: Path) -> dict[str, Any]:
    reports = [load_artifact(path) for path in paths]
    rows = _sort_rows([row_from_report(report) for report in reports])
    best = _best(rows)
    memory_preserving_best = _memory_preserving_qpruner_best(rows)
    memory_values = [row.get("qpruner_storage_reduction_pct") for row in rows if row.get("qpruner_storage_reduction_pct") is not None]
    code_cache_memory_values = [
        row.get("qpruner_code_cache_storage_reduction_pct")
        for row in rows
        if row.get("qpruner_code_cache_storage_reduction_pct") is not None
    ]
    return {
        "status": "PASS" if rows and all(row.get("status") == "PASS" for row in rows) else "ACTIONABLE" if rows else "MISSING",
        "demo_root": str(demo_root),
        "rows": rows,
        "best": best,
        "memory_preserving_qpruner_best": memory_preserving_best,
        "cache_effect": _cache_effect(rows),
        "long_decode_effect": _long_decode_effect(rows),
        "runtime_diagnosis": _runtime_diagnosis(rows),
        "memory": {
            "best_qpruner_storage_reduction_pct": max(memory_values) if memory_values else None,
            "best_qpruner_code_cache_storage_reduction_pct": max(code_cache_memory_values)
            if code_cache_memory_values
            else None,
        },
        "readout": readout(best),
        "evidence": [
            "artifacts/compressed_native_sweep_summary.json",
            "reports/compressed-native-cache-long-decode-sweep.md",
            "reports/compressed-native-cache-long-decode-sweep.csv",
            "reports/compressed-native-cache-long-decode-sweep.svg",
            *[str(path) for path in paths],
        ],
    }


def markdown(summary: dict[str, Any]) -> str:
    memory_best = (
        summary.get("memory_preserving_qpruner_best", {})
        if isinstance(summary.get("memory_preserving_qpruner_best"), dict)
        else {}
    )
    lines = [
        "# Compressed-Native Cache/Long-Decode Sweep",
        "",
        summary.get("readout", "missing"),
        "",
        "## Effects",
        "",
        "- Cache effect at max_new_tokens={tokens}: CAP {cap}, QPruner {qpruner} speedup delta.".format(
            tokens=summary.get("cache_effect", {}).get("max_new_tokens"),
            cap=fmt(summary.get("cache_effect", {}).get("cap_speedup_delta")),
            qpruner=fmt(summary.get("cache_effect", {}).get("qpruner_speedup_delta")),
        ),
        "- Long-decode effect under cache: max_new_tokens {src} -> {dst}, CAP {cap}, QPruner {qpruner} speedup delta.".format(
            src=summary.get("long_decode_effect", {}).get("from_max_new_tokens"),
            dst=summary.get("long_decode_effect", {}).get("to_max_new_tokens"),
            cap=fmt(summary.get("long_decode_effect", {}).get("cap_speedup_delta")),
            qpruner=fmt(summary.get("long_decode_effect", {}).get("qpruner_speedup_delta")),
        ),
        "- QPruner storage reduction {value}%.".format(
            value=fmt(summary.get("memory", {}).get("best_qpruner_storage_reduction_pct"))
        ),
        "",
        "## Memory-preserving QPruner runtime",
        "",
        "- Best memory-preserving QPruner runtime `{label}`: strategy `{strategy}`, cache mode `{mode}`, QPruner {speedup}x baseline, code-cache storage reduction {storage}%, dense cached weight modules {dense_modules}.".format(
            label=memory_best.get("label", "missing"),
            strategy=memory_best.get("qpruner_runtime_strategy", "missing"),
            mode=memory_best.get("qpruner_cache_mode", "missing"),
            speedup=fmt(memory_best.get("qpruner_vs_baseline_speedup")),
            storage=fmt(
                memory_best.get("qpruner_code_cache_storage_reduction_pct")
                if memory_best.get("qpruner_code_cache_storage_reduction_pct") is not None
                else memory_best.get("qpruner_storage_reduction_pct")
            ),
            dense_modules=fmt(memory_best.get("qpruner_cached_dense_weight_modules")),
        ),
        "",
        "## Runtime Diagnosis",
        "",
        "- Baseline decode scaling {base}x; CAP {cap}x; QPruner {qpruner}x.".format(
            base=fmt(summary.get("runtime_diagnosis", {}).get("baseline_decode_scaling")),
            cap=fmt(summary.get("runtime_diagnosis", {}).get("cap_decode_scaling")),
            qpruner=fmt(summary.get("runtime_diagnosis", {}).get("qpruner_decode_scaling")),
        ),
        "- CAP scaling gap {cap_gap}x; QPruner scaling gap {q_gap}x vs baseline.".format(
            cap_gap=fmt(summary.get("runtime_diagnosis", {}).get("cap_scaling_gap_vs_baseline")),
            q_gap=fmt(summary.get("runtime_diagnosis", {}).get("qpruner_scaling_gap_vs_baseline")),
        ),
        "- {tradeoff}".format(
            tradeoff=summary.get("runtime_diagnosis", {}).get("cap_runtime_tradeoff", "missing"),
        ),
        "- {tradeoff}".format(
            tradeoff=summary.get("runtime_diagnosis", {}).get("cache_tradeoff", "missing"),
        ),
        "- {tradeoff}".format(
            tradeoff=summary.get("runtime_diagnosis", {}).get(
                "memory_preserving_qpruner_tradeoff", "missing"
            ),
        ),
        "- Next action: {action}".format(
            action=summary.get("runtime_diagnosis", {}).get("next_action", "missing"),
        ),
        "",
        "## Sweep Table",
        "",
        "| Run label | Max new tokens | Cache | CAP strategy | Baseline tok/s | CAP tok/s | CAP speedup | QPruner tok/s | QPruner speedup | Peak MB |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.get("rows", []):
        lines.append(
            "| {label} | {tokens} | {cache} | {cap_strategy} | {base} | {cap} | {cap_speedup} | {q} | {q_speedup} | {peak} |".format(
                label=row["label"],
                tokens=row["max_new_tokens"],
                cache=row["inference_cache_enabled"],
                cap_strategy=fmt(row.get("cap_runtime_strategy")),
                base=fmt(row.get("baseline_tokens_per_s")),
                cap=fmt(row.get("cap_tokens_per_s")),
                cap_speedup=fmt(row.get("cap_vs_baseline_speedup")),
                q=fmt(row.get("qpruner_tokens_per_s")),
                q_speedup=fmt(row.get("qpruner_vs_baseline_speedup")),
                peak=fmt(row.get("peak_mem_mb")),
            )
        )
    lines.extend(
        [
            "",
            "## Evidence",
            "",
            "- `reports/compressed-native-cache-long-decode-sweep.svg`",
            "- `reports/compressed-native-cache-long-decode-sweep.csv`",
            "- `artifacts/compressed_native_sweep_summary.json`",
        ]
    )
    return "\n".join(lines) + "\n"


def csv_text(summary: dict[str, Any]) -> str:
    fields = [
        "run_label",
        "max_new_tokens",
        "inference_cache_enabled",
        "qpruner_cache_mode",
        "cap_runtime_strategy",
        "cap_runtime_storage_format",
        "cap_dense_sparse_buffers",
        "baseline_tokens_per_s",
        "cap_tokens_per_s",
        "cap_vs_baseline_speedup",
        "qpruner_tokens_per_s",
        "qpruner_vs_baseline_speedup",
        "qpruner_runtime_strategy",
        "qpruner_storage_reduction_pct",
        "qpruner_code_cache_storage_reduction_pct",
        "qpruner_cached_dense_weight_modules",
        "qpruner_cached_code_modules",
        "qpruner_scaled_code_matmul_modules",
        "qpruner_shape_aware_cache_modules",
        "peak_mem_mb",
    ]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for row in summary.get("rows", []):
        writer.writerow(
            {
                "run_label": row.get("label"),
                **{field: row.get(field) for field in fields if field != "run_label"},
            }
        )
    return output.getvalue()


def svg(summary: dict[str, Any]) -> str:
    rows = summary.get("rows", [])
    width = 1080
    height = 520
    left = 250
    top = 92
    bar_width = 590
    row_h = 60
    max_speedup = max(
        [float(row.get("qpruner_vs_baseline_speedup") or 0.0) for row in rows]
        + [float(row.get("cap_vs_baseline_speedup") or 0.0) for row in rows]
        + [1.0],
    )
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Compressed-native cache long decode sweep">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#111827}.muted{fill:#6b7280}.grid{stroke:#e5e7eb}.baseline{stroke:#111827;stroke-width:2;stroke-dasharray:6 5}",
        "</style>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        '<text x="64" y="44" font-size="27" font-weight="700">Compressed-native cache/long-decode sweep</text>',
        '<text x="64" y="70" font-size="14" class="muted">Speedup vs dense baseline; 1.0x line marks latency-positive compressed serving</text>',
    ]
    for tick in range(6):
        x = left + (tick / 5) * bar_width
        value = (tick / 5) * max_speedup
        parts.extend(
            [
                f'<line class="grid" x1="{x:.1f}" y1="{top - 8}" x2="{x:.1f}" y2="{top + max(len(rows), 1) * row_h}"/>',
                f'<text x="{x:.1f}" y="{top - 18}" font-size="11" text-anchor="middle" class="muted">{value:.2f}x</text>',
            ]
        )
    one_x = left + min(1.0 / max_speedup, 1.0) * bar_width
    parts.append(f'<line class="baseline" x1="{one_x:.1f}" y1="{top - 8}" x2="{one_x:.1f}" y2="{top + max(len(rows), 1) * row_h}"/>')
    for index, row in enumerate(rows):
        y = top + index * row_h
        label = row["label"]
        if row["inference_cache_enabled"] and row["max_new_tokens"] > 1:
            tag = "cache on / long decode"
        elif row["inference_cache_enabled"]:
            tag = "cache on"
        else:
            tag = "cache off"
        cap_w = _bar(row.get("cap_vs_baseline_speedup"), max_speedup, bar_width)
        q_w = _bar(row.get("qpruner_vs_baseline_speedup"), max_speedup, bar_width)
        parts.extend(
            [
                f'<text x="64" y="{y + 18}" font-size="14" font-weight="700">{html.escape(label)}</text>',
                f'<text x="64" y="{y + 38}" font-size="12" class="muted">{tag}, max_new_tokens={row["max_new_tokens"]}</text>',
                f'<rect x="{left}" y="{y}" width="{cap_w:.1f}" height="18" fill="#2563eb"><title>CAP {fmt(row.get("cap_vs_baseline_speedup"))}x</title></rect>',
                f'<rect x="{left}" y="{y + 23}" width="{q_w:.1f}" height="18" fill="#dc2626"><title>QPruner {fmt(row.get("qpruner_vs_baseline_speedup"))}x</title></rect>',
                f'<text x="{left + cap_w + 8:.1f}" y="{y + 14}" font-size="12">CAP {fmt(row.get("cap_vs_baseline_speedup"))}x</text>',
                f'<text x="{left + q_w + 8:.1f}" y="{y + 37}" font-size="12">QPruner {fmt(row.get("qpruner_vs_baseline_speedup"))}x</text>',
            ]
        )
    parts.extend(
        [
            f'<rect x="870" y="108" width="150" height="92" fill="#f9fafb" stroke="#e5e7eb"/>',
            '<rect x="890" y="132" width="20" height="12" fill="#2563eb"/><text x="920" y="143" font-size="13">CAP</text>',
            '<rect x="890" y="160" width="20" height="12" fill="#dc2626"/><text x="920" y="171" font-size="13">QPruner</text>',
            f'<text x="64" y="{height - 34}" font-size="13" class="muted">{html.escape(str(summary.get("readout", "")))}</text>',
        ]
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _bar(value: Any, max_value: float, width: float) -> float:
    number = _number(value)
    if number is None or max_value <= 0:
        return 0.0
    return max(1.0, number / max_value * width)


def write_outputs(summary: dict[str, Any], demo_root: Path) -> None:
    artifacts = demo_root / "artifacts"
    reports = demo_root / "reports"
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    (artifacts / OUT_JSON).write_text(json.dumps(summary, indent=2, sort_keys=True))
    (reports / OUT_MD).write_text(markdown(summary))
    (reports / OUT_CSV).write_text(csv_text(summary))
    (reports / OUT_SVG).write_text(svg(summary))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize compressed-native cache/long-decode sweep artifacts")
    parser.add_argument("--demo-root", type=Path, default=Path("/mnt/nvme/622/tidal-demo"))
    parser.add_argument("artifacts", nargs="+", type=Path)
    args = parser.parse_args(argv)
    summary = build_summary(args.artifacts, demo_root=args.demo_root)
    write_outputs(summary, args.demo_root)
    print(
        "COMPRESSED_NATIVE_SWEEP_SUMMARY {json} {markdown} {csv} {svg}".format(
            json=args.demo_root / "artifacts" / OUT_JSON,
            markdown=args.demo_root / "reports" / OUT_MD,
            csv=args.demo_root / "reports" / OUT_CSV,
            svg=args.demo_root / "reports" / OUT_SVG,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
