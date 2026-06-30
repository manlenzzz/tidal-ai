#!/usr/bin/env python
"""Diagnose why the current Ascend serving path is not yet showing speedup."""
from __future__ import annotations

import argparse
import json
import platform
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


METHODS = ("baseline", "cap", "qpruner")
VLLM_BENCHMARK_ARTIFACTS = (
    "vllm_serving_benchmark_tiny_qwen3_serving_vllm_metadata_shim_npu.json",
    "vllm_serving_benchmark_tiny_qwen3_serving_vllm_selector_shim_npu.json",
)
TORCH_FALLBACK_ARTIFACT = "torch_serving_fallback_tiny_qwen3_serving_fallback_npu.json"
COMPRESSED_NATIVE_ARTIFACTS = (
    "compressed_native_torch_serving_tiny_qwen3_native_serving_scaled_code_matmul_npu.json",
    "compressed_native_torch_serving_tiny_qwen3_native_serving_code_cache_npu.json",
    "compressed_native_torch_serving_tiny_qwen3_native_serving_npu.json",
)
QPRUNER_PACKED_DECODE_ARTIFACT = "qpruner_packed_decode_benchmark_tiny_qwen3_qpruner_decode_npu.json"
QPRUNER_PACKED_DECODE_SHAPE_SWEEP_ARTIFACT = "qpruner_packed_decode_benchmark_qwen3_06b_shape_sweep_npu.json"
QPRUNER_PACKED_DECODE_SHAPE_SWEEP_PATTERN = "qpruner_packed_decode_benchmark_qwen3_06b*shape_sweep*_npu.json"
QWEN_QPRUNER_NATIVE_PROFILE_ARTIFACT = "qwen_qpruner_native_profile_qwen3_06b_native_profile_npu.json"
QWEN_QPRUNER_NATIVE_PROFILE_PATTERN = "qwen_qpruner_native_profile_*.json"
QWEN_QPRUNER_GROUPED_REPLAY_PATTERN = "qwen_qpruner_grouped_replay_*.json"
QWEN_GENERATE_ARTIFACT = "qwen_compression_generate_qwen3_06b_generate_npu.json"
QWEN_QUALITY_ARTIFACT = "qwen_compression_quality_qwen3_06b_quality_npu.json"
NPU_OCCUPANCY_ARTIFACT = "current_npu_occupancy.json"


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def first_existing_json(artifacts: Path, names: Sequence[str]) -> tuple[str | None, dict[str, Any] | None]:
    for name in names:
        payload = read_json(artifacts / name)
        if payload is not None:
            return name, payload
    return None, None


def best_benchmark_artifact(artifacts: Path, pattern: str, fallback_names: Sequence[str]) -> tuple[str | None, dict[str, Any] | None]:
    candidates: list[tuple[tuple[int, int, float], str, dict[str, Any]]] = []
    seen: set[str] = set()
    paths = list(artifacts.glob(pattern))
    paths.extend(artifacts / name for name in fallback_names)
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
        candidates.append(((status_score, max_new_tokens, path.stat().st_mtime), path.name, payload))
    if not candidates:
        return None, None
    _, name, payload = max(candidates, key=lambda row: row[0])
    return name, payload


def best_qpruner_shape_sweep_artifact(artifacts: Path) -> tuple[str | None, dict[str, Any] | None]:
    candidates: list[tuple[tuple[int, int, float], str, dict[str, Any]]] = []
    seen: set[str] = set()
    paths = list(artifacts.glob(QPRUNER_PACKED_DECODE_SHAPE_SWEEP_PATTERN))
    paths.append(artifacts / QPRUNER_PACKED_DECODE_SHAPE_SWEEP_ARTIFACT)
    for path in paths:
        if path.name in seen:
            continue
        seen.add(path.name)
        payload = read_json(path)
        if payload is None:
            continue
        summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
        status_score = 1 if payload.get("status") == "PASS" else 0
        try:
            shape_count = int(summary.get("shape_count") or 0)
        except (TypeError, ValueError):
            shape_count = 0
        if shape_count <= 0 and isinstance(payload.get("shape_sweep"), list):
            shape_count = len(payload["shape_sweep"])
        candidates.append(((status_score, shape_count, path.stat().st_mtime), path.name, payload))
    if not candidates:
        return None, None
    _, name, payload = max(candidates, key=lambda row: row[0])
    return name, payload


def _qwen_native_profile_speedup(payload: dict[str, Any]) -> float:
    qpruner = payload.get("qpruner", {}) if isinstance(payload.get("qpruner"), dict) else {}
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
    for value in (summary.get("qpruner_vs_baseline_speedup"), qpruner.get("latency_speedup")):
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _qwen_native_profile_code_cache_storage_reduction(payload: dict[str, Any]) -> float:
    qpruner = payload.get("qpruner", {}) if isinstance(payload.get("qpruner"), dict) else {}
    try:
        return float(qpruner.get("code_cache_storage_reduction_pct"))
    except (TypeError, ValueError):
        return -999.0


def _qwen_native_profile_release_packed(payload: dict[str, Any]) -> int:
    qpruner = payload.get("qpruner", {}) if isinstance(payload.get("qpruner"), dict) else {}
    return int(bool(payload.get("qpruner_release_packed_after_cache") or qpruner.get("release_packed_after_cache")))


def _qwen_native_profile_stable_measurement(payload: dict[str, Any]) -> int:
    try:
        iters = int(payload.get("iters") or 0)
    except (TypeError, ValueError):
        iters = 0
    try:
        warmup = int(payload.get("warmup") or 0)
    except (TypeError, ValueError):
        warmup = 0
    return int(iters >= 3 and warmup >= 1)


def _qwen_native_profile_paired_measurement(payload: dict[str, Any]) -> int:
    try:
        paired_rounds = int(payload.get("paired_rounds") or 0)
    except (TypeError, ValueError):
        paired_rounds = 0
    paired_summary = payload.get("paired_summary", {}) if isinstance(payload.get("paired_summary"), dict) else {}
    try:
        paired_samples = int(paired_summary.get("samples") or 0)
    except (TypeError, ValueError):
        paired_samples = 0
    measurement_basis = (
        payload.get("summary", {}).get("measurement_basis")
        if isinstance(payload.get("summary"), dict)
        else None
    )
    return int(paired_rounds >= 3 and paired_samples >= paired_rounds * 2 and measurement_basis == "paired_interleaved_median")


def _qwen_native_profile_measurement_quality(payload: dict[str, Any]) -> int:
    if _qwen_native_profile_paired_measurement(payload):
        return 2
    if _qwen_native_profile_stable_measurement(payload):
        return 1
    return 0


def best_qwen_qpruner_native_profile_artifact(artifacts: Path) -> tuple[str | None, dict[str, Any] | None]:
    candidates: list[tuple[tuple[int, int, int, int, float, float], str, dict[str, Any]]] = []
    seen: set[str] = set()
    paths = list(artifacts.glob(QWEN_QPRUNER_NATIVE_PROFILE_PATTERN))
    paths.append(artifacts / QWEN_QPRUNER_NATIVE_PROFILE_ARTIFACT)
    for path in paths:
        if path.name in seen:
            continue
        seen.add(path.name)
        payload = read_json(path)
        if payload is None:
            continue
        status_score = 1 if payload.get("status") == "PASS" else 0
        qpruner = payload.get("qpruner", {}) if isinstance(payload.get("qpruner"), dict) else {}
        try:
            target_layers = int(qpruner.get("targeted_layers") or payload.get("target_layer_limit") or 0)
        except (TypeError, ValueError):
            target_layers = 0
        try:
            max_new_tokens = int(payload.get("max_new_tokens") or 0)
        except (TypeError, ValueError):
            max_new_tokens = 0
        candidates.append(
            (
                (
                    status_score,
                    target_layers,
                    max_new_tokens,
                    _qwen_native_profile_measurement_quality(payload),
                    _qwen_native_profile_speedup(payload),
                    path.stat().st_mtime,
                ),
                path.name,
                payload,
            )
        )
    if not candidates:
        return None, None
    _, name, payload = max(candidates, key=lambda row: row[0])
    return name, payload


def best_qwen_qpruner_native_memory_profile_artifact(artifacts: Path) -> tuple[str | None, dict[str, Any] | None]:
    candidates: list[tuple[tuple[int, int, int, float, int, int, float, float], str, dict[str, Any]]] = []
    seen: set[str] = set()
    paths = list(artifacts.glob(QWEN_QPRUNER_NATIVE_PROFILE_PATTERN))
    paths.append(artifacts / QWEN_QPRUNER_NATIVE_PROFILE_ARTIFACT)
    for path in paths:
        if path.name in seen:
            continue
        seen.add(path.name)
        payload = read_json(path)
        if payload is None:
            continue
        status_score = 1 if payload.get("status") == "PASS" else 0
        qpruner = payload.get("qpruner", {}) if isinstance(payload.get("qpruner"), dict) else {}
        try:
            target_layers = int(qpruner.get("targeted_layers") or payload.get("target_layer_limit") or 0)
        except (TypeError, ValueError):
            target_layers = 0
        try:
            max_new_tokens = int(payload.get("max_new_tokens") or 0)
        except (TypeError, ValueError):
            max_new_tokens = 0
        candidates.append(
            (
                (
                    status_score,
                    target_layers,
                    max_new_tokens,
                    _qwen_native_profile_code_cache_storage_reduction(payload),
                    _qwen_native_profile_release_packed(payload),
                    _qwen_native_profile_measurement_quality(payload),
                    _qwen_native_profile_speedup(payload),
                    path.stat().st_mtime,
                ),
                path.name,
                payload,
            )
        )
    if not candidates:
        return None, None
    _, name, payload = max(candidates, key=lambda row: row[0])
    return name, payload


def _qwen_native_profile_speed_first_tradeoff(payload: dict[str, Any]) -> int:
    qpruner = payload.get("qpruner", {}) if isinstance(payload.get("qpruner"), dict) else {}
    return int(
        bool(payload.get("qpruner_prebuild_scaled_code_dtype_cache"))
        or as_float(qpruner.get("cached_scaled_code_bytes")) not in (None, 0)
        or bool(payload.get("qpruner_dense_cache_budget_bytes"))
        or as_float(qpruner.get("cached_dense_weight_bytes")) not in (None, 0)
    )


def best_qwen_qpruner_native_speed_first_profile_artifact(artifacts: Path) -> tuple[str | None, dict[str, Any] | None]:
    candidates: list[tuple[tuple[int, int, int, int, int, float, float], str, dict[str, Any]]] = []
    seen: set[str] = set()
    paths = list(artifacts.glob(QWEN_QPRUNER_NATIVE_PROFILE_PATTERN))
    paths.append(artifacts / QWEN_QPRUNER_NATIVE_PROFILE_ARTIFACT)
    for path in paths:
        if path.name in seen:
            continue
        seen.add(path.name)
        payload = read_json(path)
        if payload is None:
            continue
        status_score = 1 if payload.get("status") == "PASS" else 0
        qpruner = payload.get("qpruner", {}) if isinstance(payload.get("qpruner"), dict) else {}
        try:
            target_layers = int(qpruner.get("targeted_layers") or payload.get("target_layer_limit") or 0)
        except (TypeError, ValueError):
            target_layers = 0
        try:
            max_new_tokens = int(payload.get("max_new_tokens") or 0)
        except (TypeError, ValueError):
            max_new_tokens = 0
        candidates.append(
            (
                (
                    status_score,
                    target_layers,
                    max_new_tokens,
                    _qwen_native_profile_speed_first_tradeoff(payload),
                    _qwen_native_profile_measurement_quality(payload),
                    _qwen_native_profile_speedup(payload),
                    path.stat().st_mtime,
                ),
                path.name,
                payload,
            )
        )
    if not candidates:
        return None, None
    _, name, payload = max(candidates, key=lambda row: row[0])
    return name, payload


def _qwen_grouped_replay_best_speedup(payload: dict[str, Any]) -> float:
    best_group = payload.get("best_group", {}) if isinstance(payload.get("best_group"), dict) else {}
    grouped = (
        best_group.get("grouped_scaled_code_matmul", {})
        if isinstance(best_group.get("grouped_scaled_code_matmul"), dict)
        else {}
    )
    return as_float(grouped.get("speedup_vs_sequential_scaled_code")) or 0.0


def best_qwen_qpruner_grouped_replay_artifact(artifacts: Path) -> tuple[str | None, dict[str, Any] | None]:
    candidates: list[tuple[tuple[int, int, int, float, float], str, dict[str, Any]]] = []
    for path in artifacts.glob(QWEN_QPRUNER_GROUPED_REPLAY_PATTERN):
        payload = read_json(path)
        if payload is None:
            continue
        status_score = 1 if payload.get("status") == "PASS" else 0
        try:
            target_layers = int(payload.get("target_layer_limit") or 0)
        except (TypeError, ValueError):
            target_layers = 0
        try:
            group_count = int(payload.get("group_count") or 0)
        except (TypeError, ValueError):
            group_count = 0
        candidates.append(
            (
                (
                    status_score,
                    target_layers,
                    group_count,
                    _qwen_grouped_replay_best_speedup(payload),
                    path.stat().st_mtime,
                ),
                path.name,
                payload,
            )
        )
    if not candidates:
        return None, None
    _, name, payload = max(candidates, key=lambda row: row[0])
    return name, payload


def qwen_qpruner_native_bits_tradeoff(artifacts: Path) -> list[dict[str, Any]]:
    candidates: dict[int, tuple[tuple[int, int, float, float], str, dict[str, Any]]] = {}
    seen: set[str] = set()
    paths = list(artifacts.glob(QWEN_QPRUNER_NATIVE_PROFILE_PATTERN))
    paths.append(artifacts / QWEN_QPRUNER_NATIVE_PROFILE_ARTIFACT)
    for path in paths:
        if path.name in seen:
            continue
        seen.add(path.name)
        payload = read_json(path)
        if payload is None or payload.get("status") != "PASS":
            continue
        qpruner = payload.get("qpruner", {}) if isinstance(payload.get("qpruner"), dict) else {}
        try:
            target_layers = int(qpruner.get("targeted_layers") or payload.get("target_layer_limit") or 0)
            max_new_tokens = int(payload.get("max_new_tokens") or 0)
        except (TypeError, ValueError):
            continue
        if target_layers < 196 or max_new_tokens < 16:
            continue
        if not bool(payload.get("qpruner_release_packed_after_cache") or qpruner.get("release_packed_after_cache")):
            continue
        average_bits = as_float(qpruner.get("average_bits") or payload.get("qpruner_average_bits"))
        if average_bits is None:
            continue
        bits_bucket = int(round(average_bits))
        if bits_bucket not in {2, 3, 4, 5, 6, 7, 8}:
            continue
        score = (
            _qwen_native_profile_measurement_quality(payload),
            max_new_tokens,
            _qwen_native_profile_speedup(payload),
            path.stat().st_mtime,
        )
        current = candidates.get(bits_bucket)
        if current is None or score > current[0]:
            candidates[bits_bucket] = (score, path.name, payload)
    rows: list[dict[str, Any]] = []
    for bits_bucket in sorted(candidates):
        _, name, payload = candidates[bits_bucket]
        row = qwen_qpruner_native_profile_summary(name, payload)
        row["bits_label"] = f"bits{bits_bucket}"
        rows.append(row)
    return rows


def qpruner_cache_mode(payload: dict[str, Any] | None) -> str | None:
    if not isinstance(payload, dict):
        return None
    qpruner = payload.get("qpruner", {}) if isinstance(payload.get("qpruner"), dict) else {}
    metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}
    for value in (payload.get("qpruner_cache_mode"), metadata.get("qpruner_cache_mode"), qpruner.get("cache_mode")):
        if value:
            return str(value)
    return None


def qpruner_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(payload, dict) and isinstance(payload.get("qpruner"), dict):
        return payload["qpruner"]
    return {}


def qpruner_runtime_strategy(payload: dict[str, Any] | None) -> str | None:
    qpruner = qpruner_payload(payload)
    value = qpruner.get("runtime_strategy")
    return str(value) if value else None


def qpruner_dense_cached_modules(payload: dict[str, Any] | None) -> float | None:
    return as_float(qpruner_payload(payload).get("cached_dense_weight_modules"))


def uses_qpruner_dense_weight_cache(payload: dict[str, Any] | None) -> bool:
    if qpruner_runtime_strategy(payload) == "dense_weight_cache":
        return True
    dense_modules = qpruner_dense_cached_modules(payload)
    if dense_modules is not None and dense_modules > 0:
        return True
    return qpruner_cache_mode(payload) == "dense"


def is_memory_preserving_compressed_native(payload: dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    if payload.get("serving_dense_export") is True:
        return False
    if uses_qpruner_dense_weight_cache(payload):
        return False
    qpruner = qpruner_payload(payload)
    return any(
        qpruner.get(key) is not None
        for key in (
            "targeted_storage_reduction_pct",
            "code_cache_storage_reduction_pct",
            "targeted_param_reduction_pct",
        )
    )


def qpruner_shape_policy_match_count(payload: dict[str, Any] | None) -> int:
    qpruner = qpruner_payload(payload)
    plan = qpruner.get("shape_aware_plan", {}) if isinstance(qpruner.get("shape_aware_plan"), dict) else {}
    try:
        return int(plan.get("shape_policy_match_count") or 0)
    except (TypeError, ValueError):
        return 0


def best_compressed_native_artifact(artifacts: Path) -> tuple[str | None, dict[str, Any] | None]:
    candidates: list[tuple[tuple[int, int, int, int, float, int, float], str, dict[str, Any]]] = []
    seen: set[str] = set()
    paths = list(artifacts.glob("compressed_native_torch_serving_*.json"))
    paths.extend(artifacts / name for name in COMPRESSED_NATIVE_ARTIFACTS)
    for path in paths:
        if path.name in seen:
            continue
        seen.add(path.name)
        payload = read_json(path)
        if payload is None:
            continue
        status_score = 1 if payload.get("status") == "PASS" else 0
        memory_score = 1 if is_memory_preserving_compressed_native(payload) else 0
        shape_policy_score = 1 if qpruner_shape_policy_match_count(payload) > 0 else 0
        cache_score = {"shape-aware-code": 3, "scaled-code-matmul": 2, "code": 1}.get(str(qpruner_cache_mode(payload)), 0)
        qpruner = qpruner_payload(payload)
        qpruner_tokens = as_float(qpruner.get("tokens_per_s")) or 0.0
        try:
            max_new_tokens = int(payload.get("max_new_tokens") or 0)
        except (TypeError, ValueError):
            max_new_tokens = 0
        candidates.append(
            (
                (
                    status_score,
                    memory_score,
                    shape_policy_score,
                    cache_score,
                    qpruner_tokens,
                    max_new_tokens,
                    path.stat().st_mtime,
                ),
                path.name,
                payload,
            )
        )
    if not candidates:
        return None, None
    _, name, payload = max(candidates, key=lambda row: row[0])
    return name, payload


def fastest_compressed_native_qpruner_artifact(artifacts: Path) -> tuple[str | None, dict[str, Any] | None]:
    candidates: list[tuple[tuple[int, float, float, int, float], str, dict[str, Any]]] = []
    seen: set[str] = set()
    paths = list(artifacts.glob("compressed_native_torch_serving_*.json"))
    paths.extend(artifacts / name for name in COMPRESSED_NATIVE_ARTIFACTS)
    for path in paths:
        if path.name in seen:
            continue
        seen.add(path.name)
        payload = read_json(path)
        if payload is None:
            continue
        qpruner = qpruner_payload(payload)
        status_score = 1 if payload.get("status") == "PASS" and qpruner.get("status", "PASS") == "PASS" else 0
        qpruner_tokens = as_float(qpruner.get("tokens_per_s")) or 0.0
        qpruner_speedup = as_float(qpruner.get("latency_speedup"))
        if qpruner_speedup is None:
            summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
            qpruner_speedup = as_float(summary.get("qpruner_vs_baseline_speedup")) or 0.0
        try:
            max_new_tokens = int(payload.get("max_new_tokens") or 0)
        except (TypeError, ValueError):
            max_new_tokens = 0
        candidates.append(((status_score, qpruner_tokens, qpruner_speedup or 0.0, max_new_tokens, path.stat().st_mtime), path.name, payload))
    if not candidates:
        return None, None
    _, name, payload = max(candidates, key=lambda row: row[0])
    return name, payload


def fmt(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def speedup(numerator: Any, denominator: Any) -> float | None:
    top = as_float(numerator)
    bottom = as_float(denominator)
    if top is None or bottom in (None, 0):
        return None
    return round(top / bottom, 3)


def method_label(method: str) -> str:
    return {"baseline": "baseline", "cap": "CAP", "qpruner": "QPruner"}.get(method, method)


def artifact_ref(name: str | None) -> str:
    return f"artifacts/{name}" if name else "missing"


def best_vllm_parallel_suite(artifacts: Path) -> tuple[str | None, dict[str, Any] | None]:
    candidates: list[tuple[tuple[int, int, float], str, dict[str, Any]]] = []
    for path in artifacts.glob("multicard_parallel_suite_vllm_metadata_sync*_metrics.json"):
        payload = read_json(path)
        if payload is None:
            continue
        status_score = 1 if payload.get("status") == "PASS" else 0
        try:
            world_size = int(payload.get("world_size") or 0)
        except (TypeError, ValueError):
            world_size = 0
        candidates.append(((status_score, world_size, path.stat().st_mtime), path.name, payload))
    if not candidates:
        return None, None
    _, name, payload = max(candidates, key=lambda row: row[0])
    return name, payload


def current_vllm_rerun_artifact(artifacts: Path) -> tuple[str | None, dict[str, Any] | None]:
    candidates: list[tuple[tuple[int, int, int, float], str, dict[str, Any]]] = []
    for path in artifacts.glob("vllm_serving_benchmark_*hbm*.json"):
        payload = read_json(path)
        if payload is None:
            continue
        parallel_score = 1 if payload.get("parallel_methods") else 0
        status_score = 1 if payload.get("status") == "PASS" else 0
        try:
            max_new_tokens = int(payload.get("max_new_tokens") or 0)
        except (TypeError, ValueError):
            max_new_tokens = 0
        candidates.append(((parallel_score, status_score, max_new_tokens, path.stat().st_mtime), path.name, payload))
    if not candidates:
        return None, None
    _, name, payload = max(candidates, key=lambda row: row[0])
    return name, payload


def _root_cause_line(text: str) -> str | None:
    markers = (
        "acl.rt",
        "ModuleNotFoundError:",
        "ImportError:",
        "RuntimeError:",
        "ValueError:",
        "TypeError:",
        "AttributeError:",
    )
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if any(marker in line for marker in markers):
            return line
    return None


def extract_failure_root_cause(payload: dict[str, Any] | None) -> str | None:
    if not isinstance(payload, dict):
        return None
    exports = payload.get("exports", {}) if isinstance(payload.get("exports"), dict) else {}
    text_parts: list[str] = []
    for method in METHODS:
        item = exports.get(method)
        if not isinstance(item, dict) or item.get("status") == "PASS":
            continue
        text_parts.extend(
            str(item.get(key) or "")
            for key in ("stderr", "stderr_tail", "stdout", "stdout_tail", "error", "traceback")
        )
    text_parts.extend(str(payload.get(key) or "") for key in ("stderr", "stdout", "error", "traceback"))
    return _root_cause_line("\n".join(text_parts))


def vllm_rerun_summary(name: str | None, payload: dict[str, Any] | None) -> dict[str, Any]:
    exports = payload.get("exports", {}) if isinstance(payload, dict) and isinstance(payload.get("exports"), dict) else {}
    summary = payload.get("summary", {}) if isinstance(payload, dict) and isinstance(payload.get("summary"), dict) else {}
    failing_methods = [
        method
        for method in METHODS
        if isinstance(exports.get(method), dict) and exports.get(method, {}).get("status") != "PASS"
    ]
    return {
        "artifact": name,
        "status": payload.get("status") if payload else "MISSING",
        "run_label": payload.get("run_label") if payload else None,
        "backend": payload.get("backend") if payload else None,
        "max_new_tokens": payload.get("max_new_tokens") if payload else None,
        "parallel_methods": payload.get("parallel_methods") if payload else None,
        "method_cards": payload.get("method_cards") if payload else None,
        "best_method": summary.get("best_method"),
        "best_tokens_per_s": summary.get("best_tokens_per_s"),
        "tokens_per_s": extract_method_tokens(payload),
        "speedups": extract_speedups(payload),
        "failing_methods": failing_methods,
        "root_cause": extract_failure_root_cause(payload),
    }


def extract_method_tokens(payload: dict[str, Any] | None) -> dict[str, float | None]:
    exports = payload.get("exports", {}) if isinstance(payload, dict) and isinstance(payload.get("exports"), dict) else {}
    return {
        method: as_float(exports.get(method, {}).get("tokens_per_s"))
        if isinstance(exports.get(method), dict)
        else None
        for method in METHODS
    }


def extract_speedups(payload: dict[str, Any] | None) -> dict[str, float | None]:
    summary = payload.get("summary", {}) if isinstance(payload, dict) and isinstance(payload.get("summary"), dict) else {}
    tokens = extract_method_tokens(payload)
    return {
        "cap_vs_baseline": as_float(summary.get("cap_vs_baseline_speedup"))
        if summary.get("cap_vs_baseline_speedup") is not None
        else speedup(tokens.get("cap"), tokens.get("baseline")),
        "qpruner_vs_baseline": as_float(summary.get("qpruner_vs_baseline_speedup"))
        if summary.get("qpruner_vs_baseline_speedup") is not None
        else speedup(tokens.get("qpruner"), tokens.get("baseline")),
    }


def benchmark_summary(name: str | None, payload: dict[str, Any] | None) -> dict[str, Any]:
    summary = payload.get("summary", {}) if isinstance(payload, dict) and isinstance(payload.get("summary"), dict) else {}
    paired_summary = (
        payload.get("paired_summary", {})
        if isinstance(payload, dict) and isinstance(payload.get("paired_summary"), dict)
        else {}
    )
    return {
        "artifact": name,
        "status": payload.get("status") if payload else "MISSING",
        "backend": payload.get("backend") if payload else None,
        "run_label": payload.get("run_label") if payload else None,
        "max_new_tokens": payload.get("max_new_tokens") if payload else None,
        "iters": payload.get("iters") if payload else None,
        "warmup": payload.get("warmup") if payload else None,
        "best_method": summary.get("best_method"),
        "best_tokens_per_s": summary.get("best_tokens_per_s"),
        "tokens_per_s": extract_method_tokens(payload),
        "speedups": extract_speedups(payload),
    }


def parallel_method_bests(payload: dict[str, Any] | None) -> dict[str, float]:
    bests: dict[str, float] = {}
    workers = payload.get("workers", []) if isinstance(payload, dict) else []
    if not isinstance(workers, list):
        return bests
    for worker in workers:
        if not isinstance(worker, dict):
            continue
        metrics = worker.get("metrics", {}) if isinstance(worker.get("metrics"), dict) else {}
        method = metrics.get("method")
        value = as_float(metrics.get("tokens_per_s"))
        if method in METHODS and value is not None:
            bests[str(method)] = max(bests.get(str(method), value), value)
    return bests


def parallel_summary(name: str | None, payload: dict[str, Any] | None) -> dict[str, Any]:
    aggregate = payload.get("aggregate", {}) if isinstance(payload, dict) and isinstance(payload.get("aggregate"), dict) else {}
    return {
        "artifact": name,
        "status": payload.get("status") if payload else "MISSING",
        "run_label": payload.get("run_label") if payload else None,
        "world_size": payload.get("world_size") if payload else None,
        "cards": payload.get("cards", []) if payload else [],
        "start_window_s": payload.get("start_window_s") if payload else None,
        "release_lag_window_s": payload.get("release_lag_window_s") if payload else None,
        "launched_synchronously": payload.get("launched_synchronously") if payload else None,
        "pass_count": aggregate.get("pass_count"),
        "best_vllm_tokens_per_s": aggregate.get("best_vllm_tokens_per_s"),
        "method_bests": parallel_method_bests(payload),
    }


def compression_generate_summary(name: str, payload: dict[str, Any] | None) -> dict[str, Any]:
    cap = payload.get("cap", {}) if isinstance(payload, dict) and isinstance(payload.get("cap"), dict) else {}
    qpruner = payload.get("qpruner", {}) if isinstance(payload, dict) and isinstance(payload.get("qpruner"), dict) else {}
    return {
        "artifact": name,
        "status": payload.get("status") if payload else "MISSING",
        "backend": payload.get("backend") if payload else None,
        "inference_cache_enabled": payload.get("inference_cache_enabled") if payload else None,
        "serving_dense_export": payload.get("serving_dense_export") if payload else None,
        "target_layer_limit": payload.get("target_layer_limit") if payload else None,
        "targeted_layers_total": payload.get("targeted_layers_total") if payload else None,
        "max_new_tokens": payload.get("max_new_tokens") if payload else None,
        "cap_latency_speedup": cap.get("latency_speedup"),
        "qpruner_latency_speedup": qpruner.get("latency_speedup"),
        "cap_cache_modules": cap.get("cache_modules"),
        "qpruner_cache_modules": qpruner.get("cache_modules"),
        "cap_exported_dense_linears": cap.get("exported_dense_linears"),
        "qpruner_exported_dense_linears": qpruner.get("exported_dense_linears"),
        "cap_targeted_compression_ratio": cap.get("targeted_compression_ratio"),
        "qpruner_average_bits": qpruner.get("average_bits"),
    }


def compression_quality_summary(name: str, payload: dict[str, Any] | None) -> dict[str, Any]:
    cap = payload.get("cap", {}) if isinstance(payload, dict) and isinstance(payload.get("cap"), dict) else {}
    qpruner = payload.get("qpruner", {}) if isinstance(payload, dict) and isinstance(payload.get("qpruner"), dict) else {}
    baseline = payload.get("baseline", {}) if isinstance(payload, dict) and isinstance(payload.get("baseline"), dict) else {}
    return {
        "artifact": name,
        "status": payload.get("status") if payload else "MISSING",
        "target_layer_limit": payload.get("target_layer_limit") if payload else None,
        "targeted_layers_total": payload.get("targeted_layers_total") if payload else None,
        "baseline_loss": baseline.get("loss"),
        "cap_loss_delta": cap.get("loss_delta"),
        "qpruner_loss_delta": qpruner.get("loss_delta"),
        "cap_targeted_compression_ratio": cap.get("targeted_compression_ratio"),
        "qpruner_average_bits": qpruner.get("average_bits"),
    }


def compressed_native_summary(name: str | None, payload: dict[str, Any] | None) -> dict[str, Any]:
    baseline = payload.get("baseline", {}) if isinstance(payload, dict) and isinstance(payload.get("baseline"), dict) else {}
    cap = payload.get("cap", {}) if isinstance(payload, dict) and isinstance(payload.get("cap"), dict) else {}
    qpruner = (
        payload.get("qpruner", {}) if isinstance(payload, dict) and isinstance(payload.get("qpruner"), dict) else {}
    )
    qpruner_shape_plan = qpruner.get("shape_aware_plan", {}) if isinstance(qpruner.get("shape_aware_plan"), dict) else {}
    summary = payload.get("summary", {}) if isinstance(payload, dict) and isinstance(payload.get("summary"), dict) else {}
    return {
        "artifact": name,
        "status": payload.get("status") if payload else "MISSING",
        "backend": payload.get("backend") if payload else None,
        "run_label": payload.get("run_label") if payload else None,
        "max_new_tokens": payload.get("max_new_tokens") if payload else None,
        "serving_dense_export": payload.get("serving_dense_export") if payload else None,
        "qpruner_cache_mode": qpruner_cache_mode(payload),
        "best_method": summary.get("best_method"),
        "best_tokens_per_s": summary.get("best_tokens_per_s"),
        "tokens_per_s": {
            "baseline": as_float(baseline.get("tokens_per_s")),
            "cap": as_float(cap.get("tokens_per_s")),
            "qpruner": as_float(qpruner.get("tokens_per_s")),
        },
        "latency_ms": {
            "baseline": as_float(baseline.get("latency_ms")),
            "cap": as_float(cap.get("latency_ms")),
            "qpruner": as_float(qpruner.get("latency_ms")),
        },
        "speedups": {
            "cap_vs_baseline": as_float(summary.get("cap_vs_baseline_speedup"))
            if summary.get("cap_vs_baseline_speedup") is not None
            else speedup(baseline.get("tokens_per_s"), cap.get("tokens_per_s")),
            "qpruner_vs_baseline": as_float(summary.get("qpruner_vs_baseline_speedup"))
            if summary.get("qpruner_vs_baseline_speedup") is not None
            else speedup(baseline.get("tokens_per_s"), qpruner.get("tokens_per_s")),
        },
        "cap_runtime_strategy": cap.get("runtime_strategy"),
        "qpruner_runtime_strategy": qpruner.get("runtime_strategy"),
        "qpruner_cached_dense_weight_modules": as_float(qpruner.get("cached_dense_weight_modules")),
        "qpruner_cached_code_modules": as_float(qpruner.get("cached_code_modules")),
        "qpruner_scaled_code_matmul_modules": as_float(qpruner.get("scaled_code_matmul_modules")),
        "qpruner_shape_policy_source": qpruner_shape_plan.get("shape_policy_source"),
        "qpruner_shape_policy_match_count": qpruner_shape_plan.get("shape_policy_match_count"),
        "qpruner_shape_strategy_source_counts": qpruner_shape_plan.get("strategy_source_counts"),
        "qpruner_runtime_profile": qpruner.get("runtime_profile") if isinstance(qpruner.get("runtime_profile"), dict) else None,
        "cap_memory_reduction_pct": as_float(cap.get("targeted_param_reduction_pct")),
        "qpruner_memory_reduction_pct": as_float(qpruner.get("targeted_param_reduction_pct")),
        "cap_storage_reduction_pct": as_float(cap.get("targeted_storage_reduction_pct")),
        "qpruner_storage_reduction_pct": as_float(qpruner.get("targeted_storage_reduction_pct")),
        "qpruner_code_cache_storage_reduction_pct": as_float(qpruner.get("code_cache_storage_reduction_pct")),
    }


def qwen_qpruner_native_profile_summary(name: str | None, payload: dict[str, Any] | None) -> dict[str, Any]:
    baseline = payload.get("baseline", {}) if isinstance(payload, dict) and isinstance(payload.get("baseline"), dict) else {}
    qpruner = (
        payload.get("qpruner", {}) if isinstance(payload, dict) and isinstance(payload.get("qpruner"), dict) else {}
    )
    summary = payload.get("summary", {}) if isinstance(payload, dict) and isinstance(payload.get("summary"), dict) else {}
    paired_summary = (
        payload.get("paired_summary", {})
        if isinstance(payload, dict) and isinstance(payload.get("paired_summary"), dict)
        else {}
    )
    return {
        "artifact": name,
        "status": payload.get("status") if payload else "MISSING",
        "backend": payload.get("backend") if payload else None,
        "model_id": payload.get("model_id") if payload else None,
        "run_label": payload.get("run_label") if payload else None,
        "max_new_tokens": payload.get("max_new_tokens") if payload else None,
        "iters": payload.get("iters") if payload else None,
        "warmup": payload.get("warmup") if payload else None,
        "paired_rounds": payload.get("paired_rounds") if payload else None,
        "paired_samples": paired_summary.get("samples"),
        "measurement_basis": summary.get("measurement_basis"),
        "target_layer_limit": payload.get("target_layer_limit") if payload else None,
        "targeted_layers_total": payload.get("targeted_layers_total") if payload else None,
        "target_layer_coverage_pct": payload.get("target_layer_coverage_pct") if payload else None,
        "qpruner_cache_mode": qpruner_cache_mode(payload),
        "serving_dense_export": payload.get("serving_dense_export") if payload else None,
        "qpruner_release_packed_after_cache": payload.get("qpruner_release_packed_after_cache") if payload else None,
        "qpruner_cache_aux_tensors": payload.get("qpruner_cache_aux_tensors") if payload else None,
        "qpruner_prebuild_scaled_code_dtype_cache": payload.get("qpruner_prebuild_scaled_code_dtype_cache")
        if payload
        else None,
        "qpruner_dense_cache_budget_bytes": as_float(payload.get("qpruner_dense_cache_budget_bytes")) if payload else None,
        "tokens_per_s": {
            "baseline": as_float(baseline.get("tokens_per_s")),
            "qpruner": as_float(qpruner.get("tokens_per_s")),
        },
        "latency_ms": {
            "baseline": as_float(baseline.get("latency_ms")),
            "qpruner": as_float(qpruner.get("latency_ms")),
        },
        "speedups": {
            "qpruner_vs_baseline": as_float(summary.get("qpruner_vs_baseline_speedup"))
            if summary.get("qpruner_vs_baseline_speedup") is not None
            else as_float(qpruner.get("latency_speedup")),
        },
        "qpruner_runtime_strategy": qpruner.get("runtime_strategy"),
        "qpruner_storage_reduction_pct": as_float(qpruner.get("targeted_storage_reduction_pct")),
        "qpruner_code_cache_storage_reduction_pct": as_float(qpruner.get("code_cache_storage_reduction_pct")),
        "qpruner_code_cache_storage_bytes": as_float(qpruner.get("code_cache_storage_bytes")),
        "qpruner_cached_aux_modules": as_float(qpruner.get("cached_aux_modules")),
        "qpruner_cached_aux_bytes": as_float(qpruner.get("cached_aux_bytes")),
        "qpruner_cached_scale_bytes": as_float(qpruner.get("cached_scale_bytes")),
        "qpruner_cached_bias_bytes": as_float(qpruner.get("cached_bias_bytes")),
        "qpruner_cached_scaled_code_bytes": as_float(qpruner.get("cached_scaled_code_bytes")),
        "qpruner_cached_dense_weight_bytes": as_float(qpruner.get("cached_dense_weight_bytes")),
        "qpruner_dense_cache_budget_plan": qpruner.get("dense_cache_budget_plan")
        if isinstance(qpruner.get("dense_cache_budget_plan"), dict)
        else None,
        "qpruner_live_compressed_payload_storage_bytes": as_float(qpruner.get("live_compressed_payload_storage_bytes")),
        "qpruner_released_packed_code_bytes": as_float(qpruner.get("released_packed_code_bytes")),
        "qpruner_packed_code_bytes": as_float(qpruner.get("packed_code_bytes")),
        "qpruner_release_packed_after_cache": qpruner.get("release_packed_after_cache"),
        "qpruner_runtime_profile": qpruner.get("runtime_profile") if isinstance(qpruner.get("runtime_profile"), dict) else None,
    }


def qpruner_packed_decode_summary(name: str, payload: dict[str, Any] | None) -> dict[str, Any]:
    code_cached = payload.get("code_cached", {}) if isinstance(payload, dict) and isinstance(payload.get("code_cached"), dict) else {}
    scaled = (
        payload.get("scaled_code_matmul", {})
        if isinstance(payload, dict) and isinstance(payload.get("scaled_code_matmul"), dict)
        else {}
    )
    cached = payload.get("cached", {}) if isinstance(payload, dict) and isinstance(payload.get("cached"), dict) else {}
    uncached = payload.get("uncached", {}) if isinstance(payload, dict) and isinstance(payload.get("uncached"), dict) else {}
    grouped = (
        payload.get("grouped_scaled_code_matmul", {})
        if isinstance(payload, dict) and isinstance(payload.get("grouped_scaled_code_matmul"), dict)
        else {}
    )
    sequential_group = (
        payload.get("sequential_scaled_code_matmul_group", {})
        if isinstance(payload, dict) and isinstance(payload.get("sequential_scaled_code_matmul_group"), dict)
        else {}
    )
    return {
        "artifact": name,
        "status": payload.get("status") if payload else "MISSING",
        "backend": payload.get("backend") if payload else None,
        "device": payload.get("device") if payload else None,
        "bits": payload.get("bits") if payload else None,
        "shape": payload.get("shape") if payload else None,
        "storage_reduction_pct": as_float(payload.get("storage_reduction_pct")) if payload else None,
        "code_cache_storage_reduction_pct": as_float(payload.get("code_cache_storage_reduction_pct")) if payload else None,
        "uncached_latency_ms": as_float(uncached.get("latency_ms")),
        "code_cached_latency_ms": as_float(code_cached.get("latency_ms")),
        "code_cached_speedup_vs_uncached": as_float(code_cached.get("speedup_vs_uncached")),
        "scaled_code_latency_ms": as_float(scaled.get("latency_ms")),
        "scaled_code_speedup_vs_uncached": as_float(scaled.get("speedup_vs_uncached")),
        "scaled_code_peak_mem_mb": as_float(scaled.get("peak_mem_mb")),
        "dense_cache_speedup_vs_uncached": as_float(cached.get("speedup_vs_uncached")),
        "grouped_module_count": as_float(grouped.get("module_count")),
        "grouped_sequential_latency_ms": as_float(sequential_group.get("latency_ms")),
        "grouped_latency_ms": as_float(grouped.get("latency_ms")),
        "grouped_scaled_code_speedup_vs_sequential": as_float(grouped.get("speedup_vs_sequential_scaled_code")),
        "grouped_code_cache_bytes": as_float(grouped.get("grouped_code_cache_bytes")),
        "grouped_aux_cache_bytes": as_float(grouped.get("grouped_aux_cache_bytes")),
        "grouped_scaled_code_cache_bytes": as_float(grouped.get("grouped_scaled_code_cache_bytes")),
        "grouped_max_abs_diff_vs_sequential": as_float(
            payload.get("grouped_max_abs_diff_vs_sequential_scaled_code") if payload else None
        ),
        "next_action": payload.get("next_action") if payload else None,
    }


def qpruner_packed_decode_shape_sweep_summary(name: str, payload: dict[str, Any] | None) -> dict[str, Any]:
    summary = payload.get("summary", {}) if isinstance(payload, dict) and isinstance(payload.get("summary"), dict) else {}
    best_memory = (
        summary.get("best_memory_preserving", {})
        if isinstance(summary.get("best_memory_preserving"), dict)
        else {}
    )
    best_latency = summary.get("best_latency", {}) if isinstance(summary.get("best_latency"), dict) else {}
    shape_count = summary.get("shape_count")
    if shape_count is None and isinstance(payload, dict) and isinstance(payload.get("shape_sweep"), list):
        shape_count = len(payload["shape_sweep"])
    return {
        "artifact": name,
        "status": payload.get("status") if payload else "MISSING",
        "backend": payload.get("backend") if payload else None,
        "device": payload.get("device") if payload else None,
        "bits": payload.get("bits") if payload else None,
        "shape_preset": payload.get("shape_preset") if payload else None,
        "shape_count": shape_count,
        "best_memory_preserving": {
            "label": best_memory.get("label"),
            "path": best_memory.get("path"),
            "strategy": best_memory.get("strategy"),
            "latency_ms": as_float(best_memory.get("latency_ms")),
            "speedup_vs_uncached": as_float(best_memory.get("speedup_vs_uncached")),
            "storage_reduction_pct": as_float(best_memory.get("storage_reduction_pct")),
            "code_cache_storage_reduction_pct": as_float(best_memory.get("code_cache_storage_reduction_pct")),
        },
        "best_latency": {
            "label": best_latency.get("label"),
            "path": best_latency.get("path"),
            "strategy": best_latency.get("strategy"),
            "latency_ms": as_float(best_latency.get("latency_ms")),
            "speedup_vs_uncached": as_float(best_latency.get("speedup_vs_uncached")),
        },
        "next_action": payload.get("next_action") if payload else None,
    }


def qwen_qpruner_grouped_replay_summary(name: str | None, payload: dict[str, Any] | None) -> dict[str, Any]:
    best = payload.get("best_group", {}) if isinstance(payload, dict) and isinstance(payload.get("best_group"), dict) else {}
    grouped = (
        best.get("grouped_scaled_code_matmul", {})
        if isinstance(best.get("grouped_scaled_code_matmul"), dict)
        else {}
    )
    sequential = (
        best.get("sequential_scaled_code_matmul_group", {})
        if isinstance(best.get("sequential_scaled_code_matmul_group"), dict)
        else {}
    )
    return {
        "artifact": name,
        "status": payload.get("status") if payload else "MISSING",
        "backend": payload.get("backend") if payload else None,
        "model_id": payload.get("model_id") if payload else None,
        "device": payload.get("device") if payload else None,
        "dtype": payload.get("dtype") if payload else None,
        "target_layer_limit": payload.get("target_layer_limit") if payload else None,
        "targeted_layers_total": payload.get("targeted_layers_total") if payload else None,
        "quantized_layers": payload.get("quantized_layers") if payload else None,
        "cache_modules": payload.get("cache_modules") if payload else None,
        "aux_cache_modules": payload.get("aux_cache_modules") if payload else None,
        "qpruner_release_packed_after_cache": payload.get("qpruner_release_packed_after_cache") if payload else None,
        "qpruner_prebuild_scaled_code_dtype_cache": payload.get("qpruner_prebuild_scaled_code_dtype_cache") if payload else None,
        "targeted_storage_reduction_pct": as_float(payload.get("targeted_storage_reduction_pct")) if payload else None,
        "code_cache_storage_reduction_pct": as_float(payload.get("code_cache_storage_reduction_pct")) if payload else None,
        "cached_scaled_code_bytes": as_float(payload.get("cached_scaled_code_bytes")) if payload else None,
        "cached_aux_bytes": as_float(payload.get("cached_aux_bytes")) if payload else None,
        "group_count": payload.get("group_count") if payload else None,
        "available_group_count": payload.get("available_group_count") if payload else None,
        "best_group": {
            "role": best.get("role"),
            "shape": best.get("shape") if isinstance(best.get("shape"), dict) else None,
            "module_count": best.get("module_count"),
            "available_module_count": best.get("available_module_count"),
            "sample_module_names": best.get("sample_module_names", []),
            "sequential_latency_ms": as_float(sequential.get("latency_ms")),
            "grouped_latency_ms": as_float(grouped.get("latency_ms")),
            "speedup_vs_sequential_scaled_code": as_float(grouped.get("speedup_vs_sequential_scaled_code")),
            "memory_strategy": grouped.get("memory_strategy"),
            "grouped_code_cache_bytes": as_float(best.get("grouped_code_cache_bytes")),
            "grouped_aux_cache_bytes": as_float(best.get("grouped_aux_cache_bytes")),
            "grouped_scaled_code_cache_bytes": as_float(best.get("grouped_scaled_code_cache_bytes")),
            "max_abs_diff_vs_sequential_scaled_code": as_float(best.get("max_abs_diff_vs_sequential_scaled_code")),
        },
        "next_action": payload.get("next_action") if payload else None,
    }


def grouped_shape_row_summary(row: dict[str, Any], *, artifact: str) -> dict[str, Any] | None:
    grouped = row.get("grouped_scaled_code_matmul") if isinstance(row.get("grouped_scaled_code_matmul"), dict) else {}
    if not grouped:
        return None
    sequential = (
        row.get("sequential_scaled_code_matmul_group")
        if isinstance(row.get("sequential_scaled_code_matmul_group"), dict)
        else {}
    )
    return {
        "artifact": artifact,
        "label": row.get("label"),
        "shape": row.get("shape") if isinstance(row.get("shape"), dict) else None,
        "module_count": as_float(grouped.get("module_count")),
        "sequential_latency_ms": as_float(sequential.get("latency_ms")),
        "grouped_latency_ms": as_float(grouped.get("latency_ms")),
        "speedup_vs_sequential_scaled_code": as_float(grouped.get("speedup_vs_sequential_scaled_code")),
        "grouped_code_cache_bytes": as_float(grouped.get("grouped_code_cache_bytes")),
        "grouped_aux_cache_bytes": as_float(grouped.get("grouped_aux_cache_bytes")),
        "grouped_scaled_code_cache_bytes": as_float(grouped.get("grouped_scaled_code_cache_bytes")),
        "max_abs_diff_vs_sequential": as_float(row.get("grouped_max_abs_diff_vs_sequential_scaled_code")),
    }


def qpruner_grouped_shape_sweep_summary(name: str | None, payload: dict[str, Any] | None) -> dict[str, Any]:
    rows = payload.get("shape_sweep", []) if isinstance(payload, dict) and isinstance(payload.get("shape_sweep"), list) else []
    grouped_rows = [
        row_summary
        for row in rows
        if isinstance(row, dict)
        for row_summary in [grouped_shape_row_summary(row, artifact=name or "missing")]
        if row_summary is not None
    ]
    best = max(
        grouped_rows,
        key=lambda item: as_float(item.get("speedup_vs_sequential_scaled_code")) or 0.0,
        default=None,
    )
    return {
        "artifact": name,
        "status": payload.get("status") if payload else "MISSING",
        "backend": payload.get("backend") if payload else None,
        "device": payload.get("device") if payload else None,
        "bits": payload.get("bits") if payload else None,
        "shape_preset": payload.get("shape_preset") if payload else None,
        "grouped_shape_count": len(grouped_rows),
        "grouped_projections": grouped_rows,
        "best_grouped_projection": best,
        "next_action": payload.get("next_action") if payload else None,
    }


def best_grouped_qpruner_shape_sweep_artifact(artifacts: Path) -> tuple[str | None, dict[str, Any] | None]:
    candidates: list[tuple[tuple[int, float, int, float], str, dict[str, Any]]] = []
    seen: set[str] = set()
    paths = list(artifacts.glob(QPRUNER_PACKED_DECODE_SHAPE_SWEEP_PATTERN))
    paths.append(artifacts / QPRUNER_PACKED_DECODE_SHAPE_SWEEP_ARTIFACT)
    for path in paths:
        if path.name in seen:
            continue
        seen.add(path.name)
        payload = read_json(path)
        if payload is None:
            continue
        summary = qpruner_grouped_shape_sweep_summary(path.name, payload)
        best = summary.get("best_grouped_projection")
        if not isinstance(best, dict):
            continue
        best_speedup = as_float(best.get("speedup_vs_sequential_scaled_code")) or 0.0
        if best_speedup <= 0.0:
            continue
        status_score = 1 if payload.get("status") == "PASS" else 0
        try:
            grouped_count = int(summary.get("grouped_shape_count") or 0)
        except (TypeError, ValueError):
            grouped_count = 0
        actual_preset_score = 1 if payload.get("shape_preset") == "qwen3-0.6b-actual" else 0
        candidates.append(
            (
                (status_score, actual_preset_score, grouped_count, best_speedup, path.stat().st_mtime),
                path.name,
                payload,
            )
        )
    if not candidates:
        return None, None
    _, name, payload = max(candidates, key=lambda row: row[0])
    return name, payload


def projection_role_from_name(name: Any) -> str | None:
    text = str(name or "")
    for role in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        if text.endswith(role) or f"_{role}" in text or f".{role}" in text:
            return role
    return None


def _shape_dim(shape: Any, key: str) -> int | None:
    if not isinstance(shape, dict):
        return None
    try:
        return int(shape.get(key))
    except (TypeError, ValueError):
        return None


def grouped_candidate_profile_match(candidate: dict[str, Any], runtime_profile: dict[str, Any]) -> dict[str, Any]:
    top_modules = runtime_profile.get("top_modules", []) if isinstance(runtime_profile, dict) else []
    if not isinstance(top_modules, list):
        top_modules = []
    candidate_role = projection_role_from_name(candidate.get("label"))
    in_features = _shape_dim(candidate.get("shape"), "in_features")
    out_features = _shape_dim(candidate.get("shape"), "out_features")
    hits = 0
    calls = 0.0
    time_ms = 0.0
    names: list[str] = []
    for module in top_modules:
        if not isinstance(module, dict):
            continue
        module_name = str(module.get("name") or "")
        module_role = projection_role_from_name(module_name)
        role_match = candidate_role is not None and module_role == candidate_role
        try:
            module_in = int(module.get("in_features"))
            module_out = int(module.get("out_features"))
        except (TypeError, ValueError):
            module_in = None
            module_out = None
        shape_match = (
            in_features is not None
            and out_features is not None
            and module_in == in_features
            and module_out == out_features
        )
        if not (role_match or shape_match):
            continue
        hits += 1
        calls += as_float(module.get("calls")) or 0.0
        time_ms += as_float(module.get("time_ms")) or 0.0
        if module_name:
            names.append(module_name)
    return {
        "profile_top_module_hits": hits,
        "profile_top_module_calls": calls,
        "profile_top_module_time_ms": round(time_ms, 6),
        "profile_top_module_names": names,
    }


def qpruner_grouped_projection_plan_summary(
    native_profile: dict[str, Any],
    grouped_shape_sweep: dict[str, Any],
) -> dict[str, Any]:
    runtime_profile = (
        native_profile.get("qpruner_runtime_profile", {})
        if isinstance(native_profile.get("qpruner_runtime_profile"), dict)
        else {}
    )
    grouped_rows = (
        grouped_shape_sweep.get("grouped_projections", [])
        if isinstance(grouped_shape_sweep.get("grouped_projections"), list)
        else []
    )
    candidates: list[dict[str, Any]] = []
    for row in grouped_rows:
        if not isinstance(row, dict):
            continue
        speed = as_float(row.get("speedup_vs_sequential_scaled_code"))
        if speed is None or speed <= 1.0:
            continue
        enriched = dict(row)
        enriched.update(grouped_candidate_profile_match(enriched, runtime_profile))
        candidates.append(enriched)
    candidates.sort(
        key=lambda item: (
            int((as_float(item.get("profile_top_module_hits")) or 0) > 0),
            as_float(item.get("speedup_vs_sequential_scaled_code")) or 0.0,
            as_float(item.get("profile_top_module_time_ms")) or 0.0,
        ),
        reverse=True,
    )
    best = candidates[0] if candidates else None
    shape_preset = grouped_shape_sweep.get("shape_preset")
    native_ready = native_profile.get("status") == "PASS" and runtime_profile.get("status") == "PASS"
    grouped_ready = grouped_shape_sweep.get("status") == "PASS" and bool(candidates)
    if not native_ready or not grouped_ready:
        status = "MISSING_EVIDENCE"
    elif shape_preset != "qwen3-0.6b-actual":
        status = "STALE_SHAPE_PRESET"
    else:
        status = "ACTIONABLE"
    return {
        "status": status,
        "profile_artifact": native_profile.get("artifact"),
        "shape_artifact": grouped_shape_sweep.get("artifact"),
        "shape_preset": shape_preset,
        "candidate_count": len(candidates),
        "best_candidate": best,
        "candidates": candidates,
        "full_model": {
            "target_layers": native_profile.get("target_layer_limit"),
            "targeted_layers_total": native_profile.get("targeted_layers_total"),
            "max_new_tokens": native_profile.get("max_new_tokens"),
            "measurement_basis": native_profile.get("measurement_basis"),
            "qpruner_vs_baseline_speedup": (
                native_profile.get("speedups", {}).get("qpruner_vs_baseline")
                if isinstance(native_profile.get("speedups"), dict)
                else None
            ),
            "runtime_strategy": native_profile.get("qpruner_runtime_strategy"),
            "total_forward_calls": runtime_profile.get("total_forward_calls"),
            "total_forward_time_ms": runtime_profile.get("total_forward_time_ms"),
            "forward_share_pct": runtime_profile.get("estimated_forward_share_pct"),
            "code_cache_storage_reduction_pct": native_profile.get("qpruner_code_cache_storage_reduction_pct"),
            "code_cache_storage_bytes": native_profile.get("qpruner_code_cache_storage_bytes"),
            "cached_aux_bytes": native_profile.get("qpruner_cached_aux_bytes"),
            "cached_scaled_code_bytes": native_profile.get("qpruner_cached_scaled_code_bytes"),
            "release_packed_after_cache": native_profile.get("qpruner_release_packed_after_cache"),
        },
        "recommendation": (
            "Prototype the grouped/fused QPruner kernel for the best actual Qwen projection family, "
            "then rerun the full-model native profile before claiming end-to-end speedup."
        ),
    }


def resource_blocker_summary(name: str, payload: dict[str, Any] | None) -> dict[str, Any]:
    container = payload.get("container", {}) if isinstance(payload, dict) and isinstance(payload.get("container"), dict) else {}
    service = payload.get("service", {}) if isinstance(payload, dict) and isinstance(payload.get("service"), dict) else {}
    return {
        "artifact": name,
        "status": payload.get("status") if payload else "MISSING",
        "reason": payload.get("reason") if payload else None,
        "occupied_cards": payload.get("occupied_cards") if payload else None,
        "total_cards": payload.get("total_cards") if payload else None,
        "container_id": container.get("id"),
        "container_name": container.get("name"),
        "container_image": container.get("image"),
        "service_model": service.get("model"),
        "service_command": service.get("command"),
        "tensor_parallel_size": service.get("tensor_parallel_size"),
        "policy": payload.get("policy") if payload else None,
        "next_action": payload.get("next_action") if payload else None,
    }


def diagnosis_entry(identifier: str, title: str, evidence: str, recommendation: str) -> dict[str, str]:
    return {"id": identifier, "title": title, "evidence": evidence, "recommendation": recommendation}


def build_diagnoses(report: dict[str, Any]) -> list[dict[str, str]]:
    diagnoses: list[dict[str, str]] = []
    sequential = report["sequential_vllm"]
    torch_fallback = report["torch_fallback"]
    parallel = report["vllm_parallel_sync"]
    current_vllm = report.get("current_vllm_rerun", {})
    generate = report["compression_generate"]
    compressed_native = report.get("compressed_native", {})
    compressed_native_fastest = report.get("compressed_native_fastest", {})
    packed_decode = report.get("qpruner_packed_decode", {})
    packed_decode_shape_sweep = report.get("qpruner_packed_decode_shape_sweep", {})
    grouped_shape_sweep = report.get("qpruner_grouped_shape_sweep", {})
    real_grouped_replay = report.get("qwen_qpruner_grouped_replay", {})
    qwen_native_profile = report.get("qwen_qpruner_native_profile", {})
    grouped_projection_plan = report.get("qpruner_grouped_projection_plan", {})
    resource_blocker = report.get("resource_blocker", {})
    vllm_speedups = sequential.get("speedups", {})
    vllm_tokens = sequential.get("tokens_per_s", {})
    cap_speedup = as_float(vllm_speedups.get("cap_vs_baseline"))
    qpruner_speedup = as_float(vllm_speedups.get("qpruner_vs_baseline"))
    compressed_speedups = [value for value in (cap_speedup, qpruner_speedup) if value is not None]
    if current_vllm.get("status") == "FAIL":
        diagnoses.append(
            diagnosis_entry(
                "vllm_current_hbm_parallel_rerun_failed",
                "Current vLLM HBM rerun fails before benchmark metrics",
                "artifact {artifact}, parallel_methods={parallel}, method_cards={cards}, failing methods={methods}; "
                "root cause: {root}.".format(
                    artifact=artifact_ref(current_vllm.get("artifact")),
                    parallel=fmt(current_vllm.get("parallel_methods")),
                    cards=fmt(current_vllm.get("method_cards")),
                    methods=", ".join(method_label(method) for method in current_vllm.get("failing_methods", []))
                    or "missing",
                    root=fmt(current_vllm.get("root_cause")),
                ),
                "Treat older vLLM PASS throughput as historical baseline evidence; fix the vLLM-Ascend ACL Python import path before claiming current HBM or multi-card vLLM results.",
            )
        )
    if sequential.get("status") == "PASS" and compressed_speedups and max(compressed_speedups) < 1.03:
        diagnoses.append(
            diagnosis_entry(
                "vllm_compressed_not_faster_than_baseline",
                "vLLM compressed path is not faster than baseline",
                "baseline {baseline} tokens/s, CAP {cap} tokens/s ({cap_speedup}x), "
                "QPruner {qpruner} tokens/s ({q_speedup}x).".format(
                    baseline=fmt(vllm_tokens.get("baseline")),
                    cap=fmt(vllm_tokens.get("cap")),
                    qpruner=fmt(vllm_tokens.get("qpruner")),
                    cap_speedup=fmt(cap_speedup),
                    q_speedup=fmt(qpruner_speedup),
                ),
                "Do not claim vLLM serving speedup yet; use this as the baseline for kernel/export optimization.",
            )
        )
    max_new_tokens = sequential.get("max_new_tokens")
    if as_float(max_new_tokens) is not None and float(max_new_tokens) <= 4:
        diagnoses.append(
            diagnosis_entry(
                "short_decode_hides_steady_state",
                "short decode hides steady-state throughput",
                f"vLLM benchmark uses max_new_tokens={max_new_tokens}; load and scheduler overhead can dominate.",
                "Run longer decode slices such as max_new_tokens=16/32 with warmup and multiple iterations.",
            )
        )
    dense_or_cache = bool(generate.get("serving_dense_export")) or bool(generate.get("inference_cache_enabled"))
    exported_dense = [
        as_float(generate.get("cap_exported_dense_linears")),
        as_float(generate.get("qpruner_exported_dense_linears")),
    ]
    cache_modules = [
        as_float(generate.get("cap_cache_modules")),
        as_float(generate.get("qpruner_cache_modules")),
    ]
    if dense_or_cache or any((value or 0) > 0 for value in exported_dense + cache_modules):
        diagnoses.append(
            diagnosis_entry(
                "serving_export_dense_or_cache_overhead",
                "serving export is dense or cache-heavy",
                "serving_dense_export={dense}, inference_cache_enabled={cache}, "
                "exported_dense_linears CAP/QPruner={cap_dense}/{q_dense}, cache_modules={cap_cache}/{q_cache}.".format(
                    dense=generate.get("serving_dense_export"),
                    cache=generate.get("inference_cache_enabled"),
                    cap_dense=fmt(generate.get("cap_exported_dense_linears")),
                    q_dense=fmt(generate.get("qpruner_exported_dense_linears")),
                    cap_cache=fmt(generate.get("cap_cache_modules")),
                    q_cache=fmt(generate.get("qpruner_cache_modules")),
                ),
                "Inspect the export path and replace dense materialization/cache overhead with compression-aware kernels.",
            )
        )
    torch_q_speedup = as_float(torch_fallback.get("speedups", {}).get("qpruner_vs_baseline"))
    if torch_q_speedup is not None and torch_q_speedup > 1.0 and (qpruner_speedup is None or qpruner_speedup <= 1.0):
        diagnoses.append(
            diagnosis_entry(
                "torch_fallback_qpruner_beats_vllm_qpruner",
                "torch_npu fallback shows QPruner benefit while vLLM does not",
                "torch_npu fallback qpruner_vs_baseline={torch_speedup}x; "
                "vLLM qpruner_vs_baseline={vllm_speedup}x.".format(
                    torch_speedup=fmt(torch_q_speedup),
                    vllm_speedup=fmt(qpruner_speedup),
                ),
                "Use torch_npu as a functional compressed reference and profile the vLLM wrapper/export boundary.",
            )
        )
    native_q_speedup = as_float(compressed_native.get("speedups", {}).get("qpruner_vs_baseline"))
    native_q_storage = as_float(compressed_native.get("qpruner_storage_reduction_pct"))
    if compressed_native.get("status") == "PASS" and native_q_speedup is not None and native_q_speedup < 1.0:
        diagnoses.append(
            diagnosis_entry(
                "compressed_native_qpruner_end_to_end_not_faster",
                "compressed-native QPruner preserves memory but is not end-to-end faster",
                "compressed-native QPruner qpruner_vs_baseline={speedup}x with {storage}% targeted storage "
                "reduction, runtime={runtime}, cache_mode={cache_mode}, shape_policy_matches={matches}, "
                "strategy_sources={sources}.".format(
                    speedup=fmt(native_q_speedup),
                    storage=fmt(native_q_storage),
                    runtime=fmt(compressed_native.get("qpruner_runtime_strategy")),
                    cache_mode=fmt(compressed_native.get("qpruner_cache_mode")),
                    matches=fmt(compressed_native.get("qpruner_shape_policy_match_count")),
                    sources=fmt(compressed_native.get("qpruner_shape_strategy_source_counts")),
                ),
                "Keep the compressed-native path as the memory reference; move the packed decode speedup into full-model kernels.",
            )
        )
    runtime_profile = compressed_native.get("qpruner_runtime_profile", {})
    if isinstance(runtime_profile, dict) and runtime_profile.get("status") == "PASS":
        top_modules = runtime_profile.get("top_modules", [])
        top_module = top_modules[0] if isinstance(top_modules, list) and top_modules and isinstance(top_modules[0], dict) else {}
        diagnoses.append(
            diagnosis_entry(
                "qpruner_runtime_profile_forward_overhead",
                "QPruner runtime profile shows full-model QuantizedLinear overhead",
                "QPruner runtime profile: {calls} QuantizedLinear forwards take {time} ms "
                "({share}% of measured generate latency); runtime={runtime}; strategy timing={strategies}; "
                "top module {top} uses {strategy} for {top_calls} calls at {top_time} ms.".format(
                    calls=fmt(runtime_profile.get("total_forward_calls")),
                    time=fmt(runtime_profile.get("total_forward_time_ms")),
                    share=fmt(runtime_profile.get("estimated_forward_share_pct")),
                    runtime=fmt(runtime_profile.get("runtime_strategy")),
                    strategies=fmt(runtime_profile.get("by_strategy")),
                    top=fmt(top_module.get("name")),
                    strategy=fmt(top_module.get("strategy")),
                    top_calls=fmt(top_module.get("calls")),
                    top_time=fmt(top_module.get("time_ms")),
                ),
                "Use the profile to target fused QPruner projection kernels or batched dequantization instead of only changing cache policy.",
            )
        )
    qwen_profile = qwen_native_profile.get("qpruner_runtime_profile", {})
    if isinstance(qwen_profile, dict) and qwen_profile.get("status") == "PASS":
        top_modules = qwen_profile.get("top_modules", [])
        top_module = top_modules[0] if isinstance(top_modules, list) and top_modules and isinstance(top_modules[0], dict) else {}
        diagnoses.append(
            diagnosis_entry(
                "qwen_qpruner_native_profile_confirms_model_overhead",
                "Qwen3 QPruner native profile confirms full-model overhead",
                "Qwen3 QPruner native profile: target layers {limit}/{total}, max_new_tokens={tokens}, "
                "qpruner_vs_baseline={speedup}x, "
                "storage reduction {storage}%, code-cache storage reduction {code_storage}%, "
                "aux-cache bytes={aux_bytes}; "
                "{calls} QuantizedLinear forwards take {time} ms ({share}% of measured generate latency); "
                "top module {top} uses {strategy} for {top_calls} calls at {top_time} ms.".format(
                    limit=fmt(qwen_native_profile.get("target_layer_limit")),
                    total=fmt(qwen_native_profile.get("targeted_layers_total")),
                    tokens=fmt(qwen_native_profile.get("max_new_tokens")),
                    speedup=fmt(qwen_native_profile.get("speedups", {}).get("qpruner_vs_baseline")),
                    storage=fmt(qwen_native_profile.get("qpruner_storage_reduction_pct")),
                    code_storage=fmt(qwen_native_profile.get("qpruner_code_cache_storage_reduction_pct")),
                    aux_bytes=fmt(qwen_native_profile.get("qpruner_cached_aux_bytes")),
                    calls=fmt(qwen_profile.get("total_forward_calls")),
                    time=fmt(qwen_profile.get("total_forward_time_ms")),
                    share=fmt(qwen_profile.get("estimated_forward_share_pct")),
                    top=fmt(top_module.get("name")),
                    strategy=fmt(top_module.get("strategy")),
                    top_calls=fmt(top_module.get("calls")),
                    top_time=fmt(top_module.get("time_ms")),
                ),
                "Use the Qwen3 profile as the next full-model target for fused/batched QuantizedLinear kernels before claiming end-to-end acceleration.",
            )
        )
    qwen_memory_profile = report.get("qwen_qpruner_native_memory_profile", {})
    memory_code_storage = as_float(qwen_memory_profile.get("qpruner_code_cache_storage_reduction_pct"))
    memory_release = bool(qwen_memory_profile.get("qpruner_release_packed_after_cache"))
    if qwen_memory_profile.get("status") == "PASS" and memory_code_storage is not None and memory_code_storage > 0:
        diagnoses.append(
            diagnosis_entry(
                "qwen_qpruner_native_release_packed_memory_positive",
                "Qwen3 QPruner native profile keeps code-cache memory savings after releasing packed codes",
                "Qwen3 memory-first native profile: target layers {limit}/{total}, max_new_tokens={tokens}, "
                "qpruner_vs_baseline={speedup}x, "
                "code-cache storage reduction {code_storage}%, released_packed={release}, "
                "aux-cache bytes={aux_bytes}, released packed bytes={released}, "
                "live payload bytes={live_payload}, runtime={runtime}.".format(
                    limit=fmt(qwen_memory_profile.get("target_layer_limit")),
                    total=fmt(qwen_memory_profile.get("targeted_layers_total")),
                    tokens=fmt(qwen_memory_profile.get("max_new_tokens")),
                    speedup=fmt(qwen_memory_profile.get("speedups", {}).get("qpruner_vs_baseline")),
                    code_storage=fmt(memory_code_storage),
                    release=fmt(memory_release),
                    aux_bytes=fmt(qwen_memory_profile.get("qpruner_cached_aux_bytes")),
                    released=fmt(qwen_memory_profile.get("qpruner_released_packed_code_bytes")),
                    live_payload=fmt(qwen_memory_profile.get("qpruner_live_compressed_payload_storage_bytes")),
                    runtime=fmt(qwen_memory_profile.get("qpruner_runtime_strategy")),
                ),
                "Keep this as the memory-first QPruner runtime evidence; optimize fused/batched kernels separately for speed.",
            )
        )
    fastest_q_speedup = as_float(compressed_native_fastest.get("speedups", {}).get("qpruner_vs_baseline"))
    fastest_runtime = compressed_native_fastest.get("qpruner_runtime_strategy")
    fastest_dense_modules = as_float(compressed_native_fastest.get("qpruner_cached_dense_weight_modules"))
    if (
        compressed_native_fastest.get("status") == "PASS"
        and fastest_q_speedup is not None
        and fastest_q_speedup >= 1.0
        and (fastest_runtime == "dense_weight_cache" or (fastest_dense_modules or 0) > 0)
    ):
        diagnoses.append(
            diagnosis_entry(
                "compressed_native_fastest_uses_dense_weight_cache",
                "Fastest compressed-native QPruner path uses dense weight cache",
                "compressed-native fastest QPruner qpruner_vs_baseline={speedup}x with runtime={runtime}, "
                "cached_dense_weight_modules={modules}; this is a latency reference, not the memory-preserving path.".format(
                    speedup=fmt(fastest_q_speedup),
                    runtime=fmt(fastest_runtime),
                    modules=fmt(fastest_dense_modules),
                ),
                "Report dense-cache latency separately from pruning/quantization memory savings and optimize the code/scaled-code path next.",
            )
        )
    scaled_speedup = as_float(packed_decode.get("scaled_code_speedup_vs_uncached"))
    code_speedup = as_float(packed_decode.get("code_cached_speedup_vs_uncached"))
    best_decode_speedup = max([value for value in (scaled_speedup, code_speedup) if value is not None], default=None)
    if packed_decode.get("status") == "PASS" and best_decode_speedup is not None and best_decode_speedup > 1.0:
        diagnoses.append(
            diagnosis_entry(
                "qpruner_packed_decode_kernel_positive",
                "QPruner packed decode kernel microbenchmark is latency-positive",
                "QPruner packed decode scaled-code speedup {scaled}x, code-cache speedup {code}x, "
                "code-cache storage reduction {storage}%, scaled-code peak {peak} MB.".format(
                    scaled=fmt(scaled_speedup),
                    code=fmt(code_speedup),
                    storage=fmt(packed_decode.get("code_cache_storage_reduction_pct")),
                    peak=fmt(packed_decode.get("scaled_code_peak_mem_mb")),
                ),
                "Use this as the kernel target for the compressed-native and vLLM integration work.",
            )
        )
    grouped_speedup = as_float(packed_decode.get("grouped_scaled_code_speedup_vs_sequential"))
    if packed_decode.get("status") == "PASS" and grouped_speedup is not None and grouped_speedup > 1.0:
        diagnoses.append(
            diagnosis_entry(
                "qpruner_grouped_projection_kernel_positive",
                "QPruner grouped projection diagnostic is latency-positive",
                "QPruner grouped projection diagnostic speedup {speedup}x across {modules} same-shape modules; "
                "grouped code-cache bytes={codes}, aux-cache bytes={aux}, max_abs_diff={diff}.".format(
                    speedup=fmt(grouped_speedup),
                    modules=fmt(packed_decode.get("grouped_module_count")),
                    codes=fmt(packed_decode.get("grouped_code_cache_bytes")),
                    aux=fmt(packed_decode.get("grouped_aux_cache_bytes")),
                    diff=fmt(packed_decode.get("grouped_max_abs_diff_vs_sequential")),
                ),
                "Use this to size the next NPU fused/grouped projection kernel without treating dense caches as compression evidence.",
            )
        )
    grouped_shape_best = (
        grouped_shape_sweep.get("best_grouped_projection", {})
        if isinstance(grouped_shape_sweep.get("best_grouped_projection"), dict)
        else {}
    )
    grouped_shape_speedup = as_float(grouped_shape_best.get("speedup_vs_sequential_scaled_code"))
    if grouped_shape_sweep.get("status") == "PASS" and grouped_shape_speedup is not None and grouped_shape_speedup > 1.0:
        diagnoses.append(
            diagnosis_entry(
                "qpruner_grouped_shape_sweep_positive",
                "QPruner Qwen3 grouped shape sweep identifies a fusion target",
                "QPruner Qwen3 grouped shape sweep best grouped projection {label}: grouped speedup {speedup}x, "
                "sequential {seq} ms, grouped {grouped} ms, grouped code-cache bytes={codes}, aux-cache bytes={aux}.".format(
                    label=fmt(grouped_shape_best.get("label")),
                    speedup=fmt(grouped_shape_speedup),
                    seq=fmt(grouped_shape_best.get("sequential_latency_ms")),
                    grouped=fmt(grouped_shape_best.get("grouped_latency_ms")),
                    codes=fmt(grouped_shape_best.get("grouped_code_cache_bytes")),
                    aux=fmt(grouped_shape_best.get("grouped_aux_cache_bytes")),
                ),
                "Prioritize the listed projection family when prototyping the NPU grouped/fused QPruner kernel.",
            )
        )
    real_replay_best = (
        real_grouped_replay.get("best_group", {})
        if isinstance(real_grouped_replay.get("best_group"), dict)
        else {}
    )
    real_replay_speedup = as_float(real_replay_best.get("speedup_vs_sequential_scaled_code"))
    if real_grouped_replay.get("status") == "PASS" and real_replay_speedup is not None and real_replay_speedup > 1.0:
        diagnoses.append(
            diagnosis_entry(
                "qwen_qpruner_real_grouped_replay_positive",
                "Qwen3 real QPruner module grouped replay is latency-positive",
                "Qwen3 real grouped replay: target layers {limit}/{total}, quantized layers {quantized}, "
                "groups {groups}, best role {role}, grouped speedup {speedup}x, sequential {seq} ms, "
                "grouped {grouped} ms, code-cache storage reduction {code_storage}%, "
                "grouped code-cache bytes={codes}, grouped scaled-code bytes={scaled}, max_abs_diff={diff}.".format(
                    limit=fmt(real_grouped_replay.get("target_layer_limit")),
                    total=fmt(real_grouped_replay.get("targeted_layers_total")),
                    quantized=fmt(real_grouped_replay.get("quantized_layers")),
                    groups=fmt(real_grouped_replay.get("group_count")),
                    role=fmt(real_replay_best.get("role")),
                    speedup=fmt(real_replay_speedup),
                    seq=fmt(real_replay_best.get("sequential_latency_ms")),
                    grouped=fmt(real_replay_best.get("grouped_latency_ms")),
                    code_storage=fmt(real_grouped_replay.get("code_cache_storage_reduction_pct")),
                    codes=fmt(real_replay_best.get("grouped_code_cache_bytes")),
                    scaled=fmt(real_replay_best.get("grouped_scaled_code_cache_bytes")),
                    diff=fmt(real_replay_best.get("max_abs_diff_vs_sequential_scaled_code")),
                ),
                "Use this real-module replay as the next bridge from synthetic shape sweeps to a fused full-model QPruner kernel.",
            )
        )
    plan_best = (
        grouped_projection_plan.get("best_candidate", {})
        if isinstance(grouped_projection_plan.get("best_candidate"), dict)
        else {}
    )
    plan_full_model = (
        grouped_projection_plan.get("full_model", {})
        if isinstance(grouped_projection_plan.get("full_model"), dict)
        else {}
    )
    if grouped_projection_plan.get("status") == "ACTIONABLE" and plan_best:
        diagnoses.append(
            diagnosis_entry(
                "qpruner_full_model_grouped_projection_plan",
                "QPruner full-model grouped projection plan is actionable",
                "QPruner grouped projection plan uses {shape_preset} shapes and full-model profile {profile}; "
                "best candidate {label} speedup {speedup}x, top-module hits {hits}, "
                "forward share {share}%, code-cache storage reduction {code_storage}%, "
                "cached scaled-code bytes {scaled_bytes}.".format(
                    shape_preset=fmt(grouped_projection_plan.get("shape_preset")),
                    profile=artifact_ref(grouped_projection_plan.get("profile_artifact")),
                    label=fmt(plan_best.get("label")),
                    speedup=fmt(plan_best.get("speedup_vs_sequential_scaled_code")),
                    hits=fmt(plan_best.get("profile_top_module_hits")),
                    share=fmt(plan_full_model.get("forward_share_pct")),
                    code_storage=fmt(plan_full_model.get("code_cache_storage_reduction_pct")),
                    scaled_bytes=fmt(plan_full_model.get("cached_scaled_code_bytes")),
                ),
                "Implement the grouped/fused QPruner kernel for this projection family and rerun the full-model native profile.",
            )
        )
    shape_best = (
        packed_decode_shape_sweep.get("best_memory_preserving", {})
        if isinstance(packed_decode_shape_sweep.get("best_memory_preserving"), dict)
        else {}
    )
    shape_best_speedup = as_float(shape_best.get("speedup_vs_uncached"))
    if packed_decode_shape_sweep.get("status") == "PASS" and shape_best_speedup is not None and shape_best_speedup > 1.0:
        diagnoses.append(
            diagnosis_entry(
                "qpruner_qwen_shape_sweep_memory_path_positive",
                "QPruner Qwen3 shape sweep has a latency-positive memory-preserving path",
                "QPruner Qwen3 shape sweep best memory-preserving {strategy} on {label}: "
                "speedup {speedup}x, latency {latency} ms, storage reduction {storage}%, "
                "code-cache storage reduction {code_storage}%.".format(
                    strategy=fmt(shape_best.get("strategy")),
                    label=fmt(shape_best.get("label")),
                    speedup=fmt(shape_best_speedup),
                    latency=fmt(shape_best.get("latency_ms")),
                    storage=fmt(shape_best.get("storage_reduction_pct")),
                    code_storage=fmt(shape_best.get("code_cache_storage_reduction_pct")),
                ),
                "Use the Qwen3 shape sweep to select code-cache vs scaled-code per projection family before vLLM integration.",
            )
        )
    if parallel.get("status") == "PASS":
        diagnoses.append(
            diagnosis_entry(
                "multi_card_sync_ready_not_speedup",
                "multi-card sync is ready but is not itself a speedup claim",
                "{world}-card vLLM synchronized slice PASS with start_window={window}s and pass_count={passes}; "
                "method bests are {methods}.".format(
                    world=fmt(parallel.get("world_size")),
                    window=fmt(parallel.get("start_window_s")),
                    passes=fmt(parallel.get("pass_count")),
                    methods=", ".join(
                        f"{method_label(method)}={fmt(value)} tokens/s"
                        for method, value in sorted(parallel.get("method_bests", {}).items())
                    )
                    or "missing",
                ),
                "Keep the synchronized harness for parallel experiments; optimize the compressed serving path before presenting acceleration.",
            )
        )
    if resource_blocker.get("status") == "BLOCKED":
        diagnoses.append(
            diagnosis_entry(
                "long_decode_waiting_for_free_cards",
                "Long-decode serving validation is waiting for free cards",
                "{occupied}/{total} NPU cards are occupied by container {container} running {model} "
                "with tensor_parallel_size={tp}.".format(
                    occupied=fmt(resource_blocker.get("occupied_cards")),
                    total=fmt(resource_blocker.get("total_cards")),
                    container=fmt(resource_blocker.get("container_name")),
                    model=fmt(resource_blocker.get("service_model")),
                    tp=fmt(resource_blocker.get("tensor_parallel_size")),
                ),
                "Do not stop unrelated services; rerun long-decode vLLM/torch_npu benchmarks after the cards are released.",
            )
        )
    return diagnoses


def next_optimization_target(report: dict[str, Any]) -> dict[str, Any]:
    sequential = report.get("sequential_vllm", {})
    max_new_tokens = as_float(sequential.get("max_new_tokens"))
    if max_new_tokens is not None and max_new_tokens >= 16 and sequential.get("status") == "PASS":
        return {
            "title": "Optimize compressed serving export and kernels",
            "steps": [
                f"Long-decode vLLM serving benchmark already covers max_new_tokens={int(max_new_tokens)}; use it as the optimization baseline.",
                "Profile dense materialization, cache rebuild cost, and wrapper overhead in CAP/QPruner serving exports.",
                "Prototype compression-aware matmul/attention integration before claiming end-to-end serving speedup.",
            ],
        }
    return {
        "title": "Reduce dense export/cache overhead and measure longer decode",
        "steps": [
            "Run vLLM and torch_npu serving benchmarks at max_new_tokens=16/32 with warmup.",
            "Profile the compressed serving export boundary for dense materialization and cache rebuild cost.",
            "Prototype compression-aware matmul/attention integration before claiming end-to-end serving speedup.",
        ],
    }


def build_report(demo_root: Path) -> dict[str, Any]:
    artifacts = demo_root / "artifacts"
    vllm_name, vllm = best_benchmark_artifact(
        artifacts,
        "vllm_serving_benchmark_*.json",
        VLLM_BENCHMARK_ARTIFACTS,
    )
    current_vllm_name, current_vllm = current_vllm_rerun_artifact(artifacts)
    parallel_name, parallel = best_vllm_parallel_suite(artifacts)
    torch_fallback_name, torch_fallback = best_benchmark_artifact(
        artifacts,
        "torch_serving_fallback_*.json",
        [TORCH_FALLBACK_ARTIFACT],
    )
    compressed_native_name, compressed_native = best_compressed_native_artifact(artifacts)
    compressed_native_fastest_name, compressed_native_fastest = fastest_compressed_native_qpruner_artifact(artifacts)
    qpruner_packed_decode = read_json(artifacts / QPRUNER_PACKED_DECODE_ARTIFACT)
    qpruner_packed_decode_shape_sweep_name, qpruner_packed_decode_shape_sweep = best_qpruner_shape_sweep_artifact(
        artifacts
    )
    qpruner_grouped_shape_sweep_name, qpruner_grouped_shape_sweep = best_grouped_qpruner_shape_sweep_artifact(
        artifacts
    )
    qwen_qpruner_grouped_replay_name, qwen_qpruner_grouped_replay = best_qwen_qpruner_grouped_replay_artifact(
        artifacts
    )
    qwen_qpruner_native_profile_name, qwen_qpruner_native_profile = best_qwen_qpruner_native_profile_artifact(
        artifacts
    )
    (
        qwen_qpruner_native_memory_profile_name,
        qwen_qpruner_native_memory_profile,
    ) = best_qwen_qpruner_native_memory_profile_artifact(artifacts)
    (
        qwen_qpruner_native_speed_first_profile_name,
        qwen_qpruner_native_speed_first_profile,
    ) = best_qwen_qpruner_native_speed_first_profile_artifact(artifacts)
    qwen_generate = read_json(artifacts / QWEN_GENERATE_ARTIFACT)
    qwen_quality = read_json(artifacts / QWEN_QUALITY_ARTIFACT)
    npu_occupancy = read_json(artifacts / NPU_OCCUPANCY_ARTIFACT)
    report = {
        "status": "UNKNOWN",
        "demo_root": str(demo_root),
        "platform": platform.platform(),
        "engineering_baseline": engineering_baseline_summary(),
        "paper_baseline_alignment": paper_baseline_alignment(),
        "sequential_vllm": benchmark_summary(vllm_name, vllm),
        "current_vllm_rerun": vllm_rerun_summary(current_vllm_name, current_vllm),
        "vllm_parallel_sync": parallel_summary(parallel_name, parallel),
        "torch_fallback": benchmark_summary(torch_fallback_name or TORCH_FALLBACK_ARTIFACT, torch_fallback),
        "compressed_native": compressed_native_summary(compressed_native_name, compressed_native),
        "compressed_native_fastest": compressed_native_summary(compressed_native_fastest_name, compressed_native_fastest),
        "qpruner_packed_decode": qpruner_packed_decode_summary(QPRUNER_PACKED_DECODE_ARTIFACT, qpruner_packed_decode),
        "qpruner_packed_decode_shape_sweep": qpruner_packed_decode_shape_sweep_summary(
            qpruner_packed_decode_shape_sweep_name or QPRUNER_PACKED_DECODE_SHAPE_SWEEP_ARTIFACT,
            qpruner_packed_decode_shape_sweep,
        ),
        "qpruner_grouped_shape_sweep": qpruner_grouped_shape_sweep_summary(
            qpruner_grouped_shape_sweep_name,
            qpruner_grouped_shape_sweep,
        ),
        "qwen_qpruner_grouped_replay": qwen_qpruner_grouped_replay_summary(
            qwen_qpruner_grouped_replay_name,
            qwen_qpruner_grouped_replay,
        ),
        "qwen_qpruner_native_profile": qwen_qpruner_native_profile_summary(
            qwen_qpruner_native_profile_name or QWEN_QPRUNER_NATIVE_PROFILE_ARTIFACT,
            qwen_qpruner_native_profile,
        ),
        "qwen_qpruner_native_memory_profile": qwen_qpruner_native_profile_summary(
            qwen_qpruner_native_memory_profile_name or QWEN_QPRUNER_NATIVE_PROFILE_ARTIFACT,
            qwen_qpruner_native_memory_profile,
        ),
        "qwen_qpruner_native_speed_first_profile": qwen_qpruner_native_profile_summary(
            qwen_qpruner_native_speed_first_profile_name or QWEN_QPRUNER_NATIVE_PROFILE_ARTIFACT,
            qwen_qpruner_native_speed_first_profile,
        ),
        "qwen_qpruner_native_bits_tradeoff": qwen_qpruner_native_bits_tradeoff(artifacts),
        "compression_generate": compression_generate_summary(QWEN_GENERATE_ARTIFACT, qwen_generate),
        "compression_quality": compression_quality_summary(QWEN_QUALITY_ARTIFACT, qwen_quality),
        "resource_blocker": resource_blocker_summary(NPU_OCCUPANCY_ARTIFACT, npu_occupancy),
    }
    report["qpruner_grouped_projection_plan"] = qpruner_grouped_projection_plan_summary(
        report["qwen_qpruner_native_memory_profile"],
        report["qpruner_grouped_shape_sweep"],
    )
    report["diagnoses"] = build_diagnoses(report)
    report["next_optimization_target"] = next_optimization_target(report)
    missing = [
        artifact_ref(name)
        for name, payload in (
            (vllm_name or VLLM_BENCHMARK_ARTIFACTS[0], vllm),
            (parallel_name or "multicard_parallel_suite_vllm_metadata_sync*_metrics.json", parallel),
            (TORCH_FALLBACK_ARTIFACT, torch_fallback),
            (compressed_native_name or COMPRESSED_NATIVE_ARTIFACTS[0], compressed_native),
            (compressed_native_fastest_name or COMPRESSED_NATIVE_ARTIFACTS[-1], compressed_native_fastest),
            (QPRUNER_PACKED_DECODE_ARTIFACT, qpruner_packed_decode),
            (qpruner_packed_decode_shape_sweep_name or QPRUNER_PACKED_DECODE_SHAPE_SWEEP_ARTIFACT, qpruner_packed_decode_shape_sweep),
            (QWEN_GENERATE_ARTIFACT, qwen_generate),
            (QWEN_QUALITY_ARTIFACT, qwen_quality),
        )
        if payload is None
    ]
    report["missing_artifacts"] = missing
    report["status"] = "MISSING_EVIDENCE" if missing else "ACTIONABLE"
    return report


def markdown(report: dict[str, Any]) -> str:
    sequential = report["sequential_vllm"]
    current_vllm = report.get("current_vllm_rerun", {})
    parallel = report["vllm_parallel_sync"]
    torch_fallback = report["torch_fallback"]
    compressed_native = report["compressed_native"]
    compressed_native_fastest = report.get("compressed_native_fastest", {})
    packed_decode = report["qpruner_packed_decode"]
    packed_decode_shape_sweep = report.get("qpruner_packed_decode_shape_sweep", {})
    grouped_shape_sweep = report.get("qpruner_grouped_shape_sweep", {})
    real_grouped_replay = report.get("qwen_qpruner_grouped_replay", {})
    qwen_native_profile = report.get("qwen_qpruner_native_profile", {})
    qwen_memory_profile = report.get("qwen_qpruner_native_memory_profile", {})
    qwen_speed_profile = report.get("qwen_qpruner_native_speed_first_profile", {})
    qwen_bits_tradeoff = report.get("qwen_qpruner_native_bits_tradeoff", [])
    grouped_projection_plan = report.get("qpruner_grouped_projection_plan", {})
    generate = report["compression_generate"]
    quality = report["compression_quality"]
    resource_blocker = report.get("resource_blocker", {})
    method_bests = parallel.get("method_bests", {})
    runtime_profile = (
        compressed_native.get("qpruner_runtime_profile", {})
        if isinstance(compressed_native.get("qpruner_runtime_profile"), dict)
        else {}
    )
    qwen_runtime_profile = (
        qwen_native_profile.get("qpruner_runtime_profile", {})
        if isinstance(qwen_native_profile.get("qpruner_runtime_profile"), dict)
        else {}
    )
    grouped_shape_best = (
        grouped_shape_sweep.get("best_grouped_projection", {})
        if isinstance(grouped_shape_sweep.get("best_grouped_projection"), dict)
        else {}
    )
    real_grouped_best = (
        real_grouped_replay.get("best_group", {})
        if isinstance(real_grouped_replay.get("best_group"), dict)
        else {}
    )
    grouped_plan_best = (
        grouped_projection_plan.get("best_candidate", {})
        if isinstance(grouped_projection_plan.get("best_candidate"), dict)
        else {}
    )
    grouped_plan_full_model = (
        grouped_projection_plan.get("full_model", {})
        if isinstance(grouped_projection_plan.get("full_model"), dict)
        else {}
    )
    lines = [
        "# Inference Bottleneck Diagnosis",
        "",
        "This report separates the multi-card synchronization evidence from the serving-speedup claim.",
        "",
        engineering_baseline_note(),
        "",
        paper_baseline_note(),
        "",
        "| Evidence | Artifact | Status | Key metrics |",
        "|---|---|---:|---|",
        "| Sequential vLLM serving | `{artifact}` | {status} | baseline {baseline} tokens/s; CAP {cap} tokens/s ({cap_speedup}x); QPruner {q} tokens/s ({q_speedup}x); max_new_tokens={tokens} |".format(
            artifact=artifact_ref(sequential.get("artifact")),
            status=sequential.get("status"),
            baseline=fmt(sequential.get("tokens_per_s", {}).get("baseline")),
            cap=fmt(sequential.get("tokens_per_s", {}).get("cap")),
            q=fmt(sequential.get("tokens_per_s", {}).get("qpruner")),
            cap_speedup=fmt(sequential.get("speedups", {}).get("cap_vs_baseline")),
            q_speedup=fmt(sequential.get("speedups", {}).get("qpruner_vs_baseline")),
            tokens=fmt(sequential.get("max_new_tokens")),
        ),
        "| Current vLLM HBM rerun | `{artifact}` | {status} | parallel_methods={parallel}; method_cards={cards}; failing_methods={methods}; root={root} |".format(
            artifact=artifact_ref(current_vllm.get("artifact")),
            status=current_vllm.get("status"),
            parallel=fmt(current_vllm.get("parallel_methods")),
            cards=fmt(current_vllm.get("method_cards")),
            methods=", ".join(method_label(method) for method in current_vllm.get("failing_methods", []))
            or "missing",
            root=fmt(current_vllm.get("root_cause")),
        ),
        "| Synchronized vLLM slice | `{artifact}` | {status} | {world}-card start_window={window}s; pass_count={passes}; best_vllm_tokens_per_s={best} |".format(
            artifact=artifact_ref(parallel.get("artifact")),
            status=parallel.get("status"),
            world=fmt(parallel.get("world_size")),
            window=fmt(parallel.get("start_window_s")),
            passes=fmt(parallel.get("pass_count")),
            best=fmt(parallel.get("best_vllm_tokens_per_s")),
        ),
        "| torch_npu fallback | `{artifact}` | {status} | baseline {baseline} tokens/s; CAP {cap} tokens/s; QPruner {q} tokens/s; qpruner_vs_baseline={speedup}x |".format(
            artifact=artifact_ref(torch_fallback.get("artifact")),
            status=torch_fallback.get("status"),
            baseline=fmt(torch_fallback.get("tokens_per_s", {}).get("baseline")),
            cap=fmt(torch_fallback.get("tokens_per_s", {}).get("cap")),
            q=fmt(torch_fallback.get("tokens_per_s", {}).get("qpruner")),
            speedup=fmt(torch_fallback.get("speedups", {}).get("qpruner_vs_baseline")),
        ),
        "| compressed-native memory-preserving torch_npu | `{artifact}` | {status} | baseline {baseline} tokens/s; QPruner {q} tokens/s; qpruner_vs_baseline={speedup}x; runtime={runtime}; storage reduction={storage}% |".format(
            artifact=artifact_ref(compressed_native.get("artifact")),
            status=compressed_native.get("status"),
            baseline=fmt(compressed_native.get("tokens_per_s", {}).get("baseline")),
            q=fmt(compressed_native.get("tokens_per_s", {}).get("qpruner")),
            speedup=fmt(compressed_native.get("speedups", {}).get("qpruner_vs_baseline")),
            runtime=fmt(compressed_native.get("qpruner_runtime_strategy")),
            storage=fmt(compressed_native.get("qpruner_storage_reduction_pct")),
        ),
        "| QPruner runtime profile | `{artifact}` | {status} | forwards={calls}; forward_time={time} ms; share={share}%; runtime={runtime} |".format(
            artifact=artifact_ref(compressed_native.get("artifact")),
            status=fmt(runtime_profile.get("status")) if runtime_profile else "missing",
            calls=fmt(runtime_profile.get("total_forward_calls")) if runtime_profile else "missing",
            time=fmt(runtime_profile.get("total_forward_time_ms")) if runtime_profile else "missing",
            share=fmt(runtime_profile.get("estimated_forward_share_pct")) if runtime_profile else "missing",
            runtime=fmt(runtime_profile.get("runtime_strategy")) if runtime_profile else "missing",
        ),
        "| compressed-native fastest torch_npu | `{artifact}` | {status} | baseline {baseline} tokens/s; QPruner {q} tokens/s; qpruner_vs_baseline={speedup}x; runtime={runtime}; cached dense modules={dense} |".format(
            artifact=artifact_ref(compressed_native_fastest.get("artifact")),
            status=compressed_native_fastest.get("status"),
            baseline=fmt(compressed_native_fastest.get("tokens_per_s", {}).get("baseline")),
            q=fmt(compressed_native_fastest.get("tokens_per_s", {}).get("qpruner")),
            speedup=fmt(compressed_native_fastest.get("speedups", {}).get("qpruner_vs_baseline")),
            runtime=fmt(compressed_native_fastest.get("qpruner_runtime_strategy")),
            dense=fmt(compressed_native_fastest.get("qpruner_cached_dense_weight_modules")),
        ),
        "| QPruner packed decode | `{artifact}` | {status} | scaled-code speedup {scaled}x; code-cache speedup {code}x; code-cache storage reduction={storage}%; scaled-code peak={peak} MB |".format(
            artifact=artifact_ref(packed_decode.get("artifact")),
            status=packed_decode.get("status"),
            scaled=fmt(packed_decode.get("scaled_code_speedup_vs_uncached")),
            code=fmt(packed_decode.get("code_cached_speedup_vs_uncached")),
            storage=fmt(packed_decode.get("code_cache_storage_reduction_pct")),
            peak=fmt(packed_decode.get("scaled_code_peak_mem_mb")),
        ),
        "| QPruner grouped projection diagnostic | `{artifact}` | {status} | modules={modules}; sequential={seq} ms; grouped={grouped} ms; speedup={speedup}x; grouped code-cache bytes={codes}; aux-cache bytes={aux}; max_abs_diff={diff} |".format(
            artifact=artifact_ref(packed_decode.get("artifact")),
            status=packed_decode.get("status"),
            modules=fmt(packed_decode.get("grouped_module_count")),
            seq=fmt(packed_decode.get("grouped_sequential_latency_ms")),
            grouped=fmt(packed_decode.get("grouped_latency_ms")),
            speedup=fmt(packed_decode.get("grouped_scaled_code_speedup_vs_sequential")),
            codes=fmt(packed_decode.get("grouped_code_cache_bytes")),
            aux=fmt(packed_decode.get("grouped_aux_cache_bytes")),
            diff=fmt(packed_decode.get("grouped_max_abs_diff_vs_sequential")),
        ),
        "| QPruner Qwen3 shape sweep | `{artifact}` | {status} | preset={preset}; shapes={count}; best memory-preserving {strategy} on {label}: {speedup}x, latency={latency} ms |".format(
            artifact=artifact_ref(packed_decode_shape_sweep.get("artifact")),
            status=packed_decode_shape_sweep.get("status"),
            preset=fmt(packed_decode_shape_sweep.get("shape_preset")),
            count=fmt(packed_decode_shape_sweep.get("shape_count")),
            strategy=fmt(packed_decode_shape_sweep.get("best_memory_preserving", {}).get("strategy")),
            label=fmt(packed_decode_shape_sweep.get("best_memory_preserving", {}).get("label")),
            speedup=fmt(packed_decode_shape_sweep.get("best_memory_preserving", {}).get("speedup_vs_uncached")),
            latency=fmt(packed_decode_shape_sweep.get("best_memory_preserving", {}).get("latency_ms")),
        ),
        "| QPruner Qwen3 grouped shape sweep | `{artifact}` | {status} | grouped shapes={count}; best grouped projection {label}: grouped speedup {speedup}x, sequential={seq} ms, grouped={grouped} ms, grouped code-cache bytes={codes}, aux-cache bytes={aux} |".format(
            artifact=artifact_ref(grouped_shape_sweep.get("artifact")),
            status=grouped_shape_sweep.get("status"),
            count=fmt(grouped_shape_sweep.get("grouped_shape_count")),
            label=fmt(grouped_shape_best.get("label")),
            speedup=fmt(grouped_shape_best.get("speedup_vs_sequential_scaled_code")),
            seq=fmt(grouped_shape_best.get("sequential_latency_ms")),
            grouped=fmt(grouped_shape_best.get("grouped_latency_ms")),
            codes=fmt(grouped_shape_best.get("grouped_code_cache_bytes")),
            aux=fmt(grouped_shape_best.get("grouped_aux_cache_bytes")),
        ),
        "| Qwen3 QPruner real grouped replay | `{artifact}` | {status} | target layers {limit}/{total}; quantized layers={quantized}; groups={groups}; best role={role}; grouped speedup={speedup}x; sequential={seq} ms; grouped={grouped} ms; code-cache storage reduction={code_storage}%; grouped code-cache bytes={codes}; grouped scaled-code bytes={scaled}; max_abs_diff={diff} |".format(
            artifact=artifact_ref(real_grouped_replay.get("artifact")),
            status=real_grouped_replay.get("status"),
            limit=fmt(real_grouped_replay.get("target_layer_limit")),
            total=fmt(real_grouped_replay.get("targeted_layers_total")),
            quantized=fmt(real_grouped_replay.get("quantized_layers")),
            groups=fmt(real_grouped_replay.get("group_count")),
            role=fmt(real_grouped_best.get("role")),
            speedup=fmt(real_grouped_best.get("speedup_vs_sequential_scaled_code")),
            seq=fmt(real_grouped_best.get("sequential_latency_ms")),
            grouped=fmt(real_grouped_best.get("grouped_latency_ms")),
            code_storage=fmt(real_grouped_replay.get("code_cache_storage_reduction_pct")),
            codes=fmt(real_grouped_best.get("grouped_code_cache_bytes")),
            scaled=fmt(real_grouped_best.get("grouped_scaled_code_cache_bytes")),
            diff=fmt(real_grouped_best.get("max_abs_diff_vs_sequential_scaled_code")),
        ),
        "| QPruner full-model grouped projection plan | `{artifact}` | {status} | preset={preset}; candidates={count}; best={label}; grouped speedup={speedup}x; top-module hits={hits}; forwards={calls}; forward share={share}%; code-cache storage reduction {code_storage}%; cached scaled-code bytes={scaled_bytes} |".format(
            artifact=artifact_ref(grouped_projection_plan.get("shape_artifact")),
            status=grouped_projection_plan.get("status"),
            preset=fmt(grouped_projection_plan.get("shape_preset")),
            count=fmt(grouped_projection_plan.get("candidate_count")),
            label=fmt(grouped_plan_best.get("label")),
            speedup=fmt(grouped_plan_best.get("speedup_vs_sequential_scaled_code")),
            hits=fmt(grouped_plan_best.get("profile_top_module_hits")),
            calls=fmt(grouped_plan_full_model.get("total_forward_calls")),
            share=fmt(grouped_plan_full_model.get("forward_share_pct")),
            code_storage=fmt(grouped_plan_full_model.get("code_cache_storage_reduction_pct")),
            scaled_bytes=fmt(grouped_plan_full_model.get("cached_scaled_code_bytes")),
        ),
        "| Qwen3 QPruner native profile | `{artifact}` | {status} | target layers {limit}/{total}; max_new_tokens={tokens}; iters={iters}; warmup={warmup}; paired_rounds={paired}; measurement={measurement}; qpruner_vs_baseline={speedup}x; forwards={calls}; forward_time={time} ms; storage reduction={storage}%; aux-cache modules={aux_modules}; aux-cache bytes={aux_bytes} |".format(
            artifact=artifact_ref(qwen_native_profile.get("artifact")),
            status=qwen_native_profile.get("status"),
            limit=fmt(qwen_native_profile.get("target_layer_limit")),
            total=fmt(qwen_native_profile.get("targeted_layers_total")),
            tokens=fmt(qwen_native_profile.get("max_new_tokens")),
            iters=fmt(qwen_native_profile.get("iters")),
            warmup=fmt(qwen_native_profile.get("warmup")),
            paired=fmt(qwen_native_profile.get("paired_rounds")),
            measurement=fmt(qwen_native_profile.get("measurement_basis")),
            speedup=fmt(qwen_native_profile.get("speedups", {}).get("qpruner_vs_baseline")),
            calls=fmt(qwen_runtime_profile.get("total_forward_calls")) if qwen_runtime_profile else "missing",
            time=fmt(qwen_runtime_profile.get("total_forward_time_ms")) if qwen_runtime_profile else "missing",
            storage=fmt(qwen_native_profile.get("qpruner_storage_reduction_pct")),
            aux_modules=fmt(qwen_native_profile.get("qpruner_cached_aux_modules")),
            aux_bytes=fmt(qwen_native_profile.get("qpruner_cached_aux_bytes")),
        ),
        "| Qwen3 QPruner speed-first native profile | `{artifact}` | {status} | target layers {limit}/{total}; max_new_tokens={tokens}; paired_rounds={paired}; measurement={measurement}; qpruner_vs_baseline={speedup}x; runtime={runtime}; prebuild_scaled_code_dtype_cache={prebuild}; scaled-code dtype-cache bytes={scaled_bytes}; dense-cache bytes={dense_bytes}; dense-cache budget bytes={dense_budget}; code-cache storage reduction={code_storage}% |".format(
            artifact=artifact_ref(qwen_speed_profile.get("artifact")),
            status=qwen_speed_profile.get("status"),
            limit=fmt(qwen_speed_profile.get("target_layer_limit")),
            total=fmt(qwen_speed_profile.get("targeted_layers_total")),
            tokens=fmt(qwen_speed_profile.get("max_new_tokens")),
            paired=fmt(qwen_speed_profile.get("paired_rounds")),
            measurement=fmt(qwen_speed_profile.get("measurement_basis")),
            speedup=fmt(qwen_speed_profile.get("speedups", {}).get("qpruner_vs_baseline")),
            runtime=fmt(qwen_speed_profile.get("qpruner_runtime_strategy")),
            prebuild=fmt(qwen_speed_profile.get("qpruner_prebuild_scaled_code_dtype_cache")),
            scaled_bytes=fmt(qwen_speed_profile.get("qpruner_cached_scaled_code_bytes")),
            dense_bytes=fmt(qwen_speed_profile.get("qpruner_cached_dense_weight_bytes")),
            dense_budget=fmt(qwen_speed_profile.get("qpruner_dense_cache_budget_bytes")),
            code_storage=fmt(qwen_speed_profile.get("qpruner_code_cache_storage_reduction_pct")),
        ),
        "| Qwen3 QPruner memory-first native profile | `{artifact}` | {status} | target layers {limit}/{total}; max_new_tokens={tokens}; iters={iters}; warmup={warmup}; paired_rounds={paired}; measurement={measurement}; qpruner_vs_baseline={speedup}x; code-cache storage reduction={code_storage}%; release_packed={release}; aux-cache bytes={aux_bytes}; released packed bytes={released}; live payload bytes={live_payload} |".format(
            artifact=artifact_ref(qwen_memory_profile.get("artifact")),
            status=qwen_memory_profile.get("status"),
            limit=fmt(qwen_memory_profile.get("target_layer_limit")),
            total=fmt(qwen_memory_profile.get("targeted_layers_total")),
            tokens=fmt(qwen_memory_profile.get("max_new_tokens")),
            iters=fmt(qwen_memory_profile.get("iters")),
            warmup=fmt(qwen_memory_profile.get("warmup")),
            paired=fmt(qwen_memory_profile.get("paired_rounds")),
            measurement=fmt(qwen_memory_profile.get("measurement_basis")),
            speedup=fmt(qwen_memory_profile.get("speedups", {}).get("qpruner_vs_baseline")),
            code_storage=fmt(qwen_memory_profile.get("qpruner_code_cache_storage_reduction_pct")),
            release=fmt(qwen_memory_profile.get("qpruner_release_packed_after_cache")),
            aux_bytes=fmt(qwen_memory_profile.get("qpruner_cached_aux_bytes")),
            released=fmt(qwen_memory_profile.get("qpruner_released_packed_code_bytes")),
            live_payload=fmt(qwen_memory_profile.get("qpruner_live_compressed_payload_storage_bytes")),
        ),
        *[
            "| Qwen3 QPruner native bits tradeoff | `{artifact}` | {status} | {bits}; qpruner_vs_baseline={speedup}x; storage reduction={storage}%; code-cache storage reduction={code_storage}%; released packed bytes={released}; live payload bytes={live_payload} |".format(
                artifact=artifact_ref(row.get("artifact")),
                status=row.get("status"),
                bits=fmt(row.get("bits_label")),
                speedup=fmt(row.get("speedups", {}).get("qpruner_vs_baseline")),
                storage=fmt(row.get("qpruner_storage_reduction_pct")),
                code_storage=fmt(row.get("qpruner_code_cache_storage_reduction_pct")),
                released=fmt(row.get("qpruner_released_packed_code_bytes")),
                live_payload=fmt(row.get("qpruner_live_compressed_payload_storage_bytes")),
            )
            for row in qwen_bits_tradeoff
            if isinstance(row, dict)
        ],
        "| Compression generate | `{artifact}` | {status} | serving_dense_export={dense}; cache={cache}; CAP speedup {cap_speedup}x; QPruner speedup {q_speedup}x |".format(
            artifact=artifact_ref(generate.get("artifact")),
            status=generate.get("status"),
            dense=generate.get("serving_dense_export"),
            cache=generate.get("inference_cache_enabled"),
            cap_speedup=fmt(generate.get("cap_latency_speedup")),
            q_speedup=fmt(generate.get("qpruner_latency_speedup")),
        ),
        "| Compression quality | `{artifact}` | {status} | CAP loss delta {cap_loss}; QPruner loss delta {q_loss}; target layers {limit}/{total} |".format(
            artifact=artifact_ref(quality.get("artifact")),
            status=quality.get("status"),
            cap_loss=fmt(quality.get("cap_loss_delta")),
            q_loss=fmt(quality.get("qpruner_loss_delta")),
            limit=fmt(quality.get("target_layer_limit")),
            total=fmt(quality.get("targeted_layers_total")),
        ),
        "| External NPU occupancy | `{artifact}` | {status} | occupied_cards={occupied}/{total}; container={container}; model={model}; tensor_parallel_size={tp}; policy={policy} |".format(
            artifact=artifact_ref(resource_blocker.get("artifact")),
            status=resource_blocker.get("status"),
            occupied=fmt(resource_blocker.get("occupied_cards")),
            total=fmt(resource_blocker.get("total_cards")),
            container=fmt(resource_blocker.get("container_name")),
            model=fmt(resource_blocker.get("service_model")),
            tp=fmt(resource_blocker.get("tensor_parallel_size")),
            policy=fmt(resource_blocker.get("policy")),
        ),
        "",
        "## Multi-Card Sync Evidence",
        "",
        "{world}-card vLLM synchronized slice {status}: start_window={window}s, release_lag_window={release}s, pass_count={passes}.".format(
            world=fmt(parallel.get("world_size")),
            status=parallel.get("status"),
            window=fmt(parallel.get("start_window_s")),
            release=fmt(parallel.get("release_lag_window_s")),
            passes=fmt(parallel.get("pass_count")),
        ),
    ]
    for method in METHODS:
        if method in method_bests:
            lines.append(f"- {method_label(method)} vLLM {fmt(method_bests[method])} tokens/s")
    lines.extend(["", "## Diagnoses", ""])
    for item in report.get("diagnoses", []):
        lines.extend(
            [
                f"### {item['title']}",
                "",
                f"- Evidence: {item['evidence']}",
                f"- Recommendation: {item['recommendation']}",
                "",
            ]
        )
    next_target = report.get("next_optimization_target", {})
    lines.extend(["## Next optimization target", "", str(next_target.get("title") or "missing")])
    for step in next_target.get("steps", []):
        lines.append(f"- {step}")
    if report.get("missing_artifacts"):
        lines.extend(["", "## Missing Artifacts", ""])
        lines.extend(f"- `{name}`" for name in report["missing_artifacts"])
    return "\n".join(lines) + "\n"


def write_outputs(report: dict[str, Any], demo_root: Path) -> None:
    artifacts = demo_root / "artifacts"
    reports = demo_root / "reports"
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    (artifacts / "inference_bottleneck_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    (reports / "inference-bottleneck-report.md").write_text(markdown(report))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write an Ascend serving bottleneck diagnosis report")
    parser.add_argument("--demo-root", default="/mnt/nvme/622/tidal-demo")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(Path(args.demo_root))
    write_outputs(report, Path(args.demo_root))
    print("INFERENCE_BOTTLENECK_REPORT " + json.dumps(report, sort_keys=True))
    return 0 if report.get("status") != "MISSING_EVIDENCE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
