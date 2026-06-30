#!/usr/bin/env python
"""Write a requirement-level coverage audit for the Ascend 910B objective."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from tidal.reports.multicard_selection import best_multicard_qwen_lora_finetune  # noqa: E402
from tidal.reports.qwen_grouped_mlp_sweep import (  # noqa: E402
    best_qwen_grouped_mlp_memory_tradeoff,
    best_qwen_grouped_mlp_sweep,
    qwen_grouped_mlp_memory_tradeoff_readout,
    qwen_grouped_mlp_sweep_evidence,
    qwen_grouped_mlp_sweep_readout,
)
from tidal.reports.qwen_native_tradeoff import (  # noqa: E402
    best_qwen_native_8card_profile_summary,
    qwen_native_8card_profile_evidence,
    qwen_native_8card_profile_readout,
)


QPRUNER_PACKED_DECODE_ARTIFACT = "qpruner_packed_decode_benchmark_tiny_qwen3_qpruner_decode_npu.json"
QPRUNER_PACKED_DECODE_REPORT = "qpruner-packed-decode-benchmark-tiny_qwen3_qpruner_decode_npu.md"
QWEN_QPRUNER_NATIVE_PROFILE_ARTIFACT = "qwen_qpruner_native_profile_qwen3_06b_native_profile_npu.json"
QWEN_QPRUNER_NATIVE_PROFILE_REPORT = "qwen-qpruner-native-profile-qwen3_06b_native_profile_npu.md"
QWEN_QPRUNER_NATIVE_PROFILE_PATTERN = "qwen_qpruner_native_profile_*.json"
CHOICE_ACCURACY_ARTIFACT = "qwen_compression_choice_accuracy_qwen3_06b_choice_accuracy_npu.json"
CHOICE_ACCURACY_REPORT = "qwen-compression-choice-accuracy-qwen3_06b_choice_accuracy_npu.md"
MULTITASK_CHOICE_ACCURACY_ARTIFACT = "qwen_compression_choice_accuracy_qwen3_06b_multitask_choice_npu.json"
MULTITASK_CHOICE_ACCURACY_REPORT = "qwen-compression-choice-accuracy-qwen3_06b_multitask_choice_npu.md"
FULLTARGET_QPRUNER_CHOICE_ARTIFACT = (
    "qwen_compression_choice_accuracy_qwen3_06b_multitask_choice_fulltarget_qpruner_npu.json"
)
FULLTARGET_QPRUNER_CHOICE_REPORT = (
    "qwen-compression-choice-accuracy-qwen3_06b_multitask_choice_fulltarget_qpruner_npu.md"
)


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def fmt(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def fmt_accuracy(value: Any) -> str:
    if value is None:
        return "missing"
    try:
        return f"{float(value) * 100.0:.3f}%"
    except (TypeError, ValueError):
        return str(value)


def artifact_name(name: str) -> str:
    return f"artifacts/{name}"


def report_name(name: str) -> str:
    return f"reports/{name}"


def status_is_pass(payload: dict[str, Any] | None) -> bool:
    return bool(payload and payload.get("status") == "PASS")


def has_full_scale_coverage(limit: Any, total: Any) -> bool:
    try:
        return int(limit) >= int(total) and int(total) > 0
    except (TypeError, ValueError):
        return False


def existing(demo_root: Path, paths: Sequence[str]) -> list[str]:
    return [path for path in paths if (demo_root / path).exists()]


def missing(demo_root: Path, paths: Sequence[str]) -> list[str]:
    return [path for path in paths if not (demo_root / path).exists()]


def qwen_native_profile_report_name(artifact_name: str | None) -> str:
    if not artifact_name:
        return QWEN_QPRUNER_NATIVE_PROFILE_REPORT
    prefix = "qwen_qpruner_native_profile_"
    if not artifact_name.startswith(prefix) or not artifact_name.endswith(".json"):
        return QWEN_QPRUNER_NATIVE_PROFILE_REPORT
    run_label = artifact_name[len(prefix) : -len(".json")]
    return f"qwen-qpruner-native-profile-{run_label}.md"


def qpruner_shape_sweep_report_name(artifact_name: str | None) -> str | None:
    if not artifact_name:
        return None
    prefix = "qpruner_packed_decode_benchmark_"
    if not artifact_name.startswith(prefix) or not artifact_name.endswith(".json"):
        return None
    run_label = artifact_name[len(prefix) : -len(".json")]
    return f"qpruner-packed-decode-benchmark-{run_label}.md"


def selected_qpruner_shape_sweep(inference_bottleneck: dict[str, Any] | None) -> dict[str, Any]:
    if not inference_bottleneck:
        return {}
    sweep = inference_bottleneck.get("qpruner_packed_decode_shape_sweep")
    return sweep if isinstance(sweep, dict) else {}


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


def best_qwen_qpruner_native_profile_artifact(artifacts: Path) -> tuple[str, dict[str, Any] | None]:
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
        return QWEN_QPRUNER_NATIVE_PROFILE_ARTIFACT, None
    _, name, payload = max(candidates, key=lambda row: row[0])
    return name, payload


def best_qwen_qpruner_native_memory_profile_artifact(artifacts: Path) -> tuple[str, dict[str, Any] | None]:
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
        return QWEN_QPRUNER_NATIVE_PROFILE_ARTIFACT, None
    _, name, payload = max(candidates, key=lambda row: row[0])
    return name, payload


def first_diagnosis_title(report: dict[str, Any] | None) -> str:
    diagnoses = report.get("diagnoses", []) if report else []
    for item in diagnoses:
        if isinstance(item, dict) and item.get("title"):
            return str(item["title"])
    return "missing"


def qpruner_packed_decode_summary(report: dict[str, Any] | None) -> str:
    if not report:
        return "packed decode missing"
    code_cached = report.get("code_cached", {}) if isinstance(report.get("code_cached"), dict) else {}
    scaled_code = report.get("scaled_code_matmul", {}) if isinstance(report.get("scaled_code_matmul"), dict) else {}
    cached = report.get("cached", {}) if isinstance(report.get("cached"), dict) else {}
    return (
        "QPruner packed decode runtime {runtime}, packed decode storage {storage}%, "
        "code-cache storage {code_storage}%, code-cache {code_latency} ms, "
        "code-cache speedup {code_speedup}x, scaled-code matmul {scaled_latency} ms, "
        "scaled-code speedup {scaled_speedup}x, scaled-code peak {scaled_peak} MB, "
        "cache speedup {speedup}x"
    ).format(
        runtime=report.get("runtime_storage_format", "missing"),
        storage=fmt(report.get("storage_reduction_pct")),
        code_storage=fmt(report.get("code_cache_storage_reduction_pct")),
        code_latency=fmt(code_cached.get("latency_ms")),
        code_speedup=fmt(code_cached.get("speedup_vs_uncached")),
        scaled_latency=fmt(scaled_code.get("latency_ms")),
        scaled_speedup=fmt(scaled_code.get("speedup_vs_uncached")),
        scaled_peak=fmt(scaled_code.get("peak_mem_mb")),
        speedup=fmt(cached.get("speedup_vs_uncached")),
    )


def qwen_qpruner_native_profile_summary(report: dict[str, Any] | None) -> str:
    if not report:
        return "Qwen3 QPruner native profile missing"
    qpruner = report.get("qpruner", {}) if isinstance(report.get("qpruner"), dict) else {}
    summary = report.get("summary", {}) if isinstance(report.get("summary"), dict) else {}
    profile = qpruner.get("runtime_profile", {}) if isinstance(qpruner.get("runtime_profile"), dict) else {}
    return (
        "Qwen3 QPruner native profile {speedup}x baseline, target layers {limit}/{total}, "
        "max_new_tokens {tokens}, "
        "storage {storage}% reduction, code-cache storage {code_storage}% reduction, "
        "{calls} QuantizedLinear forwards, runtime {runtime}"
    ).format(
        speedup=fmt(summary.get("qpruner_vs_baseline_speedup") or qpruner.get("latency_speedup")),
        limit=fmt(report.get("target_layer_limit")),
        total=fmt(report.get("targeted_layers_total")),
        tokens=fmt(report.get("max_new_tokens")),
        storage=fmt(qpruner.get("targeted_storage_reduction_pct")),
        code_storage=fmt(qpruner.get("code_cache_storage_reduction_pct")),
        calls=fmt(profile.get("total_forward_calls")),
        runtime=fmt(qpruner.get("runtime_strategy") or profile.get("runtime_strategy")),
    )


def qwen_qpruner_native_memory_profile_summary(report: dict[str, Any] | None) -> str:
    if not report:
        return "Qwen3 QPruner memory-first native profile missing"
    qpruner = report.get("qpruner", {}) if isinstance(report.get("qpruner"), dict) else {}
    summary = report.get("summary", {}) if isinstance(report.get("summary"), dict) else {}
    return (
        "Qwen3 QPruner memory-first native profile {speedup}x baseline, target layers {limit}/{total}, "
        "max_new_tokens {tokens}, "
        "code-cache storage {code_storage}% reduction, released packed codes {release}, "
        "released packed bytes {released}, live payload bytes {live_payload}, runtime {runtime}"
    ).format(
        speedup=fmt(summary.get("qpruner_vs_baseline_speedup") or qpruner.get("latency_speedup")),
        limit=fmt(report.get("target_layer_limit")),
        total=fmt(report.get("targeted_layers_total")),
        tokens=fmt(report.get("max_new_tokens")),
        code_storage=fmt(qpruner.get("code_cache_storage_reduction_pct")),
        release=fmt(bool(report.get("qpruner_release_packed_after_cache") or qpruner.get("release_packed_after_cache"))),
        released=fmt(qpruner.get("released_packed_code_bytes")),
        live_payload=fmt(qpruner.get("live_compressed_payload_storage_bytes")),
        runtime=fmt(qpruner.get("runtime_strategy")),
    )


def qwen_native_bits_tradeoff_summary(inference_bottleneck: dict[str, Any] | None) -> str:
    rows = (
        inference_bottleneck.get("qwen_qpruner_native_bits_tradeoff", [])
        if isinstance(inference_bottleneck, dict)
        else []
    )
    if not isinstance(rows, list) or not rows:
        return "Qwen3 native bits tradeoff missing"
    parts: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        parts.append(
            "{bits} {speedup}x/{storage}% packed storage/{code_storage}% code-cache storage".format(
                bits=fmt(row.get("bits_label")),
                speedup=fmt(row.get("speedups", {}).get("qpruner_vs_baseline")),
                storage=fmt(row.get("qpruner_storage_reduction_pct")),
                code_storage=fmt(row.get("qpruner_code_cache_storage_reduction_pct")),
            )
        )
    return "Qwen3 native bits tradeoff " + ", ".join(parts) if parts else "Qwen3 native bits tradeoff missing"


def qwen_native_bits_tradeoff_artifacts(inference_bottleneck: dict[str, Any] | None) -> list[str]:
    rows = (
        inference_bottleneck.get("qwen_qpruner_native_bits_tradeoff", [])
        if isinstance(inference_bottleneck, dict)
        else []
    )
    if not isinstance(rows, list):
        return []
    artifacts: list[str] = []
    for row in rows:
        if isinstance(row, dict) and row.get("artifact"):
            artifacts.append(artifact_name(str(row["artifact"])))
    return artifacts


def qpruner_shape_sweep_summary(report: dict[str, Any]) -> str:
    if not report:
        return "Qwen3 projection shape sweep missing"
    best = report.get("best_memory_preserving", {}) if isinstance(report.get("best_memory_preserving"), dict) else {}
    return (
        "Qwen3 projection shape sweep bits {bits}, shapes {count}, best memory path {path} on {label}, "
        "latency {latency} ms, speedup {speedup}x, code-cache storage {code_storage}%"
    ).format(
        bits=fmt(report.get("bits")),
        count=fmt(report.get("shape_count")),
        path=fmt(best.get("path") or best.get("strategy")),
        label=fmt(best.get("label")),
        latency=fmt(best.get("latency_ms")),
        speedup=fmt(best.get("speedup_vs_uncached")),
        code_storage=fmt(best.get("code_cache_storage_reduction_pct")),
    )


def inference_grouped_replay_summary(grouped: dict[str, Any] | None) -> str:
    if not grouped:
        return "8-card grouped replay missing"
    monitor = grouped.get("monitor", {}) if isinstance(grouped.get("monitor"), dict) else {}
    return (
        "8-card grouped replay {status}, world {world}, memory-native workers {memory}/{passes}, "
        "code-cache storage {storage}% reduction, mean grouped speedup {mean_speedup}x, "
        "best grouped speedup {best_speedup}x, monitor {monitor_status}, "
        "{aicore_samples} all-8 AICore samples, max active cards {max_active}, "
        "released packed-code bytes {released}, live compressed payload bytes {payload}"
    ).format(
        status=grouped.get("status", "missing"),
        world=fmt(grouped.get("world_size")),
        memory=fmt(grouped.get("memory_native_worker_count")),
        passes=fmt(grouped.get("pass_count")),
        storage=fmt(grouped.get("code_cache_storage_reduction_pct")),
        mean_speedup=fmt(grouped.get("mean_grouped_speedup")),
        best_speedup=fmt(grouped.get("best_grouped_speedup")),
        monitor_status=monitor.get("status", "missing"),
        aicore_samples=fmt(monitor.get("samples_with_8_nonzero_aicore")),
        max_active=fmt(monitor.get("max_active_process_cards")),
        released=fmt(grouped.get("released_packed_code_bytes_total")),
        payload=fmt(grouped.get("live_compressed_payload_storage_bytes_total")),
    )


def inference_grouped_replay_evidence(grouped: dict[str, Any] | None) -> list[str]:
    if not grouped:
        return []
    monitor = grouped.get("monitor", {}) if isinstance(grouped.get("monitor"), dict) else {}
    evidence: list[str] = []
    for item in (grouped.get("artifact"), monitor.get("artifact")):
        if item:
            evidence.append(str(item))
    return evidence


def choice_accuracy_summary(report: dict[str, Any] | None) -> str:
    if not report:
        return "external choice accuracy missing"
    methods = report.get("methods", {}) if isinstance(report.get("methods"), dict) else {}
    baseline = methods.get("baseline", {}) if isinstance(methods.get("baseline"), dict) else {}
    cap = methods.get("cap", {}) if isinstance(methods.get("cap"), dict) else {}
    qpruner = methods.get("qpruner", {}) if isinstance(methods.get("qpruner"), dict) else {}
    prefix = "multi-task external choice accuracy" if int(report.get("task_count") or 0) > 1 else "external choice accuracy"
    sample_count = baseline.get("total") or len(report.get("samples", []))
    sample_suffix = f"; samples {sample_count}" if sample_count else ""
    task_suffix = f"; tasks {report.get('task_count')}" if report.get("task_count") else ""
    return (
        "{prefix} baseline {baseline}, CAP {cap}, QPruner {qpruner}; "
        "choice accuracy target layers {limit}/{total}{sample_suffix}{task_suffix}"
    ).format(
        prefix=prefix,
        baseline=fmt_accuracy(baseline.get("accuracy")),
        cap=fmt_accuracy(cap.get("accuracy")),
        qpruner=fmt_accuracy(qpruner.get("accuracy")),
        limit=report.get("target_layer_limit", "missing"),
        total=report.get("targeted_layers_total", "missing"),
        sample_suffix=sample_suffix,
        task_suffix=task_suffix,
    )


def qpruner_choice_target_layers(report: dict[str, Any] | None) -> Any:
    if not report:
        return None
    methods = report.get("methods", {}) if isinstance(report.get("methods"), dict) else {}
    qpruner = methods.get("qpruner", {}) if isinstance(methods.get("qpruner"), dict) else {}
    return qpruner.get("targeted_layers") or report.get("target_layer_limit")


def has_full_target_qpruner_choice(report: dict[str, Any] | None) -> bool:
    if not status_is_pass(report):
        return False
    methods = report.get("methods", {}) if isinstance(report.get("methods"), dict) else {}
    if "baseline" not in methods or "qpruner" not in methods:
        return False
    return has_full_scale_coverage(qpruner_choice_target_layers(report), report.get("targeted_layers_total"))


def fulltarget_qpruner_choice_summary(report: dict[str, Any] | None) -> str:
    if not report:
        return "full-target QPruner choice accuracy missing"
    methods = report.get("methods", {}) if isinstance(report.get("methods"), dict) else {}
    baseline = methods.get("baseline", {}) if isinstance(methods.get("baseline"), dict) else {}
    qpruner = methods.get("qpruner", {}) if isinstance(methods.get("qpruner"), dict) else {}
    sample_count = baseline.get("total") or len(report.get("samples", []))
    sample_suffix = f"; samples {sample_count}" if sample_count else ""
    task_suffix = f"; tasks {report.get('task_count')}" if report.get("task_count") else ""
    compression_time = qpruner.get("compression_time_s")
    compression_suffix = f"; QPruner compression time {float(compression_time):.3f}s" if compression_time is not None else ""
    return (
        "full-target QPruner choice accuracy baseline {baseline}, QPruner {qpruner}; "
        "choice accuracy target layers {limit}/{total}{sample_suffix}{task_suffix}{compression_suffix}"
    ).format(
        baseline=fmt_accuracy(baseline.get("accuracy")),
        qpruner=fmt_accuracy(qpruner.get("accuracy")),
        limit=fmt(qpruner_choice_target_layers(report)),
        total=report.get("targeted_layers_total", "missing"),
        sample_suffix=sample_suffix,
        task_suffix=task_suffix,
        compression_suffix=compression_suffix,
    )


def section(
    *,
    section_id: str,
    title: str,
    status: str,
    evidence: list[str],
    evidence_summary: str,
    remaining_gap: str,
    next_action: str,
) -> dict[str, Any]:
    return {
        "id": section_id,
        "title": title,
        "status": status,
        "evidence": evidence,
        "evidence_summary": evidence_summary,
        "remaining_gap": remaining_gap,
        "next_action": next_action,
    }


def build_audit(demo_root: Path) -> dict[str, Any]:
    artifacts = demo_root / "artifacts"
    sync = read_json(artifacts / "multicard_sync_summary.json")
    compatibility = read_json(artifacts / "compatibility_issues.json")
    inference_summary = read_json(artifacts / "inference_acceleration_summary.json")
    inference_bottleneck = read_json(artifacts / "inference_bottleneck_report.json")
    qpruner_packed_decode = read_json(artifacts / QPRUNER_PACKED_DECODE_ARTIFACT)
    qwen_qpruner_native_profile_name, qwen_qpruner_native_profile = best_qwen_qpruner_native_profile_artifact(
        artifacts
    )
    qwen_qpruner_native_profile_report = qwen_native_profile_report_name(qwen_qpruner_native_profile_name)
    (
        qwen_qpruner_native_memory_profile_name,
        qwen_qpruner_native_memory_profile,
    ) = best_qwen_qpruner_native_memory_profile_artifact(artifacts)
    qwen_qpruner_native_memory_profile_report = qwen_native_profile_report_name(
        qwen_qpruner_native_memory_profile_name
    )
    qwen_grouped_mlp_sweep = best_qwen_grouped_mlp_sweep(artifacts)
    qwen_grouped_mlp_memory_tradeoff = best_qwen_grouped_mlp_memory_tradeoff(artifacts)
    qwen_native_8card_profile = best_qwen_native_8card_profile_summary(artifacts)
    qwen_lora = best_multicard_qwen_lora_finetune(artifacts)
    qwen_lora_artifact = str(
        (qwen_lora or {}).get("_artifact_name")
        or "multicard_qwen_lora_finetune_qwen3_06b_lora_sync_npu.json"
    )
    qwen_bslora = read_json(artifacts / "tiny_qwen_bslora_finetune_qwen3_06b_bslora_npu.json")
    compression_sync = read_json(artifacts / "multicard_compression_sync_summary.json")
    qwen_quality = read_json(artifacts / "qwen_compression_quality_qwen3_06b_quality_npu.json")
    qpruner_scale_quality = read_json(artifacts / "qwen_qpruner_scale_quality_summary.json")
    fixed_choice_accuracy = read_json(artifacts / CHOICE_ACCURACY_ARTIFACT)
    multitask_choice_accuracy = read_json(artifacts / MULTITASK_CHOICE_ACCURACY_ARTIFACT)
    fulltarget_qpruner_choice = read_json(artifacts / FULLTARGET_QPRUNER_CHOICE_ARTIFACT)
    choice_accuracy = multitask_choice_accuracy or fixed_choice_accuracy
    has_multitask_choice_accuracy = status_is_pass(multitask_choice_accuracy)
    fulltarget_qpruner_choice_ready = has_full_target_qpruner_choice(fulltarget_qpruner_choice)

    baseline_evidence = [
        artifact_name("multicard_sync_summary.json"),
        artifact_name("compatibility_issues.json"),
    ]
    baseline_ready = bool(sync and sync.get("status") == "PASS" and compatibility)
    baseline_section = section(
        section_id="ascend_910b_baseline",
        title="Ascend 910B baseline",
        status="READY" if baseline_ready else "MISSING",
        evidence=baseline_evidence,
        evidence_summary="8-card HCCL sync {status}, world_size={world}; compatibility report {compat}".format(
            status=(sync or {}).get("status", "missing"),
            world=(sync or {}).get("world_size", "missing"),
            compat=(compatibility or {}).get("status", "missing"),
        ),
        remaining_gap="No baseline blocker found in required smoke artifacts" if baseline_ready else "Run NPU smoke workflows and compatibility report",
        next_action="Keep compatibility_issues.json current after each rerun.",
    )

    inference_payload = (
        inference_summary.get("summary", {}) if isinstance((inference_summary or {}).get("summary"), dict) else {}
    )
    best = (
        inference_payload.get("best_throughput", {})
        if isinstance(inference_payload.get("best_throughput"), dict)
        else {}
    )
    native = (
        inference_payload.get("best_native_memory", {})
        if isinstance(inference_payload.get("best_native_memory"), dict)
        else {}
    )
    grouped_replay = (
        inference_payload.get("grouped_replay", {})
        if isinstance(inference_payload.get("grouped_replay"), dict)
        else {}
    )
    inference_gap = first_diagnosis_title(inference_bottleneck)
    inference_next_action = "Add bottleneck diagnosis and rerun serving benchmark"
    if isinstance((inference_bottleneck or {}).get("next_optimization_target"), dict):
        inference_next_action = inference_bottleneck["next_optimization_target"].get(
            "title",
            inference_next_action,
        )
    if isinstance(qpruner_packed_decode, dict) and qpruner_packed_decode.get("next_action"):
        inference_next_action = qpruner_packed_decode["next_action"]
    qpruner_shape_sweep = selected_qpruner_shape_sweep(inference_bottleneck)
    qpruner_shape_sweep_artifact = qpruner_shape_sweep.get("artifact")
    qpruner_shape_sweep_report = qpruner_shape_sweep_report_name(qpruner_shape_sweep_artifact)
    qpruner_shape_sweep_evidence = []
    if qpruner_shape_sweep_artifact:
        qpruner_shape_sweep_evidence.append(artifact_name(qpruner_shape_sweep_artifact))
    if qpruner_shape_sweep_report:
        qpruner_shape_sweep_evidence.append(report_name(qpruner_shape_sweep_report))
    inference_section = section(
        section_id="inference_acceleration",
        title="Inference acceleration",
        status="ACTIONABLE" if inference_summary else "MISSING",
        evidence=[
            artifact_name("inference_acceleration_summary.json"),
            artifact_name("inference_bottleneck_report.json"),
            artifact_name(QPRUNER_PACKED_DECODE_ARTIFACT),
            *qpruner_shape_sweep_evidence,
            artifact_name(qwen_qpruner_native_profile_name),
            artifact_name(qwen_qpruner_native_memory_profile_name),
            *qwen_native_bits_tradeoff_artifacts(inference_bottleneck),
            *qwen_grouped_mlp_sweep_evidence(qwen_grouped_mlp_sweep),
            *qwen_grouped_mlp_sweep_evidence(qwen_grouped_mlp_memory_tradeoff),
            *qwen_native_8card_profile_evidence(qwen_native_8card_profile),
            *inference_grouped_replay_evidence(grouped_replay),
            report_name("inference-acceleration-summary.md"),
            report_name("inference-bottleneck-report.md"),
            report_name(QPRUNER_PACKED_DECODE_REPORT),
            report_name(qwen_qpruner_native_profile_report),
            report_name(qwen_qpruner_native_memory_profile_report),
        ],
        evidence_summary=(
            "{track}/{method} {tokens} tokens/s; native memory {native_method} {memory}%, storage {storage}%; "
            "{packed}; {shape_sweep}; {qwen_native}; {qwen_memory_native}; {bits_tradeoff}; {grouped_mlp}; "
            "{native_8card}; {grouped_replay}"
        ).format(
            track=best.get("track", "missing"),
            method=best.get("method_label") or best.get("method", "missing"),
            tokens=fmt(best.get("tokens_per_s")),
            native_method=native.get("method_label") or native.get("method", "missing"),
            memory=fmt(native.get("memory_reduction_pct")),
            storage=fmt(native.get("storage_reduction_pct")),
            packed=qpruner_packed_decode_summary(qpruner_packed_decode),
            shape_sweep=qpruner_shape_sweep_summary(qpruner_shape_sweep),
            qwen_native=qwen_qpruner_native_profile_summary(qwen_qpruner_native_profile),
            qwen_memory_native=qwen_qpruner_native_memory_profile_summary(qwen_qpruner_native_memory_profile),
            bits_tradeoff=qwen_native_bits_tradeoff_summary(inference_bottleneck),
            grouped_mlp=(
                qwen_grouped_mlp_sweep_readout(qwen_grouped_mlp_sweep)
                + "; "
                + qwen_grouped_mlp_memory_tradeoff_readout(qwen_grouped_mlp_memory_tradeoff)
            ),
            native_8card=qwen_native_8card_profile_readout(qwen_native_8card_profile)
            or "Qwen3 8-card memory-native native profile missing",
            grouped_replay=inference_grouped_replay_summary(grouped_replay),
        ),
        remaining_gap=inference_gap,
        next_action=inference_next_action,
    )

    lora_agg = qwen_lora.get("aggregate", {}) if isinstance((qwen_lora or {}).get("aggregate"), dict) else {}
    finetune_ready = (
        status_is_pass(qwen_lora)
        and bool(lora_agg.get("distributed_reduce_consistent"))
        and bool(lora_agg.get("adapter_sync_consistent"))
        and status_is_pass(qwen_bslora)
    )
    finetune_section = section(
        section_id="finetuning_effect",
        title="Fine-tuning effect",
        status="READY" if finetune_ready else "PARTIAL_READY" if qwen_lora or qwen_bslora else "MISSING",
        evidence=[
            artifact_name(qwen_lora_artifact),
            artifact_name("tiny_qwen_bslora_finetune_qwen3_06b_bslora_npu.json"),
        ],
        evidence_summary="RankAdaptor LoRA validation delta {delta}; BSLoRA loss delta {bslora_delta}".format(
            delta=fmt(lora_agg.get("validation_loss_delta_avg")),
            bslora_delta=fmt((qwen_bslora or {}).get("loss_delta")),
        ),
        remaining_gap="Longer validation and generation-quality rubric are still needed",
        next_action="Run a larger validation/generation set once runtime budget is available.",
    )

    generate = (
        compression_sync.get("qwen_compression_generate", {})
        if isinstance((compression_sync or {}).get("qwen_compression_generate"), dict)
        else {}
    )
    memory = (
        compression_sync.get("compressed_native_memory", {})
        if isinstance((compression_sync or {}).get("compressed_native_memory"), dict)
        else {}
    )
    alignment = (
        compression_sync.get("baseline_alignment", {})
        if isinstance((compression_sync or {}).get("baseline_alignment"), dict)
        else {}
    )
    scale_best = (
        qpruner_scale_quality.get("best_under_loss_delta", {})
        if isinstance((qpruner_scale_quality or {}).get("best_under_loss_delta"), dict)
        else {}
    )
    scale_limit = scale_best.get("target_layer_limit") or (qpruner_scale_quality or {}).get("max_target_layer_limit")
    scale_total = scale_best.get("targeted_layers_total") or (qpruner_scale_quality or {}).get("targeted_layers_total")
    scale_full_coverage = has_full_scale_coverage(scale_limit, scale_total)
    scale_summary = "scale sweep missing"
    scale_gap = "Scale target-layer coverage and add larger PPL/accuracy table beyond current pilot"
    if qpruner_scale_quality:
        scale_summary = (
            "scale sweep {limit}/{total} target layers ({coverage}% coverage), "
            "{memory}% targeted memory reduction, loss-derived approximate PPL {baseline_ppl} -> {qpruner_ppl}"
        ).format(
            limit=fmt(scale_limit),
            total=fmt(scale_total),
            coverage=fmt(scale_best.get("coverage_pct") or (qpruner_scale_quality or {}).get("max_coverage_pct")),
            memory=fmt(scale_best.get("memory_reduction_pct")),
            baseline_ppl=fmt(scale_best.get("baseline_ppl")),
            qpruner_ppl=fmt(scale_best.get("qpruner_ppl")),
        )
        scale_gap = (
            "Scale beyond {limit}/{total} target layers and add external accuracy table beyond held-out "
            "loss-derived approximate PPL"
        ).format(limit=fmt(scale_limit), total=fmt(scale_total))
        if scale_full_coverage:
            scale_gap = (
                "Full target-layer coverage is present at {limit}/{total}; add external accuracy table beyond "
                "held-out loss-derived approximate PPL"
            ).format(limit=fmt(scale_limit), total=fmt(scale_total))
    if choice_accuracy:
        scale_gap = (
            "Scale beyond {limit}/{total} target layers and add broader external benchmark tasks beyond "
            "fixed prompt-choice table"
        ).format(limit=fmt(scale_limit), total=fmt(scale_total))
        if scale_full_coverage:
            scale_gap = (
                "Full target-layer coverage is present at {limit}/{total}; add broader external benchmark "
                "tasks beyond fixed prompt-choice table"
            ).format(limit=fmt(scale_limit), total=fmt(scale_total))
        if has_multitask_choice_accuracy:
            scale_gap = (
                "Full target-layer coverage is present at {limit}/{total}; expand from the local multi-task "
                "choice probe to a larger external benchmark suite"
            ).format(limit=fmt(scale_limit), total=fmt(scale_total))
    if fulltarget_qpruner_choice_ready and scale_full_coverage:
        scale_gap = (
            "Full target-layer coverage is present at {limit}/{total}; full-target QPruner probe is demo-ready; "
            "expand to a larger external benchmark suite for paper-scale reporting"
        ).format(limit=fmt(scale_limit), total=fmt(scale_total))
    compression_ready = status_is_pass(compression_sync) and status_is_pass(qwen_quality)
    compression_demo_ready = compression_ready and scale_full_coverage and fulltarget_qpruner_choice_ready
    choice_summary_parts = [choice_accuracy_summary(choice_accuracy)]
    if fulltarget_qpruner_choice:
        choice_summary_parts.append(fulltarget_qpruner_choice_summary(fulltarget_qpruner_choice))
    compression_section = section(
        section_id="model_compression",
        title="Model compression",
        status="READY" if compression_demo_ready else "PARTIAL_READY" if compression_ready else "MISSING",
        evidence=[
            artifact_name("multicard_compression_sync_summary.json"),
            artifact_name("qwen_compression_quality_qwen3_06b_quality_npu.json"),
            artifact_name("qwen_qpruner_scale_quality_summary.json"),
            artifact_name(CHOICE_ACCURACY_ARTIFACT),
            artifact_name(MULTITASK_CHOICE_ACCURACY_ARTIFACT),
            artifact_name(FULLTARGET_QPRUNER_CHOICE_ARTIFACT),
            report_name("multicard-compression-sync-summary.md"),
            report_name("qwen-qpruner-scale-quality-summary.md"),
            report_name(CHOICE_ACCURACY_REPORT),
            report_name(MULTITASK_CHOICE_ACCURACY_REPORT),
            report_name(FULLTARGET_QPRUNER_CHOICE_REPORT),
        ],
        evidence_summary=(
            "QPruner {qpruner} tokens/s total, {speedup}x baseline; "
            "QPruner storage reduction {storage}%; {scale}; {choice}; {alignment}"
        ).format(
            qpruner=fmt(generate.get("qpruner_tokens_per_s_total")),
            speedup=fmt(generate.get("qpruner_speedup_vs_baseline")),
            storage=fmt(memory.get("qpruner_storage_reduction_pct")),
            scale=scale_summary,
            choice="; ".join(choice_summary_parts),
            alignment=alignment.get("QPruner", "baseline alignment missing"),
        ),
        remaining_gap=scale_gap,
        next_action=(
            "Keep memory-native path, expand external benchmark tasks, and rerun quality-memory frontier."
            if scale_full_coverage
            else "Increase layer coverage, keep memory-native path, and rerun quality-memory frontier."
        ),
    )

    materials = [
        report_name("ascend-910b-demo-progress.md"),
        report_name("demo-storyboard.md"),
        report_name("demo-readiness-report.md"),
        "recordings/ascend-910b-demo-playback-latest.typescript",
    ]
    material_missing = missing(demo_root, materials)
    materials_section = section(
        section_id="demo_materials",
        title="Demo materials",
        status="READY" if not material_missing else "MISSING",
        evidence=materials,
        evidence_summary="Preserved progress, storyboard, readiness, and terminal playback materials",
        remaining_gap="No material gap found" if not material_missing else "Missing: " + ", ".join(material_missing),
        next_action="Regenerate this audit after every benchmark rerun.",
    )

    sections = [baseline_section, inference_section, finetune_section, compression_section, materials_section]
    if any(item["status"] == "MISSING" for item in sections):
        status = "INCOMPLETE"
    elif any(item["status"] in {"ACTIONABLE", "PARTIAL_READY"} for item in sections):
        status = "PARTIAL_READY"
    else:
        status = "READY"
    return {
        "status": status,
        "demo_root": str(demo_root),
        "sections": sections,
        "evidence": [
            artifact_name("objective_coverage_audit.json"),
            report_name("objective-coverage-audit.md"),
        ],
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# TIDAL-AI Ascend Objective Coverage Audit",
        "",
        f"- Overall status: `{report.get('status', 'missing')}`",
        f"- Demo root: `{report.get('demo_root', 'missing')}`",
        "",
        "| Objective slice | Status | Evidence summary | Remaining gap |",
        "|---|---:|---|---|",
    ]
    for section_data in report.get("sections", []):
        lines.append(
            "| {title} | {status} | {summary} | {gap} |".format(
                title=section_data.get("title", "missing"),
                status=section_data.get("status", "missing"),
                summary=section_data.get("evidence_summary", "missing"),
                gap=section_data.get("remaining_gap", "missing"),
            )
        )
    lines.extend(["", "## Evidence", ""])
    for section_data in report.get("sections", []):
        lines.append(f"### {section_data.get('title', 'missing')}")
        for item in section_data.get("evidence", []):
            lines.append(f"- `{item}`")
        lines.append(f"- Next action: {section_data.get('next_action', 'missing')}")
        lines.append("")
    lines.extend(["## Audit Artifacts", "", "- `artifacts/objective_coverage_audit.json`", "- `reports/objective-coverage-audit.md`"])
    return "\n".join(lines) + "\n"


def write_outputs(demo_root: Path) -> tuple[Path, Path]:
    artifacts = demo_root / "artifacts"
    reports = demo_root / "reports"
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    report = build_audit(demo_root)
    json_path = artifacts / "objective_coverage_audit.json"
    md_path = reports / "objective-coverage-audit.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    md_path.write_text(markdown(report))
    return json_path, md_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write TIDAL-AI Ascend objective coverage audit")
    parser.add_argument("--demo-root", default="/mnt/nvme/622/tidal-demo")
    args = parser.parse_args(argv)
    json_path, md_path = write_outputs(Path(args.demo_root))
    print(f"OBJECTIVE_COVERAGE_AUDIT {json_path} {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
