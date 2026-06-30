#!/usr/bin/env python
"""Replay preserved Ascend 910B demo artifacts for terminal recording."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from tidal.reports.baselines import engineering_baseline_note, paper_baseline_note
from tidal.reports.multicard_selection import best_multicard_qwen_compression_generate
from tidal.reports.npu_monitor_selection import (
    best_npu_utilization_monitor,
    npu_monitor_peak_score,
)
from tidal.reports.qwen_grouped_mlp_sweep import (
    best_qwen_grouped_mlp_memory_tradeoff,
    best_qwen_grouped_mlp_sweep,
    qwen_grouped_mlp_memory_tradeoff_readout,
    qwen_grouped_mlp_sweep_evidence,
    qwen_grouped_mlp_sweep_readout,
    qwen_grouped_mlp_sweep_report_name,
)
from tidal.reports.qwen_grouped_replay_selection import (
    best_multicard_qwen_qpruner_grouped_replay,
    multicard_qwen_qpruner_grouped_replay_report_name as grouped_replay_report_name,
)
from tidal.reports.qwen_native_tradeoff import qwen_native_bits_tradeoff_readout, qwen_native_profile_readout


COMPRESSION_ARTIFACTS = (
    "tiny_qwen_compression_generate_tiny_qwen3_generate_serving_export_npu.json",
    "tiny_qwen_compression_generate_tiny_qwen3_generate_cache_npu.json",
    "tiny_qwen_compression_tiny_qwen3_compression_npu.json",
)
MULTICARD_GENERATE_ARTIFACTS = (
    "multicard_tiny_qwen_generate_tiny_qwen3_multicard_generate_npu.json",
    "multicard_tiny_qwen_generate_cpu_test.json",
)
DEFAULT_VLLM_PROBE_ARTIFACT = "vllm_serving_probe_tiny_qwen3_serving_export_npu.json"
PATCH_VLLM_PROBE_ARTIFACT = "vllm_serving_probe_tiny_qwen3_serving_export_npu_preload_patch.json"
SHIM_VLLM_PROBE_ARTIFACT = "vllm_serving_probe_tiny_qwen3_serving_export_npu_selector_shim.json"
VLLM_SERVING_BENCHMARK_ARTIFACTS = (
    "vllm_serving_benchmark_tiny_qwen3_serving_vllm_metadata_shim_npu.json",
    "vllm_serving_benchmark_tiny_qwen3_serving_vllm_selector_shim_npu.json",
)
TORCH_SERVING_FALLBACK_ARTIFACT = "torch_serving_fallback_tiny_qwen3_serving_fallback_npu.json"
COMPRESSED_NATIVE_TORCH_SERVING_ARTIFACT = "compressed_native_torch_serving_tiny_qwen3_native_serving_npu.json"
QPRUNER_PACKED_DECODE_ARTIFACT = "qpruner_packed_decode_benchmark_tiny_qwen3_qpruner_decode_npu.json"
QPRUNER_DTYPE_SHAPE_SWEEP_ARTIFACT = "qpruner_packed_decode_benchmark_qwen3_06b_shape_sweep_dtype_preserving_npu.json"
QPRUNER_DTYPE_SHAPE_SWEEP_REPORT = "qpruner-packed-decode-benchmark-qwen3_06b_shape_sweep_dtype_preserving_npu.md"
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
INFERENCE_ACCELERATION_SUMMARY_ARTIFACT = "inference_acceleration_summary.json"
COMPRESSED_NATIVE_SWEEP_SUMMARY_ARTIFACT = "compressed_native_sweep_summary.json"
MULTICARD_COMPRESSION_SYNC_SUMMARY_ARTIFACT = "multicard_compression_sync_summary.json"
OBJECTIVE_COVERAGE_AUDIT_ARTIFACT = "objective_coverage_audit.json"
DEMO_READINESS_REPORT_ARTIFACT = "demo_readiness_report.json"
PAPER_BASELINE_COVERAGE_AUDIT_ARTIFACT = "paper_baseline_coverage_audit.json"
MULTICARD_PARALLEL_SUITE_ARTIFACT = "multicard_parallel_suite_demo_parallel_suite_npu.json"
VLLM_PARALLEL_SUITE_ARTIFACT = "multicard_parallel_suite_vllm_metadata_sync_npu_metrics.json"
VLLM_PARALLEL_SUITE_GLOB = "multicard_parallel_suite_vllm_metadata_sync*_metrics.json"
INFERENCE_BOTTLENECK_ARTIFACT = "inference_bottleneck_report.json"
QWEN3_BSLORA_ARTIFACT = "tiny_qwen_bslora_finetune_qwen3_06b_bslora_npu.json"


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def first_json(artifacts: Path, names: Sequence[str]) -> dict[str, Any] | None:
    for name in names:
        payload = read_json(artifacts / name)
        if payload is not None:
            return payload
    return None


def best_vllm_parallel_suite(artifacts: Path) -> dict[str, Any] | None:
    candidates: list[tuple[tuple[int, int, float], dict[str, Any]]] = []
    for path in artifacts.glob(VLLM_PARALLEL_SUITE_GLOB):
        payload = read_json(path)
        if payload is None:
            continue
        payload["_artifact_name"] = path.name
        status_score = 1 if payload.get("status") == "PASS" else 0
        try:
            world_size = int(payload.get("world_size") or 0)
        except (TypeError, ValueError):
            world_size = 0
        candidates.append(((status_score, world_size, path.stat().st_mtime), payload))
    if not candidates:
        payload = read_json(artifacts / VLLM_PARALLEL_SUITE_ARTIFACT)
        if payload is not None:
            payload["_artifact_name"] = VLLM_PARALLEL_SUITE_ARTIFACT
        return payload
    return max(candidates, key=lambda row: row[0])[1]


def int_score(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def float_score(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def multicard_qwen_qpruner_grouped_replay_report_name(replay: dict[str, Any] | None) -> str:
    return f"reports/{grouped_replay_report_name(replay)}"


def multicard_qwen_qpruner_grouped_replay_lines(replay: dict[str, Any] | None) -> list[str]:
    if not isinstance(replay, dict) or not replay:
        return ["8-card grouped replay: missing; Qwen3 QPruner grouped replay memory-native: missing"]
    aggregate = replay.get("aggregate", {}) if isinstance(replay.get("aggregate"), dict) else {}
    released_bytes = aggregate.get("released_packed_code_bytes_total")
    live_payload_bytes = aggregate.get(
        "live_compressed_payload_storage_bytes_total",
        aggregate.get("live_compressed_payload_bytes_total"),
    )
    artifact = str(replay.get("_artifact_name") or "multicard_qwen_qpruner_grouped_replay_missing.json")
    return [
        (
            "8-card grouped replay: {status}; Qwen3 QPruner grouped replay memory-native: {status}, world {world}, "
            "memory-native workers {memory}/{passes}, code-cache storage {storage}%"
        ).format(
            status=replay.get("status", "missing"),
            world=replay.get("world_size", "missing"),
            memory=fmt(aggregate.get("memory_native_worker_count")),
            passes=fmt(aggregate.get("pass_count")),
            storage=fmt(aggregate.get("min_code_cache_storage_reduction_pct")),
        ),
        (
            "Qwen3 QPruner grouped replay performance: best role {role}, "
            "best grouped speedup {speedup}x, mean {mean}x"
        ).format(
            role=aggregate.get("best_role", "missing"),
            speedup=fmt(aggregate.get("best_grouped_speedup")),
            mean=fmt(aggregate.get("mean_grouped_speedup")),
        ),
        (
            "Qwen3 QPruner grouped replay memory bytes: released packed-code bytes {released}, "
            "live compressed payload bytes {payload}"
        ).format(
            released=fmt(float(released_bytes)) if released_bytes is not None else fmt(None),
            payload=fmt(float(live_payload_bytes)) if live_payload_bytes is not None else fmt(None),
        ),
        f"Qwen3 QPruner grouped replay report: {multicard_qwen_qpruner_grouped_replay_report_name(replay)}",
        f"Qwen3 QPruner grouped replay artifact: artifacts/{artifact}",
    ]


def qwen_grouped_mlp_sweep_lines(sweep: dict[str, Any] | None) -> list[str]:
    if not isinstance(sweep, dict) or not sweep:
        return ["Qwen3 grouped-MLP full-model sweep: missing"]
    artifact = str(sweep.get("_artifact_name") or "qwen3_06b_native_profile_8card_grouped_mlp_sweep_missing.json")
    report = qwen_grouped_mlp_sweep_report_name(sweep)
    return [
        qwen_grouped_mlp_sweep_readout(sweep),
        f"Qwen3 grouped-MLP sweep report: reports/{report}",
        f"Qwen3 grouped-MLP sweep artifact: artifacts/{artifact}",
    ]


def qwen_grouped_mlp_memory_tradeoff_lines(sweep: dict[str, Any] | None) -> list[str]:
    if not isinstance(sweep, dict) or not sweep:
        return ["Qwen3 grouped-MLP memory-first tradeoff: missing"]
    artifact = str(sweep.get("_artifact_name") or "qwen3_06b_native_profile_8card_grouped_mlp_sweep_missing.json")
    report = qwen_grouped_mlp_sweep_report_name(sweep)
    return [
        qwen_grouped_mlp_memory_tradeoff_readout(sweep),
        f"Qwen3 grouped-MLP memory-first report: reports/{report}",
        f"Qwen3 grouped-MLP memory-first artifact: artifacts/{artifact}",
    ]


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


def qwen_native_bits_tradeoff_artifacts(inference_bottleneck: dict[str, Any] | None) -> list[str]:
    if not inference_bottleneck:
        return []
    rows = inference_bottleneck.get("qwen_qpruner_native_bits_tradeoff")
    if not isinstance(rows, list):
        return []
    artifacts: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        artifact = row.get("artifact")
        if not isinstance(artifact, str) or not artifact or artifact in seen:
            continue
        seen.add(artifact)
        artifacts.append(artifact)
    return artifacts


def selected_qwen_speed_first_profile(inference_bottleneck: dict[str, Any] | None) -> dict[str, Any]:
    if not inference_bottleneck:
        return {}
    profile = inference_bottleneck.get("qwen_qpruner_native_speed_first_profile")
    return profile if isinstance(profile, dict) else {}


def qwen_speed_first_readout(inference_bottleneck: dict[str, Any] | None) -> str | None:
    profile = selected_qwen_speed_first_profile(inference_bottleneck)
    if not profile:
        return None
    return (
        "Qwen3 speed-first native profile: {speedup}x baseline, measurement {measurement}, "
        "paired_rounds {paired}, code-cache storage {code_storage}%, scaled-code dtype-cache bytes "
        "{scaled_bytes}, dense-cache bytes {dense_bytes}, dense-cache budget bytes {dense_budget}, "
        "prebuild dtype cache {prebuild}"
    ).format(
        speedup=fmt((profile.get("speedups") or {}).get("qpruner_vs_baseline")),
        measurement=fmt(profile.get("measurement_basis")),
        paired=fmt(profile.get("paired_rounds")),
        code_storage=fmt(profile.get("qpruner_code_cache_storage_reduction_pct")),
        scaled_bytes=fmt(profile.get("qpruner_cached_scaled_code_bytes")),
        dense_bytes=fmt(profile.get("qpruner_cached_dense_weight_bytes")),
        dense_budget=fmt(profile.get("qpruner_dense_cache_budget_bytes")),
        prebuild=profile.get("qpruner_prebuild_scaled_code_dtype_cache", "missing"),
    )


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


def issue_classes(probe: dict[str, Any] | None) -> str:
    if not probe:
        return "missing"
    classes = [
        str(row.get("id"))
        for row in probe.get("issue_classes", [])
        if isinstance(row, dict) and row.get("id")
    ]
    return ", ".join(classes) if classes else "none"


def api_diagnostic_summary(probe: dict[str, Any] | None) -> list[str]:
    if not probe:
        return []
    api = probe.get("api_diagnostics")
    if not isinstance(api, dict):
        return []
    missing_fields = ", ".join(api.get("missing_patch_config_fields") or []) or "none"
    missing_parameters = ", ".join(api.get("missing_patch_get_attn_backend_parameters") or []) or "none"
    return [
        "vLLM API diagnostics: {status}, missing patch config {fields}".format(
            status=api.get("status", "missing"),
            fields=missing_fields,
        ),
        f"vLLM patch get_attn_backend missing: {missing_parameters}",
    ]


def probe_load_summary(probe: dict[str, Any] | None) -> str:
    if not probe:
        return "Transformers load through both exports: missing"
    exports = probe.get("exports", {})
    cap = exports.get("cap", {}) if isinstance(exports.get("cap"), dict) else {}
    qpruner = exports.get("qpruner", {}) if isinstance(exports.get("qpruner"), dict) else {}
    return (
        "Transformers load through both exports: CAP {cap_hf}, QPruner {q_hf}; "
        "vLLM load: CAP {cap_vllm}, QPruner {q_vllm}"
    ).format(
        cap_hf=cap.get("transformers_load", "missing"),
        q_hf=qpruner.get("transformers_load", "missing"),
        cap_vllm=cap.get("vllm_load", "missing"),
        q_vllm=qpruner.get("vllm_load", "missing"),
    )


def torch_fallback_lines(fallback: dict[str, Any] | None) -> list[str]:
    if not fallback:
        return ["Torch serving fallback: missing"]
    summary = fallback.get("summary", {}) if isinstance(fallback.get("summary"), dict) else {}
    exports = fallback.get("exports", {}) if isinstance(fallback.get("exports"), dict) else {}
    baseline = exports.get("baseline", {}) if isinstance(exports.get("baseline"), dict) else {}
    cap = exports.get("cap", {}) if isinstance(exports.get("cap"), dict) else {}
    qpruner = exports.get("qpruner", {}) if isinstance(exports.get("qpruner"), dict) else {}
    return [
        "Torch serving fallback: {status}, backend {backend}, best {best} {tps} tokens/s".format(
            status=fallback.get("status", "missing"),
            backend=fallback.get("backend", "missing"),
            best=summary.get("best_method", "missing"),
            tps=fmt(summary.get("best_tokens_per_s")),
        ),
        "Fallback baseline {baseline_tps} tokens/s, CAP {cap_tps} tokens/s, QPruner {q_tps} tokens/s".format(
            baseline_tps=fmt(baseline.get("tokens_per_s")),
            cap_tps=fmt(cap.get("tokens_per_s")),
            q_tps=fmt(qpruner.get("tokens_per_s")),
        ),
        "Fallback speedups vs baseline: CAP {cap_speedup}x, QPruner {q_speedup}x".format(
            cap_speedup=fmt(summary.get("cap_vs_baseline_speedup")),
            q_speedup=fmt(summary.get("qpruner_vs_baseline_speedup")),
        ),
    ]


def compressed_native_lines(report: dict[str, Any] | None) -> list[str]:
    if not report:
        return ["Compressed-native torch serving: missing"]
    summary = report.get("summary", {}) if isinstance(report.get("summary"), dict) else {}
    memory = report.get("memory_reference", {}) if isinstance(report.get("memory_reference"), dict) else {}
    baseline = report.get("baseline", {}) if isinstance(report.get("baseline"), dict) else {}
    cap = report.get("cap", {}) if isinstance(report.get("cap"), dict) else {}
    qpruner = report.get("qpruner", {}) if isinstance(report.get("qpruner"), dict) else {}
    return [
        "Compressed-native torch serving: {status}, backend {backend}, dense export {dense}".format(
            status=report.get("status", "missing"),
            backend=report.get("backend", "missing"),
            dense=report.get("serving_dense_export", "missing"),
        ),
        "Native baseline {baseline} tokens/s, CAP {cap} tokens/s, QPruner {qpruner} tokens/s".format(
            baseline=fmt(baseline.get("tokens_per_s")),
            cap=fmt(cap.get("tokens_per_s")),
            qpruner=fmt(qpruner.get("tokens_per_s")),
        ),
        "Native compressed modules: CAP packed {packed}, QPruner quantized {quantized}, exported dense {cap_dense}/{q_dense}".format(
            packed=fmt(cap.get("packed_layers")),
            quantized=fmt(qpruner.get("quantized_layers")),
            cap_dense=fmt(cap.get("exported_dense_linears")),
            q_dense=fmt(qpruner.get("exported_dense_linears")),
        ),
        "Native memory reductions: CAP {cap}%, QPruner {qpruner}%, dense export erases savings {erases}".format(
            cap=fmt(memory.get("cap_targeted_param_reduction_pct")),
            qpruner=fmt(memory.get("qpruner_targeted_param_reduction_pct")),
            erases=summary.get("dense_export_erases_storage_savings", "missing"),
        ),
        "Native storage footprint: baseline {baseline} bytes, CAP {cap} bytes, QPruner {qpruner} bytes".format(
            baseline=fmt(memory.get("baseline_targeted_storage_bytes")),
            cap=fmt(memory.get("cap_targeted_storage_bytes")),
            qpruner=fmt(memory.get("qpruner_targeted_storage_bytes")),
        ),
        "Native storage reductions: CAP {cap}%, QPruner {qpruner}%".format(
            cap=fmt(memory.get("cap_targeted_storage_reduction_pct")),
            qpruner=fmt(memory.get("qpruner_targeted_storage_reduction_pct")),
        ),
    ]


def qpruner_packed_decode_lines(report: dict[str, Any] | None) -> list[str]:
    if not report:
        return ["QPruner packed decode: missing"]
    uncached = report.get("uncached", {}) if isinstance(report.get("uncached"), dict) else {}
    code_cached = report.get("code_cached", {}) if isinstance(report.get("code_cached"), dict) else {}
    scaled_code = report.get("scaled_code_matmul", {}) if isinstance(report.get("scaled_code_matmul"), dict) else {}
    cached = report.get("cached", {}) if isinstance(report.get("cached"), dict) else {}
    dense = report.get("dense", {}) if isinstance(report.get("dense"), dict) else {}
    return [
        "QPruner packed decode: {status}, device {device}, storage {storage}% reduction, runtime {runtime}".format(
            status=report.get("status", "missing"),
            device=report.get("device", "missing"),
            storage=fmt(report.get("storage_reduction_pct")),
            runtime=report.get("runtime_storage_format", "missing"),
        ),
        "Packed decode latency: uncached {uncached} ms, cached {cached} ms, dense {dense} ms".format(
            uncached=fmt(uncached.get("latency_ms")),
            cached=fmt(cached.get("latency_ms")),
            dense=fmt(dense.get("latency_ms")),
        ),
        "Int8 code-cache path: code-cache {latency} ms, code-cache speedup {speedup}x, code-cache storage {storage}% reduction".format(
            latency=fmt(code_cached.get("latency_ms")),
            speedup=fmt(code_cached.get("speedup_vs_uncached")),
            storage=fmt(report.get("code_cache_storage_reduction_pct")),
        ),
        "Scaled code matmul path: scaled-code {latency} ms, scaled-code speedup {speedup}x, peak {peak} MB".format(
            latency=fmt(scaled_code.get("latency_ms")),
            speedup=fmt(scaled_code.get("speedup_vs_uncached")),
            peak=fmt(scaled_code.get("peak_mem_mb")),
        ),
        "Decode cache speedup: {cached}x vs uncached; dense equivalent {dense}x".format(
            cached=fmt(cached.get("speedup_vs_uncached")),
            dense=fmt(dense.get("speedup_vs_uncached")),
        ),
        "Packed storage: codes {codes} bytes, payload {payload} bytes vs dense {dense} bytes".format(
            codes=fmt(report.get("packed_weight_code_bytes")),
            payload=fmt(report.get("quantized_payload_bytes")),
            dense=fmt(report.get("dense_payload_bytes")),
        ),
        f"Next kernel target: {report.get('next_action', 'missing')}",
    ]


def inference_acceleration_summary_lines(summary: dict[str, Any] | None) -> list[str]:
    if not summary:
        return ["Inference acceleration summary: missing"]
    payload = summary.get("summary", {}) if isinstance(summary.get("summary"), dict) else {}
    best = payload.get("best_throughput", {}) if isinstance(payload.get("best_throughput"), dict) else {}
    native = payload.get("best_native_memory", {}) if isinstance(payload.get("best_native_memory"), dict) else {}
    bottleneck = summary.get("vllm_bottleneck", {}) if isinstance(summary.get("vllm_bottleneck"), dict) else {}
    return [
        f"Inference acceleration summary: {summary.get('status', 'missing')}",
        engineering_baseline_note(),
        paper_baseline_note(),
        "Inference summary best throughput: {track} / {method} {tokens} tokens/s".format(
            track=best.get("track", "missing"),
            method=best.get("method_label") or best.get("method", "missing"),
            tokens=fmt(best.get("tokens_per_s")),
        ),
        "Inference summary native memory: {method} saves {memory}% targeted memory, {storage}% targeted storage".format(
            method=native.get("method_label") or native.get("method", "missing"),
            memory=fmt(native.get("memory_reduction_pct")),
            storage=fmt(native.get("storage_reduction_pct")),
        ),
        f"Inference summary vLLM bottleneck: {bottleneck.get('title', 'missing')}",
    ]


def compressed_native_sweep_lines(summary: dict[str, Any] | None) -> list[str]:
    if not summary:
        return ["Compressed-native sweep: missing"]
    best = summary.get("best", {}) if isinstance(summary.get("best"), dict) else {}
    cache = summary.get("cache_effect", {}) if isinstance(summary.get("cache_effect"), dict) else {}
    long_decode = (
        summary.get("long_decode_effect", {}) if isinstance(summary.get("long_decode_effect"), dict) else {}
    )
    diagnosis = summary.get("runtime_diagnosis", {}) if isinstance(summary.get("runtime_diagnosis"), dict) else {}
    memory = summary.get("memory", {}) if isinstance(summary.get("memory"), dict) else {}
    memory_best = (
        summary.get("memory_preserving_qpruner_best", {})
        if isinstance(summary.get("memory_preserving_qpruner_best"), dict)
        else {}
    )
    return [
        f"Compressed-native sweep: {summary.get('status', 'missing')}",
        f"Compressed-native sweep readout: {summary.get('readout', 'missing')}",
        "Compressed-native sweep best: {method} {speedup}x baseline, max_new_tokens {tokens}, cache {cache}, storage reduction {storage}%".format(
            method=best.get("method_label") or best.get("method", "missing"),
            speedup=fmt(best.get("speedup")),
            tokens=best.get("max_new_tokens", "missing"),
            cache=best.get("inference_cache_enabled", "missing"),
            storage=fmt(best.get("storage_reduction_pct")),
        ),
        "Cache effect: max_new_tokens {tokens}, QPruner speedup delta {delta}x".format(
            tokens=cache.get("max_new_tokens", "missing"),
            delta=fmt(cache.get("qpruner_speedup_delta")),
        ),
        "Long decode effect: {short} -> {long} tokens, QPruner speedup delta {delta}x".format(
            short=long_decode.get("from_max_new_tokens", "missing"),
            long=long_decode.get("to_max_new_tokens", "missing"),
            delta=fmt(long_decode.get("qpruner_speedup_delta")),
        ),
        "CAP runtime tradeoff: {tradeoff}".format(
            tradeoff=diagnosis.get("cap_runtime_tradeoff", "missing"),
        ),
        "Memory-preserving QPruner runtime: {label}, strategy {strategy}, cache mode {mode}, speedup {speedup}x, code-cache storage reduction {storage}%, shape policy matches {matches}, source counts {source_counts}, source {source}".format(
            label=memory_best.get("label", "missing"),
            strategy=memory_best.get("qpruner_runtime_strategy", "missing"),
            mode=memory_best.get("qpruner_cache_mode", "missing"),
            speedup=fmt(memory_best.get("qpruner_vs_baseline_speedup")),
            storage=fmt(
                memory_best.get("qpruner_code_cache_storage_reduction_pct")
                if memory_best.get("qpruner_code_cache_storage_reduction_pct") is not None
                else memory_best.get("qpruner_storage_reduction_pct")
            ),
            matches=fmt(memory_best.get("qpruner_shape_policy_match_count")),
            source_counts=memory_best.get("qpruner_shape_strategy_source_counts", "missing"),
            source=memory_best.get("qpruner_shape_policy_source", "missing"),
        ),
        "Runtime diagnosis: QPruner scaling gap {gap}x, QPruner cache peak +{mem} MB, next {action}".format(
            gap=fmt(diagnosis.get("qpruner_scaling_gap_vs_baseline")),
            mem=fmt(diagnosis.get("qpruner_cache_peak_mem_delta_mb")),
            action=diagnosis.get("next_action", "missing"),
        ),
        "Memory effect: QPruner storage reduction {storage}%".format(
            storage=fmt(memory.get("best_qpruner_storage_reduction_pct")),
        ),
    ]


def multicard_compression_sync_lines(summary: dict[str, Any] | None) -> list[str]:
    if not summary:
        return ["Multi-card compression sync: missing"]
    generate = (
        summary.get("qwen_compression_generate", {})
        if isinstance(summary.get("qwen_compression_generate"), dict)
        else {}
    )
    memory = (
        summary.get("compressed_native_memory", {})
        if isinstance(summary.get("compressed_native_memory"), dict)
        else {}
    )
    parallel = summary.get("parallel_suite", {}) if isinstance(summary.get("parallel_suite"), dict) else {}
    artifact = generate.get("artifact")
    report = generate.get("report")
    return [
        f"Multi-card compression sync: {summary.get('status', 'missing')}",
        f"Multi-card compression sync readout: {summary.get('readout', 'missing')}",
        "Multi-card compression sync QPruner: {qpruner} tokens/s total, {speedup}x baseline, all-reduce {consistent}".format(
            qpruner=fmt(generate.get("qpruner_tokens_per_s_total")),
            speedup=fmt(generate.get("qpruner_speedup_vs_baseline")),
            consistent=generate.get("distributed_reduce_consistent", "missing"),
        ),
        "Multi-card compression sync generate: target layers {limit}/{total}, artifact {artifact}, report {report}".format(
            limit=fmt(generate.get("target_layer_limit")),
            total=fmt(generate.get("targeted_layers_total")),
            artifact=f"artifacts/{artifact}" if artifact else "missing",
            report=f"reports/{report}" if report else "missing",
        ),
        "Multi-card compression sync memory: QPruner storage reduction {storage}%, cache peak +{mem} MB".format(
            storage=fmt(memory.get("qpruner_storage_reduction_pct")),
            mem=fmt(memory.get("qpruner_cache_peak_mem_delta_mb")),
        ),
        "Multi-card compression sync window: start {start}s, release lag {lag}s".format(
            start=fmt(parallel.get("start_window_s")),
            lag=fmt(parallel.get("release_lag_window_s")),
        ),
    ]


def npu_monitor_artifact_name(monitor: dict[str, Any] | None) -> str | None:
    if not monitor:
        return None
    artifact = monitor.get("_artifact_name")
    if artifact:
        return str(artifact)
    run_label = monitor.get("run_label")
    return f"npu_monitor_{run_label}.json" if run_label else None


def npu_monitor_log_path(monitor: dict[str, Any] | None, demo_root: Path) -> str | None:
    if not monitor or not monitor.get("monitor_log"):
        return None
    log_path = Path(str(monitor["monitor_log"]))
    try:
        return str(log_path.relative_to(demo_root))
    except ValueError:
        pass
    try:
        return str(log_path.resolve().relative_to(demo_root.resolve()))
    except (OSError, ValueError):
        return str(log_path)


def npu_monitor_aicore_peaks(monitor: dict[str, Any] | None) -> str:
    if not monitor or not isinstance(monitor.get("max_aicore_by_card"), dict):
        return "missing"
    peaks = monitor["max_aicore_by_card"]

    def key_order(key: Any) -> tuple[int, str]:
        try:
            return int(str(key)), str(key)
        except ValueError:
            return 9999, str(key)

    return ", ".join(f"{key}={peaks[key]}" for key in sorted(peaks, key=key_order)) or "missing"


def npu_monitor_readout(monitor: dict[str, Any] | None) -> str:
    if not monitor:
        return "NPU utilization monitor: missing"
    return (
        "NPU utilization monitor: {status}, {samples} samples, max active process cards {active}, "
        "{process_samples} samples with all 8 process cards, {aicore_samples} samples with all 8 AICore nonzero, "
        "per-card AICore peaks {peaks}"
    ).format(
        status=monitor.get("status", "missing"),
        samples=monitor.get("samples", "missing"),
        active=monitor.get("max_active_process_cards", "missing"),
        process_samples=monitor.get("samples_with_8_active_process_cards", "missing"),
        aicore_samples=monitor.get("samples_with_8_nonzero_aicore", "missing"),
        peaks=npu_monitor_aicore_peaks(monitor),
    )


def npu_monitor_lines(monitor: dict[str, Any] | None, demo_root: Path) -> list[str]:
    artifact = npu_monitor_artifact_name(monitor)
    log_path = npu_monitor_log_path(monitor, demo_root)
    lines = [npu_monitor_readout(monitor)]
    if artifact:
        lines.append(f"Monitor artifact: artifacts/{artifact}")
    if log_path:
        lines.append(f"Monitor log: {log_path}")
    return lines


def objective_coverage_lines(audit: dict[str, Any] | None) -> list[str]:
    if not audit:
        return ["Objective coverage audit: missing"]
    lines = [f"Objective coverage audit: {audit.get('status', 'missing')}"]
    for item in audit.get("sections", []):
        if not isinstance(item, dict):
            continue
        lines.append(
            "Objective coverage section: {title} {status}; evidence {summary}; gap {gap}".format(
                title=item.get("title", "missing"),
                status=item.get("status", "missing"),
                summary=item.get("evidence_summary", "missing"),
                gap=item.get("remaining_gap", "missing"),
            )
        )
    return lines


def demo_readiness_lines(report: dict[str, Any] | None) -> list[str]:
    if not report:
        return ["Demo readiness: missing"]
    lines = [f"Demo readiness: {report.get('status', 'missing')}"]
    for item in report.get("sections", []):
        if not isinstance(item, dict):
            continue
        title = item.get("title") or item.get("id", "missing")
        lines.append(f"Readiness section: {title} {item.get('status', 'missing')}")
        for gap in item.get("gaps", []):
            lines.append(f"Readiness gap: {gap}")
    return lines


def method_label(method: str) -> str:
    return {"baseline": "baseline", "cap": "CAP", "qpruner": "QPruner"}.get(method, method)


def first_root_cause_line(text: str) -> str | None:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        for marker in ("AttributeError:", "RuntimeError:", "ValueError:", "TypeError:", "ImportError:"):
            if marker in line:
                return line[line.index(marker) :]
    return None


def vllm_failure_summary(benchmark: dict[str, Any] | None) -> str:
    if not benchmark:
        return "missing"
    exports = benchmark.get("exports", {})
    if not isinstance(exports, dict):
        return "missing"
    failures = []
    root_cause = None
    for method in ("baseline", "cap", "qpruner"):
        payload = exports.get(method, {}) if isinstance(exports.get(method), dict) else {}
        if payload.get("status") == "PASS":
            continue
        if not payload:
            continue
        failures.append((method_label(method), payload.get("error_type") or payload.get("status") or "unknown"))
        if root_cause is None:
            root_cause = first_root_cause_line(
                "\n".join(str(payload.get(key) or "") for key in ("stdout_tail", "stderr_tail", "error"))
            )
    if not failures:
        return "no failing methods"
    labels = ", ".join(label for label, _ in failures)
    error_types = {error_type for _, error_type in failures}
    error_text = failures[0][1] if len(error_types) == 1 else ", ".join(
        f"{label}={error_type}" for label, error_type in failures
    )
    summary = f"{labels}: {error_text}"
    if root_cause:
        summary += f"; {root_cause}"
    return summary


def vllm_benchmark_lines(benchmark: dict[str, Any] | None) -> list[str]:
    if not benchmark:
        return ["vLLM metadata-shim benchmark: missing"]
    summary = benchmark.get("summary", {}) if isinstance(benchmark.get("summary"), dict) else {}
    exports = benchmark.get("exports", {}) if isinstance(benchmark.get("exports"), dict) else {}
    baseline = exports.get("baseline", {}) if isinstance(exports.get("baseline"), dict) else {}
    cap = exports.get("cap", {}) if isinstance(exports.get("cap"), dict) else {}
    qpruner = exports.get("qpruner", {}) if isinstance(exports.get("qpruner"), dict) else {}
    lines = [
        "vLLM metadata-shim benchmark: {status}, backend {backend}, best {best} {tps} tokens/s".format(
            status=benchmark.get("status", "missing"),
            backend=benchmark.get("backend", "missing"),
            best=summary.get("best_method", "missing"),
            tps=fmt(summary.get("best_tokens_per_s")),
        ),
        "vLLM metadata shim: {enabled}, status {status}".format(
            enabled=benchmark.get("preload_vllm_ascend_metadata_shim", "missing"),
            status=(benchmark.get("metadata_shim", {}) if isinstance(benchmark.get("metadata_shim"), dict) else {}).get(
                "status", "missing"
            ),
        ),
        "vLLM baseline {baseline_tps} tokens/s, CAP {cap_tps} tokens/s, QPruner {q_tps} tokens/s".format(
            baseline_tps=fmt(baseline.get("tokens_per_s")),
            cap_tps=fmt(cap.get("tokens_per_s")),
            q_tps=fmt(qpruner.get("tokens_per_s")),
        ),
        "vLLM speedups vs baseline: CAP {cap_speedup}x, QPruner {q_speedup}x".format(
            cap_speedup=fmt(summary.get("cap_vs_baseline_speedup")),
            q_speedup=fmt(summary.get("qpruner_vs_baseline_speedup")),
        ),
    ]
    if benchmark.get("status") != "PASS":
        failure = vllm_failure_summary(benchmark)
        if "; " in failure:
            summary, detail = failure.split("; ", 1)
            lines.append(f"vLLM benchmark root cause: {summary}")
            lines.append(f"vLLM benchmark root detail: {detail}")
        else:
            lines.append(f"vLLM benchmark root cause: {failure}")
    return lines


def current_vllm_hbm_rerun_line(report: dict[str, Any] | None) -> str | None:
    current = report.get("current_vllm_rerun", {}) if report else {}
    if not isinstance(current, dict) or not current:
        return None
    cards = current.get("method_cards")
    card_text = ",".join(str(card) for card in cards) if isinstance(cards, list) else "missing"
    artifact = current.get("artifact")
    artifact_text = f"artifacts/{artifact}" if artifact else "missing"
    root_cause = current.get("root_cause") or "none"
    speedups = current.get("speedups", {}) if isinstance(current.get("speedups"), dict) else {}
    return (
        "Current vLLM HBM rerun: {status}, parallel={parallel}, cards {cards}, "
        "max_new_tokens={tokens}, best {best} {best_tps} tokens/s, "
        "qpruner_vs_baseline {q_speedup}x, artifact {artifact}, root cause {root_cause}"
    ).format(
        status=current.get("status", "missing"),
        parallel=current.get("parallel_methods", "missing"),
        cards=card_text,
        tokens=current.get("max_new_tokens", "missing"),
        best=current.get("best_method", "missing"),
        best_tps=fmt(current.get("best_tokens_per_s")),
        q_speedup=fmt(speedups.get("qpruner_vs_baseline")),
        artifact=artifact_text,
        root_cause=root_cause,
    )


def first_diagnosis(report: dict[str, Any] | None) -> dict[str, Any]:
    diagnoses = report.get("diagnoses", []) if report else []
    for item in diagnoses:
        if isinstance(item, dict):
            return item
    return {}


def inference_bottleneck_lines(report: dict[str, Any] | None) -> list[str]:
    if not report:
        return ["Inference bottleneck diagnosis: missing"]
    diagnoses = [item for item in report.get("diagnoses", []) if isinstance(item, dict)]
    diagnosis = diagnoses[0] if diagnoses else first_diagnosis(report)
    target = report.get("next_optimization_target", {})
    target_title = target.get("title", "missing") if isinstance(target, dict) else "missing"
    blocker = report.get("resource_blocker", {}) if isinstance(report.get("resource_blocker"), dict) else {}
    lines = [f"Inference bottleneck diagnosis: {report.get('status', 'missing')}"]
    if diagnosis.get("title"):
        lines.append(f"Bottleneck: {diagnosis['title']}")
    if diagnosis.get("evidence"):
        lines.append(f"Bottleneck evidence: {diagnosis['evidence']}")
    for item in diagnoses[1:]:
        if item.get("title"):
            lines.append(f"Bottleneck detail: {item['title']}")
        if item.get("evidence"):
            lines.append(f"Bottleneck detail evidence: {item['evidence']}")
    bits_readout = qwen_native_bits_tradeoff_readout(report)
    profile_readout = qwen_native_profile_readout(report)
    if profile_readout:
        lines.append(profile_readout)
    if bits_readout:
        lines.append(bits_readout)
    speed_readout = qwen_speed_first_readout(report)
    if speed_readout:
        lines.append(speed_readout)
    current_rerun = current_vllm_hbm_rerun_line(report)
    if current_rerun:
        lines.append(current_rerun)
    if blocker.get("status") == "BLOCKED":
        lines.append(
            "Resource blocker: {status}, occupied cards {occupied}/{total}".format(
                status=blocker.get("status", "missing"),
                occupied=fmt(blocker.get("occupied_cards")),
                total=fmt(blocker.get("total_cards")),
            )
        )
        lines.append(
            "Resource blocker service: {container}, {model}, tensor_parallel_size={tp}".format(
                container=blocker.get("container_name", "missing"),
                model=blocker.get("service_model", "missing"),
                tp=fmt(blocker.get("tensor_parallel_size")),
            )
        )
        lines.append(f"Resource blocker policy: {blocker.get('policy', 'missing')}")
    else:
        lines.append("Resource blocker: none")
    lines.append(f"Next optimization target: {target_title}")
    return lines


def join_values(values: Any) -> str:
    if not isinstance(values, list):
        return "missing"
    return ", ".join(str(value) for value in values) or "missing"


def paper_baseline_evidence_metrics(metrics: Any) -> str:
    if not isinstance(metrics, dict) or not metrics:
        return "missing"
    return ", ".join(f"{key} {fmt(value)}" for key, value in metrics.items())


def paper_baseline_audit_lines(report: dict[str, Any] | None) -> list[str]:
    if not report:
        return ["Paper baseline coverage audit: missing"]

    lines = [f"Paper baseline coverage audit: {report.get('status', 'missing')}"]
    note = report.get("engineering_baseline_note", {})
    if isinstance(note, dict):
        lines.append(
            "Paper baseline audit note: {role}, not paper baseline {not_paper}".format(
                role=note.get("role", "missing"),
                not_paper=note.get("not_paper_baseline", "missing"),
            )
        )
    elif note:
        lines.append(f"Paper baseline audit note: {note}")

    for readout in report.get("video_readout", []):
        lines.append(f"Paper baseline audit readout: {readout}")

    evidence = report.get("evidence", {})
    if isinstance(evidence, dict):
        qpruner = evidence.get("qpruner_llm_pruner_full196", {})
        if isinstance(qpruner, dict):
            lines.append(
                "Paper baseline audit evidence: QPruner LLM-Pruner-style {status}, target layers {layers}/{total}; "
                "prune {prune}%, loss delta {loss}, speedup {speedup}x".format(
                    status=qpruner.get("status", "missing"),
                    layers=fmt(qpruner.get("targeted_layers")),
                    total=fmt(qpruner.get("targeted_layers_total")),
                    prune=fmt(qpruner.get("targeted_param_reduction_pct")),
                    loss=fmt(qpruner.get("loss_delta")),
                    speedup=fmt(qpruner.get("latency_speedup")),
                )
            )
        cap = evidence.get("cap_wanda_sparsegpt_full196", {})
        if isinstance(cap, dict):
            target_layers = "{layers}/{total}".format(
                layers=fmt(cap.get("targeted_layers")),
                total=fmt(cap.get("targeted_layers_total")),
            )
            lines.append(
                "Paper baseline audit evidence: CAP Wanda {status}, target layers {target_layers}; "
                "sparsity {sparsity}%, loss delta {loss}, speedup {speedup}x".format(
                    status=cap.get("status", "missing"),
                    target_layers=target_layers,
                    sparsity=fmt(cap.get("wanda_targeted_param_reduction_pct")),
                    loss=fmt(cap.get("wanda_loss_delta")),
                    speedup=fmt(cap.get("wanda_latency_speedup")),
                )
            )
            lines.append(
                "Paper baseline audit evidence: CAP SparseGPT {status}, target layers {target_layers}; "
                "sparsity {sparsity}%, loss delta {loss}, speedup {speedup}x".format(
                    status=cap.get("status", "missing"),
                    target_layers=target_layers,
                    sparsity=fmt(cap.get("sparsegpt_targeted_param_reduction_pct")),
                    loss=fmt(cap.get("sparsegpt_loss_delta")),
                    speedup=fmt(cap.get("sparsegpt_latency_speedup")),
                )
            )
        rankadaptor = evidence.get("rankadaptor_existing", {})
        if isinstance(rankadaptor, dict):
            lines.append(
                "Paper baseline audit evidence: RankAdaptor LoRA/BSLoRA {status}, target modules {target_modules}; "
                "LoRA world {world}, val delta {val_delta}, BSLoRA loss delta {bslora_delta}".format(
                    status=rankadaptor.get("status", "missing"),
                    target_modules=fmt(rankadaptor.get("qwen3_bslora_target_module_count")),
                    world=fmt(rankadaptor.get("qwen3_lora_world_size")),
                    val_delta=fmt(rankadaptor.get("qwen3_lora_validation_loss_delta_avg")),
                    bslora_delta=fmt(rankadaptor.get("qwen3_bslora_loss_delta")),
                )
            )
        recovery = evidence.get("rankadaptor_recovery_controlled", {})
        if isinstance(recovery, dict):
            lines.append(
                "Paper baseline audit evidence: RankAdaptor recovery baselines {status}, best {best}; "
                "no-recovery delta {none}, LoRA delta {lora}, AdaLoRA-style delta {adalora}".format(
                    status=recovery.get("status", "missing"),
                    best=recovery.get("best_recovery_method", "missing"),
                    none=fmt(recovery.get("no_recovery_loss_delta")),
                    lora=fmt(recovery.get("lora_loss_delta")),
                    adalora=fmt(recovery.get("adalora_style_loss_delta")),
                )
            )
        memory = evidence.get("memory_first_existing", {})
        if isinstance(memory, dict):
            lines.append(
                "Paper baseline audit memory-first: CAP targeted memory -{cap}%, "
                "QPruner targeted memory -{qpruner}%, coverage {coverage}%".format(
                    cap=fmt(memory.get("cap_targeted_memory_reduction_pct")),
                    qpruner=fmt(memory.get("qpruner_targeted_memory_reduction_pct")),
                    coverage=fmt(memory.get("target_layer_coverage_pct")),
                )
            )

    for item in report.get("current_evidence", []):
        if not isinstance(item, dict):
            continue
        lines.append(
            "Paper baseline audit evidence: {method} {baseline} {status}, target layers {target_layers}; metrics {metrics}".format(
                method=item.get("method", "missing"),
                baseline=item.get("paper_baseline", "missing"),
                status=item.get("status", "missing"),
                target_layers=item.get("target_layers", "missing"),
                metrics=paper_baseline_evidence_metrics(item.get("metrics")),
            )
        )

    for gap in report.get("remaining_gaps", []):
        lines.append(f"Paper baseline audit gap: {gap}")
    for item in report.get("coverage_decisions", []):
        if not isinstance(item, dict) or not item.get("remaining_gap"):
            continue
        lines.append(
            "Paper baseline audit gap: {area}: {gap}".format(
                area=item.get("area", "missing"),
                gap=item.get("remaining_gap", "missing"),
            )
        )
    for run in report.get("next_runs", []):
        lines.append(f"Paper baseline audit next run: {run}")
    return lines


def compression_memory_video_readout(report: dict[str, Any]) -> str:
    if report.get("video_readout"):
        return str(report["video_readout"])
    memory = report.get("compression_memory", {}) if isinstance(report.get("compression_memory"), dict) else {}
    gap = report.get("memory_gap", {}) if isinstance(report.get("memory_gap"), dict) else {}
    exports = report.get("serving_export_memory", {}) if isinstance(report.get("serving_export_memory"), dict) else {}
    cap_memory = memory.get("cap", {}) if isinstance(memory.get("cap"), dict) else {}
    q_memory = memory.get("qpruner", {}) if isinstance(memory.get("qpruner"), dict) else {}
    cap_export = exports.get("cap", {}) if isinstance(exports.get("cap"), dict) else {}
    q_export = exports.get("qpruner", {}) if isinstance(exports.get("qpruner"), dict) else {}
    cap_ratio = cap_export.get("size_ratio_vs_baseline") or cap_export.get("size_ratio_vs_engineering_reference")
    q_ratio = q_export.get("size_ratio_vs_baseline") or q_export.get("size_ratio_vs_engineering_reference")
    if gap.get("dense_export_erases_storage_savings"):
        cap_ratio = 1.0 if cap_ratio is None else cap_ratio
        q_ratio = 1.0 if q_ratio is None else q_ratio
    return (
        "Memory-first compression readout: CAP targeted memory -{cap}%, QPruner targeted memory -{q}%, "
        "target layers {limit}/{total}; dense HF export ratio CAP/QPruner {cap_ratio}x/{q_ratio}x, "
        "so memory savings require the compressed-native runtime path."
    ).format(
        cap=fmt(cap_memory.get("targeted_param_reduction_pct")),
        q=fmt(q_memory.get("targeted_param_reduction_pct")),
        limit=fmt(memory.get("target_layer_limit")),
        total=fmt(memory.get("targeted_layers_total")),
        cap_ratio=fmt(cap_ratio),
        q_ratio=fmt(q_ratio),
    )


def compression_memory_lines(
    report: dict[str, Any] | None, *, include_legacy_baseline_evidence: bool = True
) -> list[str]:
    if not report:
        return ["Compression memory report: missing"]
    paper = report.get("paper_baselines", {}) if isinstance(report.get("paper_baselines"), dict) else {}
    memory = report.get("compression_memory", {}) if isinstance(report.get("compression_memory"), dict) else {}
    gap = report.get("memory_gap", {}) if isinstance(report.get("memory_gap"), dict) else {}
    reference = report.get("engineering_reference", {}) if isinstance(report.get("engineering_reference"), dict) else {}
    qpruner = paper.get("qpruner", {}) if isinstance(paper.get("qpruner"), dict) else {}
    cap = paper.get("cap", {}) if isinstance(paper.get("cap"), dict) else {}
    rankadaptor = paper.get("rankadaptor", {}) if isinstance(paper.get("rankadaptor"), dict) else {}
    cap_memory = memory.get("cap", {}) if isinstance(memory.get("cap"), dict) else {}
    q_memory = memory.get("qpruner", {}) if isinstance(memory.get("qpruner"), dict) else {}
    lines = [
        f"Compression memory report: {report.get('status', 'missing')}",
        compression_memory_video_readout(report),
        f"QPruner paper baseline: {join_values(qpruner.get('primary'))}",
        f"CAP paper baselines: {join_values(cap.get('pruning_baselines'))}",
        f"RankAdaptor paper recovery baselines: {join_values(rankadaptor.get('recovery_baselines'))}",
        "Engineering reference baseline: {role}, not paper baseline {not_paper}".format(
            role=reference.get("serving_baseline_role", "missing"),
            not_paper=reference.get("not_paper_baseline", "missing"),
        ),
        "Compression memory reductions: CAP {cap}%, QPruner {qpruner}%, target coverage {coverage}%".format(
            cap=fmt(cap_memory.get("targeted_param_reduction_pct")),
            qpruner=fmt(q_memory.get("targeted_param_reduction_pct")),
            coverage=fmt(memory.get("target_layer_coverage_pct")),
        ),
        "Serving memory gap: dense export {dense}, erases storage savings {erases}".format(
            dense=gap.get("serving_dense_export", "missing"),
            erases=gap.get("dense_export_erases_storage_savings", "missing"),
        ),
    ]
    lines = [line for line in lines if line]
    for item in report.get("baseline_alignment_matrix", []):
        if not isinstance(item, dict):
            continue
        lines.append(
            "Baseline alignment: {method} {role} {baselines} -> {demo_role}".format(
                method=item.get("method", "missing"),
                role=item.get("paper_baseline_role", "missing"),
                baselines=join_values(item.get("paper_baselines")),
                demo_role=item.get("demo_reference_role", "missing"),
            )
        )
    if include_legacy_baseline_evidence:
        for item in report.get("paper_baseline_evidence", []):
            if not isinstance(item, dict):
                continue
            lines.append(
                "Paper baseline evidence: {method} {baseline} {status}; metrics {metrics}; artifacts {artifacts}".format(
                    method=item.get("method", "missing"),
                    baseline=item.get("paper_baseline", "missing"),
                    status=item.get("status", "missing"),
                    metrics=paper_baseline_evidence_metrics(item.get("metrics")),
                    artifacts=join_values(item.get("artifacts")),
                )
            )
    if report.get("next_action"):
        lines.append(f"Compression memory next action: {report['next_action']}")
    return lines


def quality_memory_sweep_lines(demo_root: Path) -> list[str]:
    path = demo_root / "reports" / "qwen-compression-quality-memory-sweep.md"
    if not path.exists():
        return ["Qwen3 quality-memory sweep: missing"]
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if line.startswith("Best memory-quality point"):
            return [f"Qwen3 quality-memory sweep: {line}"]
    return ["Qwen3 quality-memory sweep: present without frontier summary"]


def qpruner_quality_memory_sweep_lines(demo_root: Path) -> list[str]:
    path = demo_root / "reports" / "qwen-qpruner-quality-memory-sweep.md"
    if not path.exists():
        return ["Qwen3 QPruner-only quality-memory sweep: missing"]
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if line.startswith("Best memory-quality point"):
            return [f"Qwen3 QPruner-only quality-memory sweep: {line}"]
    return ["Qwen3 QPruner-only quality-memory sweep: present without frontier summary"]


def qpruner_scale_quality_lines(demo_root: Path) -> list[str]:
    payload = read_json(demo_root / "artifacts" / "qwen_qpruner_scale_quality_summary.json")
    if isinstance(payload, dict) and payload.get("readout"):
        return [f"Qwen3 QPruner scale-quality summary: {payload['readout']}"]
    path = demo_root / "reports" / "qwen-qpruner-scale-quality-summary.md"
    if not path.exists():
        return ["Qwen3 QPruner scale-quality summary: missing"]
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if line.startswith("QPruner scale-quality sweep"):
            return [f"Qwen3 QPruner scale-quality summary: {line}"]
    return ["Qwen3 QPruner scale-quality summary: present without readout"]


def choice_accuracy_lines(report: dict[str, Any] | None) -> list[str]:
    if not report:
        return ["Qwen3 compression choice accuracy: missing"]
    methods = report.get("methods", {}) if isinstance(report.get("methods"), dict) else {}
    baseline = methods.get("baseline", {}) if isinstance(methods.get("baseline"), dict) else {}
    cap = methods.get("cap", {}) if isinstance(methods.get("cap"), dict) else {}
    qpruner = methods.get("qpruner", {}) if isinstance(methods.get("qpruner"), dict) else {}
    prefix = "multi-task choice accuracy" if int(report.get("task_count") or 0) > 1 else "choice accuracy"
    sample_count = baseline.get("total") or len(report.get("samples", []))
    sample_suffix = f"; samples {sample_count}" if sample_count else ""
    task_suffix = f"; tasks {report.get('task_count')}" if report.get("task_count") else ""
    if "cap" not in methods and "qpruner" in methods:
        qpruner_layers = qpruner.get("targeted_layers") or report.get("target_layer_limit", "missing")
        compression_time = qpruner.get("compression_time_s")
        compression_suffix = (
            f"; QPruner compression time {float(compression_time):.3f}s" if compression_time is not None else ""
        )
        return [
            f"Qwen3 compression choice accuracy: {report.get('status', 'missing')}",
            (
                "Qwen3 compression choice accuracy: full-target QPruner choice accuracy baseline {baseline}, "
                "QPruner {qpruner}; choice accuracy target layers {limit}/{total}{sample_suffix}{task_suffix}"
                "{compression_suffix}"
            ).format(
                baseline=fmt_accuracy(baseline.get("accuracy")),
                qpruner=fmt_accuracy(qpruner.get("accuracy")),
                limit=qpruner_layers,
                total=report.get("targeted_layers_total", "missing"),
                sample_suffix=sample_suffix,
                task_suffix=task_suffix,
                compression_suffix=compression_suffix,
            ),
        ]
    return [
        f"Qwen3 compression choice accuracy: {report.get('status', 'missing')}",
        (
            "Qwen3 compression choice accuracy: {prefix} baseline {baseline}, "
            "CAP {cap}, QPruner {qpruner}; choice accuracy target layers {limit}/{total}{sample_suffix}{task_suffix}"
        ).format(
            prefix=prefix,
            baseline=fmt_accuracy(baseline.get("accuracy")),
            cap=fmt_accuracy(cap.get("accuracy")),
            qpruner=fmt_accuracy(qpruner.get("accuracy")),
            limit=report.get("target_layer_limit", "missing"),
            total=report.get("targeted_layers_total", "missing"),
            sample_suffix=sample_suffix,
            task_suffix=task_suffix,
        ),
    ]


def qpruner_frontier_chart_lines(demo_root: Path) -> list[str]:
    svg = demo_root / "reports" / "qwen-qpruner-quality-memory-frontier.svg"
    markdown = demo_root / "reports" / "qwen-qpruner-quality-memory-frontier.md"
    if not svg.exists():
        return ["Qwen3 QPruner frontier chart: missing"]
    suffix = " plus Markdown embed" if markdown.exists() else " without Markdown embed"
    return [f"Qwen3 QPruner frontier chart: reports/{svg.name}{suffix}"]


def parallel_suite_lines(parallel_suite: dict[str, Any] | None) -> list[str]:
    if not parallel_suite:
        return ["Parallel suite: missing"]
    counts = parallel_suite.get("task_counts", {}) if isinstance(parallel_suite.get("task_counts"), dict) else {}
    aggregate = parallel_suite.get("aggregate", {}) if isinstance(parallel_suite.get("aggregate"), dict) else {}
    cards = ",".join(str(card) for card in parallel_suite.get("cards", []))
    task_text = ", ".join(f"{task}={count}" for task, count in sorted(counts.items())) or "missing"
    return [
        "Parallel suite: {status}, workers {workers}, cards {cards}".format(
            status=parallel_suite.get("status", "missing"),
            workers=parallel_suite.get("world_size", "missing"),
            cards=cards or "missing",
        ),
        "Parallel suite start window: {window}s, synchronized {sync}".format(
            window=fmt(parallel_suite.get("start_window_s")),
            sync=parallel_suite.get("launched_synchronously", "missing"),
        ),
        "Parallel suite sync gate: {gate}, release lag window {release_window}s".format(
            gate=fmt(parallel_suite.get("sync_start_target_ts")),
            release_window=fmt(parallel_suite.get("release_lag_window_s")),
        ),
        f"Parallel suite tasks: {task_text}",
        "Parallel suite best serving fallback: {tps} tokens/s".format(
            tps=fmt(aggregate.get("best_tokens_per_s")),
        ),
    ]


def vllm_worker_metrics(parallel_suite: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not parallel_suite:
        return {}
    metrics_by_method: dict[str, dict[str, Any]] = {}
    for worker in parallel_suite.get("workers", []):
        if not isinstance(worker, dict):
            continue
        metrics = worker.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        method = metrics.get("method")
        if method:
            existing = metrics_by_method.get(str(method))
            current_tps = metrics.get("tokens_per_s")
            existing_tps = existing.get("tokens_per_s") if existing else None
            if existing is None or (current_tps is not None and (existing_tps is None or float(current_tps) > float(existing_tps))):
                metrics_by_method[str(method)] = metrics
    return metrics_by_method


def vllm_parallel_suite_lines(parallel_suite: dict[str, Any] | None) -> list[str]:
    if not parallel_suite:
        return ["Synchronized vLLM slice: missing"]
    counts = parallel_suite.get("task_counts", {}) if isinstance(parallel_suite.get("task_counts"), dict) else {}
    aggregate = parallel_suite.get("aggregate", {}) if isinstance(parallel_suite.get("aggregate"), dict) else {}
    cards = ",".join(str(card) for card in parallel_suite.get("cards", []))
    task_text = ", ".join(f"{task}={count}" for task, count in sorted(counts.items())) or "missing"
    metrics_by_method = vllm_worker_metrics(parallel_suite)

    def method_tps(method: str) -> str:
        return fmt(metrics_by_method.get(method, {}).get("tokens_per_s"))

    def method_status(method: str) -> str:
        metrics = metrics_by_method.get(method, {})
        shim = metrics.get("metadata_shim_status", "missing")
        backend = metrics.get("metadata_backend_forward_status", "missing")
        return f"{shim}/{backend}"

    return [
        "Synchronized vLLM slice: {status}, workers {workers}, cards {cards}".format(
            status=parallel_suite.get("status", "missing"),
            workers=parallel_suite.get("world_size", "missing"),
            cards=cards or "missing",
        ),
        "Synchronized vLLM slice start window: {window}s, synchronized {sync}".format(
            window=fmt(parallel_suite.get("start_window_s")),
            sync=parallel_suite.get("launched_synchronously", "missing"),
        ),
        "Synchronized vLLM slice tasks: {tasks}, best vLLM {tps} tokens/s".format(
            tasks=task_text,
            tps=fmt(aggregate.get("best_vllm_tokens_per_s")),
        ),
        "Synchronized vLLM methods: QPruner {qpruner} tokens/s, CAP {cap} tokens/s, baseline {baseline} tokens/s".format(
            qpruner=method_tps("qpruner"),
            cap=method_tps("cap"),
            baseline=method_tps("baseline"),
        ),
        "Synchronized vLLM metadata shim: QPruner {qpruner}, CAP {cap}, baseline {baseline}".format(
            qpruner=method_status("qpruner"),
            cap=method_status("cap"),
            baseline=method_status("baseline"),
        ),
    ]


def parallel_suite_report_name(parallel_suite: dict[str, Any] | None, fallback: str) -> str:
    if not parallel_suite:
        return fallback
    run_label = parallel_suite.get("run_label")
    if not run_label:
        artifact_name = str(parallel_suite.get("_artifact_name", ""))
        prefix = "multicard_parallel_suite_"
        suffix = ".json"
        if artifact_name.startswith(prefix) and artifact_name.endswith(suffix):
            run_label = artifact_name[len(prefix) : -len(suffix)]
    if run_label:
        return f"reports/multicard-parallel-suite-{run_label}.md"
    return fallback


def manifest_lines(manifest: dict[str, Any] | None) -> list[str]:
    if manifest is None:
        return ["Manifest: missing"]
    lines = [f"Manifest: {manifest.get('status', 'missing')}"]
    for stage in manifest.get("stages", []):
        if not isinstance(stage, dict):
            continue
        lines.append(
            "  - {name}: {status} ({seconds}s, {log})".format(
                name=stage.get("name", "unknown"),
                status=stage.get("status", "missing"),
                seconds=fmt(stage.get("seconds")),
                log=stage.get("log_path", "missing"),
            )
        )
    return lines


def model_inventory_lines(model_inventory: dict[str, Any] | None) -> list[str]:
    if model_inventory is None:
        return ["Model inventory: missing"]
    lines = [f"Model inventory: {model_inventory.get('status', 'missing')}"]
    for row in model_inventory.get("target_candidates", []):
        lines.append(
            "Target model {model}: {status}".format(
                model=row.get("model_id", "missing"),
                status=row.get("status", "missing"),
            )
        )
    for row in model_inventory.get("tiny_fixtures", []):
        lines.append(
            "Tiny fixture {model}: {status}, {mb} MB".format(
                model=row.get("name") or row.get("model_id", "missing"),
                status=row.get("status", "missing"),
                mb=fmt(row.get("parameter_mb")),
            )
        )
    return lines


def demo_storyline(demo_root: Path) -> str:
    artifacts = demo_root / "artifacts"
    manifest = read_json(artifacts / "ascend_910b_demo_manifest.json")
    model_inventory = read_json(artifacts / "model_inventory.json")
    sync = read_json(artifacts / "multicard_sync_summary.json")
    lora_sync = read_json(artifacts / "rankadaptor_lora_sync_summary.json")
    tiny_lora = read_json(artifacts / "tiny_qwen_lora_finetune_tiny_qwen3_lora_npu.json")
    tiny_bslora = read_json(artifacts / "tiny_qwen_bslora_finetune_tiny_qwen3_bslora_npu.json")
    qwen3_bslora = read_json(artifacts / QWEN3_BSLORA_ARTIFACT)
    bslora_sync = read_json(artifacts / "multicard_tiny_qwen_bslora_finetune_tiny_qwen3_bslora_sync_npu.json")
    compression = first_json(artifacts, COMPRESSION_ARTIFACTS)
    multicard_generate = first_json(artifacts, MULTICARD_GENERATE_ARTIFACTS)
    vllm_probe = read_json(artifacts / DEFAULT_VLLM_PROBE_ARTIFACT)
    vllm_patch_probe = read_json(artifacts / PATCH_VLLM_PROBE_ARTIFACT)
    vllm_selector_shim_probe = read_json(artifacts / SHIM_VLLM_PROBE_ARTIFACT)
    vllm_serving_benchmark = first_json(artifacts, VLLM_SERVING_BENCHMARK_ARTIFACTS)
    inference_bottleneck = read_json(artifacts / INFERENCE_BOTTLENECK_ARTIFACT)
    torch_serving_fallback = read_json(artifacts / TORCH_SERVING_FALLBACK_ARTIFACT)
    compressed_native = read_json(artifacts / COMPRESSED_NATIVE_TORCH_SERVING_ARTIFACT)
    qpruner_packed_decode = read_json(artifacts / QPRUNER_PACKED_DECODE_ARTIFACT)
    inference_acceleration_summary = read_json(artifacts / INFERENCE_ACCELERATION_SUMMARY_ARTIFACT)
    compressed_native_sweep_summary = read_json(artifacts / COMPRESSED_NATIVE_SWEEP_SUMMARY_ARTIFACT)
    multicard_compression_sync_summary = read_json(artifacts / MULTICARD_COMPRESSION_SYNC_SUMMARY_ARTIFACT)
    multicard_qwen_qpruner_grouped_replay = best_multicard_qwen_qpruner_grouped_replay(artifacts)
    qwen_grouped_mlp_sweep = best_qwen_grouped_mlp_sweep(artifacts)
    qwen_grouped_mlp_memory_tradeoff = best_qwen_grouped_mlp_memory_tradeoff(artifacts)
    npu_monitor = best_npu_utilization_monitor(artifacts)
    objective_coverage_audit = read_json(artifacts / OBJECTIVE_COVERAGE_AUDIT_ARTIFACT)
    demo_readiness_report = read_json(artifacts / DEMO_READINESS_REPORT_ARTIFACT)
    qwen_native_profile_name, _ = best_qwen_qpruner_native_profile_artifact(artifacts)
    qwen_native_profile_report = qwen_native_profile_report_name(qwen_native_profile_name)
    qwen_native_memory_profile_name, _ = best_qwen_qpruner_native_memory_profile_artifact(artifacts)
    qwen_native_memory_profile_report = qwen_native_profile_report_name(qwen_native_memory_profile_name)
    qwen_speed_first_profile = selected_qwen_speed_first_profile(inference_bottleneck)
    qwen_speed_first_artifact = qwen_speed_first_profile.get("artifact")
    qwen_speed_first_report = qwen_native_profile_report_name(qwen_speed_first_artifact)
    qwen_native_bits_artifacts = qwen_native_bits_tradeoff_artifacts(inference_bottleneck)
    qwen_native_profile_reports = list(
        dict.fromkeys(
            [
                qwen_native_profile_report,
                qwen_native_memory_profile_report,
                qwen_speed_first_report,
                *(qwen_native_profile_report_name(name) for name in qwen_native_bits_artifacts),
            ]
        )
    )
    qwen_native_profile_artifacts = list(
        dict.fromkeys(
            [
                qwen_native_profile_name,
                qwen_native_memory_profile_name,
                qwen_speed_first_artifact,
                *qwen_native_bits_artifacts,
            ]
        )
    )
    qpruner_shape_sweep = selected_qpruner_shape_sweep(inference_bottleneck)
    qpruner_shape_sweep_artifact = qpruner_shape_sweep.get("artifact")
    qpruner_shape_sweep_report = qpruner_shape_sweep_report_name(qpruner_shape_sweep_artifact)
    npu_monitor_artifact = npu_monitor_artifact_name(npu_monitor)
    npu_monitor_log = npu_monitor_log_path(npu_monitor, demo_root)
    multicard_parallel_suite = read_json(artifacts / MULTICARD_PARALLEL_SUITE_ARTIFACT)
    vllm_parallel_suite = best_vllm_parallel_suite(artifacts)
    vllm_parallel_report = parallel_suite_report_name(
        vllm_parallel_suite,
        "reports/multicard-parallel-suite-vllm_metadata_sync_npu_metrics.md",
    )
    qwen3_snapshot = read_json(artifacts / "model_snapshot_qwen3_06b.json")
    qwen3_inference = read_json(artifacts / "ascend_inference_qwen3_06b_torch_npu.json")
    multicard_qwen_inference = read_json(artifacts / "multicard_qwen_inference_qwen3_06b_npu.json")
    qwen_compression_quality = read_json(artifacts / "qwen_compression_quality_qwen3_06b_quality_npu.json")
    compression_memory = read_json(artifacts / "compression_memory_report.json")
    paper_baseline_audit = read_json(artifacts / PAPER_BASELINE_COVERAGE_AUDIT_ARTIFACT)
    choice_accuracy = (
        read_json(artifacts / FULLTARGET_QPRUNER_CHOICE_ARTIFACT)
        or read_json(artifacts / MULTITASK_CHOICE_ACCURACY_ARTIFACT)
        or read_json(artifacts / CHOICE_ACCURACY_ARTIFACT)
    )
    multicard_qwen_compression_quality = read_json(
        artifacts / "multicard_qwen_compression_quality_qwen3_06b_quality_sync_npu.json"
    )
    multicard_qwen_pattern_quality = read_json(
        artifacts / "multicard_qwen_compression_quality_qwen3_06b_quality_pattern_2card_npu.json"
    )
    qwen_compression_generate = read_json(artifacts / "qwen_compression_generate_qwen3_06b_generate_npu.json")
    multicard_qwen_lora_finetune = read_json(
        artifacts / "multicard_qwen_lora_finetune_qwen3_06b_lora_sync_npu.json"
    )
    multicard_qwen_compression_generate = best_multicard_qwen_compression_generate(artifacts)
    multicard_qwen_compression_generate_artifact = (
        multicard_qwen_compression_generate or {}
    ).get("_artifact_name")
    multicard_qwen_compression_generate_report = (
        multicard_qwen_compression_generate or {}
    ).get("_report_name")
    multicard_qwen_pattern_generate = read_json(
        artifacts / "multicard_qwen_compression_generate_qwen3_06b_generate_pattern_2card_npu.json"
    )

    lines = [
        "# Ascend 910B TIDAL-AI Demo Playback",
        "",
        f"Demo root: {demo_root}",
        "",
        "## 1. One-command demo run",
        *manifest_lines(manifest),
        "",
        "## 2. Model inventory",
        *model_inventory_lines(model_inventory),
        "",
        "## 3. Multi-card synchronization",
    ]
    if sync is None:
        lines.append("8-card HCCL sync: missing")
    else:
        lines.append(
            "8-card HCCL sync: {status}, world size {world}, backend {backend}".format(
                status=sync.get("status", "missing"),
                world=sync.get("world_size", "missing"),
                backend=sync.get("backend", "missing"),
            )
        )

    lines.extend(["", "## 4. Qwen3-0.6B runnable torch_npu inference"])
    if qwen3_inference is None:
        lines.append("Qwen3-0.6B torch_npu inference: missing")
    else:
        lines.append(
            "Qwen3-0.6B torch_npu inference: {status}, backend {backend}, device {device}".format(
                status=qwen3_inference.get("status", "missing"),
                backend=qwen3_inference.get("backend", "missing"),
                device=qwen3_inference.get("device", "missing"),
            )
        )
        lines.append(
            "Qwen3-0.6B latency: {latency} ms, {tps} tokens/s, peak {peak} MB".format(
                latency=fmt(qwen3_inference.get("latency_ms")),
                tps=fmt(qwen3_inference.get("tokens_per_s")),
                peak=fmt(qwen3_inference.get("peak_mem_mb")),
            )
        )
    if qwen3_snapshot is None:
        lines.append("Qwen3-0.6B snapshot: missing")
    else:
        lines.append(
            "Qwen3-0.6B snapshot: {status} via {provider}, {mb} MB".format(
                status=qwen3_snapshot.get("status", "missing"),
                provider=qwen3_snapshot.get("provider", "missing"),
                mb=fmt(qwen3_snapshot.get("parameter_mb")),
            )
        )
    if multicard_qwen_inference is None:
        lines.append("Qwen3-0.6B 8-card inference: missing")
    else:
        aggregate = multicard_qwen_inference.get("aggregate", {})
        lines.append(
            "Qwen3-0.6B 8-card inference: {status}, world size {world}, backend {backend}".format(
                status=multicard_qwen_inference.get("status", "missing"),
                world=multicard_qwen_inference.get("world_size", "missing"),
                backend=multicard_qwen_inference.get("backend", "missing"),
            )
        )
        lines.append(
            "Qwen3-0.6B all-reduce totals: {consistent}, {tps} tokens/s, peak {peak} MB".format(
                consistent=aggregate.get("distributed_reduce_consistent", "missing"),
                tps=fmt(aggregate.get("tokens_per_s_total")),
                peak=fmt(aggregate.get("peak_mem_mb_total")),
            )
        )
    if qwen_compression_generate is None:
        lines.append("Qwen3-0.6B CAP/QPruner generate: missing")
    else:
        cap = qwen_compression_generate.get("cap", {})
        qpruner = qwen_compression_generate.get("qpruner", {})
        lines.append(
            "Qwen3-0.6B CAP/QPruner generate: {status}, target layers {limit}/{total}".format(
                status=qwen_compression_generate.get("status", "missing"),
                limit=qwen_compression_generate.get("target_layer_limit", "missing"),
                total=qwen_compression_generate.get("targeted_layers_total", "missing"),
            )
        )
        lines.append(
            "Qwen3 CAP speedup {cap_speedup}, ratio {cap_ratio}x; QPruner speedup {q_speedup}, bits {q_bits}".format(
                cap_speedup=fmt(cap.get("latency_speedup")),
                cap_ratio=fmt(cap.get("targeted_compression_ratio")),
                q_speedup=fmt(qpruner.get("latency_speedup")),
                q_bits=fmt(qpruner.get("average_bits")),
            )
        )
    if qwen_compression_quality is None:
        lines.append("Qwen3-0.6B CAP/QPruner quality: missing")
    else:
        cap = qwen_compression_quality.get("cap", {})
        qpruner = qwen_compression_quality.get("qpruner", {})
        lines.append(
            "Qwen3-0.6B CAP/QPruner quality: {status}, target layers {limit}/{total}".format(
                status=qwen_compression_quality.get("status", "missing"),
                limit=qwen_compression_quality.get("target_layer_limit", "missing"),
                total=qwen_compression_quality.get("targeted_layers_total", "missing"),
            )
        )
        lines.append(
            "Qwen3 quality loss: baseline {baseline}, CAP delta {cap_delta}, QPruner delta {q_delta}".format(
                baseline=fmt(qwen_compression_quality.get("baseline", {}).get("loss")),
                cap_delta=fmt(cap.get("loss_delta")),
                q_delta=fmt(qpruner.get("loss_delta")),
            )
        )
    lines.extend(
        compression_memory_lines(
            compression_memory,
            include_legacy_baseline_evidence=paper_baseline_audit is None,
        )
    )
    lines.extend(paper_baseline_audit_lines(paper_baseline_audit))
    lines.extend(quality_memory_sweep_lines(demo_root))
    lines.extend(qpruner_quality_memory_sweep_lines(demo_root))
    lines.extend(qpruner_scale_quality_lines(demo_root))
    lines.extend(choice_accuracy_lines(choice_accuracy))
    lines.extend(qpruner_frontier_chart_lines(demo_root))
    if multicard_qwen_compression_quality is None:
        lines.append("Qwen3-0.6B 8-card CAP/QPruner quality: missing")
    else:
        aggregate = multicard_qwen_compression_quality.get("aggregate", {})
        lines.append(
            "Qwen3-0.6B 8-card CAP/QPruner quality: {status}, world size {world}, backend {backend}".format(
                status=multicard_qwen_compression_quality.get("status", "missing"),
                world=multicard_qwen_compression_quality.get("world_size", "missing"),
                backend=multicard_qwen_compression_quality.get("backend", "missing"),
            )
        )
        lines.append(
            "Qwen3 quality all-reduce: {consistent}, baseline loss {baseline}, CAP delta {cap_delta}, QPruner delta {q_delta}".format(
                consistent=aggregate.get("distributed_reduce_consistent", "missing"),
                baseline=fmt(aggregate.get("baseline_loss_avg")),
                cap_delta=fmt(aggregate.get("cap_loss_delta_avg")),
                q_delta=fmt(aggregate.get("qpruner_loss_delta_avg")),
            )
        )
    if multicard_qwen_pattern_quality is None:
        lines.append("Qwen3 patterned quality sync: missing")
    else:
        aggregate = multicard_qwen_pattern_quality.get("aggregate", {})
        lines.append(
            "Qwen3 patterned quality sync: {status}, world size {world}, backend {backend}, pattern {pattern}, target layers {limit}/{total}".format(
                status=multicard_qwen_pattern_quality.get("status", "missing"),
                world=multicard_qwen_pattern_quality.get("world_size", "missing"),
                backend=multicard_qwen_pattern_quality.get("backend", "missing"),
                pattern=multicard_qwen_pattern_quality.get("target_layer_pattern", "missing"),
                limit=multicard_qwen_pattern_quality.get("target_layer_limit", "missing"),
                total=multicard_qwen_pattern_quality.get("targeted_layers_total", "missing"),
            )
        )
        lines.append(
            "Qwen3 patterned quality all-reduce: {consistent}, baseline loss {baseline}, CAP delta {cap_delta}, WANDA delta {wanda_delta}, SparseGPT delta {sparsegpt_delta}, QPruner delta {q_delta}".format(
                consistent=aggregate.get("distributed_reduce_consistent", "missing"),
                baseline=fmt(aggregate.get("baseline_loss_avg")),
                cap_delta=fmt(aggregate.get("cap_loss_delta_avg")),
                wanda_delta=fmt(aggregate.get("wanda_loss_delta_avg")),
                sparsegpt_delta=fmt(aggregate.get("sparsegpt_loss_delta_avg")),
                q_delta=fmt(aggregate.get("qpruner_loss_delta_avg")),
            )
        )
    if multicard_qwen_lora_finetune is None:
        lines.append("Qwen3-0.6B 8-card LoRA fine-tune: missing")
    else:
        aggregate = multicard_qwen_lora_finetune.get("aggregate", {})
        lines.append(
            "Qwen3-0.6B 8-card LoRA fine-tune: {status}, world size {world}, backend {backend}".format(
                status=multicard_qwen_lora_finetune.get("status", "missing"),
                world=multicard_qwen_lora_finetune.get("world_size", "missing"),
                backend=multicard_qwen_lora_finetune.get("backend", "missing"),
            )
        )
        lines.append(
            "Qwen3 LoRA all-reduce loss: {consistent}, {initial} -> {final}, delta {delta}".format(
                consistent=aggregate.get("distributed_reduce_consistent", "missing"),
                initial=fmt(aggregate.get("initial_loss_avg")),
                final=fmt(aggregate.get("final_loss_avg")),
                delta=fmt(aggregate.get("loss_delta_avg")),
            )
        )
        if aggregate.get("validation_initial_loss_avg") is not None:
            lines.append(
                "Qwen3 LoRA validation loss: {initial} -> {final}, delta {delta}".format(
                    initial=fmt(aggregate.get("validation_initial_loss_avg")),
                    final=fmt(aggregate.get("validation_final_loss_avg")),
                    delta=fmt(aggregate.get("validation_loss_delta_avg")),
                )
            )
        if multicard_qwen_lora_finetune.get("after_generate"):
            lines.append(f"Qwen3 LoRA generation before: {multicard_qwen_lora_finetune.get('before_generate', 'missing')}")
            lines.append(f"Qwen3 LoRA generation after: {multicard_qwen_lora_finetune.get('after_generate', 'missing')}")
        lines.append(
            "Qwen3 LoRA adapter sync: {adapter}, peak {peak} MB, passing ranks {passes}".format(
                adapter=aggregate.get("adapter_sync_consistent", "missing"),
                peak=fmt(aggregate.get("peak_mem_mb_total")),
                passes=fmt(aggregate.get("pass_count")),
            )
        )
    if multicard_qwen_compression_generate is None:
        lines.append("Qwen3-0.6B 8-card CAP/QPruner generate: missing")
    else:
        aggregate = multicard_qwen_compression_generate.get("aggregate", {})
        artifact = multicard_qwen_compression_generate.get("_artifact_name")
        report = multicard_qwen_compression_generate.get("_report_name")
        lines.append(
            "Qwen3-0.6B 8-card CAP/QPruner generate: {status}, world size {world}, backend {backend}, target layers {limit}/{total}".format(
                status=multicard_qwen_compression_generate.get("status", "missing"),
                world=multicard_qwen_compression_generate.get("world_size", "missing"),
                backend=multicard_qwen_compression_generate.get("backend", "missing"),
                limit=fmt(multicard_qwen_compression_generate.get("target_layer_limit")),
                total=fmt(multicard_qwen_compression_generate.get("targeted_layers_total")),
            )
        )
        lines.append(
            "Qwen3 compressed all-reduce: {consistent}, CAP {cap} tokens/s, QPruner {qpruner} tokens/s".format(
                consistent=aggregate.get("distributed_reduce_consistent", "missing"),
                cap=fmt(aggregate.get("cap_tokens_per_s_total")),
                qpruner=fmt(aggregate.get("qpruner_tokens_per_s_total")),
            )
        )
        if artifact:
            lines.append(f"Qwen3 compressed generate artifact: artifacts/{artifact}")
        if report:
            lines.append(f"Qwen3 compressed generate report: reports/{report}")
    if multicard_qwen_pattern_generate is None:
        lines.append("Qwen3 patterned generate sync: missing")
    else:
        aggregate = multicard_qwen_pattern_generate.get("aggregate", {})
        lines.append(
            "Qwen3 patterned generate sync: {status}, world size {world}, backend {backend}, pattern {pattern}, target layers {limit}/{total}".format(
                status=multicard_qwen_pattern_generate.get("status", "missing"),
                world=multicard_qwen_pattern_generate.get("world_size", "missing"),
                backend=multicard_qwen_pattern_generate.get("backend", "missing"),
                pattern=multicard_qwen_pattern_generate.get("target_layer_pattern", "missing"),
                limit=multicard_qwen_pattern_generate.get("target_layer_limit", "missing"),
                total=multicard_qwen_pattern_generate.get("targeted_layers_total", "missing"),
            )
        )
        lines.append(
            "Qwen3 patterned generate all-reduce: {consistent}, baseline {baseline} tokens/s, CAP {cap} tokens/s, QPruner {qpruner} tokens/s".format(
                consistent=aggregate.get("distributed_reduce_consistent", "missing"),
                baseline=fmt(aggregate.get("baseline_tokens_per_s_total")),
                cap=fmt(aggregate.get("cap_tokens_per_s_total")),
                qpruner=fmt(aggregate.get("qpruner_tokens_per_s_total")),
            )
        )

    lines.extend(["", "## 5. RankAdaptor LoRA training sync"])
    if lora_sync is None:
        lines.append("RankAdaptor LoRA sync: missing")
    else:
        lines.append(
            "RankAdaptor LoRA sync: {status}, world size {world}, trainable adapter params {params}".format(
                status=lora_sync.get("status", "missing"),
                world=lora_sync.get("world_size", "missing"),
                params=lora_sync.get("trainable_adapter_params", "missing"),
            )
        )

    lines.extend(["", "## 6. TinyQwen LoRA fine-tune"])
    if tiny_lora is None:
        lines.append("TinyQwen LoRA fine-tune: missing")
    else:
        lines.append(
            "TinyQwen LoRA fine-tune: {status}, loss {initial} -> {final}, delta {delta}".format(
                status=tiny_lora.get("status", "missing"),
                initial=fmt(tiny_lora.get("initial_loss")),
                final=fmt(tiny_lora.get("final_loss")),
                delta=fmt(tiny_lora.get("loss_delta")),
            )
        )
        lines.append(f"Generation before: {tiny_lora.get('before_generate', 'missing')}")
        lines.append(f"Generation after: {tiny_lora.get('after_generate', 'missing')}")

    lines.extend(["", "## 7. TinyQwen BSLoRA shared fine-tune"])
    if tiny_bslora is None:
        lines.append("TinyQwen BSLoRA fine-tune: missing")
    else:
        lines.append(
            "TinyQwen BSLoRA fine-tune: {status}, loss {initial} -> {final}, delta {delta}".format(
                status=tiny_bslora.get("status", "missing"),
                initial=fmt(tiny_bslora.get("initial_loss")),
                final=fmt(tiny_bslora.get("final_loss")),
                delta=fmt(tiny_bslora.get("loss_delta")),
            )
        )
        lines.append(
            "BSLoRA sharing: trainable {trainable} vs unshared {unshared} adapter params".format(
                trainable=tiny_bslora.get("trainable_adapter_params", "missing"),
                unshared=tiny_bslora.get("unshared_adapter_params", "missing"),
            )
        )

    lines.extend(["", "## 8. Qwen3-0.6B BSLoRA shared fine-tune"])
    if qwen3_bslora is None:
        lines.append("Qwen3 BSLoRA fine-tune: missing")
    else:
        lines.append(
            "Qwen3 BSLoRA fine-tune: {status}, loss {initial} -> {final}, delta {delta}".format(
                status=qwen3_bslora.get("status", "missing"),
                initial=fmt(qwen3_bslora.get("initial_loss")),
                final=fmt(qwen3_bslora.get("final_loss")),
                delta=fmt(qwen3_bslora.get("loss_delta")),
            )
        )
        lines.append(
            "Qwen3 BSLoRA sharing: target modules {modules}, trainable {trainable} vs unshared {unshared} adapter params".format(
                modules=qwen3_bslora.get("target_module_count", "missing"),
                trainable=qwen3_bslora.get("trainable_adapter_params", "missing"),
                unshared=qwen3_bslora.get("unshared_adapter_params", "missing"),
            )
        )
        lines.append(
            "Qwen3 BSLoRA tokens trained {tokens}, peak {peak} MB".format(
                tokens=qwen3_bslora.get("tokens_trained", "missing"),
                peak=fmt(qwen3_bslora.get("peak_mem_mb")),
            )
        )

    lines.extend(["", "## 9. TinyQwen BSLoRA 8-card sync"])
    if bslora_sync is None:
        lines.append("8-card BSLoRA sync: missing")
    else:
        aggregate = bslora_sync.get("aggregate", {})
        lines.append(
            "8-card BSLoRA sync: {status}, world size {world}, backend {backend}".format(
                status=bslora_sync.get("status", "missing"),
                world=bslora_sync.get("world_size", "missing"),
                backend=bslora_sync.get("backend", "missing"),
            )
        )
        lines.append(
            "BSLoRA all-reduce: {reduce}, adapter checksum sync {adapter}".format(
                reduce=aggregate.get("distributed_reduce_consistent", "missing"),
                adapter=aggregate.get("adapter_sync_consistent", "missing"),
            )
        )
        lines.append(
            "BSLoRA average loss: {initial} -> {final}, delta {delta}".format(
                initial=fmt(aggregate.get("initial_loss_avg")),
                final=fmt(aggregate.get("final_loss_avg")),
                delta=fmt(aggregate.get("loss_delta_avg")),
            )
        )

    lines.extend(["", "## 10. CAP/QPruner compressed generate"])
    if compression is None:
        lines.append("TinyQwen compressed generate: missing")
    else:
        baseline = compression.get("baseline", {})
        cap = compression.get("cap", {})
        qpruner = compression.get("qpruner", {})
        lines.append(
            "TinyQwen compressed generate: {status}, backend {backend}, serving dense export {export}".format(
                status=compression.get("status", "missing"),
                backend=compression.get("backend", "missing"),
                export=compression.get("serving_dense_export", "missing"),
            )
        )
        lines.append(
            "Baseline generate: {latency} ms, {tps} tokens/s, peak {peak} MB".format(
                latency=fmt(baseline.get("latency_ms")),
                tps=fmt(baseline.get("tokens_per_s")),
                peak=fmt(compression.get("peak_mem_mb")),
            )
        )
        lines.append(
            "CAP compression: {ratio}x targeted, speedup {speedup}".format(
                ratio=fmt(cap.get("targeted_compression_ratio")),
                speedup=fmt(cap.get("latency_speedup")),
            )
        )
        lines.append(
            "QPruner compression: {bits} avg bits, speedup {speedup}".format(
                bits=fmt(qpruner.get("average_bits")),
                speedup=fmt(qpruner.get("latency_speedup")),
            )
        )

    lines.extend(["", "## 11. Multi-card compressed generate"])
    if multicard_generate is None:
        lines.append("Multi-card compressed generate: missing")
    else:
        aggregate = multicard_generate.get("aggregate", {})
        lines.append(
            "Multi-card compressed generate: {status}, world size {world}, backend {backend}".format(
                status=multicard_generate.get("status", "missing"),
                world=multicard_generate.get("world_size", "missing"),
                backend=multicard_generate.get("backend", "missing"),
            )
        )
        lines.append(
            "all-reduce throughput totals consistent: {consistent}".format(
                consistent=aggregate.get("distributed_reduce_consistent", "missing"),
            )
        )
        lines.append(f"Baseline total: {fmt(aggregate.get('baseline_tokens_per_s_total'))} tokens/s")
        lines.append(f"CAP total: {fmt(aggregate.get('cap_tokens_per_s_total'))} tokens/s")
        lines.append(f"QPruner total: {fmt(aggregate.get('qpruner_tokens_per_s_total'))} tokens/s")

    lines.extend(["", "## 12. torch_npu serving fallback"])
    lines.extend(torch_fallback_lines(torch_serving_fallback))
    lines.extend(compressed_native_lines(compressed_native))
    lines.extend(qpruner_packed_decode_lines(qpruner_packed_decode))
    lines.extend(inference_acceleration_summary_lines(inference_acceleration_summary))
    lines.extend(compressed_native_sweep_lines(compressed_native_sweep_summary))
    lines.extend(multicard_compression_sync_lines(multicard_compression_sync_summary))
    lines.extend(multicard_qwen_qpruner_grouped_replay_lines(multicard_qwen_qpruner_grouped_replay))
    lines.extend(qwen_grouped_mlp_sweep_lines(qwen_grouped_mlp_sweep))
    lines.extend(qwen_grouped_mlp_memory_tradeoff_lines(qwen_grouped_mlp_memory_tradeoff))
    lines.extend(["", "## 13. 8-card utilization proof"])
    lines.extend(npu_monitor_lines(npu_monitor, demo_root))
    lines.extend(demo_readiness_lines(demo_readiness_report))
    lines.extend(objective_coverage_lines(objective_coverage_audit))

    lines.extend(["", "## 14. Synchronized multi-card parallel suite"])
    lines.extend(parallel_suite_lines(multicard_parallel_suite))
    lines.extend(vllm_parallel_suite_lines(vllm_parallel_suite))

    lines.extend(["", "## 15. vLLM serving compatibility boundary"])
    if vllm_probe is None:
        lines.append("vLLM serving probe: missing")
    else:
        lines.append(
            "vLLM serving probe: {status}, classes {classes}".format(
                status=vllm_probe.get("status", "missing"),
                classes=issue_classes(vllm_probe),
            )
        )
        lines.append(probe_load_summary(vllm_probe))
        lines.extend(api_diagnostic_summary(vllm_probe))
        acl = vllm_probe.get("acl_diagnostics", {}) if isinstance(vllm_probe.get("acl_diagnostics"), dict) else {}
        lines.append(
            "ACL import for default probe: {acl}, gpu_memory_utilization {gpu}".format(
                acl=acl.get("import_acl", "missing"),
                gpu=fmt(vllm_probe.get("gpu_memory_utilization")),
            )
        )
    if vllm_patch_probe is None:
        lines.append("vLLM patch-preload probe: missing")
    else:
        lines.append(
            "vLLM patch-preload probe: {status}, classes {classes}".format(
                status=vllm_patch_probe.get("status", "missing"),
                classes=issue_classes(vllm_patch_probe),
            )
        )
        lines.append(
            "Patch preload flag: {flag}, export run label {label}".format(
                flag=vllm_patch_probe.get("preload_vllm_ascend_patch", "missing"),
                label=vllm_patch_probe.get("export_run_label", "missing"),
            )
        )
    if vllm_selector_shim_probe is None:
        lines.append("vLLM selector-shim probe: missing")
    else:
        lines.append(
            "vLLM selector-shim probe: {status}, classes {classes}".format(
                status=vllm_selector_shim_probe.get("status", "missing"),
                classes=issue_classes(vllm_selector_shim_probe),
            )
        )
        lines.append(probe_load_summary(vllm_selector_shim_probe))
        lines.append(
            "Selector shim flag: {flag}, export run label {label}".format(
                flag=vllm_selector_shim_probe.get("preload_vllm_ascend_selector_shim", "missing"),
                label=vllm_selector_shim_probe.get("export_run_label", "missing"),
            )
        )
    lines.extend(vllm_benchmark_lines(vllm_serving_benchmark))
    lines.extend(["", "## 16. Inference bottleneck diagnosis", *inference_bottleneck_lines(inference_bottleneck)])

    lines.extend(
        [
            "",
            "## 17. Evidence to show on camera",
            f"- reports/ascend-910b-demo-manifest.md",
            f"- reports/ascend-910b-demo-progress.md",
            f"- reports/demo-storyboard.md",
            f"- reports/demo-readiness-report.md",
            f"- reports/model-inventory.md",
            f"- reports/model-snapshot-qwen3_06b.md",
            f"- reports/ascend-inference-qwen3_06b_torch_npu.md",
            f"- reports/multicard-qwen-inference-qwen3_06b_npu.md",
            f"- reports/qwen-compression-quality-qwen3_06b_quality_npu.md",
            f"- reports/qwen-llm-pruner-baseline-qwen3_06b_llm_pruner_npu.md",
            f"- reports/compression-memory-report.md",
            f"- reports/qwen-compression-quality-memory-sweep.md",
            f"- reports/qwen-qpruner-quality-memory-sweep.md",
            f"- reports/qwen-qpruner-scale-quality-summary.md",
            f"- reports/{CHOICE_ACCURACY_REPORT}",
            f"- reports/{MULTITASK_CHOICE_ACCURACY_REPORT}",
            f"- reports/{FULLTARGET_QPRUNER_CHOICE_REPORT}",
            f"- reports/qwen-qpruner-quality-memory-frontier.svg",
            f"- reports/qwen-qpruner-quality-memory-frontier.md",
            f"- reports/multicard-qwen-compression-quality-qwen3_06b_quality_sync_npu.md",
            f"- reports/multicard-qwen-compression-quality-qwen3_06b_quality_pattern_2card_npu.md",
            f"- reports/qwen-compression-generate-qwen3_06b_generate_npu.md",
            f"- reports/qwen-compression-generate-sweep.md",
            f"- reports/multicard-qwen-lora-finetune-qwen3_06b_lora_sync_npu.md",
            f"- reports/tiny-qwen-bslora-finetune-qwen3_06b_bslora_npu.md",
            f"- reports/multicard-qwen-compression-generate-qwen3_06b_compression_generate_npu.md",
            f"- reports/multicard-qwen-compression-generate-qwen3_06b_generate_pattern_2card_npu.md",
            f"- reports/multicard-tiny-qwen-bslora-finetune-tiny_qwen3_bslora_sync_npu.md",
            f"- reports/multicard-tiny-qwen-generate-tiny_qwen3_multicard_generate_npu.md",
            f"- reports/tiny-qwen-compression-generate-tiny_qwen3_generate_serving_export_npu.md",
            f"- reports/torch-serving-fallback-tiny_qwen3_serving_fallback_npu.md",
            f"- reports/compressed-native-torch-serving-tiny_qwen3_native_serving_npu.md",
            f"- reports/inference-acceleration-summary.md",
            f"- reports/inference-acceleration-summary.csv",
            f"- reports/inference-acceleration-summary.svg",
            f"- reports/compressed-native-cache-long-decode-sweep.md",
            f"- reports/compressed-native-cache-long-decode-sweep.csv",
            f"- reports/compressed-native-cache-long-decode-sweep.svg",
            f"- reports/qpruner-packed-decode-benchmark-tiny_qwen3_qpruner_decode_npu.md",
            f"- reports/qpruner-packed-decode-benchmark-qwen3_06b_shape_sweep_npu.md",
            *([] if not qpruner_shape_sweep_report else [f"- reports/{qpruner_shape_sweep_report}"]),
            f"- reports/{QPRUNER_DTYPE_SHAPE_SWEEP_REPORT}",
            *(f"- reports/{report}" for report in qwen_native_profile_reports),
            f"- reports/{qwen_grouped_mlp_sweep_report_name(qwen_grouped_mlp_sweep)}",
            f"- reports/multicard-compression-sync-summary.md",
            f"- reports/multicard-compression-sync-summary.csv",
            f"- reports/multicard-compression-sync-summary.svg",
            *(
                []
                if not multicard_qwen_compression_generate_report
                else [f"- reports/{multicard_qwen_compression_generate_report}"]
            ),
            f"- reports/objective-coverage-audit.md",
            f"- reports/multicard-parallel-suite-demo_parallel_suite_npu.md",
            f"- {vllm_parallel_report}",
            f"- reports/vllm-serving-probe-tiny_qwen3_serving_export_npu.md",
            f"- reports/vllm-serving-probe-tiny_qwen3_serving_export_npu_preload_patch.md",
            f"- reports/vllm-serving-probe-tiny_qwen3_serving_export_npu_selector_shim.md",
            f"- reports/vllm-serving-benchmark-tiny_qwen3_serving_vllm_metadata_shim_npu.md",
            f"- reports/inference-bottleneck-report.md",
            f"- artifacts/inference_acceleration_summary.json",
            f"- artifacts/compressed_native_sweep_summary.json",
            f"- artifacts/qpruner_packed_decode_benchmark_tiny_qwen3_qpruner_decode_npu.json",
            f"- artifacts/qpruner_packed_decode_benchmark_qwen3_06b_shape_sweep_npu.json",
            *([] if not qpruner_shape_sweep_artifact else [f"- artifacts/{qpruner_shape_sweep_artifact}"]),
            f"- artifacts/{QPRUNER_DTYPE_SHAPE_SWEEP_ARTIFACT}",
            *(f"- artifacts/{artifact}" for artifact in qwen_native_profile_artifacts),
            *(
                f"- {item}"
                for item in qwen_grouped_mlp_sweep_evidence(qwen_grouped_mlp_sweep)
                if not item.startswith("reports/")
            ),
            *(
                f"- {item}"
                for item in qwen_grouped_mlp_sweep_evidence(qwen_grouped_mlp_memory_tradeoff)
                if not item.startswith("reports/")
            ),
            f"- artifacts/{CHOICE_ACCURACY_ARTIFACT}",
            f"- artifacts/{MULTITASK_CHOICE_ACCURACY_ARTIFACT}",
            f"- artifacts/{FULLTARGET_QPRUNER_CHOICE_ARTIFACT}",
            f"- artifacts/multicard_compression_sync_summary.json",
            *(
                []
                if not multicard_qwen_compression_generate_artifact
                else [f"- artifacts/{multicard_qwen_compression_generate_artifact}"]
            ),
            *([] if not npu_monitor_artifact else [f"- artifacts/{npu_monitor_artifact}"]),
            *([] if not npu_monitor_log else [f"- {npu_monitor_log}"]),
            f"- artifacts/objective_coverage_audit.json",
            f"- logs/*.log",
        ]
    )
    return "\n".join(lines) + "\n"


def record_command(demo_root: Path) -> tuple[list[str], Path]:
    recordings = demo_root / "recordings"
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    transcript = recordings / f"ascend-910b-demo-playback-{timestamp}.typescript"
    command = [
        "script",
        "-q",
        str(transcript),
        "-c",
        f"{shutil.which('python') or sys.executable} {Path(__file__).resolve()} --demo-root {demo_root}",
    ]
    return command, transcript


def latest_recording_path(demo_root: Path) -> Path:
    return demo_root / "recordings" / "ascend-910b-demo-playback-latest.typescript"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay preserved Ascend 910B demo artifacts")
    parser.add_argument("--demo-root", default="/mnt/nvme/622/tidal-demo")
    parser.add_argument("--record", action="store_true", help="Record this playback with the system script command.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    demo_root = Path(args.demo_root)
    if args.record:
        command, transcript = record_command(demo_root)
        transcript.parent.mkdir(parents=True, exist_ok=True)
        if shutil.which("script") is None:
            print("script command is not available; run without --record to print playback.")
            return 1
        print(f"Recording terminal playback to {transcript}")
        result = subprocess.run(command, check=False)
        if result.returncode == 0 and transcript.exists():
            shutil.copyfile(transcript, latest_recording_path(demo_root))
        return result.returncode
    print(demo_storyline(demo_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
