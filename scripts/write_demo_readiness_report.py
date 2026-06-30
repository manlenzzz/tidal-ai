#!/usr/bin/env python
"""Map the Ascend 910B demo objective to concrete evidence and remaining gaps."""
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

from tidal.reports.baselines import engineering_baseline_note, paper_baseline_note
from tidal.reports.multicard_selection import (
    best_multicard_qwen_compression_generate,
    best_multicard_qwen_lora_finetune,
    multicard_qwen_compression_generate_report_name,
)
from tidal.reports.qwen_grouped_mlp_sweep import (
    best_qwen_grouped_mlp_memory_tradeoff,
    best_qwen_grouped_mlp_sweep,
    qwen_grouped_mlp_memory_tradeoff_readout,
    qwen_grouped_mlp_sweep_evidence,
    qwen_grouped_mlp_sweep_readout,
)
from tidal.reports.qwen_native_tradeoff import (
    best_qwen_native_8card_profile_summary,
    qwen_native_8card_profile_evidence,
    qwen_native_8card_profile_readout,
    qwen_native_bits_tradeoff_evidence,
    qwen_native_bits_tradeoff_readout,
    qwen_native_profile_readout,
)


QPRUNER_PACKED_DECODE_ARTIFACT = "qpruner_packed_decode_benchmark_tiny_qwen3_qpruner_decode_npu.json"
QPRUNER_PACKED_DECODE_REPORT = "qpruner-packed-decode-benchmark-tiny_qwen3_qpruner_decode_npu.md"
QPRUNER_SHAPE_SWEEP_ARTIFACT = "qpruner_packed_decode_benchmark_qwen3_06b_shape_sweep_npu.json"
QPRUNER_SHAPE_SWEEP_REPORT = "qpruner-packed-decode-benchmark-qwen3_06b_shape_sweep_npu.md"
QPRUNER_DTYPE_SHAPE_SWEEP_ARTIFACT = "qpruner_packed_decode_benchmark_qwen3_06b_shape_sweep_dtype_preserving_npu.json"
QPRUNER_DTYPE_SHAPE_SWEEP_REPORT = "qpruner-packed-decode-benchmark-qwen3_06b_shape_sweep_dtype_preserving_npu.md"
CHOICE_ACCURACY_ARTIFACT = "qwen_compression_choice_accuracy_qwen3_06b_choice_accuracy_npu.json"
CHOICE_ACCURACY_REPORT = "qwen-compression-choice-accuracy-qwen3_06b_choice_accuracy_npu.md"
CHOICE_ACCURACY_CSV = "qwen-compression-choice-accuracy-qwen3_06b_choice_accuracy_npu.csv"
MULTITASK_CHOICE_ACCURACY_ARTIFACT = "qwen_compression_choice_accuracy_qwen3_06b_multitask_choice_npu.json"
MULTITASK_CHOICE_ACCURACY_REPORT = "qwen-compression-choice-accuracy-qwen3_06b_multitask_choice_npu.md"
MULTITASK_CHOICE_ACCURACY_CSV = "qwen-compression-choice-accuracy-qwen3_06b_multitask_choice_npu.csv"
FULLTARGET_QPRUNER_CHOICE_ARTIFACT = (
    "qwen_compression_choice_accuracy_qwen3_06b_multitask_choice_fulltarget_qpruner_npu.json"
)
FULLTARGET_QPRUNER_CHOICE_REPORT = (
    "qwen-compression-choice-accuracy-qwen3_06b_multitask_choice_fulltarget_qpruner_npu.md"
)
FULLTARGET_QPRUNER_CHOICE_CSV = (
    "qwen-compression-choice-accuracy-qwen3_06b_multitask_choice_fulltarget_qpruner_npu.csv"
)


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def first_existing_json(artifacts: Path, names: Sequence[str]) -> dict[str, Any] | None:
    for name in names:
        payload = read_json(artifacts / name)
        if payload is not None:
            return payload
    return None


def artifact_from_payload(payload: dict[str, Any] | None, default: str) -> str:
    return str((payload or {}).get("_artifact_name") or default)


def report_from_payload(payload: dict[str, Any] | None, default: str) -> str:
    return str((payload or {}).get("_report_name") or default)


def qpruner_shape_sweep_report_name(artifact_name: str | None) -> str | None:
    if not artifact_name:
        return None
    prefix = "qpruner_packed_decode_benchmark_"
    if not artifact_name.startswith(prefix) or not artifact_name.endswith(".json"):
        return None
    run_label = artifact_name[len(prefix) : -len(".json")]
    return f"qpruner-packed-decode-benchmark-{run_label}.md"


def qwen_native_profile_report_name(artifact_name: str | None) -> str | None:
    if not artifact_name:
        return None
    prefix = "qwen_qpruner_native_profile_"
    if not artifact_name.startswith(prefix) or not artifact_name.endswith(".json"):
        return None
    run_label = artifact_name[len(prefix) : -len(".json")]
    return f"qwen-qpruner-native-profile-{run_label}.md"


def selected_qpruner_shape_sweep(inference_bottleneck: dict[str, Any] | None) -> dict[str, Any]:
    if not inference_bottleneck:
        return {}
    sweep = inference_bottleneck.get("qpruner_packed_decode_shape_sweep")
    return sweep if isinstance(sweep, dict) else {}


def selected_qwen_native_profile(inference_bottleneck: dict[str, Any] | None) -> dict[str, Any]:
    if not inference_bottleneck:
        return {}
    profile = inference_bottleneck.get("qwen_qpruner_native_profile")
    return profile if isinstance(profile, dict) else {}


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


def best_vllm_parallel_suite(artifacts: Path) -> dict[str, Any] | None:
    candidates: list[tuple[tuple[int, int, float], dict[str, Any]]] = []
    for path in artifacts.glob("multicard_parallel_suite_vllm_metadata_sync*_metrics.json"):
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
    return max(candidates, key=lambda row: row[0])[1] if candidates else None


def vllm_parallel_artifact_name(parallel_suite: dict[str, Any] | None) -> str:
    return str((parallel_suite or {}).get("_artifact_name") or "multicard_parallel_suite_vllm_metadata_sync_npu_metrics.json")


def vllm_parallel_report_name(parallel_suite: dict[str, Any] | None) -> str:
    run_label = (parallel_suite or {}).get("run_label")
    if not run_label:
        artifact_name = str((parallel_suite or {}).get("_artifact_name") or "")
        prefix = "multicard_parallel_suite_"
        suffix = ".json"
        if artifact_name.startswith(prefix) and artifact_name.endswith(suffix):
            run_label = artifact_name[len(prefix) : -len(suffix)]
    if run_label:
        return f"multicard-parallel-suite-{run_label}.md"
    return "multicard-parallel-suite-vllm_metadata_sync_npu_metrics.md"


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


def issue_classes(probe: dict[str, Any] | None) -> str:
    if not probe:
        return "missing"
    rows = probe.get("issue_classes", [])
    classes = [str(row.get("id")) for row in rows if isinstance(row, dict) and row.get("id")]
    return ", ".join(classes) if classes else "none"


def method_label(method: str) -> str:
    return {"baseline": "baseline", "cap": "CAP", "qpruner": "QPruner"}.get(method, method)


def join_values(values: Any) -> str:
    if not isinstance(values, list):
        return "missing"
    return ", ".join(str(value) for value in values) or "missing"


def paper_baseline_evidence_metrics(metrics: Any) -> str:
    if not isinstance(metrics, dict) or not metrics:
        return "missing"
    return ", ".join(f"{key} {fmt(value)}" for key, value in metrics.items())


def paper_baseline_evidence_metrics_inline(metrics: Any) -> str:
    text = paper_baseline_evidence_metrics(metrics)
    return f"; metrics {text}" if text != "missing" else ""


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


def paper_baseline_audit_metrics(report: dict[str, Any] | None) -> list[str]:
    if not report:
        return ["Paper baseline audit missing"]

    metrics = [f"Paper baseline audit {report.get('status', 'missing')}"]
    evidence = report.get("evidence", {}) if isinstance(report.get("evidence"), dict) else {}

    qpruner = evidence.get("qpruner_llm_pruner_full196", {})
    if isinstance(qpruner, dict) and qpruner:
        metrics.append(
            "QPruner LLM-Pruner-style {status}, target layers {layers}/{total}; prune {prune}%, "
            "loss delta {loss}, speedup {speedup}x".format(
                status=qpruner.get("status", "missing"),
                layers=fmt(qpruner.get("targeted_layers")),
                total=fmt(qpruner.get("targeted_layers_total")),
                prune=fmt(qpruner.get("targeted_param_reduction_pct")),
                loss=fmt(qpruner.get("loss_delta")),
                speedup=fmt(qpruner.get("latency_speedup")),
            )
        )

    cap = evidence.get("cap_wanda_sparsegpt_full196", {})
    if isinstance(cap, dict) and cap:
        target_layers = "{layers}/{total}".format(
            layers=fmt(cap.get("targeted_layers")),
            total=fmt(cap.get("targeted_layers_total")),
        )
        metrics.append(
            "CAP Wanda {status}, target layers {target_layers}; sparsity {sparsity}%, "
            "loss delta {loss}, speedup {speedup}x".format(
                status=cap.get("status", "missing"),
                target_layers=target_layers,
                sparsity=fmt(cap.get("wanda_targeted_param_reduction_pct")),
                loss=fmt(cap.get("wanda_loss_delta")),
                speedup=fmt(cap.get("wanda_latency_speedup")),
            )
        )
        metrics.append(
            "CAP SparseGPT {status}, target layers {target_layers}; sparsity {sparsity}%, "
            "loss delta {loss}, speedup {speedup}x".format(
                status=cap.get("status", "missing"),
                target_layers=target_layers,
                sparsity=fmt(cap.get("sparsegpt_targeted_param_reduction_pct")),
                loss=fmt(cap.get("sparsegpt_loss_delta")),
                speedup=fmt(cap.get("sparsegpt_latency_speedup")),
            )
        )

    rankadaptor = evidence.get("rankadaptor_recovery_controlled", {})
    if isinstance(rankadaptor, dict) and rankadaptor:
        metrics.append(
            "RankAdaptor recovery baselines {status}, best {best}; no-recovery delta {none}, "
            "LoRA delta {lora}, AdaLoRA-style delta {adalora}".format(
                status=rankadaptor.get("status", "missing"),
                best=rankadaptor.get("best_recovery_method", "missing"),
                none=fmt(rankadaptor.get("no_recovery_loss_delta")),
                lora=fmt(rankadaptor.get("lora_loss_delta")),
                adalora=fmt(rankadaptor.get("adalora_style_loss_delta")),
            )
        )

    return metrics


def compression_memory_metrics(
    report: dict[str, Any] | None, *, include_legacy_baseline_evidence: bool = True
) -> list[str]:
    if not report:
        return ["compression memory report missing"]
    paper = report.get("paper_baselines", {}) if isinstance(report.get("paper_baselines"), dict) else {}
    memory = report.get("compression_memory", {}) if isinstance(report.get("compression_memory"), dict) else {}
    gap = report.get("memory_gap", {}) if isinstance(report.get("memory_gap"), dict) else {}
    qpruner = paper.get("qpruner", {}) if isinstance(paper.get("qpruner"), dict) else {}
    cap = paper.get("cap", {}) if isinstance(paper.get("cap"), dict) else {}
    metrics = [
        compression_memory_video_readout(report),
        "QPruner paper baseline {baseline}; recovery baselines {recovery}".format(
            baseline=join_values(qpruner.get("primary")),
            recovery=join_values(qpruner.get("recovery_baselines")),
        ),
        "CAP paper baselines {baselines}".format(
            baselines=join_values(cap.get("pruning_baselines")),
        ),
        "memory reductions CAP {cap}%, QPruner {qpruner}%; target coverage {coverage}%".format(
            cap=fmt((memory.get("cap", {}) if isinstance(memory.get("cap"), dict) else {}).get("targeted_param_reduction_pct")),
            qpruner=fmt(
                (memory.get("qpruner", {}) if isinstance(memory.get("qpruner"), dict) else {}).get(
                    "targeted_param_reduction_pct"
                )
            ),
            coverage=fmt(memory.get("target_layer_coverage_pct")),
        ),
        "dense export erases storage savings {erases}; next {next_action}".format(
            erases=gap.get("dense_export_erases_storage_savings", "missing"),
            next_action=report.get("next_action", "missing"),
        ),
    ]
    for item in report.get("baseline_alignment_matrix", []):
        if not isinstance(item, dict):
            continue
        metrics.append(
            "baseline alignment {method}: {role} {baselines} -> {demo_role}".format(
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
            metrics.append(
                "paper baseline evidence {method} {baseline} {status}{metric_text}".format(
                    method=item.get("method", "missing"),
                    baseline=item.get("paper_baseline", "missing"),
                    status=item.get("status", "missing"),
                    metric_text=paper_baseline_evidence_metrics_inline(item.get("metrics")),
                )
            )
    return metrics


def quality_memory_sweep_text(demo_root: Path) -> str:
    path = demo_root / "reports" / "qwen-compression-quality-memory-sweep.md"
    if not path.exists():
        return "quality-memory sweep missing"
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if line.startswith("Best memory-quality point"):
            return line
    return "quality-memory sweep present without frontier summary"


def qpruner_quality_memory_sweep_text(demo_root: Path) -> str:
    path = demo_root / "reports" / "qwen-qpruner-quality-memory-sweep.md"
    if not path.exists():
        return "QPruner-only quality-memory sweep missing"
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if line.startswith("Best memory-quality point"):
            return f"QPruner-only quality-memory sweep: {line}"
    return "QPruner-only quality-memory sweep present without frontier summary"


def qpruner_scale_quality_text(demo_root: Path) -> str:
    payload = read_json(demo_root / "artifacts" / "qwen_qpruner_scale_quality_summary.json")
    if isinstance(payload, dict) and payload.get("readout"):
        return str(payload["readout"])
    path = demo_root / "reports" / "qwen-qpruner-scale-quality-summary.md"
    if not path.exists():
        return "QPruner scale-quality summary missing"
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if line.startswith("QPruner scale-quality sweep"):
            return line
    return "QPruner scale-quality summary present without readout"


def choice_accuracy_text(report: dict[str, Any] | None) -> str:
    if not report:
        return "choice accuracy table missing"
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
        return (
            "full-target QPruner choice accuracy baseline {baseline}, QPruner {qpruner}; "
            "choice accuracy target layers {limit}/{total}{sample_suffix}{task_suffix}{compression_suffix}"
        ).format(
            baseline=fmt_accuracy(baseline.get("accuracy")),
            qpruner=fmt_accuracy(qpruner.get("accuracy")),
            limit=qpruner_layers,
            total=report.get("targeted_layers_total", "missing"),
            sample_suffix=sample_suffix,
            task_suffix=task_suffix,
            compression_suffix=compression_suffix,
        )
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


def compressed_native_metrics(report: dict[str, Any] | None) -> list[str]:
    if not report:
        return ["compressed-native torch_npu benchmark missing"]
    summary = report.get("summary", {}) if isinstance(report.get("summary"), dict) else {}
    memory = report.get("memory_reference", {}) if isinstance(report.get("memory_reference"), dict) else {}
    baseline = report.get("baseline", {}) if isinstance(report.get("baseline"), dict) else {}
    cap = report.get("cap", {}) if isinstance(report.get("cap"), dict) else {}
    qpruner = report.get("qpruner", {}) if isinstance(report.get("qpruner"), dict) else {}
    return [
        "compressed-native torch_npu baseline {baseline} tokens/s, CAP {cap} tokens/s, QPruner {qpruner} tokens/s, qpruner_vs_baseline={speedup}x".format(
            baseline=fmt(baseline.get("tokens_per_s")),
            cap=fmt(cap.get("tokens_per_s")),
            qpruner=fmt(qpruner.get("tokens_per_s")),
            speedup=fmt(summary.get("qpruner_vs_baseline_speedup")),
        ),
        "compressed-native modules packed {packed}, quantized {quantized}, exported dense {cap_dense}/{q_dense}".format(
            packed=fmt(cap.get("packed_layers")),
            quantized=fmt(qpruner.get("quantized_layers")),
            cap_dense=fmt(cap.get("exported_dense_linears")),
            q_dense=fmt(qpruner.get("exported_dense_linears")),
        ),
        "native memory reductions CAP {cap}%, QPruner {qpruner}%; dense export {dense}".format(
            cap=fmt(memory.get("cap_targeted_param_reduction_pct")),
            qpruner=fmt(memory.get("qpruner_targeted_param_reduction_pct")),
            dense=report.get("serving_dense_export", "missing"),
        ),
        "native storage bytes baseline {baseline}, CAP {cap}, QPruner {qpruner}".format(
            baseline=fmt(memory.get("baseline_targeted_storage_bytes")),
            cap=fmt(memory.get("cap_targeted_storage_bytes")),
            qpruner=fmt(memory.get("qpruner_targeted_storage_bytes")),
        ),
        "native storage reductions CAP {cap}%, QPruner {qpruner}%".format(
            cap=fmt(memory.get("cap_targeted_storage_reduction_pct")),
            qpruner=fmt(memory.get("qpruner_targeted_storage_reduction_pct")),
        ),
    ]


def inference_acceleration_summary_metrics(summary: dict[str, Any] | None) -> list[str]:
    if not summary:
        return ["inference acceleration summary missing"]
    payload = summary.get("summary", {}) if isinstance(summary.get("summary"), dict) else {}
    best = payload.get("best_throughput", {}) if isinstance(payload.get("best_throughput"), dict) else {}
    native = payload.get("best_native_memory", {}) if isinstance(payload.get("best_native_memory"), dict) else {}
    grouped = payload.get("grouped_replay", {}) if isinstance(payload.get("grouped_replay"), dict) else {}
    bottleneck = summary.get("vllm_bottleneck", {}) if isinstance(summary.get("vllm_bottleneck"), dict) else {}
    metrics = [
        "inference summary best {track}/{method} {tokens} tokens/s".format(
            track=best.get("track", "missing"),
            method=best.get("method_label") or best.get("method", "missing"),
            tokens=fmt(best.get("tokens_per_s")),
        ),
        "inference summary native memory {method} {memory}%, storage {storage}%".format(
            method=native.get("method_label") or native.get("method", "missing"),
            memory=fmt(native.get("memory_reduction_pct")),
            storage=fmt(native.get("storage_reduction_pct")),
        ),
        f"inference summary bottleneck {bottleneck.get('title', 'missing')}",
    ]
    if grouped:
        monitor = grouped.get("monitor", {}) if isinstance(grouped.get("monitor"), dict) else {}
        metrics.append(
            "inference summary grouped replay {status}, world {world}, memory-native workers {memory}/{passes}, "
            "code-cache storage {storage}%, mean grouped speedup {mean_speedup}x, "
            "best grouped speedup {best_speedup}x, {aicore_samples} all-8 AICore samples, "
            "released packed-code bytes {released}, live compressed payload bytes {payload}".format(
                status=grouped.get("status", "missing"),
                world=fmt(grouped.get("world_size")),
                memory=fmt(grouped.get("memory_native_worker_count")),
                passes=fmt(grouped.get("pass_count")),
                storage=fmt(grouped.get("code_cache_storage_reduction_pct")),
                mean_speedup=fmt(grouped.get("mean_grouped_speedup")),
                best_speedup=fmt(grouped.get("best_grouped_speedup")),
                aicore_samples=fmt(monitor.get("samples_with_8_nonzero_aicore")),
                released=fmt(grouped.get("released_packed_code_bytes_total")),
                payload=fmt(grouped.get("live_compressed_payload_storage_bytes_total")),
            )
        )
    return metrics


def inference_acceleration_summary_evidence(summary: dict[str, Any] | None) -> list[str]:
    if not summary:
        return []
    payload = summary.get("summary", {}) if isinstance(summary.get("summary"), dict) else {}
    grouped = payload.get("grouped_replay", {}) if isinstance(payload.get("grouped_replay"), dict) else {}
    monitor = grouped.get("monitor", {}) if isinstance(grouped.get("monitor"), dict) else {}
    evidence: list[str] = []
    for item in (grouped.get("artifact"), monitor.get("artifact")):
        if item:
            evidence.append(str(item))
    return evidence


def compressed_native_sweep_metrics(summary: dict[str, Any] | None) -> list[str]:
    if not summary:
        return ["compressed-native sweep missing"]
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
        "compressed-native sweep {status}: {readout}".format(
            status=summary.get("status", "missing"),
            readout=summary.get("readout", "missing"),
        ),
        "compressed-native sweep best {method} {speedup}x baseline at max_new_tokens={tokens}, cache={cache}".format(
            method=best.get("method_label") or best.get("method", "missing"),
            speedup=fmt(best.get("speedup")),
            tokens=best.get("max_new_tokens", "missing"),
            cache=best.get("inference_cache_enabled", "missing"),
        ),
        "compressed-native sweep cache qpruner delta {delta}x at max_new_tokens={tokens}".format(
            delta=fmt(cache.get("qpruner_speedup_delta")),
            tokens=cache.get("max_new_tokens", "missing"),
        ),
        "compressed-native sweep long decode qpruner delta {delta}x from {short} to {long} tokens".format(
            delta=fmt(long_decode.get("qpruner_speedup_delta")),
            short=long_decode.get("from_max_new_tokens", "missing"),
            long=long_decode.get("to_max_new_tokens", "missing"),
        ),
        "compressed-native sweep memory-preserving QPruner {label}, strategy {strategy}, cache mode {mode}, speedup {speedup}x, code-cache storage reduction {storage}%".format(
            label=memory_best.get("label", "missing"),
            strategy=memory_best.get("qpruner_runtime_strategy", "missing"),
            mode=memory_best.get("qpruner_cache_mode", "missing"),
            speedup=fmt(memory_best.get("qpruner_vs_baseline_speedup")),
            storage=fmt(
                memory_best.get("qpruner_code_cache_storage_reduction_pct")
                if memory_best.get("qpruner_code_cache_storage_reduction_pct") is not None
                else memory_best.get("qpruner_storage_reduction_pct")
            ),
        ),
        "compressed-native sweep runtime diagnosis QPruner scaling gap {gap}x, QPruner cache peak +{mem} MB, next {action}".format(
            gap=fmt(diagnosis.get("qpruner_scaling_gap_vs_baseline")),
            mem=fmt(diagnosis.get("qpruner_cache_peak_mem_delta_mb")),
            action=diagnosis.get("next_action", "missing"),
        ),
        "compressed-native sweep QPruner storage reduction {storage}%".format(
            storage=fmt(memory.get("best_qpruner_storage_reduction_pct")),
        ),
    ]


def qpruner_packed_decode_metrics(report: dict[str, Any] | None) -> list[str]:
    if not report:
        return ["QPruner packed decode missing"]
    uncached = report.get("uncached", {}) if isinstance(report.get("uncached"), dict) else {}
    code_cached = report.get("code_cached", {}) if isinstance(report.get("code_cached"), dict) else {}
    scaled_code = report.get("scaled_code_matmul", {}) if isinstance(report.get("scaled_code_matmul"), dict) else {}
    cached = report.get("cached", {}) if isinstance(report.get("cached"), dict) else {}
    dense = report.get("dense", {}) if isinstance(report.get("dense"), dict) else {}
    return [
        "QPruner packed decode {status} on {device}: runtime {runtime}, storage {storage}%".format(
            status=report.get("status", "missing"),
            device=report.get("device", "missing"),
            runtime=report.get("runtime_storage_format", "missing"),
            storage=fmt(report.get("storage_reduction_pct")),
        ),
        "QPruner packed decode latency uncached {uncached} ms, cache {cache} ms, dense {dense} ms; cache speedup {speedup}x".format(
            uncached=fmt(uncached.get("latency_ms")),
            cache=fmt(cached.get("latency_ms")),
            dense=fmt(dense.get("latency_ms")),
            speedup=fmt(cached.get("speedup_vs_uncached")),
        ),
        "QPruner int8 code-cache storage {storage}%; code-cache {latency} ms; code-cache speedup {speedup}x".format(
            storage=fmt(report.get("code_cache_storage_reduction_pct")),
            latency=fmt(code_cached.get("latency_ms")),
            speedup=fmt(code_cached.get("speedup_vs_uncached")),
        ),
        "QPruner scaled-code matmul {latency} ms; scaled-code speedup {speedup}x; scaled-code peak {peak} MB".format(
            latency=fmt(scaled_code.get("latency_ms")),
            speedup=fmt(scaled_code.get("speedup_vs_uncached")),
            peak=fmt(scaled_code.get("peak_mem_mb")),
        ),
        "QPruner packed decode payload {payload} bytes vs dense {dense} bytes; next {action}".format(
            payload=fmt(report.get("quantized_payload_bytes")),
            dense=fmt(report.get("dense_payload_bytes")),
            action=report.get("next_action", "missing"),
        ),
    ]


def multicard_compression_sync_metrics(summary: dict[str, Any] | None) -> list[str]:
    if not summary:
        return ["multi-card compression sync missing"]
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
    return [
        "multi-card compression sync {status}: {readout}".format(
            status=summary.get("status", "missing"),
            readout=summary.get("readout", "missing"),
        ),
        "multi-card compression sync QPruner {qpruner} tokens/s total, {speedup}x baseline, all-reduce {consistent}".format(
            qpruner=fmt(generate.get("qpruner_tokens_per_s_total")),
            speedup=fmt(generate.get("qpruner_speedup_vs_baseline")),
            consistent=generate.get("distributed_reduce_consistent", "missing"),
        ),
        "multi-card compression sync memory QPruner storage reduction {storage}%, cache peak +{mem} MB".format(
            storage=fmt(memory.get("qpruner_storage_reduction_pct")),
            mem=fmt(memory.get("qpruner_cache_peak_mem_delta_mb")),
        ),
        "multi-card compression sync window start {start}s, release lag {lag}s".format(
            start=fmt(parallel.get("start_window_s")),
            lag=fmt(parallel.get("release_lag_window_s")),
        ),
    ]


def objective_coverage_metrics(audit: dict[str, Any] | None) -> list[str]:
    if not audit:
        return ["objective coverage audit missing"]
    metrics = [f"objective coverage audit={audit.get('status', 'missing')}"]
    for item in audit.get("sections", []):
        if not isinstance(item, dict):
            continue
        metrics.append(
            "objective coverage {title}: {status}, {summary}; gap {gap}".format(
                title=item.get("title", "missing"),
                status=item.get("status", "missing"),
                summary=item.get("evidence_summary", "missing"),
                gap=item.get("remaining_gap", "missing"),
            )
        )
    return metrics


def objective_section(audit: dict[str, Any] | None, section_id: str) -> dict[str, Any] | None:
    if not audit:
        return None
    for item in audit.get("sections", []):
        if isinstance(item, dict) and item.get("id") == section_id:
            return item
    return None


def objective_actionable_gap(audit: dict[str, Any] | None, section_id: str) -> str | None:
    item = objective_section(audit, section_id)
    if not item:
        return None
    status = item.get("status")
    if status not in {"ACTIONABLE", "PARTIAL_READY"}:
        return None
    gap = item.get("remaining_gap") or "see objective coverage audit"
    return f"objective audit {section_id.replace('_', ' ')} {status}: {gap}"


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


def first_diagnosis_title(report: dict[str, Any] | None) -> str:
    diagnoses = report.get("diagnoses", []) if report else []
    for item in diagnoses:
        if isinstance(item, dict) and item.get("title"):
            return str(item["title"])
    return "none" if report else "missing"


def inference_bottleneck_metrics(report: dict[str, Any] | None) -> list[str]:
    diagnoses = report.get("diagnoses", []) if report else []
    metrics: list[str] = []
    current = report.get("current_vllm_rerun", {}) if report else {}
    if isinstance(current, dict) and current:
        cards = current.get("method_cards")
        card_text = ",".join(str(card) for card in cards) if isinstance(cards, list) else "missing"
        artifact = current.get("artifact")
        artifact_text = f"artifacts/{artifact}" if artifact else "missing"
        speedups = current.get("speedups", {}) if isinstance(current.get("speedups"), dict) else {}
        metrics.append(
            (
                "current vLLM HBM rerun {status}: parallel={parallel}, cards {cards}, "
                "max_new_tokens={tokens}, best {best} {best_tps} tokens/s, "
                "qpruner_vs_baseline {q_speedup}x, artifact {artifact}"
            ).format(
                status=current.get("status", "missing"),
                parallel=current.get("parallel_methods", "missing"),
                cards=card_text,
                tokens=current.get("max_new_tokens", "missing"),
                best=current.get("best_method", "missing"),
                best_tps=fmt(current.get("best_tokens_per_s")),
                q_speedup=fmt(speedups.get("qpruner_vs_baseline")),
                artifact=artifact_text,
            )
        )
    for index, item in enumerate(diagnoses, start=1):
        if not isinstance(item, dict):
            continue
        title = item.get("title")
        evidence = item.get("evidence")
        if title:
            metrics.append(f"bottleneck diagnosis {index}: {title}")
        if evidence:
            metrics.append(f"bottleneck diagnosis {index} evidence: {evidence}")
    return metrics


def resource_blocker_metric(report: dict[str, Any] | None) -> str:
    blocker = report.get("resource_blocker", {}) if report else {}
    if not isinstance(blocker, dict) or blocker.get("status") != "BLOCKED":
        return "resource blocker none"
    return (
        "resource blocker {status}: occupied_cards={occupied}/{total}, container={container}, "
        "model={model}, tensor_parallel_size={tp}, policy={policy}"
    ).format(
        status=blocker.get("status", "missing"),
        occupied=fmt(blocker.get("occupied_cards")),
        total=fmt(blocker.get("total_cards")),
        container=blocker.get("container_name", "missing"),
        model=blocker.get("service_model", "missing"),
        tp=fmt(blocker.get("tensor_parallel_size")),
        policy=blocker.get("policy", "missing"),
    )


def missing_artifacts(demo_root: Path, names: Sequence[str]) -> list[str]:
    artifacts = demo_root / "artifacts"
    return [artifact_name(name) for name in names if not (artifacts / name).exists()]


def section(
    *,
    section_id: str,
    title: str,
    status: str,
    summary: str,
    evidence: list[str],
    metrics: list[str],
    gaps: list[str],
    next_action: str,
) -> dict[str, Any]:
    return {
        "id": section_id,
        "title": title,
        "status": status,
        "summary": summary,
        "evidence": evidence,
        "metrics": metrics,
        "gaps": gaps,
        "next_action": next_action,
    }


def build_report(demo_root: Path) -> dict[str, Any]:
    artifacts = demo_root / "artifacts"
    sync = read_json(artifacts / "multicard_sync_summary.json")
    lora_sync = read_json(artifacts / "rankadaptor_lora_sync_summary.json")
    compatibility = read_json(artifacts / "compatibility_issues.json")
    qwen3_snapshot = read_json(artifacts / "model_snapshot_qwen3_06b.json")
    qwen3_inference = read_json(artifacts / "ascend_inference_qwen3_06b_torch_npu.json")
    fallback = read_json(artifacts / "torch_serving_fallback_tiny_qwen3_serving_fallback_npu.json")
    compressed_native = read_json(artifacts / "compressed_native_torch_serving_tiny_qwen3_native_serving_npu.json")
    inference_acceleration_summary = read_json(artifacts / "inference_acceleration_summary.json")
    compressed_native_sweep_summary = read_json(artifacts / "compressed_native_sweep_summary.json")
    qpruner_packed_decode = read_json(artifacts / QPRUNER_PACKED_DECODE_ARTIFACT)
    multicard_compression_sync_summary = read_json(artifacts / "multicard_compression_sync_summary.json")
    parallel = read_json(artifacts / "multicard_parallel_suite_demo_parallel_suite_npu.json")
    qwen_lora = best_multicard_qwen_lora_finetune(artifacts)
    qwen_lora_artifact = artifact_from_payload(
        qwen_lora,
        "multicard_qwen_lora_finetune_qwen3_06b_lora_sync_npu.json",
    )
    bslora = read_json(artifacts / "multicard_tiny_qwen_bslora_finetune_tiny_qwen3_bslora_sync_npu.json")
    qwen_bslora = read_json(artifacts / "tiny_qwen_bslora_finetune_qwen3_06b_bslora_npu.json")
    qwen_quality = read_json(artifacts / "qwen_compression_quality_qwen3_06b_quality_npu.json")
    qwen_generate = read_json(artifacts / "qwen_compression_generate_qwen3_06b_generate_npu.json")
    qwen_pattern_quality = read_json(
        artifacts / "multicard_qwen_compression_quality_qwen3_06b_quality_pattern_2card_npu.json"
    )
    qwen_pattern_generate = read_json(
        artifacts / "multicard_qwen_compression_generate_qwen3_06b_generate_pattern_2card_npu.json"
    )
    qwen_multicard_generate = best_multicard_qwen_compression_generate(artifacts)
    compression_memory = read_json(artifacts / "compression_memory_report.json")
    paper_baseline_coverage_audit = read_json(artifacts / "paper_baseline_coverage_audit.json")
    choice_accuracy = (
        read_json(artifacts / FULLTARGET_QPRUNER_CHOICE_ARTIFACT)
        or read_json(artifacts / MULTITASK_CHOICE_ACCURACY_ARTIFACT)
        or read_json(artifacts / CHOICE_ACCURACY_ARTIFACT)
    )
    vllm_probe = read_json(artifacts / "vllm_serving_probe_tiny_qwen3_serving_export_npu.json")
    vllm_selector_shim_probe = read_json(
        artifacts / "vllm_serving_probe_tiny_qwen3_serving_export_npu_selector_shim.json"
    )
    vllm_serving_benchmark = first_existing_json(
        artifacts,
        [
            "vllm_serving_benchmark_tiny_qwen3_serving_vllm_metadata_shim_npu.json",
            "vllm_serving_benchmark_tiny_qwen3_serving_vllm_selector_shim_npu.json",
        ],
    )
    vllm_parallel = best_vllm_parallel_suite(artifacts)
    inference_bottleneck = read_json(artifacts / "inference_bottleneck_report.json")
    objective_coverage_audit = read_json(artifacts / "objective_coverage_audit.json")
    qwen_grouped_mlp_sweep = best_qwen_grouped_mlp_sweep(artifacts)
    qwen_grouped_mlp_memory_tradeoff = best_qwen_grouped_mlp_memory_tradeoff(artifacts)
    qwen_native_8card_profile = best_qwen_native_8card_profile_summary(artifacts)

    fallback_exports = fallback.get("exports", {}) if fallback else {}
    fallback_summary = fallback.get("summary", {}) if fallback else {}
    fallback_baseline = fallback_exports.get("baseline", {}) if isinstance(fallback_exports.get("baseline"), dict) else {}
    fallback_cap = fallback_exports.get("cap", {}) if isinstance(fallback_exports.get("cap"), dict) else {}
    fallback_qpruner = fallback_exports.get("qpruner", {}) if isinstance(fallback_exports.get("qpruner"), dict) else {}
    parallel_counts = parallel.get("task_counts", {}) if parallel else {}
    qwen_lora_aggregate = qwen_lora.get("aggregate", {}) if qwen_lora else {}
    bslora_aggregate = bslora.get("aggregate", {}) if bslora else {}
    quality_cap = qwen_quality.get("cap", {}) if qwen_quality else {}
    quality_qpruner = qwen_quality.get("qpruner", {}) if qwen_quality else {}
    generate_cap = qwen_generate.get("cap", {}) if qwen_generate else {}
    generate_qpruner = qwen_generate.get("qpruner", {}) if qwen_generate else {}
    pattern_quality_aggregate = qwen_pattern_quality.get("aggregate", {}) if qwen_pattern_quality else {}
    pattern_generate_aggregate = qwen_pattern_generate.get("aggregate", {}) if qwen_pattern_generate else {}
    vllm_benchmark_exports = vllm_serving_benchmark.get("exports", {}) if vllm_serving_benchmark else {}
    vllm_benchmark_summary = vllm_serving_benchmark.get("summary", {}) if vllm_serving_benchmark else {}
    vllm_parallel_aggregate = vllm_parallel.get("aggregate", {}) if vllm_parallel else {}
    vllm_benchmark_baseline = (
        vllm_benchmark_exports.get("baseline", {})
        if isinstance(vllm_benchmark_exports.get("baseline"), dict)
        else {}
    )
    vllm_benchmark_cap = (
        vllm_benchmark_exports.get("cap", {}) if isinstance(vllm_benchmark_exports.get("cap"), dict) else {}
    )
    vllm_benchmark_qpruner = (
        vllm_benchmark_exports.get("qpruner", {})
        if isinstance(vllm_benchmark_exports.get("qpruner"), dict)
        else {}
    )

    baseline_missing = missing_artifacts(
        demo_root,
        [
            "multicard_sync_summary.json",
            "rankadaptor_lora_sync_summary.json",
            "compatibility_issues.json",
            "multicard_parallel_suite_demo_parallel_suite_npu.json",
        ],
    )
    baseline_ready = (
        status_is_pass(sync)
        and int(sync.get("world_size", 0)) >= 1
        and status_is_pass(lora_sync)
        and status_is_pass(parallel)
        and compatibility is not None
    )
    baseline_section = section(
        section_id="ascend_910b_baseline",
        title="1. Ascend 910B Baseline",
        status="READY" if baseline_ready else "INCOMPLETE",
        summary="CAP/QPruner/RankAdaptor workflows have NPU evidence"
        if baseline_ready
        else "Missing evidence for one or more baseline workflows",
        evidence=[
            artifact_name("multicard_sync_summary.json"),
            artifact_name("rankadaptor_lora_sync_summary.json"),
            artifact_name("multicard_parallel_suite_demo_parallel_suite_npu.json"),
            artifact_name("compatibility_issues.json"),
            report_name("compatibility-issues.md"),
        ],
        metrics=[
            "8-card HCCL sync {status}, world_size={world}, backend={backend}".format(
                status=(sync or {}).get("status", "missing"),
                world=(sync or {}).get("world_size", "missing"),
                backend=(sync or {}).get("backend", "missing"),
            ),
            "parallel suite {status}, workers {workers}, start window {window}s, tasks {tasks}".format(
                status=(parallel or {}).get("status", "missing"),
                workers=(parallel or {}).get("world_size", "missing"),
                window=fmt((parallel or {}).get("start_window_s")),
                tasks=", ".join(f"{key}={value}" for key, value in sorted(parallel_counts.items())) or "missing",
            ),
            "parallel suite sync gate {gate}, release lag window {release_window}s".format(
                gate=fmt((parallel or {}).get("sync_start_target_ts")),
                release_window=fmt((parallel or {}).get("release_lag_window_s")),
            ),
        ],
        gaps=[f"Missing evidence: {item}" for item in baseline_missing],
        next_action="Keep compatibility_issues.json current after each backend or workflow rerun.",
    )

    inference_missing = missing_artifacts(
        demo_root,
        [
            "model_snapshot_qwen3_06b.json",
            "ascend_inference_qwen3_06b_torch_npu.json",
            "torch_serving_fallback_tiny_qwen3_serving_fallback_npu.json",
            "compressed_native_torch_serving_tiny_qwen3_native_serving_npu.json",
            "vllm_serving_probe_tiny_qwen3_serving_export_npu.json",
        ],
    )
    inference_ready = status_is_pass(qwen3_inference) and status_is_pass(fallback) and status_is_pass(compressed_native)
    inference_gaps = [f"Missing evidence: {item}" for item in inference_missing]
    vllm_selector_shim_ready = status_is_pass(vllm_selector_shim_probe)
    vllm_benchmark_ready = status_is_pass(vllm_serving_benchmark)
    vllm_benchmark_failed = bool(vllm_serving_benchmark and vllm_serving_benchmark.get("status") == "FAIL")
    vllm_benchmark_failure = vllm_failure_summary(vllm_serving_benchmark)
    if vllm_benchmark_ready:
        vllm_metric = (
            "vLLM benchmark best {best} {tps} tokens/s, baseline {baseline}, CAP {cap}, "
            "QPruner {qpruner}, qpruner_vs_baseline={speedup}x"
        ).format(
            best=vllm_benchmark_summary.get("best_method", "missing"),
            tps=fmt(vllm_benchmark_summary.get("best_tokens_per_s")),
            baseline=fmt(vllm_benchmark_baseline.get("tokens_per_s")),
            cap=fmt(vllm_benchmark_cap.get("tokens_per_s")),
            qpruner=fmt(vllm_benchmark_qpruner.get("tokens_per_s")),
            speedup=fmt(vllm_benchmark_summary.get("qpruner_vs_baseline_speedup")),
        )
    elif vllm_benchmark_failed:
        inference_gaps.append(f"vLLM benchmark failed: {vllm_benchmark_failure}")
        vllm_metric = f"vLLM benchmark failed: {vllm_benchmark_failure}"
    elif vllm_selector_shim_ready:
        inference_gaps.append(
            "vLLM selector shim loads exports, but latency/throughput benchmark evidence is missing"
        )
        vllm_metric = "vLLM selector shim PASS for CAP/QPruner exports without system package changes"
    elif vllm_probe and vllm_probe.get("status") != "PASS":
        inference_gaps.append(
            f"vLLM blocked at {issue_classes(vllm_probe)}; torch_npu fallback remains runnable"
        )
        vllm_metric = f"vLLM default probe blocked at {issue_classes(vllm_probe)}"
    else:
        vllm_metric = f"vLLM default probe {(vllm_probe or {}).get('status', 'missing')}"
    vllm_parallel_metric = (
        "vLLM synchronized slice {status}, workers {workers}, start window {window}s, "
        "best_vllm_tokens_per_s={best}"
    ).format(
        status=(vllm_parallel or {}).get("status", "missing"),
        workers=(vllm_parallel or {}).get("world_size", "missing"),
        window=fmt((vllm_parallel or {}).get("start_window_s")),
        best=fmt(vllm_parallel_aggregate.get("best_vllm_tokens_per_s")),
    )
    qpruner_shape_sweep = selected_qpruner_shape_sweep(inference_bottleneck)
    qpruner_shape_sweep_artifact = qpruner_shape_sweep.get("artifact")
    qpruner_shape_sweep_report = qpruner_shape_sweep_report_name(qpruner_shape_sweep_artifact)
    qpruner_shape_sweep_evidence = []
    if qpruner_shape_sweep_artifact:
        qpruner_shape_sweep_evidence.append(artifact_name(qpruner_shape_sweep_artifact))
    if qpruner_shape_sweep_report:
        qpruner_shape_sweep_evidence.append(report_name(qpruner_shape_sweep_report))
    qwen_native_profile = selected_qwen_native_profile(inference_bottleneck)
    qwen_native_profile_artifact = qwen_native_profile.get("artifact")
    qwen_native_profile_report = qwen_native_profile_report_name(qwen_native_profile_artifact)
    qwen_native_profile_evidence = []
    if qwen_native_profile_artifact:
        qwen_native_profile_evidence.append(artifact_name(qwen_native_profile_artifact))
    if qwen_native_profile_report:
        qwen_native_profile_evidence.append(report_name(qwen_native_profile_report))
    qwen_speed_first_profile = selected_qwen_speed_first_profile(inference_bottleneck)
    qwen_speed_first_artifact = qwen_speed_first_profile.get("artifact")
    qwen_speed_first_report = qwen_native_profile_report_name(qwen_speed_first_artifact)
    qwen_speed_first_evidence = []
    if qwen_speed_first_artifact:
        qwen_speed_first_evidence.append(artifact_name(qwen_speed_first_artifact))
    if qwen_speed_first_report:
        qwen_speed_first_evidence.append(report_name(qwen_speed_first_report))
    qwen_speed_first_metric = qwen_speed_first_readout(inference_bottleneck)
    qwen_native_profile_metric = qwen_native_profile_readout(inference_bottleneck)
    qwen_native_bits_readout = qwen_native_bits_tradeoff_readout(inference_bottleneck)
    qwen_native_bits_evidence = qwen_native_bits_tradeoff_evidence(inference_bottleneck)
    qwen_grouped_mlp_readout = qwen_grouped_mlp_sweep_readout(qwen_grouped_mlp_sweep)
    qwen_grouped_mlp_evidence = qwen_grouped_mlp_sweep_evidence(qwen_grouped_mlp_sweep)
    qwen_grouped_mlp_memory_readout = qwen_grouped_mlp_memory_tradeoff_readout(
        qwen_grouped_mlp_memory_tradeoff
    )
    qwen_grouped_mlp_memory_evidence = qwen_grouped_mlp_sweep_evidence(qwen_grouped_mlp_memory_tradeoff)
    qwen_native_8card_profile_readout_text = qwen_native_8card_profile_readout(qwen_native_8card_profile)
    qwen_native_8card_profile_evidence_paths = qwen_native_8card_profile_evidence(qwen_native_8card_profile)
    inference_objective_gap = objective_actionable_gap(objective_coverage_audit, "inference_acceleration")
    if inference_objective_gap:
        inference_gaps.append(inference_objective_gap)
    if inference_ready and inference_objective_gap:
        inference_status = "READY_WITH_ACTIONABLE_GAP"
    elif inference_ready and inference_gaps:
        inference_status = "READY_WITH_BACKEND_GAP"
    elif inference_ready:
        inference_status = "READY"
    else:
        inference_status = "INCOMPLETE"
    inference_section = section(
        section_id="inference_acceleration",
        title="2. Inference Acceleration",
        status=inference_status,
        summary=(
            "torch_npu and vLLM metadata-shim baseline/CAP/QPruner benchmarks exist"
            if vllm_benchmark_ready
            else "torch_npu baseline/CAP/QPruner benchmark exists; vLLM selector shim is runnable"
            if vllm_selector_shim_ready
            else "torch_npu baseline/CAP/QPruner benchmark exists; vLLM boundary is isolated"
        )
        if inference_ready
        else "Missing torch_npu serving benchmark evidence",
        evidence=[
            artifact_name("model_snapshot_qwen3_06b.json"),
            artifact_name("ascend_inference_qwen3_06b_torch_npu.json"),
            artifact_name("torch_serving_fallback_tiny_qwen3_serving_fallback_npu.json"),
            artifact_name("compressed_native_torch_serving_tiny_qwen3_native_serving_npu.json"),
            artifact_name("vllm_serving_probe_tiny_qwen3_serving_export_npu.json"),
            artifact_name("vllm_serving_probe_tiny_qwen3_serving_export_npu_selector_shim.json"),
            artifact_name("vllm_serving_benchmark_tiny_qwen3_serving_vllm_metadata_shim_npu.json"),
            artifact_name(vllm_parallel_artifact_name(vllm_parallel)),
            artifact_name("inference_acceleration_summary.json"),
            *inference_acceleration_summary_evidence(inference_acceleration_summary),
            artifact_name("compressed_native_sweep_summary.json"),
            artifact_name(QPRUNER_PACKED_DECODE_ARTIFACT),
            artifact_name(QPRUNER_SHAPE_SWEEP_ARTIFACT),
            *qpruner_shape_sweep_evidence,
            *qwen_native_profile_evidence,
            *qwen_speed_first_evidence,
            *qwen_native_bits_evidence,
            *qwen_grouped_mlp_evidence,
            *qwen_grouped_mlp_memory_evidence,
            *qwen_native_8card_profile_evidence_paths,
            artifact_name(QPRUNER_DTYPE_SHAPE_SWEEP_ARTIFACT),
            artifact_name("inference_bottleneck_report.json"),
            report_name("torch-serving-fallback-tiny_qwen3_serving_fallback_npu.md"),
            report_name("compressed-native-torch-serving-tiny_qwen3_native_serving_npu.md"),
            report_name("inference-acceleration-summary.md"),
            report_name("inference-acceleration-summary.csv"),
            report_name("inference-acceleration-summary.svg"),
            report_name("compressed-native-cache-long-decode-sweep.md"),
            report_name("compressed-native-cache-long-decode-sweep.csv"),
            report_name("compressed-native-cache-long-decode-sweep.svg"),
            report_name(QPRUNER_PACKED_DECODE_REPORT),
            report_name(QPRUNER_SHAPE_SWEEP_REPORT),
            report_name(QPRUNER_DTYPE_SHAPE_SWEEP_REPORT),
            report_name("vllm-serving-probe-tiny_qwen3_serving_export_npu_selector_shim.md"),
            report_name("vllm-serving-benchmark-tiny_qwen3_serving_vllm_metadata_shim_npu.md"),
            report_name(vllm_parallel_report_name(vllm_parallel)),
            report_name("inference-bottleneck-report.md"),
        ],
        metrics=[
            "Qwen3-0.6B snapshot {status} via {provider}, {mb} MB".format(
                status=(qwen3_snapshot or {}).get("status", "missing"),
                provider=(qwen3_snapshot or {}).get("provider", "missing"),
                mb=fmt((qwen3_snapshot or {}).get("parameter_mb")),
            ),
            "torch_npu inference {status}, latency {latency} ms, {tps} tokens/s, peak {peak} MB".format(
                status=(qwen3_inference or {}).get("status", "missing"),
                latency=fmt((qwen3_inference or {}).get("latency_ms")),
                tps=fmt((qwen3_inference or {}).get("tokens_per_s")),
                peak=fmt((qwen3_inference or {}).get("peak_mem_mb")),
            ),
            "serving fallback baseline {baseline} tokens/s, CAP {cap} tokens/s, QPruner {qpruner} tokens/s, qpruner_vs_baseline={speedup}x".format(
                baseline=fmt(fallback_baseline.get("tokens_per_s")),
                cap=fmt(fallback_cap.get("tokens_per_s")),
                qpruner=fmt(fallback_qpruner.get("tokens_per_s")),
                speedup=fmt(fallback_summary.get("qpruner_vs_baseline_speedup")),
            ),
            engineering_baseline_note(),
            paper_baseline_note(),
            *compressed_native_metrics(compressed_native),
            *inference_acceleration_summary_metrics(inference_acceleration_summary),
            *compressed_native_sweep_metrics(compressed_native_sweep_summary),
            *qpruner_packed_decode_metrics(qpruner_packed_decode),
            *([] if not qwen_native_profile_metric else [qwen_native_profile_metric]),
            *([] if not qwen_native_bits_readout else [qwen_native_bits_readout]),
            qwen_grouped_mlp_readout,
            qwen_grouped_mlp_memory_readout,
            *([] if not qwen_native_8card_profile_readout_text else [qwen_native_8card_profile_readout_text]),
            vllm_metric,
            vllm_parallel_metric,
            "bottleneck diagnosis {status}: {diagnosis}".format(
                status=(inference_bottleneck or {}).get("status", "missing"),
                diagnosis=first_diagnosis_title(inference_bottleneck),
            ),
            *inference_bottleneck_metrics(inference_bottleneck),
            *( [qwen_speed_first_metric] if qwen_speed_first_metric else [] ),
            resource_blocker_metric(inference_bottleneck),
        ],
        gaps=inference_gaps,
        next_action=(
            "Scale benchmark prompts/tokens and start bottleneck work on the vLLM and torch_npu paths."
            if vllm_benchmark_ready
            else "Align vLLM-Ascend attention metadata handling and rerun the metadata-shim benchmark."
            if vllm_benchmark_failed
            else "Run vLLM latency/throughput benchmark with the selector shim on baseline/CAP/QPruner exports."
            if vllm_selector_shim_ready
            else "Scale benchmark prompts/tokens and retry vLLM-Ascend after attention selector API versions are aligned."
        ),
    )

    finetune_missing = missing_artifacts(
        demo_root,
        [
            "rankadaptor_lora_sync_summary.json",
            qwen_lora_artifact,
            "multicard_tiny_qwen_bslora_finetune_tiny_qwen3_bslora_sync_npu.json",
            "tiny_qwen_bslora_finetune_qwen3_06b_bslora_npu.json",
        ],
    )
    finetune_ready = (
        status_is_pass(lora_sync)
        and status_is_pass(qwen_lora)
        and status_is_pass(bslora)
        and status_is_pass(qwen_bslora)
    )
    finetune_section = section(
        section_id="finetuning_effect",
        title="3. Fine-Tuning Effect",
        status="READY" if finetune_ready else "INCOMPLETE",
        summary="RankAdaptor/LoRA/BSLoRA fine-tuning has synchronized Ascend evidence"
        if finetune_ready
        else "Missing one or more fine-tuning evidence artifacts",
        evidence=[
            artifact_name("rankadaptor_lora_sync_summary.json"),
            artifact_name(qwen_lora_artifact),
            artifact_name("multicard_tiny_qwen_bslora_finetune_tiny_qwen3_bslora_sync_npu.json"),
            artifact_name("tiny_qwen_bslora_finetune_qwen3_06b_bslora_npu.json"),
        ],
        metrics=[
            "RankAdaptor sync {status}, world_size={world}, trainable adapters={params}".format(
                status=(lora_sync or {}).get("status", "missing"),
                world=(lora_sync or {}).get("world_size", "missing"),
                params=(lora_sync or {}).get("trainable_adapter_params", "missing"),
            ),
            "Qwen3 LoRA validation delta {delta}, adapter sync {sync}, pass_count {passes}".format(
                delta=fmt(qwen_lora_aggregate.get("validation_loss_delta_avg")),
                sync=qwen_lora_aggregate.get("adapter_sync_consistent", "missing"),
                passes=qwen_lora_aggregate.get("pass_count", "missing"),
            ),
            "BSLoRA loss delta {delta}, adapter sync {sync}".format(
                delta=fmt(bslora_aggregate.get("loss_delta_avg")),
                sync=bslora_aggregate.get("adapter_sync_consistent", "missing"),
            ),
            "Qwen3 BSLoRA loss delta {delta}, target modules {modules}, tokens trained {tokens}".format(
                delta=fmt((qwen_bslora or {}).get("loss_delta")),
                modules=(qwen_bslora or {}).get("target_module_count", "missing"),
                tokens=(qwen_bslora or {}).get("tokens_trained", "missing"),
            ),
            "Qwen3 BSLoRA sharing {trainable} vs {unshared} adapter params".format(
                trainable=(qwen_bslora or {}).get("trainable_adapter_params", "missing"),
                unshared=(qwen_bslora or {}).get("unshared_adapter_params", "missing"),
            ),
        ],
        gaps=[f"Missing evidence: {item}" for item in finetune_missing],
        next_action="Add a longer validation set and generation-quality rubric once runtime budget is available.",
    )

    compression_missing = missing_artifacts(
        demo_root,
        [
            "qwen_compression_quality_qwen3_06b_quality_npu.json",
            "qwen_compression_generate_qwen3_06b_generate_npu.json",
            "multicard_qwen_compression_quality_qwen3_06b_quality_pattern_2card_npu.json",
            "multicard_qwen_compression_generate_qwen3_06b_generate_pattern_2card_npu.json",
            "multicard_compression_sync_summary.json",
            "qwen_qpruner_scale_quality_summary.json",
            CHOICE_ACCURACY_ARTIFACT,
            MULTITASK_CHOICE_ACCURACY_ARTIFACT,
            FULLTARGET_QPRUNER_CHOICE_ARTIFACT,
        ],
    )
    compression_ready = (
        status_is_pass(qwen_quality)
        and status_is_pass(qwen_generate)
        and status_is_pass(qwen_pattern_quality)
        and status_is_pass(qwen_pattern_generate)
        and status_is_pass(multicard_compression_sync_summary)
        and status_is_pass(choice_accuracy)
    )
    compression_section = section(
        section_id="model_compression",
        title="4. Model Compression",
        status="READY" if compression_ready else "INCOMPLETE",
        summary="CAP/QPruner have reproducible Ascend compression quality and generate benchmarks"
        if compression_ready
        else "Missing compression quality or generate benchmark evidence",
        evidence=[
            artifact_name("qwen_compression_quality_qwen3_06b_quality_npu.json"),
            artifact_name("qwen_compression_generate_qwen3_06b_generate_npu.json"),
            artifact_name("multicard_qwen_compression_quality_qwen3_06b_quality_pattern_2card_npu.json"),
            artifact_name("multicard_qwen_compression_generate_qwen3_06b_generate_pattern_2card_npu.json"),
            artifact_name(
                artifact_from_payload(
                    qwen_multicard_generate,
                    "multicard_qwen_compression_generate_qwen3_06b_compression_generate_npu.json",
                )
            ),
            artifact_name("multicard_compression_sync_summary.json"),
            artifact_name("compression_memory_report.json"),
            artifact_name("paper_baseline_coverage_audit.json"),
            artifact_name("qwen_llm_pruner_baseline_qwen3_06b_llm_pruner_npu.json"),
            artifact_name("qwen_qpruner_scale_quality_summary.json"),
            artifact_name(CHOICE_ACCURACY_ARTIFACT),
            artifact_name(MULTITASK_CHOICE_ACCURACY_ARTIFACT),
            artifact_name(FULLTARGET_QPRUNER_CHOICE_ARTIFACT),
            report_name("qwen-compression-quality-qwen3_06b_quality_npu.md"),
            report_name("qwen-compression-generate-qwen3_06b_generate_npu.md"),
            report_name("qwen-llm-pruner-baseline-qwen3_06b_llm_pruner_npu.md"),
            report_name("multicard-qwen-compression-quality-qwen3_06b_quality_pattern_2card_npu.md"),
            report_name("multicard-qwen-compression-generate-qwen3_06b_generate_pattern_2card_npu.md"),
            report_name(
                report_from_payload(
                    qwen_multicard_generate,
                    multicard_qwen_compression_generate_report_name(
                        "multicard_qwen_compression_generate_qwen3_06b_compression_generate_npu.json"
                    )
                    or "multicard-qwen-compression-generate-qwen3_06b_compression_generate_npu.md",
                )
            ),
            report_name("qwen-compression-quality-memory-sweep.md"),
            report_name("qwen-qpruner-quality-memory-sweep.md"),
            report_name("qwen-qpruner-scale-quality-summary.md"),
            report_name("qwen-qpruner-scale-quality-summary.csv"),
            report_name(CHOICE_ACCURACY_REPORT),
            report_name(CHOICE_ACCURACY_CSV),
            report_name(MULTITASK_CHOICE_ACCURACY_REPORT),
            report_name(MULTITASK_CHOICE_ACCURACY_CSV),
            report_name(FULLTARGET_QPRUNER_CHOICE_REPORT),
            report_name(FULLTARGET_QPRUNER_CHOICE_CSV),
            report_name("qwen-qpruner-quality-memory-frontier.svg"),
            report_name("qwen-qpruner-quality-memory-frontier.md"),
            report_name("multicard-compression-sync-summary.md"),
            report_name("multicard-compression-sync-summary.csv"),
            report_name("multicard-compression-sync-summary.svg"),
            artifact_name("objective_coverage_audit.json"),
            report_name("objective-coverage-audit.md"),
            report_name("compression-memory-report.md"),
            report_name("paper-baseline-coverage-audit.md"),
        ],
        metrics=[
            "target layers {limit}/{total}, baseline loss {loss}, CAP loss delta {cap_delta}, QPruner loss delta {q_delta}".format(
                limit=(qwen_quality or {}).get("target_layer_limit", "missing"),
                total=(qwen_quality or {}).get("targeted_layers_total", "missing"),
                loss=fmt((qwen_quality or {}).get("baseline", {}).get("loss") if qwen_quality else None),
                cap_delta=fmt(quality_cap.get("loss_delta")),
                q_delta=fmt(quality_qpruner.get("loss_delta")),
            ),
            "CAP compression ratio {cap_ratio}, CAP generate speedup {cap_speedup}x".format(
                cap_ratio=fmt(quality_cap.get("targeted_compression_ratio")),
                cap_speedup=fmt(generate_cap.get("latency_speedup")),
            ),
            "QPruner average bits {bits}, QPruner generate speedup {speedup}x".format(
                bits=fmt(quality_qpruner.get("average_bits")),
                speedup=fmt(generate_qpruner.get("latency_speedup")),
            ),
            "patterned Qwen3 quality sync {status}, world_size={world}, pattern {pattern}, target layers {limit}/{total}, CAP loss delta {cap_delta}, WANDA loss delta {wanda_delta}, SparseGPT loss delta {sparsegpt_delta}, QPruner loss delta {q_delta}".format(
                status=(qwen_pattern_quality or {}).get("status", "missing"),
                world=(qwen_pattern_quality or {}).get("world_size", "missing"),
                pattern=(qwen_pattern_quality or {}).get("target_layer_pattern", "missing"),
                limit=(qwen_pattern_quality or {}).get("target_layer_limit", "missing"),
                total=(qwen_pattern_quality or {}).get("targeted_layers_total", "missing"),
                cap_delta=fmt(pattern_quality_aggregate.get("cap_loss_delta_avg")),
                wanda_delta=fmt(pattern_quality_aggregate.get("wanda_loss_delta_avg")),
                sparsegpt_delta=fmt(pattern_quality_aggregate.get("sparsegpt_loss_delta_avg")),
                q_delta=fmt(pattern_quality_aggregate.get("qpruner_loss_delta_avg")),
            ),
            "patterned Qwen3 generate sync {status}, CAP {cap} tokens/s, QPruner {qpruner} tokens/s, all-reduce {consistent}".format(
                status=(qwen_pattern_generate or {}).get("status", "missing"),
                cap=fmt(pattern_generate_aggregate.get("cap_tokens_per_s_total")),
                qpruner=fmt(pattern_generate_aggregate.get("qpruner_tokens_per_s_total")),
                consistent=pattern_generate_aggregate.get("distributed_reduce_consistent", "missing"),
            ),
            quality_memory_sweep_text(demo_root),
            qpruner_quality_memory_sweep_text(demo_root),
            qpruner_scale_quality_text(demo_root),
            choice_accuracy_text(choice_accuracy),
            *multicard_compression_sync_metrics(multicard_compression_sync_summary),
            *paper_baseline_audit_metrics(paper_baseline_coverage_audit),
            *compression_memory_metrics(
                compression_memory,
                include_legacy_baseline_evidence=paper_baseline_coverage_audit is None,
            ),
        ],
        gaps=[f"Missing evidence: {item}" for item in compression_missing],
        next_action="Increase target-layer coverage beyond the pilot limit and add broader external benchmark tasks.",
    )

    materials = [
        report_name("ascend-910b-demo-progress.md"),
        report_name("demo-storyboard.md"),
        report_name("demo-readiness-report.md"),
        report_name("objective-coverage-audit.md"),
        report_name("multicard-compression-sync-summary.md"),
        report_name("multicard-compression-sync-summary.csv"),
        report_name("multicard-compression-sync-summary.svg"),
        artifact_name("objective_coverage_audit.json"),
        "recordings/ascend-910b-demo-playback-latest.typescript",
    ]
    preexisting_materials = [item for item in materials if item != report_name("demo-readiness-report.md")]
    material_gaps = [f"Missing evidence: {item}" for item in preexisting_materials if not (demo_root / item).exists()]
    demo_section = section(
        section_id="demo_materials",
        title="5. Demo Materials",
        status="READY" if not material_gaps else "INCOMPLETE",
        summary="Recording materials are preserved under the demo root"
        if not material_gaps
        else "Some recording materials are not present yet",
        evidence=materials,
        metrics=[
            f"demo_root={demo_root}",
            "storyboard=reports/demo-storyboard.md",
            "terminal_recording=recordings/ascend-910b-demo-playback-latest.typescript",
            *objective_coverage_metrics(objective_coverage_audit),
        ],
        gaps=material_gaps,
        next_action="Regenerate storyboard, progress, compatibility, and playback after every benchmark rerun.",
    )

    sections = [baseline_section, inference_section, finetune_section, compression_section, demo_section]
    incomplete = [entry for entry in sections if entry["status"] == "INCOMPLETE"]
    partial_gap = [
        entry
        for entry in sections
        if entry["status"] in {"READY_WITH_BACKEND_GAP", "READY_WITH_ACTIONABLE_GAP"}
    ]
    overall_status = "INCOMPLETE" if incomplete else ("PARTIAL_READY" if partial_gap else "READY")
    return {
        "status": overall_status,
        "demo_root": str(demo_root),
        "sections": sections,
        "platform": platform.platform(),
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Ascend 910B Demo Readiness Report",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Demo root: `{report.get('demo_root')}`",
        "",
    ]
    for entry in report.get("sections", []):
        lines.extend(
            [
                f"## {entry.get('title')}",
                "",
                f"- Status: `{entry.get('status')}`",
                f"- Summary: {entry.get('summary')}",
                "",
                "### Evidence",
                "",
            ]
        )
        for evidence in entry.get("evidence", []):
            lines.append(f"- `{evidence}`")
        lines.extend(["", "### Metrics", ""])
        for metric in entry.get("metrics", []):
            lines.append(f"- {metric}")
        gaps = entry.get("gaps", [])
        if gaps:
            lines.extend(["", "### Gaps", ""])
            for gap in gaps:
                lines.append(f"- {gap}")
        lines.extend(["", f"Next action: {entry.get('next_action')}", ""])
    return "\n".join(lines).rstrip() + "\n"


def write_demo_readiness_report(demo_root: Path) -> Path:
    artifacts = demo_root / "artifacts"
    reports = demo_root / "reports"
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    report = build_report(demo_root)
    json_path = artifacts / "demo_readiness_report.json"
    md_path = reports / "demo-readiness-report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    md_path.write_text(markdown(report))
    return md_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write Ascend 910B demo readiness report")
    parser.add_argument("--demo-root", default="/mnt/nvme/622/tidal-demo")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = write_demo_readiness_report(Path(args.demo_root))
    print(f"DEMO_READINESS_REPORT {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
