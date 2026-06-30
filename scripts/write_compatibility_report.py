#!/usr/bin/env python
"""Write a consolidated Ascend 910B compatibility issue report."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any, Sequence

from tidal.reports.multicard_selection import best_multicard_qwen_compression_generate


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


def status_of(payload: dict[str, Any] | None) -> str:
    return str(payload.get("status")) if payload else "MISSING_EVIDENCE"


def package_status(probe: dict[str, Any] | None, name: str) -> str:
    if not probe:
        return "MISSING_EVIDENCE"
    packages = probe.get("packages", {})
    return str(packages.get(name, {}).get("status", "MISSING"))


def workflow_smoke_section(parallel_smokes: dict[str, Any] | None, workflow: str) -> dict[str, Any]:
    if not parallel_smokes:
        return {}
    for run in parallel_smokes.get("runs", []):
        if not isinstance(run, dict):
            continue
        workflows = run.get("workflows", [])
        if run.get("requested_workflow") == workflow or workflow in workflows:
            section = run.get(workflow, {})
            return section if isinstance(section, dict) else {}
    return {}


def item(item_id: str, severity: str, title: str, evidence: str, action: str) -> dict[str, str]:
    return {
        "id": item_id,
        "severity": severity,
        "title": title,
        "evidence": evidence,
        "action": action,
    }


def build_report(demo_root: Path) -> dict[str, Any]:
    artifacts = demo_root / "artifacts"
    inference_probe = read_json(artifacts / "ascend_inference_probe.json")
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
    parallel_smokes = read_json(artifacts / "parallel_ascend_smokes.json")
    serving_export = read_json(artifacts / "tiny_qwen_serving_export_tiny_qwen3_serving_export_npu.json")
    torch_serving_fallback = read_json(artifacts / "torch_serving_fallback_tiny_qwen3_serving_fallback_npu.json")
    multicard_parallel_suite = read_json(artifacts / "multicard_parallel_suite_demo_parallel_suite_npu.json")
    sync = read_json(artifacts / "multicard_sync_summary.json")
    lora_sync = read_json(artifacts / "rankadaptor_lora_sync_summary.json")
    multicard_generate = read_json(artifacts / "multicard_tiny_qwen_generate_tiny_qwen3_multicard_generate_npu.json")
    multicard_bslora = read_json(
        artifacts / "multicard_tiny_qwen_bslora_finetune_tiny_qwen3_bslora_sync_npu.json"
    )
    compression = read_json(artifacts / "tiny_qwen_compression_tiny_qwen3_compression_npu.json")
    qwen35_inference = read_json(artifacts / "ascend_inference_qwen35_08b_torch_npu.json")
    qwen35_snapshot = read_json(artifacts / "model_snapshot_qwen35_08b.json")
    qwen3_inference = read_json(artifacts / "ascend_inference_qwen3_06b_torch_npu.json")
    qwen3_snapshot = read_json(artifacts / "model_snapshot_qwen3_06b.json")
    multicard_qwen_inference = read_json(artifacts / "multicard_qwen_inference_qwen3_06b_npu.json")
    qwen_compression_quality = read_json(artifacts / "qwen_compression_quality_qwen3_06b_quality_npu.json")
    multicard_qwen_compression_quality = read_json(
        artifacts / "multicard_qwen_compression_quality_qwen3_06b_quality_sync_npu.json"
    )
    qwen_compression_generate = read_json(artifacts / "qwen_compression_generate_qwen3_06b_generate_npu.json")
    multicard_qwen_lora_finetune = read_json(
        artifacts / "multicard_qwen_lora_finetune_qwen3_06b_lora_sync_npu.json"
    )
    multicard_qwen_compression_generate = best_multicard_qwen_compression_generate(artifacts)

    issues: list[dict[str, str]] = []
    notes: list[dict[str, str]] = []
    ready: list[dict[str, str]] = []

    if qwen35_snapshot and qwen35_snapshot.get("status") in {"FOUND", "DOWNLOADED"}:
        ready.append(
            item(
                "qwen35_snapshot_ready",
                "info",
                "Qwen3.5-0.8B snapshot is available under the demo model root",
                (
                    f"path={qwen35_snapshot.get('model_path')}, "
                    f"parameter_mb={qwen35_snapshot.get('parameter_mb')}"
                ),
                "Use this as the modern-model track once the backend supports model_type qwen3_5.",
            )
        )

    if qwen35_inference and qwen35_inference.get("status") == "BACKEND_ERROR":
        error_text = str(qwen35_inference.get("error", ""))
        if "qwen3_5" in error_text and "does not recognize this architecture" in error_text:
            transformer_version = (
                (inference_probe or {})
                .get("packages", {})
                .get("transformers", {})
                .get("version", "unknown")
            )
            issues.append(
                item(
                    "qwen35_transformers_unsupported",
                    "medium",
                    "Qwen3.5-0.8B is downloaded but current Transformers does not recognize qwen3_5",
                    (
                        f"{error_text}\n"
                        f"model_path={qwen35_inference.get('model_path')}; "
                        f"transformers={transformer_version}"
                    ),
                    "Use a container with Transformers support for qwen3_5, or run the next benchmark on Qwen3-0.6B/Qwen2.5-0.5B while keeping Qwen3.5 as the latest-model readiness artifact.",
                )
            )

    if qwen3_inference and qwen3_inference.get("status") == "PASS":
        snapshot_status = qwen3_snapshot.get("status") if qwen3_snapshot else "missing"
        ready.append(
            item(
                "qwen3_torch_npu_ready",
                "info",
                "Qwen3-0.6B torch_npu inference is ready",
                (
                    f"snapshot_status={snapshot_status}, "
                    f"provider={(qwen3_snapshot or {}).get('provider', 'missing')}, "
                    f"device={qwen3_inference.get('device')}, "
                    f"latency_ms={qwen3_inference.get('latency_ms')}, "
                    f"tokens_per_s={qwen3_inference.get('tokens_per_s')}, "
                    f"peak_mem_mb={qwen3_inference.get('peak_mem_mb')}"
                ),
                "Use this as the modern runnable torch_npu model benchmark while Qwen3.5 waits for qwen3_5 backend support.",
            )
        )

    multicard_qwen_aggregate = multicard_qwen_inference.get("aggregate", {}) if multicard_qwen_inference else {}
    if (
        multicard_qwen_inference
        and multicard_qwen_inference.get("status") == "PASS"
        and multicard_qwen_aggregate.get("distributed_reduce_consistent")
    ):
        ready.append(
            item(
                "multicard_qwen_inference_ready",
                "info",
                "Qwen3-0.6B 8-card torch_npu inference is synchronized",
                (
                    f"world_size={multicard_qwen_inference.get('world_size')}, "
                    f"backend={multicard_qwen_inference.get('backend')}, "
                    f"tokens_per_s_total={multicard_qwen_aggregate.get('tokens_per_s_total')}, "
                    f"generated_tokens_total={multicard_qwen_aggregate.get('generated_tokens_total')}, "
                    f"peak_mem_mb_total={multicard_qwen_aggregate.get('peak_mem_mb_total')}, "
                    f"pass_count={multicard_qwen_aggregate.get('pass_count')}"
                ),
                "Use this as the modern multi-card torch_npu inference evidence.",
            )
        )

    if qwen_compression_generate and qwen_compression_generate.get("status") == "PASS":
        qwen_cap = qwen_compression_generate.get("cap", {})
        qwen_qpruner = qwen_compression_generate.get("qpruner", {})
        ready.append(
            item(
                "qwen_compression_generate_ready",
                "info",
                "Qwen3-0.6B CAP/QPruner torch_npu compressed generate pilot is ready",
                (
                    f"target_layers={qwen_compression_generate.get('target_layer_limit')}/"
                    f"{qwen_compression_generate.get('targeted_layers_total')}, "
                    f"cap_speedup={qwen_cap.get('latency_speedup')}, "
                    f"cap_ratio={qwen_cap.get('targeted_compression_ratio')}, "
                    f"qpruner_speedup={qwen_qpruner.get('latency_speedup')}, "
                    f"qpruner_bits={qwen_qpruner.get('average_bits')}"
                ),
                "Use this as the first modern-model CAP/QPruner compression benchmark and scale the target layer limit after timing is acceptable.",
            )
        )

    if qwen_compression_quality and qwen_compression_quality.get("status") == "PASS":
        qwen_cap = qwen_compression_quality.get("cap", {})
        qwen_qpruner = qwen_compression_quality.get("qpruner", {})
        ready.append(
            item(
                "qwen_compression_quality_ready",
                "info",
                "Qwen3-0.6B CAP/QPruner torch_npu compression quality pilot is ready",
                (
                    f"target_layers={qwen_compression_quality.get('target_layer_limit')}/"
                    f"{qwen_compression_quality.get('targeted_layers_total')}, "
                    f"baseline_loss={qwen_compression_quality.get('baseline', {}).get('loss')}, "
                    f"cap_loss_delta={qwen_cap.get('loss_delta')}, "
                    f"cap_ratio={qwen_cap.get('targeted_compression_ratio')}, "
                    f"qpruner_loss_delta={qwen_qpruner.get('loss_delta')}, "
                    f"qpruner_bits={qwen_qpruner.get('average_bits')}"
                ),
                "Use this as the modern-model compression quality evidence alongside the compressed-generate throughput pilot.",
            )
        )

    multicard_qwen_lora_aggregate = multicard_qwen_lora_finetune.get("aggregate", {}) if multicard_qwen_lora_finetune else {}
    if (
        multicard_qwen_lora_finetune
        and multicard_qwen_lora_finetune.get("status") == "PASS"
        and multicard_qwen_lora_aggregate.get("distributed_reduce_consistent")
        and multicard_qwen_lora_aggregate.get("adapter_sync_consistent")
    ):
        ready.append(
            item(
                "multicard_qwen_lora_finetune_ready",
                "info",
                "Qwen3-0.6B 8-card RankAdaptor LoRA fine-tuning is synchronized",
                (
                    f"world_size={multicard_qwen_lora_finetune.get('world_size')}, "
                    f"backend={multicard_qwen_lora_finetune.get('backend')}, "
                    f"target_modules={multicard_qwen_lora_finetune.get('target_module_count')}/"
                    f"{multicard_qwen_lora_finetune.get('targeted_modules_total')}, "
                    f"loss_delta_avg={multicard_qwen_lora_aggregate.get('loss_delta_avg')}, "
                    f"adapter_sync={multicard_qwen_lora_aggregate.get('adapter_sync_consistent')}, "
                    f"pass_count={multicard_qwen_lora_aggregate.get('pass_count')}"
                ),
                "Use this as the modern multi-card fine-tuning improvement evidence.",
            )
        )

    multicard_qwen_compression_aggregate = (
        multicard_qwen_compression_generate.get("aggregate", {}) if multicard_qwen_compression_generate else {}
    )
    if (
        multicard_qwen_compression_generate
        and multicard_qwen_compression_generate.get("status") == "PASS"
        and multicard_qwen_compression_aggregate.get("distributed_reduce_consistent")
    ):
        ready.append(
            item(
                "multicard_qwen_compression_generate_ready",
                "info",
                "Qwen3-0.6B 8-card CAP/QPruner compressed generate is synchronized",
                (
                    f"world_size={multicard_qwen_compression_generate.get('world_size')}, "
                    f"backend={multicard_qwen_compression_generate.get('backend')}, "
                    f"target_layers={multicard_qwen_compression_generate.get('target_layer_limit')}/"
                    f"{multicard_qwen_compression_generate.get('targeted_layers_total')}, "
                    f"baseline_tokens_per_s_total={multicard_qwen_compression_aggregate.get('baseline_tokens_per_s_total')}, "
                    f"cap_tokens_per_s_total={multicard_qwen_compression_aggregate.get('cap_tokens_per_s_total')}, "
                    f"qpruner_tokens_per_s_total={multicard_qwen_compression_aggregate.get('qpruner_tokens_per_s_total')}, "
                    f"peak_mem_mb_total={multicard_qwen_compression_aggregate.get('peak_mem_mb_total')}, "
                    f"pass_count={multicard_qwen_compression_aggregate.get('pass_count')}"
                ),
                "Use this as the modern multi-card CAP/QPruner generate evidence.",
            )
        )

    multicard_qwen_quality_aggregate = (
        multicard_qwen_compression_quality.get("aggregate", {}) if multicard_qwen_compression_quality else {}
    )
    if (
        multicard_qwen_compression_quality
        and multicard_qwen_compression_quality.get("status") == "PASS"
        and multicard_qwen_quality_aggregate.get("distributed_reduce_consistent")
    ):
        ready.append(
            item(
                "multicard_qwen_compression_quality_ready",
                "info",
                "Qwen3-0.6B 8-card CAP/QPruner compression quality is synchronized",
                (
                    f"world_size={multicard_qwen_compression_quality.get('world_size')}, "
                    f"backend={multicard_qwen_compression_quality.get('backend')}, "
                    f"target_layers={multicard_qwen_compression_quality.get('target_layer_limit')}/"
                    f"{multicard_qwen_compression_quality.get('targeted_layers_total')}, "
                    f"baseline_loss_avg={multicard_qwen_quality_aggregate.get('baseline_loss_avg')}, "
                    f"cap_loss_delta_avg={multicard_qwen_quality_aggregate.get('cap_loss_delta_avg')}, "
                    f"qpruner_loss_delta_avg={multicard_qwen_quality_aggregate.get('qpruner_loss_delta_avg')}, "
                    f"pass_count={multicard_qwen_quality_aggregate.get('pass_count')}"
                ),
                "Use this as the modern multi-card compression quality evidence.",
            )
        )

    vllm_selector_shim_ready = bool(vllm_selector_shim_probe and vllm_selector_shim_probe.get("status") == "PASS")
    if vllm_selector_shim_ready:
        ready.append(
            item(
                "vllm_selector_shim_ready",
                "info",
                "Project-local vLLM-Ascend selector shim can load CAP/QPruner serving exports",
                (
                    f"export_run_label={vllm_selector_shim_probe.get('export_run_label')}, "
                    f"preload_vllm_ascend_patch={vllm_selector_shim_probe.get('preload_vllm_ascend_patch')}, "
                    "preload_vllm_ascend_selector_shim="
                    f"{vllm_selector_shim_probe.get('preload_vllm_ascend_selector_shim')}, "
                    f"cap_vllm_load={vllm_selector_shim_probe.get('exports', {}).get('cap', {}).get('vllm_load')}, "
                    "qpruner_vllm_load="
                    f"{vllm_selector_shim_probe.get('exports', {}).get('qpruner', {}).get('vllm_load')}"
                ),
                "Use this artifact for the demo serving path, then add latency/throughput benchmarking on top of the same shim.",
            )
        )

    if vllm_serving_benchmark and vllm_serving_benchmark.get("status") == "PASS":
        summary = vllm_serving_benchmark.get("summary", {})
        exports = vllm_serving_benchmark.get("exports", {})
        baseline_export = exports.get("baseline", {}) if isinstance(exports.get("baseline"), dict) else {}
        cap_export = exports.get("cap", {}) if isinstance(exports.get("cap"), dict) else {}
        qpruner_export = exports.get("qpruner", {}) if isinstance(exports.get("qpruner"), dict) else {}
        ready.append(
            item(
                "vllm_serving_benchmark_ready",
                "info",
                "vLLM-Ascend metadata-shim serving benchmark can generate from baseline/CAP/QPruner exports",
                (
                    f"backend={vllm_serving_benchmark.get('backend')}, "
                    f"best_method={summary.get('best_method')}, "
                    f"best_tokens_per_s={summary.get('best_tokens_per_s')}, "
                    f"baseline_tokens_per_s={baseline_export.get('tokens_per_s')}, "
                    f"cap_tokens_per_s={cap_export.get('tokens_per_s')}, "
                    f"qpruner_tokens_per_s={qpruner_export.get('tokens_per_s')}, "
                    f"qpruner_vs_baseline_speedup={summary.get('qpruner_vs_baseline_speedup')}"
                ),
                "Use this as the vLLM serving latency/throughput/memory evidence; next scale prompts/tokens and tune bottlenecks.",
            )
        )

    if vllm_probe is None:
        issues.append(
            item(
                "vllm_serving_probe_missing",
                "medium",
                "vLLM serving probe evidence is missing",
                "No vllm_serving_probe_tiny_qwen3_serving_export_npu.json artifact was found.",
                "Run scripts/probe_vllm_serving_exports.py after exporting serving directories.",
            )
        )
    else:
        issue_text = "\n".join(str(issue) for issue in vllm_probe.get("issues", []))
        acl_diag = vllm_probe.get("acl_diagnostics") or {}
        acl_candidates = acl_diag.get("candidate_pythonpath_entries") or []
        issue_class_ids = {str(entry.get("id")) for entry in vllm_probe.get("issue_classes", []) if isinstance(entry, dict)}
        package_diag = vllm_probe.get("package_diagnostics") or {}
        api_diag = vllm_probe.get("api_diagnostics") or {}
        if "vllm_attention_selector_api_mismatch" in issue_class_ids:
            mismatch_item = item(
                "vllm_attention_selector_api_mismatch",
                "high",
                "vLLM-Ascend and vLLM attention selector APIs appear mismatched",
                (
                    f"{issue_text}\n"
                    f"vLLM={package_diag.get('vllm', {}).get('version')} "
                    f"({package_diag.get('vllm', {}).get('file')}); "
                    f"vLLM-Ascend={package_diag.get('vllm-ascend', {}).get('version')} "
                    f"({package_diag.get('vllm-ascend', {}).get('file')}); "
                    f"ACL import={acl_diag.get('import_acl')}, origin={acl_diag.get('acl_origin')}; "
                    f"api_status={api_diag.get('status', 'missing')}; "
                    "missing_patch_config_fields="
                    f"{api_diag.get('missing_patch_config_fields', [])}; "
                    "missing_patch_get_attn_backend_parameters="
                    f"{api_diag.get('missing_patch_get_attn_backend_parameters', [])}"
                ),
                "Use a vLLM/vLLM-Ascend pair with matching attention selector API, then rerun the serving load and throughput benchmark.",
            )
            if vllm_selector_shim_ready:
                notes.append(
                    item(
                        "vllm_attention_selector_api_mismatch_mitigated",
                        "note",
                        "Upstream vLLM-Ascend API mismatch is mitigated by the project-local selector shim",
                        mismatch_item["evidence"],
                        "Keep this note visible for upstream/container cleanup, but use the selector-shim PASS artifact for the demo.",
                    )
                )
            else:
                issues.append(mismatch_item)
        elif "vllm_device_memory_pressure" in issue_class_ids:
            issues.append(
                item(
                    "vllm_device_memory_pressure",
                    "high",
                    "vLLM-Ascend initialization is blocked by low free HBM on the selected device",
                    (
                        f"{issue_text}\n"
                        f"ACL import={acl_diag.get('import_acl')}, origin={acl_diag.get('acl_origin')}\n"
                        f"gpu_memory_utilization={vllm_probe.get('gpu_memory_utilization')}"
                    ),
                    "Free or select an idle Ascend card, then rerun the vLLM probe with the project path prepended to PYTHONPATH.",
                )
            )
        elif "No module named 'acl'" in issue_text:
            if acl_diag.get("import_acl") == "MISSING" and acl_candidates:
                issues.append(
                    item(
                        "vllm_acl_pythonpath_hidden",
                        "high",
                        "vLLM-Ascend cannot see CANN ACL because PYTHONPATH hides CANN site-packages",
                        (
                            f"{issue_text}\n"
                            f"PYTHONPATH={acl_diag.get('pythonpath', '')}\n"
                            f"CANN candidates={acl_candidates}"
                        ),
                        "Run the probe/serving process with the project path prepended to the existing container PYTHONPATH, not replacing it.",
                    )
                )
            else:
                issues.append(
                    item(
                        "vllm_acl_python_binding_missing",
                        "high",
                        "vLLM-Ascend cannot initialize because Python module acl is missing",
                        issue_text,
                        "Use or fix a vLLM-Ascend container that exposes CANN ACL Python bindings inside Python.",
                    )
                )
        elif vllm_probe.get("status") == "PASS":
            ready.append(
                item(
                    "vllm_serving_ready",
                    "info",
                    "vLLM can load CAP/QPruner serving exports",
                    "vLLM serving probe status is PASS.",
                    "Proceed to vLLM latency/throughput benchmark against baseline/CAP/QPruner.",
                )
            )
        else:
            issues.append(
                item(
                    "vllm_serving_probe_failed",
                    "high",
                    "vLLM serving probe failed",
                    issue_text or f"Probe status: {vllm_probe.get('status')}",
                    "Inspect the vLLM probe report and fix the backend/container issue before serving benchmark.",
                )
            )

    if serving_export is None:
        issues.append(
            item(
                "serving_export_missing",
                "medium",
                "CAP/QPruner serving export evidence is missing",
                "No tiny_qwen_serving_export_tiny_qwen3_serving_export_npu.json artifact was found.",
                "Run scripts/export_tiny_qwen_compressed_serving.py.",
            )
        )
    elif serving_export.get("status") == "PASS":
        ready.append(
            item(
                "hf_serving_export_ready",
                "info",
                "CAP/QPruner Hugging Face serving exports are loadable",
                "Serving export status is PASS and HF load checks passed.",
                "Use the export directories for vLLM-Ascend once the backend environment is fixed.",
            )
        )
        if "does not preserve compressed storage" in str(serving_export.get("storage_note", "")):
            notes.append(
                item(
                    "serving_dense_export_storage_tradeoff",
                    "note",
                    "Dense serving export is compatible but not storage-compressed",
                    str(serving_export.get("storage_note")),
                    "Use this export for backend compatibility; keep native CAP/QPruner modules for compressed-storage experiments.",
                )
            )

    if torch_serving_fallback and torch_serving_fallback.get("status") == "PASS":
        summary = torch_serving_fallback.get("summary", {})
        exports = torch_serving_fallback.get("exports", {})
        cap_export = exports.get("cap", {}) if isinstance(exports.get("cap"), dict) else {}
        qpruner_export = exports.get("qpruner", {}) if isinstance(exports.get("qpruner"), dict) else {}
        ready.append(
            item(
                "torch_serving_fallback_ready",
                "info",
                "torch_npu Transformers serving fallback can generate from CAP/QPruner exports",
                (
                    f"backend={torch_serving_fallback.get('backend')}, "
                    f"device={torch_serving_fallback.get('device')}, "
                    f"best_method={summary.get('best_method')}, "
                    f"best_tokens_per_s={summary.get('best_tokens_per_s')}, "
                    f"cap_tokens_per_s={cap_export.get('tokens_per_s')}, "
                    f"qpruner_tokens_per_s={qpruner_export.get('tokens_per_s')}"
                ),
                "Use this as the runnable serving benchmark while vLLM-Ascend is blocked by attention selector API mismatch.",
            )
        )

    parallel_aggregate = multicard_parallel_suite.get("aggregate", {}) if multicard_parallel_suite else {}
    if (
        multicard_parallel_suite
        and multicard_parallel_suite.get("status") == "PASS"
        and multicard_parallel_suite.get("launched_synchronously")
    ):
        ready.append(
            item(
                "multicard_parallel_suite_ready",
                "info",
                "Synchronized multi-card parallel suite is ready for video",
                (
                    f"world_size={multicard_parallel_suite.get('world_size')}, "
                    f"cards={multicard_parallel_suite.get('cards')}, "
                    f"sync_start_target_ts={multicard_parallel_suite.get('sync_start_target_ts')}, "
                    f"start_window_s={multicard_parallel_suite.get('start_window_s')}, "
                    f"release_lag_window_s={multicard_parallel_suite.get('release_lag_window_s')}, "
                    f"tasks={multicard_parallel_suite.get('task_counts')}, "
                    f"pass_count={parallel_aggregate.get('pass_count')}, "
                    f"best_tokens_per_s={parallel_aggregate.get('best_tokens_per_s')}"
                ),
                "Use this as the explicit evidence that CAP/QPruner/RankAdaptor/serving workers can occupy multiple cards concurrently.",
            )
        )
    elif multicard_parallel_suite:
        notes.append(
            item(
                "multicard_parallel_suite_not_synchronized",
                "note",
                "Multi-card parallel suite exists but is not synchronized cleanly",
                (
                    f"status={multicard_parallel_suite.get('status')}, "
                    f"launched_synchronously={multicard_parallel_suite.get('launched_synchronously')}, "
                    f"start_window_s={multicard_parallel_suite.get('start_window_s')}, "
                    f"release_lag_window_s={multicard_parallel_suite.get('release_lag_window_s')}"
                ),
                "Rerun scripts/multicard_parallel_suite.py with idle cards before using it as video evidence.",
            )
        )

    cap_smoke = workflow_smoke_section(parallel_smokes, "cap")
    qpruner_smoke = workflow_smoke_section(parallel_smokes, "qpruner")
    rankadaptor_smoke = workflow_smoke_section(parallel_smokes, "rankadaptor")
    smoke_sections = {
        "cap": cap_smoke,
        "qpruner": qpruner_smoke,
        "rankadaptor": rankadaptor_smoke,
    }
    if parallel_smokes and parallel_smokes.get("status") == "PASS" and all(
        section.get("status") == "PASS" for section in smoke_sections.values()
    ):
        ready.append(
            item(
                "tiny_workflow_smokes_ready",
                "info",
                "CAP/QPruner/RankAdaptor tiny workflow smokes run on Ascend NPU",
                (
                    f"cap={cap_smoke.get('status')}, packed={cap_smoke.get('packed_count')}, "
                    f"cap_device={cap_smoke.get('forward_output_device')}, "
                    f"qpruner={qpruner_smoke.get('status')}, quantized={qpruner_smoke.get('quantized_count')}, "
                    f"qpruner_device={qpruner_smoke.get('forward_output_device')}, "
                    f"rankadaptor={rankadaptor_smoke.get('status')}, profiles={rankadaptor_smoke.get('profile_count')}, "
                    f"rankadaptor_peft_applied={rankadaptor_smoke.get('peft_applied')}"
                ),
                "Use this as the baseline proof that the three core workflows can execute through torch_npu; keep the per-workflow logs for video material.",
            )
        )
    else:
        details = ", ".join(
            f"{name}={section.get('status', 'MISSING')}" for name, section in smoke_sections.items()
        )
        issues.append(
            item(
                "tiny_workflow_smokes_missing_or_failed",
                "medium",
                "CAP/QPruner/RankAdaptor tiny workflow smokes are not fully proven",
                f"Status: {status_of(parallel_smokes)}; {details}",
                "Run scripts/run_parallel_ascend_smokes.sh inside the target container and preserve the per-workflow JSON/log artifacts.",
            )
        )

    for package_name in ("torch", "torch_npu", "transformers"):
        if package_status(inference_probe, package_name) != "FOUND":
            issues.append(
                item(
                    f"package_{package_name}_not_ready",
                    "high",
                    f"{package_name} is not confirmed available",
                    f"Package status: {package_status(inference_probe, package_name)}",
                    "Run scripts/ascend_inference_probe.py inside the target container and fix missing backend packages.",
                )
            )

    if package_status(inference_probe, "peft") == "FOUND":
        ready.append(
            item(
                "peft_available",
                "info",
                "PEFT is available for RankAdaptor/LoRA workflows",
                "ascend_inference_probe package status for peft is FOUND.",
                "Continue RankAdaptor/LoRA and BSLoRA-style fine-tuning experiments.",
            )
        )
    else:
        notes.append(
            item(
                "peft_not_confirmed",
                "note",
                "PEFT availability is not confirmed",
                f"Package status: {package_status(inference_probe, 'peft')}",
                "Manual LoRA smoke remains valid; PEFT wrapping should be rechecked before full BSLoRA integration.",
            )
        )

    if sync and sync.get("status") == "PASS":
        ready.append(
            item(
                "hccl_sync_ready",
                "info",
                "8-card HCCL synchronization is ready",
                f"world_size={sync.get('world_size')}, backend={sync.get('backend')}",
                "Use this launch path for distributed training and benchmark stages.",
            )
        )
    else:
        issues.append(
            item(
                "hccl_sync_missing_or_failed",
                "high",
                "8-card HCCL sync is not proven",
                f"Status: {status_of(sync)}",
                "Run scripts/ascend_multicard_sync_smoke.py on the selected cards.",
            )
        )

    if lora_sync and lora_sync.get("status") == "PASS":
        ready.append(
            item(
                "rankadaptor_lora_sync_ready",
                "info",
                "RankAdaptor LoRA synchronized training smoke is ready",
                f"world_size={lora_sync.get('world_size')}, backend={lora_sync.get('backend')}",
                "Use this as the micro fine-tuning synchronization evidence.",
            )
        )
    else:
        issues.append(
            item(
                "rankadaptor_lora_sync_missing_or_failed",
                "medium",
                "RankAdaptor LoRA sync smoke is not proven",
                f"Status: {status_of(lora_sync)}",
                "Run scripts/rankadaptor_lora_sync_smoke.py.",
            )
        )

    bslora_aggregate = multicard_bslora.get("aggregate", {}) if multicard_bslora else {}
    if (
        multicard_bslora
        and multicard_bslora.get("status") == "PASS"
        and bslora_aggregate.get("distributed_reduce_consistent")
        and bslora_aggregate.get("adapter_sync_consistent")
    ):
        ready.append(
            item(
                "tiny_qwen_bslora_sync_ready",
                "info",
                "TinyQwen BSLoRA/shared-LoRA synchronized fine-tuning is ready",
                (
                    f"world_size={multicard_bslora.get('world_size')}, "
                    f"backend={multicard_bslora.get('backend')}, "
                    f"final_loss_avg={bslora_aggregate.get('final_loss_avg')}"
                ),
                "Use this as the real TinyQwen multi-card fine-tuning synchronization evidence.",
            )
        )
    else:
        notes.append(
            item(
                "tiny_qwen_bslora_sync_missing",
                "note",
                "TinyQwen BSLoRA synchronized fine-tuning evidence is not in the bundle",
                f"Status: {status_of(multicard_bslora)}",
                "Run scripts/multicard_tiny_qwen_bslora_finetune.py.",
            )
        )

    aggregate = multicard_generate.get("aggregate", {}) if multicard_generate else {}
    if multicard_generate and multicard_generate.get("status") == "PASS" and aggregate.get("distributed_reduce_consistent"):
        ready.append(
            item(
                "multicard_generate_ready",
                "info",
                "Multi-card TinyQwen compressed generate benchmark is synchronized",
                (
                    f"world_size={multicard_generate.get('world_size')}, "
                    f"baseline_total={aggregate.get('baseline_tokens_per_s_total')}, "
                    f"cap_total={aggregate.get('cap_tokens_per_s_total')}, "
                    f"qpruner_total={aggregate.get('qpruner_tokens_per_s_total')}"
                ),
                "Use this as the current torch_npu multi-card inference acceleration evidence.",
            )
        )
    else:
        issues.append(
            item(
                "multicard_generate_missing_or_inconsistent",
                "medium",
                "Multi-card compressed generate benchmark is not proven",
                f"Status: {status_of(multicard_generate)}",
                "Run scripts/multicard_tiny_qwen_generate_benchmark.py.",
            )
        )

    if compression and compression.get("status") == "PASS":
        ready.append(
            item(
                "compression_loss_benchmark_ready",
                "info",
                "CAP/QPruner loss and latency benchmark is available",
                (
                    f"baseline_loss={compression.get('baseline', {}).get('loss')}, "
                    f"cap_delta={compression.get('cap', {}).get('loss_delta')}, "
                    f"qpruner_delta={compression.get('qpruner', {}).get('loss_delta')}"
                ),
                "Use this as the current tiny-model PPL-like quality evidence.",
            )
        )
    else:
        notes.append(
            item(
                "compression_loss_benchmark_missing",
                "note",
                "CAP/QPruner loss benchmark is not in the evidence bundle",
                f"Status: {status_of(compression)}",
                "Run scripts/tiny_qwen_compression_benchmark.py to add loss deltas to the demo.",
            )
        )

    status = "ISSUES_FOUND" if issues else "READY_WITH_NOTES" if notes else "READY"
    return {
        "status": status,
        "platform": platform.platform(),
        "demo_root": str(demo_root),
        "issues": issues,
        "ready": ready,
        "notes": notes,
        "sources": {
            "inference_probe": status_of(inference_probe),
            "vllm_probe": status_of(vllm_probe),
            "vllm_selector_shim_probe": status_of(vllm_selector_shim_probe),
            "vllm_serving_benchmark": status_of(vllm_serving_benchmark),
            "parallel_ascend_smokes": status_of(parallel_smokes),
            "serving_export": status_of(serving_export),
            "torch_serving_fallback": status_of(torch_serving_fallback),
            "multicard_parallel_suite": status_of(multicard_parallel_suite),
            "multicard_sync": status_of(sync),
            "rankadaptor_lora_sync": status_of(lora_sync),
            "tiny_qwen_bslora_sync": status_of(multicard_bslora),
            "multicard_generate": status_of(multicard_generate),
            "compression_loss": status_of(compression),
            "qwen35_inference": status_of(qwen35_inference),
            "qwen35_snapshot": status_of(qwen35_snapshot),
            "qwen3_inference": status_of(qwen3_inference),
            "qwen3_snapshot": status_of(qwen3_snapshot),
            "multicard_qwen_inference": status_of(multicard_qwen_inference),
            "qwen_compression_quality": status_of(qwen_compression_quality),
            "multicard_qwen_compression_quality": status_of(multicard_qwen_compression_quality),
            "qwen_compression_generate": status_of(qwen_compression_generate),
            "multicard_qwen_lora_finetune": status_of(multicard_qwen_lora_finetune),
            "multicard_qwen_compression_generate": status_of(multicard_qwen_compression_generate),
        },
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# TIDAL-AI Ascend 910B Compatibility Issues",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Demo root: `{report.get('demo_root')}`",
        "",
        "## Blocking / Action Items",
        "",
    ]
    issues = report.get("issues") or []
    if issues:
        for entry in issues:
            lines.extend(
                [
                    f"### {entry['title']}",
                    "",
                    f"- Severity: `{entry['severity']}`",
                    f"- ID: `{entry['id']}`",
                    f"- Evidence: {entry['evidence']}",
                    f"- Next action: {entry['action']}",
                    "",
                ]
            )
    else:
        lines.extend(["- None", ""])

    lines.extend(["## Ready Evidence", ""])
    ready = report.get("ready") or []
    if ready:
        for entry in ready:
            lines.append(f"- `{entry['id']}`: {entry['title']} ({entry['evidence']})")
    else:
        lines.append("- None")

    lines.extend(["", "## Notes", ""])
    notes = report.get("notes") or []
    if notes:
        for entry in notes:
            lines.append(f"- `{entry['id']}`: {entry['title']} - {entry['evidence']}")
    else:
        lines.append("- None")

    lines.extend(["", "## Source Artifacts", ""])
    for name, status in sorted((report.get("sources") or {}).items()):
        lines.append(f"- `{name}`: `{status}`")
    return "\n".join(lines) + "\n"


def write_compatibility_report(demo_root: Path) -> Path:
    artifacts = demo_root / "artifacts"
    reports = demo_root / "reports"
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    report = build_report(demo_root)
    (artifacts / "compatibility_issues.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    output = reports / "compatibility-issues.md"
    output.write_text(markdown(report))
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write Ascend 910B compatibility issue report")
    parser.add_argument("--demo-root", default="/mnt/nvme/622/tidal-demo")
    args = parser.parse_args(argv)
    output = write_compatibility_report(Path(args.demo_root))
    print(f"COMPATIBILITY_REPORT {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
