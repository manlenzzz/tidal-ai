#!/usr/bin/env python
"""Run the recording-friendly Ascend 910B TIDAL-AI demo pipeline."""

import argparse
import json
import platform
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class Stage:
    name: str
    command: list[str]
    required: bool = True


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Ascend 910B TIDAL-AI demo pipeline")
    parser.add_argument("--demo-root", default="/mnt/nvme/622/tidal-demo")
    parser.add_argument("--model-id", default="TinyQwen3-Offline")
    parser.add_argument("--model-path", default="/mnt/nvme/622/models/TinyQwen3-Offline")
    parser.add_argument("--qwen-model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--qwen-model-path", default="/mnt/nvme/622/models/Qwen3-0.6B")
    parser.add_argument("--device", default="npu")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--cards", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--skip-multicard-sync", action="store_true")
    parser.add_argument("--skip-lora-sync", action="store_true")
    parser.add_argument("--skip-tiny-lora", action="store_true")
    parser.add_argument("--skip-tiny-bslora", action="store_true")
    parser.add_argument("--skip-multicard-tiny-bslora", action="store_true")
    parser.add_argument("--skip-compression-generate", action="store_true")
    parser.add_argument("--skip-multicard-compression-generate", action="store_true")
    parser.add_argument("--skip-qwen-compression-quality", action="store_true")
    parser.add_argument("--skip-multicard-qwen-compression-quality", action="store_true")
    parser.add_argument("--skip-compressed-native-torch-serving", action="store_true")
    parser.add_argument("--skip-serving-export", action="store_true")
    parser.add_argument("--skip-torch-serving-fallback", action="store_true")
    parser.add_argument("--skip-multicard-parallel-suite", action="store_true")
    parser.add_argument("--skip-qwen-qpruner-grouped-replay-utilmon", action="store_true")
    parser.add_argument("--skip-vllm-probe", action="store_true")
    parser.add_argument("--skip-vllm-serving-benchmark", action="store_true")
    parser.add_argument("--lora-steps", type=int, default=5)
    parser.add_argument("--sync-steps", type=int, default=1)
    parser.add_argument("--compression-iters", type=int, default=3)
    parser.add_argument("--compression-warmup", type=int, default=1)
    parser.add_argument("--qwen-quality-iters", type=int, default=3)
    parser.add_argument("--qwen-quality-warmup", type=int, default=1)
    parser.add_argument("--qwen-quality-target-layer-limit", type=int, default=2)
    parser.add_argument("--qwen-grouped-replay-batch-size", type=int, default=32)
    parser.add_argument("--qwen-grouped-replay-iters", type=int, default=1000)
    parser.add_argument("--qwen-grouped-replay-warmup", type=int, default=3)
    parser.add_argument("--qwen-grouped-replay-monitor-interval", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def script_path(name: str) -> str:
    return str(Path(__file__).resolve().parent / name)


def parse_card_ids(raw: str) -> list[str]:
    cards = [part.strip() for part in str(raw).split(",") if part.strip()]
    return cards or ["0"]


def card_count(raw: str) -> int:
    return len(parse_card_ids(raw))


def method_cards_for_methods(raw: str, *, method_count: int = 3) -> str:
    cards = parse_card_ids(raw)
    selected = [cards[index % len(cards)] for index in range(method_count)]
    return ",".join(selected)


def build_stage_plan(args: argparse.Namespace) -> list[Stage]:
    demo_root = str(Path(args.demo_root))
    model_path = str(Path(args.model_path))
    qwen_model_path = str(Path(args.qwen_model_path))
    python = sys.executable
    selected_cards = str(args.cards)
    selected_card_count = card_count(selected_cards)
    stages = [
        Stage("prepare_workspace", ["bash", script_path("prepare_demo_workspace.sh"), demo_root]),
        Stage(
            "model_inventory",
            [
                python,
                script_path("write_model_inventory_report.py"),
                "--demo-root",
                demo_root,
            ],
        ),
        Stage(
            "create_tiny_qwen_fixture",
            [
                python,
                script_path("create_tiny_qwen_fixture.py"),
                "--output-dir",
                model_path,
                "--vocab-size",
                "256",
                "--hidden-size",
                "64",
                "--intermediate-size",
                "128",
            ],
        ),
    ]
    if not args.skip_multicard_sync:
        stages.append(
            Stage(
                "multicard_sync",
                [
                    python,
                    script_path("ascend_multicard_sync_smoke.py"),
                    "--demo-root",
                    demo_root,
                    "--device",
                    args.device,
                    "--cards",
                    args.cards,
                    "--steps",
                    str(args.sync_steps),
                ],
            )
        )
    if not args.skip_lora_sync:
        stages.append(
            Stage(
                "rankadaptor_lora_sync",
                [
                    python,
                    script_path("rankadaptor_lora_sync_smoke.py"),
                    "--demo-root",
                    demo_root,
                    "--device",
                    args.device,
                    "--cards",
                    args.cards,
                    "--steps",
                    str(args.lora_steps),
                ],
            )
        )
    if not args.skip_tiny_lora:
        stages.append(
            Stage(
                "tiny_qwen_lora_finetune",
                [
                    python,
                    script_path("tiny_qwen_lora_finetune.py"),
                    "--demo-root",
                    demo_root,
                    "--model-id",
                    args.model_id,
                    "--model-path",
                    model_path,
                    "--device",
                    args.device,
                    "--dtype",
                    args.dtype,
                    "--steps",
                    str(args.lora_steps),
                    "--max-new-tokens",
                    str(args.max_new_tokens),
                    "--run-label",
                    "tiny_qwen3_lora_npu",
                ],
            )
        )
    if not args.skip_tiny_bslora:
        stages.append(
            Stage(
                "tiny_qwen_bslora_finetune",
                [
                    python,
                    script_path("tiny_qwen_bslora_finetune.py"),
                    "--demo-root",
                    demo_root,
                    "--model-id",
                    args.model_id,
                    "--model-path",
                    model_path,
                    "--device",
                    args.device,
                    "--dtype",
                    args.dtype,
                    "--steps",
                    str(args.lora_steps),
                    "--max-new-tokens",
                    str(args.max_new_tokens),
                    "--run-label",
                    "tiny_qwen3_bslora_npu",
                ],
            )
        )
    if not args.skip_multicard_tiny_bslora:
        stages.append(
            Stage(
                "multicard_tiny_qwen_bslora_finetune",
                [
                    python,
                    script_path("multicard_tiny_qwen_bslora_finetune.py"),
                    "--demo-root",
                    demo_root,
                    "--model-id",
                    args.model_id,
                    "--model-path",
                    model_path,
                    "--device",
                    args.device,
                    "--dtype",
                    args.dtype,
                    "--cards",
                    args.cards,
                    "--steps",
                    str(args.lora_steps),
                    "--run-label",
                    "tiny_qwen3_bslora_sync_npu",
                ],
            )
        )
    if not args.skip_compression_generate:
        stages.append(
            Stage(
                "compression_generate",
                [
                    python,
                    script_path("tiny_qwen_compression_generate_benchmark.py"),
                    "--demo-root",
                    demo_root,
                    "--model-id",
                    args.model_id,
                    "--model-path",
                    model_path,
                    "--device",
                    args.device,
                    "--dtype",
                    args.dtype,
                    "--max-new-tokens",
                    str(args.max_new_tokens),
                    "--iters",
                    str(args.compression_iters),
                    "--warmup",
                    str(args.compression_warmup),
                    "--cap-budget",
                    "8192",
                    "--cap-max-iter",
                    "4",
                    "--cap-policy-steps",
                    "1",
                    "--cap-samples-per-step",
                    "1",
                    "--qpruner-average-bits",
                    "4.0",
                    "--enable-inference-cache",
                    "--export-dense-for-serving",
                    "--run-label",
                    "tiny_qwen3_generate_serving_export_npu",
                ],
            )
        )
    if not args.skip_multicard_compression_generate:
        stages.append(
            Stage(
                "multicard_compression_generate",
                [
                    python,
                    script_path("multicard_tiny_qwen_generate_benchmark.py"),
                    "--demo-root",
                    demo_root,
                    "--model-id",
                    args.model_id,
                    "--model-path",
                    model_path,
                    "--device",
                    args.device,
                    "--dtype",
                    args.dtype,
                    "--cards",
                    args.cards,
                    "--max-new-tokens",
                    str(args.max_new_tokens),
                    "--iters",
                    str(args.compression_iters),
                    "--warmup",
                    str(args.compression_warmup),
                    "--cap-budget",
                    "8192",
                    "--cap-max-iter",
                    "4",
                    "--cap-policy-steps",
                    "1",
                    "--cap-samples-per-step",
                    "1",
                    "--qpruner-average-bits",
                    "4.0",
                    "--enable-inference-cache",
                    "--export-dense-for-serving",
                    "--run-label",
                    "tiny_qwen3_multicard_generate_npu",
                ],
            )
        )
    if not args.skip_qwen_compression_quality:
        stages.append(
            Stage(
                "qwen_compression_quality",
                [
                    python,
                    script_path("qwen_compression_quality_benchmark.py"),
                    "--demo-root",
                    demo_root,
                    "--model-id",
                    args.qwen_model_id,
                    "--model-path",
                    qwen_model_path,
                    "--device",
                    args.device,
                    "--dtype",
                    args.dtype,
                    "--max-length",
                    "96",
                    "--iters",
                    str(args.qwen_quality_iters),
                    "--warmup",
                    str(args.qwen_quality_warmup),
                    "--target-layer-limit",
                    str(args.qwen_quality_target_layer_limit),
                    "--cap-budget",
                    "1048576",
                    "--cap-max-iter",
                    "1",
                    "--cap-policy-steps",
                    "1",
                    "--cap-samples-per-step",
                    "1",
                    "--qpruner-average-bits",
                    "4.0",
                    "--enable-inference-cache",
                    "--inplace-compression",
                    "--run-label",
                    "qwen3_06b_quality_npu",
                ],
            )
        )
    if not args.skip_multicard_qwen_compression_quality:
        stages.append(
            Stage(
                "multicard_qwen_compression_quality",
                [
                    python,
                    script_path("multicard_qwen_compression_quality_benchmark.py"),
                    "--demo-root",
                    demo_root,
                    "--model-id",
                    args.qwen_model_id,
                    "--model-path",
                    qwen_model_path,
                    "--device",
                    args.device,
                    "--dtype",
                    args.dtype,
                    "--cards",
                    args.cards,
                    "--max-length",
                    "96",
                    "--iters",
                    str(max(1, min(int(args.qwen_quality_iters), 1))),
                    "--warmup",
                    "0",
                    "--target-layer-limit",
                    str(args.qwen_quality_target_layer_limit),
                    "--cap-budget",
                    "1048576",
                    "--cap-max-iter",
                    "1",
                    "--cap-policy-steps",
                    "1",
                    "--cap-samples-per-step",
                    "1",
                    "--qpruner-average-bits",
                    "4.0",
                    "--enable-inference-cache",
                    "--inplace-compression",
                    "--timeout-seconds",
                    "1200",
                    "--run-label",
                    "qwen3_06b_quality_sync_npu",
                ],
            )
        )
    if not args.skip_compressed_native_torch_serving:
        stages.append(
            Stage(
                "compressed_native_torch_serving",
                [
                    python,
                    script_path("compressed_native_torch_serving_benchmark.py"),
                    "--demo-root",
                    demo_root,
                    "--model-id",
                    args.model_id,
                    "--model-path",
                    model_path,
                    "--device",
                    args.device,
                    "--dtype",
                    args.dtype,
                    "--max-new-tokens",
                    str(args.max_new_tokens),
                    "--iters",
                    str(args.compression_iters),
                    "--warmup",
                    str(args.compression_warmup),
                    "--cap-budget",
                    "8192",
                    "--cap-max-iter",
                    "4",
                    "--cap-policy-steps",
                    "1",
                    "--cap-samples-per-step",
                    "1",
                    "--qpruner-average-bits",
                    "4.0",
                    "--enable-inference-cache",
                    "--run-label",
                    "tiny_qwen3_native_serving_npu",
                ],
            )
        )
        stages.append(
            Stage(
                "compressed_native_torch_serving_code_cache",
                [
                    python,
                    script_path("compressed_native_torch_serving_benchmark.py"),
                    "--demo-root",
                    demo_root,
                    "--model-id",
                    args.model_id,
                    "--model-path",
                    model_path,
                    "--device",
                    args.device,
                    "--dtype",
                    args.dtype,
                    "--max-new-tokens",
                    str(args.max_new_tokens),
                    "--iters",
                    str(args.compression_iters),
                    "--warmup",
                    str(args.compression_warmup),
                    "--cap-budget",
                    "8192",
                    "--cap-max-iter",
                    "4",
                    "--cap-policy-steps",
                    "1",
                    "--cap-samples-per-step",
                    "1",
                    "--qpruner-average-bits",
                    "4.0",
                    "--qpruner-cache-mode",
                    "code",
                    "--run-label",
                    "tiny_qwen3_native_serving_code_cache_npu",
                ],
            )
        )
        stages.append(
            Stage(
                "compressed_native_torch_serving_scaled_code_matmul",
                [
                    python,
                    script_path("compressed_native_torch_serving_benchmark.py"),
                    "--demo-root",
                    demo_root,
                    "--model-id",
                    args.model_id,
                    "--model-path",
                    model_path,
                    "--device",
                    args.device,
                    "--dtype",
                    args.dtype,
                    "--max-new-tokens",
                    str(args.max_new_tokens),
                    "--iters",
                    str(args.compression_iters),
                    "--warmup",
                    str(args.compression_warmup),
                    "--cap-budget",
                    "8192",
                    "--cap-max-iter",
                    "4",
                    "--cap-policy-steps",
                    "1",
                    "--cap-samples-per-step",
                    "1",
                    "--qpruner-average-bits",
                    "4.0",
                    "--qpruner-cache-mode",
                    "scaled-code-matmul",
                    "--run-label",
                    "tiny_qwen3_native_serving_scaled_code_matmul_npu",
                ],
            )
        )
    if not args.skip_serving_export:
        stages.append(
            Stage(
                "serving_export",
                [
                    python,
                    script_path("export_tiny_qwen_compressed_serving.py"),
                    "--demo-root",
                    demo_root,
                    "--model-id",
                    args.model_id,
                    "--model-path",
                    model_path,
                    "--device",
                    args.device,
                    "--dtype",
                    args.dtype,
                    "--cap-budget",
                    "8192",
                    "--cap-max-iter",
                    "4",
                    "--cap-policy-steps",
                    "1",
                    "--cap-samples-per-step",
                    "1",
                    "--qpruner-average-bits",
                    "4.0",
                    "--run-label",
                    "tiny_qwen3_serving_export_npu",
                ],
            )
        )
    if not args.skip_torch_serving_fallback:
        stages.append(
            Stage(
                "torch_serving_fallback",
                [
                    python,
                    script_path("torch_serving_fallback_benchmark.py"),
                    "--demo-root",
                    demo_root,
                    "--export-run-label",
                    "tiny_qwen3_serving_export_npu",
                    "--run-label",
                    "tiny_qwen3_serving_fallback_npu",
                    "--device",
                    args.device,
                    "--dtype",
                    args.dtype,
                    "--max-new-tokens",
                    str(args.max_new_tokens),
                    "--iters",
                    str(args.compression_iters),
                    "--warmup",
                    str(args.compression_warmup),
                ],
            )
        )
    if not args.skip_multicard_parallel_suite:
        stages.append(
            Stage(
                "multicard_parallel_suite",
                [
                    python,
                    script_path("multicard_parallel_suite.py"),
                    "--project-root",
                    str(Path(__file__).resolve().parents[1]),
                    "--demo-root",
                    demo_root,
                    "--cards",
                    args.cards,
                    "--tasks",
                    "workflow_cap,workflow_qpruner,rankadaptor,torch_serving_fallback,vllm_serving_benchmark",
                    "--run-label",
                    "demo_parallel_suite_npu",
                    "--device",
                    args.device,
                    "--dtype",
                    args.dtype,
                    "--export-run-label",
                    "tiny_qwen3_serving_export_npu",
                    "--iters",
                    str(args.compression_iters),
                    "--warmup",
                    str(args.compression_warmup),
                    "--max-new-tokens",
                    str(args.max_new_tokens),
                ],
            )
        )
    if not args.skip_qwen_qpruner_grouped_replay_utilmon:
        run_label = "qwen3_06b_real_grouped_replay_8card_bits8_releasepacked_npu_utilmon_b32_i1000"
        stages.append(
            Stage(
                "qwen_qpruner_grouped_replay_8card_utilmon",
                [
                    python,
                    script_path("run_with_npu_monitor.py"),
                    "--demo-root",
                    demo_root,
                    "--run-label",
                    run_label,
                    "--interval-seconds",
                    str(args.qwen_grouped_replay_monitor_interval),
                    "--expected-cards",
                    str(selected_card_count),
                    "--",
                    python,
                    script_path("multicard_qwen_qpruner_grouped_replay.py"),
                    "--project-root",
                    str(Path(__file__).resolve().parents[1]),
                    "--demo-root",
                    demo_root,
                    "--cards",
                    selected_cards,
                    "--run-label",
                    run_label,
                    "--model-id",
                    args.qwen_model_id,
                    "--model-path",
                    qwen_model_path,
                    "--device",
                    args.device,
                    "--dtype",
                    args.dtype,
                    "--batch-size",
                    str(args.qwen_grouped_replay_batch_size),
                    "--iters",
                    str(args.qwen_grouped_replay_iters),
                    "--warmup",
                    str(args.qwen_grouped_replay_warmup),
                    "--max-modules",
                    "8",
                    "--min-group-modules",
                    "2",
                    "--max-groups",
                    "0",
                    "--target-layer-limit",
                    "0",
                    "--qpruner-average-bits",
                    "8.0",
                    "--dense-weight-bits",
                    "16.0",
                    "--qpruner-release-packed-after-cache",
                    "--max-start-skew-seconds",
                    "1.0",
                    "--sync-start-delay-seconds",
                    "5.0",
                ],
            )
        )
    if not args.skip_vllm_probe:
        stages.extend(
            [
                Stage(
                    "vllm_serving_probe",
                    [
                        python,
                        script_path("probe_vllm_serving_exports.py"),
                        "--demo-root",
                        demo_root,
                        "--run-label",
                        "tiny_qwen3_serving_export_npu",
                        "--gpu-memory-utilization",
                        "0.05",
                    ],
                    required=False,
                ),
                Stage(
                    "vllm_serving_probe_preload_patch",
                    [
                        python,
                        script_path("probe_vllm_serving_exports.py"),
                        "--demo-root",
                        demo_root,
                        "--run-label",
                        "tiny_qwen3_serving_export_npu_preload_patch",
                        "--export-run-label",
                        "tiny_qwen3_serving_export_npu",
                        "--gpu-memory-utilization",
                        "0.05",
                        "--preload-vllm-ascend-patch",
                    ],
                    required=False,
                ),
                Stage(
                    "vllm_serving_probe_selector_shim",
                    [
                        python,
                        script_path("probe_vllm_serving_exports.py"),
                        "--demo-root",
                        demo_root,
                        "--run-label",
                        "tiny_qwen3_serving_export_npu_selector_shim",
                        "--export-run-label",
                        "tiny_qwen3_serving_export_npu",
                        "--gpu-memory-utilization",
                        "0.05",
                        "--preload-vllm-ascend-patch",
                        "--preload-vllm-ascend-selector-shim",
                    ],
                    required=False,
                ),
            ]
        )
    if not args.skip_vllm_serving_benchmark:
        stages.append(
            Stage(
                "vllm_serving_benchmark_hbm_parallel",
                [
                    python,
                    script_path("vllm_serving_benchmark.py"),
                    "--demo-root",
                    demo_root,
                    "--run-label",
                    "tiny_qwen3_serving_vllm_metadata_shim_npu_hbm_parallel",
                    "--export-run-label",
                    "tiny_qwen3_serving_export_npu",
                    "--dtype",
                    args.dtype,
                    "--max-new-tokens",
                    str(args.max_new_tokens),
                    "--iters",
                    str(args.compression_iters),
                    "--warmup",
                    str(args.compression_warmup),
                    "--gpu-memory-utilization",
                    "0.05",
                    "--parallel-methods",
                    "--method-cards",
                    method_cards_for_methods(selected_cards),
                    "--preload-vllm-ascend-patch",
                    "--preload-vllm-ascend-selector-shim",
                    "--preload-vllm-ascend-metadata-shim",
                ],
                required=False,
            )
        )
    stages.append(
        Stage(
            "compression_memory_report",
            [
                python,
                script_path("write_compression_memory_report.py"),
                "--demo-root",
                demo_root,
                "--export-run-label",
                "tiny_qwen3_serving_export_npu",
            ],
            required=False,
        )
    )
    stages.append(
        Stage(
            "qpruner_scale_quality_summary",
            [python, script_path("write_qwen_qpruner_scale_quality_summary.py"), "--demo-root", demo_root],
            required=False,
        )
    )
    stages.append(
        Stage(
            "objective_coverage_audit",
            [python, script_path("write_objective_coverage_audit.py"), "--demo-root", demo_root],
            required=False,
        )
    )
    stages.append(
        Stage(
            "progress_report",
            [python, script_path("write_demo_progress_report.py"), "--demo-root", demo_root],
            required=False,
        )
    )
    stages.append(
        Stage(
            "demo_storyboard",
            [python, script_path("write_demo_storyboard.py"), "--demo-root", demo_root],
            required=False,
        )
    )
    stages.append(
        Stage(
            "demo_readiness_report",
            [python, script_path("write_demo_readiness_report.py"), "--demo-root", demo_root],
            required=False,
        )
    )
    stages.append(
        Stage(
            "objective_coverage_audit_final",
            [python, script_path("write_objective_coverage_audit.py"), "--demo-root", demo_root],
            required=False,
        )
    )
    return stages


def ensure_demo_dirs(demo_root: Path) -> None:
    for name in ("artifacts", "logs", "reports", "recordings", "scripts"):
        (demo_root / name).mkdir(parents=True, exist_ok=True)


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def command_text(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run_stage(stage: Stage, *, demo_root: Path, dry_run: bool) -> dict[str, object]:
    log_path = demo_root / "logs" / f"{stage.name}.log"
    started = time.time()
    record: dict[str, object] = {
        "name": stage.name,
        "required": stage.required,
        "command": stage.command,
        "command_text": command_text(stage.command),
        "log_path": rel(log_path, demo_root),
    }
    if dry_run:
        record.update({"status": "SKIPPED", "returncode": None, "seconds": 0.0})
        return record

    with log_path.open("w") as log:
        log.write(f"$ {record['command_text']}\n\n")
        log.flush()
        proc = subprocess.run(stage.command, stdout=log, stderr=subprocess.STDOUT, text=True, check=False)
    seconds = time.time() - started
    status = "PASS" if proc.returncode == 0 else "FAIL"
    record.update({"status": status, "returncode": proc.returncode, "seconds": seconds})
    return record


def manifest_status(stage_records: list[dict[str, object]], *, dry_run: bool) -> str:
    if dry_run:
        return "DRY_RUN"
    required = [stage for stage in stage_records if stage.get("required", True)]
    return "PASS" if required and all(stage.get("status") == "PASS" for stage in required) else "FAIL"


def fmt_seconds(value: object) -> str:
    if isinstance(value, (float, int)):
        return f"{float(value):.3f}"
    return "0.000"


def manifest_markdown(manifest: dict[str, object]) -> str:
    lines = [
        "# Ascend 910B One-Command Demo Manifest",
        "",
        f"- Overall status: `{manifest.get('status')}`",
        f"- Demo root: `{manifest.get('demo_root', '')}`",
        f"- Device: `{manifest.get('device', '')}`",
        f"- Cards: `{manifest.get('cards', '')}`",
        "",
        "| Stage | Status | Seconds | Log |",
        "|---|---:|---:|---|",
    ]
    for stage in manifest.get("stages", []):
        if not isinstance(stage, dict):
            continue
        lines.append(
            "| {name} | {status} | {seconds} | `{log_path}` |".format(
                name=stage.get("name", ""),
                status=stage.get("status", ""),
                seconds=fmt_seconds(stage.get("seconds")),
                log_path=stage.get("log_path", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Recording Notes",
            "",
            "Run this script from inside the project container. It preserves every stage log and JSON artifact under the demo root for later video recording.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_manifest(manifest: dict[str, object], demo_root: Path) -> None:
    artifacts = demo_root / "artifacts"
    reports = demo_root / "reports"
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    (artifacts / "ascend_910b_demo_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    (reports / "ascend-910b-demo-manifest.md").write_text(manifest_markdown(manifest))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    demo_root = Path(args.demo_root)
    ensure_demo_dirs(demo_root)
    stages = build_stage_plan(args)
    stage_records: list[dict[str, object]] = []
    for stage in stages:
        record = run_stage(stage, demo_root=demo_root, dry_run=bool(args.dry_run))
        stage_records.append(record)
        print(f"ASCEND_910B_DEMO_STAGE {record['name']} {record['status']}")
        if record["status"] == "FAIL" and stage.required:
            break

    manifest = {
        "status": manifest_status(stage_records, dry_run=bool(args.dry_run)),
        "demo_root": str(demo_root),
        "model_id": args.model_id,
        "model_path": args.model_path,
        "qwen_model_id": args.qwen_model_id,
        "qwen_model_path": args.qwen_model_path,
        "device": args.device,
        "dtype": args.dtype,
        "cards": args.cards,
        "dry_run": bool(args.dry_run),
        "platform": platform.platform(),
        "stages": stage_records,
    }
    write_manifest(manifest, demo_root)
    print("ASCEND_910B_DEMO_MANIFEST " + json.dumps(manifest, sort_keys=True))
    return 0 if manifest["status"] in {"PASS", "DRY_RUN"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
