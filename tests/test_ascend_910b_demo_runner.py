import importlib.util
import json
from pathlib import Path


def load_runner_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_ascend_910b_demo.py"
    spec = importlib.util.spec_from_file_location("run_ascend_910b_demo", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_stage_plan_includes_recording_friendly_defaults(tmp_path):
    runner = load_runner_module()
    assert runner.card_count("0,1") == 2
    assert runner.method_cards_for_methods("0,1", method_count=3) == "0,1,0"
    args = runner.parse_args(
        [
            "--demo-root",
            str(tmp_path / "demo"),
            "--model-path",
            str(tmp_path / "model"),
            "--device",
            "cpu",
            "--cards",
            "0,1",
            "--dry-run",
        ]
    )

    stages = runner.build_stage_plan(args)

    assert [stage.name for stage in stages] == [
        "prepare_workspace",
        "model_inventory",
        "create_tiny_qwen_fixture",
        "multicard_sync",
        "rankadaptor_lora_sync",
        "tiny_qwen_lora_finetune",
        "tiny_qwen_bslora_finetune",
        "multicard_tiny_qwen_bslora_finetune",
        "compression_generate",
        "multicard_compression_generate",
            "qwen_compression_quality",
            "multicard_qwen_compression_quality",
            "compressed_native_torch_serving",
            "compressed_native_torch_serving_code_cache",
            "compressed_native_torch_serving_scaled_code_matmul",
        "serving_export",
        "torch_serving_fallback",
        "multicard_parallel_suite",
        "qwen_qpruner_grouped_replay_8card_utilmon",
        "vllm_serving_probe",
        "vllm_serving_probe_preload_patch",
        "vllm_serving_probe_selector_shim",
        "vllm_serving_benchmark_hbm_parallel",
        "compression_memory_report",
        "qpruner_scale_quality_summary",
        "objective_coverage_audit",
        "progress_report",
        "demo_storyboard",
        "demo_readiness_report",
        "objective_coverage_audit_final",
    ]
    assert "--run-label" in stages[6].command
    assert "tiny_qwen3_bslora_npu" in stages[6].command
    assert "--run-label" in stages[7].command
    assert "tiny_qwen3_bslora_sync_npu" in stages[7].command
    assert "--export-dense-for-serving" in stages[8].command
    assert "--enable-inference-cache" in stages[8].command
    assert "--export-dense-for-serving" in stages[9].command
    assert "--enable-inference-cache" in stages[9].command
    assert "qwen_compression_quality_benchmark.py" in stages[10].command[1]
    assert "--model-id" in stages[10].command
    assert "Qwen/Qwen3-0.6B" in stages[10].command
    assert "--model-path" in stages[10].command
    assert "/mnt/nvme/622/models/Qwen3-0.6B" in stages[10].command
    assert "--target-layer-limit" in stages[10].command
    assert "qwen3_06b_quality_npu" in stages[10].command
    assert "multicard_qwen_compression_quality_benchmark.py" in stages[11].command[1]
    assert "--cards" in stages[11].command
    assert "0,1" in stages[11].command
    assert "qwen3_06b_quality_sync_npu" in stages[11].command
    assert "compressed_native_torch_serving_benchmark.py" in stages[12].command[1]
    assert "--enable-inference-cache" in stages[12].command
    assert "--run-label" in stages[12].command
    assert "tiny_qwen3_native_serving_npu" in stages[12].command
    assert stages[12].required is True
    assert "compressed_native_torch_serving_benchmark.py" in stages[13].command[1]
    assert "--qpruner-cache-mode" in stages[13].command
    assert "code" in stages[13].command
    assert "--run-label" in stages[13].command
    assert "tiny_qwen3_native_serving_code_cache_npu" in stages[13].command
    assert stages[13].required is True
    assert "compressed_native_torch_serving_benchmark.py" in stages[14].command[1]
    assert "--qpruner-cache-mode" in stages[14].command
    assert "scaled-code-matmul" in stages[14].command
    assert "--run-label" in stages[14].command
    assert "tiny_qwen3_native_serving_scaled_code_matmul_npu" in stages[14].command
    assert stages[14].required is True
    assert "export_tiny_qwen_compressed_serving.py" in stages[15].command[1]
    assert "--run-label" in stages[15].command
    assert "tiny_qwen3_serving_export_npu" in stages[15].command
    assert stages[15].required is True
    assert "torch_serving_fallback_benchmark.py" in stages[16].command[1]
    assert "--export-run-label" in stages[16].command
    assert "tiny_qwen3_serving_export_npu" in stages[16].command
    assert "tiny_qwen3_serving_fallback_npu" in stages[16].command
    assert stages[16].required is True
    assert "multicard_parallel_suite.py" in stages[17].command[1]
    assert "--tasks" in stages[17].command
    assert "workflow_cap,workflow_qpruner,rankadaptor,torch_serving_fallback,vllm_serving_benchmark" in stages[17].command
    assert "--cards" in stages[17].command
    assert "0,1" in stages[17].command
    assert "demo_parallel_suite_npu" in stages[17].command
    assert stages[17].required is True
    assert "run_with_npu_monitor.py" in stages[18].command[1]
    assert "--run-label" in stages[18].command
    assert "qwen3_06b_real_grouped_replay_8card_bits8_releasepacked_npu_utilmon_b32_i1000" in stages[18].command
    assert "--expected-cards" in stages[18].command
    assert "2" in stages[18].command
    assert "--" in stages[18].command
    replay_command = stages[18].command[stages[18].command.index("--") + 1 :]
    assert "multicard_qwen_qpruner_grouped_replay.py" in replay_command[1]
    assert "--cards" in replay_command
    assert "0,1" in replay_command
    assert "--batch-size" in replay_command
    assert "32" in replay_command
    assert "--iters" in replay_command
    assert "1000" in replay_command
    assert "--target-layer-limit" in replay_command
    assert "0" in replay_command
    assert "--qpruner-average-bits" in replay_command
    assert "8.0" in replay_command
    assert "--qpruner-release-packed-after-cache" in replay_command
    assert "--qpruner-prebuild-scaled-code-dtype-cache" not in replay_command
    assert stages[18].required is True
    assert "probe_vllm_serving_exports.py" in stages[19].command[1]
    assert "tiny_qwen3_serving_export_npu" in stages[19].command
    assert "--gpu-memory-utilization" in stages[19].command
    assert stages[19].required is False
    assert "probe_vllm_serving_exports.py" in stages[20].command[1]
    assert "tiny_qwen3_serving_export_npu_preload_patch" in stages[20].command
    assert "--export-run-label" in stages[20].command
    assert "--preload-vllm-ascend-patch" in stages[20].command
    assert stages[20].required is False
    assert "probe_vllm_serving_exports.py" in stages[21].command[1]
    assert "tiny_qwen3_serving_export_npu_selector_shim" in stages[21].command
    assert "--export-run-label" in stages[21].command
    assert "--preload-vllm-ascend-patch" in stages[21].command
    assert "--preload-vllm-ascend-selector-shim" in stages[21].command
    assert stages[21].required is False
    assert "vllm_serving_benchmark.py" in stages[22].command[1]
    assert "tiny_qwen3_serving_vllm_metadata_shim_npu_hbm_parallel" in stages[22].command
    assert "--export-run-label" in stages[22].command
    assert "--parallel-methods" in stages[22].command
    assert "--method-cards" in stages[22].command
    assert "0,1,0" in stages[22].command
    assert "--preload-vllm-ascend-patch" in stages[22].command
    assert "--preload-vllm-ascend-selector-shim" in stages[22].command
    assert "--preload-vllm-ascend-metadata-shim" in stages[22].command
    assert stages[22].required is False
    assert "write_compression_memory_report.py" in stages[23].command[1]
    assert "--export-run-label" in stages[23].command
    assert "tiny_qwen3_serving_export_npu" in stages[23].command
    assert stages[23].required is False
    assert "write_qwen_qpruner_scale_quality_summary.py" in stages[24].command[1]
    assert str(tmp_path / "demo") in stages[24].command
    assert stages[24].required is False
    assert "write_objective_coverage_audit.py" in stages[25].command[1]
    assert str(tmp_path / "demo") in stages[25].command
    assert stages[25].required is False
    assert "write_demo_progress_report.py" in stages[26].command[1]
    assert str(tmp_path / "demo") in stages[26].command
    assert "write_demo_storyboard.py" in stages[27].command[1]
    assert str(tmp_path / "demo") in stages[27].command
    assert "write_demo_readiness_report.py" in stages[28].command[1]
    assert str(tmp_path / "demo") in stages[28].command
    assert "write_objective_coverage_audit.py" in stages[29].command[1]
    assert str(tmp_path / "demo") in stages[29].command
    assert stages[29].required is False
    assert str(tmp_path / "demo") in stages[0].command


def test_stage_plan_can_skip_qwen_compression_quality(tmp_path):
    runner = load_runner_module()
    args = runner.parse_args(
        [
            "--demo-root",
            str(tmp_path / "demo"),
            "--model-path",
            str(tmp_path / "model"),
            "--device",
            "cpu",
            "--cards",
            "0,1",
            "--skip-qwen-compression-quality",
            "--skip-multicard-qwen-compression-quality",
            "--dry-run",
        ]
    )

    stages = runner.build_stage_plan(args)
    names = [stage.name for stage in stages]

    assert "qwen_compression_quality" not in names
    assert "multicard_qwen_compression_quality" not in names
    assert "serving_export" in names


def test_stage_plan_can_skip_torch_serving_fallback(tmp_path):
    runner = load_runner_module()
    args = runner.parse_args(
        [
            "--demo-root",
            str(tmp_path / "demo"),
            "--model-path",
            str(tmp_path / "model"),
            "--device",
            "cpu",
            "--cards",
            "0,1",
            "--skip-torch-serving-fallback",
            "--dry-run",
        ]
    )

    names = [stage.name for stage in runner.build_stage_plan(args)]

    assert "serving_export" in names
    assert "compressed_native_torch_serving" in names
    assert "torch_serving_fallback" not in names
    assert "multicard_parallel_suite" in names
    assert "vllm_serving_probe" in names


def test_stage_plan_can_skip_multicard_parallel_suite(tmp_path):
    runner = load_runner_module()
    args = runner.parse_args(
        [
            "--demo-root",
            str(tmp_path / "demo"),
            "--model-path",
            str(tmp_path / "model"),
            "--device",
            "cpu",
            "--cards",
            "0,1",
            "--skip-multicard-parallel-suite",
            "--dry-run",
        ]
    )

    names = [stage.name for stage in runner.build_stage_plan(args)]

    assert "torch_serving_fallback" in names
    assert "multicard_parallel_suite" not in names
    assert "vllm_serving_probe" in names


def test_stage_plan_can_skip_vllm_serving_benchmark(tmp_path):
    runner = load_runner_module()
    args = runner.parse_args(
        [
            "--demo-root",
            str(tmp_path / "demo"),
            "--model-path",
            str(tmp_path / "model"),
            "--device",
            "cpu",
            "--cards",
            "0,1",
            "--skip-vllm-serving-benchmark",
            "--dry-run",
        ]
    )

    names = [stage.name for stage in runner.build_stage_plan(args)]

    assert "vllm_serving_probe_selector_shim" in names
    assert "vllm_serving_benchmark_hbm_parallel" not in names
    assert "progress_report" in names


def test_dry_run_writes_manifest_without_running_stages(tmp_path):
    runner = load_runner_module()

    rc = runner.main(
        [
            "--demo-root",
            str(tmp_path / "demo"),
            "--model-path",
            str(tmp_path / "model"),
            "--device",
            "cpu",
            "--cards",
            "0,1",
            "--dry-run",
        ]
    )

    manifest = json.loads((tmp_path / "demo" / "artifacts" / "ascend_910b_demo_manifest.json").read_text())
    assert rc == 0
    assert manifest["status"] == "DRY_RUN"
    assert manifest["device"] == "cpu"
    assert manifest["cards"] == "0,1"
    assert manifest["stages"][0]["status"] == "SKIPPED"
    assert manifest["stages"][0]["log_path"].endswith("prepare_workspace.log")
    assert (tmp_path / "demo" / "reports" / "ascend-910b-demo-manifest.md").exists()


def test_demo_manifest_markdown_summarizes_stage_statuses():
    runner = load_runner_module()

    text = runner.manifest_markdown(
        {
            "status": "PASS",
            "stages": [
                {
                    "name": "multicard_sync",
                    "status": "PASS",
                    "seconds": 1.25,
                    "log_path": "logs/multicard_sync.log",
                }
            ],
        }
    )

    assert "Ascend 910B One-Command Demo Manifest" in text
    assert "| multicard_sync | PASS | 1.250 | `logs/multicard_sync.log` |" in text
