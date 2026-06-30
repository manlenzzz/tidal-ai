import importlib.util
import json
from pathlib import Path


def load_writer_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "write_paper_baseline_coverage_audit.py"
    spec = importlib.util.spec_from_file_location("write_paper_baseline_coverage_audit", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def populate_demo(demo_root: Path) -> None:
    artifacts = demo_root / "artifacts"
    write_json(
        artifacts / "qwen_llm_pruner_baseline_qwen3_06b_llm_pruner_full196_npu.json",
        {
            "status": "PASS",
            "targeted_layers": 196,
            "targeted_layers_total": 196,
            "targeted_param_reduction_pct": 25.0,
            "loss_delta": 10.007,
            "latency_speedup": 1.007,
            "tokens_per_s": 537.654,
            "peak_mem_mb": 3506.7,
            "compression_time_s": 0.572,
            "claim_scope": "lightweight Ascend structural-pruning baseline evidence, not official full LLM-Pruner reproduction",
        },
    )
    write_json(
        artifacts / "paper_baseline_qwen3_06b_wanda_sparsegpt_full196_npu.json",
        {
            "status": "PASS",
            "targeted_layers": 196,
            "targeted_layers_total": 196,
            "wanda_targeted_param_reduction_pct": 50.0,
            "wanda_loss_delta": 0.662,
            "wanda_latency_speedup": 0.976,
            "sparsegpt_targeted_param_reduction_pct": 50.0,
            "sparsegpt_loss_delta": 11.018,
            "sparsegpt_latency_speedup": 0.929,
        },
    )
    write_json(
        artifacts / "compression_memory_report.json",
        {
            "status": "ACTIONABLE",
            "compression_memory": {
                "cap": {"targeted_param_reduction_pct": 66.667},
                "qpruner": {"targeted_param_reduction_pct": 50.0},
                "target_layer_coverage_pct": 100.0,
            },
        },
    )
    write_json(
        artifacts / "finetuning_effect_summary.json",
        {
            "status": "PASS",
            "qwen3_lora_world_size": 8,
            "qwen3_lora_validation_loss_delta_avg": -0.008,
            "qwen3_bslora_target_module_count": 196,
            "qwen3_bslora_loss_delta": -0.289,
        },
    )
    write_json(
        artifacts / "rankadaptor_recovery_baseline_suite_qwen3_06b_recovery_baselines_npu.json",
        {
            "status": "PASS",
            "all_methods_passed": True,
            "best_recovery_method": "lora",
            "paper_baselines": ["without recovery", "LoRA", "AdaLoRA-style"],
            "target_modules_total": 196,
            "methods": {
                "no_recovery": {
                    "status": "PASS",
                    "paper_baseline": "without recovery",
                    "loss_delta": 0.0,
                    "trainable_adapter_params": 0,
                },
                "lora": {
                    "status": "PASS",
                    "paper_baseline": "LoRA",
                    "loss_delta": -0.484,
                    "trainable_adapter_params": 26624,
                },
                "adalora_style": {
                    "status": "PASS",
                    "paper_baseline": "AdaLoRA-style",
                    "loss_delta": 0.102,
                    "trainable_adapter_params": 39936,
                    "claim_scope": "controlled AdaLoRA-style approximation, not official AdaLoRA implementation",
                },
            },
        },
    )


def test_paper_baseline_audit_uses_rankadaptor_recovery_suite(tmp_path):
    writer = load_writer_module()
    demo_root = tmp_path / "demo"
    populate_demo(demo_root)

    report = writer.build_audit(demo_root)

    assert report["status"] == "PASS"
    evidence = report["evidence"]["rankadaptor_recovery_controlled"]
    assert evidence["status"] == "PASS"
    assert evidence["all_methods_passed"] is True
    assert evidence["best_recovery_method"] == "lora"
    assert evidence["lora_loss_delta"] == -0.484
    assert evidence["adalora_style_loss_delta"] == 0.102
    rank_decision = [item for item in report["coverage_decisions"] if item["area"] == "RankAdaptor"][0]
    assert "controlled LoRA / AdaLoRA-style / no-recovery" in rank_decision["decision"]
    assert "Official AdaLoRA" in rank_decision["remaining_gap"]
    assert any("RankAdaptor recovery baselines now have controlled Ascend evidence" in item for item in report["video_readout"])
    assert not any("Need controlled LoRA vs AdaLoRA" in item.get("remaining_gap", "") for item in report["coverage_decisions"])
    text = writer.markdown(report)
    assert "RankAdaptor recovery controlled baselines" in text
    assert "AdaLoRA-style" in text


def test_paper_baseline_audit_accepts_nested_full_target_artifact_schema(tmp_path):
    writer = load_writer_module()
    demo_root = tmp_path / "demo"
    populate_demo(demo_root)
    artifacts = demo_root / "artifacts"
    write_json(
        artifacts / "qwen_llm_pruner_baseline_qwen3_06b_llm_pruner_full196_npu.json",
        {
            "status": "PASS",
            "target_layer_limit": 196,
            "targeted_layers_total": 196,
            "llm_pruner": {
                "status": "PASS",
                "targeted_layers": 196,
                "targeted_param_reduction_pct": 25.0,
                "loss_delta": 10.007,
                "latency_speedup": 1.007,
                "tokens_per_s": 537.654,
                "peak_mem_mb": 3506.7,
                "compression_time_s": 0.572,
            },
        },
    )
    write_json(
        artifacts / "paper_baseline_qwen3_06b_wanda_sparsegpt_full196_npu.json",
        {
            "status": "PASS",
            "target_layer_limit": 196,
            "targeted_layers_total": 196,
            "wanda": {
                "status": "PASS",
                "targeted_layers": 196,
                "targeted_param_reduction_pct": 50.0,
                "loss_delta": 0.662,
                "latency_speedup": 0.976,
                "compression_time_s": 1.25,
            },
            "sparsegpt": {
                "status": "PASS",
                "targeted_layers": 196,
                "targeted_param_reduction_pct": 50.0,
                "loss_delta": 11.018,
                "latency_speedup": 0.929,
                "compression_time_s": 1.5,
            },
        },
    )

    report = writer.build_audit(demo_root)

    qpruner = report["evidence"]["qpruner_llm_pruner_full196"]
    cap = report["evidence"]["cap_wanda_sparsegpt_full196"]
    assert qpruner["targeted_layers"] == 196
    assert qpruner["targeted_param_reduction_pct"] == 25.0
    assert qpruner["loss_delta"] == 10.007
    assert cap["targeted_layers"] == 196
    assert cap["wanda_loss_delta"] == 0.662
    assert cap["sparsegpt_loss_delta"] == 11.018
    assert any("196/196 LLM-Pruner-style Ascend evidence" in item for item in report["video_readout"])
    assert any("Wanda loss delta 0.662, SparseGPT loss delta 11.018" in item for item in report["video_readout"])


def test_write_outputs_creates_json_and_markdown(tmp_path):
    writer = load_writer_module()
    demo_root = tmp_path / "demo"
    populate_demo(demo_root)

    json_path, md_path = writer.write_outputs(demo_root)

    assert json_path.name == "paper_baseline_coverage_audit.json"
    assert md_path.name == "paper-baseline-coverage-audit.md"
    payload = json.loads(json_path.read_text())
    assert payload["evidence"]["rankadaptor_recovery_controlled"]["status"] == "PASS"
    assert "RankAdaptor recovery controlled baselines" in md_path.read_text()
