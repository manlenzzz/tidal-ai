import importlib.util
import json
from pathlib import Path


def load_report_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "write_compression_memory_report.py"
    assert script.exists(), f"missing compression memory report script: {script}"
    spec = importlib.util.spec_from_file_location("write_compression_memory_report", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def write_export(path: Path, *, model_bytes: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    write_json(
        path / "config.json",
        {
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "torch_dtype": "float16",
        },
    )
    write_json(path / "generation_config.json", {"do_sample": False})
    write_json(path / "tokenizer_config.json", {"model_max_length": 2048})
    (path / "model.safetensors").write_bytes(b"0" * model_bytes)


def populate_memory_artifacts(demo_root: Path) -> None:
    artifacts = demo_root / "artifacts"
    exports = demo_root / "serving_exports" / "tiny_qwen3_serving_export_npu"
    for method in ("baseline", "cap", "qpruner"):
        write_export(exports / method, model_bytes=1024)
    write_json(
        artifacts / "qwen_compression_quality_qwen3_06b_quality_npu.json",
        {
            "status": "PASS",
            "target_layer_limit": 2,
            "targeted_layers_total": 196,
            "baseline": {"loss": 6.797},
            "cap": {"loss_delta": -0.134, "targeted_compression_ratio": 3.0},
            "wanda": {
                "status": "PASS",
                "loss_delta": 0.088,
                "targeted_param_reduction_pct": 50.0,
                "latency_speedup": 1.125,
                "targeted_layers": 2,
            },
            "sparsegpt": {
                "status": "PASS",
                "loss_delta": 0.077,
                "targeted_param_reduction_pct": 50.0,
                "latency_speedup": 1.111,
                "targeted_layers": 2,
            },
            "qpruner": {"loss_delta": 0.276, "average_bits": 4.0},
        },
    )
    write_json(
        artifacts / "qwen_compression_generate_qwen3_06b_generate_npu.json",
        {
            "status": "PASS",
            "serving_dense_export": True,
            "inference_cache_enabled": True,
            "target_layer_limit": 2,
            "targeted_layers_total": 196,
            "baseline": {"peak_mem_mb": 1159.8},
            "cap": {
                "targeted_compression_ratio": 3.0,
                "targeted_param_reduction_pct": 66.667,
                "peak_mem_mb": 1160.0,
                "exported_dense_linears": 2,
                "cache_modules": 2,
            },
            "qpruner": {
                "average_bits": 4.0,
                "targeted_param_reduction_pct": 75.0,
                "peak_mem_mb": 1160.0,
                "exported_dense_linears": 2,
                "cache_modules": 2,
            },
        },
    )
    write_json(
        artifacts / "vllm_serving_benchmark_tiny_qwen3_serving_vllm_metadata_shim_npu_long_decode.json",
        {
            "status": "PASS",
            "backend": "vllm_ascend_generate",
            "max_new_tokens": 16,
            "exports": {
                "baseline": {"status": "PASS", "tokens_per_s": 377.673, "peak_mem_mb": 0.0},
                "cap": {"status": "PASS", "tokens_per_s": 361.231, "peak_mem_mb": 0.0},
                "qpruner": {"status": "PASS", "tokens_per_s": 355.682, "peak_mem_mb": 0.0},
            },
            "summary": {"cap_vs_baseline_speedup": 0.956, "qpruner_vs_baseline_speedup": 0.942},
        },
    )


def test_memory_report_prioritizes_compression_savings_and_dense_export_gap(tmp_path):
    module = load_report_module()
    demo_root = tmp_path / "demo"
    populate_memory_artifacts(demo_root)

    report = module.build_report(demo_root, "tiny_qwen3_serving_export_npu")
    text = module.markdown(report)

    assert report["status"] == "ACTIONABLE"
    assert report["compression_memory"]["cap"]["targeted_param_reduction_pct"] == 66.667
    assert report["compression_memory"]["qpruner"]["targeted_param_reduction_pct"] == 75.0
    assert report["compression_memory"]["target_layer_coverage_pct"] == 1.02
    assert report["serving_export_memory"]["baseline"]["model_bytes"] == 1024
    assert report["serving_export_memory"]["cap"]["size_ratio_vs_baseline"] == 1.0
    assert report["serving_export_memory"]["qpruner"]["format"] == "dense_hf"
    assert report["engineering_reference"]["serving_baseline_role"] == "uncompressed_serving_export"
    assert report["engineering_reference"]["not_paper_baseline"] is True
    assert report["paper_baselines"]["qpruner"]["primary"] == ["LLM-Pruner"]
    assert "LoftQ" in report["paper_baselines"]["qpruner"]["recovery_baselines"]
    assert {"SparseGPT", "Wanda"} <= set(report["paper_baselines"]["cap"]["pruning_baselines"])
    assert {"LoRA", "AdaLoRA"} <= set(report["paper_baselines"]["rankadaptor"]["recovery_baselines"])
    assert report["baseline_alignment_matrix"][0] == {
        "method": "QPruner",
        "paper_baseline_role": "pruning baseline",
        "paper_baselines": ["LLM-Pruner"],
        "demo_reference_role": "compressed-vs-uncompressed Ascend runtime reference",
        "demo_artifacts": [
            "artifacts/qwen_compression_quality_qwen3_06b_quality_npu.json",
            "artifacts/qwen_compression_generate_qwen3_06b_generate_npu.json",
            "artifacts/compressed_native_sweep_summary.json",
        ],
        "current_status": "NPU algorithm evidence exists; paper-baseline rerun is not claimed.",
    }
    assert report["baseline_alignment_matrix"][1]["method"] == "CAP"
    assert "SparseGPT" in report["baseline_alignment_matrix"][1]["paper_baselines"]
    assert report["baseline_alignment_matrix"][2]["method"] == "RankAdaptor"
    assert report["baseline_alignment_matrix"][2]["demo_reference_role"] == "LoRA/BSLoRA recovery evidence on Ascend"
    assert report["baseline_alignment_matrix"][3]["method"] == "Engineering runtime baseline"
    assert report["baseline_alignment_matrix"][3]["paper_baseline_role"] == "not a paper baseline"
    assert report["paper_baseline_evidence"][0] == {
        "method": "CAP",
        "paper_baseline": "Wanda",
        "status": "RUN_ON_ASCEND",
        "evidence_role": "actual paper pruning baseline",
        "artifacts": ["artifacts/qwen_compression_quality_qwen3_06b_quality_npu.json"],
        "metrics": {
            "loss_delta": 0.088,
            "targeted_param_reduction_pct": 50.0,
            "latency_speedup": 1.125,
            "targeted_layers": 2,
        },
        "next_action": "Use this as the first paper-baseline comparison point; add SparseGPT/DSNoT/OATS/OWL/AlphaPruning when ported.",
    }
    assert report["paper_baseline_evidence"][1]["paper_baseline"] == "SparseGPT / DSNoT / OATS / OWL / AlphaPruning"
    assert report["paper_baseline_evidence"][1]["status"] == "PARTIAL_RUN_ON_ASCEND"
    assert report["paper_baseline_evidence"][1]["metrics"]["sparsegpt"]["loss_delta"] == 0.077
    assert report["paper_baseline_evidence"][1]["pending_baselines"] == ["DSNoT", "OATS", "OWL", "AlphaPruning"]
    assert report["paper_baseline_evidence"][2]["paper_baseline"] == "LLM-Pruner"
    assert report["paper_baseline_evidence"][2]["status"] == "PENDING_NOT_RUN_ON_ASCEND"
    assert report["baseline_alignment_matrix"][1]["current_status"] == (
        "Wanda paper baseline has an Ascend NPU quality artifact; "
        "SparseGPT/DSNoT/OATS/OWL/AlphaPruning are not claimed as rerun."
    )
    assert report["memory_gap"]["serving_dense_export"] is True
    assert report["memory_gap"]["dense_export_erases_storage_savings"] is True
    assert report["memory_first_summary"] == {
        "headline": "Compression demo is memory-first: CAP saves 66.667% targeted memory and QPruner saves 75.000% targeted memory across 2/196 Qwen3 target layers.",
        "serving_gap": "Current dense HF serving export is 1.000x/1.000x of the uncompressed reference, so the video should show compressed-native/runtime-preserved memory savings before claiming serving memory savings.",
        "next_action": "Keep compression in pruned/quantized form inside serving runtime so memory savings survive inference.",
    }
    assert report["video_readout"] == (
        "Memory-first compression readout: CAP targeted memory -66.667%, QPruner targeted memory -75.000%, "
        "target layers 2/196; dense HF export ratio CAP/QPruner 1.000x/1.000x, so memory savings require "
        "the compressed-native runtime path."
    )
    assert {item["id"] for item in report["findings"]} >= {
        "compression_memory_savings_exist",
        "dense_serving_export_erases_memory_savings",
        "benchmark_peak_memory_not_yet_proving_savings",
    }
    assert report["next_action"] == "Keep compression in pruned/quantized form inside serving runtime so memory savings survive inference."
    assert "Compression Memory Report" in text
    assert "memory is the primary compression win" in text
    assert "Memory-first compression readout: CAP targeted memory -66.667%, QPruner targeted memory -75.000%" in text
    assert "Current dense HF serving export is 1.000x/1.000x of the uncompressed reference" in text
    assert "Paper Baselines" in text
    assert "Baseline Alignment Matrix" in text
    assert "Paper Baseline Evidence" in text
    assert "| CAP | Wanda | RUN_ON_ASCEND | artifacts/qwen_compression_quality_qwen3_06b_quality_npu.json | loss_delta=0.088, targeted_param_reduction_pct=50.000, latency_speedup=1.125, targeted_layers=2 |" in text
    assert "| CAP | SparseGPT / DSNoT / OATS / OWL / AlphaPruning | PARTIAL_RUN_ON_ASCEND | artifacts/qwen_compression_quality_qwen3_06b_quality_npu.json | sparsegpt: loss_delta=0.077, targeted_param_reduction_pct=50.000, latency_speedup=1.111, targeted_layers=2; pending=DSNoT, OATS, OWL, AlphaPruning |" in text
    assert "| QPruner | LLM-Pruner | PENDING_NOT_RUN_ON_ASCEND | missing | missing |" in text
    assert "| QPruner | pruning baseline | LLM-Pruner | compressed-vs-uncompressed Ascend runtime reference |" in text
    assert "| CAP | pruning / joint-compression / SVD baselines | SparseGPT, Wanda" in text
    assert "| RankAdaptor | recovery baselines | LoRA, AdaLoRA, without recovery | LoRA/BSLoRA recovery evidence on Ascend |" in text
    assert "| Engineering runtime baseline | not a paper baseline | uncompressed serving export | runtime sanity, latency, throughput, and memory reference |" in text
    assert "QPruner paper baseline: LLM-Pruner" in text
    assert "CAP paper baselines: SparseGPT, Wanda" in text
    assert "Engineering reference: `baseline` is the uncompressed serving export, not the paper baseline." in text
    assert "QPruner targeted memory reduction 75.000%" in text
    assert "dense HF export size ratio 1.000x" in text


def test_memory_report_prefers_qpruner_scale_quality_summary_for_full_coverage_readout(tmp_path):
    module = load_report_module()
    demo_root = tmp_path / "demo"
    populate_memory_artifacts(demo_root)
    write_json(
        demo_root / "artifacts" / "qwen_qpruner_scale_quality_summary.json",
        {
            "status": "PASS",
            "targeted_layers_total": 196,
            "max_target_layer_limit": 196,
            "max_coverage_pct": 100.0,
            "best_under_loss_delta": {
                "run_label": "qwen3_06b_qpruner_sweep_proj_layers160_196_npu_layers196_bits8",
                "target_layer_limit": 196,
                "targeted_layers_total": 196,
                "coverage_pct": 100.0,
                "memory_reduction_pct": 50.0,
                "average_bits": 8.0,
                "loss_delta": -0.003,
            },
            "readout": (
                "QPruner scale-quality sweep reaches 196/196 target layers (100.000% coverage); "
                "best under loss delta <= 0.500 is qwen3_06b_qpruner_sweep_proj_layers160_196_npu_layers196_bits8 "
                "with 50.000% targeted memory reduction and loss-derived approximate PPL 894.892 -> 891.377 (-0.393%)."
            ),
        },
    )

    report = module.build_report(demo_root, "tiny_qwen3_serving_export_npu")
    text = module.markdown(report)

    qpruner = report["compression_memory"]["qpruner"]
    assert report["compression_memory"]["target_layer_limit"] == 196
    assert report["compression_memory"]["targeted_layers_total"] == 196
    assert report["compression_memory"]["target_layer_coverage_pct"] == 100.0
    assert qpruner["targeted_param_reduction_pct"] == 50.0
    assert qpruner["average_bits"] == 8.0
    assert qpruner["scale_quality_artifact"] == "qwen_qpruner_scale_quality_summary.json"
    assert report["memory_first_summary"]["headline"] == (
        "Compression demo is memory-first: CAP saves 66.667% targeted memory and QPruner saves "
        "50.000% targeted memory across 196/196 Qwen3 target layers."
    )
    assert report["video_readout"].startswith(
        "Memory-first compression readout: CAP targeted memory -66.667%, QPruner targeted memory -50.000%, "
        "target layers 196/196;"
    )
    assert "QPruner targeted memory reduction 50.000%" in text
    assert "target layers 196/196" in text
    assert "qwen_qpruner_scale_quality_summary.json" in text


def test_memory_report_falls_back_to_multicard_wanda_evidence(tmp_path):
    module = load_report_module()
    demo_root = tmp_path / "demo"
    populate_memory_artifacts(demo_root)
    artifacts = demo_root / "artifacts"
    quality_path = artifacts / "qwen_compression_quality_qwen3_06b_quality_npu.json"
    quality = json.loads(quality_path.read_text())
    quality.pop("wanda")
    quality_path.write_text(json.dumps(quality))
    write_json(
        artifacts / "multicard_qwen_compression_quality_qwen3_06b_quality_pattern_2card_npu.json",
        {
            "status": "PASS",
            "world_size": 2,
            "target_layer_limit": 4,
            "targeted_layers_total": 196,
            "aggregate": {
                "distributed_reduce_consistent": True,
                "wanda_loss_delta_avg": -0.079,
                "wanda_tokens_per_s_total": 604.0,
            },
            "ranks": [
                {
                    "benchmark": {
                        "wanda": {
                            "status": "PASS",
                            "targeted_param_reduction_pct": 50.0,
                            "targeted_layers": 4,
                        }
                    }
                }
            ],
        },
    )

    report = module.build_report(demo_root, "tiny_qwen3_serving_export_npu")
    text = module.markdown(report)

    assert report["paper_baseline_evidence"][0] == {
        "method": "CAP",
        "paper_baseline": "Wanda",
        "status": "RUN_ON_ASCEND",
        "evidence_role": "actual paper pruning baseline",
        "artifacts": ["artifacts/multicard_qwen_compression_quality_qwen3_06b_quality_pattern_2card_npu.json"],
        "metrics": {
            "loss_delta": -0.079,
            "targeted_param_reduction_pct": 50.0,
            "tokens_per_s_total": 604.0,
            "targeted_layers": 4,
            "world_size": 2,
            "distributed_reduce_consistent": True,
        },
        "next_action": "Use this as the first paper-baseline comparison point; add SparseGPT/DSNoT/OATS/OWL/AlphaPruning when ported.",
    }
    assert "| CAP | Wanda | RUN_ON_ASCEND | artifacts/multicard_qwen_compression_quality_qwen3_06b_quality_pattern_2card_npu.json | loss_delta=-0.079, targeted_param_reduction_pct=50.000, tokens_per_s_total=604.000, targeted_layers=4, world_size=2, distributed_reduce_consistent=True |" in text


def test_memory_report_finds_sparsegpt_evidence_in_separate_quality_artifact(tmp_path):
    module = load_report_module()
    demo_root = tmp_path / "demo"
    populate_memory_artifacts(demo_root)
    artifacts = demo_root / "artifacts"
    quality_path = artifacts / "qwen_compression_quality_qwen3_06b_quality_npu.json"
    quality = json.loads(quality_path.read_text())
    quality.pop("sparsegpt")
    quality_path.write_text(json.dumps(quality))
    write_json(
        artifacts / "qwen_compression_quality_qwen3_06b_quality_sparsegpt_npu.json",
        {
            "status": "PASS",
            "target_layer_limit": 4,
            "targeted_layers_total": 196,
            "sparsegpt": {
                "status": "PASS",
                "loss_delta": 0.123,
                "targeted_param_reduction_pct": 50.0,
                "latency_speedup": 0.955,
                "targeted_layers": 4,
            },
        },
    )

    report = module.build_report(demo_root, "tiny_qwen3_serving_export_npu")
    evidence = report["paper_baseline_evidence"][1]

    assert evidence["status"] == "PARTIAL_RUN_ON_ASCEND"
    assert evidence["artifacts"] == ["artifacts/qwen_compression_quality_qwen3_06b_quality_sparsegpt_npu.json"]
    assert evidence["metrics"]["sparsegpt"] == {
        "loss_delta": 0.123,
        "targeted_param_reduction_pct": 50.0,
        "latency_speedup": 0.955,
        "targeted_layers": 4,
    }
    assert evidence["pending_baselines"] == ["DSNoT", "OATS", "OWL", "AlphaPruning"]


def test_memory_report_prefers_multicard_sparsegpt_evidence(tmp_path):
    module = load_report_module()
    demo_root = tmp_path / "demo"
    populate_memory_artifacts(demo_root)
    artifacts = demo_root / "artifacts"
    quality_path = artifacts / "qwen_compression_quality_qwen3_06b_quality_npu.json"
    quality = json.loads(quality_path.read_text())
    quality.pop("sparsegpt")
    quality_path.write_text(json.dumps(quality))
    write_json(
        artifacts / "qwen_compression_quality_qwen3_06b_quality_sparsegpt_npu.json",
        {
            "status": "PASS",
            "sparsegpt": {
                "status": "PASS",
                "loss_delta": 0.123,
                "targeted_param_reduction_pct": 50.0,
                "targeted_layers": 4,
            },
        },
    )
    write_json(
        artifacts / "multicard_qwen_compression_quality_qwen3_06b_quality_sparsegpt_2card_npu.json",
        {
            "status": "PASS",
            "world_size": 2,
            "target_layer_limit": 4,
            "targeted_layers_total": 196,
            "aggregate": {
                "distributed_reduce_consistent": True,
                "sparsegpt_loss_delta_avg": 0.118,
                "sparsegpt_tokens_per_s_total": 612.0,
            },
            "ranks": [
                {
                    "benchmark": {
                        "sparsegpt": {
                            "status": "PASS",
                            "targeted_param_reduction_pct": 50.0,
                            "targeted_layers": 4,
                        }
                    }
                }
            ],
        },
    )

    report = module.build_report(demo_root, "tiny_qwen3_serving_export_npu")
    evidence = report["paper_baseline_evidence"][1]
    text = module.markdown(report)

    assert evidence["status"] == "PARTIAL_RUN_ON_ASCEND"
    assert evidence["artifacts"] == [
        "artifacts/multicard_qwen_compression_quality_qwen3_06b_quality_sparsegpt_2card_npu.json"
    ]
    assert evidence["metrics"]["sparsegpt"] == {
        "loss_delta": 0.118,
        "targeted_param_reduction_pct": 50.0,
        "tokens_per_s_total": 612.0,
        "targeted_layers": 4,
        "world_size": 2,
        "distributed_reduce_consistent": True,
    }
    assert "tokens_per_s_total=612.000" in text
    assert "distributed_reduce_consistent=True" in text


def test_memory_report_finds_llm_pruner_qpruner_paper_baseline_evidence(tmp_path):
    module = load_report_module()
    demo_root = tmp_path / "demo"
    populate_memory_artifacts(demo_root)
    write_json(
        demo_root / "artifacts" / "qwen_llm_pruner_baseline_qwen3_06b_llm_pruner_npu.json",
        {
            "status": "PASS",
            "paper_baseline": "LLM-Pruner",
            "baseline_family": "LLM-Pruner",
            "baseline_variant": "llm_pruner_style_structured_channel_pruning",
            "claim_scope": "lightweight Ascend structural-pruning baseline evidence, not official full LLM-Pruner reproduction",
            "device": "npu",
            "target_layer_limit": 4,
            "targeted_layers_total": 196,
            "llm_pruner": {
                "status": "PASS",
                "loss_delta": 0.144,
                "targeted_param_reduction_pct": 25.0,
                "latency_speedup": 1.037,
                "tokens_per_s": 512.25,
                "targeted_layers": 4,
                "compression_time_s": 0.42,
            },
        },
    )

    report = module.build_report(demo_root, "tiny_qwen3_serving_export_npu")
    evidence = report["paper_baseline_evidence"][2]
    text = module.markdown(report)

    assert evidence == {
        "method": "QPruner",
        "paper_baseline": "LLM-Pruner",
        "status": "PARTIAL_RUN_ON_ASCEND",
        "evidence_role": "LLM-Pruner-style structural pruning paper-baseline compatibility point",
        "artifacts": ["artifacts/qwen_llm_pruner_baseline_qwen3_06b_llm_pruner_npu.json"],
        "metrics": {
            "loss_delta": 0.144,
            "targeted_param_reduction_pct": 25.0,
            "latency_speedup": 1.037,
            "tokens_per_s": 512.25,
            "targeted_layers": 4,
            "compression_time_s": 0.42,
            "claim_scope": "lightweight Ascend structural-pruning baseline evidence, not official full LLM-Pruner reproduction",
        },
        "next_action": "Replace the lightweight structural baseline with an official full LLM-Pruner port before claiming full QPruner paper-baseline reproduction.",
    }
    assert "| QPruner | LLM-Pruner | PARTIAL_RUN_ON_ASCEND | artifacts/qwen_llm_pruner_baseline_qwen3_06b_llm_pruner_npu.json | loss_delta=0.144, targeted_param_reduction_pct=25.000, latency_speedup=1.037, tokens_per_s=512.250, targeted_layers=4, compression_time_s=0.420, claim_scope=lightweight Ascend structural-pruning baseline evidence, not official full LLM-Pruner reproduction |" in text


def test_cli_writes_json_and_markdown(tmp_path):
    module = load_report_module()
    demo_root = tmp_path / "demo"
    populate_memory_artifacts(demo_root)

    rc = module.main(["--demo-root", str(demo_root), "--export-run-label", "tiny_qwen3_serving_export_npu"])

    assert rc == 0
    payload = json.loads((demo_root / "artifacts" / "compression_memory_report.json").read_text())
    markdown = (demo_root / "reports" / "compression-memory-report.md").read_text()
    assert payload["status"] == "ACTIONABLE"
    assert "Compression Memory Report" in markdown
