import importlib.util
import json
from pathlib import Path


def load_compat_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "write_compatibility_report.py"
    spec = importlib.util.spec_from_file_location("write_compatibility_report", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_build_report_extracts_vllm_acl_and_serving_tradeoff(tmp_path):
    compat = load_compat_module()
    demo_root = tmp_path / "demo"
    artifacts = demo_root / "artifacts"
    write_json(
        artifacts / "vllm_serving_probe_tiny_qwen3_serving_export_npu.json",
        {
            "status": "FAIL",
            "vllm_available": True,
            "exports": {
                "cap": {"exists": True, "transformers_load": "PASS", "vllm_load": "FAIL"},
                "qpruner": {"exists": True, "transformers_load": "PASS", "vllm_load": "FAIL"},
            },
            "issues": [
                "ModuleNotFoundError: No module named 'acl' from vllm_ascend/device_allocator/camem.py"
            ],
        },
    )
    write_json(
        artifacts / "tiny_qwen_serving_export_tiny_qwen3_serving_export_npu.json",
        {
            "status": "PASS",
            "storage_note": "Serving dense export preserves approximate weights but does not preserve compressed storage.",
            "exports": {
                "cap": {"status": "PASS", "load_check": "PASS", "targeted_compression_ratio": 19.0},
                "qpruner": {"status": "PASS", "load_check": "PASS", "average_bits": 3.789},
            },
        },
    )
    write_json(
        artifacts / "ascend_inference_probe.json",
        {
            "packages": {
                "peft": {"status": "FOUND", "version": "0.13"},
                "accelerate": {"status": "FOUND", "version": "1.0"},
                "vllm": {"status": "FOUND", "version": "0.10"},
            },
            "vllm_ascend": {"status": "FOUND", "modules": ["vllm_ascend"]},
        },
    )
    write_json(
        artifacts / "multicard_sync_summary.json",
        {"status": "PASS", "world_size": 8, "backend": "hccl"},
    )
    write_json(
        artifacts / "rankadaptor_lora_sync_summary.json",
        {"status": "PASS", "world_size": 8, "backend": "hccl"},
    )
    write_json(
        artifacts / "multicard_tiny_qwen_generate_tiny_qwen3_multicard_generate_npu.json",
        {"status": "PASS", "world_size": 8, "aggregate": {"distributed_reduce_consistent": True}},
    )
    write_json(
        artifacts / "multicard_tiny_qwen_bslora_finetune_tiny_qwen3_bslora_sync_npu.json",
        {
            "status": "PASS",
            "world_size": 8,
            "backend": "hccl",
            "aggregate": {"distributed_reduce_consistent": True, "adapter_sync_consistent": True},
        },
    )

    report = compat.build_report(demo_root)

    assert report["status"] == "ISSUES_FOUND"
    assert any(item["id"] == "vllm_acl_python_binding_missing" for item in report["issues"])
    assert any(item["id"] == "serving_dense_export_storage_tradeoff" for item in report["notes"])
    assert any(item["id"] == "rankadaptor_lora_sync_ready" for item in report["ready"])
    assert any(item["id"] == "multicard_generate_ready" for item in report["ready"])
    assert any(item["id"] == "tiny_qwen_bslora_sync_ready" for item in report["ready"])


def test_build_report_classifies_hidden_acl_pythonpath(tmp_path):
    compat = load_compat_module()
    demo_root = tmp_path / "demo"
    artifacts = demo_root / "artifacts"
    write_json(
        artifacts / "vllm_serving_probe_tiny_qwen3_serving_export_npu.json",
        {
            "status": "FAIL",
            "vllm_available": True,
            "exports": {
                "cap": {"exists": True, "transformers_load": "PASS", "vllm_load": "FAIL"},
                "qpruner": {"exists": True, "transformers_load": "PASS", "vllm_load": "FAIL"},
            },
            "issues": ["ModuleNotFoundError: No module named 'acl'"],
            "acl_diagnostics": {
                "import_acl": "MISSING",
                "pythonpath": "/mnt/nvme/622/tidal-ai",
                "candidate_pythonpath_entries": ["/usr/local/Ascend/cann-8.5.1/python/site-packages"],
            },
        },
    )
    write_json(
        artifacts / "ascend_inference_probe.json",
        {
            "packages": {
                "torch": {"status": "FOUND"},
                "torch_npu": {"status": "FOUND"},
                "transformers": {"status": "FOUND"},
            }
        },
    )
    write_json(artifacts / "multicard_sync_summary.json", {"status": "PASS", "world_size": 8, "backend": "hccl"})

    report = compat.build_report(demo_root)

    hidden_acl_issue = next(item for item in report["issues"] if item["id"] == "vllm_acl_pythonpath_hidden")
    assert "PYTHONPATH" in hidden_acl_issue["title"]
    assert "/usr/local/Ascend/cann-8.5.1/python/site-packages" in hidden_acl_issue["evidence"]


def test_build_report_classifies_vllm_memory_pressure(tmp_path):
    compat = load_compat_module()
    demo_root = tmp_path / "demo"
    artifacts = demo_root / "artifacts"
    write_json(
        artifacts / "vllm_serving_probe_tiny_qwen3_serving_export_npu.json",
        {
            "status": "FAIL",
            "vllm_available": True,
            "gpu_memory_utilization": 0.05,
            "exports": {
                "cap": {"exists": True, "transformers_load": "PASS", "vllm_load": "FAIL"},
                "qpruner": {"exists": True, "transformers_load": "PASS", "vllm_load": "FAIL"},
            },
            "issues": ["ValueError: Free memory on device (5.06/60.96 GiB) on startup is less than desired GPU memory"],
            "issue_classes": [{"id": "vllm_device_memory_pressure"}],
            "acl_diagnostics": {
                "import_acl": "FOUND",
                "acl_origin": "/usr/local/Ascend/cann-8.5.1/python/site-packages/acl.so",
            },
        },
    )
    write_json(
        artifacts / "ascend_inference_probe.json",
        {
            "packages": {
                "torch": {"status": "FOUND"},
                "torch_npu": {"status": "FOUND"},
                "transformers": {"status": "FOUND"},
            }
        },
    )
    write_json(artifacts / "multicard_sync_summary.json", {"status": "PASS", "world_size": 8, "backend": "hccl"})

    report = compat.build_report(demo_root)

    memory_issue = next(item for item in report["issues"] if item["id"] == "vllm_device_memory_pressure")
    assert "HBM" in memory_issue["title"]
    assert "acl.so" in memory_issue["evidence"]


def test_build_report_classifies_vllm_attention_selector_api_mismatch(tmp_path):
    compat = load_compat_module()
    demo_root = tmp_path / "demo"
    artifacts = demo_root / "artifacts"
    write_json(
        artifacts / "vllm_serving_probe_tiny_qwen3_serving_export_npu.json",
        {
            "status": "FAIL",
            "vllm_available": True,
            "gpu_memory_utilization": 0.05,
            "exports": {
                "cap": {"exists": True, "transformers_load": "PASS", "vllm_load": "FAIL"},
                "qpruner": {"exists": True, "transformers_load": "PASS", "vllm_load": "FAIL"},
            },
            "issues": [
                "TypeError: AttentionSelectorConfig.__new__() got an unexpected keyword argument "
                "'use_per_head_v1_attention'"
            ],
            "issue_classes": [{"id": "vllm_attention_selector_api_mismatch"}],
            "acl_diagnostics": {
                "import_acl": "FOUND",
                "acl_origin": "/usr/local/Ascend/cann-8.5.1/python/site-packages/acl.so",
            },
            "package_diagnostics": {
                "vllm": {"version": "0.18.0+empty", "file": "/vllm-workspace/vllm/vllm/__init__.py"},
                "vllm-ascend": {"version": "0.1.dev2924+ge5f7e2f43", "file": "/vllm-workspace/vllm-ascend"},
            },
            "api_diagnostics": {
                "status": "MISMATCH",
                "missing_patch_config_fields": ["use_per_head_quant_scales"],
                "missing_patch_get_attn_backend_parameters": ["use_per_head_quant_scales", "num_heads"],
            },
        },
    )
    write_json(
        artifacts / "ascend_inference_probe.json",
        {
            "packages": {
                "torch": {"status": "FOUND"},
                "torch_npu": {"status": "FOUND"},
                "transformers": {"status": "FOUND"},
            }
        },
    )
    write_json(artifacts / "multicard_sync_summary.json", {"status": "PASS", "world_size": 8, "backend": "hccl"})

    report = compat.build_report(demo_root)

    mismatch = next(item for item in report["issues"] if item["id"] == "vllm_attention_selector_api_mismatch")
    assert "API" in mismatch["title"]
    assert "0.18.0+empty" in mismatch["evidence"]
    assert "0.1.dev2924" in mismatch["evidence"]
    assert "use_per_head_quant_scales" in mismatch["evidence"]
    assert "num_heads" in mismatch["evidence"]


def test_build_report_marks_selector_shim_vllm_ready(tmp_path):
    compat = load_compat_module()
    demo_root = tmp_path / "demo"
    artifacts = demo_root / "artifacts"
    write_json(
        artifacts / "vllm_serving_probe_tiny_qwen3_serving_export_npu.json",
        {
            "status": "FAIL",
            "issue_classes": [{"id": "vllm_attention_selector_api_mismatch"}],
            "issues": ["TypeError: get_attn_backend() got an unexpected keyword argument"],
            "api_diagnostics": {
                "status": "MISMATCH",
                "missing_patch_config_fields": ["use_per_head_quant_scales"],
                "missing_patch_get_attn_backend_parameters": ["use_per_head_quant_scales", "num_heads"],
            },
            "package_diagnostics": {
                "vllm": {"version": "0.18.0+empty"},
                "vllm-ascend": {"version": "0.1.dev2924+ge5f7e2f43"},
            },
        },
    )
    write_json(
        artifacts / "vllm_serving_probe_tiny_qwen3_serving_export_npu_selector_shim.json",
        {
            "status": "PASS",
            "preload_vllm_ascend_patch": True,
            "preload_vllm_ascend_selector_shim": True,
            "export_run_label": "tiny_qwen3_serving_export_npu",
            "exports": {
                "cap": {"exists": True, "transformers_load": "PASS", "vllm_load": "PASS"},
                "qpruner": {"exists": True, "transformers_load": "PASS", "vllm_load": "PASS"},
            },
        },
    )
    write_json(
        artifacts / "ascend_inference_probe.json",
        {
            "packages": {
                "torch": {"status": "FOUND"},
                "torch_npu": {"status": "FOUND"},
                "transformers": {"status": "FOUND"},
            }
        },
    )
    write_json(artifacts / "multicard_sync_summary.json", {"status": "PASS", "world_size": 8, "backend": "hccl"})

    report = compat.build_report(demo_root)

    assert not any(item["id"] == "vllm_attention_selector_api_mismatch" for item in report["issues"])
    assert any(item["id"] == "vllm_selector_shim_ready" for item in report["ready"])
    note = next(item for item in report["notes"] if item["id"] == "vllm_attention_selector_api_mismatch_mitigated")
    assert "project-local selector shim" in note["title"]
    assert "use_per_head_quant_scales" in note["evidence"]
    assert report["sources"]["vllm_selector_shim_probe"] == "PASS"


def test_build_report_marks_torch_serving_fallback_ready(tmp_path):
    compat = load_compat_module()
    demo_root = tmp_path / "demo"
    artifacts = demo_root / "artifacts"
    write_json(
        artifacts / "torch_serving_fallback_tiny_qwen3_serving_fallback_npu.json",
        {
            "status": "PASS",
            "backend": "torch_generate",
            "device": "npu",
            "summary": {"best_method": "qpruner", "best_tokens_per_s": 456.0},
            "exports": {
                "cap": {"status": "PASS", "tokens_per_s": 432.0, "peak_mem_mb": 40.0},
                "qpruner": {"status": "PASS", "tokens_per_s": 456.0, "peak_mem_mb": 42.0},
            },
        },
    )
    write_json(
        artifacts / "ascend_inference_probe.json",
        {
            "packages": {
                "torch": {"status": "FOUND"},
                "torch_npu": {"status": "FOUND"},
                "transformers": {"status": "FOUND"},
            }
        },
    )
    write_json(artifacts / "multicard_sync_summary.json", {"status": "PASS", "world_size": 8, "backend": "hccl"})

    report = compat.build_report(demo_root)

    ready = next(item for item in report["ready"] if item["id"] == "torch_serving_fallback_ready")
    assert "torch_generate" in ready["evidence"]
    assert "456.0" in ready["evidence"]


def test_build_report_marks_vllm_serving_benchmark_ready(tmp_path):
    compat = load_compat_module()
    demo_root = tmp_path / "demo"
    artifacts = demo_root / "artifacts"
    write_json(
        artifacts / "vllm_serving_benchmark_tiny_qwen3_serving_vllm_metadata_shim_npu.json",
        {
            "status": "PASS",
            "backend": "vllm_ascend_generate",
            "summary": {
                "best_method": "qpruner",
                "best_tokens_per_s": 222.0,
                "cap_vs_baseline_speedup": 1.05,
                "qpruner_vs_baseline_speedup": 1.11,
            },
            "exports": {
                "baseline": {"status": "PASS", "tokens_per_s": 200.0, "peak_mem_mb": 512.0},
                "cap": {"status": "PASS", "tokens_per_s": 210.0, "peak_mem_mb": 512.0},
                "qpruner": {"status": "PASS", "tokens_per_s": 222.0, "peak_mem_mb": 512.0},
            },
        },
    )
    write_json(
        artifacts / "ascend_inference_probe.json",
        {
            "packages": {
                "torch": {"status": "FOUND"},
                "torch_npu": {"status": "FOUND"},
                "transformers": {"status": "FOUND"},
            }
        },
    )
    write_json(artifacts / "multicard_sync_summary.json", {"status": "PASS", "world_size": 8, "backend": "hccl"})

    report = compat.build_report(demo_root)

    ready = next(item for item in report["ready"] if item["id"] == "vllm_serving_benchmark_ready")
    assert "vllm_ascend_generate" in ready["evidence"]
    assert "222.0" in ready["evidence"]
    assert "qpruner_vs_baseline_speedup=1.11" in ready["evidence"]
    assert report["sources"]["vllm_serving_benchmark"] == "PASS"


def test_build_report_marks_parallel_suite_ready(tmp_path):
    compat = load_compat_module()
    demo_root = tmp_path / "demo"
    artifacts = demo_root / "artifacts"
    write_json(
        artifacts / "multicard_parallel_suite_demo_parallel_suite_npu.json",
        {
            "status": "PASS",
            "world_size": 4,
            "cards": [0, 1, 2, 3],
            "launched_synchronously": True,
            "start_window_s": 0.084,
            "task_counts": {"workflow_cap": 1, "workflow_qpruner": 1, "rankadaptor": 1},
            "aggregate": {"pass_count": 4, "fail_count": 0, "best_tokens_per_s": 456.0},
        },
    )
    write_json(
        artifacts / "ascend_inference_probe.json",
        {
            "packages": {
                "torch": {"status": "FOUND"},
                "torch_npu": {"status": "FOUND"},
                "transformers": {"status": "FOUND"},
            }
        },
    )
    write_json(artifacts / "multicard_sync_summary.json", {"status": "PASS", "world_size": 8, "backend": "hccl"})

    report = compat.build_report(demo_root)

    ready = next(item for item in report["ready"] if item["id"] == "multicard_parallel_suite_ready")
    assert "start_window_s=0.084" in ready["evidence"]
    assert "workflow_cap" in ready["evidence"]
    assert report["sources"]["multicard_parallel_suite"] == "PASS"


def test_build_report_marks_tiny_workflow_smokes_ready(tmp_path):
    compat = load_compat_module()
    demo_root = tmp_path / "demo"
    artifacts = demo_root / "artifacts"
    write_json(
        artifacts / "parallel_ascend_smokes.json",
        {
            "status": "PASS",
            "runs": [
                {
                    "status": "PASS",
                    "requested_workflow": "cap",
                    "workflows": ["cap"],
                    "device": "npu",
                    "npu_count": 1,
                    "cap": {
                        "status": "PASS",
                        "seconds": 0.416,
                        "packed_count": 6,
                        "module_devices": ["npu"],
                        "forward_output_device": "npu",
                    },
                },
                {
                    "status": "PASS",
                    "requested_workflow": "qpruner",
                    "workflows": ["qpruner"],
                    "device": "npu",
                    "npu_count": 1,
                    "qpruner": {
                        "status": "PASS",
                        "seconds": 0.352,
                        "quantized_count": 6,
                        "module_devices": ["npu"],
                        "forward_output_device": "npu",
                    },
                },
                {
                    "status": "PASS",
                    "requested_workflow": "rankadaptor",
                    "workflows": ["rankadaptor"],
                    "device": "npu",
                    "npu_count": 1,
                    "rankadaptor": {
                        "status": "PASS",
                        "seconds": 0.034,
                        "profile_count": 6,
                        "module_devices": ["npu"],
                        "peft_applied": False,
                    },
                },
            ],
        },
    )
    write_json(
        artifacts / "ascend_inference_probe.json",
        {
            "packages": {
                "torch": {"status": "FOUND"},
                "torch_npu": {"status": "FOUND"},
                "transformers": {"status": "FOUND"},
            }
        },
    )
    write_json(artifacts / "multicard_sync_summary.json", {"status": "PASS", "world_size": 8, "backend": "hccl"})

    report = compat.build_report(demo_root)

    ready = next(item for item in report["ready"] if item["id"] == "tiny_workflow_smokes_ready")
    assert "CAP/QPruner/RankAdaptor" in ready["title"]
    assert "cap=PASS" in ready["evidence"]
    assert "packed=6" in ready["evidence"]
    assert "qpruner=PASS" in ready["evidence"]
    assert "quantized=6" in ready["evidence"]
    assert "rankadaptor=PASS" in ready["evidence"]
    assert "profiles=6" in ready["evidence"]
    assert report["sources"]["parallel_ascend_smokes"] == "PASS"


def test_build_report_flags_missing_tiny_workflow_smokes(tmp_path):
    compat = load_compat_module()
    demo_root = tmp_path / "demo"
    artifacts = demo_root / "artifacts"
    write_json(
        artifacts / "ascend_inference_probe.json",
        {
            "packages": {
                "torch": {"status": "FOUND"},
                "torch_npu": {"status": "FOUND"},
                "transformers": {"status": "FOUND"},
            }
        },
    )
    write_json(artifacts / "multicard_sync_summary.json", {"status": "PASS", "world_size": 8, "backend": "hccl"})

    report = compat.build_report(demo_root)

    issue = next(item for item in report["issues"] if item["id"] == "tiny_workflow_smokes_missing_or_failed")
    assert "CAP/QPruner/RankAdaptor tiny workflow smokes" in issue["title"]
    assert "MISSING_EVIDENCE" in issue["evidence"]
    assert report["sources"]["parallel_ascend_smokes"] == "MISSING_EVIDENCE"


def test_build_report_classifies_qwen35_transformers_unsupported(tmp_path):
    compat = load_compat_module()
    demo_root = tmp_path / "demo"
    artifacts = demo_root / "artifacts"
    write_json(
        artifacts / "ascend_inference_qwen35_08b_torch_npu.json",
        {
            "status": "BACKEND_ERROR",
            "model_id": "Qwen/Qwen3.5-0.8B",
            "model_path": "/mnt/nvme/622/models/Qwen3.5-0.8B",
            "backend": "torch",
            "error_type": "ValueError",
            "error": "The checkpoint you are trying to load has model type `qwen3_5` but Transformers does not recognize this architecture.",
        },
    )
    write_json(
        artifacts / "model_snapshot_qwen35_08b.json",
        {
            "status": "FOUND",
            "model_id": "Qwen/Qwen3.5-0.8B",
            "model_path": "/mnt/nvme/622/models/Qwen3.5-0.8B",
            "parameter_mb": 1666.014,
        },
    )
    write_json(
        artifacts / "ascend_inference_probe.json",
        {
            "packages": {
                "torch": {"status": "FOUND"},
                "torch_npu": {"status": "FOUND"},
                "transformers": {"status": "FOUND", "version": "4.57.6"},
            }
        },
    )
    write_json(artifacts / "multicard_sync_summary.json", {"status": "PASS", "world_size": 8, "backend": "hccl"})

    report = compat.build_report(demo_root)

    issue = next(item for item in report["issues"] if item["id"] == "qwen35_transformers_unsupported")
    assert "Qwen3.5" in issue["title"]
    assert "qwen3_5" in issue["evidence"]
    assert "4.57.6" in issue["evidence"]


def test_build_report_marks_qwen3_torch_npu_ready(tmp_path):
    compat = load_compat_module()
    demo_root = tmp_path / "demo"
    artifacts = demo_root / "artifacts"
    write_json(
        artifacts / "model_snapshot_qwen3_06b.json",
        {
            "status": "DOWNLOADED",
            "provider": "modelscope",
            "model_id": "Qwen/Qwen3-0.6B",
            "model_path": "/mnt/nvme/622/models/Qwen3-0.6B",
            "parameter_mb": 1433.659,
        },
    )
    write_json(
        artifacts / "ascend_inference_qwen3_06b_torch_npu.json",
        {
            "status": "PASS",
            "model_id": "Qwen/Qwen3-0.6B",
            "backend": "torch",
            "device": "npu",
            "tokens_per_s": 11.923,
            "latency_ms": 1341.941,
            "peak_mem_mb": 1159.8,
        },
    )
    write_json(
        artifacts / "ascend_inference_probe.json",
        {
            "packages": {
                "torch": {"status": "FOUND"},
                "torch_npu": {"status": "FOUND"},
                "transformers": {"status": "FOUND", "version": "4.57.6"},
            }
        },
    )
    write_json(artifacts / "multicard_sync_summary.json", {"status": "PASS", "world_size": 8, "backend": "hccl"})

    report = compat.build_report(demo_root)

    ready = next(item for item in report["ready"] if item["id"] == "qwen3_torch_npu_ready")
    assert "Qwen3-0.6B" in ready["title"]
    assert "tokens_per_s=11.923" in ready["evidence"]
    assert report["sources"]["qwen3_inference"] == "PASS"
    assert report["sources"]["qwen3_snapshot"] == "DOWNLOADED"


def test_build_report_marks_multicard_qwen_inference_ready(tmp_path):
    compat = load_compat_module()
    demo_root = tmp_path / "demo"
    artifacts = demo_root / "artifacts"
    write_json(
        artifacts / "multicard_qwen_inference_qwen3_06b_npu.json",
        {
            "status": "PASS",
            "model_id": "Qwen/Qwen3-0.6B",
            "world_size": 8,
            "backend": "hccl",
            "aggregate": {
                "distributed_reduce_consistent": True,
                "tokens_per_s_total": 95.384,
                "generated_tokens_total": 128,
                "peak_mem_mb_total": 9278.4,
                "pass_count": 8,
            },
        },
    )
    write_json(
        artifacts / "ascend_inference_probe.json",
        {
            "packages": {
                "torch": {"status": "FOUND"},
                "torch_npu": {"status": "FOUND"},
                "transformers": {"status": "FOUND"},
            }
        },
    )
    write_json(artifacts / "multicard_sync_summary.json", {"status": "PASS", "world_size": 8, "backend": "hccl"})

    report = compat.build_report(demo_root)

    ready = next(item for item in report["ready"] if item["id"] == "multicard_qwen_inference_ready")
    assert "Qwen3-0.6B" in ready["title"]
    assert "world_size=8" in ready["evidence"]
    assert "tokens_per_s_total=95.384" in ready["evidence"]
    assert report["sources"]["multicard_qwen_inference"] == "PASS"


def test_build_report_marks_multicard_qwen_lora_finetune_ready(tmp_path):
    compat = load_compat_module()
    demo_root = tmp_path / "demo"
    artifacts = demo_root / "artifacts"
    write_json(
        artifacts / "multicard_qwen_lora_finetune_qwen3_06b_lora_sync_npu.json",
        {
            "status": "PASS",
            "model_id": "Qwen/Qwen3-0.6B",
            "world_size": 8,
            "backend": "hccl",
            "target_module_limit": 2,
            "target_module_count": 2,
            "targeted_modules_total": 196,
            "trainable_adapter_params": 4096,
            "aggregate": {
                "distributed_reduce_consistent": True,
                "adapter_sync_consistent": True,
                "initial_loss_avg": 12.5,
                "final_loss_avg": 12.1,
                "loss_delta_avg": -0.4,
                "peak_mem_mb_total": 9600.0,
                "pass_count": 8,
            },
        },
    )
    write_json(
        artifacts / "ascend_inference_probe.json",
        {
            "packages": {
                "torch": {"status": "FOUND"},
                "torch_npu": {"status": "FOUND"},
                "transformers": {"status": "FOUND"},
            }
        },
    )

    report = compat.build_report(demo_root)

    ready = next(item for item in report["ready"] if item["id"] == "multicard_qwen_lora_finetune_ready")
    assert "Qwen3-0.6B" in ready["title"]
    assert "world_size=8" in ready["evidence"]
    assert "target_modules=2/196" in ready["evidence"]
    assert "loss_delta_avg=-0.4" in ready["evidence"]
    assert "adapter_sync=True" in ready["evidence"]
    assert report["sources"]["multicard_qwen_lora_finetune"] == "PASS"


def test_build_report_marks_qwen_compression_generate_ready(tmp_path):
    compat = load_compat_module()
    demo_root = tmp_path / "demo"
    artifacts = demo_root / "artifacts"
    write_json(
        artifacts / "qwen_compression_generate_qwen3_06b_generate_npu.json",
        {
            "status": "PASS",
            "model_id": "Qwen/Qwen3-0.6B",
            "backend": "torch_generate",
            "device": "npu",
            "target_layer_limit": 2,
            "targeted_layers_total": 196,
            "baseline": {"tokens_per_s": 66.667, "targeted_layers": 2},
            "cap": {"latency_speedup": 1.091, "targeted_compression_ratio": 24.0, "targeted_layers": 2},
            "qpruner": {"latency_speedup": 1.111, "average_bits": 4.0, "targeted_layers": 2},
        },
    )
    write_json(
        artifacts / "ascend_inference_probe.json",
        {
            "packages": {
                "torch": {"status": "FOUND"},
                "torch_npu": {"status": "FOUND"},
                "transformers": {"status": "FOUND"},
            }
        },
    )
    write_json(artifacts / "multicard_sync_summary.json", {"status": "PASS", "world_size": 8, "backend": "hccl"})

    report = compat.build_report(demo_root)

    ready = next(item for item in report["ready"] if item["id"] == "qwen_compression_generate_ready")
    assert "Qwen3-0.6B" in ready["title"]
    assert "target_layers=2/196" in ready["evidence"]
    assert "cap_speedup=1.091" in ready["evidence"]
    assert "qpruner_bits=4.0" in ready["evidence"]
    assert report["sources"]["qwen_compression_generate"] == "PASS"


def test_build_report_marks_qwen_compression_quality_ready(tmp_path):
    compat = load_compat_module()
    demo_root = tmp_path / "demo"
    artifacts = demo_root / "artifacts"
    write_json(
        artifacts / "qwen_compression_quality_qwen3_06b_quality_npu.json",
        {
            "status": "PASS",
            "model_id": "Qwen/Qwen3-0.6B",
            "backend": "torch_forward",
            "device": "npu",
            "target_layer_limit": 2,
            "targeted_layers_total": 196,
            "baseline": {"loss": 11.5, "tokens_per_s": 66.667, "targeted_layers": 2},
            "cap": {"loss_delta": 0.125, "targeted_compression_ratio": 24.0, "targeted_layers": 2},
            "qpruner": {"loss_delta": 0.0625, "average_bits": 4.0, "targeted_layers": 2},
        },
    )
    write_json(
        artifacts / "ascend_inference_probe.json",
        {
            "packages": {
                "torch": {"status": "FOUND"},
                "torch_npu": {"status": "FOUND"},
                "transformers": {"status": "FOUND"},
            }
        },
    )
    write_json(artifacts / "multicard_sync_summary.json", {"status": "PASS", "world_size": 8, "backend": "hccl"})

    report = compat.build_report(demo_root)

    ready = next(item for item in report["ready"] if item["id"] == "qwen_compression_quality_ready")
    assert "Qwen3-0.6B" in ready["title"]
    assert "target_layers=2/196" in ready["evidence"]
    assert "baseline_loss=11.5" in ready["evidence"]
    assert "cap_loss_delta=0.125" in ready["evidence"]
    assert "qpruner_loss_delta=0.0625" in ready["evidence"]
    assert report["sources"]["qwen_compression_quality"] == "PASS"


def test_build_report_marks_multicard_qwen_compression_generate_ready(tmp_path):
    compat = load_compat_module()
    demo_root = tmp_path / "demo"
    artifacts = demo_root / "artifacts"
    write_json(
        artifacts / "multicard_qwen_compression_generate_qwen3_06b_compression_generate_npu.json",
        {
            "status": "PASS",
            "model_id": "Qwen/Qwen3-0.6B",
            "world_size": 8,
            "backend": "hccl",
            "target_layer_limit": 2,
            "targeted_layers_total": 196,
            "aggregate": {
                "distributed_reduce_consistent": True,
                "baseline_tokens_per_s_total": 70.0,
                "cap_tokens_per_s_total": 210.0,
                "qpruner_tokens_per_s_total": 224.0,
                "peak_mem_mb_total": 27914.4,
                "pass_count": 8,
            },
        },
    )
    write_json(
        artifacts / "ascend_inference_probe.json",
        {
            "packages": {
                "torch": {"status": "FOUND"},
                "torch_npu": {"status": "FOUND"},
                "transformers": {"status": "FOUND"},
            }
        },
    )

    report = compat.build_report(demo_root)

    ready = next(item for item in report["ready"] if item["id"] == "multicard_qwen_compression_generate_ready")
    assert "Qwen3-0.6B" in ready["title"]
    assert "world_size=8" in ready["evidence"]
    assert "target_layers=2/196" in ready["evidence"]
    assert "cap_tokens_per_s_total=210.0" in ready["evidence"]
    assert "qpruner_tokens_per_s_total=224.0" in ready["evidence"]
    assert report["sources"]["multicard_qwen_compression_generate"] == "PASS"


def test_build_report_prefers_stronger_multicard_qwen_compression_generate_artifact(tmp_path):
    compat = load_compat_module()
    demo_root = tmp_path / "demo"
    artifacts = demo_root / "artifacts"
    write_json(
        artifacts / "multicard_qwen_compression_generate_qwen3_06b_compression_generate_npu.json",
        {
            "status": "PASS",
            "model_id": "Qwen/Qwen3-0.6B",
            "world_size": 8,
            "backend": "hccl",
            "target_layer_limit": 2,
            "targeted_layers_total": 196,
            "aggregate": {
                "distributed_reduce_consistent": True,
                "baseline_tokens_per_s_total": 70.0,
                "cap_tokens_per_s_total": 210.0,
                "qpruner_tokens_per_s_total": 224.0,
                "peak_mem_mb_total": 27914.4,
                "pass_count": 8,
            },
        },
    )
    write_json(
        artifacts / "multicard_qwen_compression_generate_qwen3_06b_compression_generate_8card_target8_long8_npu.json",
        {
            "status": "PASS",
            "model_id": "Qwen/Qwen3-0.6B",
            "world_size": 8,
            "backend": "hccl",
            "target_layer_limit": 8,
            "targeted_layers_total": 196,
            "aggregate": {
                "distributed_reduce_consistent": True,
                "baseline_tokens_per_s_total": 104.491,
                "cap_tokens_per_s_total": 208.332,
                "qpruner_tokens_per_s_total": 211.526,
                "peak_mem_mb_total": 30172.0,
                "pass_count": 8,
            },
        },
    )

    report = compat.build_report(demo_root)

    ready = next(item for item in report["ready"] if item["id"] == "multicard_qwen_compression_generate_ready")
    assert "target_layers=8/196" in ready["evidence"]
    assert "qpruner_tokens_per_s_total=211.526" in ready["evidence"]


def test_build_report_marks_multicard_qwen_compression_quality_ready(tmp_path):
    compat = load_compat_module()
    demo_root = tmp_path / "demo"
    artifacts = demo_root / "artifacts"
    write_json(
        artifacts / "multicard_qwen_compression_quality_qwen3_06b_quality_sync_npu.json",
        {
            "status": "PASS",
            "model_id": "Qwen/Qwen3-0.6B",
            "world_size": 8,
            "backend": "hccl",
            "target_layer_limit": 2,
            "targeted_layers_total": 196,
            "aggregate": {
                "distributed_reduce_consistent": True,
                "baseline_loss_avg": 11.5,
                "cap_loss_delta_avg": 0.125,
                "qpruner_loss_delta_avg": 0.0625,
                "peak_mem_mb_total": 27914.4,
                "pass_count": 8,
            },
        },
    )
    write_json(
        artifacts / "ascend_inference_probe.json",
        {
            "packages": {
                "torch": {"status": "FOUND"},
                "torch_npu": {"status": "FOUND"},
                "transformers": {"status": "FOUND"},
            }
        },
    )

    report = compat.build_report(demo_root)

    ready = next(item for item in report["ready"] if item["id"] == "multicard_qwen_compression_quality_ready")
    assert "Qwen3-0.6B" in ready["title"]
    assert "world_size=8" in ready["evidence"]
    assert "target_layers=2/196" in ready["evidence"]
    assert "baseline_loss_avg=11.5" in ready["evidence"]
    assert "cap_loss_delta_avg=0.125" in ready["evidence"]
    assert "qpruner_loss_delta_avg=0.0625" in ready["evidence"]
    assert report["sources"]["multicard_qwen_compression_quality"] == "PASS"


def test_write_compatibility_report_outputs_markdown(tmp_path):
    compat = load_compat_module()
    demo_root = tmp_path / "demo"
    write_json(demo_root / "artifacts" / "ascend_inference_probe.json", {"packages": {}, "vllm_ascend": {}})

    output = compat.write_compatibility_report(demo_root)

    text = output.read_text()
    assert output == demo_root / "reports" / "compatibility-issues.md"
    assert "TIDAL-AI Ascend 910B Compatibility Issues" in text
    assert "MISSING_EVIDENCE" in text
    assert (demo_root / "artifacts" / "compatibility_issues.json").exists()
