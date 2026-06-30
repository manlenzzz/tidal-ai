import importlib.util
import json
from pathlib import Path


def load_writer_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "write_current_npu_occupancy.py"
    assert script.exists(), f"missing occupancy writer: {script}"
    spec = importlib.util.spec_from_file_location("write_current_npu_occupancy", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cli_writes_normalized_npu_occupancy_artifact(tmp_path):
    module = load_writer_module()
    demo_root = tmp_path / "demo"

    rc = module.main(
        [
            "--demo-root",
            str(demo_root),
            "--occupied-cards",
            "8",
            "--total-cards",
            "8",
            "--container-id",
            "0f0d94d365a0",
            "--container-name",
            "tdqs_qwen3-30b-a3b",
            "--container-image",
            "quay.io/ascend/vllm-ascend:v0.20.2rc1",
            "--service-model",
            "Qwen3-30B-A3B",
            "--service-command",
            "vllm serve /root/.cache --served-model-name Qwen3-30B-A3B",
            "--tensor-parallel-size",
            "8",
        ]
    )

    payload = json.loads((demo_root / "artifacts" / "current_npu_occupancy.json").read_text())
    assert rc == 0
    assert payload["status"] == "BLOCKED"
    assert payload["occupied_cards"] == 8
    assert payload["total_cards"] == 8
    assert payload["container"]["name"] == "tdqs_qwen3-30b-a3b"
    assert payload["service"]["model"] == "Qwen3-30B-A3B"
    assert payload["service"]["tensor_parallel_size"] == 8
    assert payload["policy"] == "do_not_kill_unrelated_container"


def test_cli_marks_zero_occupied_cards_as_free_by_default(tmp_path):
    module = load_writer_module()
    demo_root = tmp_path / "demo"

    rc = module.main(
        [
            "--demo-root",
            str(demo_root),
            "--occupied-cards",
            "0",
            "--total-cards",
            "8",
            "--reason",
            "no_running_npu_processes_found",
            "--container-id",
            "3ee",
            "--container-name",
            "tidal-ai",
            "--container-image",
            "project-container",
            "--service-model",
            "none",
            "--service-command",
            "npu-smi reports no running processes",
            "--tensor-parallel-size",
            "0",
        ]
    )

    payload = json.loads((demo_root / "artifacts" / "current_npu_occupancy.json").read_text())
    assert rc == 0
    assert payload["status"] == "FREE"
    assert payload["occupied_cards"] == 0
