import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def load_inventory_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "write_model_inventory_report.py"
    spec = importlib.util.spec_from_file_location("write_model_inventory_report", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_model(path: Path, *, model_type: str = "qwen3", hidden_size: int = 64, weight_bytes: int = 1024) -> None:
    path.mkdir(parents=True)
    (path / "config.json").write_text(
        json.dumps({"model_type": model_type, "hidden_size": hidden_size, "architectures": ["Qwen3ForCausalLM"]})
    )
    (path / "tokenizer_config.json").write_text("{}")
    (path / "model.safetensors").write_bytes(b"0" * weight_bytes)


def test_model_inventory_classifies_tiny_fixture_and_missing_targets(tmp_path):
    inventory = load_inventory_module()
    write_model(tmp_path / "TinyQwen3-Offline", weight_bytes=512)

    report = inventory.build_report(
        roots=[tmp_path],
        target_model_ids=["Qwen/Qwen3-0.6B", "Qwen/Qwen2.5-0.5B-Instruct"],
        tiny_model_names=["TinyQwen3-Offline"],
    )

    assert report["status"] == "ONLY_TINY_FIXTURE"
    assert report["tiny_fixtures"][0]["status"] == "FOUND"
    assert report["tiny_fixtures"][0]["parameter_bytes"] == 512
    assert all(candidate["status"] == "MISSING" for candidate in report["target_candidates"])
    assert report["recommended_next_action"].startswith("Place a modern local model")


def test_default_targets_prioritize_latest_small_qwen():
    inventory = load_inventory_module()

    assert inventory.DEFAULT_TARGET_MODELS[0] == "Qwen/Qwen3.5-0.8B"
    assert "Qwen/Qwen3-0.6B" in inventory.DEFAULT_TARGET_MODELS
    assert "/mnt/nvme/622/models" in inventory.recommended_next_action("ONLY_TINY_FIXTURE")
    assert "Qwen/Qwen3.5-0.8B" in inventory.recommended_next_action("ONLY_TINY_FIXTURE")


def test_model_inventory_detects_real_target_snapshot(tmp_path):
    inventory = load_inventory_module()
    snapshot = tmp_path / "models--Qwen--Qwen3-0.6B" / "snapshots" / "abc"
    write_model(snapshot, hidden_size=1024, weight_bytes=2048)

    report = inventory.build_report(
        roots=[tmp_path],
        target_model_ids=["Qwen/Qwen3-0.6B"],
        tiny_model_names=["TinyQwen3-Offline"],
    )

    assert report["status"] == "TARGET_MODEL_FOUND"
    assert report["target_candidates"][0]["status"] == "FOUND"
    assert report["target_candidates"][0]["path"] == str(snapshot)
    assert report["target_candidates"][0]["source"] == "huggingface_cache"
    assert report["target_candidates"][0]["parameter_bytes"] == 2048


def test_write_model_inventory_report_outputs_json_and_markdown(tmp_path):
    inventory = load_inventory_module()
    roots = tmp_path / "models"
    write_model(roots / "TinyQwen3-Offline", weight_bytes=512)
    demo_root = tmp_path / "demo"

    output = inventory.write_model_inventory_report(
        demo_root=demo_root,
        roots=[roots],
        target_model_ids=["Qwen/Qwen3-0.6B"],
        tiny_model_names=["TinyQwen3-Offline"],
    )

    payload = json.loads((demo_root / "artifacts/model_inventory.json").read_text())
    assert output == demo_root / "reports/model-inventory.md"
    assert payload["status"] == "ONLY_TINY_FIXTURE"
    text = output.read_text()
    assert "Model Inventory" in text
    assert "TinyQwen3-Offline" in text
    assert "Qwen/Qwen3-0.6B" in text
    assert "ONLY_TINY_FIXTURE" in text


def test_model_inventory_script_runs_from_cli(tmp_path):
    roots = tmp_path / "models"
    write_model(roots / "TinyQwen3-Offline", weight_bytes=512)
    demo_root = tmp_path / "demo"
    script = Path(__file__).resolve().parents[1] / "scripts" / "write_model_inventory_report.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--demo-root",
            str(demo_root),
            "--root",
            str(roots),
            "--target-model-id",
            "Qwen/Qwen3-0.6B",
            "--tiny-model-name",
            "TinyQwen3-Offline",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=60,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "MODEL_INVENTORY" in proc.stdout
    assert (demo_root / "artifacts/model_inventory.json").exists()
