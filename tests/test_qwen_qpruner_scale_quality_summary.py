import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path


def load_summary_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "write_qwen_qpruner_scale_quality_summary.py"
    spec = importlib.util.spec_from_file_location("write_qwen_qpruner_scale_quality_summary", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_qpruner_point(
    path: Path,
    *,
    run_label: str,
    target_layer_limit: int,
    targeted_layers_total: int,
    average_bits: float,
    loss_delta: float,
    tokens_per_s: float,
    actual_targeted_layers: int | None = None,
) -> None:
    baseline_loss = 6.0
    actual_targeted_layers = actual_targeted_layers or target_layer_limit
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "backend": "torch_forward",
                "model_id": "Qwen/Qwen3-0.6B",
                "model_path": "/mnt/nvme/622/models/Qwen3-0.6B",
                "device": "npu",
                "dtype": "torch.float16",
                "target_layer_pattern": r"self_attn\.(q_proj|k_proj)$",
                "target_layer_limit": target_layer_limit,
                "targeted_layers_total": targeted_layers_total,
                "run_label": run_label,
                "baseline": {
                    "status": "PASS",
                    "loss": baseline_loss,
                    "latency_ms": 92.0,
                    "tokens_per_s": 420.0,
                    "targeted_layers": actual_targeted_layers,
                    "targeted_params": 1_000_000 * actual_targeted_layers,
                },
                "qpruner": {
                    "status": "PASS",
                    "loss": baseline_loss + loss_delta,
                    "loss_delta": loss_delta,
                    "latency_ms": 88.0,
                    "tokens_per_s": tokens_per_s,
                    "average_bits": average_bits,
                    "compression_time_s": 3.5,
                    "bitwidths": {"a": int(average_bits), "b": int(average_bits)},
                },
                "peak_mem_mb": 2048.0,
            },
            indent=2,
        )
    )


def test_scale_quality_summary_quantifies_coverage_memory_and_loss_derived_ppl(tmp_path):
    summary_mod = load_summary_module()
    artifacts = tmp_path / "demo" / "artifacts"
    low_bits = artifacts / "qwen_compression_quality_qwen3_06b_qpruner_sweep_layers4_bits4.json"
    mid_scale = artifacts / "qwen_compression_quality_qwen3_06b_qpruner_sweep_layers8_bits6.json"
    best_scale = artifacts / "qwen_compression_quality_qwen3_06b_qpruner_sweep_layers16_bits6.json"
    write_qpruner_point(
        low_bits,
        run_label="qwen3_06b_qpruner_sweep_layers4_bits4",
        target_layer_limit=4,
        targeted_layers_total=196,
        average_bits=4.0,
        loss_delta=1.25,
        tokens_per_s=430.0,
    )
    write_qpruner_point(
        mid_scale,
        run_label="qwen3_06b_qpruner_sweep_layers8_bits6",
        target_layer_limit=8,
        targeted_layers_total=196,
        average_bits=6.0,
        loss_delta=0.20,
        tokens_per_s=425.0,
    )
    write_qpruner_point(
        best_scale,
        run_label="qwen3_06b_qpruner_sweep_layers16_bits6",
        target_layer_limit=16,
        targeted_layers_total=196,
        average_bits=6.0,
        loss_delta=0.25,
        tokens_per_s=418.0,
    )

    report = summary_mod.build_summary([low_bits, mid_scale, best_scale], max_loss_delta=0.5)

    assert report["status"] == "PASS"
    assert report["max_target_layer_limit"] == 16
    assert report["targeted_layers_total"] == 196
    assert report["max_coverage_pct"] == 8.163
    assert report["best_under_loss_delta"]["run_label"] == "qwen3_06b_qpruner_sweep_layers16_bits6"
    assert report["best_under_loss_delta"]["memory_reduction_pct"] == 62.5
    assert report["best_under_loss_delta"]["coverage_pct"] == 8.163
    assert report["best_under_loss_delta"]["baseline_ppl"] == round(math.exp(6.0), 3)
    assert report["best_under_loss_delta"]["qpruner_ppl"] == round(math.exp(6.25), 3)
    assert "16/196 target layers" in report["readout"]
    assert "62.500% targeted memory reduction" in report["readout"]
    assert "loss-derived approximate PPL" in report["readout"]

    markdown = summary_mod.markdown(report)
    assert "# Qwen QPruner Scale-Quality Summary" in markdown
    assert "| qwen3_06b_qpruner_sweep_layers16_bits6 | PASS | 16 / 196 | 8.163% | 6.000 | 62.500% | 0.250 | 403.429 | 518.013 | 28.403% |" in markdown
    assert "Best scale-quality point under loss delta <= 0.500" in markdown
    assert "loss-derived approximate PPL" in markdown
    assert f"- `{best_scale}`" in markdown


def test_cli_writes_scale_quality_summary_artifacts(tmp_path):
    demo_root = tmp_path / "demo"
    artifact = demo_root / "artifacts" / "qwen_compression_quality_qwen3_06b_qpruner_sweep_layers16_bits6.json"
    write_qpruner_point(
        artifact,
        run_label="qwen3_06b_qpruner_sweep_layers16_bits6",
        target_layer_limit=16,
        targeted_layers_total=196,
        average_bits=6.0,
        loss_delta=0.25,
        tokens_per_s=418.0,
    )
    script = Path(__file__).resolve().parents[1] / "scripts" / "write_qwen_qpruner_scale_quality_summary.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--demo-root",
            str(demo_root),
            "--artifact-glob",
            "qwen_compression_quality_qwen3_06b_qpruner_sweep_layers*_bits*.json",
            "--max-loss-delta",
            "0.5",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary_json = demo_root / "artifacts" / "qwen_qpruner_scale_quality_summary.json"
    summary_md = demo_root / "reports" / "qwen-qpruner-scale-quality-summary.md"
    summary_csv = demo_root / "reports" / "qwen-qpruner-scale-quality-summary.csv"
    assert summary_json.exists()
    assert summary_md.exists()
    assert summary_csv.exists()
    payload = json.loads(summary_json.read_text())
    assert payload["status"] == "PASS"
    assert payload["best_under_loss_delta"]["coverage_pct"] == 8.163
    assert "Qwen QPruner Scale-Quality Summary" in summary_md.read_text()
    assert "run_label,status,target_layer_limit,targeted_layers_total,coverage_pct,average_bits" in summary_csv.read_text()


def test_cli_default_glob_includes_projected_layer_scale_artifacts(tmp_path):
    demo_root = tmp_path / "demo"
    old_artifact = demo_root / "artifacts" / "qwen_compression_quality_qwen3_06b_qpruner_sweep_layers128_bits8.json"
    projected_artifact = (
        demo_root
        / "artifacts"
        / "qwen_compression_quality_qwen3_06b_qpruner_sweep_proj_layers160_196_npu_layers196_bits8.json"
    )
    write_qpruner_point(
        old_artifact,
        run_label="qwen3_06b_qpruner_sweep_layers128_bits8",
        target_layer_limit=128,
        targeted_layers_total=196,
        average_bits=8.0,
        loss_delta=0.1,
        tokens_per_s=500.0,
        actual_targeted_layers=56,
    )
    write_qpruner_point(
        projected_artifact,
        run_label="qwen3_06b_qpruner_sweep_proj_layers160_196_npu_layers196_bits8",
        target_layer_limit=196,
        targeted_layers_total=196,
        average_bits=8.0,
        loss_delta=-0.02,
        tokens_per_s=480.0,
        actual_targeted_layers=196,
    )
    script = Path(__file__).resolve().parents[1] / "scripts" / "write_qwen_qpruner_scale_quality_summary.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--demo-root",
            str(demo_root),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads((demo_root / "artifacts" / "qwen_qpruner_scale_quality_summary.json").read_text())
    assert payload["max_target_layer_limit"] == 196
    assert payload["max_coverage_pct"] == 100.0
    assert payload["best_under_loss_delta"]["run_label"] == (
        "qwen3_06b_qpruner_sweep_proj_layers160_196_npu_layers196_bits8"
    )
    assert str(projected_artifact) in payload["inputs"]


def test_scale_quality_summary_uses_actual_targeted_layers_when_request_exceeds_matches(tmp_path):
    summary_mod = load_summary_module()
    artifacts = tmp_path / "demo" / "artifacts"
    requested_high = artifacts / "qwen_compression_quality_qwen3_06b_qpruner_sweep_layers128_bits8.json"
    write_qpruner_point(
        requested_high,
        run_label="qwen3_06b_qpruner_sweep_layers128_bits8",
        target_layer_limit=128,
        targeted_layers_total=196,
        average_bits=8.0,
        loss_delta=0.1,
        tokens_per_s=500.0,
        actual_targeted_layers=56,
    )

    report = summary_mod.build_summary([requested_high], max_loss_delta=0.5)

    point = report["points"][0]
    assert point["requested_target_layer_limit"] == 128
    assert point["target_layer_limit"] == 56
    assert point["coverage_pct"] == 28.571
    assert report["max_target_layer_limit"] == 56
    assert report["max_coverage_pct"] == 28.571
    assert "56/196 target layers" in report["readout"]
