import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def load_summary_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "summarize_qwen_compression_generate_sweep.py"
    spec = importlib.util.spec_from_file_location("summarize_qwen_compression_generate_sweep", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_artifact(
    path: Path,
    *,
    run_label: str,
    target_limit: int,
    total_layers: int,
    baseline_ms: float,
    cap_ms: float,
    qpruner_ms: float,
) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "model_id": "Qwen/Qwen3-0.6B",
                "device": "npu",
                "dtype": "float16",
                "target_layer_limit": target_limit,
                "targeted_layers_total": total_layers,
                "run_label": run_label,
                "baseline": {
                    "status": "PASS",
                    "latency_ms": baseline_ms,
                    "tokens_per_s": 8 / baseline_ms * 1000,
                    "targeted_layers": target_limit,
                },
                "cap": {
                    "status": "PASS",
                    "latency_ms": cap_ms,
                    "tokens_per_s": 8 / cap_ms * 1000,
                    "latency_speedup": round(baseline_ms / cap_ms, 3),
                    "targeted_compression_ratio": 3.0,
                    "targeted_layers": target_limit,
                },
                "qpruner": {
                    "status": "PASS",
                    "latency_ms": qpruner_ms,
                    "tokens_per_s": 8 / qpruner_ms * 1000,
                    "latency_speedup": round(baseline_ms / qpruner_ms, 3),
                    "average_bits": 4.0,
                    "targeted_layers": target_limit,
                },
                "peak_mem_mb": 3489.3,
            },
            indent=2,
        )
    )


def test_markdown_summary_includes_qwen_generate_sweep_rows_and_best_take(tmp_path):
    summary = load_summary_module()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    l2 = artifacts / "qwen_compression_generate_qwen3_06b_generate_l2_npu.json"
    l4 = artifacts / "qwen_compression_generate_qwen3_06b_generate_l4_npu.json"
    write_artifact(
        l2,
        run_label="qwen3_06b_generate_l2_npu",
        target_limit=2,
        total_layers=196,
        baseline_ms=956.512,
        cap_ms=298.476,
        qpruner_ms=300.391,
    )
    write_artifact(
        l4,
        run_label="qwen3_06b_generate_l4_npu",
        target_limit=4,
        total_layers=196,
        baseline_ms=900.0,
        cap_ms=310.0,
        qpruner_ms=320.0,
    )

    text = summary.markdown_summary([l4, l2])

    assert "Qwen Compression Generate Sweep" in text
    assert "| qwen3_06b_generate_l2 | PASS | Qwen/Qwen3-0.6B | 2 / 196 |" in text
    assert "| qwen3_06b_generate_l4 | PASS | Qwen/Qwen3-0.6B | 4 / 196 |" in text
    assert "3.205" in text
    assert "4.000" in text
    assert "Best CAP speedup is 3.205x at 2/196 target layers" in text
    assert f"- `{l2}`" in text
    assert f"- `{l4}`" in text
    assert text.index("qwen3_06b_generate_l2") < text.index("qwen3_06b_generate_l4")


def test_cli_writes_qwen_generate_sweep_summary(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    output = tmp_path / "reports" / "qwen-compression-generate-sweep.md"
    artifact = artifacts / "qwen_compression_generate_qwen3_06b_generate_l2_npu.json"
    write_artifact(
        artifact,
        run_label="qwen3_06b_generate_l2_npu",
        target_limit=2,
        total_layers=196,
        baseline_ms=956.512,
        cap_ms=298.476,
        qpruner_ms=300.391,
    )
    script = Path(__file__).resolve().parents[1] / "scripts" / "summarize_qwen_compression_generate_sweep.py"

    proc = subprocess.run(
        [sys.executable, str(script), str(artifact), "--output", str(output)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert output.exists()
    assert "Qwen Compression Generate Sweep" in output.read_text()
