import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def load_summary_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "summarize_qwen_compression_quality_memory_sweep.py"
    spec = importlib.util.spec_from_file_location("summarize_qwen_compression_quality_memory_sweep", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_quality_artifact(
    path: Path,
    *,
    run_label: str,
    qpruner_bits: float,
    qpruner_loss_delta: float,
    qpruner_tokens_per_s: float,
    qpruner_memory_bits: int,
    bitwidths: dict[str, int],
) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "backend": "torch_forward",
                "model_id": "Qwen/Qwen3-0.6B",
                "device": "npu",
                "dtype": "torch.float16",
                "target_layer_limit": 4,
                "targeted_layers_total": 196,
                "target_layer_pattern": r"self_attn\.(q_proj|k_proj)$",
                "run_label": run_label,
                "baseline": {
                    "status": "PASS",
                    "loss": 6.797,
                    "latency_ms": 78.0,
                    "tokens_per_s": 500.0,
                    "targeted_params": 6_291_456,
                    "targeted_layers": 4,
                },
                "cap": {
                    "status": "PASS",
                    "loss_delta": 0.156,
                    "tokens_per_s": 504.0,
                    "targeted_compression_ratio": 6.0,
                },
                "wanda": {
                    "status": "PASS",
                    "loss_delta": -0.079,
                    "tokens_per_s": 501.0,
                    "targeted_param_reduction_pct": 50.0,
                },
                "qpruner": {
                    "status": "PASS",
                    "loss_delta": qpruner_loss_delta,
                    "tokens_per_s": qpruner_tokens_per_s,
                    "average_bits": qpruner_bits,
                    "memory_bits": qpruner_memory_bits,
                    "bitwidths": bitwidths,
                },
                "peak_mem_mb": 2430.4,
            },
            indent=2,
        )
    )


def test_markdown_summary_shows_quality_memory_frontier(tmp_path):
    summary = load_summary_module()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    l4 = artifacts / "qwen_compression_quality_qwen3_06b_quality_pattern_npu.json"
    l6 = artifacts / "qwen_compression_quality_qwen3_06b_quality_pattern_bits6_npu.json"
    l8 = artifacts / "qwen_compression_quality_qwen3_06b_quality_pattern_bits8_npu.json"
    write_quality_artifact(
        l4,
        run_label="qwen3_06b_quality_pattern_npu",
        qpruner_bits=4.0,
        qpruner_loss_delta=5.287,
        qpruner_tokens_per_s=505.468,
        qpruner_memory_bits=25_165_824,
        bitwidths={"a": 2, "b": 4, "c": 4, "d": 8},
    )
    write_quality_artifact(
        l6,
        run_label="qwen3_06b_quality_pattern_bits6_npu",
        qpruner_bits=5.333333333333333,
        qpruner_loss_delta=-0.224,
        qpruner_tokens_per_s=526.190,
        qpruner_memory_bits=33_554_432,
        bitwidths={"a": 4, "b": 4, "c": 8, "d": 8},
    )
    write_quality_artifact(
        l8,
        run_label="qwen3_06b_quality_pattern_bits8_npu",
        qpruner_bits=8.0,
        qpruner_loss_delta=-0.010,
        qpruner_tokens_per_s=582.228,
        qpruner_memory_bits=50_331_648,
        bitwidths={"a": 8, "b": 8, "c": 8, "d": 8},
    )

    text = summary.markdown_summary([l8, l4, l6], max_loss_delta=0.5)

    assert "Qwen Compression Quality-Memory Sweep" in text
    assert "| qwen3_06b_quality_pattern | PASS | 4 / 196 | 4.000 | 75.000% | 5.287 |" in text
    assert "| qwen3_06b_quality_pattern_bits6 | PASS | 4 / 196 | 5.333 | 66.667% | -0.224 |" in text
    assert "| qwen3_06b_quality_pattern_bits8 | PASS | 4 / 196 | 8.000 | 50.000% | -0.010 |" in text
    assert "Best memory-quality point under loss delta <= 0.500 is qwen3_06b_quality_pattern_bits6" in text
    assert "4-bit saves the most memory but exceeds the quality threshold" in text
    assert "2-bit:1, 4-bit:2, 8-bit:1" in text
    assert f"- `{l4}`" in text
    assert text.index("qwen3_06b_quality_pattern |") < text.index("qwen3_06b_quality_pattern_bits6")


def test_cli_writes_quality_memory_sweep_summary(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    output = tmp_path / "reports" / "qwen-compression-quality-memory-sweep.md"
    artifact = artifacts / "qwen_compression_quality_qwen3_06b_quality_pattern_bits6_npu.json"
    write_quality_artifact(
        artifact,
        run_label="qwen3_06b_quality_pattern_bits6_npu",
        qpruner_bits=5.333333333333333,
        qpruner_loss_delta=-0.224,
        qpruner_tokens_per_s=526.190,
        qpruner_memory_bits=33_554_432,
        bitwidths={"a": 4, "b": 8},
    )
    script = Path(__file__).resolve().parents[1] / "scripts" / "summarize_qwen_compression_quality_memory_sweep.py"

    proc = subprocess.run(
        [sys.executable, str(script), str(artifact), "--output", str(output), "--max-loss-delta", "0.5"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert output.exists()
    assert "Qwen Compression Quality-Memory Sweep" in output.read_text()
