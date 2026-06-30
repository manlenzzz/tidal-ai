import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def load_summary_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "summarize_tiny_qwen_compression_sweep.py"
    spec = importlib.util.spec_from_file_location("summarize_tiny_qwen_compression_sweep", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_artifact(path: Path, *, label: str, batch: int, seq: int, base: float, cap: float, qpruner: float) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "batch_size": batch,
                "seq_len": seq,
                "baseline": {"latency_ms": base, "tokens_per_s": batch * seq / base * 1000},
                "cap": {
                    "status": "PASS",
                    "latency_ms": cap,
                    "tokens_per_s": batch * seq / cap * 1000,
                    "latency_speedup": round(base / cap, 3),
                    "targeted_compression_ratio": 19.0,
                    "loss_delta": 0.01,
                },
                "qpruner": {
                    "status": "PASS",
                    "latency_ms": qpruner,
                    "tokens_per_s": batch * seq / qpruner * 1000,
                    "latency_speedup": round(base / qpruner, 3),
                    "average_bits": 3.789,
                    "loss_delta": 0.02,
                },
                "inference_cache_enabled": True,
                "cap_cached": {
                    "status": "PASS",
                    "latency_ms": cap - 0.2,
                    "latency_speedup": round(base / (cap - 0.2), 3),
                    "cache_modules": 7,
                },
                "qpruner_cached": {
                    "status": "PASS",
                    "latency_ms": qpruner - 0.7,
                    "latency_speedup": round(base / (qpruner - 0.7), 3),
                    "cache_modules": 7,
                },
                "peak_mem_mb": 25.0,
                "run_label": label,
            },
            indent=2,
        )
    )


def test_markdown_summary_includes_sweep_rows_and_bottleneck_note(tmp_path):
    summary = load_summary_module()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    write_artifact(artifacts / "b1s16.json", label="b1s16", batch=1, seq=16, base=3.5, cap=4.3, qpruner=4.4)
    write_artifact(artifacts / "b4s64.json", label="b4s64", batch=4, seq=64, base=3.6, cap=4.1, qpruner=4.0)

    text = summary.markdown_summary([artifacts / "b1s16.json", artifacts / "b4s64.json"])

    assert "TinyQwen Compression Sweep" in text
    assert "| b1s16 | 1 | 16 |" in text
    assert "| b4s64 | 4 | 64 |" in text
    assert "19.000x" in text
    assert "3.789" in text
    assert "CAP cached speedup" in text
    assert "QPruner cached speedup" in text
    assert "cached compressed path is latency-positive" in text


def test_cli_writes_summary_markdown(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    output = tmp_path / "summary.md"
    write_artifact(artifacts / "b1s16.json", label="b1s16", batch=1, seq=16, base=3.5, cap=4.3, qpruner=4.4)
    script = Path(__file__).resolve().parents[1] / "scripts" / "summarize_tiny_qwen_compression_sweep.py"

    proc = subprocess.run(
        [sys.executable, str(script), str(artifacts / "b1s16.json"), "--output", str(output)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert output.exists()
    assert "TinyQwen Compression Sweep" in output.read_text()
