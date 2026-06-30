import importlib.util
import subprocess
import sys
from pathlib import Path


def load_chart_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "plot_qwen_qpruner_quality_memory_frontier.py"
    spec = importlib.util.spec_from_file_location("plot_qwen_qpruner_quality_memory_frontier", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


CSV_TEXT = """label,status,target_layer_limit,targeted_layers_total,requested_bits,actual_bits,target_memory_reduction_pct,loss_delta,tokens_per_s,bitwidth_mix
layers4_bits4,PASS,4,196,4.0,4.0,75.0,5.287,548.923,"2-bit:1, 4-bit:2, 8-bit:1"
layers4_bits6,PASS,4,196,6.0,5.333333333333333,66.667,-0.224,549.0,"4-bit:2, 8-bit:2"
layers4_bits8,PASS,4,196,8.0,8.0,50.0,-0.010,547.763,8-bit:4
layers8_bits4,PASS,8,196,4.0,4.0,75.0,6.148,557.845,4-bit:8
layers8_bits6,PASS,8,196,6.0,6.0,62.5,0.234,559.571,"4-bit:4, 8-bit:4"
layers8_bits8,PASS,8,196,8.0,8.0,50.0,-0.005,559.325,8-bit:8
layers16_bits4,PASS,16,196,4.0,4.0,75.0,7.751,557.0,4-bit:16
layers16_bits6,PASS,16,196,6.0,6.0,62.5,0.993,559.0,"4-bit:8, 8-bit:8"
layers16_bits8,PASS,16,196,8.0,8.0,50.0,-0.012,559.0,8-bit:16
"""


def test_frontier_chart_svg_marks_threshold_and_best_point(tmp_path):
    chart = load_chart_module()
    csv_path = tmp_path / "sweep.csv"
    csv_path.write_text(CSV_TEXT)

    rows = chart.load_rows(csv_path)
    best = chart.best_memory_quality_point(rows, max_loss_delta=0.5)
    svg = chart.render_svg(rows, best=best, max_loss_delta=0.5)

    assert best["label"] == "layers4_bits6"
    assert "<svg" in svg
    assert "QPruner quality-memory frontier" in svg
    assert "loss delta <= 0.500" in svg
    assert "Best: layers4_bits6" in svg
    assert "66.667% memory reduction" in svg
    assert "target layers 4" in svg
    assert "target layers 8" in svg
    assert "target layers 16" in svg
    assert "data-label=\"layers16_bits6\"" in svg


def test_frontier_chart_cli_writes_svg_and_markdown_embed(tmp_path):
    csv_path = tmp_path / "sweep.csv"
    csv_path.write_text(CSV_TEXT)
    output = tmp_path / "frontier.svg"
    markdown = tmp_path / "frontier.md"
    script = Path(__file__).resolve().parents[1] / "scripts" / "plot_qwen_qpruner_quality_memory_frontier.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--csv",
            str(csv_path),
            "--output",
            str(output),
            "--markdown",
            str(markdown),
            "--max-loss-delta",
            "0.5",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "QPRUNER_FRONTIER_CHART" in proc.stdout
    assert output.exists()
    assert markdown.exists()
    assert "Best: layers4_bits6" in output.read_text()
    assert "![QPruner quality-memory frontier](frontier.svg)" in markdown.read_text()
