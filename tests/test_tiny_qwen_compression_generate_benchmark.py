import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import torch


def load_fixture_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "create_tiny_qwen_fixture.py"
    spec = importlib.util.spec_from_file_location("create_tiny_qwen_fixture", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_generate_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "tiny_qwen_compression_generate_benchmark.py"
    spec = importlib.util.spec_from_file_location("tiny_qwen_compression_generate_benchmark", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_default_prompts_are_offline_and_nonempty():
    bench = load_generate_module()

    assert len(bench.DEFAULT_PROMPTS) >= 2
    assert all(isinstance(prompt, str) and prompt for prompt in bench.DEFAULT_PROMPTS)


def test_markdown_report_marks_serving_dense_export():
    bench = load_generate_module()

    text = bench.markdown_report(
        {
            "model_id": "TinyQwen3-Offline",
            "model_path": "/models/tiny",
            "device": "npu",
            "dtype": "torch.float16",
            "download_attempted": False,
            "serving_dense_export": True,
            "baseline": {"status": "PASS", "latency_ms": 10.0, "tokens_per_s": 100.0},
            "cap": {
                "status": "PASS",
                "latency_ms": 9.0,
                "tokens_per_s": 111.0,
                "latency_speedup": 1.111,
                "targeted_compression_ratio": 19.0,
                "cache_modules": 7,
                "exported_dense_linears": 7,
            },
            "qpruner": {
                "status": "PASS",
                "latency_ms": 8.0,
                "tokens_per_s": 125.0,
                "latency_speedup": 1.25,
                "average_bits": 3.789,
                "cache_modules": 7,
                "exported_dense_linears": 7,
            },
        }
    )

    assert "Serving dense export" in text
    assert "| Exported dense linears | - | 7 | 7 |" in text


class RecordingTokenizer:
    def __init__(self):
        self.padding_side = "right"
        self.padding_sides_seen = []

    def __call__(self, prompts, return_tensors=None, padding=True, truncation=True, max_length=128):
        assert return_tensors == "pt"
        assert padding is True
        assert truncation is True
        self.padding_sides_seen.append(self.padding_side)
        input_ids = torch.arange(len(prompts) * 4, dtype=torch.long).reshape(len(prompts), 4)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }


class FakeGenerateModel(torch.nn.Module):
    def generate(self, input_ids, attention_mask=None, max_new_tokens=2, do_sample=False):
        suffix = torch.ones((input_ids.shape[0], max_new_tokens), dtype=input_ids.dtype, device=input_ids.device)
        return torch.cat([input_ids, suffix], dim=1)


def test_run_generate_left_pads_decoder_only_batches():
    bench = load_generate_module()
    tokenizer = RecordingTokenizer()

    metrics = bench.run_generate(
        model=FakeGenerateModel(),
        tokenizer=tokenizer,
        prompts=["short", "a longer prompt"],
        device=torch.device("cpu"),
        max_new_tokens=2,
        iters=1,
        warmup=0,
    )

    assert tokenizer.padding_sides_seen == ["left"]
    assert tokenizer.padding_side == "left"
    assert metrics["tokenizer_padding_side"] == "left"


def test_tiny_qwen_compression_generate_benchmark_runs_offline_cpu(tmp_path):
    fixture = load_fixture_module()
    model_dir = tmp_path / "TinyQwen3"
    demo_root = tmp_path / "demo"
    fixture.create_fixture(model_dir, vocab_size=64, hidden_size=32, intermediate_size=64)
    script = Path(__file__).resolve().parents[1] / "scripts" / "tiny_qwen_compression_generate_benchmark.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--demo-root",
            str(demo_root),
            "--model-path",
            str(model_dir),
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--max-new-tokens",
            "2",
            "--iters",
            "1",
            "--warmup",
            "0",
            "--cap-budget",
            "4096",
            "--cap-max-iter",
            "2",
            "--cap-policy-steps",
            "1",
            "--cap-samples-per-step",
            "1",
            "--enable-inference-cache",
            "--export-dense-for-serving",
            "--run-label",
            "cpu_test",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=180,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads((demo_root / "artifacts" / "tiny_qwen_compression_generate_cpu_test.json").read_text())
    assert payload["status"] == "PASS"
    assert payload["download_attempted"] is False
    assert payload["backend"] == "torch_generate"
    assert payload["baseline"]["status"] == "PASS"
    assert payload["baseline"]["generated_tokens"] > 0
    assert payload["cap"]["status"] == "PASS"
    assert payload["cap"]["targeted_compression_ratio"] > 1
    assert payload["qpruner"]["status"] == "PASS"
    assert payload["qpruner"]["average_bits"] <= 4.0
    assert payload["inference_cache_enabled"] is True
    assert payload["serving_dense_export"] is True
    assert payload["cap"]["exported_dense_linears"] > 0
    assert payload["qpruner"]["exported_dense_linears"] > 0
