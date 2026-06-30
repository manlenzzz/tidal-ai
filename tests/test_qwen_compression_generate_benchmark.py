import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from torch import nn


def load_fixture_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "create_tiny_qwen_fixture.py"
    spec = importlib.util.spec_from_file_location("create_tiny_qwen_fixture", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_qwen_generate_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "qwen_compression_generate_benchmark.py"
    spec = importlib.util.spec_from_file_location("qwen_compression_generate_benchmark", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_select_target_names_supports_regex_before_limit():
    bench = load_qwen_generate_module()

    class ToyModel:
        def named_modules(self):
            for name in [
                "model.layers.0.self_attn.q_proj",
                "model.layers.0.self_attn.k_proj",
                "model.layers.0.self_attn.v_proj",
                "model.layers.0.mlp.gate_proj",
                "model.layers.1.self_attn.q_proj",
                "model.layers.1.self_attn.k_proj",
                "model.layers.1.mlp.up_proj",
                "model.layers.2.self_attn.q_proj",
            ]:
                yield name, nn.Linear(2, 2)

    names, total = bench.select_target_names(
        ToyModel(),
        limit=4,
        pattern=r"self_attn\.(q_proj|k_proj)$",
    )

    assert total == 8
    assert names == [
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.1.self_attn.q_proj",
        "model.layers.1.self_attn.k_proj",
    ]


def test_build_prompts_repeats_defaults_for_heavier_decode_batches():
    bench = load_qwen_generate_module()

    prompts = bench.build_prompts(prompt_repeat=3)

    assert len(prompts) == len(bench.DEFAULT_PROMPTS) * 3
    assert prompts[: len(bench.DEFAULT_PROMPTS)] == list(bench.DEFAULT_PROMPTS)
    assert prompts[len(bench.DEFAULT_PROMPTS) : 2 * len(bench.DEFAULT_PROMPTS)] == [
        f"{prompt} [repeat 2]" for prompt in bench.DEFAULT_PROMPTS
    ]
    assert prompts[2 * len(bench.DEFAULT_PROMPTS) :] == [
        f"{prompt} [repeat 3]" for prompt in bench.DEFAULT_PROMPTS
    ]


def test_qwen_markdown_report_explains_target_layer_limit():
    bench = load_qwen_generate_module()

    text = bench.markdown_report(
        {
            "status": "PASS",
            "model_id": "Qwen/Qwen3-0.6B",
            "model_path": "/mnt/nvme/622/models/Qwen3-0.6B",
            "device": "npu",
            "dtype": "torch.float16",
            "prompt_repeat": 4,
            "per_rank_prompt_count": 8,
            "per_rank_generated_tokens_target": 64,
            "target_layer_limit": 2,
            "targeted_layers_total": 196,
            "targeted_layer_names_sample": ["model.layers.0.self_attn.q_proj"],
            "baseline": {"status": "PASS", "latency_ms": 12.0, "tokens_per_s": 80.0, "targeted_layers": 2},
            "cap": {
                "status": "PASS",
                "latency_ms": 10.0,
                "tokens_per_s": 96.0,
                "latency_speedup": 1.2,
                "targeted_layers": 2,
                "targeted_compression_ratio": 24.0,
                "cache_modules": 2,
            },
            "qpruner": {
                "status": "PASS",
                "latency_ms": 11.0,
                "tokens_per_s": 87.0,
                "latency_speedup": 1.091,
                "targeted_layers": 2,
                "average_bits": 4.0,
                "cache_modules": 2,
            },
            "peak_mem_mb": 1200.0,
        }
    )

    assert "Qwen CAP/QPruner Compressed Generate Benchmark" in text
    assert "Qwen/Qwen3-0.6B" in text
    assert "Target layer limit" in text
    assert "2 / 196" in text
    assert "model.layers.0.self_attn.q_proj" in text
    assert "Prompt repeat" in text
    assert "Per-rank prompt count" in text
    assert "Per-rank generated-token target" in text


def test_qwen_compression_generate_benchmark_runs_offline_cpu_with_layer_limit(tmp_path):
    fixture = load_fixture_module()
    model_dir = tmp_path / "TinyQwen3"
    demo_root = tmp_path / "demo"
    fixture.create_fixture(model_dir, vocab_size=64, hidden_size=32, intermediate_size=64)
    script = Path(__file__).resolve().parents[1] / "scripts" / "qwen_compression_generate_benchmark.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--demo-root",
            str(demo_root),
            "--model-id",
            "Qwen/Qwen3-0.6B",
            "--model-path",
            str(model_dir),
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--max-new-tokens",
            "1",
            "--prompt-repeat",
            "3",
            "--iters",
            "1",
            "--warmup",
            "0",
            "--cap-budget",
            "4096",
            "--cap-max-iter",
            "1",
            "--cap-policy-steps",
            "1",
            "--cap-samples-per-step",
            "1",
            "--target-layer-limit",
            "2",
            "--target-layer-pattern",
            "self_attn\\.(q_proj|k_proj)$",
            "--enable-inference-cache",
            "--export-dense-for-serving",
            "--inplace-compression",
            "--run-label",
            "cpu_test",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=180,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads((demo_root / "artifacts" / "qwen_compression_generate_cpu_test.json").read_text())
    assert payload["status"] == "PASS"
    assert payload["model_id"] == "Qwen/Qwen3-0.6B"
    assert payload["backend"] == "torch_generate"
    assert payload["prompt_repeat"] == 3
    assert payload["per_rank_prompt_count"] == 6
    assert payload["per_rank_generated_tokens_target"] == 6
    assert payload["baseline"]["batch_size"] == 6
    assert payload["cap"]["batch_size"] == 6
    assert payload["qpruner"]["batch_size"] == 6
    assert payload["target_layer_limit"] == 2
    assert payload["target_layer_pattern"] == "self_attn\\.(q_proj|k_proj)$"
    assert payload["baseline"]["targeted_layers"] == 2
    assert payload["cap"]["status"] == "PASS"
    assert payload["cap"]["targeted_layers"] == 2
    assert payload["qpruner"]["status"] == "PASS"
    assert payload["qpruner"]["targeted_layers"] == 2
    report = (demo_root / "reports" / "qwen-compression-generate-cpu_test.md").read_text()
    assert "Qwen CAP/QPruner Compressed Generate Benchmark" in report
    assert "self_attn\\.(q_proj|k_proj)$" in report
