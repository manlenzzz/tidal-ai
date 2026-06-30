import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import torch


def load_benchmark():
    script = Path(__file__).resolve().parents[1] / "scripts" / "ascend_inference_benchmark.py"
    spec = importlib.util.spec_from_file_location("ascend_inference_benchmark", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeTokenizer:
    def __init__(self):
        self.padding_side = "right"
        self.padding_sides_seen = []

    def __call__(self, prompts, return_tensors=None, padding=True, truncation=True, max_length=16):
        assert return_tensors == "pt"
        self.padding_sides_seen.append(self.padding_side)
        batch = len(prompts)
        width = min(max_length, 4)
        input_ids = torch.arange(batch * width).reshape(batch, width)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }


class TokenTypeTokenizer(FakeTokenizer):
    def __call__(self, prompts, return_tensors=None, padding=True, truncation=True, max_length=16):
        batch = super().__call__(prompts, return_tensors, padding, truncation, max_length)
        batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])
        return batch


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def generate(self, input_ids, attention_mask=None, max_new_tokens=4, do_sample=False):
        suffix = torch.ones((input_ids.shape[0], max_new_tokens), dtype=input_ids.dtype, device=input_ids.device)
        return torch.cat([input_ids, suffix], dim=1)


class FakeVllmTextOutput:
    def __init__(self, token_ids):
        self.token_ids = token_ids


class FakeVllmRequestOutput:
    def __init__(self, token_ids):
        self.outputs = [FakeVllmTextOutput(token_ids)]


class FakeVllmLLM:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = 0

    def generate(self, prompts, sampling_params):
        self.calls += 1
        return [FakeVllmRequestOutput(list(range(sampling_params.max_tokens))) for _ in prompts]


class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_tokens_per_second_counts_generated_tokens():
    bench = load_benchmark()

    assert bench.tokens_per_second(generated_tokens=32, latency_ms=8.0) == 4000.0


def test_run_torch_generate_benchmark_with_fake_model():
    bench = load_benchmark()
    device = torch.device("cpu")
    tokenizer = FakeTokenizer()

    result = bench.run_torch_generate_benchmark(
        model=FakeModel(),
        tokenizer=tokenizer,
        prompts=["hello", "world"],
        device=device,
        max_new_tokens=3,
        iters=2,
        warmup=1,
    )

    assert result["backend"] == "torch"
    assert result["status"] == "PASS"
    assert result["batch_size"] == 2
    assert result["generated_tokens"] == 6
    assert result["latency_ms"] > 0
    assert result["tokens_per_s"] > 0
    assert tokenizer.padding_sides_seen == ["left"]
    assert tokenizer.padding_side == "left"
    assert result["tokenizer_padding_side"] == "left"


def test_run_torch_generate_benchmark_filters_token_type_ids():
    bench = load_benchmark()
    device = torch.device("cpu")

    result = bench.run_torch_generate_benchmark(
        model=FakeModel(),
        tokenizer=TokenTypeTokenizer(),
        prompts=["hello", "world"],
        device=device,
        max_new_tokens=3,
        iters=1,
        warmup=0,
    )

    assert result["status"] == "PASS"


def test_run_vllm_generate_benchmark_with_fake_llm():
    bench = load_benchmark()

    result = bench.run_vllm_generate_benchmark(
        llm=FakeVllmLLM(model="/models/qwen"),
        sampling_params=FakeSamplingParams(max_tokens=5),
        prompts=["hello", "world"],
        iters=2,
        warmup=1,
    )

    assert result["backend"] == "vllm"
    assert result["status"] == "PASS"
    assert result["batch_size"] == 2
    assert result["generated_tokens"] == 10
    assert result["latency_ms"] > 0
    assert result["tokens_per_s"] > 0


def test_build_missing_model_report_does_not_require_download(tmp_path):
    bench = load_benchmark()

    report = bench.build_missing_model_report(
        model_id="Qwen/Qwen3-0.6B",
        roots=[tmp_path],
        backend="torch",
    )

    assert report["status"] == "MODEL_MISSING"
    assert report["model_id"] == "Qwen/Qwen3-0.6B"
    assert report["backend"] == "torch"
    assert report["download_attempted"] is False


def test_write_report_outputs_json_and_markdown(tmp_path):
    bench = load_benchmark()
    report = {
        "status": "PASS",
        "model_id": "Qwen/Qwen3-0.6B",
        "model_path": "/models/qwen",
        "backend": "torch",
        "device": "npu",
        "dtype": "torch.float16",
        "batch_size": 2,
        "prompt_tokens": 8,
        "generated_tokens": 16,
        "latency_ms": 12.5,
        "tokens_per_s": 1280.0,
        "peak_mem_mb": 42.0,
    }

    bench.write_report(report, tmp_path, run_label="qwen3_torch")

    payload = json.loads((tmp_path / "artifacts/ascend_inference_qwen3_torch.json").read_text())
    markdown = (tmp_path / "reports/ascend-inference-qwen3_torch.md").read_text()
    assert payload["tokens_per_s"] == 1280.0
    assert "| Backend | torch |" in markdown
    assert "| Tokens/s | 1280.0 |" in markdown
