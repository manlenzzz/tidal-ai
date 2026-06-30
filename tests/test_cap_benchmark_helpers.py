import importlib.util
from pathlib import Path
from types import SimpleNamespace


class FakeDevice:
    type = "npu"


def load_cap_benchmark():
    script = Path(__file__).resolve().parents[1] / "scripts" / "cap_benchmark.py"
    spec = importlib.util.spec_from_file_location("cap_benchmark", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_synchronize_device_calls_torch_npu(monkeypatch):
    bench = load_cap_benchmark()
    calls = []

    monkeypatch.setattr(bench.torch, "npu", SimpleNamespace(synchronize=lambda: calls.append("npu")), raising=False)

    bench.synchronize_device(FakeDevice())

    assert calls == ["npu"]


def test_tokens_per_second_uses_batch_seq_and_latency():
    bench = load_cap_benchmark()

    assert bench.tokens_per_second(batch_size=4, seq_len=32, latency_ms=8.0) == 16000.0
