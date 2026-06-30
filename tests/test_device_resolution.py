import torch

from tidal.device import resolve_device, resolve_dtype


class FakeAccelDevice:
    """Stand-in for a torch.device whose type is an accelerator."""

    def __init__(self, type_name: str):
        self.type = type_name


def test_resolve_device_explicit_cpu():
    device = resolve_device("cpu")
    assert isinstance(device, torch.device)
    assert device.type == "cpu"


def test_resolve_device_auto_falls_back_to_cpu(monkeypatch):
    # No CUDA, no NPU available -> CPU, and torch_npu must not be imported.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    import sys

    monkeypatch.setitem(sys.modules, "torch_npu", None)  # importing would fail
    device = resolve_device(None)
    assert device.type == "cpu"


def test_resolve_device_auto_prefers_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    device = resolve_device(None)
    assert device.type == "cuda"


def test_resolve_dtype_explicit_names():
    assert resolve_dtype("float32", torch.device("cpu")) is torch.float32
    assert resolve_dtype("float16", torch.device("cpu")) is torch.float16
    assert resolve_dtype("bfloat16", torch.device("cpu")) is torch.bfloat16


def test_resolve_dtype_cpu_default_is_auto():
    assert resolve_dtype(None, torch.device("cpu")) == "auto"


def test_resolve_dtype_accelerator_default_is_bfloat16():
    assert resolve_dtype(None, FakeAccelDevice("cuda")) is torch.bfloat16
    assert resolve_dtype(None, FakeAccelDevice("npu")) is torch.bfloat16


def test_calibration_batches_built_on_resolved_device():
    from tidal.workflows.common import build_calibration_batches_from_text

    class CharTokenizer:
        pad_token_id = 0
        eos_token_id = 0

        def __call__(self, text, truncation=True, max_length=8, return_attention_mask=True):
            ids = [ord(c) % 50 + 1 for c in text][:max_length]
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    device = resolve_device("cpu")
    batches = build_calibration_batches_from_text(
        calibration_data=["hello world", "device test"],
        calibration_text_field="text",
        calibration_max_samples=2,
        calibration_max_length=8,
        calibration_batch_size=1,
        tokenizer=CharTokenizer(),
        model_id=None,
        cache_dir=None,
        local_files_only=True,
        trust_remote_code=False,
        device=device,
    )
    assert batches is not None and len(batches) >= 1
    for batch in batches:
        assert batch["input_ids"].device.type == device.type
        assert batch["labels"].device.type == device.type


def test_cap_compress_accepts_device_dtype_on_cpu():
    from torch import nn

    from tidal.workflows.compression import cap_compress

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(8, 8, bias=False)
            self.gate_proj = nn.Linear(8, 16, bias=False)
            self.down_proj = nn.Linear(16, 8, bias=False)

        def named_modules(self, *args, **kwargs):
            return super().named_modules(*args, **kwargs)

    result = cap_compress(
        model=TinyModel(),
        budget=64,
        target_roles=None,
        name_filter=lambda name: True,
        device="cpu",
        dtype="float32",
        max_iter=5,
        policy_steps=2,
        samples_per_step=2,
    )
    assert result.model is not None
    assert result.summary is not None
