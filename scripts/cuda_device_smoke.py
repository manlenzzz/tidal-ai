"""CUDA smoke test for the device/dtype port. Self-contained, no HF download.

Exercises the exact paths changed by the multi-device work:
  - resolve_device / resolve_dtype on a real accelerator
  - cap_compress moving an in-memory model to the device
  - CAPPackedLinear buffers landing on the device
  - build_calibration_batches_from_text placing tensors on the device
"""
import json
import sys

import torch
from torch import nn

from tidal.device import resolve_device, resolve_dtype
from tidal.workflows.compression import cap_compress
from tidal.workflows.common import build_calibration_batches_from_text
from tidal.methods.global_rank_sparsity.torch import CAPPackedLinear

report = {}

assert torch.cuda.is_available(), "CUDA not available on this worker"
report["torch_version"] = torch.__version__
report["device_name"] = torch.cuda.get_device_name(0)

device = resolve_device(None)
report["resolved_device"] = str(device)
assert device.type == "cuda", f"expected cuda, got {device}"

dtype = resolve_dtype(None, device)
report["resolved_dtype"] = str(dtype)
assert dtype is torch.bfloat16, f"expected bf16 default on accelerator, got {dtype}"


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(16, 16, bias=False)
        self.gate_proj = nn.Linear(16, 32, bias=False)
        self.down_proj = nn.Linear(32, 16, bias=False)


result = cap_compress(
    model=TinyModel(),
    budget=128,
    target_roles=None,
    name_filter=lambda name: True,
    device="cuda",
    dtype="float32",
    max_iter=10,
    policy_steps=2,
    samples_per_step=2,
)

packed_devices = set()
for module in result.model.modules():
    if isinstance(module, CAPPackedLinear):
        packed_devices.add(module.residual_device.type)
        packed_devices.add(module.low_rank_left.device.type)
report["packed_module_devices"] = sorted(packed_devices)
assert packed_devices == {"cuda"}, f"packed buffers not on cuda: {packed_devices}"

# Forward through a packed module on cuda to confirm the recompose path runs.
sample = next(m for m in result.model.modules() if isinstance(m, CAPPackedLinear))
x = torch.randn(2, sample.in_features, device="cuda")
y = sample(x)
report["forward_output_device"] = y.device.type
assert y.device.type == "cuda"


class CharTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, text, truncation=True, max_length=8, return_attention_mask=True):
        ids = [ord(c) % 50 + 1 for c in text][:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


batches = build_calibration_batches_from_text(
    calibration_data=["hello cuda", "device smoke"],
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
batch_devices = {b["input_ids"].device.type for b in batches}
report["calibration_batch_devices"] = sorted(batch_devices)
assert batch_devices == {"cuda"}, f"calibration tensors not on cuda: {batch_devices}"

report["status"] = "PASS"
print("SMOKE_RESULT " + json.dumps(report))
sys.exit(0)
