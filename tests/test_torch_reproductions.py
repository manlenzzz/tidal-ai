import torch
from torch import nn
from peft import get_peft_model

from tidal.methods.global_rank_sparsity.torch import CAPPackedLinear, apply_cap_compression
from tidal.methods.qpruner.torch import (
    QuantizedLinear,
    apply_mixed_precision_quantization,
    build_quantization_plan,
    collect_linear_layer_sizes,
)
from tidal.methods.rankadaptor import build_lora_config, collect_linear_profiles, search_rank_allocation


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 3)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(3, 2)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def test_rankadaptor_exports_rank_pattern_to_peft_lora_config():
    model = TinyBlock()
    profiles = collect_linear_profiles(
        model,
        sensitivities={"fc1": 2.0, "fc2": 0.2},
        min_rank=1,
        max_rank=2,
        rank_step=1,
    )

    result = search_rank_allocation(profiles, budget=19)
    config = build_lora_config(result.config, alpha_multiplier=2)
    peft_model = get_peft_model(model, config)

    assert result.config == {"fc1": 2, "fc2": 1}
    assert peft_model.base_model.model.fc1.lora_A["default"].weight.shape == (2, 4)
    assert peft_model.base_model.model.fc1.lora_B["default"].weight.shape == (3, 2)
    assert peft_model.base_model.model.fc2.lora_A["default"].weight.shape == (1, 3)


def test_qpruner_replaces_torch_linear_layers_with_quantized_modules():
    torch.manual_seed(0)
    model = TinyBlock()
    sizes = collect_linear_layer_sizes(model)
    plan = build_quantization_plan(
        model,
        {"fc1": 2.0, "fc2": 0.1},
        candidate_bits=(2, 4, 8),
        max_average_bits=3.5,
    )
    quantized = apply_mixed_precision_quantization(model, plan, inplace=False)
    output = quantized(torch.randn(5, 4))

    assert sizes == {"fc1": 12, "fc2": 6}
    assert plan.bitwidths == {"fc1": 4, "fc2": 2}
    assert isinstance(quantized.fc1, QuantizedLinear)
    assert isinstance(quantized.fc2, QuantizedLinear)
    assert quantized.fc1.bits == 4
    assert quantized.fc2.bits == 2
    assert output.shape == (5, 2)
    assert isinstance(model.fc1, nn.Linear)


def test_quantized_linear_inference_cache_reuses_dequantized_weight(monkeypatch):
    torch.manual_seed(0)
    linear = nn.Linear(4, 3)
    quantized = QuantizedLinear.from_linear(linear, bits=4)
    inputs = torch.randn(5, 4)
    expected = quantized(inputs)

    quantized.enable_inference_cache(dtype=torch.float32, device=torch.device("cpu"))
    cached = quantized(inputs)
    assert torch.allclose(cached, expected)

    def fail_dequantize():
        raise AssertionError("cached inference should not dequantize every forward")

    monkeypatch.setattr(quantized, "dequantized_weight", fail_dequantize)
    cached_again = quantized(inputs)

    assert torch.allclose(cached_again, expected)


def test_quantized_linear_code_cache_reuses_unpacked_int8_codes(monkeypatch):
    torch.manual_seed(0)
    linear = nn.Linear(4, 3)
    quantized = QuantizedLinear.from_linear(linear, bits=4)
    inputs = torch.randn(5, 4)
    expected = quantized(inputs)

    quantized.enable_code_cache(device=torch.device("cpu"))
    code_cached = quantized(inputs)
    assert torch.allclose(code_cached, expected)

    def fail_unpack(*args, **kwargs):
        raise AssertionError("code cache should not unpack packed bytes every forward")

    monkeypatch.setattr(QuantizedLinear, "_unpack_weight_codes", fail_unpack)
    code_cached_again = quantized(inputs)

    assert torch.allclose(code_cached_again, expected)
    assert quantized.cached_code_bytes == linear.weight.numel()
    assert quantized.cached_weight_bytes == 0


def test_quantized_linear_code_cache_can_release_packed_codes_after_cache(monkeypatch):
    torch.manual_seed(0)
    linear = nn.Linear(4, 3)
    quantized = QuantizedLinear.from_linear(linear, bits=4)
    inputs = torch.randn(5, 4)
    expected = quantized(inputs)
    original_packed_bytes = quantized.packed_weight_codes.numel() * quantized.packed_weight_codes.element_size()

    quantized.enable_code_cache(device=torch.device("cpu"), release_packed_codes=True)

    assert original_packed_bytes > 0
    assert quantized.packed_weight_codes.numel() == 0
    assert quantized.cached_code_bytes == linear.weight.numel()
    assert quantized.released_packed_code_bytes == original_packed_bytes

    def fail_unpack(*args, **kwargs):
        raise AssertionError("released packed codes should not be unpacked while code cache is live")

    monkeypatch.setattr(QuantizedLinear, "_unpack_weight_codes", fail_unpack)
    code_cached = quantized(inputs)

    assert torch.allclose(code_cached, expected)


def test_quantized_linear_code_cache_dequantizes_codes_in_input_dtype(monkeypatch):
    torch.manual_seed(0)
    linear = nn.Linear(4, 3)
    quantized = QuantizedLinear.from_linear(linear, bits=4)
    quantized.enable_code_cache(device=torch.device("cpu"))
    inputs = torch.randn(5, 4, dtype=torch.float16)
    observed = {}
    real_dequantize = quantized.dequantized_weight_from_codes

    def capture_dequantize(codes, *, dtype=None, device=None):
        observed["dtype"] = dtype
        observed["device"] = device
        return real_dequantize(codes, dtype=dtype, device=device)

    monkeypatch.setattr(quantized, "dequantized_weight_from_codes", capture_dequantize)

    output = quantized(inputs)

    assert output.dtype == torch.float16
    assert observed["dtype"] == torch.float16
    assert observed["device"] == inputs.device


def test_quantized_linear_aux_cache_reuses_scale_and_bias_casts(monkeypatch):
    torch.manual_seed(0)
    linear = nn.Linear(4, 3)
    quantized = QuantizedLinear.from_linear(linear, bits=4)
    inputs = torch.randn(5, 4, dtype=torch.float16)
    quantized.enable_code_cache(device=torch.device("cpu"))
    expected = quantized(inputs)

    quantized.enable_aux_cache(dtype=torch.float16, device=torch.device("cpu"))

    assert quantized._cached_scale.dtype == torch.float16
    assert quantized._cached_scale.device == inputs.device
    assert quantized._cached_bias.dtype == torch.float16
    assert quantized._cached_bias.device == inputs.device
    assert quantized.cached_scale_bytes == torch.tensor([], dtype=torch.float16).element_size()
    assert quantized.cached_bias_bytes == linear.bias.numel() * torch.tensor([], dtype=torch.float16).element_size()
    assert quantized.cached_aux_bytes == quantized.cached_scale_bytes + quantized.cached_bias_bytes

    def fail_aux_to(*args, **kwargs):
        raise AssertionError("aux cache should avoid per-forward scale/bias dtype casts")

    monkeypatch.setattr(quantized.scale, "to", fail_aux_to)
    monkeypatch.setattr(quantized.bias, "to", fail_aux_to)
    observed = quantized(inputs)

    assert torch.allclose(observed, expected, atol=1e-3, rtol=1e-3)

    quantized.clear_inference_cache()

    assert quantized.cached_aux_bytes == 0
    assert quantized._cached_scale is None
    assert quantized._cached_bias is None


def test_quantized_linear_scaled_code_matmul_avoids_dequantized_weight(monkeypatch):
    torch.manual_seed(0)
    linear = nn.Linear(4, 3)
    quantized = QuantizedLinear.from_linear(linear, bits=4)
    inputs = torch.randn(5, 4)
    expected = quantized(inputs)

    quantized.enable_scaled_code_matmul(device=torch.device("cpu"))
    scaled_code = quantized(inputs)
    assert torch.allclose(scaled_code, expected, atol=1e-5, rtol=1e-5)

    def fail_dequantize_from_codes(*args, **kwargs):
        raise AssertionError("scaled code matmul should not materialize dequantized dense weight")

    monkeypatch.setattr(quantized, "dequantized_weight_from_codes", fail_dequantize_from_codes)
    scaled_code_again = quantized(inputs)

    assert torch.allclose(scaled_code_again, expected, atol=1e-5, rtol=1e-5)
    assert quantized.scaled_code_matmul_enabled is True
    assert quantized.cached_code_bytes == linear.weight.numel()
    assert quantized.cached_weight_bytes == 0


def test_quantized_linear_scaled_code_matmul_uses_input_dtype(monkeypatch):
    torch.manual_seed(0)
    linear = nn.Linear(4, 3)
    quantized = QuantizedLinear.from_linear(linear, bits=4)
    quantized.enable_scaled_code_matmul(device=torch.device("cpu"))
    inputs = torch.randn(5, 4, dtype=torch.float16)
    observed = {}
    real_linear = torch.nn.functional.linear

    def capture_linear(input, weight, bias=None):
        observed["input_dtype"] = input.dtype
        observed["weight_dtype"] = weight.dtype
        return real_linear(input, weight, bias)

    monkeypatch.setattr("tidal.methods.qpruner.torch.F.linear", capture_linear)

    output = quantized(inputs)

    assert output.dtype == torch.float16
    assert observed["input_dtype"] == torch.float16
    assert observed["weight_dtype"] == torch.float16


def test_quantized_linear_scaled_code_matmul_prebuilds_float_code_cache(monkeypatch):
    torch.manual_seed(0)
    linear = nn.Linear(4, 3)
    quantized = QuantizedLinear.from_linear(linear, bits=4)
    inputs = torch.randn(5, 4, dtype=torch.float32)
    expected = quantized(inputs)

    quantized.enable_scaled_code_matmul(device=torch.device("cpu"), dtype=torch.float32)

    assert quantized.cached_scaled_code_bytes == linear.weight.numel() * torch.tensor([], dtype=torch.float32).element_size()

    def fail_cached_code_to(*args, **kwargs):
        raise AssertionError("scaled code forward should reuse the prebuilt dtype cache")

    monkeypatch.setattr(quantized._cached_weight_codes, "to", fail_cached_code_to)
    output = quantized(inputs)

    assert torch.allclose(output, expected, atol=1e-5, rtol=1e-5)


def test_quantized_linear_packs_subbyte_codes_and_preserves_forward_math():
    torch.manual_seed(0)
    linear = nn.Linear(5, 3)
    quantized = QuantizedLinear.from_linear(linear, bits=2)
    inputs = torch.randn(4, 5)

    buffers = dict(quantized.named_buffers(recurse=False))
    assert "weight_codes" not in buffers
    assert "packed_weight_codes" in buffers
    assert quantized.packed_weight_codes.dtype == torch.uint8
    assert quantized.code_count == linear.weight.numel()
    assert quantized.packed_weight_codes.numel() == (linear.weight.numel() * quantized.bits + 7) // 8

    weight_codes = quantized.weight_codes
    expected_weight = weight_codes.to(torch.float32) * quantized.scale
    expected = torch.nn.functional.linear(inputs, expected_weight, quantized.bias)

    assert weight_codes.shape == linear.weight.shape
    assert weight_codes.dtype == torch.int8
    assert int(torch.max(torch.abs(weight_codes))) <= 1
    assert torch.allclose(quantized.dequantized_weight(), expected_weight)
    assert torch.allclose(quantized(inputs), expected)


def test_quantized_linear_packs_four_bit_codes_two_per_byte():
    torch.manual_seed(0)
    linear = nn.Linear(4, 3, bias=False)
    quantized = QuantizedLinear.from_linear(linear, bits=4)

    assert quantized.code_count == 12
    assert quantized.packed_weight_codes.numel() == 6
    assert quantized.weight_codes.shape == (3, 4)
    assert int(torch.max(torch.abs(quantized.weight_codes))) <= 7


def test_cap_replaces_linear_layers_with_low_rank_sparse_modules():
    torch.manual_seed(0)
    model = TinyBlock()
    compressed = apply_cap_compression(model, {"fc1": 9, "fc2": 7}, inplace=False, max_iter=60)
    output = compressed(torch.randn(5, 4))

    assert isinstance(compressed.fc1, CAPPackedLinear)
    assert isinstance(compressed.fc2, CAPPackedLinear)
    assert compressed.fc1.parameter_count <= 9
    assert compressed.fc2.parameter_count <= 7
    assert output.shape == (5, 2)
    assert isinstance(model.fc1, nn.Linear)
