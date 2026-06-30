import torch

from tidal.methods.qpruner.grouped import (
    grouped_same_input_scaled_code_bmm,
    grouped_same_input_scaled_code_fused_2d,
    grouped_qpruner_cache_summary,
    grouped_same_input_scaled_code,
    grouped_scaled_code_bmm,
    sequential_scaled_code_group,
)
from tidal.methods.qpruner.torch import QuantizedLinear


def make_quantized_linear(*, in_features: int = 8, out_features: int = 4, seed: int = 0) -> QuantizedLinear:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    linear = torch.nn.Linear(in_features, out_features)
    with torch.no_grad():
        linear.weight.copy_(torch.randn(out_features, in_features, generator=generator) * 0.02)
        linear.bias.copy_(torch.randn(out_features, generator=generator) * 0.02)
    module = QuantizedLinear.from_linear(linear, bits=4).eval()
    module.enable_scaled_code_matmul(device="cpu")
    module.enable_aux_cache(dtype=torch.float32, device="cpu")
    return module


def test_grouped_scaled_code_bmm_matches_sequential_scaled_code_modules():
    modules = [make_quantized_linear(seed=100 + index) for index in range(3)]
    grouped_inputs = torch.randn(3, 5, 8)

    sequential = sequential_scaled_code_group(modules, grouped_inputs)
    grouped = grouped_scaled_code_bmm(modules, grouped_inputs)

    assert grouped.shape == sequential.shape
    assert torch.max(torch.abs(grouped - sequential)).item() <= 1e-5


def test_grouped_same_input_scaled_code_matches_independent_quantized_linears():
    modules = [make_quantized_linear(seed=150 + index) for index in range(2)]
    inputs = torch.randn(2, 3, 8)

    sequential = tuple(module(inputs) for module in modules)
    grouped = grouped_same_input_scaled_code(modules, inputs)

    assert isinstance(grouped, tuple)
    assert len(grouped) == len(sequential)
    for grouped_output, sequential_output in zip(grouped, sequential):
        assert grouped_output.shape == sequential_output.shape
        assert torch.max(torch.abs(grouped_output - sequential_output)).item() <= 1e-5


def test_grouped_same_input_scaled_code_does_not_materialize_repeated_inputs(monkeypatch):
    modules = [make_quantized_linear(seed=175 + index) for index in range(2)]
    inputs = torch.randn(2, 3, 8)
    original_contiguous = torch.Tensor.contiguous
    contiguous_calls: list[tuple[int, ...]] = []

    def tracking_contiguous(self: torch.Tensor, *args, **kwargs):
        contiguous_calls.append(tuple(self.shape))
        return original_contiguous(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "contiguous", tracking_contiguous)

    grouped_same_input_scaled_code(modules, inputs)

    assert (len(modules), inputs.numel() // inputs.shape[-1], inputs.shape[-1]) not in contiguous_calls


def test_grouped_same_input_scaled_code_uses_single_fused_2d_weight_matmul(monkeypatch):
    modules = [make_quantized_linear(seed=185 + index) for index in range(2)]
    inputs = torch.randn(2, 3, 8)
    matmul_shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    original_matmul = torch.matmul

    def tracking_matmul(left: torch.Tensor, right: torch.Tensor, *args, **kwargs):
        matmul_shapes.append((tuple(left.shape), tuple(right.shape)))
        return original_matmul(left, right, *args, **kwargs)

    monkeypatch.setattr(torch, "matmul", tracking_matmul)

    grouped_same_input_scaled_code_fused_2d(modules, inputs)

    assert matmul_shapes == [((6, 8), (8, 8))]


def test_grouped_same_input_scaled_code_default_keeps_batched_weight_matmul(monkeypatch):
    modules = [make_quantized_linear(seed=195 + index) for index in range(2)]
    inputs = torch.randn(2, 3, 8)
    matmul_shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    original_matmul = torch.matmul

    def tracking_matmul(left: torch.Tensor, right: torch.Tensor, *args, **kwargs):
        matmul_shapes.append((tuple(left.shape), tuple(right.shape)))
        return original_matmul(left, right, *args, **kwargs)

    monkeypatch.setattr(torch, "matmul", tracking_matmul)

    grouped_same_input_scaled_code(modules, inputs)

    assert matmul_shapes == [((6, 8), (2, 8, 4))]


def test_grouped_same_input_scaled_code_strategy_aliases_match_outputs():
    modules = [make_quantized_linear(seed=205 + index) for index in range(2)]
    inputs = torch.randn(2, 3, 8)

    default = grouped_same_input_scaled_code(modules, inputs)
    batched = grouped_same_input_scaled_code_bmm(modules, inputs)
    fused = grouped_same_input_scaled_code_fused_2d(modules, inputs)

    for default_output, batched_output, fused_output in zip(default, batched, fused):
        assert torch.max(torch.abs(default_output - batched_output)).item() <= 1e-5
        assert torch.max(torch.abs(default_output - fused_output)).item() <= 1e-5


def test_grouped_qpruner_cache_summary_counts_memory_native_caches():
    modules = [make_quantized_linear(seed=200 + index) for index in range(2)]

    summary = grouped_qpruner_cache_summary(modules)

    assert summary == {
        "module_count": 2,
        "grouped_code_cache_bytes": sum(module.cached_code_bytes for module in modules),
        "grouped_aux_cache_bytes": sum(module.cached_aux_bytes for module in modules),
        "grouped_scaled_code_cache_bytes": sum(module.cached_scaled_code_bytes for module in modules),
    }


def test_grouped_scaled_code_bmm_rejects_mismatched_projection_shape():
    modules = [
        make_quantized_linear(in_features=8, out_features=4, seed=300),
        make_quantized_linear(in_features=8, out_features=5, seed=301),
    ]
    grouped_inputs = torch.randn(2, 5, 8)

    try:
        grouped_scaled_code_bmm(modules, grouped_inputs)
    except ValueError as exc:
        assert "same in_features and out_features" in str(exc)
    else:
        raise AssertionError("expected mismatched grouped projection shapes to be rejected")
