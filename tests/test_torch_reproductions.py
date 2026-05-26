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
