from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from tidal.qpruner import BitwidthConfig, allocate_bitwidths, bayesian_refine_bitwidths

__all__ = [
    "QuantizationPlan",
    "QuantizedLinear",
    "collect_linear_layer_sizes",
    "build_quantization_plan",
    "apply_mixed_precision_quantization",
]


@dataclass(frozen=True)
class QuantizationPlan:
    bitwidths: BitwidthConfig
    layer_sizes: dict[str, int]
    score: float | None = None


class QuantizedLinear(nn.Module):
    """Small n-bit symmetric Linear backend for QPruner plans."""

    def __init__(self, weight_codes: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor | None, *, bits: int):
        super().__init__()
        if bits < 2 or bits > 8:
            raise ValueError("bits must be in [2, 8]")
        self.bits = int(bits)
        self.in_features = int(weight_codes.shape[1])
        self.out_features = int(weight_codes.shape[0])
        self.register_buffer("weight_codes", weight_codes.to(torch.int8))
        self.register_buffer("scale", scale.reshape(()).to(torch.float32))
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)

    @classmethod
    def from_linear(cls, module: nn.Linear, *, bits: int) -> "QuantizedLinear":
        if bits < 2 or bits > 8:
            raise ValueError("bits must be in [2, 8]")
        weight = module.weight.detach().to(torch.float32)
        qmax = (1 << (bits - 1)) - 1
        max_abs = torch.max(torch.abs(weight)) if weight.numel() else torch.tensor(0.0, device=weight.device)
        if float(max_abs) == 0.0:
            scale = torch.tensor(1.0, device=weight.device)
            codes = torch.zeros_like(weight, dtype=torch.int8)
        else:
            scale = max_abs / qmax
            codes = torch.clamp(torch.round(weight / scale), -qmax, qmax).to(torch.int8)
        return cls(codes, scale, module.bias, bits=bits)

    def dequantized_weight(self) -> torch.Tensor:
        return self.weight_codes.to(torch.float32) * self.scale

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight = self.dequantized_weight().to(dtype=inputs.dtype, device=inputs.device)
        bias = self.bias.to(dtype=inputs.dtype, device=inputs.device) if self.bias is not None else None
        return F.linear(inputs, weight, bias)


def collect_linear_layer_sizes(model: nn.Module, *, name_filter: Callable[[str], bool] | None = None) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for name, module in model.named_modules():
        if not name or not isinstance(module, nn.Linear):
            continue
        if name_filter is not None and not name_filter(name):
            continue
        sizes[name] = int(module.weight.numel())
    if not sizes:
        raise ValueError("model does not contain matching torch.nn.Linear modules")
    return sizes


def _set_submodule(model: nn.Module, name: str, module: nn.Module) -> None:
    parent = model
    parts = name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], module)


def build_quantization_plan(
    model: nn.Module,
    importances: Mapping[str, float],
    *,
    candidate_bits: Sequence[int] = (2, 4, 8),
    max_memory_bits: int | None = None,
    max_average_bits: float | None = None,
    objective: Callable[[BitwidthConfig], float] | None = None,
    refine_trials: int = 0,
    seed: int | None = None,
) -> QuantizationPlan:
    layer_sizes = collect_linear_layer_sizes(model)
    if set(importances) != set(layer_sizes):
        raise ValueError("importances must contain exactly the model linear layers")
    if objective is not None and refine_trials > 0:
        result = bayesian_refine_bitwidths(
            importances,
            layer_sizes,
            objective=objective,
            candidate_bits=candidate_bits,
            max_memory_bits=max_memory_bits,
            max_average_bits=max_average_bits,
            max_trials=refine_trials,
            seed=seed,
        )
        return QuantizationPlan(result.config, layer_sizes, result.score)
    bitwidths = allocate_bitwidths(
        importances,
        layer_sizes,
        candidate_bits=candidate_bits,
        max_memory_bits=max_memory_bits,
        max_average_bits=max_average_bits,
    )
    return QuantizationPlan(bitwidths, layer_sizes, None)


def apply_mixed_precision_quantization(
    model: nn.Module,
    bitwidths: Mapping[str, int] | QuantizationPlan,
    *,
    inplace: bool = False,
) -> nn.Module:
    target = model if inplace else deepcopy(model)
    config = bitwidths.bitwidths if isinstance(bitwidths, QuantizationPlan) else dict(bitwidths)
    modules = dict(target.named_modules())
    missing = sorted(set(config) - set(modules))
    if missing:
        raise ValueError(f"unknown linear layers in bitwidths: {', '.join(missing)}")
    for name, bits in config.items():
        module = modules[name]
        if not isinstance(module, nn.Linear):
            raise ValueError(f"{name} is not a torch.nn.Linear module")
        _set_submodule(target, name, QuantizedLinear.from_linear(module, bits=int(bits)))
    return target
