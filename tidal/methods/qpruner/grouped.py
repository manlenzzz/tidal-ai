from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from tidal.methods.qpruner.torch import QuantizedLinear

__all__ = [
    "grouped_qpruner_cache_summary",
    "grouped_same_input_scaled_code",
    "grouped_same_input_scaled_code_bmm",
    "grouped_same_input_scaled_code_fused_2d",
    "grouped_scaled_code_bmm",
    "sequential_scaled_code_group",
]


def _scaled_code_weight(module: QuantizedLinear, reference: torch.Tensor) -> torch.Tensor:
    cached_codes = module._cached_weight_codes
    if cached_codes is None:
        raise RuntimeError("grouped scaled-code matmul requires enabled code caches")
    scaled_codes = module._cached_scaled_weight_codes
    if (
        scaled_codes is not None
        and scaled_codes.dtype == reference.dtype
        and scaled_codes.device == reference.device
    ):
        return scaled_codes
    return cached_codes.to(dtype=reference.dtype, device=reference.device)


def _validate_grouped_inputs(modules: Sequence[QuantizedLinear], grouped_inputs: torch.Tensor) -> None:
    if not modules:
        raise ValueError("at least one module is required")
    if grouped_inputs.ndim != 3:
        raise ValueError("grouped inputs must have shape [module_count, batch_size, in_features]")
    if grouped_inputs.shape[0] != len(modules):
        raise ValueError("grouped input module dimension must match module count")

    in_features = modules[0].in_features
    out_features = modules[0].out_features
    if grouped_inputs.shape[-1] != in_features:
        raise ValueError("grouped inputs must match module in_features")
    for module in modules:
        if module.in_features != in_features or module.out_features != out_features:
            raise ValueError("all grouped modules must have the same in_features and out_features")


def sequential_scaled_code_group(modules: Sequence[QuantizedLinear], grouped_inputs: torch.Tensor) -> torch.Tensor:
    _validate_grouped_inputs(modules, grouped_inputs)
    return torch.stack([module(grouped_inputs[index]) for index, module in enumerate(modules)], dim=0)


def grouped_scaled_code_bmm(modules: Sequence[QuantizedLinear], grouped_inputs: torch.Tensor) -> torch.Tensor:
    _validate_grouped_inputs(modules, grouped_inputs)

    code_weights: list[torch.Tensor] = []
    scales: list[torch.Tensor] = []
    biases: list[torch.Tensor] = []
    for index, module in enumerate(modules):
        module_inputs = grouped_inputs[index]
        code_weights.append(_scaled_code_weight(module, grouped_inputs))
        scales.append(module._forward_scale(module_inputs).reshape(()))
        bias = module._forward_bias(module_inputs)
        if bias is None:
            bias = torch.zeros(module.out_features, dtype=grouped_inputs.dtype, device=grouped_inputs.device)
        biases.append(bias)

    stacked_codes = torch.stack(code_weights, dim=0)
    outputs = torch.bmm(grouped_inputs, stacked_codes.transpose(1, 2))
    outputs = outputs * torch.stack(scales, dim=0).reshape(len(modules), 1, 1)
    return outputs + torch.stack(biases, dim=0).reshape(len(modules), 1, modules[0].out_features)


def grouped_same_input_scaled_code(
    modules: Sequence[QuantizedLinear],
    inputs: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    return grouped_same_input_scaled_code_bmm(modules, inputs)


def grouped_same_input_scaled_code_bmm(
    modules: Sequence[QuantizedLinear],
    inputs: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    if not modules:
        raise ValueError("at least one module is required")
    if inputs.ndim < 1:
        raise ValueError("inputs must include an in_features dimension")
    if inputs.shape[-1] != modules[0].in_features:
        raise ValueError("inputs must match module in_features")

    flat_inputs = inputs.reshape(-1, inputs.shape[-1])
    reference = inputs
    code_weights: list[torch.Tensor] = []
    scales: list[torch.Tensor] = []
    biases: list[torch.Tensor] = []
    for module in modules:
        if module.in_features != modules[0].in_features or module.out_features != modules[0].out_features:
            raise ValueError("all grouped modules must have the same in_features and out_features")
        code_weights.append(_scaled_code_weight(module, reference))
        scales.append(module._forward_scale(inputs).reshape(()))
        bias = module._forward_bias(inputs)
        if bias is None:
            bias = torch.zeros(module.out_features, dtype=inputs.dtype, device=inputs.device)
        biases.append(bias)

    stacked_codes = torch.stack(code_weights, dim=0)
    flat_outputs = torch.matmul(flat_inputs, stacked_codes.transpose(1, 2))
    flat_outputs = flat_outputs * torch.stack(scales, dim=0).reshape(len(modules), 1, 1)
    flat_outputs = flat_outputs + torch.stack(biases, dim=0).reshape(len(modules), 1, modules[0].out_features)
    output_shape = (*inputs.shape[:-1], modules[0].out_features)
    return tuple(flat_outputs[index].reshape(output_shape) for index in range(len(modules)))


def grouped_same_input_scaled_code_fused_2d(
    modules: Sequence[QuantizedLinear],
    inputs: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    if not modules:
        raise ValueError("at least one module is required")
    if inputs.ndim < 1:
        raise ValueError("inputs must include an in_features dimension")
    if inputs.shape[-1] != modules[0].in_features:
        raise ValueError("inputs must match module in_features")

    flat_inputs = inputs.reshape(-1, inputs.shape[-1])
    reference = inputs
    code_weights: list[torch.Tensor] = []
    scales: list[torch.Tensor] = []
    biases: list[torch.Tensor] = []
    for module in modules:
        if module.in_features != modules[0].in_features or module.out_features != modules[0].out_features:
            raise ValueError("all grouped modules must have the same in_features and out_features")
        code_weights.append(_scaled_code_weight(module, reference))
        scales.append(module._forward_scale(inputs).reshape(()))
        bias = module._forward_bias(inputs)
        if bias is None:
            bias = torch.zeros(module.out_features, dtype=inputs.dtype, device=inputs.device)
        biases.append(bias)

    fused_codes = torch.cat(code_weights, dim=0)
    flat_outputs = torch.matmul(flat_inputs, fused_codes.transpose(0, 1))
    flat_outputs = flat_outputs.reshape(flat_inputs.shape[0], len(modules), modules[0].out_features).transpose(0, 1)
    flat_outputs = flat_outputs * torch.stack(scales, dim=0).reshape(len(modules), 1, 1)
    flat_outputs = flat_outputs + torch.stack(biases, dim=0).reshape(len(modules), 1, modules[0].out_features)
    output_shape = (*inputs.shape[:-1], modules[0].out_features)
    return tuple(flat_outputs[index].reshape(output_shape) for index in range(len(modules)))


def grouped_qpruner_cache_summary(modules: Sequence[QuantizedLinear]) -> dict[str, Any]:
    return {
        "module_count": len(modules),
        "grouped_code_cache_bytes": sum(module.cached_code_bytes for module in modules),
        "grouped_aux_cache_bytes": sum(module.cached_aux_bytes for module in modules),
        "grouped_scaled_code_cache_bytes": sum(module.cached_scaled_code_bytes for module in modules),
    }
