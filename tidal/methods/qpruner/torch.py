from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from tidal.methods.qpruner.core import BitwidthConfig, allocate_bitwidths, bayesian_refine_bitwidths, layer_mutual_information
from tidal.model_support import compose_name_filter

__all__ = [
    "QuantizationPlan",
    "QPrunerRun",
    "QuantizedLinear",
    "collect_linear_layer_sizes",
    "build_quantization_plan",
    "apply_mixed_precision_quantization",
    "collect_linear_mutual_information",
    "run_qpruner_mixed_precision",
]


@dataclass(frozen=True)
class QuantizationPlan:
    bitwidths: BitwidthConfig
    layer_sizes: dict[str, int]
    score: float | None = None


@dataclass(frozen=True)
class QPrunerRun:
    model: nn.Module
    plan: QuantizationPlan
    importances: dict[str, float]


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


def collect_linear_layer_sizes(
    model: nn.Module,
    *,
    name_filter: Callable[[str], bool] | None = None,
    target_roles: object | None = None,
    exclude_target_roles: object | None = None,
) -> dict[str, int]:
    selected_name_filter = compose_name_filter(
        name_filter,
        target_roles=target_roles,
        exclude_roles=exclude_target_roles,
    )
    sizes: dict[str, int] = {}
    for name, module in model.named_modules():
        if not name or not isinstance(module, nn.Linear):
            continue
        if selected_name_filter is not None and not selected_name_filter(name):
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
    name_filter: Callable[[str], bool] | None = None,
    target_roles: object | None = None,
    exclude_target_roles: object | None = None,
    seed: int | None = None,
) -> QuantizationPlan:
    layer_sizes = collect_linear_layer_sizes(
        model,
        name_filter=name_filter,
        target_roles=target_roles,
        exclude_target_roles=exclude_target_roles,
    )
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



def _run_model(model: nn.Module, batch: object) -> object:
    if isinstance(batch, Mapping):
        return model(**batch)
    if isinstance(batch, tuple):
        return model(*batch)
    if isinstance(batch, list):
        return model(*batch)
    return model(batch)


def _default_predictions(output: object) -> torch.Tensor:
    if hasattr(output, "logits"):
        output = output.logits
    if isinstance(output, (tuple, list)):
        output = output[0]
    if not torch.is_tensor(output):
        raise TypeError("model output must be a Tensor, tuple/list of Tensor, or object with logits")
    tensor = output.detach()
    if tensor.ndim == 0:
        return tensor.reshape(1).cpu()
    if tensor.ndim == 1:
        return tensor.cpu()
    return torch.argmax(tensor.reshape(tensor.shape[0], -1), dim=1).cpu()


def collect_linear_mutual_information(
    model: nn.Module,
    batches: Sequence[object],
    *,
    prediction_fn: Callable[[object], torch.Tensor] | None = None,
    bins: int = 16,
    name_filter: Callable[[str], bool] | None = None,
    target_roles: object | None = None,
    exclude_target_roles: object | None = None,
) -> dict[str, float]:
    """Run representative data through a model and compute I(layer output; prediction).

    This is the QPruner paper step that records each layer output X and the final
    prediction Y on representative samples before mixed-precision allocation.
    """

    selected_name_filter = compose_name_filter(
        name_filter,
        target_roles=target_roles,
        exclude_roles=exclude_target_roles,
    )
    modules = {
        name: module
        for name, module in model.named_modules()
        if name and isinstance(module, nn.Linear) and (selected_name_filter is None or selected_name_filter(name))
    }
    if not modules:
        raise ValueError("model does not contain matching torch.nn.Linear modules")
    if not batches:
        raise ValueError("batches must not be empty")

    layer_outputs: dict[str, list[torch.Tensor]] = {name: [] for name in modules}
    handles = []

    def make_hook(name: str):
        def hook(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> None:
            value = output[0] if isinstance(output, (tuple, list)) else output
            if not torch.is_tensor(value):
                raise TypeError(f"{name} output is not a Tensor")
            layer_outputs[name].append(value.detach().cpu())

        return hook

    for name, module in modules.items():
        handles.append(module.register_forward_hook(make_hook(name)))

    was_training = model.training
    predictions: list[torch.Tensor] = []
    try:
        model.eval()
        with torch.no_grad():
            for batch in batches:
                output = _run_model(model, batch)
                pred = (prediction_fn or _default_predictions)(output)
                if not torch.is_tensor(pred):
                    pred = torch.as_tensor(pred)
                predictions.append(pred.detach().cpu().reshape(-1))
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    y = torch.cat(predictions, dim=0).numpy()
    arrays: dict[str, object] = {}
    for name, chunks in layer_outputs.items():
        if not chunks:
            raise ValueError(f"no outputs were captured for {name}")
        arrays[name] = torch.cat([chunk.reshape(chunk.shape[0], -1) for chunk in chunks], dim=0).numpy()
    return layer_mutual_information(arrays, y, bins=bins)


def run_qpruner_mixed_precision(
    model: nn.Module,
    batches: Sequence[object],
    *,
    candidate_bits: Sequence[int] = (2, 4, 8),
    max_memory_bits: int | None = None,
    max_average_bits: float | None = None,
    objective: Callable[[BitwidthConfig], float] | None = None,
    refine_trials: int = 0,
    prediction_fn: Callable[[object], torch.Tensor] | None = None,
    bins: int = 16,
    name_filter: Callable[[str], bool] | None = None,
    target_roles: object | None = None,
    exclude_target_roles: object | None = None,
    inplace: bool = False,
    seed: int | None = None,
) -> QPrunerRun:
    """Run the QPruner mixed-precision path for a pruned Torch model.

    This combines the paper steps: calibration MI collection, memory-constrained
    bitwidth initialization, optional GP Bayesian refinement, and module rewrite.
    """

    importances = collect_linear_mutual_information(
        model,
        batches,
        prediction_fn=prediction_fn,
        bins=bins,
        name_filter=name_filter,
        target_roles=target_roles,
        exclude_target_roles=exclude_target_roles,
    )
    plan = build_quantization_plan(
        model,
        importances,
        candidate_bits=candidate_bits,
        max_memory_bits=max_memory_bits,
        max_average_bits=max_average_bits,
        objective=objective,
        refine_trials=refine_trials,
        name_filter=name_filter,
        target_roles=target_roles,
        exclude_target_roles=exclude_target_roles,
        seed=seed,
    )
    quantized = apply_mixed_precision_quantization(model, plan, inplace=inplace)
    return QPrunerRun(quantized, plan, dict(importances))
