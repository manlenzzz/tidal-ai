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
        self.code_count = int(weight_codes.numel())
        packed_weight_codes = self._pack_weight_codes(weight_codes, self.bits).to(device=weight_codes.device)
        self.register_buffer("packed_weight_codes", packed_weight_codes)
        self.register_buffer("scale", scale.reshape(()).to(torch.float32))
        self.register_buffer("_cached_weight", None)
        self.register_buffer("_cached_weight_codes", None)
        self.register_buffer("_cached_scaled_weight_codes", None)
        self.register_buffer("_cached_scale", None)
        self.register_buffer("_cached_bias", None)
        self._released_packed_code_bytes = 0
        self._scaled_code_matmul_enabled = False
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)

    @staticmethod
    def _pack_weight_codes(weight_codes: torch.Tensor, bits: int) -> torch.Tensor:
        qmax = (1 << (bits - 1)) - 1
        codes = weight_codes.detach().to(device="cpu", dtype=torch.int16).reshape(-1)
        if codes.numel() == 0:
            return torch.empty(0, dtype=torch.uint8)
        if int(codes.min()) < -qmax or int(codes.max()) > qmax:
            raise ValueError(f"weight codes must be in [-{qmax}, {qmax}] for {bits}-bit quantization")
        unsigned = (codes.to(torch.long) + qmax).to(torch.long)
        packed_len = (int(unsigned.numel()) * bits + 7) // 8
        mask = (1 << bits) - 1
        if 8 % bits == 0:
            values_per_byte = 8 // bits
            pad = (-int(unsigned.numel())) % values_per_byte
            if pad:
                unsigned = torch.cat([unsigned, torch.zeros(pad, dtype=unsigned.dtype)])
            shifts = torch.arange(values_per_byte, dtype=torch.long) * bits
            packed = ((unsigned.reshape(-1, values_per_byte) & mask) << shifts).sum(dim=1)
            return packed.to(torch.uint8)

        bit_offsets = torch.arange(unsigned.numel(), dtype=torch.long) * bits
        byte_indices = bit_offsets // 8
        shifts = bit_offsets % 8
        packed = torch.zeros(packed_len, dtype=torch.long)
        packed.scatter_add_(0, byte_indices, (unsigned << shifts) & 0xFF)
        crosses_byte = shifts + bits > 8
        if bool(crosses_byte.any()):
            packed.scatter_add_(
                0,
                byte_indices[crosses_byte] + 1,
                unsigned[crosses_byte] >> (8 - shifts[crosses_byte]),
            )
        return packed.to(torch.uint8)

    @staticmethod
    def _unpack_weight_codes(
        packed_weight_codes: torch.Tensor,
        *,
        bits: int,
        code_count: int,
        out_features: int,
        in_features: int,
    ) -> torch.Tensor:
        qmax = (1 << (bits - 1)) - 1
        packed = packed_weight_codes.detach().to(device="cpu", dtype=torch.long).reshape(-1)
        if code_count == 0:
            return torch.empty((out_features, in_features), dtype=torch.int8)
        mask = (1 << bits) - 1
        if 8 % bits == 0:
            values_per_byte = 8 // bits
            shifts = torch.arange(values_per_byte, dtype=torch.long) * bits
            unsigned = ((packed.reshape(-1, 1) >> shifts) & mask).reshape(-1)[:code_count]
        else:
            bit_offsets = torch.arange(code_count, dtype=torch.long) * bits
            byte_indices = bit_offsets // 8
            shifts = bit_offsets % 8
            unsigned = packed[byte_indices] >> shifts
            crosses_byte = shifts + bits > 8
            if bool(crosses_byte.any()):
                unsigned[crosses_byte] += packed[byte_indices[crosses_byte] + 1] << (8 - shifts[crosses_byte])
            unsigned = unsigned & mask
        signed = unsigned.to(torch.int16) - qmax
        return signed.reshape(out_features, in_features).to(torch.int8)

    @property
    def code_device(self) -> torch.device:
        return self.packed_weight_codes.device

    @property
    def weight_codes(self) -> torch.Tensor:
        return self._unpack_weight_codes(
            self.packed_weight_codes,
            bits=self.bits,
            code_count=self.code_count,
            out_features=self.out_features,
            in_features=self.in_features,
        ).to(device=self.code_device)

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
        return self.dequantized_weight_from_codes(self.weight_codes)

    def dequantized_weight_from_codes(
        self,
        codes: torch.Tensor,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        target_dtype = dtype or torch.float32
        target_device = device or self.scale.device
        target_device = torch.device(target_device)
        cached_scale = self._cached_scale
        scale = (
            cached_scale
            if (
                cached_scale is not None
                and cached_scale.dtype == target_dtype
                and cached_scale.device == target_device
            )
            else self.scale.to(device=target_device, dtype=target_dtype)
        )
        return codes.to(device=target_device, dtype=target_dtype) * scale

    def enable_inference_cache(
        self,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        weight = self.dequantized_weight()
        if dtype is not None or device is not None:
            weight = weight.to(dtype=dtype or weight.dtype, device=device or weight.device)
        self._cached_weight = weight.detach()

    def enable_code_cache(
        self,
        *,
        device: torch.device | str | None = None,
        release_packed_codes: bool = False,
    ) -> None:
        codes = self.weight_codes
        if device is not None:
            codes = codes.to(device=device)
        self._cached_weight_codes = codes.detach()
        self._cached_scaled_weight_codes = None
        self._scaled_code_matmul_enabled = False
        if release_packed_codes:
            self.release_packed_codes()

    def enable_scaled_code_matmul(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        release_packed_codes: bool = False,
    ) -> None:
        self.enable_code_cache(device=device, release_packed_codes=release_packed_codes)
        if dtype is not None and self._cached_weight_codes is not None:
            self._cached_scaled_weight_codes = self._cached_weight_codes.to(dtype=dtype).detach()
        self._scaled_code_matmul_enabled = True

    def enable_aux_cache(
        self,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        target_dtype = dtype or self.scale.dtype
        target_device = device or self.scale.device
        self._cached_scale = self.scale.to(dtype=target_dtype, device=target_device).detach()
        if self.bias is None:
            self._cached_bias = None
        else:
            self._cached_bias = self.bias.to(dtype=target_dtype, device=target_device).detach()

    def release_packed_codes(self) -> None:
        if self._cached_weight_codes is None:
            raise RuntimeError("packed codes can only be released after enabling the code cache")
        packed = self.packed_weight_codes
        self._released_packed_code_bytes += int(packed.numel() * packed.element_size())
        self.packed_weight_codes = torch.empty(0, dtype=torch.uint8, device=packed.device)

    def clear_inference_cache(self) -> None:
        self._cached_weight = None
        self._cached_weight_codes = None
        self._cached_scaled_weight_codes = None
        self._cached_scale = None
        self._cached_bias = None
        self._scaled_code_matmul_enabled = False

    @property
    def scaled_code_matmul_enabled(self) -> bool:
        return bool(self._scaled_code_matmul_enabled and self._cached_weight_codes is not None)

    @property
    def cached_code_bytes(self) -> int:
        cached_codes = self._cached_weight_codes
        if cached_codes is None:
            return 0
        return int(cached_codes.numel() * cached_codes.element_size())

    @property
    def cached_scaled_code_bytes(self) -> int:
        cached_codes = self._cached_scaled_weight_codes
        if cached_codes is None:
            return 0
        return int(cached_codes.numel() * cached_codes.element_size())

    @property
    def cached_weight_bytes(self) -> int:
        cached_weight = self._cached_weight
        if cached_weight is None:
            return 0
        return int(cached_weight.numel() * cached_weight.element_size())

    @property
    def cached_scale_bytes(self) -> int:
        cached_scale = self._cached_scale
        if cached_scale is None:
            return 0
        return int(cached_scale.numel() * cached_scale.element_size())

    @property
    def cached_bias_bytes(self) -> int:
        cached_bias = self._cached_bias
        if cached_bias is None:
            return 0
        return int(cached_bias.numel() * cached_bias.element_size())

    @property
    def cached_aux_bytes(self) -> int:
        return self.cached_scale_bytes + self.cached_bias_bytes

    @property
    def released_packed_code_bytes(self) -> int:
        return int(self._released_packed_code_bytes)

    def _forward_scale(self, inputs: torch.Tensor) -> torch.Tensor:
        cached_scale = self._cached_scale
        if cached_scale is not None and cached_scale.dtype == inputs.dtype and cached_scale.device == inputs.device:
            return cached_scale
        return self.scale.to(dtype=inputs.dtype, device=inputs.device)

    def _forward_bias(self, inputs: torch.Tensor) -> torch.Tensor | None:
        if self.bias is None:
            return None
        cached_bias = self._cached_bias
        if cached_bias is not None and cached_bias.dtype == inputs.dtype and cached_bias.device == inputs.device:
            return cached_bias
        return self.bias.to(dtype=inputs.dtype, device=inputs.device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        cached_weight = self._cached_weight
        if (
            cached_weight is not None
            and cached_weight.dtype == inputs.dtype
            and cached_weight.device == inputs.device
        ):
            weight = cached_weight
        else:
            cached_codes = self._cached_weight_codes
            if (
                self.scaled_code_matmul_enabled
                and cached_codes is not None
                and cached_codes.device == inputs.device
            ):
                scaled_codes = self._cached_scaled_weight_codes
                code_weight = (
                    scaled_codes
                    if (
                        scaled_codes is not None
                        and scaled_codes.dtype == inputs.dtype
                        and scaled_codes.device == inputs.device
                    )
                    else cached_codes.to(dtype=inputs.dtype)
                )
                code_output = F.linear(inputs, code_weight, None) * self._forward_scale(inputs)
                bias = self._forward_bias(inputs)
                if bias is not None:
                    code_output = code_output + bias
                return code_output
            elif cached_codes is not None and cached_codes.device == inputs.device:
                weight = self.dequantized_weight_from_codes(
                    cached_codes,
                    dtype=inputs.dtype,
                    device=inputs.device,
                )
            else:
                weight = self.dequantized_weight().to(dtype=inputs.dtype, device=inputs.device)
        bias = self._forward_bias(inputs)
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
