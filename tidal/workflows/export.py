from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn

from tidal.methods.global_rank_sparsity.torch import CAPPackedLinear
from tidal.methods.qpruner.torch import QuantizedLinear

__all__ = ["export_compressed_linears_to_dense"]


def _dense_linear_from_weight(
    *,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    dtype: torch.dtype,
    device: torch.device,
) -> nn.Linear:
    linear = nn.Linear(int(weight.shape[1]), int(weight.shape[0]), bias=bias is not None, device=device, dtype=dtype)
    linear.weight.data.copy_(weight.to(device=device, dtype=dtype))
    if bias is not None and linear.bias is not None:
        linear.bias.data.copy_(bias.detach().to(device=device, dtype=dtype))
    linear.requires_grad_(False)
    return linear


def _export_module(module: nn.Module, *, dtype: torch.dtype | None, device: torch.device | str | None) -> nn.Module:
    if isinstance(module, CAPPackedLinear):
        weight = module.reconstructed_weight()
        target_dtype = dtype or module.residual_dtype
        target_device = torch.device(device) if device is not None else module.residual_device
        return _dense_linear_from_weight(weight=weight, bias=module.bias, dtype=target_dtype, device=target_device)
    if isinstance(module, QuantizedLinear):
        weight = module.dequantized_weight()
        target_dtype = dtype or module.scale.dtype
        target_device = torch.device(device) if device is not None else module.code_device
        return _dense_linear_from_weight(weight=weight, bias=module.bias, dtype=target_dtype, device=target_device)
    return module


def export_compressed_linears_to_dense(
    model: nn.Module,
    *,
    inplace: bool = False,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> nn.Module:
    """Replace TIDAL compressed Linear modules with plain torch.nn.Linear modules.

    The exported model is serving-compatible with backends that cannot import TIDAL
    module classes. It preserves the current approximate weights, not the compressed
    storage representation.
    """

    target = model if inplace else deepcopy(model)
    for name, child in list(target.named_children()):
        exported_child = _export_module(child, dtype=dtype, device=device)
        if exported_child is child:
            export_compressed_linears_to_dense(child, inplace=True, dtype=dtype, device=device)
        else:
            setattr(target, name, exported_child)
    return target
