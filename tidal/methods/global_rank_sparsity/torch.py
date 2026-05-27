from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tidal.model_support import canonicalize_module_name, classify_linear_module_name, compose_name_filter
from tidal.methods.global_rank_sparsity.core import (
    CAPCompression,
    CAPGlobalSearchResult,
    compress_global_rank_sparsity,
    optimize_global_rank_sparsity,
    optimize_global_rank_sparsity_for_matrices,
)

__all__ = [
    "CAPLayerResult",
    "CAPRunResult",
    "CAPTarget",
    "CAPPackedLinear",
    "apply_cap_compression",
    "apply_global_cap_compression",
    "build_cap_calibration_evaluator",
    "collect_cap_targets",
    "run_cap_compression",
    "summarize_cap_layers",
]


@dataclass(frozen=True)
class CAPLayerResult:
    name: str
    parameter_count: int
    rank: int
    sparse_entries: int


@dataclass(frozen=True)
class CAPTarget:
    name: str
    module: nn.Linear
    canonical_name: str
    role: object
    in_features: int
    out_features: int
    parameter_count: int

    def identity_compression(self) -> CAPCompression:
        weight = self.module.weight.detach().cpu().to(torch.float32).numpy()
        return CAPCompression(
            low_rank=weight,
            sparse=np.zeros_like(weight),
            rank_mask=np.ones(min(weight.shape), dtype=bool),
            sparse_mask=np.zeros_like(weight, dtype=bool),
            parameter_count=int(weight.size),
        )


@dataclass(frozen=True)
class CAPRunResult:
    compressed_model: nn.Module
    targets: tuple[CAPTarget, ...]
    global_result: CAPGlobalSearchResult
    layer_summaries: tuple[CAPLayerResult, ...]


class CAPPackedLinear(nn.Module):
    """Linear layer represented as low-rank factors plus a sparse residual."""

    def __init__(
        self,
        low_rank_left: torch.Tensor,
        low_rank_right: torch.Tensor,
        sparse: torch.Tensor,
        bias: torch.Tensor | None,
        *,
        parameter_count: int,
    ):
        super().__init__()
        if low_rank_left.ndim != 2 or low_rank_right.ndim != 2 or sparse.ndim != 2:
            raise ValueError("CAP tensors must be matrices")
        self.in_features = int(sparse.shape[1])
        self.out_features = int(sparse.shape[0])
        self.parameter_count = int(parameter_count)
        self.rank = int(low_rank_right.shape[0])
        self.sparse_entries = int(torch.count_nonzero(sparse).item())
        self.register_buffer("low_rank_left", low_rank_left.to(torch.float32))
        self.register_buffer("low_rank_right", low_rank_right.to(torch.float32))
        self.register_buffer("sparse", sparse.to(torch.float32))
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)

    @classmethod
    def from_linear(
        cls,
        module: nn.Linear,
        *,
        budget: int,
        max_iter: int = 200,
        use_policy: bool = True,
        policy_steps: int = 80,
        samples_per_step: int = 8,
        seed: int | None = None,
    ) -> "CAPPackedLinear":
        weight = module.weight.detach().cpu().to(torch.float32).numpy()
        if use_policy:
            compression = optimize_global_rank_sparsity(
                weight,
                budget=budget,
                max_iter=max_iter,
                policy_steps=policy_steps,
                samples_per_step=samples_per_step,
                seed=seed,
            ).compression
        else:
            compression = compress_global_rank_sparsity(weight, budget=budget, max_iter=max_iter)
        return cls.from_compression(module, compression)

    @classmethod
    def from_compression(cls, module: nn.Linear, compression: CAPCompression) -> "CAPPackedLinear":
        left, right = _factorize_low_rank(compression)
        sparse = torch.from_numpy(compression.sparse).to(dtype=module.weight.dtype, device=module.weight.device)
        return cls(
            left.to(dtype=module.weight.dtype, device=module.weight.device),
            right.to(dtype=module.weight.dtype, device=module.weight.device),
            sparse,
            module.bias,
            parameter_count=compression.parameter_count,
        )

    def reconstructed_weight(self) -> torch.Tensor:
        if self.rank == 0:
            low_rank = torch.zeros_like(self.sparse)
        else:
            low_rank = self.low_rank_left @ self.low_rank_right
        return low_rank + self.sparse

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight = self.reconstructed_weight().to(dtype=inputs.dtype, device=inputs.device)
        bias = self.bias.to(dtype=inputs.dtype, device=inputs.device) if self.bias is not None else None
        return F.linear(inputs, weight, bias)


def _factorize_low_rank(compression: CAPCompression) -> tuple[torch.Tensor, torch.Tensor]:
    low_rank = np.asarray(compression.low_rank, dtype=np.float32)
    if low_rank.size == 0 or not np.any(low_rank):
        rows, cols = low_rank.shape
        return torch.zeros((rows, 0), dtype=torch.float32), torch.zeros((0, cols), dtype=torch.float32)
    u, sigma, vt = np.linalg.svd(low_rank, full_matrices=False)
    keep = sigma > 1e-8
    if not np.any(keep):
        rows, cols = low_rank.shape
        return torch.zeros((rows, 0), dtype=torch.float32), torch.zeros((0, cols), dtype=torch.float32)
    left = torch.from_numpy((u[:, keep] * sigma[keep]).astype(np.float32))
    right = torch.from_numpy(vt[keep, :].astype(np.float32))
    return left, right


def _set_submodule(model: nn.Module, name: str, module: nn.Module) -> None:
    parent = model
    parts = name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], module)


def _get_submodule(model: nn.Module, name: str) -> nn.Module:
    module: nn.Module = model
    for part in name.split("."):
        module = getattr(module, part)
    return module


def collect_cap_targets(
    model: nn.Module,
    *,
    name_filter: Callable[[str], bool] | None = None,
    target_roles: object | None = None,
    exclude_target_roles: object | None = None,
) -> tuple[CAPTarget, ...]:
    selected_name_filter = compose_name_filter(
        name_filter,
        target_roles=target_roles,
        exclude_roles=exclude_target_roles,
    )
    targets: list[CAPTarget] = []
    for name, module in model.named_modules():
        if not name or not isinstance(module, nn.Linear):
            continue
        if selected_name_filter is not None and not selected_name_filter(name):
            continue
        canonical_name = canonicalize_module_name(name)
        targets.append(
            CAPTarget(
                name=name,
                module=module,
                canonical_name=canonical_name,
                role=classify_linear_module_name(canonical_name),
                in_features=int(module.in_features),
                out_features=int(module.out_features),
                parameter_count=int(module.weight.numel()),
            )
        )
    return tuple(targets)


def _patch_modules(model: nn.Module, modules: Mapping[str, nn.Module]) -> dict[str, nn.Module]:
    originals: dict[str, nn.Module] = {}
    for name, module in modules.items():
        originals[name] = _get_submodule(model, name)
        _set_submodule(model, name, module)
    return originals


def build_cap_calibration_evaluator(
    model: nn.Module,
    targets: tuple[CAPTarget, ...] | list[CAPTarget],
    calibration_batches: object,
    loss_fn: Callable[[nn.Module, object], torch.Tensor | float],
) -> Callable[[dict[str, CAPCompression]], float]:
    target_by_name = {target.name: target for target in targets}
    batches = tuple(calibration_batches)
    if not batches:
        raise ValueError("calibration_batches must not be empty")

    def evaluator(compressions: dict[str, CAPCompression]) -> float:
        packed_modules: dict[str, nn.Module] = {}
        for name, compression in compressions.items():
            if name not in target_by_name:
                continue
            packed_modules[name] = CAPPackedLinear.from_compression(target_by_name[name].module, compression)
        originals = _patch_modules(model, packed_modules)
        was_training = model.training
        model.eval()
        try:
            losses: list[float] = []
            with torch.no_grad():
                for batch in batches:
                    loss = loss_fn(model, batch)
                    loss_value = float(loss.detach().cpu().item()) if isinstance(loss, torch.Tensor) else float(loss)
                    losses.append(loss_value)
            return float(np.mean(losses))
        finally:
            _patch_modules(model, originals)
            model.train(was_training)

    return evaluator


def apply_cap_compression(
    model: nn.Module,
    budgets: Mapping[str, int],
    *,
    inplace: bool = False,
    max_iter: int = 200,
    use_policy: bool = True,
    policy_steps: int = 80,
    samples_per_step: int = 8,
    seed: int | None = None,
) -> nn.Module:
    target = model if inplace else deepcopy(model)
    modules = dict(target.named_modules())
    missing = sorted(set(budgets) - set(modules))
    if missing:
        raise ValueError(f"unknown linear layers in budgets: {', '.join(missing)}")
    for name, budget in budgets.items():
        module = modules[name]
        if not isinstance(module, nn.Linear):
            raise ValueError(f"{name} is not a torch.nn.Linear module")
        packed = CAPPackedLinear.from_linear(
            module,
            budget=int(budget),
            max_iter=max_iter,
            use_policy=use_policy,
            policy_steps=policy_steps,
            samples_per_step=samples_per_step,
            seed=seed,
        )
        _set_submodule(target, name, packed)
    return target


def _pack_global_result(
    target_model: nn.Module,
    modules: Mapping[str, nn.Linear],
    result: CAPGlobalSearchResult,
) -> None:
    for name, compression in result.compressions.items():
        _set_submodule(target_model, name, CAPPackedLinear.from_compression(modules[name], compression))


def run_cap_compression(
    model: nn.Module,
    *,
    total_budget: int,
    inplace: bool = False,
    name_filter: Callable[[str], bool] | None = None,
    target_roles: object | None = None,
    exclude_target_roles: object | None = None,
    calibration_batches: object | None = None,
    loss_fn: Callable[[nn.Module, object], torch.Tensor | float] | None = None,
    evaluator: Callable[[dict[str, CAPCompression]], float] | None = None,
    max_iter: int = 200,
    policy_steps: int = 80,
    samples_per_step: int = 8,
    seed: int | None = None,
) -> CAPRunResult:
    target = model if inplace else deepcopy(model)
    targets = collect_cap_targets(
        target,
        name_filter=name_filter,
        target_roles=target_roles,
        exclude_target_roles=exclude_target_roles,
    )
    if not targets:
        raise ValueError("model does not contain matching torch.nn.Linear modules")
    modules = {target_info.name: target_info.module for target_info in targets}
    matrices = {name: module.weight.detach().cpu().to(torch.float32).numpy() for name, module in modules.items()}
    score_fn = evaluator
    if score_fn is None and calibration_batches is not None:
        if loss_fn is None:
            raise ValueError("loss_fn is required when calibration_batches are supplied")
        score_fn = build_cap_calibration_evaluator(target, targets, calibration_batches, loss_fn)
    result = optimize_global_rank_sparsity_for_matrices(
        matrices,
        total_budget=total_budget,
        evaluator=score_fn,
        max_iter=max_iter,
        policy_steps=policy_steps,
        samples_per_step=samples_per_step,
        seed=seed,
    )
    _pack_global_result(target, modules, result)
    target._cap_global_result = result
    return CAPRunResult(
        compressed_model=target,
        targets=targets,
        global_result=result,
        layer_summaries=tuple(summarize_cap_layers(target)),
    )


def apply_global_cap_compression(
    model: nn.Module,
    *,
    total_budget: int,
    inplace: bool = False,
    name_filter: Callable[[str], bool] | None = None,
    target_roles: object | None = None,
    exclude_target_roles: object | None = None,
    evaluator: Callable[[dict[str, CAPCompression]], float] | None = None,
    max_iter: int = 200,
    policy_steps: int = 80,
    samples_per_step: int = 8,
    seed: int | None = None,
) -> nn.Module:
    return run_cap_compression(
        model,
        total_budget=total_budget,
        inplace=inplace,
        name_filter=name_filter,
        target_roles=target_roles,
        exclude_target_roles=exclude_target_roles,
        evaluator=evaluator,
        max_iter=max_iter,
        policy_steps=policy_steps,
        samples_per_step=samples_per_step,
        seed=seed,
    ).compressed_model


def summarize_cap_layers(model: nn.Module) -> list[CAPLayerResult]:
    results: list[CAPLayerResult] = []
    for name, module in model.named_modules():
        if isinstance(module, CAPPackedLinear):
            results.append(CAPLayerResult(name, module.parameter_count, module.rank, module.sparse_entries))
    return results
