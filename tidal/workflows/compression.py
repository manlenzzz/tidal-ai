from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from tidal.data import causal_lm_loss
from tidal.device import resolve_device, resolve_dtype
from tidal.methods.global_rank_sparsity.core import CAPCompression
from tidal.methods.global_rank_sparsity.torch import run_cap_compression
from tidal.methods.qpruner.torch import (
    QPrunerRun,
    apply_mixed_precision_quantization,
    build_quantization_plan,
    run_qpruner_mixed_precision,
)
from tidal.reports import save_summary_json, summarize_cap_run, summarize_qpruner_run
from tidal.workflows.common import (
    build_calibration_batches_from_text,
    load_or_use_causal_lm,
    merge_pruner_filter,
)


@dataclass(frozen=True)
class CompressionResult:
    model: object
    summary: dict[str, object]
    method_result: object

    def save(self, output_dir: str | Path) -> Path:
        return save_summary_json(self.summary, output_dir)


def cap_compress(
    *,
    model: object | None = None,
    model_id: str | None = None,
    tokenizer: object | None = None,
    cache_dir: str | None = None,
    local_files_only: bool = False,
    trust_remote_code: bool = False,
    revision: str | None = None,
    budget: int,
    pruner_targets: str | Path | Sequence[str] | None = None,
    calibration_data: str | Path | Sequence[str] | None = None,
    calibration_text_field: str = "text",
    calibration_max_samples: int = 8,
    calibration_max_length: int = 128,
    calibration_batch_size: int = 1,
    target_roles: object | None = None,
    name_filter: Callable[[str], bool] | None = None,
    evaluator: Callable[[dict[str, CAPCompression]], float] | None = None,
    max_iter: int = 20,
    policy_steps: int = 4,
    samples_per_step: int = 2,
    seed: int | None = 0,
    inplace: bool = False,
    device: object | None = None,
    dtype: object | None = None,
    rpca_backend: str = "numpy",
) -> CompressionResult:
    resolved_device = resolve_device(device)
    resolved_dtype = resolve_dtype(dtype, resolved_device)
    target_model, loaded_model_id, effective_target_roles = load_or_use_causal_lm(
        model=model,
        model_id=model_id,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        revision=revision,
        target_roles=target_roles,
        device=resolved_device,
        dtype=resolved_dtype,
    )
    pruner_names, effective_filter = merge_pruner_filter(
        pruner_targets=pruner_targets,
        target_roles=effective_target_roles,
        name_filter=name_filter,
    )
    calibration_batches = build_calibration_batches_from_text(
        calibration_data=calibration_data,
        calibration_text_field=calibration_text_field,
        calibration_max_samples=calibration_max_samples,
        calibration_max_length=calibration_max_length,
        calibration_batch_size=calibration_batch_size,
        tokenizer=tokenizer,
        model_id=model_id,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        device=resolved_device,
    )

    method_result = run_cap_compression(
        target_model,
        total_budget=budget,
        inplace=inplace,
        name_filter=effective_filter,
        target_roles=effective_target_roles,
        calibration_batches=calibration_batches,
        loss_fn=causal_lm_loss if calibration_batches is not None else None,
        evaluator=evaluator,
        max_iter=max_iter,
        policy_steps=policy_steps,
        samples_per_step=samples_per_step,
        seed=seed,
        rpca_backend=rpca_backend,
        rpca_device=resolved_device,
    )
    summary = summarize_cap_run(
        method_result,
        model_id=loaded_model_id,
        target_roles=effective_target_roles,
        pruner_target_count=len(pruner_names) if pruner_names is not None else None,
        calibration_batches=len(calibration_batches) if calibration_batches is not None else None,
    )
    return CompressionResult(method_result.compressed_model, summary, method_result)


def qpruner_compress(
    *,
    model: object | None = None,
    model_id: str | None = None,
    tokenizer: object | None = None,
    cache_dir: str | None = None,
    local_files_only: bool = False,
    trust_remote_code: bool = False,
    revision: str | None = None,
    pruner_targets: str | Path | Sequence[str] | None = None,
    calibration_batches: Sequence[object] | None = None,
    calibration_data: str | Path | Sequence[str] | None = None,
    calibration_text_field: str = "text",
    calibration_max_samples: int = 8,
    calibration_max_length: int = 128,
    calibration_batch_size: int = 1,
    importances: Mapping[str, float] | None = None,
    candidate_bits: Sequence[int] = (2, 4, 8),
    max_memory_bits: int | None = None,
    max_average_bits: float | None = None,
    objective: Callable[[dict[str, int]], float] | None = None,
    refine_trials: int = 0,
    prediction_fn: Callable[[object], object] | None = None,
    bins: int = 16,
    target_roles: object | None = None,
    name_filter: Callable[[str], bool] | None = None,
    seed: int | None = 0,
    inplace: bool = False,
    device: object | None = None,
    dtype: object | None = None,
) -> CompressionResult:
    if max_memory_bits is None and max_average_bits is None:
        raise ValueError("provide max_memory_bits or max_average_bits")
    if importances is None and calibration_batches is None and calibration_data is None:
        raise ValueError("importances, calibration_batches, or calibration_data is required")

    resolved_device = resolve_device(device)
    resolved_dtype = resolve_dtype(dtype, resolved_device)
    target_model, loaded_model_id, effective_target_roles = load_or_use_causal_lm(
        model=model,
        model_id=model_id,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        revision=revision,
        target_roles=target_roles,
        device=resolved_device,
        dtype=resolved_dtype,
    )
    pruner_names, effective_filter = merge_pruner_filter(
        pruner_targets=pruner_targets,
        target_roles=effective_target_roles,
        name_filter=name_filter,
    )

    prepared_batches = list(calibration_batches) if calibration_batches is not None else None
    if importances is None and prepared_batches is None:
        prepared_batches = build_calibration_batches_from_text(
            calibration_data=calibration_data,
            calibration_text_field=calibration_text_field,
            calibration_max_samples=calibration_max_samples,
            calibration_max_length=calibration_max_length,
            calibration_batch_size=calibration_batch_size,
            tokenizer=tokenizer,
            model_id=model_id,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            device=resolved_device,
        )

    if importances is not None:
        prepared_batches = None
        normalized_importances = {str(name): float(value) for name, value in importances.items()}
        plan = build_quantization_plan(
            target_model,
            normalized_importances,
            candidate_bits=candidate_bits,
            max_memory_bits=max_memory_bits,
            max_average_bits=max_average_bits,
            objective=objective,
            refine_trials=refine_trials,
            name_filter=effective_filter,
            target_roles=effective_target_roles,
            seed=seed,
        )
        quantized_model = apply_mixed_precision_quantization(target_model, plan, inplace=inplace)
        method_result = QPrunerRun(quantized_model, plan, normalized_importances)
    else:
        if prepared_batches is None:
            raise ValueError("calibration batches are required when importances are not provided")
        method_result = run_qpruner_mixed_precision(
            target_model,
            prepared_batches,
            candidate_bits=candidate_bits,
            max_memory_bits=max_memory_bits,
            max_average_bits=max_average_bits,
            objective=objective,
            refine_trials=refine_trials,
            prediction_fn=prediction_fn,
            bins=bins,
            name_filter=effective_filter,
            target_roles=effective_target_roles,
            inplace=inplace,
            seed=seed,
        )

    summary = summarize_qpruner_run(
        method_result,
        model_id=loaded_model_id,
        target_roles=effective_target_roles,
        pruner_target_count=len(pruner_names) if pruner_names is not None else None,
        calibration_batches=len(prepared_batches) if prepared_batches is not None else None,
    )
    return CompressionResult(method_result.model, summary, method_result)
