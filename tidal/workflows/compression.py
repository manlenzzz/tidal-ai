from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from tidal.data import build_text_calibration_batches, causal_lm_loss, load_calibration_texts
from tidal.methods.global_rank_sparsity.core import CAPCompression
from tidal.methods.global_rank_sparsity.torch import run_cap_compression
from tidal.methods.qpruner.torch import (
    QPrunerRun,
    apply_mixed_precision_quantization,
    build_quantization_plan,
    run_qpruner_mixed_precision,
)
from tidal.reports import save_summary_json, summarize_cap_run, summarize_qpruner_run
from tidal.targets import build_pruned_module_name_filter, load_pruner_target_names


@dataclass(frozen=True)
class CompressionResult:
    model: object
    summary: dict[str, object]
    method_result: object

    def save(self, output_dir: str | Path) -> Path:
        return save_summary_json(self.summary, output_dir)


def _load_or_use_model(
    *,
    model: object | None,
    model_id: str | None,
    cache_dir: str | None,
    local_files_only: bool,
    trust_remote_code: bool,
    revision: str | None,
    target_roles: object | None,
) -> tuple[object, str, object | None]:
    loaded_model_id = model_id or "in-memory"
    effective_target_roles = target_roles
    if model is not None:
        return model, loaded_model_id, effective_target_roles
    if model_id is None:
        raise ValueError("either model or model_id is required")

    from transformers import AutoModelForCausalLM

    load_kwargs: dict[str, object] = {
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
        "trust_remote_code": trust_remote_code,
        "torch_dtype": "auto",
    }
    if revision:
        load_kwargs["revision"] = revision
    target_model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    target_model.eval()
    if effective_target_roles is None:
        effective_target_roles = "modern"
    return target_model, loaded_model_id, effective_target_roles


def _merge_pruner_filter(
    *,
    pruner_targets: str | Path | Sequence[str] | None,
    target_roles: object | None,
    name_filter: Callable[[str], bool] | None,
) -> tuple[list[str] | None, Callable[[str], bool] | None]:
    if pruner_targets is None:
        return None, name_filter

    pruner_names = (
        load_pruner_target_names(pruner_targets)
        if isinstance(pruner_targets, str | Path)
        else [str(name) for name in pruner_targets]
    )
    pruner_filter = build_pruned_module_name_filter(pruner_names, target_roles=target_roles)
    if name_filter is None:
        return pruner_names, pruner_filter

    def combined_filter(name: str) -> bool:
        return bool(name_filter(name)) and pruner_filter(name)

    return pruner_names, combined_filter


def _build_calibration_batches(
    *,
    calibration_data: str | Path | Sequence[str] | None,
    calibration_text_field: str,
    calibration_max_samples: int,
    calibration_max_length: int,
    calibration_batch_size: int,
    tokenizer: object | None,
    model_id: str | None,
    cache_dir: str | None,
    local_files_only: bool,
    trust_remote_code: bool,
) -> list[object] | None:
    if calibration_data is None:
        return None
    if isinstance(calibration_data, str | Path):
        texts = load_calibration_texts(calibration_data, text_field=calibration_text_field)
    else:
        texts = [str(text) for text in calibration_data]
    texts = texts[: max(0, calibration_max_samples)]
    if tokenizer is None:
        if model_id is None:
            raise ValueError("tokenizer is required when calibration_data is used with an in-memory model")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
    return build_text_calibration_batches(
        tokenizer,
        texts,
        max_length=calibration_max_length,
        batch_size=calibration_batch_size,
        device="cpu",
    )


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
) -> CompressionResult:
    target_model, loaded_model_id, effective_target_roles = _load_or_use_model(
        model=model,
        model_id=model_id,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        revision=revision,
        target_roles=target_roles,
    )
    pruner_names, effective_filter = _merge_pruner_filter(
        pruner_targets=pruner_targets,
        target_roles=effective_target_roles,
        name_filter=name_filter,
    )
    calibration_batches = _build_calibration_batches(
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
) -> CompressionResult:
    if max_memory_bits is None and max_average_bits is None:
        raise ValueError("provide max_memory_bits or max_average_bits")
    if importances is None and calibration_batches is None and calibration_data is None:
        raise ValueError("importances, calibration_batches, or calibration_data is required")

    target_model, loaded_model_id, effective_target_roles = _load_or_use_model(
        model=model,
        model_id=model_id,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        revision=revision,
        target_roles=target_roles,
    )
    pruner_names, effective_filter = _merge_pruner_filter(
        pruner_targets=pruner_targets,
        target_roles=effective_target_roles,
        name_filter=name_filter,
    )

    prepared_batches = list(calibration_batches) if calibration_batches is not None else None
    if importances is None and prepared_batches is None:
        prepared_batches = _build_calibration_batches(
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
