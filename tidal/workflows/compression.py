from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from tidal.data import build_text_calibration_batches, causal_lm_loss, load_calibration_texts
from tidal.methods.global_rank_sparsity.core import CAPCompression
from tidal.methods.global_rank_sparsity.torch import CAPRunResult, run_cap_compression
from tidal.reports import save_summary_json, summarize_cap_run
from tidal.targets import build_pruned_module_name_filter, load_pruner_target_names


@dataclass(frozen=True)
class CompressionResult:
    model: object
    summary: dict[str, object]
    method_result: CAPRunResult

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
) -> CompressionResult:
    loaded_model_id = model_id or "in-memory"
    effective_target_roles = target_roles
    target_model = model
    if target_model is None:
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

    pruner_names: list[str] | None = None
    effective_filter = name_filter
    if pruner_targets is not None:
        pruner_names = (
            load_pruner_target_names(pruner_targets)
            if isinstance(pruner_targets, str | Path)
            else [str(name) for name in pruner_targets]
        )
        pruner_filter = build_pruned_module_name_filter(pruner_names, target_roles=effective_target_roles)
        if effective_filter is None:
            effective_filter = pruner_filter
        else:
            previous_filter = effective_filter
            effective_filter = lambda name: bool(previous_filter(name)) and pruner_filter(name)

    calibration_batches = None
    if calibration_data is not None:
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
        calibration_batches = build_text_calibration_batches(
            tokenizer,
            texts,
            max_length=calibration_max_length,
            batch_size=calibration_batch_size,
            device="cpu",
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
