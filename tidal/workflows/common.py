from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from tidal.data import build_text_calibration_batches, load_calibration_texts
from tidal.targets import build_pruned_module_name_filter, load_pruner_target_names


def load_or_use_causal_lm(
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


def merge_pruner_filter(
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


def build_calibration_batches_from_text(
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
