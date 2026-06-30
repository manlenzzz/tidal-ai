from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

from tidal.methods.rankadaptor.core import (
    ModuleProfile,
    PerformanceModel,
    build_lora_config,
    collect_linear_profiles,
    search_rank_allocation,
)
from tidal.reports import save_summary_json, summarize_rankadaptor_run
from tidal.targets import canonicalize_module_name
from tidal.workflows.common import load_or_use_causal_lm, merge_pruner_filter
from tidal.device import resolve_device, resolve_dtype


@dataclass(frozen=True)
class AdaptationResult:
    model: object
    summary: dict[str, object]
    method_result: object
    peft_config: object | None
    profiles: tuple[ModuleProfile, ...]

    def save(self, output_dir: str | Path) -> Path:
        return save_summary_json(self.summary, output_dir)


def _canonical_sensitivity_name(name: object) -> str:
    return canonicalize_module_name(str(name))


def load_sensitivity_scores(source: str | Path | Mapping[str, float] | None) -> dict[str, float] | None:
    if source is None:
        return None
    if isinstance(source, Mapping):
        return {_canonical_sensitivity_name(name): float(score) for name, score in source.items()}

    path = Path(source)
    suffix = path.suffix.lower()
    if suffix == ".json":
        loaded = json.loads(path.read_text())
        if isinstance(loaded, Mapping):
            return {_canonical_sensitivity_name(name): float(score) for name, score in loaded.items()}
        if isinstance(loaded, list):
            scores: dict[str, float] = {}
            for item in loaded:
                if not isinstance(item, Mapping):
                    continue
                name = item.get("name") or item.get("module") or item.get("module_name")
                score = item.get("sensitivity") if "sensitivity" in item else item.get("score")
                if name is not None and score is not None:
                    scores[_canonical_sensitivity_name(name)] = float(score)
            return scores
        raise ValueError("JSON sensitivities must be a mapping or list of records")

    if suffix == ".jsonl":
        scores: dict[str, float] = {}
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            item = json.loads(stripped)
            if not isinstance(item, Mapping):
                continue
            name = item.get("name") or item.get("module") or item.get("module_name")
            score = item.get("sensitivity") if "sensitivity" in item else item.get("score")
            if name is not None and score is not None:
                scores[_canonical_sensitivity_name(name)] = float(score)
        return scores

    scores: dict[str, float] = {}
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        normalized = stripped.replace(",", " ")
        parts = normalized.split()
        if len(parts) < 2:
            raise ValueError(f"invalid sensitivity line: {line!r}")
        scores[_canonical_sensitivity_name(parts[0])] = float(parts[1])
    return scores


def rankadaptor_adapt(
    *,
    model: object | None = None,
    model_id: str | None = None,
    cache_dir: str | None = None,
    local_files_only: bool = False,
    trust_remote_code: bool = False,
    revision: str | None = None,
    pruner_targets: str | Path | Sequence[str] | None = None,
    sensitivities: str | Path | Mapping[str, float] | None = None,
    budget: int,
    min_rank: int = 1,
    max_rank: int = 64,
    rank_step: int = 1,
    target_roles: object | None = None,
    name_filter: Callable[[str], bool] | None = None,
    performance_model: PerformanceModel | None = None,
    max_steps: int | None = None,
    min_gain: float = 0.0,
    alpha_multiplier: int = 2,
    apply_peft: bool = True,
    inplace: bool = False,
    seed: int | None = 0,
    peft_kwargs: Mapping[str, object] | None = None,
    device: object | None = None,
    dtype: object | None = None,
) -> AdaptationResult:
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
    sensitivity_map = load_sensitivity_scores(sensitivities)
    profiles = collect_linear_profiles(
        target_model,
        sensitivities=sensitivity_map,
        min_rank=min_rank,
        max_rank=max_rank,
        rank_step=rank_step,
        name_filter=effective_filter,
        target_roles=effective_target_roles,
    )
    search_result = search_rank_allocation(
        profiles,
        budget=budget,
        performance_model=performance_model,
        max_steps=max_steps,
        min_gain=min_gain,
    )
    output_model = target_model
    lora_config = None
    if apply_peft:
        lora_config = build_lora_config(
            search_result.config,
            alpha_multiplier=alpha_multiplier,
            **dict(peft_kwargs or {}),
        )
        from peft import get_peft_model

        peft_base = target_model if inplace else deepcopy(target_model)
        output_model = get_peft_model(peft_base, lora_config)

    summary = summarize_rankadaptor_run(
        search_result,
        profiles=profiles,
        model_id=loaded_model_id,
        target_roles=effective_target_roles,
        pruner_target_count=len(pruner_names) if pruner_names is not None else None,
        search_type="allocation",
    )
    return AdaptationResult(output_model, summary, search_result, lora_config, tuple(profiles))
