from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import torch

from tidal.model_support import (
    DEFAULT_TIDAL_TARGET_ROLES,
    LinearModuleInfo,
    ModuleRole,
    build_linear_role_filter,
    build_modern_hf_name_filter,
    build_pruned_module_name_filter,
    canonicalize_module_name,
    classify_linear_module_name,
    compose_name_filter,
    list_linear_modules,
    module_name_from_tensor_name,
    normalize_role,
    normalize_target_roles,
)

__all__ = [
    "DEFAULT_TIDAL_TARGET_ROLES",
    "LinearModuleInfo",
    "ModuleRole",
    "build_linear_role_filter",
    "build_modern_hf_name_filter",
    "build_pruned_module_name_filter",
    "canonicalize_module_name",
    "classify_linear_module_name",
    "compose_name_filter",
    "list_linear_modules",
    "load_pruner_target_names",
    "module_name_from_tensor_name",
    "normalize_role",
    "normalize_target_roles",
]


def load_pruner_target_names(path: str | Path) -> list[str]:
    target_path = Path(path)
    suffix = target_path.suffix.lower()
    if suffix in {".pt", ".pth", ".bin", ".ckpt", ".safetensors"}:
        if suffix == ".safetensors":
            try:
                from safetensors.torch import load_file
            except ImportError as exc:  # pragma: no cover - optional dependency path
                raise ImportError("loading .safetensors pruner targets requires safetensors") from exc
            loaded = load_file(str(target_path), device="cpu")
        else:
            loaded = torch.load(target_path, map_location="cpu")
        return _names_from_loaded_object(loaded)

    if suffix == ".json":
        return _names_from_loaded_object(json.loads(target_path.read_text()))

    if suffix == ".jsonl":
        names: list[str] = []
        for line in target_path.read_text().splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if isinstance(record, str):
                names.append(record)
            elif isinstance(record, Mapping):
                for key in ("name", "module", "module_name", "tensor", "key"):
                    value = record.get(key)
                    if isinstance(value, str) and value.strip():
                        names.append(value.strip())
                        break
        return names

    names = []
    for line in target_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        names.append(stripped)
    return names


def _names_from_loaded_object(value: object) -> list[str]:
    if isinstance(value, Mapping):
        if "state_dict" in value and isinstance(value["state_dict"], Mapping):
            return [str(name) for name in value["state_dict"].keys()]
        if "model" in value and isinstance(value["model"], Mapping):
            return [str(name) for name in value["model"].keys()]
        if "names" in value and isinstance(value["names"], list):
            return [str(name) for name in value["names"]]
        return [str(name) for name in value.keys()]
    if isinstance(value, list | tuple):
        names: list[str] = []
        for item in value:
            if isinstance(item, str):
                names.append(item)
            elif isinstance(item, Mapping):
                for key in ("name", "module", "module_name", "tensor", "key"):
                    candidate = item.get(key)
                    if isinstance(candidate, str) and candidate.strip():
                        names.append(candidate.strip())
                        break
        return names
    raise ValueError(f"unsupported pruner target format: {type(value).__name__}")
