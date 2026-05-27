from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable, Mapping

NameFilter = Callable[[str], bool]
RoleLike = "ModuleRole | str"


class ModuleRole(str, Enum):
    """Stable roles for linear modules in modern Hugging Face-style LLMs."""

    ATTENTION = "attention"
    MLP = "mlp"
    MOE_EXPERT = "moe_expert"
    MOE_ROUTER = "moe_router"
    OUTPUT_HEAD = "output_head"
    EMBEDDING_PROJECTION = "embedding_projection"
    OTHER = "other"


DEFAULT_TIDAL_TARGET_ROLES = frozenset(
    {
        ModuleRole.ATTENTION,
        ModuleRole.MLP,
        ModuleRole.MOE_EXPERT,
    }
)
UNSAFE_DEFAULT_EXCLUDED_ROLES = frozenset(
    {
        ModuleRole.MOE_ROUTER,
        ModuleRole.OUTPUT_HEAD,
        ModuleRole.EMBEDDING_PROJECTION,
    }
)
PRUNED_MODEL_TARGET_PRESETS = frozenset(
    {
        "llm_pruner",
        "llmpruner",
        "llm_pruning",
        "pruned",
        "pruner",
        "wanda",
    }
)
WRAPPER_PREFIX_REWRITES = (
    ("base_model.model.", ""),
    ("base_model.", ""),
    ("module.", ""),
    ("_orig_mod.", ""),
    ("wrapped_model.", ""),
    ("pruned_model.", ""),
    ("model.model.", "model."),
)
TENSOR_NAME_SUFFIXES = frozenset(
    {
        "absmax",
        "bias",
        "bias_mask",
        "col_mask",
        "g_idx",
        "mask",
        "qweight",
        "qzeros",
        "row_mask",
        "scales",
        "scaler_row",
        "sparse_mask",
        "weight",
        "weight_mask",
        "zeros",
    }
)


@dataclass(frozen=True)
class LinearModuleInfo:
    name: str
    module: object
    role: ModuleRole
    in_features: int
    out_features: int
    canonical_name: str = ""


__all__ = [
    "DEFAULT_TIDAL_TARGET_ROLES",
    "PRUNED_MODEL_TARGET_PRESETS",
    "TENSOR_NAME_SUFFIXES",
    "UNSAFE_DEFAULT_EXCLUDED_ROLES",
    "LinearModuleInfo",
    "ModuleRole",
    "build_linear_role_filter",
    "build_modern_hf_name_filter",
    "build_pruned_module_name_filter",
    "canonicalize_module_name",
    "classify_linear_module_name",
    "compose_name_filter",
    "list_linear_modules",
    "module_name_from_tensor_name",
    "normalize_role",
    "normalize_target_roles",
]


def _torch_nn() -> object:
    try:
        from torch import nn
    except ImportError as exc:  # pragma: no cover - exercised only without torch installed
        raise ImportError("modern model support requires PyTorch") from exc
    return nn


def normalize_role(role: ModuleRole | str) -> ModuleRole:
    if isinstance(role, ModuleRole):
        return role
    value = str(role).strip().lower().replace("-", "_")
    aliases = {
        "attn": ModuleRole.ATTENTION,
        "attention": ModuleRole.ATTENTION,
        "ffn": ModuleRole.MLP,
        "feed_forward": ModuleRole.MLP,
        "feedforward": ModuleRole.MLP,
        "mlp": ModuleRole.MLP,
        "expert": ModuleRole.MOE_EXPERT,
        "experts": ModuleRole.MOE_EXPERT,
        "moe_expert": ModuleRole.MOE_EXPERT,
        "router": ModuleRole.MOE_ROUTER,
        "moe_router": ModuleRole.MOE_ROUTER,
        "gate": ModuleRole.MOE_ROUTER,
        "head": ModuleRole.OUTPUT_HEAD,
        "output_head": ModuleRole.OUTPUT_HEAD,
        "embedding": ModuleRole.EMBEDDING_PROJECTION,
        "embedding_projection": ModuleRole.EMBEDDING_PROJECTION,
        "other": ModuleRole.OTHER,
    }
    if value in aliases:
        return aliases[value]
    try:
        return ModuleRole(value)
    except ValueError as exc:
        valid = ", ".join(role.value for role in ModuleRole)
        raise ValueError(f"unknown module role {role!r}; expected one of: {valid}") from exc


def normalize_target_roles(target_roles: Iterable[ModuleRole | str] | ModuleRole | str | None) -> frozenset[ModuleRole] | None:
    if target_roles is None:
        return None
    if isinstance(target_roles, (str, ModuleRole)):
        value = str(target_roles).strip().lower().replace("-", "_")
        if value in {"modern", "default", "tidal", "method", "methods"} | PRUNED_MODEL_TARGET_PRESETS:
            return DEFAULT_TIDAL_TARGET_ROLES
        return frozenset({normalize_role(target_roles)})
    return frozenset(normalize_role(role) for role in target_roles)


def _strip_wrapper_prefixes(name: str) -> str:
    normalized = name
    changed = True
    while changed:
        changed = False
        for prefix, replacement in WRAPPER_PREFIX_REWRITES:
            if normalized.startswith(prefix):
                normalized = replacement + normalized[len(prefix) :]
                changed = True
                break
    return normalized


def module_name_from_tensor_name(name: str) -> str:
    """Return a module path from a module, mask, parameter, or checkpoint key."""

    parts = [part for part in str(name).strip().split(".") if part]
    while len(parts) > 1 and parts[-1].lower() in TENSOR_NAME_SUFFIXES:
        parts.pop()
    return _strip_wrapper_prefixes(".".join(parts))


def canonicalize_module_name(name: str) -> str:
    """Canonicalize HF/pruner wrapper names while preserving structural paths."""

    return module_name_from_tensor_name(name)


def _name_parts(name: str) -> tuple[list[str], str, str]:
    lower = name.lower()
    parts = [part for part in lower.split(".") if part]
    base = parts[-1] if parts else lower
    return parts, base, lower


def _has_any(text: str, needles: Iterable[str]) -> bool:
    return any(needle in text for needle in needles)


def _path_has_any(parts: list[str], needles: Iterable[str]) -> bool:
    values = set(parts)
    return any(needle in values for needle in needles)


def classify_linear_module_name(name: str) -> ModuleRole:
    """Classify a linear module path into a stable modern-LLM role."""

    parts, base, lower = _name_parts(canonicalize_module_name(name))

    if base in {"lm_head", "language_model_head", "output_head"}:
        return ModuleRole.OUTPUT_HEAD
    if base in {"embed_out", "embed_proj", "emb_proj", "embedding_projection"}:
        return ModuleRole.EMBEDDING_PROJECTION
    if "embed" in lower and "proj" in base:
        return ModuleRole.EMBEDDING_PROJECTION

    moe_context = _has_any(lower, {"moe", "expert", "experts", "shared_expert", "shared_experts"})
    router_context = _has_any(lower, {"router", "route", "routing"})
    if router_context or (moe_context and base in {"gate", "gate_proj", "router", "router_proj"} and "experts" not in parts):
        return ModuleRole.MOE_ROUTER

    if _path_has_any(parts, {"experts", "expert", "shared_experts", "shared_expert"}) or ".experts." in lower:
        return ModuleRole.MOE_EXPERT

    attention_context = _has_any(lower, {"self_attn", ".attn.", ".attention.", "attention", "cross_attn"})
    attention_bases = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "out_proj",
        "qkv_proj",
        "wqkv",
        "w_qkv",
        "c_attn",
        "query_key_value",
        "qkv",
        "query",
        "key",
        "value",
    }
    if attention_context and base in attention_bases:
        return ModuleRole.ATTENTION
    if base in {"qkv_proj", "wqkv", "w_qkv", "c_attn", "query_key_value"}:
        return ModuleRole.ATTENTION

    mlp_context = _has_any(lower, {".mlp.", ".ffn.", ".ff.", "feed_forward", "feedforward"})
    mlp_bases = {"gate_proj", "up_proj", "down_proj", "fc1", "fc2", "w1", "w2", "w3", "c_fc", "c_proj"}
    if mlp_context and base in mlp_bases:
        return ModuleRole.MLP

    return ModuleRole.OTHER


def _role_is_selected(role: ModuleRole, include: frozenset[ModuleRole] | None, exclude: frozenset[ModuleRole] | None) -> bool:
    if include is not None and role not in include:
        return False
    if exclude is not None and role in exclude:
        return False
    return True


def build_linear_role_filter(
    *,
    target_roles: Iterable[ModuleRole | str] | ModuleRole | str | None = None,
    exclude_roles: Iterable[ModuleRole | str] | ModuleRole | str | None = None,
) -> NameFilter:
    include = normalize_target_roles(target_roles)
    exclude = normalize_target_roles(exclude_roles)

    def name_filter(name: str) -> bool:
        return _role_is_selected(classify_linear_module_name(name), include, exclude)

    return name_filter


def build_modern_hf_name_filter(
    *,
    target_roles: Iterable[ModuleRole | str] | ModuleRole | str | None = "modern",
    exclude_roles: Iterable[ModuleRole | str] | ModuleRole | str | None = None,
) -> NameFilter:
    return build_linear_role_filter(target_roles=target_roles, exclude_roles=exclude_roles)


def build_pruned_module_name_filter(
    names_or_state: Iterable[str] | Mapping[str, object],
    *,
    target_roles: Iterable[ModuleRole | str] | ModuleRole | str | None = "modern",
    exclude_roles: Iterable[ModuleRole | str] | ModuleRole | str | None = None,
) -> NameFilter:
    """Build a selector from LLM-Pruner/WANDA module lists, masks, or state dicts."""

    names = names_or_state.keys() if isinstance(names_or_state, Mapping) else names_or_state
    canonical_names = frozenset(canonicalize_module_name(name) for name in names if str(name).strip())
    role_filter = build_linear_role_filter(target_roles=target_roles, exclude_roles=exclude_roles)

    def name_filter(name: str) -> bool:
        canonical_name = canonicalize_module_name(name)
        return canonical_name in canonical_names and role_filter(canonical_name)

    return name_filter


def compose_name_filter(
    name_filter: NameFilter | None = None,
    *,
    target_roles: Iterable[ModuleRole | str] | ModuleRole | str | None = None,
    exclude_roles: Iterable[ModuleRole | str] | ModuleRole | str | None = None,
) -> NameFilter | None:
    if target_roles is None and exclude_roles is None:
        return name_filter
    role_filter = build_linear_role_filter(target_roles=target_roles, exclude_roles=exclude_roles)
    if name_filter is None:
        return role_filter

    def combined(name: str) -> bool:
        return bool(name_filter(name)) and role_filter(name)

    return combined


def list_linear_modules(
    model: object,
    *,
    target_roles: Iterable[ModuleRole | str] | ModuleRole | str | None = None,
    exclude_roles: Iterable[ModuleRole | str] | ModuleRole | str | None = None,
    name_filter: NameFilter | None = None,
) -> list[LinearModuleInfo]:
    nn = _torch_nn()
    include = normalize_target_roles(target_roles)
    exclude = normalize_target_roles(exclude_roles)
    results: list[LinearModuleInfo] = []
    for name, module in model.named_modules():
        if not name or not isinstance(module, nn.Linear):
            continue
        if name_filter is not None and not name_filter(name):
            continue
        canonical_name = canonicalize_module_name(name)
        role = classify_linear_module_name(canonical_name)
        if not _role_is_selected(role, include, exclude):
            continue
        results.append(
            LinearModuleInfo(
                name=name,
                module=module,
                role=role,
                in_features=int(module.in_features),
                out_features=int(module.out_features),
                canonical_name=canonical_name,
            )
        )
    return results
