"""Opt-in shim for vLLM list attention metadata on vLLM-Ascend.

vLLM 0.18 can store ``ForwardContext.attn_metadata`` as
``list[dict[layer_name, metadata]]`` when multiple KV cache groups are present.
The vLLM-Ascend attention backend in this demo environment still expects the
per-layer metadata object. This patch keeps the vendor install untouched and
normalizes the metadata at vLLM's attention context boundary.
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import sys
from typing import Any

TARGET_MODULE = "vllm.model_executor.layers.attention.attention"
ASCEND_BACKEND_MODULE = "vllm_ascend.attention.attention_v1"
_AUTO_FINDER_MARKER = "_tidal_vllm_ascend_metadata_auto_finder"


def _metadata_for_layer(attn_metadata: Any, layer_name: str) -> Any:
    if isinstance(attn_metadata, dict):
        return attn_metadata[layer_name]
    if isinstance(attn_metadata, list):
        for group in attn_metadata:
            if isinstance(group, dict) and layer_name in group:
                return group[layer_name]
        if len(attn_metadata) == 1:
            if isinstance(attn_metadata[0], dict):
                return next(iter(attn_metadata[0].values()))
            return attn_metadata[0]
    return attn_metadata


def _layer_name(layer: Any) -> str | None:
    return getattr(layer, "layer_name", None) or getattr(layer, "name", None)


def _patch_ascend_backend_forward(*, import_if_missing: bool = True) -> str:
    if ASCEND_BACKEND_MODULE in sys.modules:
        attention_v1 = sys.modules[ASCEND_BACKEND_MODULE]
    elif import_if_missing:
        try:
            attention_v1 = importlib.import_module(ASCEND_BACKEND_MODULE)
        except Exception:
            return "UNAVAILABLE"
    else:
        return "DEFERRED"
    patched = False
    for class_name in ("AscendAttentionBackendImpl", "AscendC8AttentionBackendImpl"):
        backend_cls = getattr(attention_v1, class_name, None)
        if backend_cls is None:
            continue
        original_forward = getattr(backend_cls, "forward", None)
        if original_forward is None:
            continue
        if getattr(original_forward, "_tidal_list_metadata_shim", False):
            patched = True
            continue

        def forward(self, layer, query, key, value, kv_cache, attn_metadata, *args, __original=original_forward, **kwargs):
            layer_name = _layer_name(layer)
            if layer_name:
                attn_metadata = _metadata_for_layer(attn_metadata, layer_name)
            return __original(self, layer, query, key, value, kv_cache, attn_metadata, *args, **kwargs)

        forward._tidal_list_metadata_shim = True  # type: ignore[attr-defined]
        backend_cls.forward = forward
        patched = True
    return "INSTALLED" if patched else "MISSING_BACKEND_CLASS"


def _patch_attention_context() -> dict[str, Any]:
    attention = importlib.import_module(TARGET_MODULE)

    if getattr(attention.get_attention_context, "_tidal_list_metadata_shim", False):
        return {
            "status": "ALREADY_INSTALLED",
            "target": f"{TARGET_MODULE}.get_attention_context",
            "supports_list_metadata": True,
        }

    def get_attention_context(layer_name: str):
        forward_context = attention.get_forward_context()
        attn_metadata = _metadata_for_layer(forward_context.attn_metadata, layer_name)
        attn_layer = forward_context.no_compile_layers[layer_name]
        kv_cache = attn_layer.kv_cache[forward_context.virtual_engine]
        slot_mapping = forward_context.slot_mapping
        assert isinstance(slot_mapping, dict), (
            f"Expected slot_mapping to be a dict, got {type(slot_mapping)}. "
        )
        layer_slot_mapping = slot_mapping.get(layer_name)
        return attn_metadata, attn_layer, kv_cache, layer_slot_mapping

    get_attention_context._tidal_list_metadata_shim = True  # type: ignore[attr-defined]
    attention.get_attention_context = get_attention_context
    return {
        "status": "INSTALLED",
        "target": f"{TARGET_MODULE}.get_attention_context",
        "supports_list_metadata": True,
    }


def install() -> dict[str, Any]:
    result = _patch_attention_context()
    result["backend_forward_status"] = _patch_ascend_backend_forward(import_if_missing=True)
    return result


class _PatchAfterImportLoader(importlib.abc.Loader):
    def __init__(self, wrapped: importlib.abc.Loader, patcher) -> None:
        self.wrapped = wrapped
        self.patcher = patcher

    def create_module(self, spec):
        if hasattr(self.wrapped, "create_module"):
            return self.wrapped.create_module(spec)  # type: ignore[misc]
        return None

    def exec_module(self, module) -> None:
        self.wrapped.exec_module(module)  # type: ignore[attr-defined]
        self.patcher()


class _AttentionMetadataShimFinder(importlib.abc.MetaPathFinder):
    _tidal_vllm_ascend_metadata_auto_finder = True

    def find_spec(self, fullname: str, path: Any = None, target: Any = None):
        patcher = {
            TARGET_MODULE: _patch_attention_context,
            ASCEND_BACKEND_MODULE: lambda: _patch_ascend_backend_forward(import_if_missing=False),
        }.get(fullname)
        if patcher is None:
            return None
        for finder in sys.meta_path:
            if finder is self or getattr(finder, _AUTO_FINDER_MARKER, False):
                continue
            find_spec = getattr(finder, "find_spec", None)
            if find_spec is None:
                continue
            spec = find_spec(fullname, path, target)
            if spec is None:
                continue
            if spec.loader is not None:
                spec.loader = _PatchAfterImportLoader(spec.loader, patcher)
            return spec
        return None


def _ensure_import_hook() -> bool:
    if any(getattr(finder, _AUTO_FINDER_MARKER, False) for finder in sys.meta_path):
        return False
    sys.meta_path.insert(0, _AttentionMetadataShimFinder())
    return True


def install_auto() -> dict[str, Any]:
    target_loaded = TARGET_MODULE in sys.modules
    backend_loaded = ASCEND_BACKEND_MODULE in sys.modules
    if target_loaded:
        result = _patch_attention_context()
        result["auto_install"] = "MODULE_ALREADY_LOADED"
    else:
        result = {
            "status": "DEFERRED",
            "target": f"{TARGET_MODULE}.get_attention_context",
            "supports_list_metadata": True,
            "auto_install": "IMPORT_HOOK",
        }

    backend_forward_status = _patch_ascend_backend_forward(import_if_missing=False) if backend_loaded else "DEFERRED"
    result["backend_forward_status"] = backend_forward_status
    hook_added = _ensure_import_hook()
    if not hook_added and result["status"] == "DEFERRED":
        return {
            "status": "ALREADY_DEFERRED",
            "target": f"{TARGET_MODULE}.get_attention_context",
            "supports_list_metadata": True,
            "auto_install": "IMPORT_HOOK",
            "backend_forward_status": backend_forward_status,
        }
    return result
