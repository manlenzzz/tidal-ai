"""Runtime shim for a vLLM/vLLM-Ascend attention selector API gap.

This module is intentionally opt-in. It patches only the in-process
``vllm.v1.attention.selector`` module after vLLM-Ascend has installed its own
patches, keeping the vendor installation untouched.
"""
from __future__ import annotations

from typing import Any, NamedTuple, cast, get_args


def install() -> dict[str, Any]:
    import torch
    import vllm.v1.attention.selector as selector
    from vllm.config.cache import CacheDType
    from vllm.v1.attention.backend import AttentionBackend, AttentionType

    cached_get_attn_backend = selector._cached_get_attn_backend

    class AttentionSelectorConfig(NamedTuple):
        head_size: int
        dtype: torch.dtype
        kv_cache_dtype: CacheDType | None
        block_size: int | None
        use_mla: bool = False
        has_sink: bool = False
        use_compress: bool = False
        use_sparse: bool = False
        use_mm_prefix: bool = False
        use_per_head_quant_scales: bool = False
        attn_type: str = AttentionType.DECODER

    def get_attn_backend(
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: str | None,
        block_size: int | None = None,
        use_mla: bool = False,
        has_sink: bool = False,
        use_compress: bool = False,
        use_sparse: bool = False,
        use_mm_prefix: bool = False,
        use_per_head_quant_scales: bool = False,
        attn_type: str | None = None,
        num_heads: int | None = None,
    ) -> type[AttentionBackend]:
        if kv_cache_dtype is not None:
            valid_cache_dtypes = get_args(CacheDType)
            assert kv_cache_dtype in valid_cache_dtypes, (
                f"Invalid kv_cache_dtype: {kv_cache_dtype}. "
                f"Valid values are: {valid_cache_dtypes}"
            )

        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        resolved_block_size = block_size
        if resolved_block_size is None:
            cache_config = getattr(vllm_config, "cache_config", None)
            if cache_config is not None and getattr(cache_config, "user_specified_block_size", False):
                resolved_block_size = cache_config.block_size

        attn_selector_config = AttentionSelectorConfig(
            head_size=head_size,
            dtype=dtype,
            kv_cache_dtype=cast(CacheDType | None, kv_cache_dtype),
            block_size=resolved_block_size,
            use_mla=use_mla,
            has_sink=has_sink,
            use_compress=use_compress,
            use_sparse=use_sparse,
            use_mm_prefix=use_mm_prefix,
            use_per_head_quant_scales=use_per_head_quant_scales,
            attn_type=attn_type or AttentionType.DECODER,
        )

        return cached_get_attn_backend(
            backend=vllm_config.attention_config.backend,
            attn_selector_config=attn_selector_config,
            num_heads=num_heads,
        )

    selector.AttentionSelectorConfig = AttentionSelectorConfig
    selector.get_attn_backend = get_attn_backend
    return {
        "status": "INSTALLED",
        "config_fields": list(AttentionSelectorConfig._fields),
        "get_attn_backend": "tidal.integrations.vllm_ascend_selector_shim.get_attn_backend",
    }
