"""Shared tokenizer setup for batched decoder-only generation benchmarks."""
from __future__ import annotations

from typing import Any


def ensure_left_padding_for_generate(tokenizer: Any) -> str | None:
    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"
        return getattr(tokenizer, "padding_side", None)
    return None
