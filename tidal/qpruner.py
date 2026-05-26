from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class QuantizedTensor:
    codes: np.ndarray
    scale: float
    bits: int

    def dequantize(self) -> np.ndarray:
        return self.codes.astype(np.float64) * self.scale


def _as_1d(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 1:
        return array
    return array.reshape(array.shape[0], -1).mean(axis=1)


def discrete_mutual_information(x: np.ndarray, y: np.ndarray, *, bins: int = 16) -> float:
    """Estimate I(X;Y) after discretizing continuous activations."""

    if bins <= 1:
        raise ValueError("bins must be greater than 1")
    x_flat = _as_1d(np.asarray(x, dtype=np.float64))
    y_flat = _as_1d(np.asarray(y, dtype=np.float64))
    if x_flat.shape[0] != y_flat.shape[0]:
        raise ValueError("x and y must have the same number of samples")

    hist, _, _ = np.histogram2d(x_flat, y_flat, bins=bins)
    total = hist.sum()
    if total == 0:
        return 0.0
    pxy = hist / total
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    nonzero = pxy > 0
    return float(np.sum(pxy[nonzero] * np.log(pxy[nonzero] / (px @ py)[nonzero])))


def layer_mutual_information(
    layer_outputs: Mapping[str, np.ndarray],
    predictions: np.ndarray,
    *,
    bins: int = 16,
) -> dict[str, float]:
    return {
        name: discrete_mutual_information(output, predictions, bins=bins)
        for name, output in layer_outputs.items()
    }


def allocate_bitwidths(
    importances: Mapping[str, float],
    layer_sizes: Mapping[str, int],
    *,
    candidate_bits: Sequence[int] = (2, 4, 8),
    max_memory_bits: int | None = None,
    max_average_bits: float | None = None,
) -> dict[str, int]:
    """Allocate mixed precision under a memory budget."""

    if set(importances) != set(layer_sizes):
        raise ValueError("importances and layer_sizes must contain the same layers")
    bits = sorted(set(int(bit) for bit in candidate_bits))
    if not bits or bits[0] <= 0:
        raise ValueError("candidate_bits must contain positive integers")
    if max_memory_bits is None:
        if max_average_bits is None:
            raise ValueError("provide max_memory_bits or max_average_bits")
        max_memory_bits = int(sum(layer_sizes.values()) * max_average_bits)

    config = {name: bits[0] for name in importances}
    used = sum(layer_sizes[name] * config[name] for name in config)
    if used > max_memory_bits:
        raise ValueError("lowest precision exceeds memory budget")

    while True:
        best: tuple[float, float, str, int, int] | None = None
        for name, current in config.items():
            index = bits.index(current)
            if index == len(bits) - 1:
                continue
            next_bit = bits[index + 1]
            added = (next_bit - current) * layer_sizes[name]
            if used + added > max_memory_bits:
                continue
            gain = max(float(importances[name]), 0.0) * np.log2(next_bit / current)
            candidate = (gain / added, gain, name, next_bit, added)
            if best is None or candidate > best:
                best = candidate
        if best is None or best[1] <= 0:
            return config
        _, _, name, next_bit, added = best
        config[name] = next_bit
        used += added


def quantize_symmetric(values: np.ndarray, *, bits: int) -> QuantizedTensor:
    """Symmetric per-tensor quantization with dequantization support."""

    if bits < 2:
        raise ValueError("bits must be >= 2")
    array = np.asarray(values, dtype=np.float64)
    qmax = (1 << (bits - 1)) - 1
    max_abs = float(np.max(np.abs(array))) if array.size else 0.0
    if max_abs == 0.0:
        return QuantizedTensor(np.zeros_like(array, dtype=np.int64), 1.0, bits)
    scale = max_abs / qmax
    codes = np.clip(np.rint(array / scale), -qmax, qmax).astype(np.int64)
    return QuantizedTensor(codes=codes, scale=scale, bits=bits)
