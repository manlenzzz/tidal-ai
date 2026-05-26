from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Segment:
    start: int
    end: int
    adapter: int


def sgmv(inputs: np.ndarray, weights: np.ndarray, segments: list[Segment]) -> np.ndarray:
    """CPU reference for segmented gather matrix-vector multiplication."""

    x = np.asarray(inputs, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 3:
        raise ValueError("weights must have shape [num_adapters, in_features, out_features]")
    if x.ndim != 2 or x.shape[1] != w.shape[1]:
        raise ValueError("inputs must have shape [batch, in_features]")

    out = np.zeros((x.shape[0], w.shape[2]), dtype=np.float64)
    for segment in segments:
        if not (0 <= segment.start <= segment.end <= x.shape[0]):
            raise ValueError("segment boundaries are out of range")
        if not (0 <= segment.adapter < w.shape[0]):
            raise ValueError("segment adapter is out of range")
        out[segment.start:segment.end] += x[segment.start:segment.end] @ w[segment.adapter]
    return out


def lora_sgmv(inputs: np.ndarray, a_weights: np.ndarray, b_weights: np.ndarray, segments: list[Segment]) -> np.ndarray:
    """CPU reference for y += x A_i B_i across adapter segments."""

    hidden = sgmv(inputs, a_weights, segments)
    return sgmv(hidden, b_weights, segments)
