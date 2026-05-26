from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import erf, sqrt
from typing import Callable, Mapping, Sequence

import numpy as np

BitwidthConfig = dict[str, int]
Objective = Callable[[BitwidthConfig], float]


@dataclass(frozen=True)
class QuantizedTensor:
    codes: np.ndarray
    scale: float
    bits: int

    def dequantize(self) -> np.ndarray:
        return self.codes.astype(np.float64) * self.scale


@dataclass(frozen=True)
class BitwidthEvaluation:
    config: BitwidthConfig
    score: float
    memory_bits: int


@dataclass(frozen=True)
class BitwidthSearchResult:
    config: BitwidthConfig
    score: float
    memory_bits: int
    evaluations: tuple[BitwidthEvaluation, ...]


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


def _normalize_bits(candidate_bits: Sequence[int]) -> list[int]:
    bits = sorted(set(int(bit) for bit in candidate_bits))
    if not bits or bits[0] <= 0:
        raise ValueError("candidate_bits must contain positive integers")
    return bits


def _memory_budget(
    layer_sizes: Mapping[str, int],
    *,
    max_memory_bits: int | None,
    max_average_bits: float | None,
) -> int:
    if any(size <= 0 for size in layer_sizes.values()):
        raise ValueError("layer_sizes must be positive")
    if max_memory_bits is None:
        if max_average_bits is None:
            raise ValueError("provide max_memory_bits or max_average_bits")
        if max_average_bits <= 0:
            raise ValueError("max_average_bits must be positive")
        return int(sum(layer_sizes.values()) * max_average_bits)
    if max_memory_bits <= 0:
        raise ValueError("max_memory_bits must be positive")
    return max_memory_bits


def config_memory_bits(config: Mapping[str, int], layer_sizes: Mapping[str, int]) -> int:
    if set(config) != set(layer_sizes):
        raise ValueError("config and layer_sizes must contain the same layers")
    return int(sum(layer_sizes[name] * int(config[name]) for name in config))


def enumerate_bitwidth_configs(
    layer_names: Sequence[str],
    layer_sizes: Mapping[str, int],
    *,
    candidate_bits: Sequence[int] = (2, 4, 8),
    max_memory_bits: int | None = None,
    max_average_bits: float | None = None,
) -> list[BitwidthConfig]:
    """Enumerate feasible mixed-precision decisions under the global memory budget."""

    names = list(layer_names)
    if len(names) != len(set(names)):
        raise ValueError("layer_names must be unique")
    if set(names) != set(layer_sizes):
        raise ValueError("layer_names and layer_sizes must contain the same layers")
    bits = _normalize_bits(candidate_bits)
    budget = _memory_budget(layer_sizes, max_memory_bits=max_memory_bits, max_average_bits=max_average_bits)

    configs: list[BitwidthConfig] = []
    for choices in product(bits, repeat=len(names)):
        config = {name: int(bit) for name, bit in zip(names, choices)}
        if config_memory_bits(config, layer_sizes) <= budget:
            configs.append(config)
    configs.sort(key=lambda cfg: (config_memory_bits(cfg, layer_sizes), tuple(cfg[name] for name in names)))
    return configs


def allocate_bitwidths(
    importances: Mapping[str, float],
    layer_sizes: Mapping[str, int],
    *,
    candidate_bits: Sequence[int] = (2, 4, 8),
    max_memory_bits: int | None = None,
    max_average_bits: float | None = None,
) -> BitwidthConfig:
    """Allocate mixed precision from MI/importance scores under a memory budget."""

    if set(importances) != set(layer_sizes):
        raise ValueError("importances and layer_sizes must contain the same layers")
    bits = _normalize_bits(candidate_bits)
    max_memory_bits = _memory_budget(
        layer_sizes,
        max_memory_bits=max_memory_bits,
        max_average_bits=max_average_bits,
    )

    config = {name: bits[0] for name in importances}
    used = config_memory_bits(config, layer_sizes)
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


def _config_features(config: Mapping[str, int], names: Sequence[str], bits: Sequence[int]) -> np.ndarray:
    lo = np.log2(min(bits))
    span = np.log2(max(bits)) - lo or 1.0
    return np.array([(np.log2(config[name]) - lo) / span for name in names], dtype=np.float64)


def _rbf_kernel(x: np.ndarray, y: np.ndarray, *, length_scale: float = 0.75) -> np.ndarray:
    diff = x[:, None, :] - y[None, :, :]
    return np.exp(-0.5 * np.sum(diff * diff, axis=2) / (length_scale * length_scale))


def _normal_pdf(value: float) -> float:
    return float(np.exp(-0.5 * value * value) / np.sqrt(2.0 * np.pi))


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def _expected_improvement(mu: float, sigma: float, best: float, exploration: float) -> float:
    if sigma <= 1e-12:
        return max(mu - best - exploration, 0.0)
    z = (mu - best - exploration) / sigma
    return (mu - best - exploration) * _normal_cdf(z) + sigma * _normal_pdf(z)


def _gp_scores(
    candidate_features: np.ndarray,
    train_features: np.ndarray,
    train_scores: np.ndarray,
    *,
    noise: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    kernel = _rbf_kernel(train_features, train_features) + noise * np.eye(train_features.shape[0])
    cross = _rbf_kernel(candidate_features, train_features)
    alpha = np.linalg.solve(kernel, train_scores)
    mu = cross @ alpha
    solved = np.linalg.solve(kernel, cross.T)
    var = np.maximum(1.0 - np.sum(cross * solved.T, axis=1), 1e-12)
    return mu, np.sqrt(var)


def bayesian_refine_bitwidths(
    importances: Mapping[str, float],
    layer_sizes: Mapping[str, int],
    *,
    objective: Objective,
    candidate_bits: Sequence[int] = (2, 4, 8),
    max_memory_bits: int | None = None,
    max_average_bits: float | None = None,
    max_trials: int = 16,
    init_config: BitwidthConfig | None = None,
    exploration: float = 0.01,
    seed: int | None = None,
) -> BitwidthSearchResult:
    """QPruner-style MI initialization followed by GP expected-improvement search."""

    if set(importances) != set(layer_sizes):
        raise ValueError("importances and layer_sizes must contain the same layers")
    if max_trials <= 0:
        raise ValueError("max_trials must be positive")
    bits = _normalize_bits(candidate_bits)
    names = list(importances)
    candidates = enumerate_bitwidth_configs(
        names,
        layer_sizes,
        candidate_bits=bits,
        max_memory_bits=max_memory_bits,
        max_average_bits=max_average_bits,
    )
    if not candidates:
        raise ValueError("no feasible bitwidth configs")

    rng = np.random.default_rng(seed)
    unevaluated = [dict(candidate) for candidate in candidates]
    evaluations: list[BitwidthEvaluation] = []

    def pop_config(config: BitwidthConfig) -> BitwidthConfig:
        for index, candidate in enumerate(unevaluated):
            if candidate == config:
                return unevaluated.pop(index)
        return unevaluated.pop(0)

    start = init_config or allocate_bitwidths(
        importances,
        layer_sizes,
        candidate_bits=bits,
        max_memory_bits=max_memory_bits,
        max_average_bits=max_average_bits,
    )
    if config_memory_bits(start, layer_sizes) > _memory_budget(
        layer_sizes,
        max_memory_bits=max_memory_bits,
        max_average_bits=max_average_bits,
    ):
        raise ValueError("init_config exceeds memory budget")

    while unevaluated and len(evaluations) < min(max_trials, len(candidates)):
        if not evaluations:
            config = pop_config(start)
        elif len(evaluations) == 1:
            config = unevaluated.pop(int(rng.integers(0, len(unevaluated))))
        else:
            train_x = np.vstack([_config_features(item.config, names, bits) for item in evaluations])
            train_y = np.asarray([item.score for item in evaluations], dtype=np.float64)
            remaining_x = np.vstack([_config_features(item, names, bits) for item in unevaluated])
            mu, sigma = _gp_scores(remaining_x, train_x, train_y)
            best_score = float(np.max(train_y))
            acquisition = np.array([
                _expected_improvement(float(m), float(s), best_score, exploration) for m, s in zip(mu, sigma)
            ])
            best_value = float(np.max(acquisition))
            ties = np.flatnonzero(np.isclose(acquisition, best_value))
            chosen = int(rng.choice(ties))
            config = unevaluated.pop(chosen)

        score = float(objective(dict(config)))
        evaluations.append(BitwidthEvaluation(dict(config), score, config_memory_bits(config, layer_sizes)))

    best = max(evaluations, key=lambda item: item.score)
    return BitwidthSearchResult(dict(best.config), best.score, best.memory_bits, tuple(evaluations))


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
