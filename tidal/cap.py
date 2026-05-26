from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RPCAResult:
    low_rank: np.ndarray
    sparse: np.ndarray
    iterations: int
    residual_norm: float


@dataclass(frozen=True)
class CAPCompression:
    low_rank: np.ndarray
    sparse: np.ndarray
    rank_mask: np.ndarray
    sparse_mask: np.ndarray
    parameter_count: int

    @property
    def reconstructed(self) -> np.ndarray:
        return self.low_rank + self.sparse


@dataclass(frozen=True)
class CAPPolicyStep:
    step: int
    reward: float
    reconstruction_error: float
    parameter_count: int


@dataclass(frozen=True)
class CAPSearchResult:
    compression: CAPCompression
    best_reward: float
    history: tuple[CAPPolicyStep, ...]


def _shrink(values: np.ndarray, tau: float) -> np.ndarray:
    return np.sign(values) * np.maximum(np.abs(values) - tau, 0.0)


def robust_pca(
    matrix: np.ndarray,
    *,
    lam: float | None = None,
    mu: float | None = None,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> RPCAResult:
    """ADMM robust PCA: minimize ||L||_* + lam ||S||_1 subject to W=L+S."""

    w = np.asarray(matrix, dtype=np.float64)
    if w.ndim != 2:
        raise ValueError("matrix must be 2D")
    rows, cols = w.shape
    lam = lam if lam is not None else 1.0 / np.sqrt(max(rows, cols))
    mu = mu if mu is not None else rows * cols / (4.0 * np.sum(np.abs(w)) + 1e-12)
    if lam <= 0 or mu <= 0:
        raise ValueError("lam and mu must be positive")

    low_rank = np.zeros_like(w)
    sparse = np.zeros_like(w)
    dual = np.zeros_like(w)
    norm_w = np.linalg.norm(w, ord="fro") + 1e-12

    residual_norm = float("inf")
    for iteration in range(1, max_iter + 1):
        u, sigma, vt = np.linalg.svd(w - sparse + dual / mu, full_matrices=False)
        sigma_shrunk = np.maximum(sigma - 1.0 / mu, 0.0)
        low_rank = (u * sigma_shrunk) @ vt
        sparse = _shrink(w - low_rank + dual / mu, lam / mu)
        residual = w - low_rank - sparse
        dual = dual + mu * residual
        residual_norm = float(np.linalg.norm(residual, ord="fro") / norm_w)
        if residual_norm <= tol:
            break
    return RPCAResult(low_rank=low_rank, sparse=sparse, iterations=iteration, residual_norm=residual_norm)


def _candidate_table(rpca: RPCAResult) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = rpca.low_rank.shape
    u, sigma, vt = np.linalg.svd(rpca.low_rank, full_matrices=False)
    sparse_abs = np.abs(rpca.sparse).ravel()

    kinds: list[int] = []
    indices: list[int] = []
    values: list[float] = []
    costs: list[int] = []
    rank_cost = rows + cols
    for index, value in enumerate(sigma):
        if value > 0:
            kinds.append(0)
            indices.append(index)
            values.append(float(value))
            costs.append(rank_cost)
    for flat_index, value in enumerate(sparse_abs):
        if value > 0:
            kinds.append(1)
            indices.append(flat_index)
            values.append(float(value))
            costs.append(1)

    return (
        np.asarray(kinds, dtype=np.int64),
        np.asarray(indices, dtype=np.int64),
        np.asarray(values, dtype=np.float64),
        np.asarray(costs, dtype=np.int64),
        sigma,
    )


def _compression_from_selection(rpca: RPCAResult, selection: np.ndarray) -> CAPCompression:
    rows, cols = rpca.low_rank.shape
    u, sigma, vt = np.linalg.svd(rpca.low_rank, full_matrices=False)
    kinds, indices, _, costs, _ = _candidate_table(rpca)
    rank_mask = np.zeros_like(sigma, dtype=bool)
    sparse_mask_flat = np.zeros(rpca.sparse.size, dtype=bool)

    for selected, kind, index in zip(selection.astype(bool), kinds, indices):
        if not selected:
            continue
        if kind == 0:
            rank_mask[index] = True
        else:
            sparse_mask_flat[index] = True

    low_rank = (u * (sigma * rank_mask)) @ vt
    sparse_mask = sparse_mask_flat.reshape(rpca.sparse.shape)
    sparse = rpca.sparse * sparse_mask
    return CAPCompression(
        low_rank=low_rank,
        sparse=sparse,
        rank_mask=rank_mask,
        sparse_mask=sparse_mask,
        parameter_count=int(np.sum(costs * selection)),
    )


def _relative_error(matrix: np.ndarray, compression: CAPCompression) -> float:
    denom = np.linalg.norm(matrix, ord="fro") + 1e-12
    return float(np.linalg.norm(matrix - compression.reconstructed, ord="fro") / denom)


def compress_global_rank_sparsity(
    matrix: np.ndarray,
    *,
    budget: int,
    lam: float | None = None,
    max_iter: int = 200,
) -> CAPCompression:
    """Compress one matrix using RPCA candidates and global score selection."""

    if budget <= 0:
        raise ValueError("budget must be positive")
    rpca = robust_pca(matrix, lam=lam, max_iter=max_iter)
    kinds, indices, values, costs, _ = _candidate_table(rpca)
    if len(values) == 0:
        zeros = np.zeros(0, dtype=bool)
        return _compression_from_selection(rpca, zeros)

    order = np.argsort(-(values / costs))
    selection = np.zeros(len(values), dtype=bool)
    used = 0
    for index in order:
        cost = int(costs[index])
        if used + cost > budget:
            continue
        selection[index] = True
        used += cost

    return _compression_from_selection(rpca, selection)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def optimize_global_rank_sparsity(
    matrix: np.ndarray,
    *,
    budget: int,
    lam: float | None = None,
    max_iter: int = 200,
    policy_steps: int = 120,
    samples_per_step: int = 8,
    learning_rate: float = 0.1,
    budget_penalty: float = 2.0,
    seed: int | None = None,
) -> CAPSearchResult:
    """CAP-style global rank/sparsity search with Bernoulli policy optimization."""

    if budget <= 0:
        raise ValueError("budget must be positive")
    if policy_steps <= 0:
        raise ValueError("policy_steps must be positive")
    if samples_per_step <= 0:
        raise ValueError("samples_per_step must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    matrix_array = np.asarray(matrix, dtype=np.float64)
    rpca = robust_pca(matrix_array, lam=lam, max_iter=max_iter)
    _, _, values, costs, _ = _candidate_table(rpca)
    if len(values) == 0:
        empty = np.zeros(0, dtype=bool)
        compression = _compression_from_selection(rpca, empty)
        step = CAPPolicyStep(1, -_relative_error(matrix_array, compression), _relative_error(matrix_array, compression), 0)
        return CAPSearchResult(compression, step.reward, tuple([step] * policy_steps))

    rng = np.random.default_rng(seed)
    greedy = compress_global_rank_sparsity(matrix_array, budget=budget, lam=lam, max_iter=max_iter)
    greedy_selection = np.zeros(len(values), dtype=bool)
    kinds, indices, _, _, _ = _candidate_table(rpca)
    for item, (kind, index) in enumerate(zip(kinds, indices)):
        if kind == 0:
            greedy_selection[item] = bool(greedy.rank_mask[index])
        else:
            greedy_selection[item] = bool(greedy.sparse_mask.ravel()[index])

    logits = np.log((values / (values.max() + 1e-12)) + 1e-3)
    logits += greedy_selection.astype(np.float64)
    best_selection = greedy_selection.copy()
    best_compression = _compression_from_selection(rpca, best_selection)
    best_error = _relative_error(matrix_array, best_compression)
    best_reward = -best_error
    baseline = best_reward
    history: list[CAPPolicyStep] = []

    for step in range(1, policy_steps + 1):
        probs = _sigmoid(logits)
        grad = np.zeros_like(logits)
        step_best_reward = -float("inf")
        step_best_error = float("inf")
        step_best_cost = 0

        for _ in range(samples_per_step):
            sample = rng.random(len(probs)) < probs
            cost = int(np.sum(costs * sample))
            compression = _compression_from_selection(rpca, sample)
            error = _relative_error(matrix_array, compression)
            overflow = max(0, cost - budget) / max(float(budget), 1.0)
            reward = -error - budget_penalty * overflow
            grad += (reward - baseline) * (sample.astype(np.float64) - probs)

            if cost <= budget and reward > best_reward:
                best_reward = reward
                best_selection = sample.copy()
                best_compression = compression
            if reward > step_best_reward:
                step_best_reward = reward
                step_best_error = error
                step_best_cost = cost

        baseline = 0.9 * baseline + 0.1 * step_best_reward
        logits += learning_rate * grad / float(samples_per_step)
        history.append(CAPPolicyStep(step, float(step_best_reward), float(step_best_error), int(step_best_cost)))

    final_compression = _compression_from_selection(rpca, best_selection)
    final_reward = -_relative_error(matrix_array, final_compression)
    return CAPSearchResult(final_compression, max(best_reward, final_reward), tuple(history))
