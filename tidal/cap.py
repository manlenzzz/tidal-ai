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
    rows, cols = rpca.low_rank.shape
    u, sigma, vt = np.linalg.svd(rpca.low_rank, full_matrices=False)

    candidates: list[tuple[float, str, int, int]] = []
    rank_cost = rows + cols
    for index, value in enumerate(sigma):
        if value > 0:
            candidates.append((float(value), "rank", index, rank_cost))
    for flat_index, value in enumerate(np.abs(rpca.sparse).ravel()):
        if value > 0:
            candidates.append((float(value), "sparse", flat_index, 1))

    candidates.sort(key=lambda item: item[0] / item[3], reverse=True)
    rank_mask = np.zeros_like(sigma, dtype=bool)
    sparse_mask_flat = np.zeros(rpca.sparse.size, dtype=bool)
    used = 0
    for _, kind, index, cost in candidates:
        if used + cost > budget:
            continue
        if kind == "rank":
            rank_mask[index] = True
        else:
            sparse_mask_flat[index] = True
        used += cost

    kept_sigma = sigma * rank_mask
    low_rank = (u * kept_sigma) @ vt
    sparse_mask = sparse_mask_flat.reshape(rpca.sparse.shape)
    sparse = rpca.sparse * sparse_mask
    return CAPCompression(
        low_rank=low_rank,
        sparse=sparse,
        rank_mask=rank_mask,
        sparse_mask=sparse_mask,
        parameter_count=used,
    )
