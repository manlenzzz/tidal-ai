import numpy as np

from tidal.methods.global_rank_sparsity.core import (
    build_cap_candidate_pool,
    robust_pca,
)


def _low_rank_plus_sparse(seed: int = 0, rows: int = 96, cols: int = 96):
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((rows, 8))
    v = rng.standard_normal((8, cols))
    low_rank = (u @ v) / np.sqrt(8) * 0.02
    sparse = np.zeros((rows, cols))
    k = int(rows * cols * 0.02)
    idx = rng.choice(rows * cols, k, replace=False)
    sparse.ravel()[idx] = rng.standard_normal(k) * 0.1
    return low_rank + sparse


def test_torch_backend_returns_numpy_result():
    matrix = _low_rank_plus_sparse()
    result = robust_pca(matrix, backend="torch", device="cpu", max_iter=80)
    assert isinstance(result.low_rank, np.ndarray)
    assert isinstance(result.sparse, np.ndarray)
    assert np.isfinite(result.residual_norm)
    assert result.low_rank.shape == matrix.shape


def test_torch_and_numpy_backends_agree():
    matrix = _low_rank_plus_sparse()
    cpu = robust_pca(matrix, backend="numpy", max_iter=120)
    gpu = robust_pca(matrix, backend="torch", device="cpu", max_iter=120)

    def rel(a, b):
        return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12))

    assert rel(cpu.low_rank, gpu.low_rank) < 1e-2
    assert rel(cpu.sparse, gpu.sparse) < 1e-2


def test_candidate_pool_torch_backend_matches_counts():
    matrix = _low_rank_plus_sparse()
    pool_np = build_cap_candidate_pool({"w": matrix}, backend="numpy", max_iter=120)
    pool_pt = build_cap_candidate_pool({"w": matrix}, backend="torch", device="cpu", max_iter=120)
    # Sparse candidate counts should be close (converged S sparsity is backend-stable).
    n_np = len(pool_np.candidates)
    n_pt = len(pool_pt.candidates)
    assert abs(n_np - n_pt) <= max(5, int(0.1 * n_np))
