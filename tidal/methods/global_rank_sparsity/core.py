from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Mapping

import numpy as np


@dataclass(frozen=True)
class RPCAResult:
    low_rank: np.ndarray
    sparse: np.ndarray
    iterations: int
    residual_norm: float


class CAPCandidateKind(str, Enum):
    """Candidate families used by CAP Stage 2 budget allocation."""

    RANK = "rank"
    SPARSE = "sparse"


@dataclass(frozen=True)
class CAPCandidate:
    id: str
    layer: str
    kind: CAPCandidateKind
    local_index: int
    value: float
    cost: int
    shape: tuple[int, int]


@dataclass(frozen=True)
class CAPCandidatePool:
    decompositions: Mapping[str, RPCAResult]
    candidates: tuple[CAPCandidate, ...]

    @property
    def layers(self) -> tuple[str, ...]:
        return tuple(self.decompositions.keys())

    @property
    def total_candidates(self) -> int:
        return len(self.candidates)

    def reconstruct(self, layer: str) -> np.ndarray:
        decomposition = self.decompositions[layer]
        return decomposition.low_rank + decomposition.sparse


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


def _candidate_kind(kind_code: int) -> CAPCandidateKind:
    return CAPCandidateKind.RANK if int(kind_code) == 0 else CAPCandidateKind.SPARSE


def _candidate_id(layer: str, kind: CAPCandidateKind, local_index: int) -> str:
    return f"{layer}:{kind.value}:{local_index}"


def _candidates_for_rpca(layer: str, rpca: RPCAResult) -> tuple[CAPCandidate, ...]:
    kinds, indices, values, costs, _ = _candidate_table(rpca)
    rows, cols = rpca.low_rank.shape
    candidates: list[CAPCandidate] = []
    for kind_code, local_index, value, cost in zip(kinds, indices, values, costs):
        kind = _candidate_kind(int(kind_code))
        index = int(local_index)
        candidates.append(
            CAPCandidate(
                id=_candidate_id(layer, kind, index),
                layer=layer,
                kind=kind,
                local_index=index,
                value=float(value),
                cost=int(cost),
                shape=(int(rows), int(cols)),
            )
        )
    return tuple(candidates)


def build_cap_candidate_pool(
    matrices: Mapping[str, np.ndarray],
    *,
    lam: float | None = None,
    max_iter: int = 200,
) -> CAPCandidatePool:
    """Run CAP Stage 1 and expose rank/sparse candidates for Stage 2."""

    if not matrices:
        raise ValueError("matrices must not be empty")
    decompositions = {
        name: robust_pca(np.asarray(matrix, dtype=np.float64), lam=lam, max_iter=max_iter)
        for name, matrix in matrices.items()
    }
    candidates: list[CAPCandidate] = []
    for name, rpca in decompositions.items():
        candidates.extend(_candidates_for_rpca(name, rpca))
    return CAPCandidatePool(decompositions=decompositions, candidates=tuple(candidates))


def _candidate_sort_key(candidate: CAPCandidate, score: float) -> tuple[float, str, int, int]:
    kind_rank = 0 if candidate.kind is CAPCandidateKind.RANK else 1
    return (-float(score), candidate.layer, kind_rank, int(candidate.local_index))


def select_cap_candidates(
    candidates: tuple[CAPCandidate, ...] | list[CAPCandidate],
    *,
    scores: Mapping[str, float] | None = None,
    budget: int,
) -> tuple[CAPCandidate, ...]:
    """Select candidates deterministically under a hard parameter budget."""

    if budget < 0:
        raise ValueError("budget must be non-negative")
    score_map = scores or {candidate.id: candidate.value / max(float(candidate.cost), 1.0) for candidate in candidates}
    selected: list[CAPCandidate] = []
    used = 0
    for candidate in sorted(candidates, key=lambda item: _candidate_sort_key(item, score_map.get(item.id, 0.0))):
        if used + candidate.cost > budget:
            continue
        selected.append(candidate)
        used += candidate.cost
    return tuple(selected)


def _selection_from_candidates(candidates: tuple[CAPCandidate, ...], selected: tuple[CAPCandidate, ...]) -> np.ndarray:
    selected_ids = {candidate.id for candidate in selected}
    return np.asarray([candidate.id in selected_ids for candidate in candidates], dtype=bool)


def _selected_candidates_from_selection(candidates: tuple[CAPCandidate, ...], selection: np.ndarray) -> tuple[CAPCandidate, ...]:
    return tuple(candidate for candidate, selected in zip(candidates, selection.astype(bool)) if bool(selected))


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
    pool = build_cap_candidate_pool({"matrix": matrix}, lam=lam, max_iter=max_iter)
    rpca = pool.decompositions["matrix"]
    if not pool.candidates:
        return _compression_from_selection(rpca, np.zeros(0, dtype=bool))

    selected = select_cap_candidates(pool.candidates, budget=budget)
    selection = _selection_from_candidates(pool.candidates, selected)
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


@dataclass(frozen=True)
class CAPGlobalPolicyStep:
    step: int
    loss: float
    parameter_count: int


@dataclass(frozen=True)
class CAPGlobalSearchResult:
    compressions: dict[str, CAPCompression]
    parameter_count: int
    best_loss: float
    history: tuple[CAPGlobalPolicyStep, ...]
    selected_candidates: tuple[CAPCandidate, ...] = ()


def _compressions_from_global_selection(
    rpcas: dict[str, RPCAResult],
    offsets: dict[str, tuple[int, int]],
    selection: np.ndarray,
) -> dict[str, CAPCompression]:
    return {
        name: _compression_from_selection(rpca, selection[start:end])
        for name, rpca in rpcas.items()
        for start, end in [offsets[name]]
    }


def _global_default_loss(matrices: dict[str, np.ndarray], compressions: dict[str, CAPCompression]) -> float:
    return float(np.mean([_relative_error(matrices[name], compression) for name, compression in compressions.items()]))


def _project_probabilities(probs: np.ndarray, costs: np.ndarray, budget: int) -> np.ndarray:
    clipped = np.clip(probs, 1e-3, 1.0 - 1e-3)
    expected = float(np.sum(clipped * costs))
    if expected > budget:
        clipped = clipped * (float(budget) / (expected + 1e-12))
    return np.clip(clipped, 1e-3, 1.0 - 1e-3)


def _select_by_probabilities(probs: np.ndarray, costs: np.ndarray, budget: int) -> np.ndarray:
    selection = np.zeros(len(probs), dtype=bool)
    used = 0
    for index in np.argsort(-probs):
        cost = int(costs[index])
        if used + cost > budget:
            continue
        selection[index] = True
        used += cost
    return selection


def optimize_global_rank_sparsity_for_matrices(
    matrices: dict[str, np.ndarray],
    *,
    total_budget: int,
    evaluator: Callable[[dict[str, CAPCompression]], float] | None = None,
    lam: float | None = None,
    max_iter: int = 200,
    policy_steps: int = 120,
    samples_per_step: int = 8,
    learning_rate: float = 0.1,
    budget_penalty: float = 2.0,
    seed: int | None = None,
) -> CAPGlobalSearchResult:
    """CAP Stage 2 global Bernoulli allocation across multiple matrices.

    ``evaluator`` receives a mapping from layer name to CAPCompression and returns
    a calibration loss. If omitted, the optimizer uses average reconstruction error.
    """

    if not matrices:
        raise ValueError("matrices must not be empty")
    if total_budget <= 0:
        raise ValueError("total_budget must be positive")
    if policy_steps <= 0 or samples_per_step <= 0:
        raise ValueError("policy_steps and samples_per_step must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    matrix_arrays = {name: np.asarray(matrix, dtype=np.float64) for name, matrix in matrices.items()}
    pool = build_cap_candidate_pool(matrix_arrays, lam=lam, max_iter=max_iter)
    rpcas = dict(pool.decompositions)
    offsets: dict[str, tuple[int, int]] = {}
    values_list: list[np.ndarray] = []
    costs_list: list[np.ndarray] = []
    cursor = 0
    for name, rpca in rpcas.items():
        _, _, values, costs, _ = _candidate_table(rpca)
        values_list.append(values)
        costs_list.append(costs)
        offsets[name] = (cursor, cursor + len(values))
        cursor += len(values)

    if cursor == 0:
        selection = np.zeros(0, dtype=bool)
        compressions = _compressions_from_global_selection(rpcas, offsets, selection)
        return CAPGlobalSearchResult(compressions, 0, _global_default_loss(matrix_arrays, compressions), tuple(), tuple())

    values = np.concatenate(values_list)
    costs = np.concatenate(costs_list).astype(np.float64)
    rng = np.random.default_rng(seed)
    probs = values / (float(values.max()) + 1e-12)
    probs = _project_probabilities(np.clip(probs, 0.05, 0.95), costs, total_budget)
    score_fn = evaluator or (lambda comps: _global_default_loss(matrix_arrays, comps))

    initial_selection = _select_by_probabilities(probs, costs, total_budget)
    initial_compressions = _compressions_from_global_selection(rpcas, offsets, initial_selection)
    best_selection = initial_selection.copy()
    best_loss = float(score_fn(initial_compressions))
    baseline = best_loss
    history: list[CAPGlobalPolicyStep] = []

    for step in range(1, policy_steps + 1):
        grad = np.zeros_like(probs)
        step_best_loss = float("inf")
        step_best_cost = 0
        for _ in range(samples_per_step):
            sample = rng.random(len(probs)) < probs
            cost = int(np.sum(costs * sample))
            compressions = _compressions_from_global_selection(rpcas, offsets, sample)
            loss = float(score_fn(compressions))
            overflow = max(0, cost - total_budget) / max(float(total_budget), 1.0)
            penalized_loss = loss + budget_penalty * overflow
            grad += (penalized_loss - baseline) * (sample.astype(np.float64) - probs) / (probs * (1.0 - probs) + 1e-8)
            if cost <= total_budget and loss < best_loss:
                best_loss = loss
                best_selection = sample.copy()
            if penalized_loss < step_best_loss:
                step_best_loss = penalized_loss
                step_best_cost = cost
        baseline = 0.9 * baseline + 0.1 * step_best_loss
        probs = _project_probabilities(probs - learning_rate * grad / float(samples_per_step), costs, total_budget)
        history.append(CAPGlobalPolicyStep(step, float(step_best_loss), int(step_best_cost)))

    deterministic = _select_by_probabilities(probs, costs, total_budget)
    deterministic_compressions = _compressions_from_global_selection(rpcas, offsets, deterministic)
    deterministic_loss = float(score_fn(deterministic_compressions))
    if deterministic_loss <= best_loss:
        final_selection = deterministic
        final_loss = deterministic_loss
    else:
        final_selection = best_selection
        final_loss = best_loss
    final_compressions = _compressions_from_global_selection(rpcas, offsets, final_selection)
    return CAPGlobalSearchResult(
        final_compressions,
        int(np.sum(costs * final_selection)),
        float(final_loss),
        tuple(history),
        _selected_candidates_from_selection(pool.candidates, final_selection),
    )
