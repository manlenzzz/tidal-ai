import numpy as np

from tidal.methods.global_rank_sparsity import (
    CAPCandidateKind,
    build_cap_candidate_pool,
    select_cap_candidates,
    optimize_global_rank_sparsity_for_matrices,
)


def structured_matrix():
    low_rank = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0],
        ]
    )
    sparse = np.zeros_like(low_rank)
    sparse[0, 2] = 4.0
    sparse[2, 0] = -3.0
    return low_rank + sparse


def test_cap_candidate_pool_exposes_stage_one_metadata_and_costs():
    pool = build_cap_candidate_pool({"layer_a": structured_matrix()}, max_iter=80)

    assert pool.layers == ("layer_a",)
    assert pool.total_candidates > 0
    assert np.allclose(pool.reconstruct("layer_a"), structured_matrix(), atol=1e-3)

    rank_candidates = [candidate for candidate in pool.candidates if candidate.kind is CAPCandidateKind.RANK]
    sparse_candidates = [candidate for candidate in pool.candidates if candidate.kind is CAPCandidateKind.SPARSE]

    assert rank_candidates
    assert sparse_candidates
    assert {candidate.cost for candidate in rank_candidates} == {6}
    assert {candidate.cost for candidate in sparse_candidates} == {1}
    assert all(candidate.layer == "layer_a" for candidate in pool.candidates)


def test_cap_candidate_selection_respects_budget_and_tie_order():
    pool = build_cap_candidate_pool({"b_layer": structured_matrix(), "a_layer": structured_matrix()}, max_iter=80)
    scores = {candidate.id: 1.0 for candidate in pool.candidates}

    selected = select_cap_candidates(pool.candidates, scores=scores, budget=7)

    assert sum(candidate.cost for candidate in selected) <= 7
    assert selected[0].layer == "a_layer"
    assert selected[0].kind is CAPCandidateKind.RANK
    assert selected[0].local_index == 0
    assert selected == select_cap_candidates(pool.candidates, scores=scores, budget=7)


def test_global_cap_search_exposes_selected_candidate_metadata_and_budget_validity():
    matrices = {
        "layer_a": structured_matrix(),
        "layer_b": structured_matrix() * 0.5,
    }
    result = optimize_global_rank_sparsity_for_matrices(
        matrices,
        total_budget=12,
        max_iter=80,
        policy_steps=8,
        samples_per_step=3,
        seed=3,
    )

    assert result.parameter_count <= 12
    assert result.selected_candidates
    assert sum(candidate.cost for candidate in result.selected_candidates) == result.parameter_count
    assert {candidate.layer for candidate in result.selected_candidates} <= set(matrices)
    for name, compression in result.compressions.items():
        assert compression.parameter_count <= result.parameter_count
        assert compression.low_rank.shape == matrices[name].shape
        assert compression.sparse.shape == matrices[name].shape
