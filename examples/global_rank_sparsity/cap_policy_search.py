import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from tidal.methods.global_rank_sparsity import optimize_global_rank_sparsity

low_rank = np.array([
    [1.0, 2.0, 3.0],
    [2.0, 4.0, 6.0],
    [3.0, 6.0, 9.0],
])
sparse = np.zeros_like(low_rank)
sparse[0, 2] = 4.0
sparse[2, 0] = -3.0
matrix = low_rank + sparse

result = optimize_global_rank_sparsity(
    matrix,
    budget=9,
    max_iter=80,
    policy_steps=40,
    samples_per_step=6,
    seed=7,
)
error = np.linalg.norm(matrix - result.compression.reconstructed) / np.linalg.norm(matrix)
print({"parameters": result.compression.parameter_count, "relative_error": round(float(error), 4)})
