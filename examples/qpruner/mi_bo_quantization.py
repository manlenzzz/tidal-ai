import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from tidal.methods.qpruner import bayesian_refine_bitwidths, layer_mutual_information

rng = np.random.default_rng(0)
predictions = rng.normal(size=128)
layer_outputs = {
    "layers.0": predictions[:, None] + 0.3 * rng.normal(size=(128, 4)),
    "layers.1": rng.normal(size=(128, 4)),
    "layers.2": 0.5 * predictions[:, None] + 0.5 * rng.normal(size=(128, 4)),
}
mi = layer_mutual_information(layer_outputs, predictions)
layer_sizes = {name: 1024 for name in mi}


def objective(config):
    return sum(mi[name] * np.log2(config[name]) for name in config)

result = bayesian_refine_bitwidths(
    mi,
    layer_sizes,
    objective=objective,
    candidate_bits=(2, 4, 8),
    max_average_bits=3.0,
    max_trials=5,
    seed=0,
)
print(result.config)
print({"score": round(result.score, 4), "memory_bits": result.memory_bits, "trials": len(result.evaluations)})
