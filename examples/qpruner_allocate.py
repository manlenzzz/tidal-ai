import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from tidal.qpruner import allocate_bitwidths, layer_mutual_information

rng = np.random.default_rng(0)
predictions = rng.normal(size=128)
layer_outputs = {
    "layers.0": predictions[:, None] + 0.3 * rng.normal(size=(128, 4)),
    "layers.1": rng.normal(size=(128, 4)),
    "layers.2": 0.5 * predictions[:, None] + 0.5 * rng.normal(size=(128, 4)),
}
mi = layer_mutual_information(layer_outputs, predictions)
print(allocate_bitwidths(mi, {name: 1024 for name in mi}, max_average_bits=3.0))
