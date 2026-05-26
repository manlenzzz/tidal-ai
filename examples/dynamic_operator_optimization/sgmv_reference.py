import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from tidal.methods.dynamic_operator_optimization.reference import Segment, lora_sgmv

inputs = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
a_weights = np.array([[[1.0], [2.0]], [[2.0], [1.0]]])
b_weights = np.array([[[3.0, 4.0]], [[1.0, 5.0]]])
segments = [Segment(0, 2, 0), Segment(2, 3, 1)]

print(lora_sgmv(inputs, a_weights, b_weights, segments))
