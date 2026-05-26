import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch import nn

from tidal.methods.qpruner.torch import run_qpruner_mixed_precision



class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 3)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(3, 2)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


torch.manual_seed(0)
model = TinyBlock()
with torch.no_grad():
    model.fc1.weight.zero_()
    model.fc1.bias.zero_()
    model.fc1.weight[0, 0] = 1.0
    model.fc1.weight[1, 1] = 1.0
    model.fc1.weight[2, 2] = 1.0
    model.fc2.weight.zero_()
    model.fc2.bias[:] = torch.tensor([0.0, 1.0])
    model.fc2.weight[0, 0] = 2.0

calibration_batches = [torch.randn(8, 4), torch.randn(8, 4)]
run = run_qpruner_mixed_precision(
    model,
    calibration_batches,
    candidate_bits=(2, 4, 8),
    max_average_bits=3.5,
    bins=4,
)
output = run.model(torch.randn(2, 4))

print({"importance": {name: round(value, 4) for name, value in run.importances.items()}})
print({"bitwidths": run.plan.bitwidths, "layer_sizes": run.plan.layer_sizes})
print({"backend": type(run.model.fc1).__name__, "output_shape": tuple(output.shape)})
