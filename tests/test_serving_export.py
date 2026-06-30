import torch
from torch import nn

from tidal.methods.global_rank_sparsity.torch import CAPPackedLinear
from tidal.methods.qpruner.torch import QuantizedLinear


class TinyCompressedBlock(nn.Module):
    def __init__(self):
        super().__init__()
        left = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)
        right = torch.tensor([[4.0, 5.0]], dtype=torch.float32)
        sparse = torch.ones(3, 2, dtype=torch.float32) * 0.25
        self.cap = CAPPackedLinear(left, right, sparse, None, parameter_count=11)
        self.quant = QuantizedLinear.from_linear(nn.Linear(3, 2), bits=4)

    def forward(self, x):
        return self.quant(self.cap(x))


def test_export_compressed_linears_to_dense_replaces_custom_modules():
    from tidal.workflows.export import export_compressed_linears_to_dense

    torch.manual_seed(0)
    model = TinyCompressedBlock()
    inputs = torch.randn(4, 2)
    expected = model(inputs)

    exported = export_compressed_linears_to_dense(model, inplace=False)
    actual = exported(inputs)

    assert exported is not model
    assert isinstance(exported.cap, nn.Linear)
    assert isinstance(exported.quant, nn.Linear)
    assert isinstance(model.cap, CAPPackedLinear)
    assert isinstance(model.quant, QuantizedLinear)
    assert torch.allclose(actual, expected, atol=1e-5)


def test_export_compressed_linears_to_dense_can_update_inplace():
    from tidal.workflows.export import export_compressed_linears_to_dense

    model = TinyCompressedBlock()
    exported = export_compressed_linears_to_dense(model, inplace=True)

    assert exported is model
    assert isinstance(model.cap, nn.Linear)
    assert isinstance(model.quant, nn.Linear)
