import torch

from src.model import MultiHeadCausalAttention, NanoForecaster, causal_mask
from src.mlp import MLPForecaster


def test_causal_mask_shape_and_values():
    expected = torch.tensor([[False, True, True], [False, False, True], [False, False, False]])
    assert torch.equal(causal_mask(3), expected)


def test_future_tokens_have_zero_attention_weight():
    layer = MultiHeadCausalAttention(8, 2, 0.0).eval()
    _, weights = layer(torch.randn(2, 5, 8), return_weights=True)
    forbidden = causal_mask(5).expand_as(weights)
    assert torch.all(weights[forbidden] == 0)


def test_output_shapes():
    x = torch.randn(4, 12, 7)
    transformer = NanoForecaster(7, 12, 6, d_model=16, n_heads=4, n_layers=1, d_ff=32)
    mlp = MLPForecaster(7, 12, 6, [16])
    assert transformer(x).shape == (4, 6)
    assert mlp(x).shape == (4, 6)

