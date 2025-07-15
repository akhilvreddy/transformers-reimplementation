import torch
from transformer.utils import generate_padding_mask, generate_subsequent_mask

def test_padding_mask():
    x = torch.tensor([[1, 2, 0], [3, 0, 0]])
    mask = generate_padding_mask(x, pad_idx=0)
    assert mask.shape == (2, 1, 1, 3)

def test_subsequent_mask():
    mask = generate_subsequent_mask(5)
    assert mask.shape == (1, 1, 5, 5)
    assert torch.equal(mask[0, 0], torch.tril(torch.ones(5, 5)).bool())