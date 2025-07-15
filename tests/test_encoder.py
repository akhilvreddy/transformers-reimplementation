import torch
from transformer.encoder import Encoder

def test_encoder_output_shape():
    encoder = Encoder(num_layers=2, d_model=512, num_heads=8, dim_ff=2048, vocab_size=1000)
    x = torch.randint(0, 1000, (4, 16))  # (B, T)
    mask = torch.ones(4, 1, 1, 16)
    out = encoder(x, mask)
    assert out.shape == (4, 16, 512)