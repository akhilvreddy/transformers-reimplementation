import torch
from transformer.attention import MultiHeadAttention, ScaledDotProductAttention

def test_multihead_attention_shapes():
    mha = MultiHeadAttention(d_model=512, num_heads=8)
    x = torch.randn(2, 10, 512)  # (B, T, D)
    out = mha(x, x, x)
    assert out.shape == (2, 10, 512)

def test_scaled_dot_product_attention_shapes():
    B, T, d_k = 2, 10, 64
    Q = torch.randn(B, T, d_k)
    K = torch.randn(B, T, d_k)
    V = torch.randn(B, T, d_k)

    attn = ScaledDotProductAttention()
    out, weights = attn(Q, K, V)

    # Output shape should match value shape
    assert out.shape == (B, T, d_k)
    
    # Attention weights should be (B, T, T)
    assert weights.shape == (B, T, T)