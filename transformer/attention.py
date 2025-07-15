import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / d_k ** 0.5
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = self.dropout(F.softmax(scores, dim=-1))
        return torch.matmul(attn, V), attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.attn = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B = q.size(0)
        T_q = q.size(1)

        def transform(x, linear):
            if x is None:
                return None
            x = linear(x)
            B, T, _ = x.size()
            return x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        q = transform(q, self.q_linear)  # (B, H, T_q, d_k)
        k = transform(k, self.k_linear)  # (B, H, T_k, d_k)
        v = transform(v, self.v_linear)  # (B, H, T_k, d_k)

        out, _ = self.attn(q, k, v, mask)  # (B, H, T_q, d_k)
        out = out.transpose(1, 2).contiguous().view(B, T_q, -1)  # (B, T_q, D)
        return self.out(out)