import torch.nn as nn
from .attention import MultiHeadAttention

# class DecoderLayer(nn.Module):
#     def __init__(self, d_model, num_heads, dim_ff, dropout=0.1):
#         super().__init__()
#         self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
#         self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
#         self.ff = nn.Sequential(
#             nn.Linear(d_model, dim_ff),
#             nn.ReLU(),
#             nn.Linear(dim_ff, d_model)
#         )
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.norm3 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x, enc_out, src_mask, tgt_mask):
#         x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
#         x = self.norm2(x + self.dropout(self.cross_attn(x, enc_out, enc_out, src_mask)))
#         x = self.norm3(x + self.dropout(self.ff(x)))
#         return x

# class Decoder(nn.Module):
#     def __init__(self, num_layers, d_model, num_heads, dim_ff, vocab_size):
#         super().__init__()
#         from .embeddings import TokenEmbedding, PositionalEncoding
#         self.embedding = TokenEmbedding(vocab_size, d_model)
#         self.pos_enc = PositionalEncoding(d_model)
#         self.layers = nn.ModuleList([
#             DecoderLayer(d_model, num_heads, dim_ff) for _ in range(num_layers)
#         ])
#         self.out = nn.Linear(d_model, vocab_size)

#     def forward(self, x, enc_out, src_mask, tgt_mask):
#         x = self.embedding(x)
#         x = self.pos_enc(x)
#         for layer in self.layers:
#             x = layer(x, enc_out, src_mask, tgt_mask)
#         return self.out(x)

class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dim_ff, vocab_size):
        super().__init__()
        from .embeddings import TokenEmbedding, PositionalEncoding
        self.embedding = TokenEmbedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, dim_ff) for _ in range(num_layers)
        ])
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, x, enc_out=None, src_mask=None, tgt_mask=None):
        x = self.embedding(x)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, enc_out=enc_out, src_mask=src_mask, tgt_mask=tgt_mask)
        return self.out(x)
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out=None, src_mask=None, tgt_mask=None):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))

        if enc_out is not None:
            x = self.norm2(x + self.dropout(self.cross_attn(x, enc_out, enc_out, src_mask)))
        else:
            x = self.norm2(x)

        x = self.norm3(x + self.dropout(self.ff(x)))
        return x