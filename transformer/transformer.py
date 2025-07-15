import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_layers=6, num_heads=8, dim_ff=2048):
        super().__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dim_ff, src_vocab_size)
        self.decoder = Decoder(num_layers, d_model, num_heads, dim_ff, tgt_vocab_size)

    def forward(self, src, tgt, src_mask, tgt_mask):
        enc_out = self.encoder(src, src_mask)
        out = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        return out