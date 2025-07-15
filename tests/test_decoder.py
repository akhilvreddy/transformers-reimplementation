import torch
from transformer.decoder import Decoder

def test_decoder_forward_pass():
    batch_size = 2
    seq_len = 5
    d_model = 512
    num_layers = 2
    num_heads = 8
    dim_ff = 2048
    vocab_size = 1000

    # inputs
    tgt = torch.randint(0, vocab_size, (batch_size, seq_len)) # (B, T)
    enc_out = torch.randn(batch_size, seq_len, d_model) # (B, T, D)
    src_mask = torch.ones(batch_size, 1, 1, seq_len).bool() # (B, 1, 1, T)
    tgt_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0).bool() # (1, 1, T, T)

    # model
    decoder = Decoder(num_layers, d_model, num_heads, dim_ff, vocab_size)
    out = decoder(tgt, enc_out, src_mask, tgt_mask)

    # should return logits for each token in tgt
    assert out.shape == (batch_size, seq_len, vocab_size)