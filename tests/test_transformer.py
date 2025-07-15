import torch
from transformer.transformer import Transformer

def test_transformer_forward_pass():
    
    batch_size = 2
    src_len = 10
    tgt_len = 7
    d_model = 512
    vocab_size = 1000
    num_layers = 2
    num_heads = 8
    dim_ff = 2048

    # dummy input
    src = torch.randint(0, vocab_size, (batch_size, src_len))  # (B, S)
    tgt = torch.randint(0, vocab_size, (batch_size, tgt_len))  # (B, T)

    # masks
    src_mask = torch.ones(batch_size, 1, 1, src_len).bool()    # (B, 1, 1, S)
    tgt_mask = torch.tril(torch.ones(tgt_len, tgt_len)).unsqueeze(0).unsqueeze(0).bool()  # (1, 1, T, T)

    # model
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        dim_ff=dim_ff
    )

    out = model(src, tgt, src_mask, tgt_mask)

    # final output: (B, T, vocab_size)
    assert out.shape == (batch_size, tgt_len, vocab_size)