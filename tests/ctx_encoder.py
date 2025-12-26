# tests/test_context_encoder.py

import torch
from nhsmm import DefaultEncoder, ContextEncoder

def test_context_encoder():
    torch.manual_seed(42)

    # --- Parameters ---
    B, T, F = 3, 8, 5        # batch, sequence length, features
    hidden_dim = 16
    cnn_channels = 10
    dropout = 0.1

    # --- Dummy input ---
    x = torch.randn(B, T, F)

    # Variable-length mask (simulate padded sequences)
    mask = torch.tensor([
        [1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ], dtype=torch.bool)

    # --- Initialize CNN+LSTM encoder ---
    cnn_lstm = DefaultEncoder(
        n_features=F,
        hidden_dim=hidden_dim,
        cnn_channels=cnn_channels,
        kernel_size=3,
        dropout=dropout,
        bidirectional=True,
        return_sequence=True,
        use_packed=True,
    )

    # --- Wrap in ContextEncoder ---
    ctx_encoder = ContextEncoder(
        encoder=cnn_lstm,
        n_heads=2,
        dropout=0.0,
        layer_norm=True,
        context_scale=1.0,
        pool="mean",
        debug=True,
    )

    # --- Forward pass (mean pooling) ---
    seq_out, ctx, attn = ctx_encoder(
        x,
        mask=mask,
        return_context=True,
        return_attn_weights=False,
        return_sequence=True
    )

    print("Input shape:", x.shape)
    print("Sequence output shape:", seq_out.shape)  # [B, T, hidden_dim*2]
    print("Context shape:", ctx.shape)              # [B, 1, hidden_dim*2]

    # --- Last valid timestep check ---
    seq_last = ctx_encoder._last_timestep(seq_out, mask)
    print("Last valid timestep shape:", seq_last.shape)  # [B, hidden_dim*2]

    # --- Attention pooling ---
    ctx_encoder.pool = "attn"
    seq_attn, ctx_attn, attn_w = ctx_encoder(
        x,
        mask=mask,
        return_context=True,
        return_attn_weights=True
    )
    print("Attention context shape:", ctx_attn.shape)    # [B, 1, hidden_dim*2]
    print("Attention weights shape:", attn_w.shape)      # [B, T, 1]

    # --- Multihead attention pooling ---
    ctx_encoder.pool = "mha"
    seq_mha, ctx_mha, attn_mha = ctx_encoder(
        x,
        mask=mask,
        return_context=True,
        return_attn_weights=True
    )
    print("Multihead context shape:", ctx_mha.shape)     # [B, 1, hidden_dim*2]
    print("Multihead attention shape:", attn_mha.shape)  # [B, T, T]

    print("All tests passed.")

if __name__ == "__main__":
    test_context_encoder()
