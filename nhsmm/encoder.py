# nhsmm/encoder.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as nnF

from nhsmm.config import logger


class DefaultEncoder(nn.Module):
    """
    Default CNN + LSTM encoder for sequences.
    Returns per-timestep features and pooled canonical context.
    """

    def __init__(
        self,
        n_features: int,
        kernel_size: int = 3,
        hidden_dim: int = 32,
        cnn_channels: int = 16,
        bidirectional: bool = True,
        return_sequence: bool = True,
        use_packed: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.cnn_channels = cnn_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bidirectional = bidirectional
        self.return_sequence = return_sequence
        self.use_packed = use_packed

        self.conv = nn.Conv1d(n_features, cnn_channels, kernel_size, padding=self.padding)
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

        self.norm = nn.LayerNorm(cnn_channels)
        self.lstm = nn.LSTM(
            cnn_channels,
            hidden_dim,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.dropout = nn.Dropout(dropout)
        self.out_dim = hidden_dim * (2 if bidirectional else 1)

        self._context = None

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, return_sequence: bool | None = None):
        # canonicalize input
        if x.ndim == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 2:
            x = x.unsqueeze(0)
        B, T, F_in = x.shape
        if F_in != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {F_in}")
        if T == 0:
            raise ValueError("Input sequence has zero length")

        # canonicalize mask
        if mask is not None:
            mask = mask.bool()
            if mask.ndim == 1:
                mask = mask.unsqueeze(0).expand(B, -1)
            elif mask.shape[0] == 1 and B > 1:
                mask = mask.expand(B, -1)
            mask = mask[:, :T]

        # --- CNN ---
        x_c = x.transpose(1, 2)  # [B,F,T]
        x_c = nnF.relu(self.conv(x_c))
        x_c = x_c.transpose(1, 2)  # [B,T,F]
        x_c = self.norm(x_c)
        x_c = self.dropout(x_c)

        # --- LSTM ---
        if self.use_packed and mask is not None:
            lengths = mask.sum(dim=1).clamp_min(1)
            packed = nn.utils.rnn.pack_padded_sequence(x_c, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out_packed, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True, total_length=T)
        else:
            out, _ = self.lstm(x_c)
        out = self.dropout(out)

        # --- Pooled canonical context ---
        if mask is not None:
            mask_f = mask.unsqueeze(-1).to(dtype=out.dtype)
            denom = mask_f.sum(dim=1).clamp_min(1.0)
            pooled = (out * mask_f).sum(dim=1) / denom
        else:
            pooled = out.mean(dim=1)
        self._context = pooled.unsqueeze(1)  # [B,1,H]

        # --- Return sequence or last valid timestep ---
        ret_seq = self.return_sequence if return_sequence is None else bool(return_sequence)
        if ret_seq:
            return out
        if mask is not None:
            idx = mask.sum(dim=1).clamp_min(1) - 1
            return out[torch.arange(B), idx]
        return out[:, -1, :]

