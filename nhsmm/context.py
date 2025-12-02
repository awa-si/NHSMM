# nhsmm/context.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal, Tuple

from nhsmm.constants import DTYPE, logger


class ContextEncoder(nn.Module):
    """
    Batch-native context encoder:
      - Supports masking and optional attention/multihead pooling
      - Returns per-timestep sequence features [B,T,F]
      - Returns pooled canonical context [B,1,F]
    """

    def __init__(
        self,
        encoder: nn.Module,
        n_heads: int = 4,
        dropout: float = 0.0,
        layer_norm: bool = True,
        context_scale: float = 1.0,
        pool: Literal["mean", "last", "max", "attn", "mha"] = "mean",
        debug: bool = False,
    ):
        super().__init__()

        self.encoder = encoder
        self.pool = pool.lower()
        self.context_scale = context_scale
        self.layer_norm = layer_norm
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.debug = debug
        self.n_heads = n_heads

        self._sequence: Optional[torch.Tensor] = None
        self._context: Optional[torch.Tensor] = None
        self._attn_vector: Optional[nn.Parameter] = None
        self._mha: Optional[nn.MultiheadAttention] = None

        if self.debug and logger:
            logger.debug(f"[ContextEncoder] Initialized with pool={self.pool}")

    # ---------------- Forward ----------------
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
        return_context: bool = False,
        return_attn_weights: bool = False,
        detach_context: bool = True,
        return_sequence: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if x.ndim == 2:
            x = x.unsqueeze(0)  # single batch

        B, T, F_in = x.shape
        mask = self._prepare_mask(mask, B, T)

        out = self.encoder(x, mask=mask)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.ndim != 3:
            raise ValueError(f"Encoder returned {out.shape}, expected [B,T,F]")

        theta = out * mask.unsqueeze(-1) if mask is not None else out
        self._sequence = theta.detach() if detach_context else theta

        pooled, attn = self._pool_context(theta, mask, return_attn_weights)

        if self.layer_norm:
            pooled = F.layer_norm(pooled, (pooled.shape[-1],))
        pooled = self.dropout_layer(torch.tanh(pooled) * self.context_scale)
        ctx = pooled.unsqueeze(1)
        self._context = ctx.detach() if detach_context else ctx

        if return_sequence:
            seq_out = theta
        else:
            seq_out = self._last_timestep(theta, mask)

        return seq_out, ctx if return_context else None, attn if return_attn_weights else None

    # ---------------- Utilities ----------------
    def _prepare_mask(self, mask: Optional[torch.BoolTensor], B: int, T: int) -> Optional[torch.BoolTensor]:
        if mask is None:
            return None
        mask = mask.bool()
        if mask.ndim == 1:
            mask = mask.unsqueeze(0).expand(B, -1)
        return mask[:, :T]

    def _last_timestep(self, theta: torch.Tensor, mask: Optional[torch.BoolTensor]) -> torch.Tensor:
        B, T, F = theta.shape
        if mask is not None:
            lengths = mask.sum(dim=1).clamp_min(1)
            idx = lengths - 1
            return theta[torch.arange(B), idx]
        return theta[:, -1, :]

    # ---------------- Pooling ----------------
    def _pool_context(
        self, theta: torch.Tensor, mask: Optional[torch.BoolTensor], ret_attn: bool
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T, F = theta.shape
        if T == 0:
            return torch.zeros((B, F), dtype=theta.dtype, device=theta.device), None

        if self.pool == "last":
            return self._last_timestep(theta, mask), None

        if self.pool == "mean":
            if mask is not None:
                denom = mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
                ctx = (theta * mask.unsqueeze(-1)).sum(dim=1) / denom
            else:
                ctx = theta.mean(dim=1)
            return ctx, None

        if self.pool == "max":
            if mask is not None:
                masked = theta.masked_fill(~mask.unsqueeze(-1), float("-inf"))
                ctx = masked.max(dim=1).values
                ctx[torch.isinf(ctx)] = 0
            else:
                ctx = theta.max(dim=1).values
            return ctx, None

        if self.pool == "attn":
            return self._attention_context(theta, mask, ret_attn)

        if self.pool == "mha":
            return self._multihead_context(theta, mask, ret_attn)

        raise ValueError(f"Invalid pooling method '{self.pool}'")

    # ---------------- Attention ----------------
    def _init_attn_vector(self, F: int, device: torch.device):
        if self._attn_vector is None or self._attn_vector.shape[0] != F:
            vec = torch.zeros(F, dtype=DTYPE, device=device)
            nn.init.normal_(vec, std=0.1)
            self._attn_vector = nn.Parameter(vec)

    def _attention_context(
        self, theta: torch.Tensor, mask: Optional[torch.BoolTensor], ret_attn: bool
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T, F = theta.shape
        self._init_attn_vector(F, theta.device)

        scores = torch.einsum("btf,f->bt", theta, self._attn_vector)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
            empty_rows = mask.sum(dim=1) == 0
            if empty_rows.any():
                scores[empty_rows, :] = 0
        attn_w = F.softmax(scores, dim=1).unsqueeze(-1)
        ctx = (attn_w * theta).sum(dim=1)
        return ctx, attn_w if ret_attn else None

    # ---------------- Multihead Attention ----------------
    def _multihead_context(
        self, theta: torch.Tensor, mask: Optional[torch.BoolTensor], ret_attn: bool
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T, F = theta.shape
        if self._mha is None:
            self._mha = nn.MultiheadAttention(F, self.n_heads, batch_first=True, device=theta.device)
        key_padding_mask = (~mask.bool()) if mask is not None else None
        out, attn = self._mha(theta, theta, theta, key_padding_mask=key_padding_mask)
        ctx = out.mean(dim=1)
        return ctx, attn if ret_attn else None

    # ---------------- Sequence / Context Access ----------------
    def get_sequence(self, detach: bool = True) -> Optional[torch.Tensor]:
        return self._sequence.detach() if detach and self._sequence is not None else self._sequence

    def get_context(self, detach: bool = True) -> Optional[torch.Tensor]:
        return self._context.detach() if detach and self._context is not None else self._context

    def set_sequence(
        self,
        sequence: torch.Tensor,
        detach: bool = True,
        recompute_context: bool = False,
        mask: Optional[torch.BoolTensor] = None,
    ):
        self._sequence = sequence.detach() if detach else sequence
        if recompute_context:
            pooled, _ = self._pool_context(sequence, mask, ret_attn=False)
            if self.layer_norm:
                pooled = F.layer_norm(pooled, (pooled.shape[-1],))
            pooled = self.dropout_layer(torch.tanh(pooled) * self.context_scale)
            self._context = pooled.unsqueeze(1)

    def set_context(self, context: torch.Tensor, detach: bool = True):
        if context.ndim == 2:
            context = context.unsqueeze(1)
        self._context = context.detach() if detach else context

    def reset(self):
        self._sequence = None
        self._context = None

