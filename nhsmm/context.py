import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal, Tuple

from nhsmm.constants import DTYPE, logger


class ContextEncoder(nn.Module):
    """
    Wrapper for sequence encoders producing:
      - sequence features [B, T, F]
      - pooled context [B, 1, F]
      - optional attention maps

    Unified context rule:
      - context always has shape [B, 1, F]
      - context.expand(S) -> [B, S, F]
    """

    def __init__(
        self,
        encoder: nn.Module,
        pool: Literal["mean", "last", "max", "attn", "mha"] = "mean",
        context_scale: float = 10.0,
        layer_norm: bool = True,
        dropout: float = 0.0,
        debug: bool = False,
        n_heads: int = 4,
    ):
        super().__init__()
        self.encoder = encoder
        self.pool = pool.lower()
        self.n_heads = n_heads
        self.layer_norm = layer_norm
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.context_scale = context_scale
        self.debug = debug

        self._context: Optional[torch.Tensor] = None
        self._attn_vector: Optional[nn.Parameter] = None
        self._mha: Optional[nn.MultiheadAttention] = None

        if self.debug and logger:
            logger.debug(f"[ContextEncoder] Initialized with pool={self.pool}")

    # ---------------- Forward ----------------
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_context: bool = False,
        return_attn_weights: bool = False,
        detach_context: bool = True,
        return_sequence: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if x.ndim == 2:
            x = x.unsqueeze(0)
        B, T, F_in = x.shape

        if mask is not None:
            mask = mask.bool()
            if mask.ndim == 1:
                mask = mask.unsqueeze(0).expand(B, -1)
            mask = mask[:, :T]

        # Encode sequence
        out = self.encoder(x, mask=mask)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.ndim != 3:
            raise ValueError(f"Encoder returned {out.shape}, expected [B, T, F]")

        theta = out
        if mask is not None:
            theta = theta * mask.unsqueeze(-1)

        # Pool to context
        pooled, attn = self._pool_context(theta, mask, return_attn_weights)

        # Normalize + scale
        if self.layer_norm:
            pooled = F.layer_norm(pooled, (pooled.shape[-1],))
        pooled = self.dropout_layer(torch.tanh(pooled) * self.context_scale)

        # canonical context [B, 1, F]
        ctx = pooled.unsqueeze(1)
        self._context = ctx.detach() if detach_context else ctx

        theta_out = theta if return_sequence else theta[:, -1, :]
        if mask is not None and not return_sequence:
            # use last valid timestep instead of simply theta[:, -1, :]
            lengths = mask.sum(dim=1).clamp_min(1)
            idx = lengths - 1
            theta_out = theta[torch.arange(B), idx]

        if return_context:
            return theta_out, ctx, attn if return_attn_weights else None
        return theta_out, None, None

    # ---------------- Pooling ----------------
    def _pool_context(self, theta: torch.Tensor, mask: Optional[torch.Tensor], ret_attn: bool) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T, F_dim = theta.shape
        if T == 0:
            return torch.zeros((B, F_dim), dtype=DTYPE), None

        if self.pool == "last":
            if mask is not None:
                lengths = mask.sum(dim=1).clamp_min(1)
                idx = lengths - 1
                ctx = theta[torch.arange(B), idx]
            else:
                ctx = theta[:, -1]
            return ctx, None

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
                ctx[torch.isinf(ctx)] = 0  # handle all-padding sequences
            else:
                ctx = theta.max(dim=1).values
            return ctx, None

        if self.pool == "attn":
            return self._attention_context(theta, mask, ret_attn)

        if self.pool == "mha":
            return self._multihead_context(theta, mask, ret_attn)

        raise ValueError(f"Invalid pooling method '{self.pool}'")

    # ---------------- Soft Attention ----------------
    def _attention_context(self, theta: torch.Tensor, mask: Optional[torch.Tensor], ret_attn: bool) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T, F_dim = theta.shape
        if self._attn_vector is None or self._attn_vector.shape[0] != F_dim:
            self._attn_vector = nn.Parameter(torch.zeros(F_dim, dtype=DTYPE))
            nn.init.normal_(self._attn_vector, std=0.1)
            if self.debug and logger:
                logger.debug("[ContextEncoder] Initialized attention vector")
        if self._attn_vector.device != theta.device:
            self._attn_vector = self._attn_vector.to(theta.device)

        scores = torch.einsum("btf,f->bt", theta, self._attn_vector)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
            scores[mask.sum(dim=1) == 0] = 0
        attn_w = F.softmax(scores, dim=1).unsqueeze(-1)
        ctx = (attn_w * theta).sum(dim=1)
        return ctx, attn_w if ret_attn else None

    # ---------------- Multihead Attention ----------------
    def _multihead_context(self, theta: torch.Tensor, mask: Optional[torch.Tensor], ret_attn: bool) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T, F_dim = theta.shape
        if self._mha is None:
            self._mha = nn.MultiheadAttention(F_dim, self.n_heads, batch_first=True)
            if self.debug and logger:
                logger.debug("[ContextEncoder] Initialized MHA pooling layer")
        key_padding_mask = ~mask if mask is not None else None
        out, attn = self._mha(theta, theta, theta, key_padding_mask=key_padding_mask)
        ctx = out.mean(dim=1)
        return ctx, attn if ret_attn else None

    # ---------------- Context Utilities ----------------
    def get_context(self, expand_to_states: Optional[int] = None, detach: bool = True) -> Optional[torch.Tensor]:
        ctx = self._context
        if ctx is None:
            return None
        if expand_to_states is not None:
            ctx = ctx.expand(-1, expand_to_states, -1)
        return ctx.detach() if detach else ctx

    def set_context(self, context: torch.Tensor, detach: bool = True):
        if context.ndim == 2:
            context = context.unsqueeze(1)
        self._context = context.detach() if detach else context

    def reset_context(self):
        self._context = None
