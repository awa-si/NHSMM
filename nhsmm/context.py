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
        context ALWAYS has shape [B, 1, F]
        context.expand(S) -> [B, S, F]
    """

    def __init__(
        self,
        encoder: nn.Module,
        n_heads: int = 4,
        pool: Literal["mean", "last", "max", "attn", "mha"] = "mean",
        device: Optional[torch.device] = None,
        layer_norm: bool = True,
        dropout: float = 0.0,
        context_scale: float = 10.0,
        debug: bool = False,
    ):
        super().__init__()
        has_params = any(p.numel() > 0 for p in encoder.parameters())
        self.device = device or (next(encoder.parameters()).device if has_params else torch.device("cpu"))
        self.encoder = encoder.to(self.device, dtype=DTYPE)

        self.pool = pool.lower()
        self.n_heads = n_heads
        self.layer_norm_flag = layer_norm
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.context_scale = context_scale
        self.debug = debug

        self._context: Optional[torch.Tensor] = None
        self._attn_vector: Optional[nn.Parameter] = None
        self._mha: Optional[nn.MultiheadAttention] = None

        self._dbg(f"ContextEncoder initialized with pool={self.pool}")

    # ---------------- Debug ----------------
    def _dbg(self, msg: str):
        if self.debug and logger:
            logger.debug(f"[ContextEncoder] {msg}")

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
        x = x.to(self.device, dtype=DTYPE)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        B, T, F_in = x.shape

        if mask is not None:
            mask = mask.to(self.device, dtype=torch.bool)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0).expand(B, -1)
            mask = mask[:, :T]

        # Encode
        out = self.encoder(x, mask=mask if hasattr(self.encoder, "forward") else None)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.ndim != 3:
            raise ValueError(f"Encoder returned {out.shape}, expected [B, T, F]")

        theta = out
        if mask is not None:
            theta = theta * mask.unsqueeze(-1)

        # Pool
        pooled, attn = self._pool_context(theta, mask, return_attn_weights)

        # Normalize + scale
        if self.layer_norm_flag:
            pooled = F.layer_norm(pooled, (pooled.shape[-1],))
        pooled = self.dropout_layer(torch.tanh(pooled) * self.context_scale)

        # Save canonical context: [B, 1, F]
        ctx = pooled.unsqueeze(1)
        self._context = ctx.detach() if detach_context else ctx

        theta_out = theta if return_sequence else theta[:, -1, :]

        if return_context:
            return (theta_out, ctx, attn if return_attn_weights else None)
        return theta_out, None, None

    # ---------------- Pooling ----------------
    def _pool_context(self, theta: torch.Tensor, mask: Optional[torch.Tensor], ret_attn: bool):
        B, T, F_dim = theta.shape
        if T == 0:
            zero = torch.zeros((B, F_dim), device=self.device, dtype=DTYPE)
            return zero, None

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
            else:
                ctx = theta.max(dim=1).values
            return ctx, None

        if self.pool == "attn":
            return self._attention_context(theta, mask, ret_attn)

        if self.pool == "mha":
            return self._multihead_context(theta, mask, ret_attn)

        raise ValueError(f"Invalid pooling method '{self.pool}'")

    # ---------------- Soft Attention ----------------
    def _attention_context(self, theta, mask, ret_attn):
        B, T, F_dim = theta.shape
        if self._attn_vector is None or self._attn_vector.shape[0] != F_dim:
            self._attn_vector = nn.Parameter(torch.zeros(F_dim, device=self.device, dtype=DTYPE))
            nn.init.normal_(self._attn_vector, std=0.1)
            self._dbg("Initialized attention vector")

        scores = torch.einsum("btf,f->bt", theta, self._attn_vector)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
            scores[mask.sum(dim=1) == 0] = 0

        attn_w = F.softmax(scores, dim=1).unsqueeze(-1)
        ctx = (attn_w * theta).sum(dim=1)
        return ctx, attn_w if ret_attn else None

    # ---------------- Multihead Attention ----------------
    def _multihead_context(self, theta, mask, ret_attn):
        B, T, F_dim = theta.shape
        if self._mha is None:
            self._mha = nn.MultiheadAttention(
                F_dim, self.n_heads, batch_first=True, device=self.device, dtype=DTYPE
            )
            self._dbg("Initialized MHA pooling layer")
        key_padding_mask = ~mask if mask is not None else None
        out, attn = self._mha(theta, theta, theta, key_padding_mask=key_padding_mask)
        ctx = out.mean(dim=1)
        return ctx, attn if ret_attn else None

    # ---------------- Context Utilities ----------------
    def get_context(self, expand_to_states: Optional[int] = None, detach: bool = True):
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

