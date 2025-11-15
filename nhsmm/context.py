import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal, Union, Tuple
from nhsmm.constants import DTYPE, EPS, logger


class ContextEncoder(nn.Module):
    """
    Sequence encoder wrapper for HSMM/HMM/CRF modules.
    Produces context suitable for Initial, Duration, and Transition distributions.

    Args:
        encoder: nn.Module producing [B, T, F] or [T, F].
        n_heads: Number of heads for multihead attention pooling.
        pool: Pooling method: "mean", "last", "max", "attn", "mha".
        device: torch.device to place module and buffers.
        layer_norm: Apply layer normalization to pooled context.
        dropout: Dropout rate on pooled context.
        context_scale: Scaling factor applied after tanh to pooled context.
        debug: Enable debug logging.
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

        self.device = device or (next(encoder.parameters()).device if any(encoder.parameters()) else torch.device("cpu"))
        self._context: Optional[torch.Tensor] = None
        self.pool = pool.lower()
        self.encoder = encoder.to(self.device, dtype=DTYPE)
        self.n_heads = n_heads
        self.layer_norm_flag = layer_norm
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.context_scale = context_scale
        self.debug = debug

        self._attn_vector = nn.Parameter(torch.empty(0, device=self.device, dtype=DTYPE))
        self._mha: Optional[nn.MultiheadAttention] = None
        self._dbg(f"ContextEncoder initialized: pool={self.pool}, device={self.device}")

    def _dbg(self, msg: str):
        if self.debug and logger:
            logger.debug(f"[ContextEncoder] {msg}")

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_context: bool = False,
        return_attn_weights: bool = False,
        detach_context: bool = True,
        return_sequence: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Encode a sequence and optionally produce pooled context for HSMM modules.

        Returns:
            theta_out: [B, F] or [B, T, F] depending on return_sequence
            context: pooled context [B, F]
            attn_weights: optional attention weights [B, T, 1] or None
        """
        x = x.to(self.device, dtype=DTYPE)
        if x.ndim == 2:
            x = x.unsqueeze(0)  # [1, T, F]
        B, T, F = x.shape

        if mask is not None:
            mask = mask.to(self.device, dtype=torch.bool)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0).expand(B, -1)

        out = self.encoder(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        assert out.ndim == 3, f"Expected [B,T,F], got {out.shape}"
        theta = out
        if mask is not None:
            theta = theta * mask.unsqueeze(-1)

        # Pooling
        context, attn_weights = self._pool_context(theta, mask, return_attn_weights)

        # Layer norm + tanh scaling + dropout
        if self.layer_norm_flag:
            context = F.layer_norm(context, context.shape[-1:])
        context = self.dropout(torch.tanh(context) * self.context_scale)

        self._context = context.detach() if detach_context else context

        theta_out = theta if return_sequence else theta[:, -1, :]

        if return_context:
            return (theta_out, context, attn_weights) if return_attn_weights else (theta_out, context)
        return theta_out

    # ---------------- Pooling ----------------
    def _pool_context(self, theta: torch.Tensor, mask: Optional[torch.Tensor], return_attn: bool = False):
        B, T, F = theta.shape
        attn_weights = None

        if T == 0:
            context = torch.zeros((B, F), device=self.device, dtype=DTYPE)
        elif self.pool == "last":
            if mask is not None:
                lengths = mask.sum(dim=1).clamp_min(1)
                idx = (lengths - 1).unsqueeze(-1).expand(-1, F)
                context = theta.gather(1, idx[:, None, :]).squeeze(1)
            else:
                context = theta[:, -1, :]
        elif self.pool == "mean":
            if mask is not None:
                denom = mask.sum(dim=1, keepdim=True).clamp_min(1)
                context = (theta * mask.unsqueeze(-1)).sum(dim=1) / denom
            else:
                context = theta.mean(dim=1)
        elif self.pool == "max":
            if mask is not None:
                masked = theta.masked_fill(~mask.unsqueeze(-1), float("-inf"))
                context = masked.max(dim=1).values
            else:
                context = theta.max(dim=1).values
        elif self.pool == "attn":
            context, attn_weights = self._attention_context(theta, mask, return_attn)
        elif self.pool == "mha":
            context, attn_weights = self._multihead_context(theta, mask, return_attn)
        else:
            raise ValueError(f"Unsupported pooling mode: {self.pool}")

        return (context, attn_weights) if return_attn else (context, None)

    # ---------------- Attention ----------------
    def _attention_context(self, theta: torch.Tensor, mask: Optional[torch.Tensor], return_attn: bool = False):
        B, T, F = theta.shape
        if self._attn_vector.numel() != F:
            self._attn_vector = nn.Parameter(torch.empty(F, device=self.device, dtype=DTYPE))
            nn.init.normal_(self._attn_vector, mean=0.0, std=0.1)
            self._dbg(f"Initialized _attn_vector (dim={F})")

        # Vectorized over batch
        attn_scores = theta @ self._attn_vector  # [B,T]
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, float("-inf"))
            zero_mask = (mask.sum(dim=1) == 0)
            if zero_mask.any():
                attn_scores[zero_mask] = 0.0

        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)  # [B,T,1]
        context = (attn_weights * theta).sum(dim=1)  # [B,F]
        return (context, attn_weights) if return_attn else (context, None)

    def _multihead_context(self, theta: torch.Tensor, mask: Optional[torch.Tensor], return_attn: bool = False):
        B, T, F = theta.shape
        if self._mha is None:
            self._mha = nn.MultiheadAttention(embed_dim=F, num_heads=self.n_heads, batch_first=True)
            self._dbg(f"Initialized MultiheadAttention(embed_dim={F}, n_heads={self.n_heads})")
        attn_mask = (~mask.bool()) if mask is not None else None
        attn_output, attn_weights = self._mha(theta, theta, theta, key_padding_mask=attn_mask)
        context = attn_output.mean(dim=1)
        return (context, attn_weights) if return_attn else (context, None)

    # ---------------- Context Utilities ----------------
    def set_context(self, context: Optional[torch.Tensor], detach: bool = True):
        if context is None:
            self._context = None
            return
        ctx = context.to(self.device, DTYPE)
        if detach:
            ctx = ctx.detach()
        if ctx.ndim == 1:
            ctx = ctx.unsqueeze(0)
        self._context = ctx

    def reset_context(self):
        self._context = None

    def get_context(self, expand_to_states: Optional[int] = None) -> Optional[torch.Tensor]:
        ctx = self._context
        if ctx is not None and expand_to_states is not None:
            ctx = ctx.unsqueeze(1).expand(-1, expand_to_states, -1)  # [B, n_states, F]
        return ctx
