import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal, Union

from nhsmm.defaults import DTYPE, logger


class ContextEncoder(nn.Module):
    """Sequence encoder wrapper for HSMM/HMM/CRF with pooling, attention, dropout, layer norm, and cached context."""

    def __init__(
        self,
        encoder: nn.Module,
        n_heads: int = 4,
        pool: Literal["mean", "last", "max", "attn", "mha"] = "mean",
        device: Optional[torch.device] = None,
        layer_norm: bool = True,
        dropout: float = 0.0,
        debug: bool = False
    ):
        super().__init__()
        self.encoder = encoder
        self.pool = pool.lower()
        self.device = device or next(encoder.parameters(), torch.tensor(0.0)).device
        self.layer_norm_flag = layer_norm
        self.n_heads = n_heads
        self.debug = debug
        self._context: Optional[torch.Tensor] = None

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self._attn_vector: Optional[nn.Parameter] = None
        self._mha: Optional[nn.MultiheadAttention] = None

        self.encoder.to(device=self.device, dtype=DTYPE)
        self._dbg(f"ContextEncoder initialized: pool={self.pool}, device={self.device}")

    # ---------------- Debug ----------------
    def _dbg(self, *args):
        if self.debug:
            logger.debug(*args)

    # ---------------- Forward ----------------
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_context: bool = False,
        return_attn_weights: bool = False
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        # Ensure x is [T, F] -> [1, T, F] for compatibility
        x = x.to(self.device, dtype=DTYPE)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if mask is not None:
            mask = mask.to(self.device, dtype=torch.bool)

        out = self.encoder(x)
        if isinstance(out, (tuple, list)):
            out = out[0]

        theta = out  # [1, T, F]
        if theta.ndim == 1:
            theta = theta.unsqueeze(0).unsqueeze(1)
        elif theta.ndim == 2:
            theta = theta.unsqueeze(0)

        if mask is not None:
            theta = theta * mask.unsqueeze(-1)

        context, attn_weights = self._pool_context(theta, mask, return_attn_weights)
        if self.layer_norm_flag:
            context = F.layer_norm(context, context.shape[-1:])
        context = self.dropout(torch.clamp(context, -10.0, 10.0))
        self._context = context.detach()

        if return_context:
            return (theta, context, attn_weights) if return_attn_weights else (theta, context)
        return theta

    # ---------------- Pooling ----------------
    def _pool_context(self, theta: torch.Tensor, mask: Optional[torch.Tensor], return_attn: bool = False):
        T, F = theta.shape[1], theta.shape[2]
        attn_weights = None

        if T == 0:
            context = torch.zeros((F,), device=self.device, dtype=DTYPE)
        elif self.pool == "last":
            if mask is not None:
                idx = mask.sum().clamp_min(1) - 1
                context = theta[0, idx, :]
            else:
                context = theta[0, -1, :]
        elif self.pool == "mean":
            context = (theta[0] if mask is None else (theta[0] * mask.unsqueeze(-1)).sum(dim=0) / mask.sum().clamp_min(1))
        elif self.pool == "max":
            masked = theta[0] if mask is None else theta[0].masked_fill(~mask.unsqueeze(-1), float('-inf'))
            context = masked.max(dim=0).values
        elif self.pool == "attn":
            context, attn_weights = self._attention_context(theta[0], mask, True)
        elif self.pool == "mha":
            context, attn_weights = self._multihead_context(theta[0].unsqueeze(0), mask, True)
        else:
            raise ValueError(f"Unsupported pooling mode: {self.pool}")

        return (context, attn_weights) if return_attn else (context, None)

    # ---------------- Attention ----------------
    def _attention_context(self, theta: torch.Tensor, mask: Optional[torch.Tensor], return_attn: bool = False):
        F = theta.shape[-1]
        if self._attn_vector is None or self._attn_vector.shape[-1] != F:
            self._attn_vector = nn.Parameter(torch.randn(F, device=self.device, dtype=DTYPE))
            self._dbg(f"Initialized _attn_vector with dim {F}")

        attn_scores = theta @ self._attn_vector
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=0).unsqueeze(-1)
        context = (attn_weights * theta).sum(dim=0)
        return (context, attn_weights) if return_attn else context

    def _multihead_context(self, theta: torch.Tensor, mask: Optional[torch.Tensor], return_attn: bool = False):
        _, T, F = theta.shape
        if self._mha is None:
            self._mha = nn.MultiheadAttention(embed_dim=F, num_heads=self.n_heads, batch_first=True)
            self._dbg(f"Initialized MultiheadAttention with embed_dim={F}, n_heads={self.n_heads}")
        attn_mask = ~mask if mask is not None else None
        attn_output, attn_weights = self._mha(theta, theta, theta, key_padding_mask=attn_mask)
        context = attn_output.mean(dim=1).squeeze(0)
        return (context, attn_weights) if return_attn else context

    # ---------------- Context Utilities ----------------
    def set_context(self, context: Optional[torch.Tensor]):
        if context is None:
            self._context = None
            return
        ctx = context.detach().to(self.device, DTYPE)
        if ctx.ndim == 1:
            ctx = ctx.unsqueeze(0)
        self._context = ctx

    def reset_context(self):
        self._context = None

    def get_context(self) -> Optional[torch.Tensor]:
        return self._context
