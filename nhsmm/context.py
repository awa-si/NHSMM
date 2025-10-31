import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal, Union, Tuple

from nhsmm.constants import DTYPE, logger


class ContextEncoder(nn.Module):
    """
    Sequence encoder wrapper for HSMM/HMM/CRF with pooling, attention, dropout, layer norm, and cached context.

    Args:
        encoder: nn.Module that outputs sequences [batch, T, F] or [T, F].
        n_heads: Number of heads for multihead attention pooling.
        pool: Pooling method: "mean", "last", "max", "attn", "mha".
        device: torch.device to place module and buffers.
        layer_norm: Apply layer normalization to pooled context.
        dropout: Dropout rate applied to pooled context.
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

        # Attention parameters
        self._attn_vector = nn.Parameter(torch.empty(0, device=self.device, dtype=DTYPE))
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
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        x = x.to(self.device, dtype=DTYPE)
        if x.ndim == 2:
            x = x.unsqueeze(0)  # [1, T, F]
        batch_size, T, F = x.shape

        if mask is not None:
            mask = mask.to(self.device, dtype=torch.bool)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0).expand(batch_size, -1)

        out = self.encoder(x)
        if isinstance(out, (tuple, list)):
            out = out[0]  # assume first element is sequence

        theta = out  # [B, T, F]

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
        B, T, F = theta.shape
        attn_weights = None

        if T == 0:
            context = torch.zeros((B, F), device=self.device, dtype=DTYPE)
        elif self.pool == "last":
            if mask is not None:
                lengths = mask.sum(dim=1).clamp_min(1)
                idx = (lengths - 1).unsqueeze(-1).expand(-1, F)
                context = theta.gather(1, idx.unsqueeze(1)).squeeze(1)
            else:
                context = theta[:, -1, :]
        elif self.pool == "mean":
            if mask is not None:
                context = (theta * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
            else:
                context = theta.mean(dim=1)
        elif self.pool == "max":
            if mask is not None:
                masked = theta.masked_fill(~mask.unsqueeze(-1), float('-inf'))
                context = masked.max(dim=1).values
            else:
                context = theta.max(dim=1).values
        elif self.pool == "attn":
            context_list = []
            attn_list = []
            for b in range(B):
                ctx, attn = self._attention_context(theta[b], mask[b] if mask is not None else None, True)
                context_list.append(ctx)
                attn_list.append(attn)
            context = torch.stack(context_list)
            attn_weights = torch.stack(attn_list) if return_attn else None
        elif self.pool == "mha":
            context, attn_weights = self._multihead_context(theta, mask, True)
        else:
            raise ValueError(f"Unsupported pooling mode: {self.pool}")

        return (context, attn_weights) if return_attn else (context, None)

    # ---------------- Attention ----------------
    def _attention_context(self, theta: torch.Tensor, mask: Optional[torch.Tensor], return_attn: bool = False):
        T, F = theta.shape
        if self._attn_vector.numel() != F:
            nn.init.normal_(self._attn_vector, mean=0.0, std=0.1)
            self._attn_vector = nn.Parameter(torch.randn(F, device=self.device, dtype=DTYPE))
            self._dbg(f"Initialized _attn_vector with dim {F}")

        attn_scores = theta @ self._attn_vector
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=0).unsqueeze(-1)
        context = (attn_weights * theta).sum(dim=0)
        return (context, attn_weights) if return_attn else context

    def _multihead_context(self, theta: torch.Tensor, mask: Optional[torch.Tensor], return_attn: bool = False):
        B, T, F = theta.shape
        if self._mha is None:
            self._mha = nn.MultiheadAttention(embed_dim=F, num_heads=self.n_heads, batch_first=True)
            self._dbg(f"Initialized MultiheadAttention with embed_dim={F}, n_heads={self.n_heads}")
        attn_mask = (~mask) if mask is not None else None
        attn_output, attn_weights = self._mha(theta, theta, theta, key_padding_mask=attn_mask)
        context = attn_output.mean(dim=1)
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
