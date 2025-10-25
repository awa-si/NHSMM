# nhsmm/context.py
import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
from typing import Optional, Literal, Union

from nhsmm.defaults import DTYPE, EPS

logger = logging.getLogger(__name__)

class ContextEncoder(nn.Module):
    """Sequence encoder wrapper for HSMM/HMM/CRF with pooling, attention, dropout, layer norm, and cached context."""

    def __init__(
        self,
        encoder: nn.Module,
        batch_first: bool = True,
        pool: Literal["mean", "last", "max", "attn", "mha"] = "mean",
        device: Optional[torch.device] = None,
        layer_norm: bool = True,
        max_seq_len: int = 1024,
        dropout: float = 0.0,
        n_heads: int = 4,
        debug: bool = False
    ):
        super().__init__()
        self.encoder = encoder
        self.batch_first = batch_first
        self.pool = pool.lower()
        self.device = device or next(encoder.parameters(), torch.tensor(0.0)).device
        self.layer_norm_flag = layer_norm
        self.max_seq_len = max_seq_len
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

    # ---------------- Masking ----------------
    def _make_mask(self, x: torch.Tensor, lengths: Optional[torch.Tensor], pad_value: float = 0.0) -> Optional[torch.Tensor]:
        if lengths is None:
            return (x.sum(dim=-1) != pad_value) if x.ndim >= 3 else None
        max_T = x.shape[1] if self.batch_first else x.shape[0]
        idx = torch.arange(max_T, device=self.device).unsqueeze(0)
        mask = idx < lengths.view(-1, 1)
        return mask

    # ---------------- Forward ----------------
    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_context: bool = False,
        return_attn_weights: bool = False
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        x = x.to(self.device, dtype=DTYPE)
        if x.ndim == 2:
            x = x.unsqueeze(0 if self.batch_first else 1)
        if not self.batch_first:
            x = x.transpose(0, 1)

        if mask is None and lengths is not None:
            mask = self._make_mask(x, lengths)
        mask = mask.to(self.device, dtype=torch.bool) if mask is not None else None

        out = self.encoder(x)
        if isinstance(out, (tuple, list)):
            out = out[0]

        # Ensure theta has shape [B, T, F]
        theta = out
        if out.ndim == 1:
            theta = out.unsqueeze(0).unsqueeze(1)
        elif out.ndim == 2:
            theta = out.unsqueeze(1) if out.shape[0] == x.shape[0] else out.unsqueeze(0)

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
                idx = mask.sum(dim=1).clamp_min(1) - 1
                context = theta.gather(1, idx.unsqueeze(-1).expand(-1, -1, F)).squeeze(1)
            else:
                context = theta[:, -1]
        elif self.pool == "mean":
            context = ((theta * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
                       if mask is not None else theta.mean(dim=1))
        elif self.pool == "max":
            masked = theta.masked_fill(~mask.unsqueeze(-1), float('-inf')) if mask is not None else theta
            context = masked.max(dim=1).values
        elif self.pool == "attn":
            context, attn_weights = self._attention_context(theta, mask, True)
        elif self.pool == "mha":
            context, attn_weights = self._multihead_context(theta, mask, True)
        else:
            raise ValueError(f"Unsupported pooling mode: {self.pool}")

        return (context, attn_weights) if return_attn else (context, None)

    # ---------------- Attention ----------------
    def _attention_context(self, theta: torch.Tensor, mask: Optional[torch.Tensor], return_attn: bool = False):
        B, T, F = theta.shape
        if self._attn_vector is None or self._attn_vector.shape[-1] != F:
            self._attn_vector = nn.Parameter(torch.randn(F, device=self.device, dtype=DTYPE))
            self._dbg(f"Initialized _attn_vector with dim {F}")

        attn_scores = theta @ self._attn_vector
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        context = (attn_weights * theta).sum(dim=1)
        return (context, attn_weights) if return_attn else context

    def _multihead_context(self, theta: torch.Tensor, mask: Optional[torch.Tensor], return_attn: bool = False):
        B, T, F = theta.shape
        if self._mha is None:
            self._mha = nn.MultiheadAttention(embed_dim=F, num_heads=self.n_heads,
                                              batch_first=True)
            self._dbg(f"Initialized MultiheadAttention with embed_dim={F}, n_heads={self.n_heads}")
        attn_mask = ~mask if mask is not None else None
        attn_output, attn_weights = self._mha(theta, theta, theta, key_padding_mask=attn_mask)
        context = attn_output.mean(dim=1)
        return (context, attn_weights) if return_attn else context

    # ---------------- Context Utilities ----------------
    def set_context(self, context: Optional[torch.Tensor], batch_size: Optional[int] = None):
        if context is None:
            self._context = None
            return
        ctx = context.detach().to(self.device, DTYPE)
        if ctx.ndim == 1:
            ctx = ctx.unsqueeze(0)
        if batch_size is not None and ctx.shape[0] == 1:
            ctx = ctx.expand(batch_size, -1)
        self._context = ctx

    def reset_context(self): self._context = None
    def get_context(self) -> Optional[torch.Tensor]: return self._context

    def _combine_context(self, theta: torch.Tensor, allow_broadcast: bool = True) -> torch.Tensor:
        ctx = self._context
        if theta is None: return ctx
        if ctx is None: return theta
        theta, ctx = theta.to(self.device, DTYPE), ctx.to(self.device, DTYPE)
        B_theta, B_ctx = theta.shape[0], ctx.shape[0]
        if B_theta != B_ctx:
            if allow_broadcast:
                if B_theta == 1: theta = theta.expand(B_ctx, *theta.shape[1:])
                elif B_ctx == 1: ctx = ctx.expand(B_theta, *ctx.shape[1:])
                else: raise ValueError(f"Batch mismatch: theta {theta.shape} vs ctx {ctx.shape}")
            else:
                raise ValueError(f"Batch mismatch and broadcasting disabled: theta {theta.shape} vs ctx {ctx.shape}")
        if theta.shape[1:-1] != ctx.shape[1:-1]:
            raise ValueError(f"Sequence dimension mismatch: theta {theta.shape} vs ctx {ctx.shape}")
        return torch.cat([theta, ctx.expand(theta.shape[0], *ctx.shape[1:])], dim=-1)

    def prepare_delta(self, delta: Optional[torch.Tensor], n_states:int, feature_dim:int, scale: float = 0.1, allow_broadcast: bool = True) -> torch.Tensor:
        B_target = self._context.shape[0] if self._context is not None else 1
        device, dtype = self.device, DTYPE
        if delta is None:
            return torch.zeros((B_target, n_states, feature_dim), dtype=dtype, device=device)
        delta = delta.to(device, dtype)
        if delta.ndim > 2:
            delta = delta.view(delta.shape[0], -1)
        B, total_features = delta.shape
        expected_features = n_states * feature_dim
        if total_features < expected_features: delta = F.pad(delta, (0, expected_features - total_features))
        elif total_features > expected_features: delta = delta[..., :expected_features]
        delta = delta.view(B, n_states, feature_dim)
        delta = scale * torch.tanh(torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0))
        if allow_broadcast and B != B_target:
            delta = delta.expand(B_target, -1, -1) if B == 1 else delta[:B_target]
        return delta

    def get_config(self) -> dict:
        return dict(
            pool=self.pool,
            n_heads=self.n_heads,
            max_seq_len=self.max_seq_len,
            dropout=self.dropout.p if isinstance(self.dropout, nn.Dropout) else 0.0,
            batch_first=self.batch_first,
            device=str(self.device)
        )
