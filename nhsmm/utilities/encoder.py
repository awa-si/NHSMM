import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Literal, Tuple

from nhsmm.defaults import DTYPE


class ContextEncoder(nn.Module):
    """
    Encoder wrapper for sequence models (HSMM/HMM/CRF/etc.).
    Features:
      - Masking for variable-length sequences
      - Pooling: mean, last, max, attention, multihead attention
      - Internal cached context (_context)
      - Safe preparation and combination of modulation tensors
    """

    def __init__(
        self,
        encoder: nn.Module,
        batch_first: bool = True,
        pool: Literal["mean", "last", "max", "attn", "mha"] = "mean",
        device: Optional[torch.device] = None,
        n_heads: int = 4,
        max_seq_len: int = 1024
    ):
        super().__init__()
        self.encoder = encoder
        self.batch_first = batch_first
        self.pool = pool.lower().strip()
        self.device = device or next(encoder.parameters(), torch.tensor(0.0)).device
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self._context: Optional[torch.Tensor] = None

        # Attention parameters
        self._attn_vector: Optional[nn.Parameter] = None
        self._mha_qkv: Optional[nn.Linear] = None
        self._mha_rel_pos: Optional[nn.Parameter] = None

        self.encoder.to(device=self.device, dtype=DTYPE)

    def _make_mask(self, x: torch.Tensor, lengths: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if lengths is None:
            return None
        lengths = lengths.view(-1)
        max_T = x.shape[1] if self.batch_first else x.shape[0]
        idx = torch.arange(max_T, device=self.device).unsqueeze(0)
        return idx < lengths.unsqueeze(1)

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_context: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = x.to(self.device, DTYPE)
        if x.ndim == 2:
            x = x.unsqueeze(0 if self.batch_first else 1)
        elif x.ndim != 3:
            raise ValueError(f"Unsupported input shape {tuple(x.shape)}, expected (T,F) or (B,T,F)")

        if not self.batch_first:
            x = x.transpose(0, 1)

        if mask is None and lengths is not None:
            mask = self._make_mask(x, lengths)
        if mask is not None:
            mask = mask.to(self.device, dtype=torch.bool)

        out = self.encoder(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if not torch.is_tensor(out):
            raise TypeError(f"Encoder must return a tensor, got {type(out)}")

        # Ensure (B,T,F)
        if out.ndim == 1:
            theta = out.unsqueeze(0).unsqueeze(1)
        elif out.ndim == 2:
            theta = out.unsqueeze(1) if out.shape[0] == x.shape[0] else out.unsqueeze(0)
        else:
            theta = out

        if mask is not None:
            theta = theta * mask.unsqueeze(-1)

        context = self._pool_context(theta, mask)
        context = F.layer_norm(context, context.shape[-1:])
        context = torch.clamp(context, -10.0, 10.0)
        self._context = context.detach()

        return (theta, context) if return_context else theta

    def _pool_context(self, theta: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if self.pool == "last":
            if mask is not None:
                idx = mask.sum(dim=1).clamp_min(1) - 1
                return theta.gather(1, idx.unsqueeze(-1).expand(-1, -1, theta.shape[-1])).squeeze(1)
            return theta[:, -1]
        elif self.pool == "mean":
            if mask is not None:
                return (theta * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
            return theta.mean(dim=1)
        elif self.pool == "max":
            masked = theta.masked_fill(~mask.unsqueeze(-1), float('-inf')) if mask is not None else theta
            return masked.max(dim=1).values
        elif self.pool == "attn":
            return self._attention_context(theta, mask)
        elif self.pool == "mha":
            return self._multihead_context(theta, mask)
        else:
            raise ValueError(f"Unsupported pooling mode: {self.pool}")

    def _attention_context(self, theta: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        hidden = theta.shape[-1]
        if self._attn_vector is None or self._attn_vector.shape[-1] != hidden:
            self._attn_vector = nn.Parameter(torch.randn(hidden, device=self.device, dtype=DTYPE))
        attn_scores = theta @ self._attn_vector
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        return (attn_weights * theta).sum(dim=1)

    def _multihead_context(self, theta: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        B, T, d_model = theta.shape
        n_heads = self.n_heads
        head_dim = d_model // n_heads
        if d_model % n_heads != 0:
            raise ValueError(f"Embedding dim {d_model} not divisible by n_heads {n_heads}")

        if self._mha_qkv is None:
            self._mha_qkv = nn.Linear(d_model, 3 * d_model, device=self.device, dtype=DTYPE)
        if self._mha_rel_pos is None or self._mha_rel_pos.shape[0] < 2*T - 1:
            self._mha_rel_pos = nn.Parameter(torch.zeros(2 * self.max_seq_len - 1, n_heads, device=self.device, dtype=DTYPE))

        qkv = self._mha_qkv(theta)
        Q, K, V = qkv.chunk(3, dim=-1)
        Q, K, V = [t.view(B, T, n_heads, head_dim).transpose(1, 2) for t in (Q, K, V)]

        attn_scores = (Q @ K.transpose(-2, -1)) / (head_dim ** 0.5)
        rel_idx = torch.arange(T, device=self.device)[:, None] - torch.arange(T, device=self.device)[None, :] + self.max_seq_len - 1
        attn_scores += self._mha_rel_pos[rel_idx].permute(2,0,1).unsqueeze(0)

        if mask is not None:
            mask_ = mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(~mask_, float('-inf'))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        context = (attn_weights @ V).transpose(1, 2).reshape(B, T, d_model)

        if mask is not None:
            return (context * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
        return context.mean(dim=1)

    # ---------------- Context Utilities ----------------

    def set_context(self, context: Optional[torch.Tensor], batch_size: Optional[int] = None):
        """Store a (B,H) or (H,) context tensor on model device/dtype."""
        if context is None:
            self._context = None
            return
        ctx = context.detach().to(device=self.device, dtype=DTYPE)
        if ctx.ndim == 1:
            ctx = ctx.unsqueeze(0)
        if batch_size is not None and ctx.shape[0] == 1:
            ctx = ctx.expand(batch_size, -1)
        self._context = ctx

    def _prepare_delta(
        self,
        delta: Optional[torch.Tensor],
        shape: Tuple[int, int, int],
        scale: float = 1.0,
        broadcast: bool = True
    ) -> torch.Tensor:
        """Reshape, pad/truncate, scale, and optionally broadcast delta tensor."""
        B_target, n_states, feat_dim = shape
        if delta is None:
            return torch.zeros((B_target, n_states, feat_dim), device=self.device, dtype=DTYPE)

        delta = delta.to(self.device, dtype=DTYPE)
        if delta.ndim == 1:
            delta = delta.unsqueeze(0)
        B, L = delta.shape[0], delta.shape[1]
        expected = n_states * feat_dim
        if L < expected:
            delta = F.pad(delta, (0, expected - L))
        elif L > expected:
            delta = delta[..., :expected]
        delta = delta.view(B, n_states, feat_dim)
        delta = scale * torch.tanh(delta)
        delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
        if broadcast and B == 1 and B_target > 1:
            delta = delta.expand(B_target, -1, -1)
        return delta

    def _combine_context(
        self,
        theta: Optional[torch.Tensor],
        allow_broadcast: bool = True
    ) -> Optional[torch.Tensor]:
        """Combine encoder output theta with cached context safely."""
        theta = theta.detach().to(self.device, DTYPE) if theta is not None else None
        ctx = self._context.detach().to(self.device, DTYPE) if self._context is not None else None

        if theta is None and ctx is None:
            return None
        if theta is None:
            return ctx
        if ctx is None:
            return theta

        if theta.ndim == 1:
            theta = theta.unsqueeze(0)
        if ctx.ndim == 1:
            ctx = ctx.unsqueeze(0)

        if theta.shape[0] != ctx.shape[0]:
            if allow_broadcast:
                if theta.shape[0] == 1:
                    theta = theta.expand(ctx.shape[0], *theta.shape[1:])
                elif ctx.shape[0] == 1:
                    ctx = ctx.expand(theta.shape[0], *ctx.shape[1:])
                else:
                    raise ValueError(f"Batch mismatch: theta {theta.shape} vs context {ctx.shape}")
            else:
                raise ValueError(f"Batch mismatch and broadcasting disabled: theta {theta.shape} vs context {ctx.shape}")

        if theta.shape[1:] != ctx.shape[1:]:
            raise ValueError(f"Feature dimension mismatch: theta {theta.shape} vs ctx {ctx.shape}")

        return torch.cat([theta, ctx], dim=-1)
