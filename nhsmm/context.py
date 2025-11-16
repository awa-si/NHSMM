import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal, Tuple
from nhsmm.constants import DTYPE, logger


class ContextEncoder(nn.Module):
    """
    Wrapper for sequence encoders (CNN+LSTM, Transformer, etc.) that
    produces sequence features, pooled context, and optional attention maps.

    Handles variable-length sequences safely and supports multiple pooling strategies.
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

        # Device
        has_params = any(p.numel() > 0 for p in encoder.parameters())
        self.device = (
            device
            or (next(encoder.parameters()).device if has_params else torch.device("cpu"))
        )

        # Encoder
        self.encoder = encoder.to(self.device, dtype=DTYPE)

        # Core settings
        self.pool = pool.lower()
        self.n_heads = n_heads
        self.layer_norm_flag = layer_norm
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.context_scale = context_scale
        self.debug = debug

        # State
        self._context: Optional[torch.Tensor] = None

        # Attention modules (lazy init)
        self._attn_vector: Optional[nn.Parameter] = None
        self._mha: Optional[nn.MultiheadAttention] = None

        self._dbg(f"Initialized ContextEncoder(pool={self.pool}, device={self.device})")

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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through encoder and pooling.

        Args:
            x: [B, T, F] input features
            mask: optional [B, T], 1 for valid, 0 for padding
            return_context: whether to return pooled context
            return_attn_weights: whether to return attention weights (if attention pooling)
            detach_context: detach pooled context from gradient
            return_sequence: return full sequence features instead of last timestep

        Returns:
            theta_out: sequence features [B, T, F] or last frame [B, F]
            optional context: [B, F]
            optional attention weights
        """
        x = x.to(self.device, dtype=DTYPE)
        if x.ndim == 2:
            x = x.unsqueeze(0)  # make batch dimension

        B, T, _ = x.shape

        # ----- Mask processing -----
        if mask is not None:
            mask = mask.to(self.device, dtype=torch.bool)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0).expand(B, -1)
            mask = mask[:, :T]

        # ----- Encode -----
        out = self.encoder(x, mask=mask if hasattr(self.encoder, "forward") else None)
        if isinstance(out, (tuple, list)):
            out = out[0]  # handle encoders that return (sequence, extra)

        if out.ndim != 3:
            raise ValueError(f"Encoder must return [B, T, F], got {out.shape}")

        theta = out

        # ----- Apply mask -----
        if mask is not None:
            theta = theta * mask.unsqueeze(-1)

        # ----- Pooling -----
        context, attn = self._pool_context(theta, mask, return_attn=return_attn_weights)

        # ----- Normalize + scale -----
        if self.layer_norm_flag:
            context = F.layer_norm(context, (context.shape[-1],))
        context = self.dropout_layer(torch.tanh(context) * self.context_scale)

        # Store context
        self._context = context.detach() if detach_context else context

        # ----- Return sequence or last -----
        theta_out = theta if return_sequence else theta[:, -1, :]

        if return_context:
            if return_attn_weights:
                return theta_out, context, attn
            return theta_out, context
        return theta_out

    # ---------------- Pooling ----------------
    def _pool_context(self, theta: torch.Tensor, mask: Optional[torch.Tensor], return_attn: bool = False):
        B, T, F_dim = theta.shape
        attn = None

        if T == 0:
            ctx = torch.zeros((B, F_dim), device=self.device, dtype=DTYPE)
            return (ctx, attn) if return_attn else (ctx, None)

        if self.pool == "last":
            if mask is not None:
                lengths = mask.sum(dim=1).clamp_min(1)
                idx = lengths - 1
                ctx = theta[torch.arange(B), idx]
            else:
                ctx = theta[:, -1, :]
        elif self.pool == "mean":
            if mask is not None:
                denom = mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
                ctx = (theta * mask.unsqueeze(-1)).sum(dim=1) / denom
            else:
                ctx = theta.mean(dim=1)
        elif self.pool == "max":
            if mask is not None:
                masked = theta.masked_fill(~mask.unsqueeze(-1), -torch.inf)
                ctx = masked.max(dim=1).values
            else:
                ctx = theta.max(dim=1).values
        elif self.pool == "attn":
            ctx, attn = self._attention_context(theta, mask, return_attn)
        elif self.pool == "mha":
            ctx, attn = self._multihead_context(theta, mask, return_attn)
        else:
            raise ValueError(f"Invalid pooling method '{self.pool}'")

        return ctx, attn if return_attn else (ctx, None)

    # ---------------- Soft Attention ----------------
    def _attention_context(self, theta: torch.Tensor, mask: Optional[torch.Tensor], return_attn: bool):
        B, T, F_dim = theta.shape
        if self._attn_vector is None or self._attn_vector.shape[0] != F_dim:
            self._attn_vector = nn.Parameter(torch.empty(F_dim, device=self.device, dtype=DTYPE))
            nn.init.normal_(self._attn_vector, std=0.1)
            self._dbg(f"Init attn vector (F={F_dim})")

        attn_scores = torch.einsum("btf,f->bt", theta, self._attn_vector)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, -torch.inf)
            attn_scores[mask.sum(dim=1) == 0] = 0.0

        attn_w = F.softmax(attn_scores, dim=1).unsqueeze(-1)
        context = (attn_w * theta).sum(dim=1)
        return (context, attn_w) if return_attn else (context, None)

    # ---------------- Multihead Attention Pool ----------------
    def _multihead_context(self, theta: torch.Tensor, mask: Optional[torch.Tensor], return_attn: bool):
        B, T, F_dim = theta.shape
        if self._mha is None:
            self._mha = nn.MultiheadAttention(embed_dim=F_dim, num_heads=self.n_heads, batch_first=True)
            self._dbg(f"Init MHA(F={F_dim}, heads={self.n_heads})")

        key_padding_mask = (~mask) if mask is not None else None
        out, attn_w = self._mha(theta, theta, theta, key_padding_mask=key_padding_mask)
        context = out.mean(dim=1)
        return (context, attn_w) if return_attn else (context, None)

    # ---------------- Context Utilities ----------------
    def get_context(self, expand_to_states: Optional[int] = None, detach: bool = True):
        ctx = self._context
        if ctx is not None and expand_to_states is not None:
            ctx = ctx.unsqueeze(1).expand(-1, expand_to_states, -1)
        return ctx.detach() if detach and ctx is not None else ctx

    def set_context(self, context: torch.Tensor, detach: bool = True):
        self._context = context.detach() if detach else context

    def reset_context(self):
        self._context = None
