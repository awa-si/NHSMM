# nhsmm/context.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal, Tuple, Callable, Dict

from nhsmm.constants import DTYPE, logger

# ----------------------------
# CNN + LSTM Encoder
# ----------------------------
class CNN_LSTM_Encoder(nn.Module):
    """
    CNN + LSTM encoder producing per-timestep features and a pooled context.

    Input:
        x: [B, T, F]
        mask: optional [B, T] (1 for valid, 0 for padding)

    Output:
        - sequence [B, T, D] if return_sequence=True
        - last valid timestep [B, D] if return_sequence=False
        - self._context: pooled [B, 1, D] always available after forward
    """

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 32,
        cnn_channels: int = 16,
        kernel_size: int = 3,
        dropout: float = 0.1,
        bidirectional: bool = True,
        return_sequence: bool = True,
        use_packed: bool = True,
    ):
        super().__init__()
        self.use_packed = bool(use_packed)
        self.return_sequence = bool(return_sequence)
        self._context: Optional[torch.Tensor] = None

        # store meta
        self.n_features = int(n_features)
        self.cnn_channels = int(cnn_channels)
        self.kernel_size = int(kernel_size)
        self.padding = self.kernel_size // 2
        self.bidirectional = bool(bidirectional)

        # CNN layer (expects [B, F, T] input)
        self.conv = nn.Conv1d(self.n_features, self.cnn_channels, self.kernel_size, padding=self.padding)
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

        # layer norm over channel dim after transpose -> [B,T,C]
        self.norm = nn.LayerNorm(self.cnn_channels, elementwise_affine=True)

        # LSTM (input_size = cnn_channels)
        self.lstm = nn.LSTM(
            input_size=self.cnn_channels,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=self.bidirectional,
        )

        self.dropout = nn.Dropout(dropout)
        self.out_dim = hidden_dim * (2 if self.bidirectional else 1)

    @property
    def device(self) -> torch.device:
        # convenient device accessor
        return next(self.parameters()).device if any(p.numel() for p in self.parameters()) else torch.device("cpu")

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, return_sequence: Optional[bool] = None):
        """
        x: [B, T, F] or [T, F] or [F] (we canonicalize)
        mask: None or [B, T] or [T] (1/0)
        """
        if x.ndim == 1:
            x = x.unsqueeze(0).unsqueeze(0)  # [1,1,F]
        elif x.ndim == 2:
            x = x.unsqueeze(0)  # [1, T, F]

        B, T, F_in = x.shape

        if F_in != self.n_features:
            raise ValueError(f"Input feature dim {F_in} != encoder.n_features {self.n_features}")

        if T == 0:
            raise ValueError("Input sequence has zero length")

        # canonicalize mask
        if mask is not None:
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
            if mask.shape[0] != B:
                # allow single mask expanded to batch
                if mask.shape[0] == 1:
                    mask = mask.expand(B, -1)
                else:
                    raise ValueError("Mask batch-size mismatch")

            mask = mask[:, :T].to(dtype=torch.bool, device=x.device)

        # --- CNN ---
        # conv expects [B, F, T]
        x_c = x.transpose(1, 2)  # [B, F, T]
        x_c = F.relu(self.conv(x_c))  # [B, C, T]
        x_c = x_c.transpose(1, 2)  # [B, T, C]
        x_c = self.norm(x_c)
        x_c = self.dropout(x_c)

        # --- LSTM ---
        use_packed = bool(self.use_packed and (mask is not None))
        if use_packed:
            # lengths must be CPU int list for pack_padded_sequence
            lengths = mask.sum(dim=1).clamp_min(1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(x_c, lengths, batch_first=True, enforce_sorted=False)
            out_packed, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True, total_length=T)
        else:
            out, _ = self.lstm(x_c)

        out = self.dropout(out)  # [B, T, D]

        # --- pooled context (mean over valid timesteps) ---
        if mask is not None:
            mask_f = mask.unsqueeze(-1).to(dtype=out.dtype)
            denom = mask_f.sum(dim=1).clamp_min(1.0)
            pooled = (out * mask_f).sum(dim=1) / denom
        else:
            pooled = out.mean(dim=1)

        # store canonical context as [B,1,D]
        self._context = pooled.unsqueeze(1)

        # decide return shape
        ret_sequence = return_sequence if return_sequence is not None else self.return_sequence
        if ret_sequence:
            return out  # [B, T, D]
        else:
            if mask is not None:
                idx = mask.sum(dim=1).clamp_min(1) - 1  # last valid index per batch
                return out[torch.arange(B, device=out.device), idx]  # [B, D]
            return out[:, -1, :]  # [B, D]


# ----------------------------
# ContextEncoder (wrapper that pools / attends)
# ----------------------------
class ContextEncoder(nn.Module):
    """
    Batch-native context encoder wrapper.

    - Accepts an `encoder` module that maps (B,T,F_in) -> (B,T,F_out)
    - Supports pooling methods: mean, last, max, attn, mha
    - Returns (sequence, context, attn) where context is [B,1,F_out]
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
        self.n_heads = int(n_heads)
        self.layer_norm = bool(layer_norm)
        self.context_scale = float(context_scale)
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.debug = bool(debug)

        self._context: Optional[torch.Tensor] = None
        self._sequence: Optional[torch.Tensor] = None

        # attention params set lazily on first use
        self._attn_vector: Optional[nn.Parameter] = None
        self._mha: Optional[nn.MultiheadAttention] = None

        # pooling registry for easy extension
        self._POOLERS: Dict[str, Callable] = {
            "mean": self._pool_mean,
            "last": self._pool_last,
            "max": self._pool_max,
            "attn": self._attention_context,
            "mha": self._multihead_context,
        }

        if self.debug and logger:
            logger.debug(f"[ContextEncoder] Initialized with pool={self.pool}")

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device if any(p.numel() for p in self.parameters()) else torch.device("cpu")

    @property
    def context_dim(self) -> Optional[int]:
        # try to infer output dim from encoder
        out_dim = getattr(self.encoder, "out_dim", None)
        return out_dim

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

        # call encoder. encoder must accept mask kwarg (many do)
        out = self.encoder(x, mask=mask) if "mask" in self._encoder_signature() else self.encoder(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.ndim != 3:
            raise ValueError(f"Encoder returned {out.shape}, expected [B,T,F]")

        # apply mask to sequence (zero-out padded timesteps)
        theta = out * mask.unsqueeze(-1).to(dtype=out.dtype, device=out.device) if mask is not None else out

        # store sequence (detach optional)
        self._sequence = theta.detach() if detach_context else theta

        # pooling
        pooled, attn = self._pool_context(theta, mask, return_attn_weights)

        # layer-norm + scaled tanh (apply scale first to avoid saturation)
        if self.layer_norm:
            pooled = F.layer_norm(pooled, (pooled.shape[-1],))
        pooled = self.dropout_layer(torch.tanh(pooled * self.context_scale))
        ctx = pooled.unsqueeze(1)  # [B,1,F]
        self._context = ctx.detach() if detach_context else ctx

        # decide sequence return
        seq_out = theta if return_sequence else self._last_timestep(theta, mask)

        return seq_out, (ctx if return_context else None), (attn if return_attn_weights else None)

    # ---------------- Utilities ----------------
    def _encoder_signature(self) -> Tuple[str, ...]:
        # cheap introspection: detect if encoder accepts "mask" kw
        try:
            return tuple(p.name for p in self.encoder.forward.__code__.co_varnames[: self.encoder.forward.__code__.co_argcount])
        except Exception:
            return tuple()

    def _prepare_mask(self, mask: Optional[torch.BoolTensor], B: int, T: int) -> Optional[torch.BoolTensor]:
        if mask is None:
            return None
        mask = mask.bool().to(device=self.device)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0).expand(B, -1)
        return mask[:, :T]

    def _last_timestep(self, theta: torch.Tensor, mask: Optional[torch.BoolTensor]) -> torch.Tensor:
        B, T, F = theta.shape
        if mask is not None:
            lengths = mask.sum(dim=1).clamp_min(1)
            idx = lengths - 1
            return theta[torch.arange(B, device=theta.device), idx]
        return theta[:, -1, :]

    # ---------------- Pool dispatchers ----------------
    def _pool_context(self, theta: torch.Tensor, mask: Optional[torch.BoolTensor], ret_attn: bool):
        if self.pool not in self._POOLERS:
            raise ValueError(f"Invalid pooling method '{self.pool}'")
        return self._POOLERS[self.pool](theta, mask, ret_attn)

    def _pool_last(self, theta: torch.Tensor, mask: Optional[torch.BoolTensor], ret_attn: bool):
        return self._last_timestep(theta, mask), None

    def _pool_mean(self, theta: torch.Tensor, mask: Optional[torch.BoolTensor], ret_attn: bool):
        B, T, F = theta.shape
        if T == 0:
            return torch.zeros((B, F), dtype=theta.dtype, device=theta.device), None
        if mask is not None:
            denom = mask.sum(dim=1).clamp_min(1).unsqueeze(-1).to(dtype=theta.dtype)
            ctx = (theta * mask.unsqueeze(-1).to(dtype=theta.dtype)).sum(dim=1) / denom
        else:
            ctx = theta.mean(dim=1)
        return ctx, None

    def _pool_max(self, theta: torch.Tensor, mask: Optional[torch.BoolTensor], ret_attn: bool):
        B, T, F = theta.shape
        if T == 0:
            return torch.zeros((B, F), dtype=theta.dtype, device=theta.device), None
        if mask is not None:
            masked = theta.masked_fill(~mask.unsqueeze(-1), float("-inf"))
            ctx = masked.max(dim=1).values
            ctx = torch.where(torch.isfinite(ctx), ctx, torch.zeros_like(ctx))
        else:
            ctx = theta.max(dim=1).values
        return ctx, None

    # ---------------- Attention (single vector) ----------------
    def _init_attn_vector(self, F: int, device: torch.device):
        if self._attn_vector is None or self._attn_vector.shape[0] != F:
            v = torch.randn(F, dtype=DTYPE, device=device) * 0.1
            self._attn_vector = nn.Parameter(v)

    def _attention_context(self, theta: torch.Tensor, mask: Optional[torch.BoolTensor], ret_attn: bool):
        B, T, F = theta.shape
        if T == 0:
            return torch.zeros((B, F), dtype=theta.dtype, device=theta.device), None
        self._init_attn_vector(F, theta.device)

        scores = torch.einsum("btf,f->bt", theta, self._attn_vector)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
            empty_rows = mask.sum(dim=1) == 0
            if empty_rows.any():
                scores[empty_rows, :] = 0.0
        attn_w = torch.nn.functional.softmax(scores, dim=1).unsqueeze(-1)  # [B, T, 1]
        ctx = (attn_w * theta).sum(dim=1)
        return ctx, (attn_w if ret_attn else None)

    # ---------------- Multi-head attention ----------------
    def _multihead_context(self, theta: torch.Tensor, mask: Optional[torch.BoolTensor], ret_attn: bool):
        B, T, F = theta.shape
        if T == 0:
            return torch.zeros((B, F), dtype=theta.dtype, device=theta.device), None

        if self._mha is None:
            # Create MHA with dropout consistent with dropout_layer (if any)
            mha_dropout = self.dropout_layer.p if isinstance(self.dropout_layer, nn.Dropout) else 0.0
            self._mha = nn.MultiheadAttention(embed_dim=F, num_heads=self.n_heads, batch_first=True, dropout=mha_dropout, device=theta.device)

        # key_padding_mask expects bool mask where True indicates padding positions
        key_padding_mask = (~mask.bool()) if mask is not None else None
        out, attn = self._mha(theta, theta, theta, key_padding_mask=key_padding_mask)
        ctx = out.mean(dim=1)
        return ctx, (attn if ret_attn else None)

    # ---------------- Accessors / Mutators ----------------
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
            pooled = self.dropout_layer(torch.tanh(pooled * self.context_scale))
            self._context = pooled.unsqueeze(1)

    def set_context(self, context: torch.Tensor, detach: bool = True):
        if context.ndim == 2:
            context = context.unsqueeze(1)
        self._context = context.detach() if detach else context

    def reset(self):
        self._sequence = None
        self._context = None
        self._attn_vector = None
        self._mha = None
