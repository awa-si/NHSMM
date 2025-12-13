# nhsmm/context.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass, field
from typing import Optional, Literal, Tuple, Callable, Dict

from nhsmm.constants import DTYPE, EPS, logger


class CNN_LSTM_Encoder(nn.Module):
    """
    CNN + LSTM encoder producing per-timestep features and a pooled context.

    Input:
        x: [B, T, F] or [T, F] or [F]
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

        # meta
        self.n_features = int(n_features)
        self.cnn_channels = int(cnn_channels)
        self.kernel_size = int(kernel_size)
        self.padding = self.kernel_size // 2
        self.bidirectional = bool(bidirectional)

        # CNN: expects [B, F, T]
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
        return next(self.parameters()).device if any(p.numel() for p in self.parameters()) else torch.device("cpu")

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, return_sequence: Optional[bool] = None):
        # canonicalize input
        if x.ndim == 1:
            x = x.unsqueeze(0).unsqueeze(0)  # [1,1,F]
        elif x.ndim == 2:
            x = x.unsqueeze(0)  # [1,T,F]
        B, T, F_in = x.shape

        if F_in != self.n_features:
            raise ValueError(f"Input feature dim {F_in} != encoder.n_features {self.n_features}")
        if T == 0:
            raise ValueError("Input sequence has zero length")

        # canonicalize mask
        if mask is not None:
            mask = mask.bool().to(device=x.device)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0).expand(B, -1)
            elif mask.shape[0] != B:
                if mask.shape[0] == 1:
                    mask = mask.expand(B, -1)
                else:
                    raise ValueError("Mask batch-size mismatch")
            mask = mask[:, :T]

        # --- CNN ---
        x_c = x.transpose(1, 2)  # [B,F,T]
        x_c = F.relu(self.conv(x_c))
        x_c = x_c.transpose(1, 2)  # [B,T,C]
        x_c = self.norm(x_c)
        x_c = self.dropout(x_c)

        # --- LSTM ---
        use_packed = self.use_packed and (mask is not None)
        if use_packed:
            lengths = mask.sum(dim=1).clamp_min(1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(x_c, lengths, batch_first=True, enforce_sorted=False)
            out_packed, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True, total_length=T)
        else:
            out, _ = self.lstm(x_c)

        out = self.dropout(out)

        # --- pooled context ---
        if mask is not None:
            mask_f = mask.unsqueeze(-1).to(dtype=out.dtype)
            denom = mask_f.sum(dim=1).clamp_min(1.0)
            pooled = (out * mask_f).sum(dim=1) / denom
        else:
            pooled = out.mean(dim=1)

        self._context = pooled.unsqueeze(1)
        ret_sequence = return_sequence if return_sequence is not None else self.return_sequence

        if ret_sequence:
            return out
        else:
            if mask is not None:
                idx = mask.sum(dim=1).clamp_min(1) - 1
                return out[torch.arange(B, device=out.device), idx]
            return out[:, -1, :]


class ContextEncoder(nn.Module):
    """
    Batch-native context encoder wrapper.

    - Accepts any encoder module mapping (B,T,F_in) -> (B,T,F_out)
    - Supports pooling: mean, last, max, attn, mha
    - Returns: (sequence, context, attn) where context=[B,1,F_out]
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
        self.n_heads = n_heads
        self.layer_norm = layer_norm
        self.context_scale = context_scale
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.debug = debug

        self._context: Optional[torch.Tensor] = None
        self._sequence: Optional[torch.Tensor] = None

        self._attn_vector: Optional[nn.Parameter] = None
        self._mha: Optional[nn.MultiheadAttention] = None

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
        return getattr(self.encoder, "out_dim", None)

    @property
    def hidden_dim(self) -> Optional[int]:
        return getattr(self.encoder, "out_dim", None)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
        return_context: bool = False,
        return_attn_weights: bool = False,
        detach_context: bool = True,
        return_sequence: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # canonicalize batch
        if x.ndim == 2:
            x = x.unsqueeze(0)
        B, T, F_in = x.shape
        mask = self._prepare_mask(mask, B, T)

        # call encoder safely
        out = self.encoder(x, mask=mask) if "mask" in self._encoder_signature() else self.encoder(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.ndim != 3:
            raise ValueError(f"Encoder returned {out.shape}, expected [B,T,F]")

        theta = out * mask.unsqueeze(-1).to(dtype=out.dtype, device=out.device) if mask is not None else out
        self._sequence = theta.detach() if detach_context else theta

        pooled, attn = self._pool_context(theta, mask, return_attn_weights)
        if self.layer_norm:
            pooled = F.layer_norm(pooled, (pooled.shape[-1],))
        pooled = self.dropout_layer(torch.tanh(pooled * self.context_scale))
        ctx = pooled.unsqueeze(1)
        self._context = ctx.detach() if detach_context else ctx

        seq_out = theta if return_sequence else self._last_timestep(theta, mask)
        return seq_out, (ctx if return_context else None), (attn if return_attn_weights else None)

    # ---------------- Utilities ----------------
    def _encoder_signature(self) -> Tuple[str, ...]:
        try:
            return tuple(p.name for p in self.encoder.forward.__code__.co_varnames[: self.encoder.forward.__code__.co_argcount])
        except Exception:
            return tuple()

    def _prepare_mask(self, mask: Optional[torch.BoolTensor], B: int, T: int) -> torch.BoolTensor:
        if mask is None:
            return torch.ones(B, T, dtype=torch.bool, device=self.device)
        mask = mask.bool().to(device=self.device)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0).expand(B, -1)
        return mask[:, :T]

    def _last_timestep(self, theta: torch.Tensor, mask: Optional[torch.BoolTensor]) -> torch.Tensor:
        B, T, F = theta.shape
        if mask is not None:
            lengths = mask.sum(dim=1)
            idx = torch.clamp(lengths - 1, min=0)
            return theta[torch.arange(B, device=theta.device), idx]
        return theta[:, -1, :]

    def _pool_context(self, theta: torch.Tensor, mask: Optional[torch.BoolTensor], ret_attn: bool):
        if self.pool not in self._POOLERS:
            raise ValueError(f"Invalid pooling method '{self.pool}'")
        return self._POOLERS[self.pool](theta, mask, ret_attn)

    def _pool_mean(self, theta: torch.Tensor, mask: Optional[torch.BoolTensor], ret_attn: bool):
        B, T, F = theta.shape
        if T == 0:
            return torch.zeros((B, F), device=theta.device, dtype=theta.dtype), None
        if mask is not None:
            denom = mask.sum(dim=1).clamp_min(1).unsqueeze(-1).to(theta.dtype)
            ctx = (theta * mask.unsqueeze(-1).to(theta.dtype)).sum(dim=1) / denom
        else:
            ctx = theta.mean(dim=1)
        return ctx, None

    def _pool_last(self, theta: torch.Tensor, mask: Optional[torch.BoolTensor], ret_attn: bool):
        return self._last_timestep(theta, mask), None

    def _pool_max(self, theta: torch.Tensor, mask: Optional[torch.BoolTensor], ret_attn: bool):
        B, T, F = theta.shape
        if T == 0:
            return torch.zeros((B, F), device=theta.device, dtype=theta.dtype), None
        if mask is not None:
            masked = theta.masked_fill(~mask.unsqueeze(-1), float("-inf"))
            ctx = masked.max(dim=1).values
            ctx = torch.where(torch.isfinite(ctx), ctx, torch.zeros_like(ctx))
        else:
            ctx = theta.max(dim=1).values
        return ctx, None

    def _init_attn_vector(self, F: int):
        if self._attn_vector is None or self._attn_vector.shape[0] != F:
            v = torch.randn(F, dtype=DTYPE, device=self.device) * 0.1
            self._attn_vector = nn.Parameter(v)

    def _attention_context(self, theta: torch.Tensor, mask: Optional[torch.BoolTensor], ret_attn: bool):
        B, T, F = theta.shape
        if T == 0:
            return torch.zeros((B, F), device=theta.device, dtype=theta.dtype), None
        self._init_attn_vector(F)
        scores = torch.einsum("btf,f->bt", theta, self._attn_vector)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
            empty_rows = mask.sum(dim=1) == 0
            if empty_rows.any():
                scores[empty_rows, :] = 0.0
        attn_w = torch.nn.functional.softmax(scores, dim=1).unsqueeze(-1)
        ctx = (attn_w * theta).sum(dim=1)
        return ctx, (attn_w if ret_attn else None)

    def _multihead_context(self, theta: torch.Tensor, mask: Optional[torch.BoolTensor], ret_attn: bool):
        B, T, F = theta.shape
        if T == 0:
            return torch.zeros((B, F), device=theta.device, dtype=theta.dtype), None
        if self._mha is None:
            mha_dropout = self.dropout_layer.p if isinstance(self.dropout_layer, nn.Dropout) else 0.0
            self._mha = nn.MultiheadAttention(embed_dim=F, num_heads=self.n_heads, batch_first=True, dropout=mha_dropout, device=self.device)
        key_padding_mask = (~mask.bool()) if mask is not None else None
        out, attn = self._mha(theta, theta, theta, key_padding_mask=key_padding_mask)
        ctx = out.mean(dim=1)
        return ctx, (attn if ret_attn else None)

    # ---------------- Accessors / Mutators ----------------
    def get_sequence(self, detach: bool = True) -> Optional[torch.Tensor]:
        return self._sequence.detach() if detach and self._sequence is not None else self._sequence

    def get_context(self, detach: bool = True) -> Optional[torch.Tensor]:
        return self._context.detach() if detach and self._context is not None else self._context

    def set_sequence(self, sequence: torch.Tensor, detach: bool = True, recompute_context: bool = False, mask: Optional[torch.BoolTensor] = None):
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


@dataclass
class SequenceSet:
    """
    Batch-major container for sequences, contexts, canonical contexts, log-probs, and masks.
    All tensors have shape [B, T, ...], sequences are padded to the same length.
    """
    lengths: torch.Tensor            # [B]
    contexts: torch.Tensor           # [B, T, H]
    sequences: torch.Tensor          # [B, T, F]
    canonical: torch.Tensor          # [B, 1, H]
    masks: torch.BoolTensor          # [B, T, 1]
    log_probs: Optional[torch.Tensor] = None  # [B, T, K]

    @classmethod
    def from_unbatched(cls,
        sequences: Sequence[torch.Tensor],
        contexts: Optional[Sequence[Optional[torch.Tensor]]] = None,
        log_probs: Optional[Sequence[Optional[torch.Tensor]]] = None,
        pad_value: float = 0.0) -> "SequenceSet":
        if not sequences:
            raise ValueError("`sequences` must be a non-empty list of tensors.")

        device, dtype = sequences[0].device, sequences[0].dtype
        B = len(sequences)
        lengths = torch.tensor([s.shape[0] for s in sequences], dtype=torch.long, device=device)
        if (lengths <= 0).any():
            raise ValueError("All sequences must have length >= 1.")

        T_max = int(lengths.max())
        F = sequences[0].shape[1] if sequences[0].ndim > 1 else 1

        # Determine context dimension
        ctx_dim = F
        if contexts is not None:
            for c in contexts:
                if c is not None:
                    ctx_dim = c.shape[-1] if c.ndim > 1 else 1
                    break

        # Allocate tensors
        seq_tensor = torch.full((B, T_max, F), pad_value, dtype=dtype, device=device)
        ctx_tensor = torch.zeros((B, T_max, ctx_dim), dtype=dtype, device=device)
        mask_tensor = torch.zeros((B, T_max, 1), dtype=torch.bool, device=device)
        logp_tensor = None

        if log_probs is not None:
            K = next((lp.shape[1] if lp.ndim > 1 else 1 for lp in log_probs if lp is not None), None)
            if K is None:
                raise ValueError("log_probs provided but all entries are None")
            logp_tensor = torch.full((B, T_max, K), float("-inf"), dtype=dtype, device=device)

        for i, s in enumerate(sequences):
            L = s.shape[0]
            seq_tensor[i, :L] = s.unsqueeze(-1) if s.ndim == 1 else s

            # Valid mask
            valid_mask = torch.ones(L, dtype=torch.bool, device=device)
            if s.ndim == 1:
                valid_mask &= ~torch.isnan(s)
            else:
                valid_mask &= ~torch.isnan(s).all(dim=1)
            mask_tensor[i, :L, 0] = valid_mask

            # Fill contexts
            if contexts and contexts[i] is not None:
                c = contexts[i]
                if c.ndim == 1:
                    ctx_tensor[i, :L] = c.unsqueeze(0).expand(L, -1)
                elif c.ndim == 2:
                    if c.shape[0] == 1:
                        ctx_tensor[i, :L] = c.expand(L, -1)
                    elif c.shape[0] == L:
                        ctx_tensor[i, :L] = c
                    else:
                        raise ValueError(f"Invalid context shape {c.shape} for sequence {i}")
                else:
                    raise ValueError(f"Unsupported context ndim {c.ndim}")
            else:
                # fallback: use sequences as context
                ctx_tensor[i, :L, :F] = seq_tensor[i, :L]

            # Fill log_probs
            if logp_tensor is not None and log_probs[i] is not None:
                lp = log_probs[i].unsqueeze(-1) if log_probs[i].ndim == 1 else log_probs[i]
                logp_tensor[i, :L] = lp

        # Canonical context: first valid timestep per sequence
        first_idx = mask_tensor.squeeze(-1).float().argmax(dim=1)
        canonical_tensor = ctx_tensor[torch.arange(B, device=device), first_idx].unsqueeze(1)

        return cls(
            sequences=seq_tensor,
            lengths=lengths,
            masks=mask_tensor,
            contexts=ctx_tensor,
            canonical=canonical_tensor,
            log_probs=logp_tensor
        )

    @property
    def n_sequences(self) -> int:
        return self.sequences.shape[0]

    @property
    def total_timesteps(self) -> int:
        return int(self.lengths.sum())

    @property
    def feature_dim(self) -> int:
        return self.sequences.shape[2]

    @property
    def context_dim(self) -> int:
        return self.contexts.shape[2]

    @property
    def device(self) -> torch.device:
        return self.sequences.device

    @property
    def dtype(self) -> torch.dtype:
        return self.sequences.dtype

    def to(self, device=None, dtype=None) -> "SequenceSet":
        kwargs = {}
        if device: kwargs["device"] = device
        if dtype: kwargs["dtype"] = dtype
        return SequenceSet(
            sequences=self.sequences.to(**kwargs),
            lengths=self.lengths.to(**kwargs),
            masks=self.masks.to(**kwargs),
            contexts=self.contexts.to(**kwargs),
            canonical=self.canonical.to(**kwargs),
            log_probs=self.log_probs.to(**kwargs) if self.log_probs is not None else None
        )

    def index_select(self, idx: torch.Tensor) -> "SequenceSet":
        return SequenceSet(
            sequences=self.sequences[idx],
            lengths=self.lengths[idx],
            masks=self.masks[idx],
            contexts=self.contexts[idx],
            canonical=self.canonical[idx],
            log_probs=self.log_probs[idx] if self.log_probs is not None else None
        )

    @staticmethod
    def batchify(items: Sequence[torch.Tensor], pad_value: float = 0.0) -> torch.Tensor:
        if not items:
            raise ValueError("Cannot batchify empty list.")
        B = len(items)
        T_max = max(t.shape[0] for t in items)
        F = items[0].shape[1] if items[0].ndim > 1 else 1
        device, dtype = items[0].device, items[0].dtype

        out = torch.full((B, T_max, F), pad_value, dtype=dtype, device=device)
        for i, t in enumerate(items):
            t2 = t if t.ndim > 1 else t.unsqueeze(-1)
            out[i, :t2.shape[0], :t2.shape[1]] = t2
        return out

    def update(self,
        encoder: ContextEncoder,
        pool: Optional[str] = None,
        detach: bool = False):
        """
        Update SequenceSet contexts and canonical context from a ContextEncoder.

        Args:
            encoder: ContextEncoder instance
            pool: optional pooling method override for this call
            detach: whether to detach encoder output
        """
        x = self.sequences
        mask = self.masks.squeeze(-1)

        # override pooling temporarily if requested
        old_pool = encoder.pool
        if pool is not None:
            encoder.pool = pool

        with torch.no_grad() if detach else contextlib.nullcontext():
            seq_out, ctx, _ = encoder(x, mask=mask, return_context=True, return_sequence=False)

        # restore original pool
        if pool is not None:
            encoder.pool = old_pool

        # ctx: [B, 1, F_out], broadcast to full sequence
        self.contexts = ctx.expand(-1, x.shape[1], -1)
        self.canonical = ctx


from dataclasses import dataclass, field
from typing import Optional, List, Dict, Union
import torch

@dataclass
class ContextRouter:
    context: torch.Tensor               # [B,T,H]
    canonical: torch.Tensor             # [B,1,H]
    names: Optional[List[str]] = None
    mask: Optional[torch.Tensor] = None # [B,T,1]
    log_probs: Optional[torch.Tensor] = None
    _cache: Dict[str, torch.Tensor] = field(default_factory=dict, init=False)

    def __post_init__(self):
        B, one, H = self.canonical.shape
        if self.canonical.ndim != 3 or one != 1:
            raise ValueError(f"canonical must be [B,1,H], got {tuple(self.canonical.shape)}")
        if self.context.ndim != 3 or self.context.shape[0] != B or self.context.shape[2] != H:
            raise ValueError(f"context must be [B,T,H] with batch/feature dims matching canonical, got {tuple(self.context.shape)}")
        if self.names is None:
            self.names = [f"context{i}" for i in range(H)]
        elif len(self.names) != H:
            raise ValueError(f"names length {len(self.names)} != feature dim {H}")
        if self.mask is None:
            self.mask = torch.ones(B, self.context.shape[1], 1, device=self.context.device, dtype=torch.bool)
        elif self.mask.ndim == 2:
            self.mask = self.mask.unsqueeze(-1)
        elif self.mask.shape != (B, self.context.shape[1], 1):
            raise ValueError(f"mask must be [B,T,1], got {self.mask.shape}")
        if self.log_probs is not None and self.log_probs.shape[0] != B:
            raise ValueError(f"log_probs must share batch dim B={B}, got {tuple(self.log_probs.shape)}")
        self.device = self.context.device
        self.dtype = self.context.dtype

    @classmethod
    def from_tensor(
        cls,
        X: "SequenceSet",
        theta: Optional[Union[torch.Tensor, "ContextRouter"]] = None,
        mode: str = "additive",
    ) -> "ContextRouter":
        """
        Build ContextRouter from a SequenceSet and optional theta.

        Args:
            X: base SequenceSet
            theta: tensor [1,H], [B,H], [T,H], [B,T,H] or ContextRouter
            mode: "additive" or "replace"
        """
        if not isinstance(X, SequenceSet):
            raise TypeError("X must be a SequenceSet")

        B, T, H = X.n_sequences, X.sequences.shape[1], X.context_dim
        canonical = X.canonical.clone()
        context = X.contexts.clone()
        mask = X.masks.clone()
        log_probs = X.log_probs.clone() if X.log_probs is not None else None
        names = [f"context{i}" for i in range(H)]

        theta_c, theta_ctx, theta_m, theta_lp, theta_names = None, None, None, None, None

        # Extract from ContextRouter if theta is one
        if isinstance(theta, ContextRouter):
            theta_c = theta.canonical
            theta_ctx = theta.context
            theta_m = theta.mask
            theta_lp = theta.log_probs
            theta_names = theta.names
        elif isinstance(theta, torch.Tensor):
            theta_ctx = theta

        # Broadcast theta_ctx to [B,T,H]
        if theta_ctx is not None:
            if theta_ctx.ndim == 1:
                theta_ctx = theta_ctx.view(1, 1, -1).expand(B, T, H)
            elif theta_ctx.ndim == 2:
                if theta_ctx.shape[0] == B:
                    theta_ctx = theta_ctx.view(B, 1, H).expand(B, T, H)
                elif theta_ctx.shape[0] == T:
                    theta_ctx = theta_ctx.unsqueeze(0).expand(B, T, H)
                else:
                    raise ValueError(f"Cannot align 2D theta {theta_ctx.shape} with context {context.shape}")
            elif theta_ctx.ndim == 3:
                if theta_ctx.shape != (B, T, H):
                    raise ValueError(f"3D theta {theta_ctx.shape} incompatible with context {context.shape}")
            else:
                raise ValueError(f"Unsupported theta ndim {theta_ctx.ndim}")

            if mode == "additive":
                context = context + theta_ctx
            elif mode == "replace":
                context = theta_ctx
                canonical = theta_ctx[:, :1, :]
            else:
                raise ValueError(f"Unsupported mode {mode}")

        # Replace attributes if theta is a ContextRouter in "replace" mode
        if mode == "replace":
            if theta_c is not None:
                canonical = theta_c
            if theta_m is not None:
                mask = theta_m
            if theta_lp is not None:
                log_probs = theta_lp
            if theta_names is not None:
                names = theta_names

        # Ensure mask shape
        if mask.ndim == 2:
            mask = mask.unsqueeze(-1)
        elif mask.shape != (B, T, 1):
            raise ValueError(f"mask must be [B,T,1], got {mask.shape}")

        return cls(
            canonical=canonical,
            context=context,
            mask=mask,
            log_probs=log_probs,
            names=names
        )

    # Convenience accessors
    def get_context(self) -> torch.Tensor: return self.context
    def get_canonical(self) -> torch.Tensor: return self.canonical
    def get_feature_names(self) -> List[str]: return self.names
    def get_log_probs(self) -> Optional[torch.Tensor]: return self.log_probs
    def get_mask(self) -> torch.Tensor: return self.mask

    # Utilities
    def select_features(self, keys: List[str]) -> "ContextRouter":
        idx = [self.names.index(k) for k in keys]
        return ContextRouter(
            canonical=self.canonical[:, :, idx],
            context=self.context[:, :, idx],
            names=keys,
            log_probs=self.log_probs,
            mask=self.mask
        )

    def to(self, device=None, dtype=None) -> "ContextRouter":
        return ContextRouter(
            canonical=self.canonical.to(device=device, dtype=dtype),
            context=self.context.to(device=device, dtype=dtype),
            names=self.names,
            log_probs=(self.log_probs.to(device=device, dtype=dtype) if self.log_probs is not None else None),
            mask=self.mask.to(device=device, dtype=dtype)
        )

    def detach(self) -> "ContextRouter":
        return ContextRouter(
            canonical=self.canonical.detach(),
            context=self.context.detach(),
            names=self.names,
            log_probs=(self.log_probs.detach() if self.log_probs is not None else None),
            mask=self.mask.detach()
        )

    def clone(self) -> "ContextRouter":
        return ContextRouter(
            canonical=self.canonical.clone(),
            context=self.context.clone(),
            names=self.names,
            log_probs=(self.log_probs.clone() if self.log_probs is not None else None),
            mask=self.mask.clone()
        )

    def __repr__(self):
        return (f"<ContextRouter canonical={tuple(self.canonical.shape)} "
                f"context={tuple(self.context.shape)} names={self.names} "
                f"log_probs={'Yes' if self.log_probs is not None else 'No'} "
                f"mask={'Yes' if self.mask is not None else 'No'} "
                f"device={self.device} dtype={self.dtype}>")

