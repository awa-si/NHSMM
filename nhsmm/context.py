# nhsmm/context.py
from __future__ import annotations
from typing import Optional, Literal, Tuple, Callable, Dict, Sequence
from dataclasses import dataclass, field
import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as nnF
from torch.nn.utils.rnn import pad_sequence

from nhsmm.constants import DTYPE, EPS, logger


@dataclass
class SequenceSet:

    lengths: torch.Tensor            # [B]
    contexts: torch.Tensor           # [B, T, H]
    sequences: torch.Tensor          # [B, T, F]
    canonical: torch.Tensor          # [B, 1, H] pooled context
    masks: torch.BoolTensor          # [B, T, 1]
    log_probs: Optional[torch.Tensor] = None  # [B, T, K]

    @classmethod
    def from_unbatched(
        cls,
        sequences: Sequence[torch.Tensor],
        contexts: Optional[Sequence[Optional[torch.Tensor]]] = None,
        log_probs: Optional[Sequence[Optional[torch.Tensor]]] = None,
        pad_value: float = 0.0,
        context_pad_value: Optional[float] = None,
    ) -> "SequenceSet":
        if not sequences:
            raise ValueError("`sequences` must be a non-empty list of tensors.")

        device, dtype = sequences[0].device, sequences[0].dtype
        B = len(sequences)
        lengths = torch.tensor([s.shape[0] for s in sequences], dtype=torch.long, device=device)
        if (lengths <= 0).any():
            raise ValueError("All sequences must have length >= 1.")

        # ---------------- Sequences ----------------
        seq_tensors = [s if s.ndim > 1 else s.unsqueeze(-1) for s in sequences]
        seq_tensor = pad_sequence(seq_tensors, batch_first=True, padding_value=pad_value)
        T_max, F = seq_tensor.shape[1], seq_tensor.shape[2]

        # ---------------- Masks ----------------
        mask_tensor = torch.arange(T_max, device=device).expand(B, T_max) < lengths.unsqueeze(1)
        mask_tensor = mask_tensor.unsqueeze(-1)

        # ---------------- Contexts ----------------
        ctx_dim = F
        if contexts is not None:
            for c in contexts:
                if c is not None:
                    ctx_dim = c.shape[-1] if c.ndim > 1 else 1
                    break

        ctx_pad = context_pad_value if context_pad_value is not None else 0.0
        ctx_tensor = torch.full((B, T_max, ctx_dim), ctx_pad, dtype=dtype, device=device)

        if contexts is not None:
            for i, c in enumerate(contexts):
                L = lengths[i]
                if c is None:
                    ctx_tensor[i, :L, :F] = seq_tensor[i, :L]
                else:
                    c_ = c if c.ndim == 2 else c.unsqueeze(0).expand(L, -1)
                    ctx_tensor[i, :L, :c_.shape[1]] = c_
        else:
            ctx_tensor[:, :, :F] = seq_tensor

        # ---------------- Log-probs ----------------
        logp_tensor = None
        if log_probs is not None:
            K = max(lp.shape[1] if lp is not None and lp.ndim > 1 else 1 for lp in log_probs)
            logp_tensor = torch.full((B, T_max, K), float("-inf"), dtype=dtype, device=device)
            for i, lp in enumerate(log_probs):
                if lp is not None:
                    lp_ = lp if lp.ndim > 1 else lp.unsqueeze(-1)
                    logp_tensor[i, :lp_.shape[0], :lp_.shape[1]] = lp_

        # ---------------- Canonical ----------------
        denom = mask_tensor.sum(dim=1).clamp_min(1).to(dtype=dtype)
        canonical_tensor = (ctx_tensor * mask_tensor.to(dtype=dtype)).sum(dim=1) / denom
        canonical_tensor = canonical_tensor.unsqueeze(1)

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

    def select(self, indices: torch.Tensor | list[int]) -> "SequenceSet":
        if isinstance(indices, list):
            indices = torch.tensor(indices, dtype=torch.long, device=self.sequences.device)
        return SequenceSet(
            sequences=self.sequences[indices],
            lengths=self.lengths[indices],
            masks=self.masks[indices],
            contexts=self.contexts[indices],
            canonical=self.canonical[indices],
            log_probs=self.log_probs[indices] if self.log_probs is not None else None
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
            t_ = t if t.ndim > 1 else t.unsqueeze(-1)
            out[i, :t_.shape[0], :t_.shape[1]] = t_
        return out

    def update(self, encoder, pool: Optional[str] = None, detach: bool = False):
        x = self.sequences
        mask = self.masks.squeeze(-1)  # [B, T]

        # Temporarily override pooling if needed
        old_pool = encoder.pool
        if pool is not None:
            encoder.pool = pool

        # Forward pass through encoder
        ctx: torch.Tensor
        with torch.no_grad() if detach else contextlib.nullcontext():
            _, ctx, _ = encoder(x, mask=mask, return_context=True, return_sequence=False)

        if pool is not None:
            encoder.pool = old_pool

        # Broadcast canonical context to per-timestep contexts
        self.contexts = ctx.expand(-1, x.shape[1], -1)
        self.canonical = ctx


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
            raise ValueError(f"context must be [B,T,H] matching canonical, got {tuple(self.context.shape)}")
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
    def from_tensor(cls,
        X: "SequenceSet",
        context: Optional[Union[torch.Tensor, "ContextRouter"]] = None,
        mode: str = "additive") -> "ContextRouter":
        """
        Build ContextRouter from a SequenceSet and optional context.
        Canonical is pooled from context unless replaced.
        """
        if not isinstance(X, SequenceSet):
            raise TypeError("X must be a SequenceSet")

        B, T, H = X.n_sequences, X.sequences.shape[1], X.context_dim
        canonical = X.canonical.clone()
        ctx = X.contexts.clone()
        mask = X.masks.clone()
        log_probs = X.log_probs.clone() if X.log_probs is not None else None
        names = [f"context{i}" for i in range(H)]

        # Process context override
        ctx_override: Optional[torch.Tensor] = None
        if isinstance(context, ContextRouter):
            ctx_override = context.context
        elif isinstance(context, torch.Tensor):
            ctx_override = context

        if ctx_override is not None:
            if ctx_override.ndim == 1:
                ctx_override = ctx_override.reshape(1, 1, H).expand(B, T, H)
            elif ctx_override.ndim == 2:
                if ctx_override.shape[0] == B and ctx_override.shape[1] == H:
                    ctx_override = ctx_override.unsqueeze(1).expand(B, T, H)
                elif ctx_override.shape[0] == T and ctx_override.shape[1] == H:
                    ctx_override = ctx_override.unsqueeze(0).expand(B, T, H)
                else:
                    raise ValueError(f"Cannot align 2D context {ctx_override.shape} with context {ctx.shape}")
            elif ctx_override.ndim == 3:
                if ctx_override.shape != (B, T, H):
                    raise ValueError(f"3D context {ctx_override.shape} incompatible with context {ctx.shape}")
            else:
                raise ValueError(f"Unsupported context ndim {ctx_override.ndim}")

            if mode == "additive":
                ctx = ctx + ctx_override
            elif mode == "replace":
                ctx = ctx_override
                canonical = ctx_override[:, :1, :]
            else:
                raise ValueError(f"Unsupported mode {mode}")

        return cls(
            canonical=canonical,
            context=ctx,
            mask=mask,
            log_probs=log_probs,
            names=names
        )

    # ---------------- Accessors ----------------
    def get_context(self) -> torch.Tensor: return self.context
    def get_canonical(self) -> torch.Tensor: return self.canonical
    def get_feature_names(self) -> List[str]: return self.names
    def get_log_probs(self) -> Optional[torch.Tensor]: return self.log_probs
    def get_mask(self) -> torch.Tensor: return self.mask

    # ---------------- Selection ----------------
    def select_features(self, keys: List[str]) -> "ContextRouter":
        idx = torch.tensor([self.names.index(k) for k in keys], device=self.context.device)
        return ContextRouter(
            canonical=self.canonical[:, :, idx],
            context=self.context[:, :, idx],
            names=keys.copy(),
            log_probs=self.log_probs,
            mask=self.mask
        )

    def select(self, indices: torch.Tensor | list[int]) -> "ContextRouter":
        if isinstance(indices, list):
            indices = torch.tensor(indices, dtype=torch.long, device=self.context.device)
        return ContextRouter(
            canonical=self.canonical[indices],
            context=self.context[indices],
            names=self.names.copy() if self.names is not None else None,
            mask=self.mask[indices] if self.mask is not None else None,
            log_probs=self.log_probs[indices] if self.log_probs is not None else None
        )

    # ---------------- Device / dtype / clone ----------------
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
            names=self.names.copy(),
            log_probs=(self.log_probs.clone() if self.log_probs is not None else None),
            mask=self.mask.clone()
        )

    def __repr__(self):
        return (f"<ContextRouter canonical={tuple(self.canonical.shape)} "
                f"context={tuple(self.context.shape)} names={self.names} "
                f"log_probs={'Yes' if self.log_probs is not None else 'No'} "
                f"mask={'Yes' if self.mask is not None else 'No'} "
                f"device={self.device} dtype={self.dtype}>")


class ContextEncoder(nn.Module):

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
        self.dropout_layer: nn.Module = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.debug = debug

        # cached outputs
        self._sequence: Optional[torch.Tensor] = None
        self._context: Optional[torch.Tensor] = None
        self._attn_vector: Optional[nn.Parameter] = None
        self._mha: Optional[nn.MultiheadAttention] = None

        self._POOLERS: Dict[str, Callable] = {
            "mean": self._pool_mean,
            "last": self._pool_last,
            "max": self._pool_max,
            "attn": self._attention_context,
            "mha": self._multihead_context,
        }

    def forward(self,
        x: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
        return_context: bool = False,
        return_attn_weights: bool = False, detach_context: bool = True,
        return_sequence: bool = False, ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:

        if x.ndim == 2: x = x.unsqueeze(0)
        B, T, _ = x.shape
        mask = self._prepare_mask(mask, B, T)

        out = self.encoder(x, mask=mask) if "mask" in self._encoder_signature() else self.encoder(x)
        if isinstance(out, (tuple, list)):
            out = out[0]

        theta = out * mask.unsqueeze(-1) if mask is not None else out
        self._sequence = theta.detach() if detach_context else theta

        pooled, attn = self._pool_context(theta, mask, return_attn_weights)
        if self.layer_norm:
            pooled = nnF.layer_norm(pooled, (pooled.shape[-1],))
        pooled = self.dropout_layer(torch.tanh(pooled * self.context_scale))
        ctx = pooled.unsqueeze(1)
        self._context = ctx.detach() if detach_context else ctx

        seq_out = theta if return_sequence else self._last_timestep(theta, mask)
        return seq_out, (ctx if return_context else None), (attn if return_attn_weights else None)

    def _encoder_signature(self) -> Tuple[str, ...]:
        try:
            return tuple(p.name for p in self.encoder.forward.__code__.co_varnames[:self.encoder.forward.__code__.co_argcount])
        except Exception:
            return tuple()

    def _prepare_mask(self, mask: Optional[torch.BoolTensor], B: int, T: int) -> torch.BoolTensor:
        if mask is None:
            return torch.ones(B, T, dtype=torch.bool)
        mask = mask.bool()
        if mask.ndim == 1:
            mask = mask.unsqueeze(0).expand(B, -1)
        elif mask.ndim == 3:
            mask = mask.squeeze(-1)
        return mask[:, :T]

    def _last_timestep(self, theta: torch.Tensor, mask: Optional[torch.BoolTensor]) -> torch.Tensor:
        if mask is not None:
            idx = torch.clamp(mask.sum(dim=1) - 1, min=0)
            return theta[torch.arange(theta.shape[0]), idx]
        return theta[:, -1, :]

    def _pool_context(self, theta: torch.Tensor, mask: Optional[torch.BoolTensor], ret_attn: bool):
        if self.pool not in self._POOLERS:
            raise ValueError(f"Invalid pooling method '{self.pool}'")
        return self._POOLERS[self.pool](theta, mask, ret_attn)

    def _pool_mean(self, theta, mask, ret_attn):
        if mask is not None:
            denom = mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
            ctx = (theta * mask.unsqueeze(-1)).sum(dim=1) / denom
        else:
            ctx = theta.mean(dim=1)
        return ctx, None

    def _pool_last(self, theta, mask, ret_attn):
        return self._last_timestep(theta, mask), None

    def _pool_max(self, theta, mask, ret_attn):
        if mask is not None:
            masked = theta.masked_fill(~mask.unsqueeze(-1), float("-inf"))
            ctx = masked.max(dim=1).values
            ctx = torch.where(torch.isfinite(ctx), ctx, torch.zeros_like(ctx))
        else:
            ctx = theta.max(dim=1).values
        return ctx, None

    def _init_attn_vector(self, F: int):
        if self._attn_vector is None or self._attn_vector.shape[0] != F:
            self._attn_vector = nn.Parameter(torch.randn(F) * 0.1)

    def _attention_context(self, theta, mask, ret_attn):
        B, T, F = theta.shape
        self._init_attn_vector(F)
        scores = torch.einsum("btf,f->bt", theta, self._attn_vector)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
            scores[mask.sum(dim=1) == 0] = 0.0
        attn_w = F.softmax(scores, dim=1).unsqueeze(-1)
        ctx = (attn_w * theta).sum(dim=1)
        return ctx, (attn_w if ret_attn else None)

    def _multihead_context(self, theta, mask, ret_attn):
        B, T, F = theta.shape
        if self._mha is None:
            self._mha = nn.MultiheadAttention(embed_dim=F, num_heads=self.n_heads, batch_first=True, dropout=self.dropout_layer.p if isinstance(self.dropout_layer, nn.Dropout) else 0.0)
        key_padding_mask = (~mask) if mask is not None else None
        out, attn = self._mha(theta, theta, theta, key_padding_mask=key_padding_mask)
        ctx = out.mean(dim=1)
        return ctx, (attn if ret_attn else None)

    def reset(self):
        self._sequence = None
        self._context = None
        self._attn_vector = None
        self._mha = None

    def encode(self,
        sequences: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
        pool: Optional[str] = None, detach: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:

        if pool is not None:
            old_pool = self.pool
            self.pool = pool
        else:
            old_pool = None

        seq_features, ctx_features, _ = self.forward(
            sequences, mask=mask, return_sequence=True, return_context=True, detach_context=detach
        )

        if old_pool is not None:
            self.pool = old_pool

        if ctx_features.ndim == 2:
            ctx_features = ctx_features.unsqueeze(1)
        elif ctx_features.ndim == 3 and ctx_features.shape[1] != 1:
            ctx_features = ctx_features.mean(dim=1, keepdim=True)

        return seq_features, ctx_features


