# nhsmm/utils.py
from __future__ import annotations

from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
import torch


@dataclass
class SequenceSet:
    """
    Batch-major container for sequences, log-probabilities, contexts, canonical contexts, and masks.
    All fields are Tensors with batch dimension first: [B, T, ...]. Sequences are padded to the same T.

    Fields
    ------
    sequences : torch.Tensor    # [B, T, F]
    lengths   : torch.Tensor    # [B] (long) valid timesteps per sequence
    masks     : torch.BoolTensor# [B, T, 1] boolean mask (True = valid)
    contexts  : torch.Tensor    # [B, T, H]
    canonical : torch.Tensor    # [B, 1, H] canonical context (first valid timestep)
    log_probs : Optional[torch.Tensor] = None  # [B, T, K] per-state log-probs (or None)
    """

    sequences: torch.Tensor
    lengths: torch.Tensor
    masks: torch.BoolTensor
    contexts: torch.Tensor
    canonical: torch.Tensor
    log_probs: Optional[torch.Tensor] = None

    # ---------------- Construction helpers ----------------
    @classmethod
    def from_unbatched(
        cls,
        sequences: Sequence[torch.Tensor],
        contexts: Optional[Sequence[Optional[torch.Tensor]]] = None,
        canonical: Optional[Sequence[Optional[torch.Tensor]]] = None,
        log_probs: Optional[Sequence[Optional[torch.Tensor]]] = None,
        masks: Optional[Sequence[Optional[torch.Tensor]]] = None,
        pad_value: float = 0.0,
    ) -> "SequenceSet":
        """
        Build a batch-major SequenceSet from lists of tensors (unbatched).
        - sequences: list of [T_i, F] or [T_i] tensors
        - contexts: optional list of [T_i, H] or [1, H] or [T_i] or None
        - canonical: optional list of [1, H] or [H]
        - log_probs: optional list of [T_i, K] or [T_i] tensors
        - masks: optional list of [T_i] or [T_i,1] boolean tensors

        Raises:
            ValueError on empty input list, mismatched feature-dimensions, or zero-length sequences.
        """
        if not isinstance(sequences, (list, tuple)) or len(sequences) == 0:
            raise ValueError("`sequences` must be a non-empty list of tensors.")

        device = sequences[0].device
        dtype = sequences[0].dtype

        # lengths and check non-empty
        lengths = torch.tensor([int(s.shape[0]) for s in sequences], dtype=torch.long, device=device)
        if (lengths <= 0).any():
            raise ValueError("All sequences must have positive length (no empty sequences allowed).")

        B = len(sequences)
        T_max = int(lengths.max().item())

        # feature dimension
        def _feat_dim(t: torch.Tensor) -> int:
            return t.shape[1] if t.ndim > 1 else 1

        feature_dim = _feat_dim(sequences[0])
        if any(_feat_dim(s) != feature_dim for s in sequences):
            raise ValueError("All sequences must have the same per-step feature dimension.")

        # context dim if provided
        ctx_dim = None
        if contexts is not None:
            # find first non-None context to infer dim
            for c in contexts:
                if c is not None:
                    ctx_dim = c.shape[-1] if c.ndim > 1 else 1
                    break
        if ctx_dim is None:
            ctx_dim = feature_dim  # fallback

        # prepare tensors
        seq_tensor = torch.full((B, T_max, feature_dim), float(pad_value), dtype=dtype, device=device)
        ctx_tensor = torch.zeros((B, T_max, ctx_dim), dtype=dtype, device=device)
        mask_tensor = torch.zeros((B, T_max, 1), dtype=torch.bool, device=device)
        logp_tensor = None
        if log_probs is not None:
            # infer K from first non-None
            K = None
            for lp in log_probs:
                if lp is not None:
                    K = lp.shape[1] if lp.ndim > 1 else 1
                    break
            if K is None:
                raise ValueError("`log_probs` provided but all entries are None")
            logp_tensor = torch.full((B, T_max, K), float("-inf"), dtype=dtype, device=device)

        for i in range(B):
            s = sequences[i]
            L = s.shape[0]
            if L == 0:
                raise ValueError("Zero-length sequence encountered; all sequences must be length >= 1.")
            # fill sequence (handle 1D -> unsqueeze)
            if s.ndim == 1:
                seq_tensor[i, :L, 0] = s.to(device=device, dtype=dtype)
            else:
                if s.shape[1] != feature_dim:
                    raise ValueError("Feature dimension mismatch when building batch.")
                seq_tensor[i, :L] = s.to(device=device, dtype=dtype)

            # mask
            if masks is None or masks[i] is None:
                mask_tensor[i, :L, 0] = True
            else:
                m = masks[i].to(device=device)
                if m.ndim == 1:
                    m = m.unsqueeze(-1)
                if m.shape[0] != L:
                    # try to reshape/trim
                    m = m.reshape(L, -1)[:, 0]
                mask_tensor[i, :L, 0] = m.view(-1)[:L].bool()

            # contexts
            if contexts is not None and contexts[i] is not None:
                c = contexts[i].to(device=device, dtype=dtype)
                if c.ndim == 1:
                    # interpret as [H] -> repeat
                    ctx_tensor[i, :L] = c.unsqueeze(0).expand(L, -1)
                elif c.ndim == 2:
                    if c.shape[0] == 1 and c.shape[1] == ctx_dim:
                        ctx_tensor[i, :L] = c.expand(L, -1)
                    elif c.shape[0] == L and c.shape[1] == ctx_dim:
                        ctx_tensor[i, :L] = c
                    else:
                        raise ValueError(f"Unsupported context shape {c.shape} for sequence {i}.")
                else:
                    raise ValueError(f"Unsupported context ndim {c.ndim} for sequence {i}.")

            # log-probs
            if logp_tensor is not None and log_probs is not None and log_probs[i] is not None:
                lp = log_probs[i].to(device=device, dtype=dtype)
                if lp.ndim == 1:
                    # interpret as [T] -> [T,1]
                    lp = lp.unsqueeze(-1)
                if lp.shape[0] != L:
                    raise ValueError("log_probs length mismatch for sequence {}".format(i))
                if lp.shape[1] != logp_tensor.shape[2]:
                    raise ValueError("log_probs feature (K) mismatch.")
                logp_tensor[i, :L] = lp

        # canonical: first valid timestep per sequence (require at least one True per row)
        valid_counts = mask_tensor.squeeze(-1).sum(dim=1)
        if (valid_counts == 0).any():
            raise ValueError("All sequences must contain at least one valid timestep (mask).")

        first_idx = mask_tensor.squeeze(-1).float().argmax(dim=1)
        canonical_tensor = ctx_tensor[torch.arange(B, device=device), first_idx].unsqueeze(1)

        return cls(
            sequences=seq_tensor,
            lengths=lengths,
            masks=mask_tensor,
            contexts=ctx_tensor,
            canonical=canonical_tensor,
            log_probs=logp_tensor,
        )

    # ---------------- Properties ----------------
    @property
    def n_sequences(self) -> int:
        return int(self.sequences.shape[0])

    @property
    def total_timesteps(self) -> int:
        return int(self.lengths.sum().item())

    @property
    def feature_dim(self) -> int:
        return int(self.sequences.shape[2])

    @property
    def context_dim(self) -> int:
        return int(self.contexts.shape[2])

    @property
    def device(self) -> torch.device:
        return self.sequences.device

    @property
    def dtype(self) -> torch.dtype:
        return self.sequences.dtype

    # ---------------- Tensor ops ----------------
    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> "SequenceSet":
        kwargs = {}
        if device is not None:
            kwargs["device"] = device
        if dtype is not None:
            kwargs["dtype"] = dtype
        return SequenceSet(
            sequences=self.sequences.to(**kwargs),
            lengths=self.lengths.to(**kwargs),
            masks=self.masks.to(**kwargs),
            contexts=self.contexts.to(**kwargs),
            canonical=self.canonical.to(**kwargs),
            log_probs=self.log_probs.to(**kwargs) if self.log_probs is not None else None,
        )

    def index_select(self, idx: torch.Tensor) -> "SequenceSet":
        """
        Select subset of batch by index tensor (1D LongTensor).
        """
        return SequenceSet(
            sequences=self.sequences.index_select(0, idx),
            lengths=self.lengths.index_select(0, idx),
            masks=self.masks.index_select(0, idx),
            contexts=self.contexts.index_select(0, idx),
            canonical=self.canonical.index_select(0, idx),
            log_probs=self.log_probs.index_select(0, idx) if self.log_probs is not None else None,
        )

    def to_padded(self, pad_value: float = 0.0) -> "SequenceSet":
        """
        Already batch-major and padded; kept for API compatibility.
        """
        return self

    # ---------------- Utility: batchify ----------------
    @staticmethod
    def batchify(items: Sequence[torch.Tensor], pad_value: float = 0.0) -> torch.Tensor:
        """
        Convert a list of [T_i, F] tensors to a batch-major tensor [B, T_max, F]
        with padding. Works for sequences, contexts, masks (1D/2D), or log-probs.

        Raises on:
            - items is None
            - inconsistent feature dimensions across items
            - empty list
        """
        if items is None:
            raise ValueError("Cannot batchify None")

        if not isinstance(items, (list, tuple)) or len(items) == 0:
            raise ValueError("`items` must be a non-empty list or tuple of tensors.")

        batch_size = len(items)
        max_len = max(int(t.shape[0]) for t in items)
        feature_dim = items[0].shape[1] if items[0].ndim > 1 else 1
        device = items[0].device
        dtype = items[0].dtype

        # check all feature dims match
        for t in items:
            fd = t.shape[1] if t.ndim > 1 else 1
            if fd != feature_dim:
                raise ValueError("All items must have the same feature dimension for batchify.")

        out = torch.full((batch_size, max_len, feature_dim), float(pad_value), dtype=dtype, device=device)

        for i, t in enumerate(items):
            t_ = t if t.ndim > 1 else t.unsqueeze(-1)
            L = t_.shape[0]
            out[i, :L, : t_.shape[1]] = t_.to(device=device, dtype=dtype)

        return out


@dataclass
class ContextFeatures:
    """
    Container for batch- and time-aware context tensors.

    Each tensor can have shape:
      - [H]       : feature-only
      - [T, H]    : time-dependent
      - [B, H]    : batch-dependent
      - [B, T, H] : batch + time-dependent

    Rules:
      - `time_dependent=True` interprets 2D tensors as [T, H].
        Otherwise, 2D tensors are interpreted as [B, H].
      - Explicit batch (B) or time (T) dimensions are inferred from tensors,
        and all other tensors are broadcast to [B, T, H].
      - Cached concatenations use (B, T) as part of the key.
    """

    n_context: int
    tensors: List[torch.Tensor]
    time_dependent: bool = False
    names: Optional[List[str]] = None
    _cache: Optional[Dict] = None

    def __post_init__(self):
        if not self.tensors or len(self.tensors) != self.n_context:
            raise ValueError(f"Expected {self.n_context} tensors, got {len(self.tensors)}")
        if self.names and len(self.names) != self.n_context:
            raise ValueError("Length of `names` must match `n_context`")
        if not all(torch.is_tensor(t) for t in self.tensors):
            raise TypeError("All elements of tensors must be torch.Tensor")

        # Check devices and dtypes are consistent
        devices = {t.device for t in self.tensors}
        dtypes = {t.dtype for t in self.tensors}
        if len(devices) > 1:
            raise ValueError("All context tensors must be on the same device")
        if len(dtypes) > 1:
            raise ValueError("All context tensors must have the same dtype")

        self._cache = {}
        feature_dims = {self._feature_dim(t) for t in self.tensors if t.ndim >= 1}
        if len(feature_dims) > 1:
            raise ValueError("Inconsistent feature dimensions across context tensors")

    @staticmethod
    def _feature_dim(t: torch.Tensor) -> int:
        return t.shape[-1] if t.ndim >= 1 else 1

    def _infer_BT(self, t: torch.Tensor) -> Tuple[Optional[int], Optional[int]]:
        """
        Return candidate (B, T) from tensor shape.
        """
        if t.ndim == 3:
            return t.shape[0], t.shape[1]
        if t.ndim == 2:
            return (None, t.shape[0]) if self.time_dependent else (t.shape[0], None)
        if t.ndim == 1:
            return None, None
        raise ValueError(f"Unsupported tensor ndim {t.ndim}")

    def infer_global_BT(self) -> Tuple[Optional[int], Optional[int]]:
        """
        Determine global batch (B) and time (T) dimensions from all tensors.
        """
        B_candidates, T_candidates = [], []
        for t in self.tensors:
            b, temp = self._infer_BT(t)
            if b is not None: B_candidates.append(b)
            if temp is not None: T_candidates.append(temp)
        return (max(B_candidates) if B_candidates else None,
                max(T_candidates) if T_candidates else None)

    def _broadcast(self, t: torch.Tensor, B: Optional[int], T: Optional[int]) -> torch.Tensor:
        """
        Broadcast a tensor to [B, T, H] shape.
        """
        H = self._feature_dim(t)

        if t.ndim == 3:
            b, temp, h = t.shape
            if B is not None and b != B: raise ValueError(f"B mismatch: {b} vs {B}")
            if T is not None and temp != T: raise ValueError(f"T mismatch: {temp} vs {T}")
            return t
        if t.ndim == 2:
            if self.time_dependent:
                temp, h = t.shape
                if T is not None and temp != T: raise ValueError(f"T mismatch: {temp} vs {T}")
                return t.unsqueeze(0).expand(B or 1, temp, h)
            else:
                b, h = t.shape
                if B is not None and b != B: raise ValueError(f"B mismatch: {b} vs {B}")
                return t.unsqueeze(1).expand(b, T or 1, h)
        if t.ndim == 1:
            return t.view(1, 1, H).expand(B or 1, T or 1, H)
        raise ValueError(f"Unsupported tensor ndim {t.ndim}")

    def concatenate(self, dim: int = -1, normalize: bool = False, eps: float = 1e-6) -> torch.Tensor:
        """
        Concatenate all context tensors to shape [B, T, H_combined].
        """
        B, T = self.infer_global_BT()
        key = (dim, normalize, B, T)
        if key in self._cache:
            return self._cache[key]

        broadcasted = [self._broadcast(t, B, T) for t in self.tensors]

        dim_adj = 3 + dim if dim < 0 else dim
        if dim_adj not in {0, 1, 2}:
            raise ValueError("`dim` must be in [-3, 2]")

        out = broadcasted[0] if len(broadcasted) == 1 else torch.cat(broadcasted, dim=dim_adj)

        if normalize:
            mean = out.mean((0, 1), keepdim=True)
            std = out.std((0, 1), keepdim=True).clamp_min(eps)
            out = (out - mean) / std

        self._cache[key] = out
        return out

    # ---------------- Utility ----------------
    def to(self, device, dtype=None) -> "ContextFeatures":
        tensors = [t.to(device=device, dtype=dtype) if dtype else t.to(device) for t in self.tensors]
        return ContextFeatures(self.n_context, tensors, self.time_dependent, self.names)

    def detach(self) -> "ContextFeatures":
        return ContextFeatures(self.n_context, [t.detach() for t in self.tensors],
                                   self.time_dependent, self.names)

    def clone(self) -> "ContextFeatures":
        return ContextFeatures(self.n_context, [t.clone() for t in self.tensors],
                                   self.time_dependent, self.names)

    def __getitem__(self, idx: Union[int, slice]) -> "ContextFeatures":
        tensors = [t[idx] for t in self.tensors]
        names = None if self.names is None else [self.names[i] for i in range(len(self.names))][idx]
        return ContextFeatures(self.n_context, tensors, self.time_dependent, names)

    def __repr__(self):
        names = self.names or [f"context{i}" for i in range(self.n_context)]
        shapes = ", ".join(f"{n}:{tuple(t.shape)}" for n, t in zip(names, self.tensors))
        td_flag = "time-varying" if self.time_dependent else "static"
        return f"<ContextFeatures[{td_flag}] {shapes}>"

