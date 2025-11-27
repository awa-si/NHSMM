# nhsmm/tools/utils.py

import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Tuple


@dataclass(frozen=False)
class SequenceSet:
    """
    Container for sequences, log-probabilities, context tensors, and masks.

    Supports:
        - Single sequence: [timesteps, features] or [timesteps]
        - Batched sequences: [batch, timesteps, features]
        - Contexts: [timesteps, hidden], [1, hidden], [batch, timesteps, hidden], [batch, hidden]
        - Masks: [timesteps], [timesteps, 1], [batch, timesteps], [batch, timesteps, 1]
    """

    sequences: List[torch.Tensor]
    lengths: Optional[List[int]] = None
    masks: Optional[List[torch.Tensor]] = None
    log_probs: Optional[List[torch.Tensor]] = None
    contexts: Optional[List[Optional[torch.Tensor]]] = None

    def __post_init__(self):
        self.sequences = self._canonicalize_list(self.sequences, "sequences")
        seqs = self.sequences

        # Infer sequence lengths if missing
        if self.lengths is None:
            self.lengths = [s.shape[0] for s in seqs]
        else:
            if len(self.lengths) != len(seqs):
                raise ValueError("`lengths` must match number of sequences.")
            for s, l in zip(seqs, self.lengths):
                if s.shape[0] != l:
                    raise ValueError("Sequence length mismatch.")

        # Canonicalize log probabilities
        if self.log_probs is not None:
            self.log_probs = self._canonicalize_list(self.log_probs, "log_probs", allow_none=False)
            if len(self.log_probs) != len(seqs):
                raise ValueError("`log_probs` length mismatch.")
            self.log_probs = [self._normalize_log_probs(lp, timesteps) for lp, timesteps in zip(self.log_probs, self.lengths)]

        # Canonicalize contexts
        if self.contexts is None:
            self.contexts = [None] * len(seqs)
        else:
            self.contexts = self._canonicalize_list(self.contexts, "contexts", allow_none=True)
            if len(self.contexts) != len(seqs):
                raise ValueError("`contexts` length mismatch.")
            self.contexts = [self._normalize_context(ctx, timesteps) if ctx is not None else None
                             for ctx, timesteps in zip(self.contexts, self.lengths)]

        # Canonicalize masks
        if self.masks is None:
            self.masks = [torch.ones(timesteps, 1, dtype=torch.bool) for timesteps in self.lengths]
        else:
            self.masks = self._canonicalize_list(self.masks, "masks", allow_none=False)
            if len(self.masks) != len(seqs):
                raise ValueError("`masks` length mismatch.")
            self.masks = [self._normalize_mask(mask, timesteps) for mask, timesteps in zip(self.masks, self.lengths)]

    # ---------------- Internal helpers ----------------
    def _canonicalize_list(self, items, name, allow_none=False):
        if torch.is_tensor(items):
            return [items]
        if isinstance(items, list):
            if allow_none:
                return [t if t is None else t for t in items]
            return [t for t in items]
        raise TypeError(f"`{name}` must be a tensor or list of tensors.")

    def _normalize_log_probs(self, tensor, timesteps):
        """
        Ensures log_probs tensor shape is [timesteps, features]
        Handles outputs from default distributions: [timesteps], [timesteps, F], [timesteps,1,F], [1,timesteps,F]
        """
        if tensor.ndim == 1 and tensor.shape[0] == timesteps:
            return tensor.unsqueeze(1)
        if tensor.ndim == 2:
            if tensor.shape[0] == timesteps:
                return tensor
            if tensor.shape[0] == 1 and tensor.shape[1] == timesteps:
                return tensor.squeeze(0).unsqueeze(1)
        if tensor.ndim == 3:
            if tensor.shape[0] == 1 and tensor.shape[1] == timesteps:
                return tensor.squeeze(0)
            if tensor.shape[1] == 1 and tensor.shape[0] == timesteps:
                return tensor.squeeze(1)
        raise ValueError(f"Invalid log_probs shape {tensor.shape} for timesteps={timesteps}")

    def _normalize_context(self, ctx, timesteps):
        """
        Normalize context tensor to [timesteps, hidden]
        Handles [timesteps,H], [1,H], [timesteps,1,H], [batch,timesteps,H], [batch,H]
        """
        if ctx.ndim == 1:
            return ctx.unsqueeze(0).expand(timesteps, -1)
        if ctx.ndim == 2:
            if ctx.shape[0] == timesteps or ctx.shape[0] == 1:
                return ctx.expand(timesteps, -1)
        if ctx.ndim == 3:
            if ctx.shape[0] == 1 and ctx.shape[1] == timesteps:
                return ctx.squeeze(0)
            if ctx.shape[1] == 1 and ctx.shape[0] == timesteps:
                return ctx.squeeze(1)
        raise ValueError(f"Unsupported context shape {ctx.shape} for timesteps={timesteps}")

    def _normalize_mask(self, mask, timesteps):
        """
        Normalize mask to [timesteps,1] boolean
        Handles [timesteps], [timesteps,1], [batch,timesteps], [batch,timesteps,1]
        """
        if mask.ndim == 1 and mask.shape[0] == timesteps:
            return mask.bool().unsqueeze(1)
        if mask.ndim == 2:
            if mask.shape == (timesteps, 1):
                return mask.bool()
            if mask.shape[0] == 1 and mask.shape[1] == timesteps:
                return mask.squeeze(0).unsqueeze(1).bool()
        if mask.ndim == 3:
            if mask.shape[0] == 1 and mask.shape[1] == timesteps:
                return mask.squeeze(0).bool()
        raise ValueError(f"Invalid mask shape {mask.shape} for timesteps={timesteps}")

    # ---------------- Properties ----------------
    @property
    def n_sequences(self):
        return len(self.sequences)

    @property
    def total_timesteps(self):
        return sum(self.lengths)

    @property
    def feature_dim(self):
        dims = {s.shape[-1] for s in self.sequences}
        if len(dims) != 1:
            raise ValueError("Feature dimension mismatch across sequences.")
        return dims.pop()

    @property
    def device(self):
        return self.sequences[0].device

    @property
    def dtype(self):
        return self.sequences[0].dtype

    def __getitem__(self, idx):
        def pick(lst):
            if lst is None:
                return None
            if isinstance(idx, slice):
                return lst[idx]
            return [lst[idx]]

        return SequenceSet(
            sequences=pick(self.sequences),
            lengths=pick(self.lengths),
            log_probs=pick(self.log_probs),
            contexts=pick(self.contexts),
            masks=pick(self.masks),
        )

    def to_tensor(self, key="sequences", pad_value=0.0):
        items = getattr(self, key)
        if items is None:
            raise ValueError(f"{key} is None.")

        batch_size = len(items)
        timesteps_max = max(t.shape[0] for t in items)
        feature_dim = items[0].shape[-1] if items[0].ndim > 1 else 1

        out = torch.full((batch_size, timesteps_max, feature_dim),
                         fill_value=pad_value,
                         dtype=self.dtype,
                         device=self.device)

        for i, t in enumerate(items):
            if t.ndim == 1:  # Single feature
                out[i, :t.shape[0], 0] = t
            else:
                out[i, :t.shape[0], :t.shape[1]] = t

        return out


@dataclass(frozen=False)
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

