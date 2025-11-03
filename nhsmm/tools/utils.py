import torch
from dataclasses import dataclass
from typing import List, Optional, Union, Tuple, Union


@dataclass(frozen=False)
class Observations:
    """Container for sequences, optional log-probs, and context vectors."""

    sequence: List[torch.Tensor]
    lengths: Optional[List[int]] = None
    log_probs: Optional[List[torch.Tensor]] = None
    context: Optional[List[Optional[torch.Tensor]]] = None
    mask: Optional[List[torch.Tensor]] = None  # new optional batch mask

    def __post_init__(self):
        if not self.sequence:
            raise ValueError("`sequence` cannot be empty.")
        if not all(isinstance(s, torch.Tensor) for s in self.sequence):
            raise TypeError("All elements in `sequence` must be torch.Tensor.")

        seq_lengths = self.lengths or [s.shape[0] for s in self.sequence]
        if any(s.shape[0] != l for s, l in zip(self.sequence, seq_lengths)):
            raise ValueError("Mismatch between sequence lengths and `lengths`.")
        object.__setattr__(self, "lengths", seq_lengths)

        if self.log_probs:
            if len(self.log_probs) != len(self.sequence):
                raise ValueError("`log_probs` length must match `sequence` length.")
            if not all(isinstance(lp, torch.Tensor) for lp in self.log_probs):
                raise TypeError("All elements in `log_probs` must be torch.Tensor.")

        if self.context:
            if len(self.context) != len(self.sequence):
                raise ValueError("`context` length must match `sequence` length.")
            if not all(c is None or isinstance(c, torch.Tensor) for c in self.context):
                raise TypeError("All elements in `context` must be torch.Tensor or None.")
        else:
            object.__setattr__(self, "context", [None] * len(self.sequence))

        if self.mask:
            if len(self.mask) != len(self.sequence):
                raise ValueError("`mask` length must match `sequence` length.")
            for m, s in zip(self.mask, self.sequence):
                if not isinstance(m, torch.Tensor):
                    raise TypeError("All elements in `mask` must be torch.Tensor.")
                if m.shape[0] != s.shape[0]:
                    raise ValueError("Each mask must match its sequence length.")
        else:
            object.__setattr__(
                self, "mask",
                [torch.ones(len_s, 1, dtype=torch.bool, device=self.sequence[0].device)
                 for len_s in seq_lengths]
            )

    # ---------------- Properties ----------------
    @property
    def n_sequences(self) -> int:
        return len(self.sequence)

    @property
    def total_length(self) -> int:
        return sum(self.lengths)

    @property
    def feature_dim(self) -> int:
        dims = {s.shape[-1] for s in self.sequence if s.ndim > 1}
        if not dims:
            return 1
        if len(dims) > 1:
            raise ValueError("Inconsistent feature dimensions across sequences.")
        return dims.pop()

    @property
    def device(self) -> torch.device:
        return self.sequence[0].device

    @property
    def dtype(self) -> torch.dtype:
        return self.sequence[0].dtype

    # ---------------- Device / Clone Ops ----------------
    def to(self, device: Union[str, torch.device], dtype: Optional[torch.dtype] = None) -> "Observations":
        dtype = dtype or self.dtype
        seqs = [s.to(device=device, dtype=dtype) for s in self.sequence]
        logs = [l.to(device=device, dtype=dtype) for l in self.log_probs] if self.log_probs else None
        ctxs = [c.to(device=device, dtype=dtype) if c is not None else None for c in self.context]
        masks = [m.to(device=device) for m in self.mask] if self.mask else None
        return Observations(seqs, self.lengths, logs, ctxs, masks)

    def detach(self) -> "Observations":
        seqs = [s.detach() for s in self.sequence]
        logs = [l.detach() for l in self.log_probs] if self.log_probs else None
        ctxs = [c.detach() if c is not None else None for c in self.context]
        masks = [m.clone() for m in self.mask] if self.mask else None
        return Observations(seqs, self.lengths, logs, ctxs, masks)

    def clone(self) -> "Observations":
        seqs = [s.clone() for s in self.sequence]
        logs = [l.clone() for l in self.log_probs] if self.log_probs else None
        ctxs = [c.clone() if c is not None else None for c in self.context]
        masks = [m.clone() for m in self.mask] if self.mask else None
        return Observations(seqs, self.lengths, logs, ctxs, masks)

    def __getitem__(self, idx: Union[int, slice]) -> "Observations":
        if isinstance(idx, int):
            seqs = [self.sequence[idx]]
            lens = [self.lengths[idx]]
            logs = [self.log_probs[idx]] if self.log_probs else None
            ctxs = [self.context[idx]] if self.context else None
            masks = [self.mask[idx]] if self.mask else None
        else:
            seqs = self.sequence[idx]
            lens = self.lengths[idx]
            logs = self.log_probs[idx] if self.log_probs else None
            ctxs = self.context[idx] if self.context else None
            masks = self.mask[idx] if self.mask else None
        return Observations(seqs, lens, logs, ctxs, masks)

    # ---------------- Normalization ----------------
    def normalize(self, mask: Optional[List[torch.Tensor]] = None, eps: float = 1e-6) -> "Observations":
        """Normalize sequences per feature, using internal or external masks."""
        normed = []
        mask_list = mask or self.mask
        for s, m in zip(self.sequence, mask_list):
            m = m.to(s.device, s.dtype)
            m = m.unsqueeze(-1) if m.ndim == 1 else m
            m_sum = m.sum(0).clamp_min(1.0)
            mean = (s * m).sum(0) / m_sum
            var = ((s - mean) ** 2 * m).sum(0) / m_sum
            std = var.sqrt().clamp_min(eps)
            normed.append(((s - mean) / std) * m + (1 - m) * s)  # keep padded entries intact
        return Observations(normed, self.lengths, self.log_probs, self.context, mask_list)


@dataclass(frozen=False)
class ContextualVariables:
    """Container for multiple context tensors with optional names and time-dependence."""

    n_context: int
    X: List[torch.Tensor]
    time_dependent: bool = False
    names: Optional[List[str]] = None

    # internal lightweight cache for concatenated context
    _cache: Optional[dict] = None

    def __post_init__(self):
        if not self.X:
            raise ValueError("`X` cannot be empty.")
        if len(self.X) != self.n_context:
            raise ValueError(f"Expected {self.n_context} context tensors, got {len(self.X)}.")
        if self.names and len(self.names) != self.n_context:
            raise ValueError("`names` length must match `n_context`.")

        devices = {x.device for x in self.X}
        if len(devices) > 1:
            raise ValueError("All context tensors must be on the same device.")
        dtypes = {x.dtype for x in self.X}
        if len(dtypes) > 1:
            raise ValueError("All context tensors must have the same dtype.")

        if self._cache is None:
            self._cache = {}

    @property
    def shape(self) -> Tuple[torch.Size, ...]:
        return tuple(x.shape for x in self.X)

    @property
    def device(self) -> torch.device:
        return self.X[0].device

    @property
    def dtype(self) -> torch.dtype:
        return self.X[0].dtype

    @property
    def feature_dim(self) -> int:
        dims = {x.shape[-1] for x in self.X if x.ndim >= 2}
        if not dims:
            return 1
        if len(dims) > 1:
            raise ValueError("Inconsistent feature dimensions across contexts.")
        return dims.pop()

    def _align_time(self, ref_len: int) -> List[torch.Tensor]:
        """Broadcast non-temporal contexts to match temporal length."""
        aligned = []
        for x in self.X:
            if x.ndim == 1 or (x.ndim == 2 and x.shape[0] != ref_len):
                aligned.append(x.expand(ref_len, -1))
            else:
                aligned.append(x)
        return aligned

    def cat(self, dim: int = -1, normalize: bool = False, eps: float = 1e-6) -> torch.Tensor:
        """Concatenate all context tensors along `dim`, optional normalization."""
        cache_key = (dim, normalize)
        if cache_key in self._cache:
            return self._cache[cache_key]

        X = self.X
        if self.time_dependent:
            ref_len = max(x.shape[0] for x in X if x.ndim >= 2)
            X = self._align_time(ref_len)

        out = X[0] if len(X) == 1 else torch.cat(X, dim=dim)
        if normalize:
            mean, std = out.mean(0, keepdim=True), out.std(0, keepdim=True).clamp_min(eps)
            out = (out - mean) / std

        self._cache[cache_key] = out
        return out

    def to(self, device: Union[str, torch.device], dtype: Optional[torch.dtype] = None) -> "ContextualVariables":
        dtype = dtype or self.dtype
        X = [x.to(device=device, dtype=dtype) for x in self.X]
        return ContextualVariables(self.n_context, X, self.time_dependent, self.names)

    def detach(self) -> "ContextualVariables":
        X = [x.detach() for x in self.X]
        return ContextualVariables(self.n_context, X, self.time_dependent, self.names)

    def clone(self) -> "ContextualVariables":
        X = [x.clone() for x in self.X]
        return ContextualVariables(self.n_context, X, self.time_dependent, self.names)

    def __getitem__(self, idx: Union[int, slice, torch.Tensor]) -> "ContextualVariables":
        """Supports indexing/slicing across all context tensors."""
        X = [x[idx] for x in self.X]
        return ContextualVariables(len(X), X, self.time_dependent, self.names)

    def __repr__(self) -> str:
        names = self.names or [f"ctx{i}" for i in range(self.n_context)]
        s = ", ".join(f"{n}:{tuple(x.shape)}" for n, x in zip(names, self.X))
        td = "time-dep" if self.time_dependent else "static"
        return f"<ContextualVariables[{td}] {s}>"

