import torch
from dataclasses import dataclass
from typing import List, Optional, Union, Tuple


@dataclass(frozen=False)
class Observations:

    sequence: List[torch.Tensor]
    lengths: Optional[List[int]] = None
    log_probs: Optional[List[torch.Tensor]] = None
    context: Optional[List[Optional[torch.Tensor]]] = None

    def __post_init__(self):
        if not self.sequence:
            raise ValueError("`sequence` cannot be empty.")
        if not all(isinstance(s, torch.Tensor) for s in self.sequence):
            raise TypeError("All elements in `sequence` must be torch.Tensor.")
        seq_lengths = self.lengths or [s.shape[0] for s in self.sequence]
        if any(s.shape[0] != l for s, l in zip(self.sequence, seq_lengths)):
            raise ValueError("Mismatch between sequence lengths and `lengths`.")
        object.__setattr__(self, "lengths", seq_lengths)

        if self.log_probs is not None:
            if len(self.log_probs) != len(self.sequence):
                raise ValueError("`log_probs` length must match `sequence` length.")
            if not all(isinstance(lp, torch.Tensor) for lp in self.log_probs):
                raise TypeError("All elements in `log_probs` must be torch.Tensor.")

        if self.context is not None:
            if len(self.context) != len(self.sequence):
                raise ValueError("`context` length must match `sequence` length.")
            if not all(c is None or isinstance(c, torch.Tensor) for c in self.context):
                raise TypeError("All elements in `context` must be torch.Tensor or None.")
        else:
            object.__setattr__(self, "context", [None] * len(self.sequence))

    @property
    def n_sequences(self) -> int:
        return len(self.sequence)

    @property
    def total_length(self) -> int:
        return sum(self.lengths)

    @property
    def feature_dim(self) -> int:
        dims = {s.shape[-1] for s in self.sequence if s.ndim > 1}
        if len(dims) > 1:
            raise ValueError("Inconsistent feature dimensions across sequences.")
        return dims.pop() if dims else 1

    @property
    def device(self) -> torch.device:
        return self.sequence[0].device

    @property
    def dtype(self) -> torch.dtype:
        return self.sequence[0].dtype

    def to(self, device: Union[str, torch.device], dtype: Optional[torch.dtype] = None) -> "Observations":
        seqs = [s.to(device=device, dtype=dtype or s.dtype) for s in self.sequence]
        logs = [l.to(device=device, dtype=dtype or l.dtype) for l in self.log_probs] if self.log_probs else None
        ctxs = [c.to(device=device, dtype=dtype or c.dtype) if c is not None else None for c in self.context]
        return Observations(seqs, self.lengths, logs, ctxs)

    def detach(self) -> "Observations":
        seqs = [s.detach() for s in self.sequence]
        logs = [l.detach() for l in self.log_probs] if self.log_probs else None
        ctxs = [c.detach() if c is not None else None for c in self.context]
        return Observations(seqs, self.lengths, logs, ctxs)

    def clone(self) -> "Observations":
        seqs = [s.clone() for s in self.sequence]
        logs = [l.clone() for l in self.log_probs] if self.log_probs else None
        ctxs = [c.clone() if c is not None else None for c in self.context]
        return Observations(seqs, self.lengths, logs, ctxs)

    def __getitem__(self, idx: int) -> "Observations":
        # single-sample only: always returns an Observations with one sequence
        seqs = [self.sequence[idx]]
        lens = [self.lengths[idx]]
        logs = [self.log_probs[idx]] if self.log_probs else None
        ctxs = [self.context[idx]] if self.context else None
        return Observations(seqs, lens, logs, ctxs)

    def normalize(self, eps: float = 1e-6) -> "Observations":
        normed = []
        for s in self.sequence:
            mean, std = s.mean(0, keepdim=True), s.std(0, keepdim=True).clamp_min(eps)
            normed.append((s - mean) / std)
        return Observations(normed, self.lengths, self.log_probs, self.context)


@dataclass(frozen=False)
class ContextualVariables:

    n_context: int
    X: List[torch.Tensor]
    time_dependent: bool = False
    names: Optional[List[str]] = None

    def __post_init__(self):
        if not self.X:
            raise ValueError("`X` cannot be empty.")
        if len(self.X) != self.n_context:
            raise ValueError(f"Expected {self.n_context} context tensors, got {len(self.X)}.")
        if self.names and len(self.names) != self.n_context:
            raise ValueError("`names` length must match `n_context`.")
        devices = {x.device for x in self.X}
        if len(devices) > 1:
            raise ValueError("All tensors must be on the same device.")
        dtypes = {x.dtype for x in self.X}
        if len(dtypes) > 1:
            raise ValueError("All tensors must have the same dtype.")

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
        if len(dims) > 1:
            raise ValueError("Inconsistent feature dimensions across contexts.")
        return dims.pop() if dims else 1

    def to(self, device: Union[str, torch.device], dtype: Optional[torch.dtype] = None) -> "ContextualVariables":
        X = [x.to(device=device, dtype=dtype or x.dtype) for x in self.X]
        return ContextualVariables(self.n_context, X, self.time_dependent, self.names)

    def detach(self) -> "ContextualVariables":
        X = [x.detach() for x in self.X]
        return ContextualVariables(self.n_context, X, self.time_dependent, self.names)

    def clone(self) -> "ContextualVariables":
        X = [x.clone() for x in self.X]
        return ContextualVariables(self.n_context, X, self.time_dependent, self.names)

    def cat(self, dim: int = -1, normalize: bool = False, eps: float = 1e-6) -> torch.Tensor:
        out = self.X[0] if len(self.X) == 1 else torch.cat(self.X, dim=dim)
        if normalize:
            mean, std = out.mean(0, keepdim=True), out.std(0, keepdim=True).clamp_min(eps)
            out = (out - mean) / std
        return out

    def normalize(self, eps: float = 1e-6) -> "ContextualVariables":
        normed = []
        for x in self.X:
            mean, std = x.mean(0, keepdim=True), x.std(0, keepdim=True).clamp_min(eps)
            normed.append((x - mean) / std)
        return ContextualVariables(self.n_context, normed, self.time_dependent, self.names)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # single-sample only: returns one context tensor
        return self.X[idx]
