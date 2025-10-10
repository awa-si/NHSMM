from typing import Tuple, List, Optional, Union
from dataclasses import dataclass, field
import torch
import torch.nn.functional as F


@dataclass(frozen=False)
class Observations:
    """Container for one or more observation sequences with optional precomputed log-probs."""
    sequence: List[torch.Tensor]
    log_probs: Optional[List[torch.Tensor]] = None
    lengths: Optional[List[int]] = None

    def __post_init__(self):
        if not self.sequence:
            raise ValueError("`sequence` cannot be empty")
        if not all(isinstance(s, torch.Tensor) for s in self.sequence):
            raise TypeError("All elements in `sequence` must be torch.Tensor")
        lengths = self.lengths or [s.shape[0] for s in self.sequence]
        if any(s.shape[0] != l for s, l in zip(self.sequence, lengths)):
            raise ValueError("Mismatch between sequence lengths and `lengths`")
        object.__setattr__(self, "lengths", lengths)

        if self.log_probs is not None:
            if len(self.log_probs) != len(self.sequence):
                raise ValueError("`log_probs` length must match `sequence` length")
            if not all(isinstance(lp, torch.Tensor) for lp in self.log_probs):
                raise TypeError("All elements in `log_probs` must be torch.Tensor")

    @property
    def n_sequences(self) -> int:
        return len(self.sequence)

    @property
    def total_length(self) -> int:
        return sum(self.lengths)

    @property
    def feature_dim(self) -> Optional[int]:
        dims = {s.shape[-1] for s in self.sequence if s.ndim > 1}
        return dims.pop() if len(dims) == 1 else None

    def to(self, device: Union[str, torch.device]) -> "Observations":
        seqs = [s.to(device) for s in self.sequence]
        logs = [l.to(device) for l in self.log_probs] if self.log_probs else None
        return Observations(sequence=seqs, log_probs=logs, lengths=self.lengths)

    def pad_sequences(self, pad_value: float = 0.0, return_mask: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        max_len = max(self.lengths)
        feat_dim = self.feature_dim or self.sequence[0].shape[-1]
        batch = torch.full((self.n_sequences, max_len, feat_dim), pad_value, dtype=self.sequence[0].dtype)
        mask = torch.zeros((self.n_sequences, max_len), dtype=torch.bool)
        for i, seq in enumerate(self.sequence):
            L = seq.shape[0]
            batch[i, :L] = seq
            mask[i, :L] = True
        return (batch, mask) if return_mask else batch

    def as_batch(self) -> torch.Tensor:
        return torch.cat(self.sequence, dim=0)

    def detach(self) -> "Observations":
        seqs = [s.detach() for s in self.sequence]
        logs = [l.detach() for l in self.log_probs] if self.log_probs else None
        return Observations(sequence=seqs, log_probs=logs, lengths=self.lengths)

    def normalize(self) -> "Observations":
        all_obs = self.as_batch()
        mean, std = all_obs.mean(0, keepdim=True), all_obs.std(0, keepdim=True).clamp_min(1e-6)
        normed = [(s - mean) / std for s in self.sequence]
        return Observations(sequence=normed, log_probs=self.log_probs, lengths=self.lengths)


@dataclass(frozen=True)
class ContextualVariables:
    """
    Container for contextual (exogenous) features for neural or hierarchical models.
    """
    n_context: int
    X: Tuple[torch.Tensor, ...]
    time_dependent: bool = field(default=False)
    names: Optional[List[str]] = None

    def __post_init__(self):
        if not self.X:
            raise ValueError("`X` cannot be empty")
        if len(self.X) != self.n_context:
            raise ValueError(f"`n_context` ({self.n_context}) != number of tensors in `X` ({len(self.X)})")
        if self.time_dependent and any(x.ndim < 2 for x in self.X):
            raise ValueError("Time-dependent context tensors must be at least 2D (T, F)")
        if self.names and len(self.names) != self.n_context:
            raise ValueError("`names` length must match `n_context`")

    @property
    def shape(self) -> Tuple[torch.Size, ...]:
        return tuple(x.shape for x in self.X)

    @property
    def device(self) -> torch.device:
        return self.X[0].device

    def cat(self, dim: int = -1, normalize: bool = False) -> torch.Tensor:
        cat = torch.cat(self.X, dim=dim)
        if normalize:
            cat = F.layer_norm(cat, cat.shape[-1:])
        return cat

    def to(self, device: Union[str, torch.device]) -> "ContextualVariables":
        X = tuple(x.to(device) for x in self.X)
        return ContextualVariables(n_context=self.n_context, X=X, time_dependent=self.time_dependent, names=self.names)

    def detach(self) -> "ContextualVariables":
        X = tuple(x.detach() for x in self.X)
        return ContextualVariables(n_context=self.n_context, X=X, time_dependent=self.time_dependent, names=self.names)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.X[idx]
