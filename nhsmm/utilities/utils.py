from typing import Tuple, List, Optional, Union
from dataclasses import dataclass, field
import torch
import torch.nn.functional as F


from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch


@dataclass(frozen=False)
class Observations:
    """Container for one or more observation sequences with optional log-probabilities."""

    sequence: List[torch.Tensor]
    log_probs: Optional[List[torch.Tensor]] = None
    lengths: Optional[List[int]] = None

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

    # ----------------------------------------------------------------------
    # Properties
    # ----------------------------------------------------------------------
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

    # ----------------------------------------------------------------------
    # Core operations
    # ----------------------------------------------------------------------
    def to(self, device: Union[str, torch.device]) -> "Observations":
        """Move all tensors to specified device."""
        seqs = [s.to(device) for s in self.sequence]
        logs = [l.to(device) for l in self.log_probs] if self.log_probs else None
        return Observations(seqs, logs, self.lengths)

    def pad_sequences(
        self, pad_value: float = 0.0, return_mask: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Pad variable-length sequences into a dense tensor [B, T, D]."""
        max_len = max(self.lengths)
        feat_dim = self.feature_dim
        dtype = self.sequence[0].dtype
        device = self.device

        batch = torch.full((self.n_sequences, max_len, feat_dim), pad_value, dtype=dtype, device=device)
        mask = torch.zeros((self.n_sequences, max_len), dtype=torch.bool, device=device)

        for i, seq in enumerate(self.sequence):
            L = seq.shape[0]
            batch[i, :L] = seq
            mask[i, :L] = True

        return (batch, mask) if return_mask else batch

    def as_batch(self) -> torch.Tensor:
        """Concatenate all sequences along time dimension."""
        return torch.cat(self.sequence, dim=0)

    def detach(self) -> "Observations":
        """Return a detached copy (no autograd tracking)."""
        seqs = [s.detach() for s in self.sequence]
        logs = [l.detach() for l in self.log_probs] if self.log_probs else None
        return Observations(seqs, logs, self.lengths)

    def normalize(self, eps: float = 1e-6) -> "Observations":
        """Apply global z-score normalization across all observations."""
        all_obs = self.as_batch()
        mean, std = all_obs.mean(0, keepdim=True), all_obs.std(0, keepdim=True).clamp_min(eps)
        normed = [(s - mean) / std for s in self.sequence]
        return Observations(normed, self.log_probs, self.lengths)

    # ----------------------------------------------------------------------
    # Utilities
    # ----------------------------------------------------------------------
    def summary(self) -> str:
        """Compact summary of observation statistics."""
        total = self.total_length
        feats = self.feature_dim
        dev = str(self.device)
        return f"{self.n_sequences} seqs | total {total} steps | dim {feats} | device {dev}"


@dataclass(frozen=True)
class ContextualVariables:
    """
    Immutable container for contextual (exogenous) variables used in hierarchical
    or neural HSMM/HMM models.

    Supports both static and time-dependent context features, with device-aware
    concatenation and optional normalization.
    """

    n_context: int
    X: Tuple[torch.Tensor, ...]
    time_dependent: bool = field(default=False)
    names: Optional[List[str]] = None

    def __post_init__(self):
        if not self.X:
            raise ValueError("`X` cannot be empty.")
        if len(self.X) != self.n_context:
            raise ValueError(f"Expected {self.n_context} context tensors, got {len(self.X)}.")
        if self.time_dependent and any(x.ndim < 2 for x in self.X):
            raise ValueError("Time-dependent context tensors must have shape (T, F) or (B, T, F).")
        if self.names and len(self.names) != self.n_context:
            raise ValueError("`names` length must match `n_context`.")
        if len({x.device for x in self.X}) > 1:
            raise ValueError("All context tensors must reside on the same device.")
        if len({x.dtype for x in self.X}) > 1:
            raise ValueError("All context tensors must share the same dtype.")

    # ----------------------------------------------------------------------
    # Properties
    # ----------------------------------------------------------------------
    @property
    def shape(self) -> Tuple[torch.Size, ...]:
        """Shapes of each context tensor."""
        return tuple(x.shape for x in self.X)

    @property
    def device(self) -> torch.device:
        """Device of the stored tensors."""
        return self.X[0].device

    @property
    def dtype(self) -> torch.dtype:
        """Data type of the stored tensors."""
        return self.X[0].dtype

    # ----------------------------------------------------------------------
    # Core methods
    # ----------------------------------------------------------------------
    def cat(self, dim: int = -1, normalize: bool = False, eps: float = 1e-6) -> torch.Tensor:
        """
        Concatenate all context tensors along a given dimension, optionally normalized.

        Args:
            dim: Dimension to concatenate on.
            normalize: Apply layer normalization.
            eps: Small epsilon to prevent numerical instability.

        Returns:
            Concatenated tensor of contextual features.
        """
        if len(self.X) == 1:
            out = self.X[0]
        else:
            out = torch.cat(self.X, dim=dim)
        return F.layer_norm(out, out.shape[-1:], eps=eps) if normalize else out

    def to(self, device: Union[str, torch.device]) -> "ContextualVariables":
        """Return a device-moved copy."""
        X = tuple(x.to(device) for x in self.X)
        return ContextualVariables(
            n_context=self.n_context, X=X,
            time_dependent=self.time_dependent, names=self.names
        )

    def detach(self) -> "ContextualVariables":
        """Return a detached copy (no autograd tracking)."""
        X = tuple(x.detach() for x in self.X)
        return ContextualVariables(
            n_context=self.n_context, X=X,
            time_dependent=self.time_dependent, names=self.names
        )

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Index into the context tuple."""
        return self.X[idx]

    # ----------------------------------------------------------------------
    # Utility
    # ----------------------------------------------------------------------
    def summary(self) -> str:
        """Compact string summary."""
        dep = "time-dependent" if self.time_dependent else "static"
        shape_str = ", ".join(str(s) for s in self.shape)
        return f"{self.n_context} {dep} contexts [{shape_str}] on {self.device}"

