import torch
import torch.nn.functional as F

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union


@dataclass(frozen=False)
class Observations:
    """
    Container for one or more observation sequences.

    Supports variable-length sequences, optional log-probabilities, device-aware operations,
    padding, batching, concatenation, normalization, and detachment.
    """

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

    @property
    def dtype(self) -> torch.dtype:
        return self.sequence[0].dtype

    def to(self, device: Union[str, torch.device]) -> "Observations":
        seqs = [s.to(device) for s in self.sequence]
        logs = [l.to(device) for l in self.log_probs] if self.log_probs else None
        return Observations(seqs, logs, self.lengths)

    def detach(self) -> "Observations":
        seqs = [s.detach() for s in self.sequence]
        logs = [l.detach() for l in self.log_probs] if self.log_probs else None
        return Observations(seqs, logs, self.lengths)

    def pad_sequences(
        self, pad_value: float = 0.0, return_mask: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Pad variable-length sequences into a dense tensor [B, T, D].
        Returns an optional mask [B, T].
        """
        B = self.n_sequences
        T = max(self.lengths)
        D = self.feature_dim
        device = self.device
        dtype = self.dtype

        batch = torch.full((B, T, D), pad_value, dtype=dtype, device=device)
        mask = torch.zeros((B, T), dtype=torch.bool, device=device)

        for i, seq in enumerate(self.sequence):
            L = seq.shape[0]
            batch[i, :L] = seq
            mask[i, :L] = True

        return (batch, mask) if return_mask else batch

    def as_batch(self) -> torch.Tensor:
        """Concatenate all sequences along the time dimension."""
        return torch.cat(self.sequence, dim=0)

    def normalize(self, eps: float = 1e-6) -> "Observations":
        """Apply global z-score normalization across all observations."""
        all_obs = self.as_batch()
        mean, std = all_obs.mean(0, keepdim=True), all_obs.std(0, keepdim=True).clamp_min(eps)
        normed = [(s - mean) / std for s in self.sequence]
        return Observations(normed, self.log_probs, self.lengths)

    def summary(self) -> str:
        return f"{self.n_sequences} seqs | total {self.total_length} steps | dim {self.feature_dim} | device {self.device}"


@dataclass(frozen=False)
class ContextualVariables:
    """
    Container for contextual variables (exogenous features) for HSMM/HMM models.

    Supports time-dependent or static contexts, padding, batching, concatenation,
    normalization, detachment, and device-aware operations.
    """

    n_context: int
    X: List[torch.Tensor]
    time_dependent: bool = False
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

    @property
    def shape(self) -> Tuple[torch.Size, ...]:
        return tuple(x.shape for x in self.X)

    @property
    def device(self) -> torch.device:
        return self.X[0].device

    @property
    def dtype(self) -> torch.dtype:
        return self.X[0].dtype

    def to(self, device: Union[str, torch.device]) -> "ContextualVariables":
        X = [x.to(device) for x in self.X]
        return ContextualVariables(self.n_context, X, self.time_dependent, self.names)

    def detach(self) -> "ContextualVariables":
        X = [x.detach() for x in self.X]
        return ContextualVariables(self.n_context, X, self.time_dependent, self.names)

    def cat(self, dim: int = -1, normalize: bool = False, eps: float = 1e-6) -> torch.Tensor:
        """Concatenate context tensors along a given dimension with optional layer normalization."""
        out = self.X[0] if len(self.X) == 1 else torch.cat(self.X, dim=dim)
        return F.layer_norm(out, out.shape[-1:], eps=eps) if normalize else out

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.X[idx]

    def pad_sequences(
        self, pad_value: float = 0.0, return_mask: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Pad time-dependent context tensors along time dimension into a batch tensor [B, T, F].
        Returns optional mask [B, T].
        """
        if not self.time_dependent:
            raise ValueError("pad_sequences only applies to time-dependent context tensors.")

        # Determine batch size and max length
        batch_size = 1 if self.X[0].ndim == 2 else self.X[0].shape[0]
        max_len = max(x.shape[1] if x.ndim == 3 else x.shape[0] for x in self.X)

        padded_list = []
        for x in self.X:
            if x.ndim == 2:  # [T, F] -> [1, T, F]
                x = x.unsqueeze(0)
            B, T, F = x.shape
            pad_tensor = torch.full((B, max_len, F), pad_value, dtype=x.dtype, device=x.device)
            pad_tensor[:, :T, :] = x
            padded_list.append(pad_tensor)

        batch = torch.cat(padded_list, dim=-1)
        mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=self.device)
        for x in self.X:
            L = x.shape[1] if x.ndim == 3 else x.shape[0]
            mask[:, :L] = True

        return (batch, mask) if return_mask else batch

    def summary(self) -> str:
        dep = "time-dependent" if self.time_dependent else "static"
        shape_str = ", ".join(str(s) for s in self.shape)
        return f"{self.n_context} {dep} contexts [{shape_str}] on {self.device}"
