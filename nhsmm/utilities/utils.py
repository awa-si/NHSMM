# utilities/utils.py
from typing import Tuple, List, Optional, Union
from dataclasses import dataclass, field

import torch.nn.functional as F
import torch


@dataclass(frozen=False)
class Observations:
    """
    Structured container for one or more observation sequences and their metadata.

    This class standardizes data flow between emission models, encoders,
    and the neural/HSMM inference routines.

    Attributes
    ----------
    sequence : list[Tensor]
        List of observation sequences. Each element has shape (T_i, F),
        where T_i is the number of time steps and F is the feature dimension.
    log_probs : list[Tensor] | None
        Optional list of log-probabilities associated with each observation
        (e.g., precomputed emission scores).
    lengths : list[int] | None
        Length of each observation sequence. If None, inferred from `sequence`.

    Notes
    -----
    * The class ensures alignment of sequence lengths and type safety.
    * It supports conversion, normalization, and batching operations.
    * Designed for variable-length sequential data common in HSMMs.
    """

    sequence: List[torch.Tensor]
    log_probs: Optional[List[torch.Tensor]] = None
    lengths: Optional[List[int]] = None

    def __post_init__(self):
        if not self.sequence:
            raise ValueError("`sequence` cannot be empty")

        if not all(isinstance(s, torch.Tensor) for s in self.sequence):
            raise TypeError("All elements in `sequence` must be torch.Tensor")

        if self.log_probs is not None:
            if len(self.log_probs) != len(self.sequence):
                raise ValueError("`log_probs` length must match `sequence` length")
            if not all(isinstance(lp, torch.Tensor) for lp in self.log_probs):
                raise TypeError("All elements in `log_probs` must be torch.Tensor")

        lengths = self.lengths or [s.shape[0] for s in self.sequence]
        if any(s.shape[0] != l for s, l in zip(self.sequence, lengths)):
            raise ValueError("Mismatch between sequence lengths and provided `lengths`")
        object.__setattr__(self, "lengths", lengths)

    # ---------------------------------------------------------------------
    # Properties
    # ---------------------------------------------------------------------

    @property
    def n_sequences(self) -> int:
        """Number of independent sequences."""
        return len(self.sequence)

    @property
    def total_length(self) -> int:
        """Sum of all sequence lengths (Σ_i T_i)."""
        return sum(self.lengths)

    @property
    def feature_dim(self) -> Optional[int]:
        """Return feature dimension if consistent across all sequences."""
        dims = {s.shape[-1] for s in self.sequence if s.ndim > 1}
        return dims.pop() if len(dims) == 1 else None

    # ---------------------------------------------------------------------
    # Core utilities
    # ---------------------------------------------------------------------

    def to(self, device: Union[str, torch.device]) -> "Observations":
        """
        Move all internal tensors to a specified device (CPU, GPU, etc.).
        """
        seqs = [s.to(device) for s in self.sequence]
        logs = [l.to(device) for l in self.log_probs] if self.log_probs else None
        return Observations(sequence=seqs, log_probs=logs, lengths=self.lengths)

    def pad_sequences(self, pad_value: float = 0.0, return_mask: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Return a zero-padded batch tensor `(B, T_max, F)` suitable for parallel processing.

        Parameters
        ----------
        pad_value : float
            Value used to fill padded positions.
        return_mask : bool
            If True, also returns a boolean mask `(B, T_max)` where True indicates valid positions.

        Returns
        -------
        padded : Tensor
            Batched padded tensor.
        mask : Tensor, optional
            Boolean mask (returned if `return_mask=True`).
        """
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
        """
        Concatenate all sequences into a single tensor of shape `(ΣT_i, F)`.

        Useful for computing global statistics (e.g., normalization).
        """
        return torch.cat(self.sequence, dim=0)

    def detach(self) -> "Observations":
        """
        Return a detached copy of this object (no gradients retained).
        """
        seqs = [s.detach() for s in self.sequence]
        logs = [l.detach() for l in self.log_probs] if self.log_probs else None
        return Observations(sequence=seqs, log_probs=logs, lengths=self.lengths)

    def normalize(self) -> "Observations":
        """
        Apply feature-wise standardization across all sequences.

        Returns
        -------
        Observations
            New instance with normalized data (mean 0, std 1).
        """
        all_obs = self.as_batch()
        mean, std = all_obs.mean(0, keepdim=True), all_obs.std(0, keepdim=True).clamp_min(1e-6)
        normed = [(s - mean) / std for s in self.sequence]
        return Observations(sequence=normed, log_probs=self.log_probs, lengths=self.lengths)


@dataclass(frozen=True)
class ContextualVariables:
    """
    Container for contextual (exogenous) features supplied to a neural or hierarchical HSMM.

    These may include:
      * latent embeddings from a neural encoder
      * control variables or regime indicators
      * time-varying external covariates

    Attributes
    ----------
    n_context : int
        Number of context tensors provided.
    X : tuple[Tensor, ...]
        Context tensors, each possibly of shape `(T, F_ctx)` or `(F_ctx,)`
        depending on whether they are time-dependent.
    time_dependent : bool
        Indicates whether the context changes over time.
    names : list[str] | None
        Optional human-readable identifiers for each context tensor.

    Notes
    -----
    * This structure allows modular composition of multiple context streams.
    * Supports dynamic concatenation, normalization, and device handling.
    """

    n_context: int
    X: Tuple[torch.Tensor, ...]
    time_dependent: bool = field(default=False)
    names: Optional[List[str]] = None

    def __post_init__(self):
        if not self.X:
            raise ValueError("`X` cannot be empty")
        if not all(isinstance(x, torch.Tensor) for x in self.X):
            raise TypeError("All elements in `X` must be torch.Tensor")
        if len(self.X) != self.n_context:
            raise ValueError(f"`n_context` ({self.n_context}) must match number of tensors in `X` ({len(self.X)})")
        if self.time_dependent and any(x.ndim < 2 for x in self.X):
            raise ValueError("Time-dependent context must have at least 2D tensors (T, F)")
        if self.names and len(self.names) != self.n_context:
            raise ValueError("`names` must have same length as `X`")

    # ---------------------------------------------------------------------
    # Properties
    # ---------------------------------------------------------------------

    @property
    def shape(self) -> Tuple[torch.Size, ...]:
        """Return tuple of shapes for each context tensor."""
        return tuple(x.shape for x in self.X)

    @property
    def device(self) -> torch.device:
        """Return the device where the context tensors reside."""
        return self.X[0].device

    # ---------------------------------------------------------------------
    # Core utilities
    # ---------------------------------------------------------------------

    def cat(self, dim: int = -1, normalize: bool = False) -> torch.Tensor:
        """
        Concatenate all context tensors along a specified dimension.

        Parameters
        ----------
        dim : int, default=-1
            Dimension along which tensors are concatenated.
        normalize : bool, default=False
            If True, apply LayerNorm to the concatenated tensor.

        Returns
        -------
        Tensor
            Combined context tensor suitable for projection layers.
        """
        cat = torch.cat(self.X, dim=dim)
        if normalize:
            cat = F.layer_norm(cat, cat.shape[-1:])
        return cat

    def to(self, device: Union[str, torch.device]) -> "ContextualVariables":
        """
        Move all contained tensors to the specified device.
        """
        X = tuple(x.to(device) for x in self.X)
        return ContextualVariables(n_context=self.n_context, X=X, time_dependent=self.time_dependent, names=self.names)

    def detach(self) -> "ContextualVariables":
        """
        Return a detached copy (without gradient tracking).
        """
        X = tuple(x.detach() for x in self.X)
        return ContextualVariables(n_context=self.n_context, X=X, time_dependent=self.time_dependent, names=self.names)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Direct tensor access by index (useful for named-context dispatch)."""
        return self.X[idx]
