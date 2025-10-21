import torch
import torch.nn.functional as F

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union


@dataclass(frozen=False)
class Observations:

    sequence: List[torch.Tensor]
    log_probs: Optional[List[torch.Tensor]] = None
    lengths: Optional[List[int]] = None
    context: Optional[List[torch.Tensor]] = None  # NEW: per-sequence context

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
            if not all(isinstance(c, torch.Tensor) for c in self.context):
                raise TypeError("All elements in `context` must be torch.Tensor.")

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

    @property
    def mean(self) -> torch.Tensor:
        return self.as_batch().mean(0)

    @property
    def std(self) -> torch.Tensor:
        return self.as_batch().std(0)

    @property
    def is_batch(self) -> bool:
        if not self.sequence:
            return False
        return self.n_sequences > 1 or self.sequence[0].shape[0] > 1

    def to(self, device: Union[str, torch.device], dtype: Optional[torch.dtype] = None) -> "Observations":
        seqs = [s.to(device=device, dtype=dtype or s.dtype) for s in self.sequence]
        logs = [l.to(device=device, dtype=dtype or l.dtype) for l in self.log_probs] if self.log_probs else None
        ctxs = [c.to(device=device, dtype=dtype or c.dtype) for c in self.context] if self.context else None
        return Observations(seqs, logs, self.lengths, ctxs)

    def detach(self) -> "Observations":
        seqs = [s.detach() for s in self.sequence]
        logs = [l.detach() for l in self.log_probs] if self.log_probs else None
        ctxs = [c.detach() for c in self.context] if self.context else None
        return Observations(seqs, logs, self.lengths, ctxs)

    def clone(self) -> "Observations":
        seqs = [s.clone() for s in self.sequence]
        logs = [l.clone() for l in self.log_probs] if self.log_probs else None
        ctxs = [c.clone() for c in self.context] if self.context else None
        return Observations(seqs, logs, self.lengths, ctxs)

    def __getitem__(self, idx: Union[int, slice]) -> "Observations":
        seqs = self.sequence[idx] if isinstance(idx, slice) else [self.sequence[idx]]
        logs = self.log_probs[idx] if self.log_probs else None
        lens = self.lengths[idx] if isinstance(idx, slice) else [self.lengths[idx]]
        ctxs = self.context[idx] if self.context else None
        return Observations(seqs, logs, lens, ctxs)

    def as_batch(self) -> torch.Tensor:
        return torch.cat(self.sequence, dim=0)

    def normalize(self, eps: float = 1e-6) -> "Observations":
        all_obs = self.as_batch()
        mean, std = all_obs.mean(0, keepdim=True), all_obs.std(0, keepdim=True).clamp_min(eps)
        normed = [(s - mean) / std for s in self.sequence]
        return Observations(normed, self.log_probs, self.lengths, self.context)

    def summary(self) -> str:
        return f"{self.n_sequences} seqs | total {self.total_length} steps | dim {self.feature_dim} | device {self.device}"

    # ----------------------------------------------------------------------
    # Unified padding / batching
    # ----------------------------------------------------------------------
    def to_batch(
        self,
        pad_value: float = 0.0,
        log_pad_value: float = -float("inf"),
        return_mask: bool = False,
        include_log_probs: bool = True,
        include_context: bool = False,  # NEW: return padded context
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]]:
        B, T, D = self.n_sequences, max(self.lengths), self.feature_dim
        device, dtype = self.device, self.dtype

        seq_batch = torch.full((B, T, D), pad_value, dtype=dtype, device=device)
        mask = torch.zeros((B, T), dtype=torch.bool, device=device)

        for i, seq in enumerate(self.sequence):
            L = seq.shape[0]
            seq_batch[i, :L] = seq
            mask[i, :L] = True

        log_batch = None
        if include_log_probs and self.log_probs is not None:
            K = self.log_probs[0].shape[1]
            log_batch = torch.full((B, T, K), log_pad_value, dtype=dtype, device=device)
            for i, lp in enumerate(self.log_probs):
                L = lp.shape[0]
                log_batch[i, :L] = lp

        ctx_batch = None
        if include_context and self.context is not None:
            H = self.context[0].shape[-1]
            ctx_batch = torch.zeros((B, T, H), dtype=dtype, device=device)
            for i, c in enumerate(self.context):
                L = c.shape[0]
                ctx_batch[i, :L] = c

        if return_mask:
            return seq_batch, mask, log_batch, ctx_batch
        return seq_batch if log_batch is None else (seq_batch, log_batch, ctx_batch)


@dataclass(frozen=False)
class ContextualVariables:
    """
    Container for static or time-dependent contextual variables (exogenous features).

    Supports:
    - Batch-aware operations for both static and time-dependent contexts
    - Device/dtype-aware operations
    - Padding, batching, and mask creation
    - Concatenation and normalization
    - Detachment and cloning
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
            raise ValueError("Time-dependent tensors must have shape [T, F] or [B, T, F].")
        if self.names and len(self.names) != self.n_context:
            raise ValueError("`names` length must match `n_context`.")
        devices = {x.device for x in self.X}
        if len(devices) > 1:
            raise ValueError("All tensors must be on the same device.")
        dtypes = {x.dtype for x in self.X}
        if len(dtypes) > 1:
            raise ValueError("All tensors must have the same dtype.")

    # ----------------------------------------------------------------------
    # Properties
    # ----------------------------------------------------------------------
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

    # ----------------------------------------------------------------------
    # Device / detachment / cloning
    # ----------------------------------------------------------------------
    def to(self, device: Union[str, torch.device], dtype: Optional[torch.dtype] = None) -> "ContextualVariables":
        X = [x.to(device=device, dtype=dtype or x.dtype) for x in self.X]
        return ContextualVariables(self.n_context, X, self.time_dependent, self.names)

    def detach(self) -> "ContextualVariables":
        X = [x.detach() for x in self.X]
        return ContextualVariables(self.n_context, X, self.time_dependent, self.names)

    def clone(self) -> "ContextualVariables":
        X = [x.clone() for x in self.X]
        return ContextualVariables(self.n_context, X, self.time_dependent, self.names)

    # ----------------------------------------------------------------------
    # Concatenation / normalization
    # ----------------------------------------------------------------------
    def cat(self, dim: int = -1, normalize: bool = False, eps: float = 1e-6) -> torch.Tensor:
        out = self.X[0] if len(self.X) == 1 else torch.cat(self.X, dim=dim)
        return F.layer_norm(out, out.shape[-1:], eps=eps) if normalize else out

    def normalize(self, eps: float = 1e-6) -> "ContextualVariables":
        all_features = self.cat(dim=-1)
        mean, std = all_features.mean(0, keepdim=True), all_features.std(0, keepdim=True).clamp_min(eps)
        normed = [(x - mean) / std for x in self.X]
        return ContextualVariables(self.n_context, normed, self.time_dependent, self.names)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.X[idx]

    # ----------------------------------------------------------------------
    # Padding / batching
    # ----------------------------------------------------------------------
    def pad_batch(
        self,
        pad_value: float = 0.0,
        return_mask: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Pad contexts into a batch tensor [B, T, F_total].
        Supports both static [B, F] and time-dependent [B, T, F] tensors.
        Returns mask [B, T] for time-dependent tensors.
        """
        if self.time_dependent:
            batch_size = 1 if self.X[0].ndim == 2 else self.X[0].shape[0]
            max_len = max(x.shape[1] if x.ndim == 3 else x.shape[0] for x in self.X)
            padded_list = []
            mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=self.device)

            for x in self.X:
                if x.ndim == 2:
                    x = x.unsqueeze(0)  # [T, F] -> [1, T, F]
                B, T, F = x.shape
                pad_tensor = torch.full((B, max_len, F), pad_value, dtype=x.dtype, device=x.device)
                pad_tensor[:, :T, :] = x
                padded_list.append(pad_tensor)
                mask[:, :T] |= True

            batch_tensor = torch.cat(padded_list, dim=-1)
            return (batch_tensor, mask) if return_mask else batch_tensor
        else:
            # static: stack along batch dim
            batch_tensor = torch.stack(self.X, dim=0)  # [B, F]
            return batch_tensor

    # ----------------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------------
    def summary(self) -> str:
        dep = "time-dependent" if self.time_dependent else "static"
        shapes = ", ".join(str(s) for s in self.shape)
        return f"{self.n_context} {dep} contexts [{shapes}] on {self.device}"
