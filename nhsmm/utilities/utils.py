import torch
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union


# -----------------------------
# Observations
# -----------------------------
@dataclass(frozen=False)
class Observations:
    sequence: List[torch.Tensor]
    lengths: Optional[List[int]] = None
    context: Optional[List[Optional[torch.Tensor]]] = None
    log_probs: Optional[List[torch.Tensor]] = None

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
        return Observations(seqs, self.lengths, ctxs, logs)

    def detach(self) -> "Observations":
        seqs = [s.detach() for s in self.sequence]
        logs = [l.detach() for l in self.log_probs] if self.log_probs else None
        ctxs = [c.detach() if c is not None else None for c in self.context]
        return Observations(seqs, self.lengths, ctxs, logs)

    def clone(self) -> "Observations":
        seqs = [s.clone() for s in self.sequence]
        logs = [l.clone() for l in self.log_probs] if self.log_probs else None
        ctxs = [c.clone() if c is not None else None for c in self.context]
        return Observations(seqs, self.lengths, ctxs, logs)

    def __getitem__(self, idx: Union[int, slice]) -> "Observations":
        seqs = self.sequence[idx] if isinstance(idx, slice) else [self.sequence[idx]]
        logs = self.log_probs[idx] if self.log_probs and isinstance(idx, slice) else \
               ([self.log_probs[idx]] if self.log_probs else None)
        lens = self.lengths[idx] if isinstance(idx, slice) else [self.lengths[idx]]
        ctxs = self.context[idx] if self.context and isinstance(idx, slice) else \
               ([self.context[idx]] if self.context else None)
        return Observations(seqs, lens, ctxs, logs)

    def as_batch(self, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None) -> torch.Tensor:
        seqs = [s.to(dtype=dtype or s.dtype, device=device or s.device) for s in self.sequence]
        return torch.cat(seqs, dim=0)

    def normalize(self, eps: float = 1e-6) -> "Observations":
        all_obs = self.as_batch()
        mean, std = all_obs.mean(0, keepdim=True), all_obs.std(0, keepdim=True).clamp_min(eps)
        normed = [(s - mean) / std for s in self.sequence]
        return Observations(normed, self.lengths, self.context, self.log_probs)

    # -----------------------------
    # Vectorized batching
    # -----------------------------
    def to_batch(
        self,
        pad_value: float = 0.0,
        log_pad_value: float = -float("inf"),
        return_mask: bool = False,
        include_log_probs: bool = True,
        include_context: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]]:
        B, T, D = self.n_sequences, max(self.lengths), self.feature_dim
        device, dtype = self.device, self.dtype

        seq_batch = torch.full((B, T, D), pad_value, dtype=dtype, device=device)
        mask = torch.zeros((B, T), dtype=torch.bool, device=device)

        lengths_tensor = torch.tensor(self.lengths, device=device)
        for i, s in enumerate(self.sequence):
            L = s.shape[0]
            seq_batch[i, :L] = s
            mask[i, :L] = True

        log_batch = None
        if include_log_probs and self.log_probs:
            K = self.log_probs[0].shape[-1]
            log_batch = torch.full((B, T, K), log_pad_value, dtype=dtype, device=device)
            for i, lp in enumerate(self.log_probs):
                if lp is not None:
                    log_batch[i, :lp.shape[0]] = lp

        ctx_batch = None
        if include_context and self.context:
            valid_ctx = [c for c in self.context if c is not None]
            if valid_ctx:
                H = valid_ctx[0].shape[-1]
                ctx_batch = torch.full((B, T, H), pad_value, dtype=dtype, device=device)
                for i, c in enumerate(self.context):
                    if c is not None:
                        ctx_batch[i, :c.shape[0]] = c

        if return_mask:
            return seq_batch, mask, log_batch, ctx_batch
        return (seq_batch, log_batch, ctx_batch) if log_batch is not None else seq_batch

    def pad_sequences(self, pad_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        return self.to_batch(pad_value=pad_value, return_mask=True, include_log_probs=False, include_context=False)[:3]


# -----------------------------
# ContextualVariables
# -----------------------------
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
        all_features = self.cat(dim=-1)
        mean, std = all_features.mean(0, keepdim=True), all_features.std(0, keepdim=True).clamp_min(eps)
        normed = [(x - mean) / std for x in self.X]
        return ContextualVariables(self.n_context, normed, self.time_dependent, self.names)

    def __getitem__(self, idx: Union[int, slice]) -> Union[torch.Tensor, "ContextualVariables"]:
        if isinstance(idx, slice):
            return ContextualVariables(self.n_context, self.X[idx], self.time_dependent, self.names)
        return self.X[idx]

    # Vectorized padding
    def pad_batch(self, pad_value: float = 0.0, return_mask: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if not self.time_dependent:
            return torch.stack(self.X, dim=0)

        B = 1 if self.X[0].ndim == 2 else self.X[0].shape[0]
        max_len = max(x.shape[1] if x.ndim == 3 else x.shape[0] for x in self.X)
        H = sum(x.shape[-1] for x in self.X)
        padded = torch.full((B, max_len, H), pad_value, dtype=self.dtype, device=self.device)
        mask = torch.zeros((B, max_len), dtype=torch.bool, device=self.device)

        offset = 0
        for x in self.X:
            x_pad = x.unsqueeze(0) if x.ndim == 2 else x
            L = x_pad.shape[1]
            F = x_pad.shape[2]
            padded[:, :L, offset:offset+F] = x_pad
            mask[:, :L] = True
            offset += F

        return (padded, mask) if return_mask else padded

    def summary(self) -> str:
        dep = "time-dependent" if self.time_dependent else "static"
        shapes = ", ".join(str(s) for s in self.shape)
        return f"{self.n_context} {dep} contexts [{shapes}] on {self.device}"
