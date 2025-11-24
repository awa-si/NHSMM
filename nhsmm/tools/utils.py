import torch
from dataclasses import dataclass
from typing import List, Optional, Union, Tuple


@dataclass(frozen=False)
class Observations:
    """
    Container for sequences, optional log-probs, context vectors, and masks.

    Accepts inputs as:
        - single-sequence: [T,F] or [T]
        - batched: [B,T,F] or [B,T]
        - context: [T,H], [1,H], [B,T,H], [B,H]
        - mask: [T], [T,1], [B,T], [B,T,1]
    """

    sequence: List[torch.Tensor]
    lengths: Optional[List[int]] = None
    log_probs: Optional[List[torch.Tensor]] = None
    context: Optional[List[Optional[torch.Tensor]]] = None
    mask: Optional[List[torch.Tensor]] = None

    # ---------------- Post Init ----------------
    def __post_init__(self):
        # Normalize all inputs into list-of-tensors form
        self.sequence = self._canonicalize_list(self.sequence, "sequence")
        seqs = self.sequence

        # Infer lengths if missing
        if self.lengths is None:
            self.lengths = [s.shape[0] for s in seqs]
        else:
            if len(self.lengths) != len(seqs):
                raise ValueError("`lengths` must match number of sequences.")
            for s, l in zip(seqs, self.lengths):
                if s.shape[0] != l:
                    raise ValueError("Mismatch between sequence and provided length.")

        # Canonicalize log_probs
        if self.log_probs is not None:
            self.log_probs = self._canonicalize_list(self.log_probs, "log_probs", allow_none=False)
            if len(self.log_probs) != len(seqs):
                raise ValueError("`log_probs` length mismatch.")

            # Normalize per-sequence shapes
            new_lp = []
            for lp, L in zip(self.log_probs, self.lengths):
                lp = self._ensure_2d_time_major(lp, L)
                new_lp.append(lp)
            self.log_probs = new_lp

        # Canonicalize context
        if self.context is None:
            self.context = [None] * len(seqs)
        else:
            self.context = self._canonicalize_list(self.context, "context", allow_none=True)
            if len(self.context) != len(seqs):
                raise ValueError("`context` length mismatch.")

            new_ctx = []
            for c, L in zip(self.context, self.lengths):
                if c is None:
                    new_ctx.append(None)
                    continue
                new_ctx.append(self._normalize_context(c, L))
            self.context = new_ctx

        # Canonicalize mask
        if self.mask is None:
            self.mask = [torch.ones(L, 1, dtype=torch.bool) for L in self.lengths]
        else:
            self.mask = self._canonicalize_list(self.mask, "mask", allow_none=False)
            if len(self.mask) != len(seqs):
                raise ValueError("`mask` length mismatch.")

            new_mask = []
            for m, L in zip(self.mask, self.lengths):
                new_mask.append(self._normalize_mask(m, L))
            self.mask = new_mask

    # ---------------- Internal helpers ----------------
    def _canonicalize_list(self, x, name, allow_none=False):
        # If given a single tensor, wrap as list
        if torch.is_tensor(x):
            return [x]

        # If x is a batch tensor [B,...]
        if isinstance(x, list):
            if allow_none:
                # allow sequences with None
                return [t if t is None else t for t in x]
            return [t for t in x]

        raise TypeError(f"`{name}` must be tensor or list of tensors.")

    def _ensure_2d_time_major(self, t, L):
        """
        Ensures:
            - 1D -> [T,1]
            - [1,T] -> squeeze to [T]
            - [T] -> [T,1]
            - [T,F] stays
        """
        if t.ndim == 1:
            if t.shape[0] == L:
                return t.unsqueeze(1)
            raise ValueError("1D tensor must have length T.")
        if t.ndim == 2:
            # could be [1,T] or [T,1] or [T,F]
            if t.shape[0] == 1 and t.shape[1] == L:
                return t.squeeze(0).unsqueeze(1)
            if t.shape[0] == L:
                return t
        raise ValueError(f"Invalid log-prob shape {t.shape} for T={L}")

    def _normalize_context(self, c, L):
        """
        Accepts:
            [H] -> expand to [T,H]
            [1,H] -> expand to [T,H]
            [T,H] -> OK
            [B,H] -> only accept B=1 -> squeeze -> [H]
            [B,T,H] -> only accept B=1 -> squeeze -> [T,H]
        """
        if c.ndim == 1:
            return c.unsqueeze(0).expand(L, -1)

        if c.ndim == 2:
            # [1,H]
            if c.shape[0] == 1:
                return c.expand(L, -1)
            # [T,H]
            if c.shape[0] == L:
                return c
            raise ValueError(f"Invalid context shape {c.shape} for T={L}")

        if c.ndim == 3:
            # [B,T,H]
            if c.shape[0] == 1:
                return c.squeeze(0)
            raise ValueError("Context batch >1 not supported per sequence.")

        raise ValueError(f"Unsupported context shape {c.shape}")

    def _normalize_mask(self, m, L):
        """
        Accept mask as:
            [T]
            [T,1]
            [1,T]
            [B,T] (B=1)
            [B,T,1] (B=1)
        Always returns [T,1] bool mask.
        """
        if m.ndim == 1:
            if m.shape[0] == L:
                return m.bool().unsqueeze(1)

        if m.ndim == 2:
            # [T,1]
            if m.shape == (L, 1):
                return m.bool()
            # [1,T]
            if m.shape[0] == 1 and m.shape[1] == L:
                return m.squeeze(0).unsqueeze(1).bool()

            # [B,T], B=1
            if m.shape[0] == 1 and m.shape[1] == L:
                return m.squeeze(0).unsqueeze(1).bool()

        if m.ndim == 3:
            # [1,T,1]
            if m.shape[0] == 1 and m.shape[1] == L:
                return m.squeeze(0).bool()

        raise ValueError(f"Invalid mask shape {m.shape} for T={L}")

    # ---------------- Properties ----------------
    @property
    def n_sequences(self):
        return len(self.sequence)

    @property
    def total_length(self):
        return sum(self.lengths)

    @property
    def feature_dim(self):
        f = {s.shape[-1] for s in self.sequence}
        if len(f) != 1:
            raise ValueError("Feature dimension mismatch.")
        return f.pop()

    @property
    def device(self):
        return self.sequence[0].device

    @property
    def dtype(self):
        return self.sequence[0].dtype

    # ---------------- Clone / detach ----------------
    def detach(self):
        return Observations(
            sequence=[s.detach() for s in self.sequence],
            lengths=list(self.lengths),
            log_probs=[lp.detach() for lp in self.log_probs] if self.log_probs is not None else None,
            context=[c.detach() if c is not None else None for c in self.context],
            mask=[m.clone() for m in self.mask],
        )

    def clone(self):
        return Observations(
            sequence=[s.clone() for s in self.sequence],
            lengths=list(self.lengths),
            log_probs=[lp.clone() for lp in self.log_probs] if self.log_probs is not None else None,
            context=[c.clone() if c is not None else None for c in self.context],
            mask=[m.clone() for m in self.mask],
        )

    # ---------------- Indexing ----------------
    def __getitem__(self, idx):
        def pick(lst):
            if lst is None:
                return None
            if isinstance(idx, slice):
                return lst[idx]
            return [lst[idx]]

        return Observations(
            sequence=pick(self.sequence),
            lengths=pick(self.lengths),
            log_probs=pick(self.log_probs),
            context=pick(self.context),
            mask=pick(self.mask),
        )

    # ---------------- to_tensor ----------------
    def to_tensor(self, key="sequence"):
        items = getattr(self, key)
        if items is None:
            raise ValueError(f"{key} is None.")
        B = len(items)
        T_max = max(t.shape[0] for t in items)
        H = items[0].shape[-1]

        out = torch.zeros(B, T_max, H, dtype=self.dtype, device=self.device)
        for i, t in enumerate(items):
            out[i, :t.shape[0]] = t
        return out

    # ---------------- Summary ----------------
    def summary(self):
        return (
            f"Observations(n_sequences={self.n_sequences}, "
            f"total_length={self.total_length}, "
            f"feature_dim={self.feature_dim}, "
            f"mask_coverage={[m.sum().item() for m in self.mask]})"
        )


from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Dict
import torch


@dataclass(frozen=False)
class ContextualVariables:
    """
    Batch- and time-aware container for context tensors.

    Supported input tensor shapes for each context:
      - [H]
      - [T, H]
      - [B, H]
      - [B, T, H]

    Important rules:
      - If `time_dependent=True`, 2-D tensors are interpreted as [T, H].
        Otherwise, 2-D tensors are interpreted as [B, H].
      - If any provided tensor has explicit batch (B) or time (T) dimensions,
        the class will attempt to infer global B and T and broadcast other
        contexts to (B, T, H) where appropriate.
      - Cache keys include (B, T) so different batch/time shapes don't reuse
        cached concatenations.
    """

    n_context: int
    X: List[torch.Tensor]
    time_dependent: bool = False
    names: Optional[List[str]] = None

    # internal cache mapping (cache_key, B, T) -> tensor
    _cache: Optional[Dict] = None

    def __post_init__(self):
        if not isinstance(self.X, list) or len(self.X) == 0:
            raise ValueError("`X` must be a non-empty list of torch.Tensor")
        if len(self.X) != self.n_context:
            raise ValueError(f"n_context ({self.n_context}) != len(X) ({len(self.X)})")
        if self.names is not None and len(self.names) != self.n_context:
            raise ValueError("`names` length must match `n_context`")

        if not all(torch.is_tensor(x) for x in self.X):
            raise TypeError("All elements of X must be torch.Tensor")

        # Validate dtypes / devices uniformity (helpful early)
        devs = {x.device for x in self.X}
        dtypes = {x.dtype for x in self.X}
        if len(devs) > 1:
            raise ValueError("All context tensors must be on the same device")
        if len(dtypes) > 1:
            raise ValueError("All context tensors must have the same dtype")

        # initialize cache
        self._cache = {}

        # Validate feature dim consistency (where determinable)
        feature_dims = {self._infer_feature_dim(x) for x in self.X if x.ndim >= 1}
        if len(feature_dims) > 1:
            raise ValueError("Inconsistent feature dimensions across context tensors")
        # ok if single context only; feature dim will be derived later as needed

    @staticmethod
    def _infer_feature_dim(x: torch.Tensor) -> int:
        if x.ndim == 1:
            return x.shape[0]
        return x.shape[-1]

    def _extract_BT(self, x: torch.Tensor) -> Tuple[Optional[int], Optional[int]]:
        """
        Returns (B, T) candidate for tensor x; None if not present.
        - ndim==3 -> (B, T)
        - ndim==2 -> (B, None) if time_dependent==False else (None, T)
        - ndim==1 -> (None, None)
        """
        if x.ndim == 3:
            return x.shape[0], x.shape[1]
        if x.ndim == 2:
            if self.time_dependent:
                return None, x.shape[0]   # interpret as [T,H]
            else:
                return x.shape[0], None   # interpret as [B,H]
        if x.ndim == 1:
            return None, None
        raise ValueError(f"Unsupported tensor ndim {x.ndim}")

    def infer_global_BT(self) -> Tuple[Optional[int], Optional[int]]:
        """
        Infer global B and T from available contexts using deterministic rules:
         - If any tensor has explicit batch (ndim==3 or 2 when time_dependent==False),
           take max batch size among those tensors as B.
         - If any tensor has explicit time (ndim==3 or 2 when time_dependent==True),
           take max time length among those tensors as T.
        Returns (B, T) with None if neither found.
        """
        B_candidates = []
        T_candidates = []
        for x in self.X:
            b, t = self._extract_BT(x)
            if b is not None:
                B_candidates.append(b)
            if t is not None:
                T_candidates.append(t)

        B = max(B_candidates) if B_candidates else None
        T = max(T_candidates) if T_candidates else None
        return B, T

    def _broadcast_to_B_T_H(self, x: torch.Tensor, B: Optional[int], T: Optional[int]) -> torch.Tensor:
        """
        Broadcast a single context tensor x into shape:
          - [B, T, H] if both B and T provided,
          - [B, 1, H] if only B provided,
          - [1, T, H] if only T provided,
          - [1, 1, H] if neither provided.

        Rules for input shapes:
          - [B, T, H] -> validated and returned
          - [T, H]     -> if time_dependent True, becomes [1, T, H] or [B, T, H] when B known
          - [B, H]     -> if time_dependent False, becomes [B, 1, H] or [B, T, H] when T known
          - [H]        -> broadcast to all dims present
        """
        h = self._infer_feature_dim(x)

        # full 3D
        if x.ndim == 3:
            bx, tx, hx = x.shape
            if B is not None and bx != B:
                raise ValueError(f"Context batch mismatch: tensor has B={bx} but expected B={B}")
            if T is not None and tx != T:
                raise ValueError(f"Context time mismatch: tensor has T={tx} but expected T={T}")
            return x

        # 2D case: interpret by time_dependent flag
        if x.ndim == 2:
            if self.time_dependent:
                # treat as [T, H]
                tx, hx = x.shape
                if T is not None and tx != T:
                    raise ValueError(f"Context time length {tx} != global T {T}")
                if B is None:
                    # return [1, T, H]
                    return x.unsqueeze(0)
                # return [B, T, H]
                return x.unsqueeze(0).expand(B, tx, hx)
            else:
                # treat as [B, H]
                bx, hx = x.shape
                if B is not None and bx != B:
                    raise ValueError(f"Context batch size {bx} != global B {B}")
                if T is None:
                    # return [B, 1, H]
                    return x.unsqueeze(1)
                # return [B, T, H]
                return x.unsqueeze(1).expand(bx, T, hx)

        # 1D case: [H]
        if x.ndim == 1:
            hx = x.shape[0]
            if B is not None and T is not None:
                return x.view(1, 1, hx).expand(B, T, hx)
            if B is not None:
                return x.view(1, hx).expand(B, hx).unsqueeze(1)  # [B,1,H]
            if T is not None:
                return x.view(1, hx).expand(T, hx).unsqueeze(0)  # [1,T,H]
            return x.view(1, 1, hx)  # [1,1,H]

        raise ValueError(f"Unsupported tensor ndim {x.ndim}")

    def cat(self, dim: int = -1, normalize: bool = False, eps: float = 1e-6) -> torch.Tensor:
        """
        Concatenate all context tensors into a single tensor.

        Output shape:
          - If global B or T inferred: returns [B, T, H_combined] (missing dims broadcasted)
          - If no B/T found: returns [1, 1, H_combined]

        Caches result with key (dim, normalize, B, T).
        """
        B, T = self.infer_global_BT()
        cache_key = (dim, normalize, B, T)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Broadcast each X to (B,T,H) semantics
        Xb = [self._broadcast_to_B_T_H(x, B, T) for x in self.X]

        # Concatenate along requested dim: for convenience, allow dim in {-1, -2, -3} relative to [B,T,H]
        # normalize to positive dim
        # we always concatenate on feature dim (last) unless user explicitly passed different value
        if dim < 0:
            dim_adj = 3 + dim  # -1 -> 2, -2 -> 1, -3 -> 0
        else:
            dim_adj = dim

        # we expect user normally wants to concat features (dim_adj==2)
        if dim_adj not in (0, 1, 2):
            raise ValueError("`dim` must be -3..2 for the resulting [B,T,H] tensor")

        out = Xb[0] if len(Xb) == 1 else torch.cat(Xb, dim=dim_adj)

        if normalize:
            # normalize per-feature across batch+time: preserve per-feature statistics
            mean = out.mean(dim=(0, 1), keepdim=True)
            std = out.std(dim=(0, 1), keepdim=True).clamp_min(eps)
            out = (out - mean) / std

        self._cache[cache_key] = out
        return out

    def to(self, device, dtype=None) -> "ContextualVariables":
        X = [x.to(device=device, dtype=dtype) if dtype is not None else x.to(device=device) for x in self.X]
        return ContextualVariables(self.n_context, X, self.time_dependent, self.names)

    def detach(self) -> "ContextualVariables":
        X = [x.detach() for x in self.X]
        return ContextualVariables(self.n_context, X, self.time_dependent, self.names)

    def clone(self) -> "ContextualVariables":
        X = [x.clone() for x in self.X]
        return ContextualVariables(self.n_context, X, self.time_dependent, self.names)

    def __getitem__(self, idx: Union[int, slice]) -> "ContextualVariables":
        Xsel = [x[idx] for x in self.X]
        return ContextualVariables(self.n_context, Xsel, self.time_dependent,
                                   None if self.names is None else [self.names[i] for i in range(len(self.names))][idx])

    def __repr__(self):
        names = self.names or [f"ctx{i}" for i in range(self.n_context)]
        specs = ", ".join(f"{n}:{tuple(x.shape)}" for n, x in zip(names, self.X))
        td = "time-vary" if self.time_dependent else "static"
        return f"<ContextualVariables[{td}] {specs}>"

