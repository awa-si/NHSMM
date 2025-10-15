# models/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Any, Literal, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, Categorical

from nhsmm import (
    utils, constraints, SeedGenerator, ConvergenceHandler
)
from nhsmm.defaults import (
    DTYPE, DefaultEmission, DefaultDuration, DefaultTransition
)


class HSMM(nn.Module, ABC):
    """
    Hidden Semi-Markov Model (HSMM) base class.

    This class implements the foundational structure for HSMMs — 
    a probabilistic sequence model that extends Hidden Markov Models (HMMs)
    by modeling explicit state durations and contextual dependencies.

    The HSMM supports modular components for:
        - **Emissions:** State-dependent observation likelihoods (Gaussian or Categorical).
        - **Transitions:** Contextual transition probabilities between states.
        - **Durations:** Explicit duration modeling per state.
        - **Context encoding:** Optional neural context conditioning for adaptive inference.

    Key Methods (to implement or override in subclasses):
        - `forward(X)`: Perform forward message passing (α-recursion).
        - `_compute_log_likelihood(X)`: Compute total log-likelihood of sequences.
        - `score(X, lengths=None, by_sample=True)`: Evaluate model likelihoods.
        - `sample(T)`: Generate sequences given model parameters.
        - `ic(X, criterion, lengths=None)`: Compute information criteria (AIC/BIC).

    Attributes:
        n_components (int): Number of hidden states.
        n_features (int): Dimensionality of observation space.
        dof (int): Model degrees of freedom, used in information criteria.
        emission_module (nn.Module): State emission probability model.
        transition_module (nn.Module): State transition probability model.
        duration_module (nn.Module): Duration distribution per state.
        encoder (Optional[nn.Module]): Optional neural encoder for context features.

    Notes:
        - Implementations should ensure numerical stability in log-space computations.
        - Duration distributions must be normalized per state.
        - Supports variable-length batches and GPU computation.
    """

    def __init__(self,
            n_states: int,
            n_features: int,
            max_duration: int,
            alpha: float = 1.0,
            seed: Optional[int] = None,
            context_dim: Optional[int] = None,
            min_covar = 1e-3,
        ):
        super().__init__()
        self.n_states = n_states
        self.alpha = float(alpha)
        self.min_covar = min_covar
        self.n_features = n_features
        self.max_duration = max_duration
        self._seed_gen = SeedGenerator(seed)
        self._context: Optional[torch.Tensor] = None

        # container for emitted distribution and other non-buffer parameters
        self._params: Dict[str, Any] = {}

        # optional external encoder (neural) for contextualization
        self.encoder: Optional[nn.Module] = None

        self.emission_module = DefaultEmission(n_states, n_features, min_covar, context_dim)
        self.duration_module = DefaultDuration(n_states, max_duration, context_dim)
        self.transition_module = DefaultTransition(n_states, context_dim)

        # initialize & register pi/A/D logits as buffers (log-space)
        self._init_buffers()
        self._params['emission_pdf'] = self.sample_emission_pdf(None)

    def _init_buffers(self):
        device = next(self.buffers(), torch.tensor(0., dtype=DTYPE)).device

        # -------------------------------
        # Safe sampling and logits helper
        # -------------------------------
        def sample_logits(shape, transitions=None):
            if transitions is None:
                probs = constraints.sample_probs(self.alpha, shape)
            else:
                probs = constraints.sample_A(self.alpha, shape[0], transitions)
            logits = torch.log(probs.clamp_min(1e-12)).to(device=device, dtype=DTYPE)
            return logits, probs.shape  # return shape for batch alignment

        # -------------------------------
        # Base HSMM parameters
        # -------------------------------
        pi_logits, pi_shape = sample_logits((self.n_states,))
        A_logits, A_shape = sample_logits((self.n_states, self.n_states), constraints.Transitions.SEMI)
        D_logits, D_shape = sample_logits((self.n_states, self.max_duration))

        self.register_buffer("_pi_logits", pi_logits)
        self.register_buffer("_A_logits", A_logits)
        self.register_buffer("_D_logits", D_logits)

        self._pi_batch_shape = pi_shape
        self._A_batch_shape = A_shape
        self._D_batch_shape = D_shape

        # -------------------------------
        # Optional super-state hierarchy
        # -------------------------------
        n_super_states = getattr(self, "n_super_states", 0)
        if n_super_states > 1:
            super_pi_logits, super_pi_shape = sample_logits((n_super_states,))
            super_A_logits, super_A_shape = sample_logits((n_super_states, n_super_states), constraints.Transitions.SEMI)
            self.register_buffer("_super_pi_logits", super_pi_logits)
            self.register_buffer("_super_A_logits", super_A_logits)

            self._super_pi_batch_shape = super_pi_shape
            self._super_A_batch_shape = super_A_shape

        # -------------------------------
        # Initialization snapshot for debugging
        # -------------------------------
        summary = [
            pi_logits.mean(), 
            A_logits.mean(), 
            D_logits.mean()
        ]
        if n_super_states > 1:
            summary += [super_pi_logits.mean(), super_A_logits.mean()]

        self.register_buffer("_init_prior_snapshot", torch.stack(summary).detach())

    @property
    def seed(self) -> Optional[int]:
        return self._seed_gen.seed

    @property
    def pi(self) -> torch.Tensor:
        return self._pi_logits

    @pi.setter
    def pi(self, logits: torch.Tensor):
        logits = logits.to(device=self._pi_logits.device, dtype=DTYPE)
        if logits.shape != (self.n_states,):
            raise ValueError(f"pi logits must have shape ({self.n_states},) but got {tuple(logits.shape)}")

        norm_val = logits.logsumexp(0)
        if not torch.allclose(norm_val, torch.tensor(0.0, dtype=DTYPE, device=logits.device), atol=1e-8):
            raise ValueError(f"pi logits must normalize (logsumexp==0); got {norm_val.item():.3e}")

        self._pi_logits.copy_(logits)

    @property
    def A(self) -> torch.Tensor:
        return self._A_logits

    @A.setter
    def A(self, logits: torch.Tensor):
        logits = logits.to(device=self._A_logits.device, dtype=DTYPE)
        if logits.shape != (self.n_states, self.n_states):
            raise ValueError(f"A logits must have shape ({self.n_states},{self.n_states})")

        row_norm = logits.logsumexp(1)
        if not torch.allclose(row_norm, torch.zeros_like(row_norm), atol=1e-8):
            raise ValueError(f"Rows of A logits must normalize (logsumexp==0); got {row_norm}")

        if not constraints.is_valid_A(logits, constraints.Transitions.SEMI):
            raise ValueError("A logits do not satisfy SEMI transition constraints")

        self._A_logits.copy_(logits)

    @property
    def D(self) -> torch.Tensor:
        return self._D_logits

    @D.setter
    def D(self, logits: torch.Tensor):
        logits = logits.to(device=self._D_logits.device, dtype=DTYPE)
        if logits.shape != (self.n_states, self.max_duration):
            raise ValueError(f"D logits must have shape ({self.n_states},{self.max_duration})")

        row_norm = logits.logsumexp(1)
        if not torch.allclose(row_norm, torch.zeros_like(row_norm), atol=1e-8):
            raise ValueError(f"Rows of D logits must normalize (logsumexp==0); got {row_norm}")

        self._D_logits.copy_(logits)

    @property
    def pdf(self) -> Any:
        """Emission distribution (torch.distributions.Distribution). Managed by subclass via hooks."""
        return self._params.get('emission_pdf')

    @property
    @abstractmethod
    def dof(self) -> int:
        """Degrees of freedom (required for IC computations)."""
        raise NotImplementedError

    @abstractmethod
    def sample_emission_pdf(self, X: Optional[torch.Tensor] = None) -> Distribution:
        raise NotImplementedError

    @abstractmethod
    def _estimate_emission_pdf(self, X: torch.Tensor, posterior: torch.Tensor, theta: Optional[utils.ContextualVariables] = None) -> Distribution:
        raise NotImplementedError

    def attach_encoder(self, encoder: nn.Module, batch_first: bool = True):
        """
        Attach a neural encoder (CNN/LSTM/Transformer) to the HSMM module.
        Automatically wraps the encoder and updates self._context for amortized inference.
        
        Args:
            encoder (nn.Module): PyTorch encoder module.
            batch_first (bool): Whether input tensors are (B,T,F) or (T,B,F).
        """
        device = next(self.parameters()).device
        encoder.to(device)

        class EncoderWrapper(nn.Module):
            def __init__(self, enc, batch_first, parent):
                super().__init__()
                self.enc = enc
                self.batch_first = batch_first
                self.parent = parent

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Ensure batch dimension
                if x.dim() == 2:  # (T, F) -> (1, T, F)
                    x = x.unsqueeze(0 if self.batch_first else 1)
                elif x.dim() != 3:
                    raise ValueError(f"Unsupported input shape {x.shape}, expected (T,F) or (B,T,F)")

                # Handle time-first if needed
                if not self.batch_first:
                    x = x.transpose(0, 1)  # (T,B,F) -> (B,T,F)

                # Forward pass
                out = self.enc(x)
                if isinstance(out, (tuple, list)):
                    out = out[0]  # support RNN/LSTM returning (out, hidden)

                if not torch.is_tensor(out):
                    raise TypeError(f"Encoder must return a tensor, got {type(out)}")

                # Standardize output to (B,T,H)
                if out.dim() == 1:  # (H,) -> (1,1,H)
                    theta = out.unsqueeze(0).unsqueeze(0)
                elif out.dim() == 2:  # (B,H) or (T,H)
                    if out.shape[0] == x.shape[0]:
                        theta = out.unsqueeze(1)  # (B,1,H)
                    else:
                        theta = out.unsqueeze(0)  # (1,T,H)
                elif out.dim() == 3:
                    theta = out
                else:
                    raise ValueError(f"Unexpected encoder output shape {out.shape}")

                # Optional: temporal pooling for context vector θ
                theta_ctx = theta.mean(dim=1)  # simple mean pooling (B,H)
                theta_ctx = nn.functional.layer_norm(theta_ctx, theta_ctx.shape[-1:])

                # Integrate into parent HSMM context
                self.parent._context = theta_ctx.detach()

                return out

        self.encoder = EncoderWrapper(encoder, batch_first, self)

    def encode_observations(self, X: torch.Tensor, pool: str = "last", store: bool = True) -> Optional[torch.Tensor]:
        """
        Encode observations into a context vector θ using the attached encoder.
        Supports CNN, RNN/LSTM, Transformer, or hybrid encoders.
        Includes optional learnable single- or multi-head attention pooling with relative positional bias.

        Args:
            X (torch.Tensor): Input observations of shape (T, F) or (B, T, F).
            pool (str): Temporal pooling mode ('last', 'mean', 'max', 'attn', 'mha').
            store (bool): Whether to store resulting context in self._context.

        Returns:
            Optional[torch.Tensor]: Context vectors θ of shape (B, H), or None.
        """
        if self.encoder is None:
            if store:
                self._context = None
            return None

        device = next(self.encoder.parameters()).device
        X = X.to(device=device, dtype=DTYPE)

        if X.ndim == 2:
            X = X.unsqueeze(0)
        elif X.ndim != 3:
            raise ValueError(f"Expected (T,F) or (B,T,F), got {X.shape}")

        try:
            out = self.encoder(X)
        except Exception as e:
            raise RuntimeError(f"Encoder forward() failed: {e}")

        if isinstance(out, (tuple, list)):
            out = out[0]

        out = out.to(dtype=DTYPE, device=device)

        if out.ndim == 3:  # (B, T, H)
            if pool == "last":
                vec = out[:, -1]
            elif pool == "mean":
                vec = out.mean(dim=1)
            elif pool == "max":
                vec, _ = out.max(dim=1)
            elif pool == "attn":
                if not hasattr(self, "_attn_vector") or self._attn_vector.shape[-1] != out.shape[-1]:
                    self._attn_vector = nn.Parameter(torch.randn(out.shape[-1], device=device, dtype=DTYPE))
                attn_scores = out @ self._attn_vector
                attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
                vec = (attn_weights * out).sum(dim=1)
            elif pool == "mha":
                n_heads = getattr(self, "_mha_heads", 4)
                head_dim = out.shape[-1] // n_heads

                if not hasattr(self, "_mha_qkv"):
                    self._mha_qkv = nn.Linear(out.shape[-1], 3 * out.shape[-1], device=device, dtype=DTYPE)
                if not hasattr(self, "_mha_rel_pos"):
                    # Relative positional bias: shape (2*T-1, n_heads)
                    self._mha_rel_pos = nn.Parameter(torch.zeros(2*out.shape[1]-1, n_heads, device=device, dtype=DTYPE))

                qkv = self._mha_qkv(out)
                Q, K, V = qkv.chunk(3, dim=-1)
                Q = Q.view(Q.shape[0], Q.shape[1], n_heads, head_dim).transpose(1, 2)
                K = K.view(K.shape[0], K.shape[1], n_heads, head_dim).transpose(1, 2)
                V = V.view(V.shape[0], V.shape[1], n_heads, head_dim).transpose(1, 2)

                attn_scores = (Q @ K.transpose(-2, -1)) / head_dim**0.5  # (B, Hh, T, T)

                # Add relative positional bias
                T = out.shape[1]
                rel_idx = torch.arange(T, device=device)[:, None] - torch.arange(T, device=device)[None, :] + T - 1
                attn_scores = attn_scores + self._mha_rel_pos[rel_idx].permute(2, 0, 1).unsqueeze(0)

                attn_weights = torch.softmax(attn_scores, dim=-1)
                context = (attn_weights @ V).transpose(1, 2).contiguous().view(out.shape[0], out.shape[1], -1)
                vec = context.mean(dim=1)
            else:
                raise ValueError(f"Unsupported pooling mode: {pool}")
        elif out.ndim == 2:
            vec = out
        else:
            raise ValueError(f"Unexpected encoder output shape: {out.shape}")

        vec = nn.functional.layer_norm(vec, vec.shape[-1:])
        vec = torch.clamp(vec, -10.0, 10.0)

        if store:
            self._context = vec.detach()

        return vec

    def _combine_context(self, X: Optional[torch.Tensor] = None, theta: Optional[torch.Tensor] = None, reduce: bool = False) -> Optional[torch.Tensor]:
        """
        Combine input features X with context theta.

        Args:
            X: (B, T, F) or (T, F)
            theta: (B, T, H), (B, H), or (1, H)
            reduce: average over temporal dimension

        Returns:
            Combined tensor (B, T, F+H) or reduced (B, 1, F+H)
        """
        if X is None and theta is None:
            return None

        # Convert to tensors
        if X is not None and not torch.is_tensor(X):
            X = torch.as_tensor(X, dtype=DTYPE, device=self.device)
        if theta is not None and not torch.is_tensor(theta):
            theta = torch.as_tensor(theta, dtype=DTYPE, device=self.device)

        # Broadcast theta
        if theta is not None:
            if theta.ndim == 1:
                theta = theta.view(1, 1, -1)
            elif theta.ndim == 2:
                theta = theta.unsqueeze(1)
            if X is not None:
                B, T = X.shape[:2]
                theta = theta.expand(B, T, -1)
            elif reduce:
                theta = theta.mean(dim=1, keepdim=True)

        combined = X if theta is None else theta if X is None else torch.cat([X, theta], dim=-1)
        if reduce and combined.ndim == 3:
            combined = combined.mean(dim=1, keepdim=True)
        return combined

    def _contextual_emission_pdf(self, X: torch.Tensor, theta: Optional[torch.Tensor] = None) -> torch.Tensor:
        device, dtype = X.device, X.dtype
        B, T = X.shape[:2]
        K = self.n_states

        base_logp = torch.nan_to_num(self.map_emission(X), nan=-1e8, neginf=-1e8, posinf=-1e8)

        if theta is None:
            return constraints.log_normalize(base_logp, dim=-1)

        # Normalize theta
        if theta.ndim == 1:
            theta = theta.view(1, 1, -1).expand(B, T, -1)
        elif theta.ndim == 2:
            theta = theta.unsqueeze(1).expand(B, T, -1)
        elif theta.ndim != 3:
            raise ValueError(f"Unexpected theta shape {theta.shape}")
        theta = theta.to(device=device, dtype=dtype)

        # Compute context shift
        if hasattr(self, "emission_context_adapter") and callable(self.emission_context_adapter):
            logp = base_logp + self.emission_context_adapter(theta)
        elif hasattr(self.emission_module, "contextual_log_prob"):
            logp = self.emission_module.contextual_log_prob(X, theta)
        else:
            if not hasattr(self, "_context_affine"):
                self._context_affine = nn.Linear(theta.shape[-1], K, device=device, dtype=dtype)
            logp = base_logp + self._context_affine(theta)

        logp = torch.nan_to_num(logp, nan=-1e8, neginf=-1e8, posinf=-1e8)
        return constraints.log_normalize(logp, dim=-1)

    def _contextual_duration_pdf(self, theta: Optional[torch.Tensor]) -> torch.Tensor:
        base_D = self._D_logits
        device, dtype = base_D.device, base_D.dtype

        if theta is None:
            return base_D

        # Reduce theta to (B, H)
        if theta.dim() == 1:
            theta = theta.unsqueeze(0)
        elif theta.dim() == 3:
            theta = theta[:, -1, :]
        theta = theta.to(device=device, dtype=dtype)

        if not hasattr(self, "_duration_adapter"):
            H = theta.shape[-1]
            self._duration_adapter = nn.Sequential(
                nn.Linear(H, H),
                nn.ReLU(),
                nn.Linear(H, self.n_states * self.max_duration)
            ).to(device=device, dtype=dtype)

        delta = self._duration_adapter(theta).view(-1, self.n_states, self.max_duration)
        log_D = base_D.unsqueeze(0) + delta
        return log_D - torch.logsumexp(log_D, dim=2, keepdim=True)

    def _contextual_transition_matrix(self, theta: Optional[torch.Tensor]) -> torch.Tensor:
        base_A = self._A_logits
        device, dtype = base_A.device, base_A.dtype

        if theta is None:
            return base_A

        # Reduce theta to (B, H)
        if theta.dim() == 1:
            theta = theta.unsqueeze(0)
        elif theta.dim() == 3:
            theta = theta[:, -1, :]
        theta = theta.to(device=device, dtype=dtype)

        if not hasattr(self, "_transition_adapter"):
            H = theta.shape[-1]
            self._transition_adapter = nn.Sequential(
                nn.Linear(H, H),
                nn.ReLU(),
                nn.Linear(H, self.n_states * self.n_states)
            ).to(device=device, dtype=dtype)

        delta = self._transition_adapter(theta).view(-1, self.n_states, self.n_states)
        log_A = base_A.unsqueeze(0) + delta

        if hasattr(constraints, "mask_invalid_transitions"):
            mask = constraints.mask_invalid_transitions(self.n_states, constraints.Transitions.SEMI).to(device)
            log_A = log_A.masked_fill(~mask.unsqueeze(0), -torch.inf)

        return log_A - torch.logsumexp(log_A, dim=2, keepdim=True)

    def map_emission(self, x: torch.Tensor, theta: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute per-state emission log-probabilities for a sequence or batch, optionally using context.

        Args:
            x (torch.Tensor): Input sequence tensor of shape
                - (T, F...) for a single sequence
                - (B, T, F...) for a batch
            theta (Optional[torch.Tensor]): Optional context tensor for conditional emission PDFs.

        Returns:
            torch.Tensor: Log-probabilities of shape
                - (T, n_states) for a single sequence
                - (B, T, n_states) for a batch
        """
        device = next(self.parameters(), torch.tensor(0.0)).device
        x = x.to(device=device, dtype=DTYPE)

        pdf = self._contextual_emission_pdf(x, theta)
        if pdf is None:
            raise RuntimeError(
                "Emission PDF not initialized. Ensure `sample_emission_pdf()` is called before `map_emission()`."
            )

        # Validate input shape against PDF event shape
        if pdf.event_shape and x.shape[-len(pdf.event_shape):] != pdf.event_shape:
            raise ValueError(
                f"Input feature shape {x.shape[-len(pdf.event_shape):]} does not match "
                f"PDF event shape {pdf.event_shape}."
            )

        # Determine batch dimensions
        batch_dims = x.shape[:-len(pdf.event_shape)]
        n_states = getattr(pdf, "batch_shape", (1,))[0]  # Assumes n_states stored in batch_shape[0]

        # Expand x to match the PDF shape: add a states dimension at the correct position
        x_exp = x.unsqueeze(-len(pdf.event_shape)-1)  # Insert before event dimensions

        try:
            log_probs = pdf.log_prob(x_exp)  # Shape: batch_dims + (n_states,) + event_shape
        except Exception as e:
            raise RuntimeError(f"Error computing log_prob: {e}")

        # Collapse event dimensions if any, leaving only batch_dims + n_states
        if log_probs.ndim > len(batch_dims) + 1:
            log_probs = log_probs.flatten(start_dim=len(batch_dims)+1).sum(dim=-1)

        return log_probs

    def check_constraints(self, value: torch.Tensor, clamp: bool = False) -> torch.Tensor:
        """
        Validate observations against the emission PDF support and event shape.
        Optionally clamps values outside the PDF support.

        Args:
            value (torch.Tensor): Input tensor of shape (batch, *event_shape).
            clamp (bool): If True, clip values to the valid support instead of raising errors.

        Returns:
            torch.Tensor: Validated (and possibly clamped) tensor.

        Raises:
            RuntimeError: If the emission PDF is not initialized.
            ValueError: If input shape does not match the PDF event shape.
        """
        pdf = self.pdf
        if pdf is None:
            raise RuntimeError(
                "Emission PDF not initialized. Ensure `sample_emission_pdf()` or `encode_observations()` has been called."
            )

        # Align device and dtype
        if hasattr(pdf, "mean") and isinstance(pdf.mean, torch.Tensor):
            value = value.to(device=pdf.mean.device, dtype=pdf.mean.dtype)

        # Vectorized support check
        support_mask = pdf.support.check(value)
        if clamp and hasattr(pdf.support, "clamp"):
            value = torch.where(support_mask, value, pdf.support.clamp(value))
        elif not torch.all(support_mask):
            bad_vals = value[~support_mask].flatten().unique()
            raise ValueError(f"Values outside PDF support detected: {bad_vals.tolist()}")

        # Validate event shape
        event_shape = pdf.event_shape or ()
        expected_ndim = len(event_shape) + 1  # +1 for batch dimension
        if value.ndim != expected_ndim:
            raise ValueError(
                f"Expected {expected_ndim}D input (batch + event dims), got {value.ndim}D."
            )

        if event_shape and tuple(value.shape[-len(event_shape):]) != tuple(event_shape):
            raise ValueError(
                f"PDF event shape mismatch: expected {tuple(event_shape)}, got {tuple(value.shape[-len(event_shape):])}."
            )

        return value

    def to_observations(self, X: torch.Tensor, lengths: Optional[List[int]] = None, theta: Optional[torch.Tensor] = None) -> utils.Observations:
        """
        Convert input tensor X into a utils.Observations object suitable for HSMM.

        Args:
            X (torch.Tensor): Input features of shape
                - (T, F) for a single sequence
                - (B, T, F) for a batch of sequences
            lengths (Optional[List[int]]): Optional list of sequence lengths. If None, entire X is one sequence.
            theta (Optional[torch.Tensor]): Optional context for emission PDFs.

        Returns:
            utils.Observations: Object containing sequences, per-state log-probabilities, and lengths.
        """
        device = next(self.parameters(), torch.tensor(0.0)).device

        # Validate constraints and move to device
        X_valid = self.check_constraints(X).to(device=device, dtype=DTYPE)

        # Determine total length of sequences
        total_len = X_valid.shape[0] if X_valid.ndim == 2 else X_valid.shape[1]

        # Determine sequence lengths
        if lengths is not None:
            if sum(lengths) != total_len:
                raise ValueError(f"Sum of lengths ({sum(lengths)}) does not match total samples ({total_len}).")
            seq_lengths = lengths
        else:
            seq_lengths = [total_len]

        # Split sequences
        sequences = []
        if X_valid.ndim == 2:  # Single sequence
            sequences = list(torch.split(X_valid, seq_lengths))
        elif X_valid.ndim == 3:  # Batch sequences
            if len(seq_lengths) == X_valid.shape[0]:
                sequences = [X_valid[i] for i in range(X_valid.shape[0])]
            else:
                sequences = list(torch.split(X_valid.reshape(-1, X_valid.shape[-1]), seq_lengths))
        else:
            raise ValueError(f"Unsupported input shape {X_valid.shape}")

        # Compute per-sequence log probabilities
        log_probs_list = []
        for idx, seq in enumerate(sequences):
            seq_theta = None
            if theta is not None:
                if theta.ndim == 2 and theta.shape[0] == len(sequences):
                    seq_theta = theta[idx].unsqueeze(0)  # (1, H) or (1, 1, H)
                else:
                    seq_theta = theta

            log_probs = self.map_emission(seq, seq_theta)
            log_probs_list.append(log_probs)

        return utils.Observations(
            sequence=sequences,
            log_probs=log_probs_list,
            lengths=seq_lengths
        )

    def sample_model_params(self, X: Optional[torch.Tensor] = None, theta: Optional[torch.Tensor] = None, inplace: bool = True) -> Dict[str, Any]:
        """
        Sample model params (π, A, D, emission_pdf) in log-space, optionally context-conditioned.
        """
        device = next(self.parameters(), torch.tensor(0.0)).device
        α = getattr(self, "alpha", 1.0)

        # --- π ---
        pi = torch.log(constraints.sample_probs(α, (self.n_states,)).to(DTYPE).to(device))

        # --- A ---
        A = torch.log(constraints.sample_A(α, self.n_states, constraints.Transitions.SEMI).to(DTYPE).to(device))
        if theta is not None and hasattr(self, "_contextual_transition_matrix"):
            A = torch.log(self._contextual_transition_matrix(theta) + 1e-12)

        # --- D ---
        D = torch.log(constraints.sample_probs(α, (self.n_states, self.max_duration)).to(DTYPE).to(device))
        if theta is not None and hasattr(self, "_contextual_duration_pdf"):
            D = torch.log(self._contextual_duration_pdf(theta) + 1e-12)

        # --- Emission ---
        emission_pdf = self.sample_emission_pdf(X)
        if theta is not None and hasattr(self, "_contextual_emission_pdf"):
            emission_pdf = self._contextual_emission_pdf(X, theta)

        params = {"pi": pi, "A": A, "D": D, "emission_pdf": emission_pdf}

        if inplace:
            if hasattr(self, "pi"): self.pi.data.copy_(pi)
            if hasattr(self, "A"): self.A.data.copy_(A)
            if hasattr(self, "D"): self.D.data.copy_(D)
            self._params["emission_pdf"] = emission_pdf

        return params

    # HSMM precessing
    def fit(
        self,
        X: torch.Tensor,
        tol: float = 1e-4,
        max_iter: int = 15,
        n_init: int = 1,
        post_conv_iter: int = 1,
        ignore_conv: bool = False,
        sample_B_from_X: bool = False,
        verbose: bool = True,
        plot_conv: bool = False,
        lengths: Optional[List[int]] = None,
        theta: Optional[torch.Tensor] = None
    ):
        """
        Fit the HSMM using EM with full context-awareness and vectorized emissions.
        """
        device = next(self.parameters(), torch.tensor(0.0)).device

        if sample_B_from_X:
            self._params['emission_pdf'] = self.sample_emission_pdf(X)

        if theta is None and self.encoder is not None:
            theta = self.encode_observations(X)

        X_valid = self.to_observations(X, lengths, theta)

        valid_theta = None
        if theta is not None:
            if theta.ndim == 2 and theta.shape[0] == len(X_valid.sequence):
                valid_theta = theta
            else:
                valid_theta = theta.repeat(len(X_valid.sequence), 1)

        # Convergence handler
        self.conv = ConvergenceHandler(
            tol=tol,
            max_iter=max_iter,
            n_init=n_init,
            post_conv_iter=post_conv_iter,
            verbose=verbose
        )

        best_state, best_score = None, -float("inf")
        seq_tensor = torch.cat(X_valid.sequence, dim=0).to(dtype=DTYPE, device=device)
        seq_offsets = [0] + list(torch.cumsum(torch.tensor(X_valid.lengths), dim=0).numpy())

        for run_idx in range(n_init):
            if verbose:
                print(f"\n=== Run {run_idx + 1}/{n_init} ===")

            if run_idx > 0:
                sampled = self.sample_model_params(X)
                self._A_logits.copy_(sampled['A'])
                self._D_logits.copy_(sampled['D'])
                self._pi_logits.copy_(sampled['pi'])
                self._params['emission_pdf'] = sampled['emission_pdf']

            base_ll = self._compute_log_likelihood(X_valid).sum()
            self.conv.push_pull(base_ll, 0, run_idx)

            for it in range(1, max_iter + 1):
                self._params['emission_pdf'] = self._contextual_emission_pdf(X_valid, valid_theta)
                A_logits = self._contextual_transition_matrix(valid_theta)
                D_logits = self._contextual_duration_pdf(valid_theta)

                if A_logits is not None:
                    self._A_logits.copy_(A_logits)
                if D_logits is not None:
                    self._D_logits.copy_(D_logits)

                new_params = self._estimate_model_params(X_valid, valid_theta)
                if 'emission_pdf' in new_params:
                    self._params['emission_pdf'] = new_params['emission_pdf']
                if 'pi' in new_params:
                    self._pi_logits.copy_(new_params['pi'])
                if 'A' in new_params:
                    self._A_logits.copy_(new_params['A'])
                if 'D' in new_params:
                    self._D_logits.copy_(new_params['D'])

                pdf = self._params['emission_pdf']
                if pdf is None or not hasattr(pdf, 'log_prob'):
                    raise RuntimeError("Emission PDF not initialized or unsupported for vectorized log_prob.")
                all_log_probs = pdf.log_prob(seq_tensor.unsqueeze(1))  # (total_T, K)
                X_valid.log_probs = [all_log_probs[seq_offsets[i]:seq_offsets[i+1]] for i in range(len(X_valid.lengths))]

                curr_ll = self._compute_log_likelihood(X_valid).sum()
                converged = self.conv.push_pull(curr_ll, it, run_idx)
                if converged and not ignore_conv:
                    if verbose:
                        print(f"[Run {run_idx + 1}] Converged at iteration {it}.")
                    break

            run_score = float(self._compute_log_likelihood(X_valid).sum().item())
            if run_score > best_score:
                best_score = run_score
                best_state = {
                    'pi': self._pi_logits.clone(),
                    'A': self._A_logits.clone(),
                    'D': self._D_logits.clone(),
                    'emission_pdf': self._params['emission_pdf']
                }

        if best_state is not None:
            self._pi_logits.copy_(best_state['pi'])
            self._A_logits.copy_(best_state['A'])
            self._D_logits.copy_(best_state['D'])
            self._params['emission_pdf'] = best_state['emission_pdf']

        if plot_conv and hasattr(self, 'conv'):
            self.conv.plot_convergence()

        return self

    def predict(
        self,
        X: torch.Tensor,
        lengths: Optional[list[int]] = None,
        algorithm: Literal["map", "viterbi"] = "viterbi",
        context: Optional[torch.Tensor] = None,
        batch_size: int = 256,
    ) -> list[torch.Tensor]:

        X_valid = self.to_observations(X, lengths)
        device = next(self.parameters(), torch.tensor(0.0)).device
        algorithm = algorithm.lower()

        if context is not None and context.dim() > 1:
            context = context.mean(dim=0, keepdim=True)

        seq_tensor = torch.cat(X_valid.sequence, dim=0).to(dtype=DTYPE, device=device)
        seq_offsets = [0] + list(torch.cumsum(torch.tensor(X_valid.lengths), dim=0).numpy())
        B = len(X_valid.lengths)
        K = getattr(self, "n_states", None)

        pdf = self._params.get("emission_pdf", None)
        if pdf is None:
            raise RuntimeError("Emission PDF not initialized.")

        total_T = seq_tensor.shape[0]
        log_probs_chunks = []

        if hasattr(pdf, "log_prob"):
            for start in range(0, total_T, batch_size):
                end = min(start + batch_size, total_T)
                chunk = seq_tensor[start:end].unsqueeze(1)  # (chunk_T, 1, F) expected by your pdf
                lp = pdf.log_prob(chunk)

                if lp.ndim > 2:
                    lp = lp.sum(dim=list(range(2, lp.ndim)))
                log_probs_chunks.append(lp)
            all_log_probs = torch.cat(log_probs_chunks, dim=0)  # (total_T, K)
        else:
            mu_var_or_logits = self.emission_module(context)
            if isinstance(mu_var_or_logits, tuple):
                mu, var = mu_var_or_logits
                var = var.clamp(min=1e-6)
                dist = torch.distributions.Normal(mu, var.sqrt())
                for start in range(0, total_T, batch_size):
                    end = min(start + batch_size, total_T)
                    chunk = seq_tensor[start:end].unsqueeze(1)
                    lp = dist.log_prob(chunk)
                    lp = lp.sum(dim=-1)
                    log_probs_chunks.append(lp)
                all_log_probs = torch.cat(log_probs_chunks, dim=0)
            else:
                logits = mu_var_or_logits
                log_soft = F.log_softmax(logits, dim=-1)
                all_log_probs = log_soft[seq_tensor.long(), :]

        X_valid.log_probs = [all_log_probs[seq_offsets[i]:seq_offsets[i + 1]] for i in range(B)]
        K = K or all_log_probs.shape[-1]

        if hasattr(self.transition_module, "log_probs"):
            A = self.transition_module.log_probs(context=context)
        else:
            trans_out = self.transition_module(context=context)
            if torch.all(trans_out >= 0) and torch.allclose(trans_out.sum(dim=-1), torch.ones_like(trans_out.sum(dim=-1)), atol=1e-4):
                A = torch.log(trans_out + 1e-12)
            else:
                A = torch.log_softmax(trans_out, dim=-1)

        if hasattr(self.duration_module, "log_probs"):
            log_D = self.duration_module.log_probs(context=context)  # (K, max_duration)
        elif hasattr(self.duration_module, "logits"):
            log_D = torch.log_softmax(self.duration_module.logits, dim=-1)
        else:
            log_D = None

        max_T = max(X_valid.lengths)
        log_B = torch.full((B, max_T, K), -torch.inf, device=device)
        for i, seq_log_probs in enumerate(X_valid.log_probs):
            log_B[i, :seq_log_probs.shape[0]] = seq_log_probs

        mask = torch.arange(max_T, device=device).unsqueeze(0) < \
               torch.tensor(X_valid.lengths, device=device).unsqueeze(1)

        if algorithm == "map":
            alpha = torch.full((B, max_T, K), -torch.inf, device=device)
            alpha[:, 0] = log_B[:, 0]
            for t in range(1, max_T):
                prev = alpha[:, t - 1].unsqueeze(2) + A
                if log_D is not None:
                    dur_idx = min(t, log_D.shape[1] - 1)
                    prev = prev + log_D[:, dur_idx].unsqueeze(0)
                alpha[:, t] = torch.logsumexp(prev, dim=1) + log_B[:, t]
                alpha[:, t] = torch.where(mask[:, t].unsqueeze(-1),
                                          alpha[:, t],
                                          alpha[:, t - 1])
            decoded = torch.argmax(alpha, dim=-1)
        elif algorithm == "viterbi":
            delta = torch.full((B, max_T, K), -torch.inf, device=device)
            psi = torch.zeros((B, max_T, K), dtype=torch.long, device=device)
            delta[:, 0] = log_B[:, 0]
            for t in range(1, max_T):
                scores = delta[:, t - 1].unsqueeze(2) + A  # (B, K, K)
                if log_D is not None:
                    dur_idx = min(t, log_D.shape[1] - 1)
                    scores = scores + log_D[:, dur_idx].unsqueeze(0)
                psi[:, t] = torch.argmax(scores, dim=1)
                delta[:, t] = torch.max(scores, dim=1).values + log_B[:, t]
                delta[:, t] = torch.where(mask[:, t].unsqueeze(-1), delta[:, t], delta[:, t - 1])

            decoded = torch.zeros((B, max_T), dtype=torch.long, device=device)
            decoded[:, -1] = torch.argmax(delta[:, -1], dim=-1)
            for t in range(max_T - 2, -1, -1):
                decoded[:, t] = torch.gather(psi[:, t + 1], 1, decoded[:, t + 1].unsqueeze(1)).squeeze(1)
                decoded[:, t] = torch.where(mask[:, t], decoded[:, t], decoded[:, t + 1])

        else:
            raise ValueError(f"Unknown decoding algorithm '{algorithm}'.")

        return [decoded[i, :L].detach().cpu() for i, L in enumerate(X_valid.lengths)]

    def score(self, X: torch.Tensor, lengths: Optional[List[int]] = None, by_sample: bool = True) -> torch.Tensor:
        """
        Compute log-likelihood(s) of input sequence(s) under the HSMM.

        Args:
            X: Input sequences, shape (B, T, F) or flattened.
            lengths: List of sequence lengths for variable-length sequences.
            by_sample: If True, return per-sequence log-likelihood; else sum.

        Returns:
            Tensor of log-likelihoods.
        """
        obs = self.to_observations(X, lengths)
        device = next(self.parameters(), torch.tensor(0.0)).device
        B = len(obs.lengths)
        max_T = max(obs.lengths)
        
        pdf = self._params.get("emission_pdf")
        if pdf is None or not hasattr(pdf, "log_prob"):
            raise RuntimeError("Emission PDF must be initialized with log_prob method.")
        
        # Concatenate all observations and compute log-probabilities
        all_seq = torch.cat(obs.sequence, dim=0).to(dtype=DTYPE, device=device)
        all_log_probs = pdf.log_prob(all_seq.unsqueeze(1))  # (total_T, K)
        
        # Compute offsets on GPU to avoid host transfer
        seq_lengths = torch.tensor([0] + obs.lengths, device=device).cumsum(dim=0)
        log_B = torch.full((B, max_T, all_log_probs.shape[-1]), -torch.inf, device=device)
        
        for i in range(B):
            start, end = seq_lengths[i].item(), seq_lengths[i+1].item()
            L = obs.lengths[i]
            log_B[i, :L] = all_log_probs[start:end]
        
        K = all_log_probs.shape[-1]

        # Transition matrix
        if hasattr(self, "transition"):
            if hasattr(self.transition, "contextual_logits"):
                log_A = self.transition.contextual_logits()
            elif hasattr(self.transition, "logits"):
                log_A = torch.log_softmax(self.transition.logits, dim=-1)
            else:
                log_A = torch.full((K, K), 1/K, device=device).log()
        else:
            log_A = torch.full((K, K), 1/K, device=device).log()
        
        # Duration log-probabilities
        dur_pdf = self._params.get("duration_pdf", None)
        if dur_pdf is not None and hasattr(dur_pdf, "log_prob"):
            log_D = dur_pdf.log_prob(torch.arange(1, max_T+1, device=device))  # (max_T,)
        else:
            log_D = torch.zeros(max_T, device=device, dtype=DTYPE)
        
        # Forward recursion
        log_alpha = torch.full((B, max_T, K), -torch.inf, device=device, dtype=DTYPE)
        log_alpha[:, 0] = log_B[:, 0]
        
        for t in range(1, max_T):
            prev = log_alpha[:, t-1].unsqueeze(2) + log_A  # (B, K, K)
            log_alpha[:, t] = torch.logsumexp(prev, dim=1) + log_B[:, t] + log_D[t-1]
        
        # Mask invalid positions
        mask = torch.arange(max_T, device=device).unsqueeze(0) < torch.tensor(obs.lengths, device=device).unsqueeze(1)
        log_alpha = log_alpha * mask.unsqueeze(-1) + (~mask).unsqueeze(-1) * (-torch.inf)
        
        # Extract log-likelihoods at sequence ends
        end_idx = torch.tensor(obs.lengths, device=device) - 1
        seq_ll = torch.logsumexp(log_alpha[torch.arange(B), end_idx], dim=-1)

        return seq_ll.detach().cpu() if by_sample else seq_ll.sum().detach().cpu()

    def ic(
        self,
        X: torch.Tensor,
        criterion: constraints.InformCriteria = constraints.InformCriteria.AIC,
        lengths: Optional[List[int]] = None,
        by_sample: bool = True
    ) -> torch.Tensor:
        """
        Compute an information criterion (AIC, BIC, etc.) for the HSMM.

        Args:
            X (torch.Tensor): Input sequences, shape (B, T, F) or (T, F).
            criterion (InformCriteria): Which information criterion to compute.
            lengths (Optional[List[int]]): Optional sequence lengths for variable-length batches.
            by_sample (bool): Whether to return per-sequence values or a scalar sum.

        Returns:
            torch.Tensor: Information criterion values (per-sample or aggregated).
        """
        device = next(self.parameters()).device

        # --- Compute log-likelihood ---
        try:
            log_likelihood = self.score(X, lengths=lengths, by_sample=by_sample).to(device)
        except Exception as e:
            raise RuntimeError(f"Failed to compute log-likelihood: {e}")

        # --- Determine total number of observations ---
        if lengths is not None:
            n_obs = max(int(sum(lengths)), 1)
        elif X.ndim == 3:  # (B, T, F)
            n_obs = max(int(X.shape[0] * X.shape[1]), 1)
        else:
            n_obs = max(int(X.shape[0]), 1)

        # --- Retrieve model degrees of freedom ---
        dof = getattr(self, "dof", None)
        if dof is None:
            raise AttributeError("Model degrees of freedom ('dof') not defined. Please set 'self.dof' during initialization.")

        # --- Compute Information Criterion ---
        ic_value = constraints.compute_information_criteria(
            n_obs=n_obs,
            log_likelihood=log_likelihood,
            dof=dof,
            criterion=criterion
        )

        # --- Safety: ensure correct dtype and finite output ---
        if not isinstance(ic_value, torch.Tensor):
            ic_value = torch.tensor(ic_value, dtype=DTYPE, device=device)

        ic_value = ic_value.nan_to_num(nan=float('inf'), posinf=float('inf'), neginf=float('inf'))

        # --- Normalize or aggregate ---
        if by_sample and ic_value.ndim == 0:
            ic_value = ic_value.unsqueeze(0)

        return ic_value.detach().cpu()

    def _map(self, X: utils.Observations) -> List[torch.Tensor]:
        gamma, _, _ = self._compute_posteriors(X)
        if gamma is None:
            raise RuntimeError("Posterior probabilities could not be computed — model parameters uninitialized.")

        device = next(self.parameters()).device
        map_sequences: List[torch.Tensor] = []

        for tens in gamma:
            if tens is None or tens.numel() == 0:
                map_sequences.append(torch.empty(0, dtype=torch.long, device=device))
                continue

            # Clean NaNs and infs
            tens = tens.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

            # Row-wise normalization with safe fallback
            row_sums = tens.sum(dim=1, keepdim=True)
            row_sums = torch.where(row_sums == 0, torch.ones_like(row_sums), row_sums)
            normalized = tens / row_sums

            # Argmax for MAP decoding
            seq_map = normalized.argmax(dim=1).to(dtype=torch.long, device=device)
            map_sequences.append(seq_map)

        return map_sequences

    def _forward(self, X: utils.Observations) -> List[torch.Tensor]:
        device = next(self.parameters(), torch.tensor(0.0)).device
        pi, A, D = self.pi, self.A, self.D
        neg_inf = -torch.inf
        alpha_vec = []

        for seq_probs, seq_len in zip(X.log_probs, X.lengths):
            if seq_len == 0:
                alpha_vec.append(torch.empty((0, self.n_states, self.max_duration), device=device))
                continue

            log_alpha = torch.full((seq_len, self.n_states, self.max_duration), neg_inf, dtype=DTYPE, device=device)
            cumsum_emit = torch.vstack((torch.zeros((1, self.n_states), dtype=DTYPE, device=device),
                                        torch.cumsum(seq_probs, dim=0)))

            max_d = min(self.max_duration, seq_len)
            durations = torch.arange(1, max_d + 1, device=device)
            emit_sums = (cumsum_emit[durations] - cumsum_emit[0]).T
            log_alpha[0, :, :max_d] = pi.unsqueeze(-1) + D[:, :max_d] + emit_sums

            for t in range(1, seq_len):
                prev_alpha = log_alpha[t - 1]
                shifted = torch.cat([prev_alpha[:, 1:], torch.full((self.n_states, 1), neg_inf, dtype=DTYPE, device=device)], dim=1)
                trans = torch.logsumexp(prev_alpha[:, 0].unsqueeze(1) + A, dim=0)
                log_alpha[t] = torch.logsumexp(torch.stack([shifted + seq_probs[t].unsqueeze(-1),
                                                            D + trans.unsqueeze(-1)]), dim=0)

            alpha_vec.append(log_alpha)

        return alpha_vec

    def _backward(self, X: utils.Observations) -> List[torch.Tensor]:
        beta_vec = []
        A, D = self.A, self.D
        neg_inf = -torch.inf

        for seq_probs, seq_len in zip(X.log_probs, X.lengths):
            device = seq_probs.device
            if seq_len == 0:
                beta_vec.append(torch.empty((0, self.n_states, self.max_duration), device=device))
                continue

            log_beta = torch.full((seq_len, self.n_states, self.max_duration), neg_inf, dtype=DTYPE, device=device)
            log_beta[-1].fill_(0.0)

            cumsum_emit = torch.vstack((torch.zeros((1, self.n_states), dtype=DTYPE, device=device),
                                        torch.cumsum(seq_probs, dim=0)))
            durations_all = torch.arange(1, self.max_duration + 1, device=device)

            for t in reversed(range(seq_len - 1)):
                max_d = min(self.max_duration, seq_len - t)
                durations = durations_all[:max_d]

                emit_sums = (cumsum_emit[t + durations] - cumsum_emit[t]).T
                dur_lp = D[:, :max_d]
                beta_next = log_beta[t + durations - 1, :, 0].T

                log_beta[t, :, 0] = torch.logsumexp(emit_sums + dur_lp + beta_next, dim=1)

                if self.max_duration > 1:
                    log_beta[t, :, 1:] = log_beta[t + 1, :, :-1] + seq_probs[t + 1].unsqueeze(-1)

            beta_vec.append(log_beta)

        return beta_vec

    def _compute_posteriors(self, X: utils.Observations) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Compute posterior expectations for HSMM sequences in log-domain:
            - gamma_vec: state marginals (T, K)
            - xi_vec: transition marginals (T-1, K, K)
            - eta_vec: state-duration marginals (T, K, Dmax)
        """
        alpha_list = self._forward(X)
        beta_list = self._backward(X)

        gamma_vec: List[torch.Tensor] = []
        xi_vec: List[torch.Tensor] = []
        eta_vec: List[torch.Tensor] = []

        device = next(self.parameters(), torch.tensor(0.0)).device
        K, Dmax = self.n_states, self.max_duration

        for seq_idx, (seq_probs, seq_len, alpha, beta) in enumerate(zip(X.log_probs, X.lengths, alpha_list, beta_list)):
            if seq_len == 0 or alpha.numel() == 0 or beta.numel() == 0:
                gamma_vec.append(torch.zeros((seq_len, K), dtype=DTYPE, device=device))
                eta_vec.append(torch.zeros((seq_len, K, Dmax), dtype=DTYPE, device=device))
                xi_vec.append(torch.zeros((max(seq_len - 1, 0), K, K), dtype=DTYPE, device=device))
                continue

            alpha = alpha.to(device=device, dtype=DTYPE)
            beta = beta.to(device=device, dtype=DTYPE)

            # ----- gamma: state marginals -----
            log_gamma = torch.logsumexp(alpha + beta, dim=2)  # sum over durations
            gamma = constraints.log_normalize(log_gamma, dim=1).exp().clamp_min(0.0)
            gamma = gamma / gamma.sum(dim=1, keepdim=True).clamp_min(1.0)
            gamma_vec.append(gamma)

            # ----- eta: state-duration marginals -----
            log_eta = alpha + beta
            eta = constraints.log_normalize(log_eta, dim=(1, 2)).exp().clamp_min(0.0)
            eta_vec.append(eta)

            # ----- xi: transition marginals -----
            if seq_len > 1:
                alpha_start = alpha[:-1, :, 0]  # durations starting a transition
                trans_alpha = alpha_start.unsqueeze(2) + self.A.unsqueeze(0).to(device=device, dtype=DTYPE)

                beta_next = beta[1:] + self.D.unsqueeze(0).to(device=device, dtype=DTYPE)
                dur_beta_sum = torch.logsumexp(beta_next, dim=2)

                log_xi = trans_alpha + dur_beta_sum.unsqueeze(1)
                xi = constraints.log_normalize(log_xi, dim=(1, 2)).exp().clamp_min(0.0)
                xi_vec.append(xi)
            else:
                xi_vec.append(torch.zeros((0, K, K), dtype=DTYPE, device=device))

        return gamma_vec, xi_vec, eta_vec

    def _estimate_model_params(self, X: utils.Observations, theta: Optional[utils.ContextualVariables] = None) -> Dict[str, Any]:
        """
        M-step: Estimate updated HSMM parameters from posterior expectations.
        """
        gamma_list, xi_list, eta_list = self._compute_posteriors(X)
        device = next(self.parameters(), torch.tensor(0.0)).device

        pi_stack = torch.stack([g[0] for g in gamma_list], dim=1).to(device=device, dtype=DTYPE)  # (n_states, n_sequences)
        new_pi = constraints.log_normalize(torch.log(pi_stack.sum(dim=1)), dim=0)

        # -------------------------------
        # A (state transition matrix)
        # -------------------------------
        xi_valid = [x for x in xi_list if x.numel() > 0]
        if xi_valid:
            xi_cat = torch.cat(xi_valid, dim=0).to(device=device, dtype=DTYPE)
            new_A = constraints.log_normalize(torch.logsumexp(xi_cat, dim=0), dim=1)
        else:
            new_A = self.A.clone()

        # -------------------------------
        # D (duration distributions)
        # -------------------------------
        eta_cat = torch.cat([e for e in eta_list], dim=0).to(device=device, dtype=DTYPE)
        new_D = constraints.log_normalize(torch.logsumexp(eta_cat, dim=0), dim=1)

        # -------------------------------
        # Emission PDF (contextual or base)
        # -------------------------------
        if X.sequence:
            all_X = torch.cat(X.sequence, dim=0).to(device=device, dtype=DTYPE)
            all_gamma = torch.cat([g for g in gamma_list], dim=0).to(device=device, dtype=DTYPE)
            new_pdf = self._estimate_emission_pdf(all_X, all_gamma, theta)
        else:
            new_pdf = self._params.get('emission_pdf')

        return {
            'pi': new_pi,
            'A': new_A,
            'D': new_D,
            'emission_pdf': new_pdf
        }

    def _viterbi(self, X: utils.Observations, duration_weight: float = 0.0) -> list[torch.Tensor]:
        """
        Duration-explicit Viterbi decoding for HSMM in log-domain.

        Args:
            X: Observation container.
            duration_weight: [0.0–1.0] strength of duration regularization.

        Returns:
            List of most probable state sequences per batch.
        """
        device = next(self.parameters(), torch.tensor(0.0)).device
        K, Dmax = self.n_states, self.max_duration
        pi, A, D = self.pi.to(device), self.A.to(device), self.D.to(device)
        neg_inf = -torch.inf

        batch_size = len(X.log_probs)
        lengths = X.lengths
        max_len = max(lengths) if lengths else 0

        V = torch.full((batch_size, max_len, K), neg_inf, dtype=DTYPE, device=device)
        best_prev = torch.full((batch_size, max_len, K), -1, dtype=torch.int64, device=device)
        best_dur = torch.zeros((batch_size, max_len, K), dtype=torch.int64, device=device)

        dur_indices = torch.arange(1, Dmax + 1, device=device)

        for b, (seq_probs, T) in enumerate(zip(X.log_probs, lengths)):
            if T == 0:
                continue
            seq_probs = seq_probs.to(device=device, dtype=DTYPE)
            cumsum_emit = torch.cat([torch.zeros((1, K), dtype=DTYPE, device=device),
                                     torch.cumsum(seq_probs, dim=0)])

            if duration_weight > 0.0:
                dur_mean = torch.softmax(D, dim=1) @ dur_indices.float()
                dur_penalty = -((dur_indices.unsqueeze(0) - dur_mean.unsqueeze(1)) ** 2) / (2 * Dmax)
                dur_lp = (1 - duration_weight) * D + duration_weight * dur_penalty
            else:
                dur_lp = D

            for t in range(T):
                max_d = min(Dmax, t + 1)
                durations = dur_indices[:max_d]
                starts = t - durations + 1
                emit_sums = (cumsum_emit[t + 1] - cumsum_emit[starts]).T  # (K, max_d)
                dur_scores = dur_lp[:, :max_d]

                if t == 0:
                    scores = pi.unsqueeze(1) + dur_scores + emit_sums
                    prev_idx = torch.full_like(scores, -1, dtype=torch.int64)
                else:
                    prev_V = V[b, t - durations]  # (max_d, K)
                    scores_plus_trans = prev_V.unsqueeze(2) + A.unsqueeze(0)  # (max_d, K, K)
                    scores_max, argmax_prev = torch.max(scores_plus_trans, dim=1)  # (max_d, K)
                    scores = scores_max.T + dur_scores + emit_sums  # (K, max_d)
                    prev_idx = argmax_prev.T  # (K, max_d)

                best_score, best_d_idx = torch.max(scores, dim=1)
                V[b, t] = best_score
                best_dur[b, t] = durations[best_d_idx]
                best_prev[b, t] = prev_idx[torch.arange(K, device=device), best_d_idx]

        # -------- Backtrace --------
        paths: list[torch.Tensor] = []
        for b, T in enumerate(lengths):
            if T == 0:
                paths.append(torch.empty((0,), dtype=torch.int64, device=device))
                continue

            t = T - 1
            cur_state = int(torch.argmax(V[b, t]).item())
            segments = []

            while t >= 0:
                d = int(best_dur[b, t, cur_state].item())
                if d <= 0:
                    break
                start = t - d + 1
                segments.append((start, t, cur_state))
                prev_state = int(best_prev[b, t, cur_state].item())
                if prev_state < 0 or start <= 0:
                    break
                t = start - 1
                cur_state = prev_state

            segments.reverse()
            if not segments:
                paths.append(torch.full((T,), cur_state, dtype=torch.int64, device=device))
                continue

            seq_path = torch.cat([
                torch.full((end - start + 1,), st, dtype=torch.int64, device=device)
                for start, end, st in segments
            ])
            paths.append(seq_path)

        return paths

    def _compute_log_likelihood(self, X: utils.Observations) -> torch.Tensor:
        """
        Compute log-likelihoods for a batch of observation sequences using the forward algorithm.

        Args:
            X (utils.Observations): Observation container with sequences of shape (T, F) or (B, T, F).

        Returns:
            torch.Tensor: Log-likelihoods for each sequence in the batch (shape: B).
        """
        log_alpha_vec = self._forward(X)
        if not log_alpha_vec:
            raise RuntimeError("Forward pass returned empty results. Model may be uninitialized.")

        device = next(self.parameters(), torch.tensor(0.0)).device
        neg_inf = -torch.inf
        log_likelihoods = []

        for log_alpha in log_alpha_vec:
            if log_alpha is None or log_alpha.numel() == 0:
                log_likelihoods.append(torch.tensor(neg_inf, device=device))
                continue

            # Ensure finite values
            log_alpha = log_alpha.nan_to_num(nan=-1e8, posinf=-1e8, neginf=-1e8)

            # Take logsumexp over last time step
            last_step = log_alpha[-1] if log_alpha.ndim >= 2 else log_alpha
            ll = torch.logsumexp(last_step, dim=-1)

            # Collapse any remaining dimensions if needed
            if ll.ndim > 0:
                ll = torch.logsumexp(ll, dim=0)

            # Final safety: ensure finite scalar
            if not torch.isfinite(ll):
                ll = torch.tensor(neg_inf, device=device)

            log_likelihoods.append(ll)

        return torch.stack(log_likelihoods, dim=0)
