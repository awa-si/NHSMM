# models/hsmm.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Any, Literal, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, Categorical

from nhsmm.defaults import Emission, Duration, Transition, DTYPE, HSMMError
from nhsmm import utils, constraints, SeedGenerator, ConvergenceHandler


class HSMM(ABC):
    """
    Hidden Semi-Markov Model (HSMM) base class.

    Supports:
        - Emissions: state-dependent observation likelihoods.
        - Transitions: contextual transition probabilities.
        - Durations: explicit duration modeling per state.
        - Optional neural context encoding.

    Key methods to implement/override:
        - forward(X)
        - _compute_log_likelihood(X)
        - score(X, lengths=None, by_sample=True)
        - sample(T)
        - information(X, criterion, lengths=None)
    """

    def __init__(
        self,
        n_states: int,
        n_features: int,
        max_duration: int,
        alpha: float = 1.0,
        seed: Optional[int] = None,
        context_dim: Optional[int] = None,
        transition_constraint: Any = None,
        min_covar: float = 1e-3,
    ):
        super().__init__()

        self.n_states = n_states
        self.n_features = n_features
        self.max_duration = max_duration
        self.min_covar = min_covar
        self.alpha = alpha

        self._params: Dict[str, Any] = {}
        self._seed_gen = SeedGenerator(seed)
        self._context: Optional[torch.Tensor] = None

        self.encoder: Optional[nn.Module] = None
        self.emission_module = Emission(n_states, n_features, min_covar, context_dim)
        self.duration_module = Duration(n_states, max_duration, context_dim)
        self.transition_module = Transition(n_states, context_dim)

        self.transition_constraint = transition_constraint or constraints.Transitions.SEMI

        self._init_buffers()
        self._params['emission_pdf'] = self.sample_emission_pdf(None)

    def _init_buffers(self):
        device = next(self.buffers(), torch.tensor(0., dtype=DTYPE)).device

        def sample_logits(shape):
            if len(shape) == 1:
                probs = constraints.sample_probs(self.alpha, shape)
            elif len(shape) == 2:
                if shape[1] == getattr(self, 'max_duration', None):
                    probs = constraints.sample_probs(self.alpha, shape)
                else:
                    probs = constraints.sample_transition(self.alpha, shape[0], self.transition_constraint)
            else:
                raise ValueError(f"Unsupported shape {shape} for logits sampling.")
            logits = torch.log(probs.clamp_min(1e-12)).to(device=device, dtype=DTYPE)
            return logits, probs.shape

        # Base parameters
        init_logits, init_shape = sample_logits((self.n_states,))
        transition_logits, transition_shape = sample_logits((self.n_states, self.n_states))
        duration_logits, duration_shape = sample_logits((self.n_states, self.max_duration))

        self.register_buffer("_transition_logits", transition_logits)
        self.register_buffer("_duration_logits", duration_logits)
        self.register_buffer("_init_logits", init_logits)

        self._transition_shape = transition_shape
        self._duration_shape = duration_shape
        self._init_batch_shape = init_shape

        # Optional super-states
        n_super_states = getattr(self, "n_super_states", 0)
        if n_super_states > 1:
            super_init_logits, super_init_shape = sample_logits((n_super_states,))
            super_transition_logits, super_transition_shape = sample_logits((n_super_states, n_super_states))
            self.register_buffer("_super_init_logits", super_init_logits)
            self.register_buffer("_super_transition_logits", super_transition_logits)

            self._super_init_batch_shape = super_init_shape
            self._super_transition_batch_shape = super_transition_shape

        # Initialization snapshot
        summary = [init_logits.mean(), transition_logits.mean(), duration_logits.mean()]
        if n_super_states > 1:
            summary += [super_init_logits.mean(), super_transition_logits.mean()]

        self.register_buffer("_init_prior_snapshot", torch.stack(summary).detach())

    @property
    def seed(self) -> Optional[int]:
        return self._seed_gen.seed

    @property
    def init_logits(self) -> torch.Tensor:
        return self._init_logits

    @init_logits.setter
    def init_logits(self, logits: torch.Tensor):
        logits = logits.to(device=self._init_logits.device, dtype=DTYPE)
        if logits.shape != (self.n_states,):
            raise ValueError(f"init_logits logits must have shape ({self.n_states},) but got {tuple(logits.shape)}")
        norm_val = logits.logsumexp(0)
        if not torch.allclose(norm_val, torch.tensor(0.0, dtype=DTYPE, device=logits.device), atol=1e-6):
            raise ValueError(f"init_logits logits must normalize (logsumexp==0); got {norm_val.item():.3e}")
        self._init_logits.copy_(logits)

    @property
    def init_probs(self) -> torch.Tensor:
        return self._init_logits.softmax(0)

    @property
    def transition_logits(self) -> torch.Tensor:
        return self._transition_logits

    @transition_logits.setter
    def transition_logits(self, logits: torch.Tensor):
        logits = logits.to(device=self._transition_logits.device, dtype=DTYPE)
        if logits.shape != (self.n_states, self.n_states):
            raise ValueError(f"transition_logits logits must have shape ({self.n_states},{self.n_states})")
        row_norm = logits.logsumexp(1)
        if not torch.allclose(row_norm, torch.zeros_like(row_norm), atol=1e-6):
            raise ValueError(f"Rows of transition_logits logits must normalize (logsumexp==0); got {row_norm}")
        if not constraints.is_valid_transition(logits, self.transition_constraint):
            raise ValueError("transition_logits logits do not satisfy transition constraints")
        self._transition_logits.copy_(logits)

    @property
    def transition_probs(self) -> torch.Tensor:
        return self._transition_logits.softmax(-1)

    @property
    def duration_logits(self) -> torch.Tensor:
        return self._duration_logits

    @duration_logits.setter
    def duration_logits(self, logits: torch.Tensor):
        logits = logits.to(device=self._duration_logits.device, dtype=DTYPE)
        if logits.shape != (self.n_states, self.max_duration):
            raise ValueError(f"duration_logits logits must have shape ({self.n_states},{self.max_duration})")
        row_norm = logits.logsumexp(1)
        if not torch.allclose(row_norm, torch.zeros_like(row_norm), atol=1e-6):
            raise ValueError(f"Rows of duration_logits logits must normalize (logsumexp==0); got {row_norm}")
        self._duration_logits.copy_(logits)

    @property
    def duration_probs(self) -> torch.Tensor:
        return self._duration_logits.softmax(-1)

    @property
    def super_init(self):
        if hasattr(self, "_super_init_logits"):
            return self._super_init_logits
        return None

    @property
    def super_transition(self):
        if hasattr(self, "_super_transition_logits"):
            return self._super_transition_logits
        return None

    @property
    def pdf(self) -> Any:
        return self._params.get('emission_pdf')

    @property
    @abstractmethod
    def dof(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def sample_emission_pdf(self, X: Optional[torch.Tensor] = None) -> Distribution:
        raise NotImplementedError

    @abstractmethod
    def _estimate_emission_pdf(
        self,
        X: torch.Tensor,
        posterior: torch.Tensor,
        theta: Optional[utils.ContextualVariables] = None
    ) -> Distribution:
        """
        Estimate the emission distribution for the HSMM M-step.

        Args:
            X: Observations tensor (total_T, F)
            posterior: Posterior state responsibilities (total_T, K)
            theta: Optional contextual variables (total_T, H) or (H,)

        Returns:
            transition_logits torch.distributions.Distribution object representing the emission probabilities.
        
        Notes:
            - Should handle both base and context-conditioned emissions.
            - Posterior-weighted statistics (e.g., mean/variance for Gaussian) should be computed here.
            - Must return a valid Distribution supporting `.log_prob` for vectorized computation.
        """
        raise NotImplementedError(
            "Subclasses must implement _estimate_emission_pdf. "
            "It should return a Distribution supporting `.log_prob` and handle optional context theta."
        )


    def _combine_context(
        self,
        X: Optional[Observations] = None,
        theta: Optional[ContextualVariables] = None,
        reduce: bool = False
    ) -> Optional[torch.Tensor]:
        """
        Combine observation features with contextual variables into a consistent batch tensor.

        Args:
            X: Observations (optional)
            theta: ContextualVariables (optional)
            reduce: If True, collapse time dimension via mean features

        Returns:
            Tensor [B, T, F+H] or [B, 1, F+H] if reduced, None if no input
        """
        if X is None and theta is None:
            return None

        # Pad sequences and extract masks
        X_batch, mask_X = (X.pad_sequences(return_mask=True) if X else (None, None))
        if theta:
            if getattr(theta, "time_dependent", False):
                theta_batch, mask_theta = theta.pad_sequences(return_mask=True)
            else:
                theta_tensor = theta.cat(dim=-1).unsqueeze(1)
                theta_batch, mask_theta = theta_tensor, None
        else:
            theta_batch, mask_theta = None, None

        # Determine batch and time dimensions
        B = X_batch.shape[0] if X_batch is not None else theta_batch.shape[0]
        T = max(X_batch.shape[1] if X_batch is not None else 0,
                theta_batch.shape[1] if theta_batch is not None else 0)

        # Pad sequences to same length
        if X_batch is not None and X_batch.shape[1] != T:
            X_batch = F.pad(X_batch, (0, 0, 0, T - X_batch.shape[1]))
        if theta_batch is not None and theta_batch.shape[1] != T:
            theta_batch = F.pad(theta_batch, (0, 0, 0, T - theta_batch.shape[1]))

        # Concatenate along feature dimension
        combined = torch.cat([t for t in [X_batch, theta_batch] if t is not None], dim=-1)

        # Optional reduction along time dimension
        if reduce:
            mask = mask_X if mask_X is not None else torch.ones(B, T, dtype=torch.bool, device=combined.device)
            combined = combined.masked_fill(~mask.unsqueeze(-1), 0.0)
            lengths = mask.sum(1).unsqueeze(-1).clamp(min=1)
            combined = combined.sum(1, keepdim=True) / lengths

        return torch.nan_to_num(combined, nan=0.0, posinf=1e8, neginf=-1e8)

    def _contextual_emission_pdf(
        self,
        X: Observations,
        theta: Optional[ContextualVariables] = None
    ) -> torch.Tensor:
        """
        Compute context-modulated emission log-probabilities for batched sequences.

        Returns: [B, T, K]
        """
        X_batch, mask = X.pad_sequences(return_mask=True)
        theta_batch = self._combine_context(X=None, theta=theta)
        base_logp = self.emission_module.log_prob(X_batch, theta_batch)
        base_logp = torch.clamp(torch.nan_to_num(base_logp, nan=-1e8, posinf=-1e8, neginf=-1e8), -1e6, 1e6)

        # Apply context delta if available
        if theta is not None:
            delta = self.emission_module._apply_context(base_logp, theta_batch)
            logp = base_logp + delta
        else:
            logp = base_logp

        # Mask padding positions
        logp = logp.masked_fill(~mask.unsqueeze(-1), -1e8)
        return F.log_softmax(logp, dim=-1)

    def _contextual_duration_pdf(
        self,
        theta: Optional[ContextualVariables] = None
    ) -> torch.Tensor:
        """
        Compute context-modulated duration log-probabilities.

        Returns: [B, K, max_duration]
        """
        base_duration = self._duration_logits.unsqueeze(0)  # [1, K, max_duration]
        theta_batch = self._combine_context(X=None, theta=theta)
        B = theta_batch.shape[0] if theta_batch is not None else 1
        log_duration = self.duration_module._apply_context(
            base_duration.expand(B, *base_duration.shape[1:]), theta_batch
        )
        log_duration = torch.nan_to_num(log_duration, nan=-1e8, posinf=-1e8, neginf=-1e8)
        return log_duration - torch.logsumexp(log_duration, dim=2, keepdim=True)

    def _contextual_transition_matrix(
        self,
        theta: Optional[ContextualVariables] = None
    ) -> torch.Tensor:
        """
        Compute context-modulated transition log-probabilities.

        Returns: [B, K, K]
        """
        base_transition = self._transition_logits.unsqueeze(0)  # [1, K, K]
        theta_batch = self._combine_context(X=None, theta=theta)
        B = theta_batch.shape[0] if theta_batch is not None else 1
        log_transition = self.transition_module._apply_context(
            base_transition.expand(B, *base_transition.shape[1:]), theta_batch
        )

        # Optional transition constraints
        if hasattr(self, "transition_constraint"):
            mask = constraints.mask_invalid_transitions(self.n_states, self.transition_constraint).to(log_transition.device)
            log_transition = log_transition.masked_fill(~mask.unsqueeze(0), -1e8)

        log_transition = torch.nan_to_num(log_transition, nan=-1e8, posinf=-1e8, neginf=-1e8)
        return log_transition - torch.logsumexp(log_transition, dim=2, keepdim=True)

    def _estimate_model_params(self, X: utils.Observations, theta: Optional[utils.ContextualVariables] = None) -> Dict[str, Any]:
        """
        M-step: Estimate updated HSMM parameters from posterior expectations.
        Supports optional context for emissions.
        """
        gamma_list, xi_list, eta_list = self._compute_posteriors(X)
        device = next(self.parameters(), torch.tensor(0.0)).device

        # -------------------------------
        # init_logits (initial state distribution)
        # -------------------------------
        pi_stack = torch.stack([g[0] for g in gamma_list], dim=1).to(device=device, dtype=DTYPE)  # (n_states, n_sequences)
        new_init = constraints.log_normalize(torch.log(pi_stack.sum(dim=1)), dim=0)

        # -------------------------------
        # transition_logits (state transition matrix)
        # -------------------------------
        xi_valid = [x for x in xi_list if x.numel() > 0]
        if xi_valid:
            xi_cat = torch.cat(xi_valid, dim=0).to(device=device, dtype=DTYPE)
            new_transition = constraints.log_normalize(torch.logsumexp(xi_cat, dim=0), dim=1)
        else:
            new_transition = self.transition_logits.clone()

        # -------------------------------
        # duration_logits (duration distributions)
        # -------------------------------
        eta_cat = torch.cat([e for e in eta_list], dim=0).to(device=device, dtype=DTYPE)
        new_duration = constraints.log_normalize(torch.logsumexp(eta_cat, dim=0), dim=1)

        # -------------------------------
        # Emission PDF (contextual or base)
        # -------------------------------
        new_pdf = self._params.get('emission_pdf')
        if X.sequence:
            all_X = torch.cat(X.sequence, dim=0).to(device=device, dtype=DTYPE)
            all_gamma = torch.cat([g for g in gamma_list], dim=0).to(device=device, dtype=DTYPE)

            if theta is not None:
                # Safely expand or reduce theta to match all_X
                if theta.ndim == 1:
                    theta_full = theta.view(1, -1).expand(all_X.shape[0], -1)
                elif theta.ndim == 2:
                    if theta.shape[0] == all_X.shape[0]:
                        theta_full = theta
                    else:
                        theta_full = theta.repeat(all_X.shape[0] // theta.shape[0], 1)
                elif theta.ndim == 3:
                    theta_full = theta[:, -1, :].repeat(all_X.shape[0] // theta.shape[0], 1)
                else:
                    raise ValueError(f"Unexpected theta shape {theta.shape}")

                new_pdf = self._contextual_emission_pdf(all_X, theta_full)
            else:
                new_pdf = self._estimate_emission_pdf(all_X, all_gamma, theta)

        return {
            'init_logits': new_init,
            'transition_logits': new_transition,
            'duration_logits': new_duration,
            'emission_pdf': new_pdf
        }

    def attach_encoder(self, encoder: nn.Module, batch_first: bool = True, pool: str = "mean"):
        """
        Attach a neural encoder (CNN/LSTM/Transformer) to the HSMM module.
        Wraps the encoder and updates self._context for amortized inference.

        Args:
            encoder (nn.Module): PyTorch encoder module.
            batch_first (bool): Whether input tensors are (B,T,F) or (T,B,F).
            pool (str): Temporal pooling mode for _context ('last', 'mean', 'max', 'attn', 'mha').
        """
        device = next(self.parameters()).device
        encoder.to(device=device, dtype=DTYPE)

        class EncoderWrapper(nn.Module):
            def __init__(self, enc: nn.Module, batch_first: bool, parent, pool: str):
                super().__init__()
                self.enc = enc
                self.batch_first = batch_first
                self.parent = parent
                self.pool = pool

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = x.to(device=device, dtype=DTYPE)

                # Ensure batch dimension
                if x.ndim == 2:  # (T,F) -> (1,T,F) or (T,1,F)
                    x = x.unsqueeze(0 if self.batch_first else 1)
                elif x.ndim != 3:
                    raise ValueError(f"Unsupported input shape {x.shape}, expected (T,F) or (B,T,F)")

                if not self.batch_first:
                    x = x.transpose(0, 1)  # (T,B,F) -> (B,T,F)

                out = self.enc(x)
                if isinstance(out, (tuple, list)):
                    out = out[0]  # RNN/LSTM style

                if not torch.is_tensor(out):
                    raise TypeError(f"Encoder must return a tensor, got {type(out)}")

                # Standardize output to (B,T,H)
                if out.ndim == 1:
                    theta = out.unsqueeze(0).unsqueeze(0)
                elif out.ndim == 2:
                    if out.shape[0] == x.shape[0]:
                        theta = out.unsqueeze(1)  # (B,1,H)
                    else:
                        theta = out.unsqueeze(0)  # (1,T,H)
                elif out.ndim == 3:
                    theta = out
                else:
                    raise ValueError(f"Unexpected encoder output shape {out.shape}")

                # --- Temporal pooling ---
                if self.pool == "last":
                    vec = theta[:, -1]
                elif self.pool == "mean":
                    vec = theta.mean(dim=1)
                elif self.pool == "max":
                    vec, _ = theta.max(dim=1)
                elif self.pool == "attn":
                    if not hasattr(self, "_attn_vector") or self._attn_vector.shape[-1] != theta.shape[-1]:
                        self._attn_vector = nn.Parameter(torch.randn(theta.shape[-1], device=device, dtype=DTYPE))
                    attn_scores = theta @ self._attn_vector
                    attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
                    vec = (attn_weights * theta).sum(dim=1)
                elif self.pool == "mha":
                    n_heads = getattr(self, "_mha_heads", 4)
                    head_dim = theta.shape[-1] // n_heads
                    if not hasattr(self, "_mha_qkv"):
                        self._mha_qkv = nn.Linear(theta.shape[-1], 3 * theta.shape[-1], device=device, dtype=DTYPE)
                    if not hasattr(self, "_mha_rel_pos"):
                        self._mha_rel_pos = nn.Parameter(torch.zeros(2*theta.shape[1]-1, n_heads, device=device, dtype=DTYPE))

                    qkv = self._mha_qkv(theta)
                    Q, K, V = qkv.chunk(3, dim=-1)
                    Q = Q.view(Q.shape[0], Q.shape[1], n_heads, head_dim).transpose(1,2)
                    K = K.view(K.shape[0], K.shape[1], n_heads, head_dim).transpose(1,2)
                    V = V.view(V.shape[0], V.shape[1], n_heads, head_dim).transpose(1,2)

                    attn_scores = (Q @ K.transpose(-2,-1)) / (head_dim ** 0.5)
                    T = theta.shape[1]
                    rel_idx = torch.arange(T, device=device)[:, None] - torch.arange(T, device=device)[None, :] + T - 1
                    attn_scores = attn_scores + self._mha_rel_pos[rel_idx].permute(2,0,1).unsqueeze(0)

                    attn_weights = torch.softmax(attn_scores, dim=-1)
                    context = (attn_weights @ V).transpose(1,2).contiguous().view(theta.shape[0], theta.shape[1], -1)
                    vec = context.mean(dim=1)
                else:
                    raise ValueError(f"Unsupported pooling mode: {self.pool}")

                vec = nn.functional.layer_norm(vec, vec.shape[-1:])
                vec = torch.clamp(vec, -10.0, 10.0)
                self.parent._context = vec.detach()

                return out

        self.encoder = EncoderWrapper(encoder, batch_first, self, pool)

    def encode_observations(self, X: torch.Tensor, pool: Optional[str] = None, store: bool = True) -> Optional[torch.Tensor]:
        """
        Encode observations into context vectors θ using the attached encoder.
        Automatically uses the encoder's pooling mode unless overridden.

        Args:
            X (torch.Tensor): Input observations of shape (T,F) or (B,T,F).
            pool (Optional[str]): Pooling mode; if None, uses encoder default.
            store (bool): Whether to store resulting context in self._context.

        Returns:
            Optional[torch.Tensor]: Context vectors θ of shape (B,H), or None.
        """
        if self.encoder is None:
            if store: self._context = None
            return None

        device = next(self.encoder.parameters()).device
        X = X.to(device=device, dtype=DTYPE)

        if X.ndim == 2: X = X.unsqueeze(0)
        elif X.ndim != 3:
            raise ValueError(f"Expected (T,F) or (B,T,F), got {X.shape}")

        # Override pooling mode if provided
        if pool is not None and hasattr(self.encoder, 'pool'):
            self.encoder.pool = pool

        try:
            out = self.encoder(X)
        except Exception as e:
            raise RuntimeError(f"Encoder forward() failed: {e}")

        # The encoder wrapper already sets _context during forward
        vec = getattr(self, "_context", None)

        if vec is None and store:
            # Fallback: simple mean pooling if _context was not set
            if out.ndim == 3:
                vec = out.mean(dim=1)
            else:
                vec = out
            vec = nn.functional.layer_norm(vec, vec.shape[-1:])
            vec = torch.clamp(vec, -10.0, 10.0)
            self._context = vec.detach()

        return vec

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
                f"Expected {expected_ndim}duration_logits input (batch + event dims), got {value.ndim}duration_logits."
            )

        if event_shape and tuple(value.shape[-len(event_shape):]) != tuple(event_shape):
            raise ValueError(
                f"PDF event shape mismatch: expected {tuple(event_shape)}, got {tuple(value.shape[-len(event_shape):])}."
            )

        return value

    def map_emission(self, x: torch.Tensor, theta: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute per-state emission log-probabilities for a sequence or batch, optionally using context.
        Falls back to self._context if theta is None.

        Args:
            x (torch.Tensor): Input sequence tensor of shape (T,F) or (B,T,F).
            theta (Optional[torch.Tensor]): Optional context tensor for conditional emission PDFs.

        Returns:
            torch.Tensor: Log-probabilities of shape (T, n_states) or (B, T, n_states).
        """
        device = next(self.parameters(), torch.tensor(0.0)).device
        x = x.to(device=device, dtype=DTYPE)

        if theta is None and hasattr(self, "_context"):
            theta = self._context

        logp_or_dist = self._contextual_emission_pdf(x, theta)

        # Convert Distribution to tensor if necessary
        if isinstance(logp_or_dist, torch.distributions.Distribution):
            event_ndim = len(logp_or_dist.event_shape)
            x_exp = x.unsqueeze(-event_ndim-1) if event_ndim > 0 else x.unsqueeze(-1)
            log_probs = logp_or_dist.log_prob(x_exp)
            if log_probs.ndim > x.ndim:
                log_probs = log_probs.flatten(start_dim=x.ndim-1).sum(dim=-1)
        elif torch.is_tensor(logp_or_dist):
            log_probs = logp_or_dist
            if log_probs.ndim < x.ndim:
                log_probs = log_probs.unsqueeze(-1)
        else:
            raise TypeError(f"Unsupported type returned from _contextual_emission_pdf: {type(logp_or_dist)}")

        return log_probs.to(dtype=DTYPE, device=device)

    def to_observations(self, X: torch.Tensor, lengths: Optional[List[int]] = None, theta: Optional[torch.Tensor] = None) -> utils.Observations:
        """
        Convert input tensor X into a utils.Observations object suitable for HSMM.
        Automatically uses self._context if theta is None.

        Args:
            X (torch.Tensor): Input tensor (T,F) or (B,T,F)
            lengths (Optional[List[int]]): Optional sequence lengths
            theta (Optional[torch.Tensor]): Optional context; defaults to self._context

        Returns:
            utils.Observations: sequences, per-state log-probabilities, and lengths
        """
        device = next(self.parameters(), torch.tensor(0.0)).device
        X_valid = self.check_constraints(X).to(device=device, dtype=DTYPE)

        # Use stored context if theta not provided
        if theta is None and hasattr(self, "_context") and self._context is not None:
            theta = self._context
            if theta.ndim == 2 and X_valid.ndim == 3:
                theta = theta.unsqueeze(1).expand(-1, X_valid.shape[1], -1)

        total_len = X_valid.shape[0] if X_valid.ndim == 2 else X_valid.shape[1]
        if lengths is not None:
            if sum(lengths) != total_len:
                raise ValueError(f"Sum of lengths ({sum(lengths)}) does not match total samples ({total_len}).")
            seq_lengths = lengths
        else:
            seq_lengths = [total_len]

        # Split sequences
        sequences = []
        if X_valid.ndim == 2:
            sequences = list(torch.split(X_valid, seq_lengths))
        elif X_valid.ndim == 3:
            if len(seq_lengths) == X_valid.shape[0]:
                sequences = [X_valid[i] for i in range(X_valid.shape[0])]
            else:
                sequences = list(torch.split(X_valid.reshape(-1, X_valid.shape[-1]), seq_lengths))
        else:
            raise ValueError(f"Unsupported input shape {X_valid.shape}")

        log_probs_list = []
        for idx, seq in enumerate(sequences):
            seq_theta = None
            if theta is not None:
                if theta.ndim == 2 and theta.shape[0] == len(sequences):
                    seq_theta = theta[idx].unsqueeze(0)
                else:
                    seq_theta = theta

            log_probs = self.map_emission(seq, seq_theta)
            if not torch.is_tensor(log_probs):
                log_probs = torch.as_tensor(log_probs, dtype=DTYPE, device=device)
            log_probs_list.append(log_probs)

        return utils.Observations(sequence=sequences, log_probs=log_probs_list, lengths=seq_lengths)

    def sample_model_params(self, X: Optional[torch.Tensor] = None, theta: Optional[torch.Tensor] = None, inplace: bool = True) -> Dict[str, Any]:
        """
        Sample model params (π, transition_logits, duration_logits, emission_pdf) in log-space, optionally context-conditioned.
        """
        device = next(self.parameters(), torch.tensor(0.0)).device
        α = getattr(self, "alpha", 1.0)

        # --- π ---
        init_logits = torch.log(constraints.sample_probs(α, (self.n_states,)).to(DTYPE).to(device))

        # --- transition_logits ---
        transition_logits = torch.log(constraints.sample_transition(α, self.n_states, constraints.Transitions.SEMI).to(DTYPE).to(device))
        if theta is not None and hasattr(self, "_contextual_transition_matrix"):
            transition_logits = torch.log(self._contextual_transition_matrix(theta) + 1e-12)

        # --- duration_logits ---
        duration_logits = torch.log(constraints.sample_probs(α, (self.n_states, self.max_duration)).to(DTYPE).to(device))
        if theta is not None and hasattr(self, "_contextual_duration_pdf"):
            duration_logits = torch.log(self._contextual_duration_pdf(theta) + 1e-12)

        # --- Emission ---
        emission_pdf = self.sample_emission_pdf(X)
        if theta is not None and hasattr(self, "_contextual_emission_pdf"):
            emission_pdf = self._contextual_emission_pdf(X, theta)

        params = {"init_logits": init_logits, "transition_logits": transition_logits, "duration_logits": duration_logits, "emission_pdf": emission_pdf}

        if inplace:
            if hasattr(self, "init_logits"): self.init_logits.data.copy_(init_logits)
            if hasattr(self, "transition_logits"): self.transition_logits.data.copy_(transition_logits)
            if hasattr(self, "duration_logits"): self.duration_logits.data.copy_(duration_logits)
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
        Fit the HSMM using EM with context-aware, vectorized emissions.
        
        Args:
            X: Observations tensor (B, T, F) or list of sequences.
            tol: Convergence tolerance.
            max_iter: Maximum EM iterations per initialization.
            n_init: Number of random initializations.
            post_conv_iter: Extra iterations after convergence.
            ignore_conv: If True, continue even after convergence.
            sample_B_from_X: Sample initial emissions from X.
            verbose: Print progress messages.
            plot_conv: Plot convergence if True.
            lengths: Optional list of sequence lengths.
            theta: Optional context tensor.
        """
        device = next(self.parameters(), torch.tensor(0.0)).device

        # Optional initialization from data
        if sample_B_from_X:
            self._params['emission_pdf'] = self.sample_emission_pdf(X)

        # Encode observations if needed
        if theta is None and getattr(self, "encoder", None):
            theta = self.encode_observations(X)

        X_valid = self.to_observations(X, lengths, theta)

        # Align theta to sequences
        if theta is not None:
            if theta.ndim == 2 and theta.shape[0] == len(X_valid.sequence):
                valid_theta = theta
            else:
                valid_theta = theta.repeat(len(X_valid.sequence), 1)
        else:
            valid_theta = None

        # Initialize convergence tracker
        self.conv = ConvergenceHandler(
            tol=tol,
            max_iter=max_iter,
            n_init=n_init,
            post_conv_iter=post_conv_iter,
            verbose=verbose
        )

        best_score = -float("inf")
        best_state = None

        # Flatten all sequences for vectorized log-prob computation
        seq_tensor = torch.cat(X_valid.sequence, dim=0).to(dtype=DTYPE, device=device)
        seq_offsets = [0] + list(torch.cumsum(torch.tensor(X_valid.lengths, device=device), dim=0).tolist())

        for run_idx in range(n_init):
            if verbose:
                print(f"\n=== Run {run_idx + 1}/{n_init} ===")

            # Re-sample parameters for subsequent runs
            if run_idx > 0:
                sampled = self.sample_model_params(X)
                self._transition_logits.copy_(sampled['transition_logits'])
                self._duration_logits.copy_(sampled['duration_logits'])
                self._init_logits.copy_(sampled['init_logits'])
                self._params['emission_pdf'] = sampled['emission_pdf']

            base_ll = self._compute_log_likelihood(X_valid).sum()
            self.conv.push_pull(base_ll, 0, run_idx)

            for it in range(1, max_iter + 1):
                # Update context-aware PDFs
                self._params['emission_pdf'] = self._contextual_emission_pdf(X_valid, valid_theta)
                transition_logits = self._contextual_transition_matrix(valid_theta)
                duration_logits = self._contextual_duration_pdf(valid_theta)

                # Copy logits safely
                if transition_logits is not None:
                    self._transition_logits.copy_(transition_logits)
                if duration_logits is not None:
                    self._duration_logits.copy_(duration_logits)

                # EM parameter update
                new_params = self._estimate_model_params(X_valid, valid_theta)
                for key in ['emission_pdf', 'init', 'transition', 'duration']:
                    if key in new_params:
                        if key == 'emission_pdf':
                            self._params[key] = new_params[key]
                        else:
                            getattr(self, f"_{key}_logits").copy_(new_params[key])

                # Compute vectorized log-probs for sequences
                pdf = self._params['emission_pdf']
                if pdf is None or not hasattr(pdf, 'log_prob'):
                    raise RuntimeError("Emission PDF not initialized or unsupported for vectorized log_prob.")
                all_log_probs = pdf.log_prob(seq_tensor.unsqueeze(1))  # (total_T, K)
                X_valid.log_probs = [all_log_probs[seq_offsets[i]:seq_offsets[i + 1]] for i in range(len(X_valid.lengths))]

                # Evaluate log-likelihood and check convergence
                curr_ll = self._compute_log_likelihood(X_valid).sum()
                converged = self.conv.push_pull(curr_ll, it, run_idx)
                if converged and not ignore_conv and verbose:
                    print(f"[Run {run_idx + 1}] Converged at iteration {it}.")
                    break

            # Track best run
            run_score = float(self._compute_log_likelihood(X_valid).sum().item())
            if run_score > best_score:
                best_score = run_score
                best_state = {
                    'init_logits': self._init_logits.clone(),
                    'transition_logits': self._transition_logits.clone(),
                    'duration_logits': self._duration_logits.clone(),
                    'emission_pdf': self._params['emission_pdf']
                }

        # Restore best parameters
        if best_state is not None:
            self._init_logits.copy_(best_state['init_logits'])
            self._transition_logits.copy_(best_state['transition_logits'])
            self._duration_logits.copy_(best_state['duration_logits'])
            self._params['emission_pdf'] = best_state['emission_pdf']

        # Optionally plot convergence
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
        """
        Decode sequences for both neural and classical HSMM using MAP or Viterbi algorithm.

        Args:
            X: Input tensor (B, T, F) or list of sequences.
            lengths: Optional list of sequence lengths.
            algorithm: Decoding mode, "map" or "viterbi".
            context: Optional context tensor (B, H), (1, H), or (H,).
            batch_size: Batch size for emission log-prob computation.

        Returns:
            List[Tensor]: Predicted discrete state indices for each sequence.
        """
        algorithm = algorithm.lower()
        device = next(self.parameters(), torch.tensor(0.0)).device
        X_valid = self.to_observations(X, lengths)

        # Normalize context
        if context is not None:
            if context.dim() == 1:
                context = context.unsqueeze(0)
            elif context.dim() > 2:
                context = context.mean(dim=0, keepdim=True)
            context = context.to(device=device, dtype=DTYPE)

        # Prepare observation tensor
        seq_tensor = torch.cat(X_valid.sequence, dim=0).to(device=device, dtype=DTYPE)
        seq_offsets = [0] + list(torch.cumsum(torch.tensor(X_valid.lengths, device=device), dim=0).tolist())
        B = len(X_valid.lengths)
        total_T = seq_tensor.shape[0]
        log_probs_chunks = []

        # --- EMISSION LOG-PROBABILITIES ---
        pdf = getattr(self, "_params", {}).get("emission_pdf", None)
        if pdf is not None and hasattr(pdf, "log_prob"):
            # Classical emission
            for start in range(0, total_T, batch_size):
                end = min(start + batch_size, total_T)
                chunk = seq_tensor[start:end].unsqueeze(1)
                lp = pdf.log_prob(chunk)
                if lp.ndim > 2:
                    lp = lp.sum(dim=list(range(2, lp.ndim)))
                log_probs_chunks.append(lp)
            all_log_probs = torch.cat(log_probs_chunks, dim=0)
        else:
            # Neural emission
            if not hasattr(self, "emission_module"):
                raise RuntimeError("No emission model found for NeuralHSMM.")
            mu_var_or_logits = self.emission_module(context)
            if isinstance(mu_var_or_logits, tuple):
                mu, var = mu_var_or_logits
                var = var.clamp(min=1e-6)
                dist = torch.distributions.Normal(mu, var.sqrt())
                for start in range(0, total_T, batch_size):
                    end = min(start + batch_size, total_T)
                    chunk = seq_tensor[start:end].unsqueeze(1)
                    lp = dist.log_prob(chunk).sum(dim=-1)
                    log_probs_chunks.append(lp)
                all_log_probs = torch.cat(log_probs_chunks, dim=0)
            else:
                logits = mu_var_or_logits
                all_log_probs = F.log_softmax(logits, dim=-1)[seq_tensor.long(), :]

        # Split per sequence
        X_valid.log_probs = [all_log_probs[seq_offsets[i]:seq_offsets[i + 1]] for i in range(B)]
        K = getattr(self, "n_states", all_log_probs.shape[-1])

        # --- TRANSITION LOGITS ---
        if hasattr(self, "transition_module"):
            if hasattr(self.transition_module, "log_probs"):
                transition_logits = self.transition_module.log_probs(context=context)
            else:
                transition_logits = torch.log_softmax(self.transition_module(context), dim=-1)
        else:
            transition_logits = getattr(self._params, "log_transition", None)
            if transition_logits is None:
                raise RuntimeError("Transition parameters missing for HSMM.")
            transition_logits = transition_logits.to(device=device)

        # --- DURATION LOG-PROBS ---
        if hasattr(self, "duration_module"):
            if hasattr(self.duration_module, "log_probs"):
                log_duration = self.duration_module.log_probs(context=context)
            elif hasattr(self.duration_module, "logits"):
                log_duration = F.log_softmax(self.duration_module.logits, dim=-1)
            else:
                log_duration = None
        else:
            log_duration = getattr(self._params, "log_duration", None)
            if log_duration is not None:
                log_duration = log_duration.to(device=device)

        # --- PAD OBSERVATIONS ---
        max_T = max(X_valid.lengths)
        log_B = torch.full((B, max_T, K), -torch.inf, device=device)
        for i, seq_lp in enumerate(X_valid.log_probs):
            log_B[i, :seq_lp.shape[0]] = seq_lp
        mask = torch.arange(max_T, device=device).unsqueeze(0) < torch.tensor(X_valid.lengths, device=device).unsqueeze(1)

        # --- DECODING ---
        if algorithm == "map":
            alpha = torch.full((B, max_T, K), -torch.inf, device=device)
            alpha[:, 0] = log_B[:, 0]
            for t in range(1, max_T):
                prev = alpha[:, t - 1].unsqueeze(2) + transition_logits
                if log_duration is not None:
                    dur_idx = min(t, log_duration.shape[1] - 1)
                    prev = prev + log_duration[:, dur_idx].unsqueeze(0)
                alpha[:, t] = torch.logsumexp(prev, dim=1) + log_B[:, t]
                alpha[:, t] = torch.where(mask[:, t].unsqueeze(-1), alpha[:, t], alpha[:, t - 1])
            decoded = torch.argmax(alpha, dim=-1)

        elif algorithm == "viterbi":
            delta = torch.full((B, max_T, K), -torch.inf, device=device)
            psi = torch.zeros((B, max_T, K), dtype=torch.long, device=device)
            delta[:, 0] = log_B[:, 0]
            for t in range(1, max_T):
                scores = delta[:, t - 1].unsqueeze(2) + transition_logits
                if log_duration is not None:
                    dur_idx = min(t, log_duration.shape[1] - 1)
                    scores = scores + log_duration[:, dur_idx].unsqueeze(0)
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

    def score(
        self,
        X: torch.Tensor,
        lengths: Optional[List[int]] = None,
        by_sample: bool = True,
        theta: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Vectorized, duration-aware log-likelihood computation for HSMM with optional context.

        Args:
            X: Input sequences (B, T, F) or flattened.
            lengths: Optional list of sequence lengths.
            by_sample: If True, return per-sequence log-likelihood; else sum.
            theta: Optional context tensor.

        Returns:
            Tensor of log-likelihoods.
        """
        obs = self.to_observations(X, lengths, theta)
        device = next(self.parameters(), torch.tensor(0.0)).device
        B = len(obs.lengths)
        max_T = max(obs.lengths) if B > 0 else 0
        K, Dmax = self.n_states, self.max_duration

        # -------- Emission log-probabilities --------
        pdf = self._params.get("emission_pdf")
        if pdf is None or not hasattr(pdf, "log_prob"):
            raise RuntimeError("Emission PDF must be initialized with log_prob method.")
        all_seq = torch.cat(obs.sequence, dim=0).to(dtype=DTYPE, device=device)
        all_log_probs = pdf.log_prob(all_seq.unsqueeze(1))  # (total_T, K)

        # Map to batch and sequence length
        seq_offsets = torch.cat([torch.tensor([0], device=device), torch.cumsum(torch.tensor(obs.lengths, device=device), dim=0)])
        log_B = torch.full((B, max_T, K), -torch.inf, device=device, dtype=DTYPE)
        for i in range(B):
            start, end = seq_offsets[i].item(), seq_offsets[i + 1].item()
            L = obs.lengths[i]
            log_B[i, :L] = all_log_probs[start:end]

        # -------- Transition and duration --------
        if hasattr(self, "transition_module"):
            log_transition = getattr(self.transition_module, "log_probs", lambda **kwargs: torch.log_softmax(self.transition_module.logits, dim=-1))()
        else:
            log_transition = torch.full((K, K), 1.0 / K, device=device).log()

        log_duration = F.log_softmax(self._duration_logits, dim=-1) if hasattr(self, "_duration_logits") else torch.zeros(K, 1, device=device, dtype=DTYPE)

        # -------- Vectorized forward (log-domain) --------
        V = torch.full((B, max_T, K), -torch.inf, device=device, dtype=DTYPE)

        # Initialize first timestep
        V[:, 0] = log_B[:, 0] + self.init_logits.to(device=device)

        # Precompute durations (Dmax) mask
        dur_idx = torch.arange(Dmax, device=device)
        for t in range(1, max_T):
            max_d = min(t + 1, Dmax)
            durations = dur_idx[:max_d]

            # Slice previous V for all durations
            prev_V = torch.stack([V[:, t - d] for d in durations], dim=1)  # (B, max_d, K)
            dur_scores = log_duration[:, :max_d].T.unsqueeze(0)  # (1, max_d, K)
            trans_scores = prev_V.unsqueeze(3) + log_transition  # (B, max_d, K, K)
            trans_max = torch.logsumexp(trans_scores, dim=2)  # (B, max_d, K)

            # Emission sum for duration segment
            emit_sums = torch.stack([log_B[:, t - d + 1:t + 1].sum(dim=1) for d in durations], dim=1)  # (B, max_d, K)

            V[:, t] = torch.logsumexp(trans_max + dur_scores + emit_sums, dim=1)

        # Mask invalid positions
        mask = torch.arange(max_T, device=device).unsqueeze(0) < torch.tensor(obs.lengths, device=device).unsqueeze(1)
        V = torch.where(mask.unsqueeze(-1), V, torch.tensor(-torch.inf, device=device))

        # Sequence log-likelihoods
        end_idx = torch.tensor(obs.lengths, device=device) - 1
        seq_ll = torch.logsumexp(V[torch.arange(B), end_idx], dim=-1)

        return seq_ll.detach().cpu() if by_sample else seq_ll.sum().detach().cpu()

    def information(
        self,
        X: torch.Tensor,
        criterion: constraints.InformCriteria = constraints.InformCriteria.AIC,
        lengths: Optional[List[int]] = None,
        by_sample: bool = True
    ) -> torch.Tensor:
        """
        Compute an information criterion (AIC, BIC, etc.) for the HSMM.

        Args:
            X: Input sequences, shape (B, T, F) or (T, F).
            criterion: Which information criterion to compute.
            lengths: Optional sequence lengths for variable-length batches.
            by_sample: Whether to return per-sequence values or a scalar sum.

        Returns:
            Tensor of information criterion values (per-sample or aggregated).
        """
        device = next(self.parameters()).device

        # --- Compute log-likelihood per sequence ---
        try:
            log_likelihood = self.score(X, lengths=lengths, by_sample=by_sample).to(device)
        except Exception as e:
            raise RuntimeError(f"Failed to compute log-likelihood: {e}")

        # --- Determine total number of observations ---
        if lengths is not None:
            n_obs = max(int(sum(lengths)), 1)
        elif X.ndim == 3:  # (B, T, F)
            n_obs = max(int(X.shape[0] * X.shape[1]), 1)
        else:  # (T, F) or flat
            n_obs = max(int(X.shape[0]), 1)

        # --- Retrieve model degrees of freedom ---
        dof = getattr(self, "dof", None)
        if dof is None:
            raise AttributeError(
                "Model degrees of freedom ('dof') not defined. "
                "Please set 'self.dof' during initialization."
            )

        # --- Compute the information criterion ---
        ic_value = constraints.compute_information_criteria(
            n_obs=n_obs,
            log_likelihood=log_likelihood,
            dof=dof,
            criterion=criterion
        )

        # --- Ensure tensor format and numeric safety ---
        if not isinstance(ic_value, torch.Tensor):
            ic_value = torch.tensor(ic_value, dtype=DTYPE, device=device)
        ic_value = ic_value.nan_to_num(nan=float('inf'), posinf=float('inf'), neginf=float('inf'))

        # --- Normalize per-sample if requested ---
        if by_sample and ic_value.ndim == 0:
            ic_value = ic_value.unsqueeze(0)

        return ic_value.detach().cpu()

    def _map(self, X: utils.Observations) -> List[torch.Tensor]:
        """MAP decoding of HSMM sequences from posterior state marginals."""
        gamma_list, _, _ = self._compute_posteriors(X)
        if gamma_list is None:
            raise RuntimeError("Posterior probabilities could not be computed — model parameters uninitialized.")

        device = next(self.parameters()).device
        map_sequences: List[torch.Tensor] = []

        for seq_gamma in gamma_list:
            if seq_gamma is None or seq_gamma.numel() == 0:
                map_sequences.append(torch.empty(0, dtype=torch.long, device=device))
                continue

            # Clean NaNs/Infs safely
            seq_gamma = torch.nan_to_num(seq_gamma, nan=0.0, posinf=0.0, neginf=0.0)

            # Row-normalize, avoid division by zero
            row_sums = seq_gamma.sum(dim=1, keepdim=True).clamp_min(1.0)
            seq_gamma_norm = seq_gamma / row_sums

            # MAP decoding: take argmax along states
            map_seq = torch.argmax(seq_gamma_norm, dim=1).to(dtype=torch.long, device=device)
            map_sequences.append(map_seq)

        return map_sequences

    def _forward(self, X: utils.Observations) -> List[torch.Tensor]:
        """Compute forward messages (log-alpha) for each sequence."""
        device = next(self.parameters(), torch.tensor(0.0)).device
        init_logits, transition_logits, duration_logits = self.init_logits, self.transition_logits, self.duration_logits
        neg_inf = -torch.inf
        alpha_list = []

        for seq_probs, seq_len in zip(X.log_probs, X.lengths):
            if seq_len == 0:
                alpha_list.append(torch.full((0, self.n_states, self.max_duration), neg_inf, device=device, dtype=DTYPE))
                continue

            log_alpha = torch.full((seq_len, self.n_states, self.max_duration), neg_inf, dtype=DTYPE, device=device)
            cumsum_emit = torch.vstack((torch.zeros((1, self.n_states), dtype=DTYPE, device=device),
                                        torch.cumsum(seq_probs, dim=0)))

            max_d = min(self.max_duration, seq_len)
            durations = torch.arange(1, max_d + 1, device=device)
            emit_sums = (cumsum_emit[durations] - cumsum_emit[0]).T

            log_alpha[0, :, :max_d] = init_logits.unsqueeze(-1) + duration_logits[:, :max_d] + emit_sums

            for t in range(1, seq_len):
                prev_alpha = log_alpha[t - 1]
                shifted = torch.cat([prev_alpha[:, 1:], torch.full((self.n_states, 1), neg_inf, dtype=DTYPE, device=device)], dim=1)
                trans = torch.logsumexp(prev_alpha[:, 0].unsqueeze(1) + transition_logits, dim=0)
                log_alpha[t] = torch.logsumexp(torch.stack([shifted + seq_probs[t].unsqueeze(-1),
                                                            duration_logits + trans.unsqueeze(-1)]), dim=0)

            alpha_list.append(log_alpha)

        return alpha_list

    def _backward(self, X: utils.Observations) -> List[torch.Tensor]:
        """Compute backward messages (log-beta) for each sequence."""
        beta_list = []
        transition_logits, duration_logits = self.transition_logits, self.duration_logits
        neg_inf = -torch.inf

        for seq_probs, seq_len in zip(X.log_probs, X.lengths):
            device = seq_probs.device
            if seq_len == 0:
                beta_list.append(torch.empty((0, self.n_states, self.max_duration), device=device))
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
                dur_lp = duration_logits[:, :max_d]
                beta_next = log_beta[t + durations - 1, :, 0].T

                log_beta[t, :, 0] = torch.logsumexp(emit_sums + dur_lp + beta_next, dim=1)

                if self.max_duration > 1:
                    log_beta[t, :, 1:] = log_beta[t + 1, :, :-1] + seq_probs[t + 1].unsqueeze(-1)

            beta_list.append(log_beta)

        return beta_list

    def _compute_posteriors(self, X: utils.Observations) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """Compute gamma, xi, eta posterior marginals for HSMM sequences."""
        alpha_list = self._forward(X)
        beta_list = self._backward(X)

        gamma_vec, xi_vec, eta_vec = [], [], []
        device = next(self.parameters(), torch.tensor(0.0)).device
        K, Dmax = self.n_states, self.max_duration

        for seq_probs, seq_len, alpha, beta in zip(X.log_probs, X.lengths, alpha_list, beta_list):
            if seq_len == 0 or alpha.numel() == 0 or beta.numel() == 0:
                gamma_vec.append(torch.zeros((seq_len, K), dtype=DTYPE, device=device))
                eta_vec.append(torch.zeros((seq_len, K, Dmax), dtype=DTYPE, device=device))
                xi_vec.append(torch.zeros((max(seq_len - 1, 0), K, K), dtype=DTYPE, device=device))
                continue

            alpha, beta = alpha.to(device=device, dtype=DTYPE), beta.to(device=device, dtype=DTYPE)

            # ----- gamma: state marginals -----
            log_gamma = torch.logsumexp(alpha + beta, dim=2)
            gamma = F.softmax(log_gamma, dim=1).clamp_min(0.0)
            gamma_vec.append(gamma)

            # ----- eta: state-duration marginals -----
            log_eta = alpha + beta
            eta = F.softmax(log_eta.flatten(start_dim=1), dim=1).view(seq_len, K, Dmax).clamp_min(0.0)
            eta_vec.append(eta)

            # ----- xi: transition marginals -----
            if seq_len > 1:
                alpha_start = alpha[:-1, :, 0]
                trans_alpha = alpha_start.unsqueeze(2) + self.transition_logits.unsqueeze(0).to(device=device, dtype=DTYPE)
                beta_next = beta[1:] + self.duration_logits.unsqueeze(0).to(device=device, dtype=DTYPE)
                dur_beta_sum = torch.logsumexp(beta_next, dim=2)
                log_xi = trans_alpha + dur_beta_sum.unsqueeze(1)
                xi = F.softmax(log_xi.flatten(start_dim=1), dim=1).view(seq_len - 1, K, K).clamp_min(0.0)
                xi_vec.append(xi)
            else:
                xi_vec.append(torch.zeros((0, K, K), dtype=DTYPE, device=device))

        return gamma_vec, xi_vec, eta_vec

    def _viterbi(self, X: utils.Observations, duration_weight: float = 0.0) -> list[torch.Tensor]:
        """
        Vectorized, duration-explicit Viterbi decoding for HSMM in log-domain.

        Args:
            X: Observation container with X.log_probs and X.lengths.
            duration_weight: [0.0–1.0] strength of duration regularization.

        Returns:
            List of most probable state sequences per batch.
        """
        device = next(self.parameters(), torch.tensor(0.0)).device
        K, Dmax = self.n_states, self.max_duration
        init_logits, transition_logits, duration_logits = self.init_logits.to(device), self.transition_logits.to(device), self.duration_logits.to(device)
        neg_inf = -torch.inf

        B = len(X.log_probs)
        lengths = X.lengths
        max_len = max(lengths) if lengths else 0

        V = torch.full((B, max_len, K), neg_inf, dtype=DTYPE, device=device)
        best_prev = torch.full((B, max_len, K), -1, dtype=torch.int64, device=device)
        best_dur = torch.zeros((B, max_len, K), dtype=torch.int64, device=device)

        dur_indices = torch.arange(1, Dmax + 1, device=device)

        # Precompute duration log-probabilities with optional weight
        if duration_weight > 0.0:
            dur_mean = torch.softmax(duration_logits, dim=1) @ dur_indices.float()
            dur_penalty = -((dur_indices.unsqueeze(0) - dur_mean.unsqueeze(1)) ** 2) / (2 * Dmax)
            dur_lp = (1 - duration_weight) * duration_logits + duration_weight * dur_penalty
        else:
            dur_lp = duration_logits  # (K, Dmax)

        # Compute cumulative emissions for each batch
        cumsum_emit = []
        for b, seq_probs in enumerate(X.log_probs):
            seq_probs = seq_probs.to(device=device, dtype=DTYPE)
            cumsum_emit.append(torch.cat([torch.zeros((1, K), dtype=DTYPE, device=device),
                                          torch.cumsum(seq_probs, dim=0)]))
        
        # Vectorized DP over duration
        for t in range(max_len):
            mask = torch.tensor([t < L for L in lengths], device=device)  # which sequences are active
            if not mask.any():
                continue
            active_idx = mask.nonzero(as_tuple=True)[0]

            for b in active_idx:
                T = lengths[b]
                max_d = min(Dmax, t + 1)
                durations = dur_indices[:max_d]
                starts = t - durations + 1
                emit_sums = (cumsum_emit[b][t + 1] - cumsum_emit[b][starts]).T  # (K, max_d)
                dur_scores = dur_lp[:, :max_d]

                if t == 0:
                    scores = init_logits.unsqueeze(1) + dur_scores + emit_sums
                    prev_idx = torch.full_like(scores, -1, dtype=torch.int64)
                else:
                    prev_V = V[b, starts]  # (max_d, K)
                    scores_plus_trans = prev_V.unsqueeze(2) + transition_logits.unsqueeze(0)  # (max_d, K, K)
                    scores_max, argmax_prev = torch.max(scores_plus_trans, dim=1)  # (max_d, K)
                    scores = scores_max.T + dur_scores + emit_sums  # (K, max_d)
                    prev_idx = argmax_prev.T

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
            # Pad if needed
            if seq_path.shape[0] < T:
                seq_path = torch.cat([seq_path, torch.full((T - seq_path.shape[0],), seq_path[-1], dtype=torch.int64, device=device)])
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
