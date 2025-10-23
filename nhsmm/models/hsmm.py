# nhsmm/models/hsmm.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Any, Literal, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, Categorical

from nhsmm.defaults import (
    ContextEncoder, Emission, Duration, Transition, DTYPE, EPS, HSMMError
)
from nhsmm import utils, constraints, SeedGenerator, ConvergenceHandler


class HSMM(nn.Module, ABC):
    """
    Hidden Semi-Markov Model (HSMM) base class.
    """

    def __init__(
        self,
        n_states: int,
        n_features: int,
        max_duration: int,
        alpha: Optional[float] = 1.0,
        min_covar: Optional[float] = None,
        seed: Optional[int] = None,
        transition_type: Any = None,
        context_dim: Optional[int] = None,
        device: Optional[torch.device] = None,
        emission_type: str = "gaussian",
        modulate_var: bool = False,
        init_emission: bool = True,  # new: defer emission PDF sampling
    ):
        super().__init__()

        # --- Device & types ---
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_states = n_states
        self.n_features = n_features
        self.max_duration = max_duration
        self.min_covar = min_covar or 1e-3
        self.alpha = alpha if alpha is not None else 1.0
        self.transition_type = transition_type or constraints.Transitions.SEMI
        self.emission_type = emission_type.lower()
        self.modulate_var = modulate_var
        self.context_dim = context_dim

        # --- Context / Encoder ---
        self._context: Optional[torch.Tensor] = None
        self.encoder: Optional[nn.Module] = None

        # --- Seed ---
        self._seed_gen = SeedGenerator(seed)

        # --- Storage ---
        self._params: Dict[str, Any] = {}

        # --- Modules & Buffers ---
        self._init_modules()
        self._init_buffers()

        # --- Optional emission PDF ---
        if init_emission:
            # ensure reproducible sampling using seed generator
            self._params['emission_pdf'] = self.sample_emission_pdf(None)
        else:
            self._params['emission_pdf'] = None

    def _init_buffers(self):
        device = next(self.buffers(), torch.zeros(1, dtype=DTYPE)).device
        EPS_val = getattr(self, "min_covar", EPS)

        def sample_probs(shape):
            probs = constraints.sample_probs(self.alpha, shape, seed=self._seed_gen())  # seeded
            probs = probs.clamp_min(EPS_val)
            logits = torch.log(probs).to(device=device, dtype=DTYPE)
            return logits, probs.shape

        def sample_transitions(n_states):
            probs = constraints.sample_transition(
                self.alpha, n_states, self.transition_type, seed=self._seed_gen()
            )
            probs = probs.clamp_min(EPS_val)
            logits = torch.log(probs).to(device=device, dtype=DTYPE)
            return logits, probs.shape

        # Base HSMM buffers
        init_logits, init_shape = sample_probs((self.n_states,))
        transition_logits, transition_shape = sample_transitions(self.n_states)
        duration_logits, duration_shape = sample_probs((self.n_states, self.max_duration))

        self.register_buffer("_init_logits", init_logits)
        self.register_buffer("_transition_logits", transition_logits)
        self.register_buffer("_duration_logits", duration_logits)

        self._init_shape = init_shape
        self._transition_shape = transition_shape
        self._duration_shape = duration_shape

        # Optional super-states
        n_super_states = getattr(self, "n_super_states", 0)
        if n_super_states > 1:
            super_init_logits, super_init_shape = sample_probs((n_super_states,))
            super_transition_logits, super_transition_shape = sample_transitions(n_super_states)
            self.register_buffer("_super_init_logits", super_init_logits)
            self.register_buffer("_super_transition_logits", super_transition_logits)
            self._super_init_shape = super_init_shape
            self._super_transition_shape = super_transition_shape

        # Snapshot for monitoring
        summary = [init_logits.mean(), transition_logits.mean(), duration_logits.mean()]
        if n_super_states > 1:
            summary += [super_init_logits.mean(), super_transition_logits.mean()]
        self.register_buffer("_init_prior_snapshot", torch.stack(summary).detach())

    def _align_logits(self, logits: torch.Tensor, B: int) -> torch.Tensor:
        """
        Expand or repeat logits to match batch size B safely.
        Supports 1D (init), 2D (transition/duration), 3D (batched) tensors.
        """
        if logits is None:
            raise ValueError("Logits cannot be None")

        logits = logits.to(self.device, dtype=DTYPE)
        B = int(B)

        if logits.ndim == 3:  # already batched
            if logits.shape[0] == B:
                return logits
            elif logits.shape[0] == 1:
                return logits.expand(B, -1, -1)
            else:
                return logits.repeat(B, 1, 1)  # repeat along batch dimension

        elif logits.ndim == 2:  # [K,K] or [K,Dmax]
            return logits.unsqueeze(0).repeat(B, 1, 1)

        elif logits.ndim == 1:  # [K] (initial state logits)
            return logits.unsqueeze(0).repeat(B, 1)

        else:
            raise ValueError(f"Unsupported logits shape {logits.shape}")

    def _reset_buffers(self):
        # Remove all model buffers initialized by _init_buffers
        buffer_names = [
            "_init_logits", "_transition_logits", "_duration_logits",
            "_super_init_logits", "_super_transition_logits",
            "_init_prior_snapshot"
        ]
        for name in buffer_names:
            if hasattr(self, name):
                self._buffers.pop(name, None)
                delattr(self, name)

        # Remove shape attributes
        shape_names = [
            "_init_shape", "_transition_shape", "_duration_shape",
            "_super_init_shape", "_super_transition_shape"
        ]
        for name in shape_names:
            if hasattr(self, name):
                delattr(self, name)

    def _init_modules(self, seed: Optional[int] = None):
        device, dtype = self.device, DTYPE

        rng = torch.Generator(device=device)
        if seed is not None:
            rng.manual_seed(seed)

        self.emission_module = Emission(
            n_states=self.n_states,
            n_features=self.n_features,
            min_covar=self.min_covar,
            context_dim=self.context_dim,
            emission_type=self.emission_type,
            modulate_var=self.modulate_var,
            device=device,
        )

        self.duration_module = Duration(
            n_states=self.n_states,
            max_duration=self.max_duration,
            context_dim=self.context_dim,
            device=device,
        )

        self.transition_module = Transition(
            n_states=self.n_states,
            context_dim=self.context_dim,
            device=device,
        )

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
        
        norm_val = logits.logsumexp(dim=0)
        zero = torch.tensor(0., dtype=DTYPE, device=logits.device)
        if not torch.allclose(norm_val, zero, atol=1e-6):
            raise ValueError(f"init_logits logits must normalize (logsumexp==0); got {norm_val.item():.3e}")

        self._init_logits.copy_(logits.to(dtype=DTYPE))

    @property
    def init_probs(self) -> torch.Tensor:
        return self._init_logits.softmax(0)

    @property
    def duration_logits(self) -> torch.Tensor:
        return self._duration_logits

    @duration_logits.setter
    def duration_logits(self, logits: torch.Tensor):
        logits = logits.to(device=self._duration_logits.device, dtype=DTYPE)
        if logits.shape != (self.n_states, self.max_duration):
            raise ValueError(f"duration_logits logits must have shape ({self.n_states},{self.max_duration})")
        
        row_norm = logits.logsumexp(dim=1)
        zeros = torch.zeros_like(row_norm)
        if not torch.allclose(row_norm, zeros, atol=1e-6):
            raise ValueError(f"Rows of duration_logits logits must normalize (logsumexp==0); got {row_norm.tolist()}")

        self._duration_logits.copy_(logits.to(dtype=DTYPE))

    @property
    def duration_probs(self) -> torch.Tensor:
        return self._duration_logits.softmax(-1)

    @property
    def transition_logits(self) -> torch.Tensor:
        return self._transition_logits

    @transition_logits.setter
    def transition_logits(self, logits: torch.Tensor, atol: float=1e-4):
        logits = logits.to(device=self._transition_logits.device, dtype=DTYPE)
        
        if logits.shape != (self.n_states, self.n_states):
            raise ValueError(f"transition_logits must have shape ({self.n_states},{self.n_states})")

        row_norm = logits.logsumexp(dim=1)
        if not torch.allclose(row_norm, torch.zeros_like(row_norm), atol=atol):
            logits = logits - row_norm.unsqueeze(1)
        
        if not constraints.is_valid_transition(logits.exp(), self.transition_type):
            print("[transition_logits] Warning: logits do not fully satisfy transition constraints, applying anyway.")
            probs = logits.exp().clamp_min(EPS)
            logits = torch.log(probs)
        
        self._transition_logits.copy_(logits.to(dtype=DTYPE))

    @property
    def transition_probs(self) -> torch.Tensor:
        return self._transition_logits.softmax(-1)

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
    def initialize_emissions(self, X: torch.Tensor, method: str = "moment") -> None:
        raise NotImplementedError("Subclasses must implement initialize_emissions().")

    @abstractmethod
    def _estimate_emission_pdf(self, X: torch.Tensor, posterior: torch.Tensor, theta: Optional[utils.ContextualVariables] = None) -> Distribution:
        raise NotImplementedError(
            "Subclasses must implement _estimate_emission_pdf. "
            "It should return a Distribution supporting `.log_prob` and handle optional context theta."
        )

    def is_batch(self, X):
        """Return True if input data is batched (3D)."""
        return X is not None and X.ndim == 3

    def attach_encoder(
        self,
        encoder: nn.Module,
        batch_first: bool = True,
        pool: Literal["mean", "last", "max", "attn", "mha"] = "mean",
        n_heads: int = 4,
        max_seq_len: int = 1024
    ) -> ContextEncoder:
        """Attach a ContextEncoder wrapper to the model."""
        self.encoder = ContextEncoder(
            pool=pool,
            n_heads=n_heads,
            encoder=encoder,
            device=self.device,
            batch_first=batch_first,
            max_seq_len=max_seq_len
        )
        return self.encoder

    def encode_observations(self,
        X: torch.Tensor,
        pool: Optional[str] = None,
        store: bool = True
    ) -> Optional[torch.Tensor]:
        """
        Encode observations into context vectors θ using the attached encoder.
        Uses the encoder's pooling mode unless overridden. Ensures HSMM._context
        is always updated with the encoder's pooled context.
        """
        if self.encoder is None:
            if store:
                self._context = None
            return None

        device = next(self.encoder.parameters()).device
        X = X.to(device=device, dtype=DTYPE)

        # Normalize input to (B, T, F)
        if X.ndim == 1:  # single feature vector (F,)
            X = X.unsqueeze(0).unsqueeze(0)
        elif X.ndim == 2:  # (T, F)
            X = X.unsqueeze(0)
        elif X.ndim != 3:
            raise ValueError(f"Expected input of shape (F,), (T,F) or (B,T,F), got {X.shape}")

        # Temporarily override pooling if requested
        original_pool = getattr(self.encoder, "pool", None)
        if pool is not None:
            self.encoder.pool = pool

        try:
            # Forward pass through encoder
            _ = self.encoder(X, return_context=True)
            vec = self.encoder.get_context()
        except Exception as e:
            raise RuntimeError(f"Encoder forward() failed: {e}")
        finally:
            # Restore original pooling
            if pool is not None and original_pool is not None:
                self.encoder.pool = original_pool

        # Store context in HSMM if requested
        if store:
            self._context = vec.detach() if vec is not None else None

        return vec

    def check_constraints(self, value: torch.Tensor, clamp: bool = False) -> torch.Tensor:
        """
        Validate observations against the emission PDF support and event shape.
        Optionally clamps values outside the PDF support.

        Supports batched sequences [B, T, F] or single observations [T, F].

        Args:
            value (torch.Tensor): Input tensor (batch dims optional + event dims)
            clamp (bool): If True, clip values to PDF support instead of erroring.

        Returns:
            torch.Tensor: Validated (and possibly clamped) tensor.

        Raises:
            RuntimeError: If the emission PDF is not initialized.
            ValueError: If input shape does not match the PDF event shape.
        """
        pdf = self.pdf
        if pdf is None:
            raise RuntimeError(
                "Emission PDF not initialized. Call `sample_emission_pdf()` or `encode_observations()` first."
            )

        # Align device/dtype
        if hasattr(pdf, "mean") and isinstance(pdf.mean, torch.Tensor):
            value = value.to(device=pdf.mean.device, dtype=pdf.mean.dtype)

        event_shape = pdf.event_shape or ()
        expected_ndim = len(event_shape)
        
        # Add batch dimension if missing
        if value.ndim == expected_ndim:
            value = value.unsqueeze(0)

        # Support check
        if hasattr(pdf.support, "check"):
            support_mask = pdf.support.check(value)
            if clamp:
                if hasattr(pdf.support, "clamp"):
                    value = torch.where(support_mask, value, pdf.support.clamp(value))
                elif hasattr(pdf.support, "lower_bound") and hasattr(pdf.support, "upper_bound"):
                    value = value.clamp(min=pdf.support.lower_bound, max=pdf.support.upper_bound)
            elif not torch.all(support_mask):
                bad_vals = value[~support_mask].flatten().unique()
                raise ValueError(f"Values outside PDF support detected: {bad_vals.tolist()}")

        # Validate event shape
        if value.ndim < 1 + len(event_shape):
            raise ValueError(f"Expected at least {1 + len(event_shape)} dims (batch + event), got {value.ndim}")

        if event_shape and tuple(value.shape[-len(event_shape):]) != tuple(event_shape):
            raise ValueError(f"PDF event shape mismatch: expected {tuple(event_shape)}, got {tuple(value.shape[-len(event_shape):])}")

        return value

    def set_context(self, ctx: torch.Tensor):
        self._context = ctx
        if self.encoder is not None:
            self.encoder.set_context(ctx)

    def reset_context(self):
        self._context = None
        if self.encoder is not None:
            self.encoder.reset_context()

    def combine_context(self, theta: Optional[torch.Tensor], allow_broadcast: bool = True):
        if self.encoder is None:
            return theta
        return self.encoder._combine_context(theta, allow_broadcast)

    def _contextual_emission_pdf(
        self,
        X: torch.Tensor,
        theta: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute context-modulated emission log-probabilities [B, T, K].
        Always returns numeric log-probs, never a Distribution.
        """
        if X.ndim == 2:
            X = X.unsqueeze(0)
        elif X.ndim != 3:
            raise ValueError(f"Unsupported input shape {X.shape}")

        B, T, _ = X.shape

        theta_batch = self.combine_context(theta)
        if theta_batch is not None and theta_batch.ndim == 2:
            theta_batch = theta_batch.unsqueeze(1).expand(-1, T, -1)

        logp_list = []
        for b in range(B):
            ctx = None if theta_batch is None else theta_batch[b]
            logp_seq = self.emission_module.log_prob(X[b], context=ctx)
            if logp_seq.ndim == 1:
                logp_seq = logp_seq.unsqueeze(-1)
            logp_list.append(logp_seq)

        logp_tensor = torch.stack(logp_list, dim=0)
        logp_tensor = torch.nan_to_num(logp_tensor, neginf=-1e8)

        # Skip normalization if already in log-space
        return logp_tensor

    def _contextual_duration_pdf(
        self,
        theta: Optional[ContextualVariables] = None
    ) -> torch.Tensor:
        """
        Compute context-modulated duration log-probabilities.

        Args:
            theta: Optional context tensor.

        Returns:
            log_duration: [B, K, max_duration]
        """
        base_logits = self._duration_logits.unsqueeze(0)  # [1, K, Dmax]
        theta_batch = self.combine_context(theta)
        B = theta_batch.shape[0] if theta_batch is not None else 1

        # Apply context modulation
        log_duration = self.duration_module._apply_context(
            base_logits.expand(B, *base_logits.shape[1:]),
            theta_batch
        )

        if not torch.is_tensor(log_duration):
            raise TypeError(f"Expected tensor from _apply_context, got {type(log_duration)}")

        # Sanitize numerical issues before normalization
        log_duration = torch.nan_to_num(log_duration, nan=-1e8, posinf=1e8)

        # Normalize along duration axis
        log_duration = log_duration - torch.logsumexp(log_duration, dim=-1, keepdim=True)

        return log_duration

    def _contextual_transition_matrix(
        self,
        theta: Optional[ContextualVariables] = None
    ) -> torch.Tensor:
        """
        Compute context-modulated transition log-probabilities.

        Args:
            theta: Optional context tensor.

        Returns:
            log_transition: [B, K, K]
        """
        base_logits = self._transition_logits.unsqueeze(0)  # [1, K, K]
        theta_batch = self.combine_context(theta)
        B = theta_batch.shape[0] if theta_batch is not None else 1

        # Contextual modulation
        log_transition = self.transition_module._apply_context(
            base_logits.expand(B, *base_logits.shape[1:]),
            theta_batch
        )

        if not torch.is_tensor(log_transition):
            raise TypeError(f"Expected tensor from _apply_context, got {type(log_transition)}")

        if log_transition.ndim == 2:
            log_transition = log_transition.unsqueeze(0)  # [1, K, K]

        # Enforce structural transition constraints
        if hasattr(self, "transition_type"):
            mask = constraints.mask_invalid_transitions(self.n_states, self.transition_type).to(log_transition.device)
            log_transition = log_transition.masked_fill(~mask.unsqueeze(0), float("-inf"))

        # Sanitize numerics
        log_transition = torch.nan_to_num(log_transition, nan=-1e8, posinf=1e8)

        # Normalize rows (each source state's outgoing transitions)
        log_transition = log_transition - torch.logsumexp(log_transition, dim=-1, keepdim=True)

        return log_transition

    def to_observations(
        self,
        X: torch.Tensor,
        lengths: Optional[List[int]] = None,
        theta: Optional[torch.Tensor] = None
    ) -> utils.Observations:
        """
        Convert raw input X (and optional context theta) into a structured Observations object.
        Supports single or batched sequences, handles context, and computes context-modulated log-probabilities.

        Args:
            X: Observations tensor of shape [T, F] or [B, T, F].
            lengths: Optional list of sequence lengths.
            theta: Optional context tensor of shape [B, H] or [B, T, H].

        Returns:
            utils.Observations(sequence, log_probs, lengths, context)
        """
        device = next(self.parameters(), torch.tensor(0.0)).device
        X = torch.as_tensor(X, dtype=DTYPE, device=device) if not torch.is_tensor(X) else X.to(dtype=DTYPE, device=device)

        # --- Validate input ---
        X_valid = self.check_constraints(X)

        # --- Align context ---
        if theta is None and getattr(self, "_context", None) is not None:
            theta = self._context

        if theta is not None:
            theta = theta.to(dtype=DTYPE, device=device)

        # --- Sequence length handling ---
        total_len = X_valid.shape[0] if X_valid.ndim == 2 else X_valid.shape[1]
        if lengths is None:
            seq_lengths = [total_len]
        else:
            if not all(isinstance(l, int) for l in lengths):
                raise TypeError("`lengths` must be a list of integers.")
            if sum(lengths) != total_len:
                raise ValueError(f"Sum of lengths ({sum(lengths)}) does not match total samples ({total_len}).")
            seq_lengths = lengths

        # --- Split sequences ---
        sequences, context_list = [], []
        start_idx = 0
        for i, seq_len in enumerate(seq_lengths):
            end_idx = start_idx + seq_len
            seq = X_valid[start_idx:end_idx] if X_valid.ndim == 2 else X_valid[:, start_idx:end_idx, :]
            sequences.append(seq)

            # Context slicing
            if theta is None:
                context_list.append(None)
            else:
                if theta.ndim == 3:  # [B, T, H]
                    ctx = theta[:, start_idx:end_idx, :] if X_valid.ndim == 3 else theta[start_idx:end_idx]
                else:  # [B, H] or single [H]
                    ctx = theta[i] if theta.shape[0] > 1 else theta
                    ctx = ctx.expand(seq_len, -1)
                context_list.append(ctx)

            start_idx = end_idx

        # --- Compute log-probabilities ---
        log_probs_list = []
        for seq, seq_theta in zip(sequences, context_list):
            pdf_or_tensor = self._contextual_emission_pdf(seq, seq_theta)

            if isinstance(pdf_or_tensor, torch.distributions.Distribution):
                log_probs = pdf_or_tensor.log_prob(seq.unsqueeze(-2))
                if log_probs.ndim > 2:
                    log_probs = log_probs.flatten(start_dim=1)
            elif torch.is_tensor(pdf_or_tensor):
                log_probs = pdf_or_tensor
                if log_probs.ndim == 1:
                    log_probs = log_probs.unsqueeze(-1)
            else:
                raise TypeError(f"_contextual_emission_pdf must return a tensor or Distribution, got {type(pdf_or_tensor)}")

            log_probs_list.append(log_probs.to(dtype=DTYPE, device=device))

        return utils.Observations(
            sequence=sequences,
            lengths=seq_lengths,
            context=context_list,
            log_probs=log_probs_list
        )

    def _estimate_model_params(
        self,
        X: utils.Observations,
        theta: Optional[utils.ContextualVariables] = None
    ) -> dict[str, Any]:
        """
        M-step: Estimate updated HSMM parameters from posterior expectations (gamma, xi, eta).
        Returns singleton-shaped initial, transition, and duration logits, plus emission_pdf
        as a torch.distributions.Distribution (or Categorical if estimator returns logits tensor).
        """
        gamma_list, xi_list, eta_list = self._compute_posteriors(X)
        device = self.device
        K = self.n_states

        def safe_sum(lst: list[Optional[torch.Tensor]], dim=0) -> Optional[torch.Tensor]:
            total = None
            for x in lst:
                if x is not None and x.numel() > 0:
                    summed = x.sum(dim=0)
                    total = summed if total is None else total + summed
            return total

        # -------------------------------
        # Initial state distribution π [K]
        # -------------------------------
        init_counts = safe_sum([g[0] for g in gamma_list if g is not None])
        if init_counts is not None:
            new_init = constraints.log_normalize(torch.clamp_min(torch.log(init_counts), EPS), dim=0)
        else:
            new_init = self.init_logits.detach().clone().to(device=device, dtype=DTYPE)

        # -------------------------------
        # Transition matrix A [K, K]
        # -------------------------------
        trans_counts = safe_sum(xi_list)
        if trans_counts is not None:
            new_transition = constraints.log_normalize(torch.clamp_min(torch.log(trans_counts), EPS), dim=1)
        else:
            new_transition = self.transition_logits.detach().clone().to(device=device, dtype=DTYPE)

        # -------------------------------
        # Duration distributions P(d | z) [K, Dmax]
        # -------------------------------
        dur_counts = safe_sum(eta_list)
        if dur_counts is not None:
            new_duration = constraints.log_normalize(torch.clamp_min(torch.log(dur_counts), EPS), dim=1)
        else:
            new_duration = self.duration_logits.detach().clone().to(device=device, dtype=DTYPE)

        # -------------------------------
        # Emission PDF (Distribution)
        # -------------------------------
        new_pdf = self._params.get("emission_pdf", None)
        if theta is None:
            theta = getattr(self, "_context", None)

        if X.sequence and any(seq.numel() > 0 for seq in X.sequence):
            all_X = torch.cat(X.sequence, dim=0).to(device=device, dtype=DTYPE)
            all_gamma = torch.cat([g for g in gamma_list if g is not None and g.numel() > 0], dim=0).to(device=device, dtype=DTYPE)

            # Align context θ
            theta_full = None
            if theta is not None:
                theta = theta.to(device=device, dtype=DTYPE)
                if theta.ndim == 1:
                    theta_full = theta.unsqueeze(0).expand(all_X.shape[0], -1)
                elif theta.ndim == 2:
                    repeats = all_X.shape[0] // theta.shape[0]
                    if all_X.shape[0] % theta.shape[0] != 0:
                        raise ValueError(f"Theta shape {tuple(theta.shape)} cannot align with sequence length {all_X.shape[0]}")
                    theta_full = theta.repeat_interleave(repeats, dim=0)
                elif theta.ndim == 3:
                    theta_full = theta.reshape(-1, theta.shape[-1])
                    if theta_full.shape[0] != all_X.shape[0]:
                        raise ValueError(f"Context θ has shape {tuple(theta.shape)} but expected total length {all_X.shape[0]}")
                else:
                    raise ValueError(f"Unsupported θ shape: {tuple(theta.shape)}")

            # Compute emission PDF
            if hasattr(self, "_estimate_emission_pdf"):
                pdf_or_tensor = self._estimate_emission_pdf(all_X, all_gamma, theta_full)
            elif hasattr(self, "_contextual_emission_pdf"):
                pdf_or_tensor = self._contextual_emission_pdf(all_X, theta_full)
            else:
                raise RuntimeError("No emission estimator available.")

            # Wrap tensor logits as Categorical if needed
            if isinstance(pdf_or_tensor, torch.distributions.Distribution):
                new_pdf = pdf_or_tensor
            elif torch.is_tensor(pdf_or_tensor):
                t = pdf_or_tensor.to(device=device, dtype=DTYPE)
                if t.ndim == 2 and t.shape[1] == K:
                    new_pdf = Categorical(logits=t)
                else:
                    raise TypeError("Tensor emission shape not (N, K); use Distribution instead.")
            else:
                raise TypeError(f"Emission estimator returned unsupported type {type(pdf_or_tensor)}")

        # -------------------------------
        # Return singleton-shaped logits
        # -------------------------------
        return {
            "init_logits": new_init.to(device=device, dtype=DTYPE),
            "transition_logits": new_transition.to(device=device, dtype=DTYPE),
            "duration_logits": new_duration.to(device=device, dtype=DTYPE),
            "emission_pdf": new_pdf,
        }

    def sample_model_params(
        self,
        X: Optional[torch.Tensor] = None,
        theta: Optional[torch.Tensor] = None,
        inplace: bool = True
    ) -> dict[str, Any]:
        """
        Sample HSMM model parameters (init_logits, transition_logits, duration_logits, emission_pdf) 
        optionally using context-conditioned theta aligned to batch/sequence dimensions.
        Ensures emission_pdf is a torch.distributions.Distribution for consistency.

        Args:
            X: Optional observation tensor [T, F] or [B, T, F].
            theta: Optional context tensor [H], [T, H], [B, H], or [B, T, H].
            inplace: If True, update model attributes.

        Returns:
            dict with sampled 'init_logits', 'transition_logits', 'duration_logits', 'emission_pdf'.
        """

        α = getattr(self, "alpha", 1.0)
        dtype, device = DTYPE, self.device
        n_states, max_dur = self.n_states, self.max_duration

        # --- Determine batch & sequence size ---
        if X is not None:
            if X.ndim == 2:
                B, T = 1, X.shape[0]
            elif X.ndim == 3:
                B, T = X.shape[:2]
            else:
                raise ValueError(f"Unexpected X shape {X.shape}")
        else:
            B, T = 1, 1

        # --- Align theta to batch & sequence ---
        aligned_theta = None
        if theta is not None:
            theta = theta.to(dtype=dtype, device=device)
            if theta.ndim == 1:
                aligned_theta = theta.view(1, -1).expand(T, -1)  # [T,H]
            elif theta.ndim == 2:
                if theta.shape[0] == T:
                    aligned_theta = theta  # [T,H]
                elif B > 1 and theta.shape[0] == B:
                    aligned_theta = theta.unsqueeze(1).expand(B, T, -1)  # [B,T,H]
                elif T % theta.shape[0] == 0:
                    aligned_theta = theta.repeat(T // theta.shape[0], 1)  # repeat across time
                else:
                    raise ValueError(f"Theta shape {theta.shape} cannot align with T={T} or B={B}")
            elif theta.ndim == 3:
                if theta.shape[0] == B and theta.shape[1] == T:
                    aligned_theta = theta  # [B,T,H]
                elif theta.shape[0] == B:
                    aligned_theta = theta[:, -1, :].unsqueeze(1).expand(B, T, -1)  # repeat last time
                else:
                    raise ValueError(f"Theta shape {theta.shape} cannot align with B={B}, T={T}")
            else:
                raise ValueError(f"Unsupported theta shape: {theta.shape}")

        # --- Sample initial logits ---
        init_logits = torch.log(
            constraints.sample_probs(α, (n_states,)).to(dtype=dtype, device=device).clamp_min(EPS)
        )

        # --- Sample transition logits ---
        if aligned_theta is not None and hasattr(self, "_contextual_transition_matrix"):
            transition_logits = torch.log(
                self._contextual_transition_matrix(aligned_theta).clamp_min(EPS)
            )
        else:
            transition_logits = torch.log(
                constraints.sample_transition(α, n_states, constraints.Transitions.SEMI)
                .to(dtype=dtype, device=device)
                .clamp_min(EPS)
            )
        transition_logits = self._align_logits(transition_logits, B)

        # --- Sample duration logits ---
        if aligned_theta is not None and hasattr(self, "_contextual_duration_pdf"):
            duration_logits = torch.log(
                self._contextual_duration_pdf(aligned_theta).clamp_min(EPS)
            )
        else:
            duration_logits = torch.log(
                constraints.sample_probs(α, (n_states, max_dur))
                .to(dtype=dtype, device=device)
                .clamp_min(EPS)
            )
        duration_logits = self._align_logits(duration_logits, B)

        # --- Sample emission PDF ---
        if aligned_theta is not None and hasattr(self, "_contextual_emission_pdf") and X is not None:
            emission_pdf = self._contextual_emission_pdf(X, aligned_theta)
        else:
            emission_pdf = self.sample_emission_pdf(X, theta=aligned_theta)

        # --- Wrap tensor as Distribution if necessary ---
        if isinstance(emission_pdf, torch.Tensor):
            if emission_pdf.ndim == 2 and emission_pdf.shape[1] == n_states:
                emission_pdf = Categorical(logits=emission_pdf)
            else:
                emission_pdf = torch.distributions.Independent(
                    torch.distributions.Delta(emission_pdf), 1
                )
        elif not isinstance(emission_pdf, torch.distributions.Distribution):
            raise TypeError(f"Emission PDF must be tensor or Distribution, got {type(emission_pdf)}")

        # --- Inplace updates ---
        if inplace:
            self.init_logits.data.copy_(init_logits)  # singleton [K]
            self.transition_logits.data.copy_(transition_logits.squeeze(0))  # [K,K]
            self.duration_logits.data.copy_(duration_logits.squeeze(0))      # [K,Dmax]
            self._params["emission_pdf"] = emission_pdf

        return {
            "init_logits": init_logits,
            "transition_logits": transition_logits,
            "duration_logits": duration_logits,
            "emission_pdf": emission_pdf
        }


    # hsmm.HSMM precessing
    def _forward(self, X: utils.Observations, theta: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """
        Vectorized forward algorithm for HSMM with optional context θ.
        Supports batched or singleton sequences.
        Returns a list of log-alpha tensors of shape (T_i, K, Dmax) per sequence.
        """
        K, Dmax = self.n_states, self.max_duration
        neg_inf = -torch.inf
        device, dtype = self.device, DTYPE

        # --- Model parameters ---
        init_logits = self.init_logits.to(device=device, dtype=dtype)           # [K]
        transition_logits = self.transition_logits.to(device=device, dtype=dtype) # [K, K]
        duration_logits = self.duration_logits.to(device=device, dtype=dtype)     # [K, Dmax]

        # --- Prepare batch ---
        if not X.is_batch:
            seq_batch = [seq.to(device=device, dtype=dtype) for seq in X.log_probs]
            lengths = X.lengths
        else:
            batch_out = X.to_batch(return_mask=False, include_log_probs=True)
            if isinstance(batch_out, tuple):
                seq_batch = batch_out[1].to(device=device, dtype=dtype)
            else:
                seq_batch = batch_out.to(device=device, dtype=dtype)
            lengths = X.lengths

        B = len(lengths)
        alpha_list: List[torch.Tensor] = []

        for b, seq_len in enumerate(lengths):
            if seq_len == 0:
                alpha_list.append(torch.full((0, K, Dmax), neg_inf, device=device, dtype=dtype))
                continue

            log_probs = seq_batch[b][:seq_len]  # ideally [T, K]

            # --- Fallback: if Distribution instead of tensor ---
            if isinstance(log_probs, torch.distributions.Distribution):
                try:
                    seq = X.sequence[b].to(device=device, dtype=dtype)
                    log_probs = log_probs.log_prob(seq.unsqueeze(-2))
                    if log_probs.ndim > 2:
                        log_probs = log_probs.sum(dim=tuple(range(2, log_probs.ndim)))
                except Exception as e:
                    raise RuntimeError(f"Failed to extract log_probs from Distribution: {e}")

            if not torch.is_tensor(log_probs):
                raise TypeError(f"log_probs must be a tensor, got {type(log_probs)}")

            # --- If context θ is provided, modulate log_probs ---
            if theta is not None:
                theta_seq = theta if theta.shape[0] == seq_len else theta.repeat_interleave(
                    seq_len // theta.shape[0], dim=0
                )
                if hasattr(self, "_contextual_emission_pdf"):
                    log_probs = self._contextual_emission_pdf(log_probs, theta_seq)

            # --- Precompute cumulative sums over durations ---
            cumsum_emit = torch.vstack((
                torch.zeros((1, K), device=device, dtype=dtype),
                torch.cumsum(log_probs, dim=0)
            ))  # [T+1, K]

            # --- Initialize log-alpha ---
            log_alpha = torch.full((seq_len, K, Dmax), neg_inf, device=device, dtype=dtype)

            max_d0 = min(Dmax, seq_len)
            durations0 = torch.arange(1, max_d0 + 1, device=device)
            emit_sums0 = (cumsum_emit[durations0] - cumsum_emit[0]).T  # [K, max_d0]
            log_alpha[0, :, :max_d0] = init_logits.unsqueeze(1) + duration_logits[:, :max_d0] + emit_sums0

            # --- Recursion ---
            for t in range(1, seq_len):
                max_dt = min(Dmax, t + 1)
                durations = torch.arange(1, max_dt + 1, device=device)
                starts = t - durations + 1

                emit_sums_t = (cumsum_emit[t + 1].unsqueeze(0) - cumsum_emit[starts]).T  # [K, max_dt]

                idx = torch.clamp(starts - 1, min=0)
                prev_alpha_first = log_alpha[idx, :, 0]  # [max_dt, K]

                mask = (starts == 0).unsqueeze(1)
                prev_alpha_first = torch.where(mask, init_logits.unsqueeze(0), prev_alpha_first)

                prev_alpha_exp = prev_alpha_first.unsqueeze(2) + transition_logits.unsqueeze(0)  # [max_dt, K, K]
                log_alpha_t = torch.logsumexp(prev_alpha_exp, dim=1).T  # [K, max_dt]

                log_alpha[t, :, :max_dt] = log_alpha_t + duration_logits[:, :max_dt] + emit_sums_t

            alpha_list.append(log_alpha)

        return alpha_list

    def _backward(self, X: utils.Observations, theta: Optional[torch.Tensor] = None) -> list[torch.Tensor]:
        """
        Vectorized backward pass for HSMM sequences (batched or singleton),
        context-aware via `theta`.
        Returns a list of log-beta tensors: shape (T_i, K, Dmax) per sequence.
        """
        device, dtype = self.device, DTYPE
        K, Dmax = self.n_states, self.max_duration
        neg_inf = -torch.inf

        # Model parameters
        transition_logits = self.transition_logits.to(device, dtype=dtype)  # [K, K]
        duration_logits = self.duration_logits.to(device, dtype=dtype)      # [K, Dmax]

        # Compute emission log-probabilities (context-aware if available)
        if theta is not None and hasattr(self, "_contextual_emission_pdf"):
            log_probs_list = []
            for seq in X.sequence:
                if seq.numel() == 0:
                    log_probs_list.append(torch.zeros((0, K), dtype=dtype, device=device))
                    continue

                pdf = self._contextual_emission_pdf(seq, theta.to(device, dtype=dtype))
                # Fallback: ensure pdf is tensor
                if isinstance(pdf, torch.distributions.Distribution):
                    # if pdf is distribution, use its log_prob over the data seq
                    seq_probs = pdf.log_prob(seq).reshape(-1, K)
                else:
                    seq_probs = torch.as_tensor(pdf, dtype=dtype, device=device)
                log_probs_list.append(seq_probs)
        else:
            log_probs_list = [
                torch.as_tensor(seq, dtype=dtype, device=device)
                for seq in X.log_probs
            ]

        beta_list = []

        for seq_idx, (seq_probs, seq_len) in enumerate(zip(log_probs_list, X.lengths)):
            if seq_len == 0:
                beta_list.append(torch.full((0, K, Dmax), neg_inf, device=device, dtype=dtype))
                continue

            # Truncate to actual sequence length and ensure correct shape
            seq_probs = seq_probs[:seq_len, :K] if seq_probs.dim() == 2 else seq_probs[:seq_len].unsqueeze(-1)
            if seq_probs.shape[1] != K:
                seq_probs = seq_probs.expand(-1, K)

            # Initialize beta
            log_beta = torch.full((seq_len, K, Dmax), neg_inf, device=device, dtype=dtype)

            # Safe cumulative sum
            cumsum_emit = torch.vstack((
                torch.zeros((1, K), dtype=dtype, device=device),
                torch.cumsum(seq_probs, dim=0)
            ))  # [T+1, K]

            log_beta[-1, :, 0] = 0.0  # last time step

            # Backward recursion
            for t in reversed(range(seq_len - 1)):
                max_dt = min(Dmax, seq_len - t)
                durations = torch.arange(1, max_dt + 1, device=device)
                ends = t + durations  # [max_dt]

                emit_start = cumsum_emit[t].unsqueeze(0).expand(max_dt, -1)  # [max_dt, K]
                emit_end = cumsum_emit[ends, :]                              # [max_dt, K]
                emit_sums = emit_end - emit_start                            # [max_dt, K]

                beta_next = log_beta[ends - 1, :, 0]                         # [max_dt, K]
                dur_scores = duration_logits[:, :max_dt].T                   # [max_dt, K]

                combined = emit_sums + beta_next + dur_scores
                log_beta[t, :, 0] = torch.logsumexp(combined.T, dim=1)

                if max_dt > 1:
                    shift_len = max_dt - 1
                    log_beta[t, :, 1:shift_len + 1] = (
                        log_beta[t + 1, :, :shift_len] + seq_probs[t + 1].unsqueeze(-1)
                    )

            beta_list.append(log_beta)

        return beta_list

    def _compute_posteriors(
        self,
        X: utils.Observations,
        theta: Optional[torch.Tensor] = None
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Compute HSMM posteriors: gamma, xi, eta (fully vectorized).

        Returns lists of tensors per sequence:
            gamma: [T_i, K]
            xi:    [T_i-1, K, K]
            eta:   [T_i, K, Dmax]

        Supports context-modulated emissions via theta.
        """
        K, Dmax = self.n_states, self.max_duration
        gamma_vec, xi_vec, eta_vec = [], [], []

        alpha_list = self._forward(X, theta=theta)
        beta_list = self._backward(X, theta=theta)

        init_logits = self.init_logits.to(device=self.device, dtype=DTYPE)
        transition_logits = self.transition_logits.to(device=self.device, dtype=DTYPE)

        for seq_probs, seq_len, alpha, beta in zip(X.log_probs, X.lengths, alpha_list, beta_list):
            if seq_len == 0:
                gamma_vec.append(torch.zeros((0, K), dtype=DTYPE, device=self.device))
                eta_vec.append(torch.zeros((0, K, Dmax), dtype=DTYPE, device=self.device))
                xi_vec.append(torch.zeros((0, K, K), dtype=DTYPE, device=self.device))
                continue

            alpha = alpha.to(self.device, dtype=DTYPE)
            beta = beta.to(self.device, dtype=DTYPE)
            seq_probs = seq_probs.to(self.device, dtype=DTYPE)

            # --- Gamma ---
            log_gamma = torch.logsumexp(alpha + beta, dim=2)
            log_gamma -= torch.logsumexp(log_gamma, dim=1, keepdim=True)
            gamma = torch.clamp(log_gamma.exp(), min=EPS)
            gamma /= gamma.sum(dim=1, keepdim=True).clamp_min(EPS)
            gamma_vec.append(gamma)

            # --- Eta (fully vectorized) ---
            # alpha + beta already shape [T, K, Dmax]
            log_eta = alpha + beta  # [T, K, Dmax]
            log_eta_flat = log_eta.view(seq_len, -1)
            log_eta_flat -= torch.logsumexp(log_eta_flat, dim=1, keepdim=True)
            eta = torch.clamp(log_eta_flat.exp().view(seq_len, K, Dmax), min=EPS)
            eta /= eta.sum(dim=(1,2), keepdim=True).clamp_min(EPS)
            eta_vec.append(eta)

            # --- Xi (vectorized over durations) ---
            if seq_len > 1:
                t_idx = torch.arange(1, seq_len, device=self.device)               # [T-1]
                d_idx = torch.arange(1, Dmax + 1, device=self.device).view(1, -1)  # [1, Dmax]
                start_idx = t_idx.view(-1, 1) - d_idx + 1                            # [T-1, Dmax]
                mask = start_idx >= 0
                start_idx_clamped = start_idx.clamp(min=0)

                # Gather previous alpha for all durations at once
                prev_alpha_first = alpha[start_idx_clamped, :, 0]                     # [T-1, Dmax, K]
                prev_alpha_first = torch.where(mask.unsqueeze(2), prev_alpha_first, init_logits.unsqueeze(0).unsqueeze(1))

                # Sum over durations
                alpha_sum = torch.logsumexp(prev_alpha_first, dim=1)                 # [T-1, K]
                beta_next = torch.logsumexp(beta[1:, :, :Dmax], dim=2)               # [T-1, K]

                log_xi = alpha_sum.unsqueeze(2) + transition_logits.unsqueeze(0) + beta_next.unsqueeze(1)  # [T-1, K, K]
                xi_seq = torch.clamp(torch.softmax(log_xi, dim=2), min=EPS)
                xi_seq /= xi_seq.sum(dim=(1,2), keepdim=True).clamp_min(EPS)
                xi_vec.append(xi_seq)
            else:
                xi_vec.append(torch.zeros((0, K, K), dtype=DTYPE, device=self.device))

        return gamma_vec, xi_vec, eta_vec

    def _map(self, X: utils.Observations) -> list[torch.Tensor]:
        """Vectorized MAP decoding of HSMM sequences from posterior state marginals."""
        gamma_list, _, _ = self._compute_posteriors(X)
        if gamma_list is None:
            raise RuntimeError("Posterior probabilities could not be computed — model parameters uninitialized.")

        # Sequence lengths and batch info
        seq_lengths = torch.tensor([g.shape[0] for g in gamma_list], device=self.device)
        B = len(seq_lengths)
        if B == 0:
            return []

        max_T = seq_lengths.max().item()
        K = self.n_states
        neg_inf = torch.tensor(-float("inf"), device=self.device, dtype=DTYPE)

        if max_T == 0:
            return [torch.empty(0, dtype=torch.long, device=self.device) for _ in range(B)]

        # Create padded tensor for gamma
        gamma_padded = neg_inf.repeat(B, max_T, K)
        for i, g in enumerate(gamma_list):
            gamma_padded[i, : g.shape[0]] = g

        # MAP decoding
        map_padded = gamma_padded.argmax(dim=-1)

        # Split sequences using indexing
        # This avoids explicit Python loops for slicing
        batch_indices = torch.arange(B, device=self.device).repeat_interleave(seq_lengths)
        time_indices = torch.cat([torch.arange(L, device=self.device) for L in seq_lengths])
        map_sequences = map_padded[batch_indices, time_indices]

        # Recover list per sequence
        splits = torch.cumsum(seq_lengths, dim=0)[:-1]
        map_sequences_list = map_sequences.split(seq_lengths.tolist())
        map_sequences_list = [seq.to(dtype=torch.long, device=self.device) for seq in map_sequences_list]

        return map_sequences_list

    def _viterbi(self, X: utils.Observations, theta: Optional[torch.Tensor] = None, duration_weight: float = 0.0) -> list[torch.Tensor]:
        """
        Batched, mostly-vectorized HSMM Viterbi decoder (batch x state x duration).
        Returns a list of integer state sequences (one per observation sequence).
        """
        device, dtype = self.device, DTYPE
        K, Dmax = self.n_states, self.max_duration

        # Quick path for empty input
        B = X.n_sequences
        lengths = X.lengths
        max_len = max(lengths) if lengths else 0
        if max_len == 0:
            return [torch.empty(0, dtype=torch.int64, device=device) for _ in range(B)]

        # Model parameters
        init_logits = self.init_logits.to(device=device, dtype=dtype)           # [K]
        transition_logits = self.transition_logits.to(device=device, dtype=dtype)  # [K,K]
        duration_logits = self.duration_logits.to(device=device, dtype=dtype)      # [K,Dmax]
        neg_inf = -torch.inf

        # Optional duration weighting
        dur_indices = torch.arange(1, Dmax + 1, device=device, dtype=dtype)
        if duration_weight > 0.0:
            dur_mean = (torch.softmax(duration_logits, dim=1) * dur_indices).sum(dim=1)  # [K]
            dur_penalty = -((dur_indices.unsqueeze(0) - dur_mean.unsqueeze(1)) ** 2) / (2 * Dmax)
            dur_lp = (1 - duration_weight) * duration_logits + duration_weight * dur_penalty
        else:
            dur_lp = duration_logits  # [K, Dmax]

        # --- Build batched emission cumsums ---
        # Use Observations.to_batch to get a padded [B, T, D] and mask
        seq_batch, mask, log_batch, ctx_batch = X.to_batch(return_mask=True, include_log_probs=True, include_context=True)
        # seq_batch: [B, T, D] (observations), log_batch: [B, T, K] (log-probs)
        # prefer context-aware emission log-probs if available
        if theta is not None and hasattr(self, "_contextual_emission_pdf"):
            # user-provided theta takes precedence
            emission = self._contextual_emission_pdf(seq_batch if seq_batch.ndim == 3 else seq_batch.unsqueeze(-1),
                                                    theta.to(device=device, dtype=dtype))  # [B_total, T?, K] or may be [B, T, K]
            # if returned shape is [B, T, K], good. If it's [total_T, K] then we need to reshape to [B, T, K]
            if emission.ndim == 2 and emission.shape[0] == seq_batch.view(-1, seq_batch.shape[-1]).shape[0]:
                # reshape to [B, T, K]
                emission = emission.view(B, seq_batch.shape[1], -1)
            emission_log = emission.to(device=device, dtype=dtype)
        else:
            emission_log = log_batch.to(device=device, dtype=dtype)  # [B, T, K]

        # mask: [B, T], emission_log: [B, T, K]
        T = seq_batch.shape[1]

        # Compute cumulative sums over time for efficient duration sums:
        # cumsum has shape [B, T+1, K], with cumsum[:,0,:] = 0
        zero_pad = torch.zeros((B, 1, K), device=device, dtype=dtype)
        cumsum_emit = torch.cat([zero_pad, torch.cumsum(emission_log, dim=1)], dim=1)  # [B, T+1, K]

        # DP tensors
        V = torch.full((B, max_len, K), neg_inf, device=device, dtype=dtype)
        best_prev = torch.full((B, max_len, K), -1, dtype=torch.int64, device=device)
        best_dur = torch.zeros((B, max_len, K), dtype=torch.int64, device=device)

        # Precompute range of durations once
        dur_range = torch.arange(1, Dmax + 1, device=device, dtype=torch.long)  # [Dmax]

        # For each time t, compute scores for all active batches
        for t in range(max_len):
            # which sequences are active at time t?
            active_mask = torch.tensor([t < L for L in lengths], device=device)
            if not active_mask.any():
                continue
            active_idx = active_mask.nonzero(as_tuple=True)[0]  # indices of active sequences
            B_active = active_idx.numel()

            max_d = min(Dmax, t + 1)  # allowed durations ending at t
            durations = dur_range[:max_d]  # [max_d]

            # starts[b, d] = t - durations[d] + 1
            # create starts: [B_active, max_d]
            starts = (t - durations.unsqueeze(0) + torch.zeros((B_active, 1), device=device, dtype=torch.long)).clamp(min=0)  # [B_active, max_d]

            # gather cumsum at starts and at t+1
            # cumsum_emit: [B, T+1, K]; we want cumsum_emit[b, starts[b,d], :] and cumsum_emit[b, t+1, :]
            # Prepare index tensors for gather:
            starts_idx = starts.unsqueeze(-1).expand(-1, -1, K)  # [B_active, max_d, K]
            # gather along time-dim (dim=1). Need to index into cumsum_emit[active_idx] first
            cumsum_active = cumsum_emit[active_idx]  # [B_active, T+1, K]
            cumsum_starts = torch.gather(cumsum_active, 1, starts_idx)  # [B_active, max_d, K]
            cumsum_end = cumsum_active[:, t + 1, :].unsqueeze(1).expand(-1, max_d, -1)  # [B_active, max_d, K]
            emit_sums = (cumsum_end - cumsum_starts).permute(0, 2, 1)  # [B_active, K, max_d]

            # duration scores, broadcast to batch: [B_active, K, max_d]
            dur_scores = dur_lp[:, :max_d].unsqueeze(0).expand(B_active, -1, -1)  # [B_active, K, max_d]

            if t == 0:
                # scores: init + dur_scores + emit_sums
                scores = init_logits.unsqueeze(0).unsqueeze(2) + dur_scores + emit_sums  # [B_active, K, max_d]
                prev_idx = torch.full((B_active, K, max_d), -1, dtype=torch.int64, device=device)
            else:
                # We need previous V values at start positions: prev_V[b,i,d,:] = V[b, starts[b,d], :]
                # gather previous V
                starts_idx_for_V = starts.unsqueeze(-1).expand(-1, -1, K)  # [B_active, max_d, K]
                prev_V = torch.gather(V[active_idx], 1, starts_idx_for_V)  # [B_active, max_d, K]
                # prev_V is [B_active, max_d, K]; we want shape [B_active, max_d, K_prev] to combine with transition
                # If the duration started at 0 (i.e. starts==0 and that implies prev state is init), override those rows
                mask_start0 = (starts == 0)  # [B_active, max_d]
                if mask_start0.any():
                    # replace prev_V[mask_start0] with init_logits
                    # create init expanded [B_active, K], then expand to positions
                    init_exp = init_logits.unsqueeze(0).unsqueeze(1).expand(B_active, max_d, K)
                    prev_V = torch.where(mask_start0.unsqueeze(-1), init_exp, prev_V)

                # Now combine prev_V with transition matrix:
                # prev_V: [B_active, max_d, K_prev], transition_logits: [K_prev, K]
                # we want best over prev_state -> curr_state: for each curr_state, max over prev_state
                # compute prev_V.unsqueeze(3) + transition_logits (broadcasted)
                prev_V_exp = prev_V.unsqueeze(3)  # [B_active, max_d, K_prev, 1]
                trans_exp = transition_logits.unsqueeze(0).unsqueeze(0)  # [1,1,K_prev,K]
                scores_plus_trans = prev_V_exp + trans_exp  # [B_active, max_d, K_prev, K]
                # max over prev_state dim (2)
                scores_max, argmax_prev = torch.max(scores_plus_trans, dim=2)  # both [B_active, max_d, K]
                # we want shape [B_active, K, max_d] like emit_sums/dur_scores
                scores = (scores_max.permute(0, 2, 1) + dur_scores + emit_sums)  # [B_active, K, max_d]
                # prev_idx should be argmax_prev permuted to [B_active, K, max_d]
                prev_idx = argmax_prev.permute(0, 2, 1).to(torch.int64)

            # pick best duration per current state
            best_score, best_d_idx = torch.max(scores, dim=2)  # [B_active, K]
            # fill global V, best_prev, best_dur for active indices
            for i, b in enumerate(active_idx):
                V[b, t] = best_score[i]
                # best duration numeric
                best_dur[b, t] = durations[best_d_idx[i]]
                # best previous state index (for each current state) = prev_idx[i, :, best_d_idx[i]]
                # prev_idx[i] shape: [K, max_d]
                chosen_prev = prev_idx[i, torch.arange(K, device=device), best_d_idx[i]]
                best_prev[b, t] = chosen_prev

        # --- Backtrace per sequence ---
        paths = []
        for b, L in enumerate(lengths):
            if L == 0:
                paths.append(torch.empty(0, dtype=torch.int64, device=device))
                continue

            t = L - 1
            # pick best curr state at final time
            cur_state = int(torch.argmax(V[b, t]).item())
            segments = []
            while t >= 0:
                d = int(best_dur[b, t, cur_state].item())
                if d <= 0:
                    break
                start = max(0, t - d + 1)
                segments.append((start, t, cur_state))
                prev_state = int(best_prev[b, t, cur_state].item())
                t = start - 1
                cur_state = prev_state if prev_state >= 0 else cur_state

            segments.reverse()
            if segments:
                seq_path = torch.cat([
                    torch.full((end - start + 1,), st, dtype=torch.int64, device=device)
                    for (start, end, st) in segments
                ])
            else:
                seq_path = torch.empty(0, dtype=torch.int64, device=device)

            # pad/truncate to L
            if seq_path.numel() < L:
                pad_len = L - seq_path.numel()
                if seq_path.numel() == 0:
                    fill_val = 0
                else:
                    fill_val = int(seq_path[-1].item())
                seq_path = torch.cat([seq_path, torch.full((pad_len,), fill_val, dtype=torch.int64, device=device)])
            elif seq_path.numel() > L:
                seq_path = seq_path[:L]

            paths.append(seq_path)

        return paths

    def _compute_log_likelihood(self, X: utils.Observations, verbose: bool = False) -> torch.Tensor:
        """
        HSMM log-likelihood computation. Supports both batched tensors and list-of-tensors outputs
        from `_forward()`. Handles variable-length sequences and stores per-sequence log-likelihoods
        in X.log_likelihoods.

        Returns:
            torch.Tensor of shape (B,) with per-sequence log-likelihoods.
        """
        alpha_out = self._forward(X)
        if alpha_out is None:
            raise RuntimeError("Forward pass returned None. Model may be uninitialized.")

        dtype, device = DTYPE, self.device
        neg_inf = float("-inf")

        # --- Convert to list of tensors for uniform processing ---
        if isinstance(alpha_out, torch.Tensor):
            if alpha_out.ndim != 4:
                raise ValueError(f"Expected [B,T,K,Dmax], got {tuple(alpha_out.shape)}")
            alpha_list = [alpha_out[b] for b in range(alpha_out.shape[0])]
        elif isinstance(alpha_out, (list, tuple)):
            if not alpha_out:
                raise RuntimeError("Forward pass returned empty list of alpha tensors.")
            alpha_list = list(alpha_out)
        else:
            raise TypeError(f"Unexpected _forward output type: {type(alpha_out).__name__}")

        B = len(alpha_list)
        seq_lengths = torch.tensor([a.shape[0] for a in alpha_list], device=device, dtype=torch.long)
        max_T = seq_lengths.max().item() if B > 0 else 0

        if max_T == 0:
            ll = torch.full((B,), neg_inf, dtype=dtype, device=device)
            X.log_likelihoods = ll
            return ll

        # --- Pad sequences to max_T for vectorized logsumexp ---
        alpha_padded = torch.nn.utils.rnn.pad_sequence(
            [a.nan_to_num(nan=neg_inf, posinf=neg_inf, neginf=neg_inf) for a in alpha_list],
            batch_first=True,
            padding_value=neg_inf
        )

        # --- Gather last valid timestep for each sequence ---
        last_idx = (seq_lengths - 1).clamp(min=0)
        last_alpha = alpha_padded[torch.arange(B, device=device), last_idx]
        ll = torch.logsumexp(last_alpha.flatten(start_dim=1), dim=-1)
        ll = torch.where(seq_lengths > 0, ll, torch.full_like(ll, neg_inf))

        X.log_likelihoods = ll.detach().clone()

        if verbose:
            print(f"[compute_ll] alpha_padded shape={alpha_padded.shape}, seq_lengths={seq_lengths.tolist()}")
            print(f"[compute_ll] log-likelihoods min={ll.min():.4f}, max={ll.max():.4f}")

        return ll

    def fit(
        self,
        X: torch.Tensor,
        n_init: int = 1,
        tol: float = 1e-4,
        max_iter: int = 15,
        post_conv_iter: int = 1,
        ignore_conv: bool = False,
        sample_D_from_X: bool = False,
        lengths: Optional[List[int]] = None,
        theta: Optional[torch.Tensor] = None,
        plot_conv: bool = False,
        verbose: bool = True,
    ):
        """
        Vectorized, GPU-friendly EM fitting for HSMM with optional batching and context support.
        """

        device = self.device

        # --- Optional emission initialization ---
        if sample_D_from_X:
            self._params['emission_pdf'] = self.sample_emission_pdf(X)

        # --- Encode observations if needed ---
        if theta is None and getattr(self, "encoder", None):
            theta = self.encode_observations(X)

        X_valid = self.to_observations(X, lengths, theta)

        # --- Align theta with sequence layout ---
        if theta is not None:
            total_len = sum(X_valid.lengths)
            if theta.shape[0] != total_len:
                if theta.shape[0] == len(X_valid.sequence):
                    theta = torch.cat(
                        [theta[i].expand(l, -1) for i, l in enumerate(X_valid.lengths)], dim=0
                    )
                elif total_len % theta.shape[0] == 0:
                    repeat_factor = total_len // theta.shape[0]
                    if total_len != theta.shape[0] * repeat_factor:
                        raise ValueError(
                            f"Theta repetition misaligned: {theta.shape[0]} x {repeat_factor} != {total_len}"
                        )
                    theta = theta.repeat(repeat_factor, 1)
                else:
                    raise ValueError(f"Theta shape {theta.shape} cannot align with total length {total_len}")
            valid_theta = theta.to(device)
        else:
            valid_theta = None

        # --- Convergence tracker ---
        self.conv = ConvergenceHandler(
            tol=tol, max_iter=max_iter, n_init=n_init, post_conv_iter=post_conv_iter, verbose=verbose
        )

        best_score, best_state = -float("inf"), None

        # --- Flatten sequences ---
        nonempty_idx = [i for i, l in enumerate(X_valid.lengths) if l > 0]
        seq_tensor = torch.cat([X_valid.sequence[i] for i in nonempty_idx], dim=0).to(DTYPE).to(device)
        seq_offsets = torch.cumsum(
            torch.tensor([0] + [X_valid.lengths[i] for i in nonempty_idx], device=device), dim=0
        )

        def _squeeze_logits(t: torch.Tensor) -> torch.Tensor:
            return t.squeeze(0) if t.ndim == 3 and t.shape[0] == 1 else t

        # --- EM runs ---
        for run_idx in range(n_init):
            if verbose:
                print(f"\n=== Run {run_idx + 1}/{n_init} ===")

            # Controlled randomization
            g = torch.Generator(device=device).manual_seed(run_idx + 12345)

            if run_idx > 0:
                sampled = self.sample_model_params(X, valid_theta, generator=g)
                self._init_logits.copy_(_squeeze_logits(sampled['init_logits']))
                self._transition_logits.copy_(_squeeze_logits(sampled['transition_logits']))
                self._duration_logits.copy_(_squeeze_logits(sampled['duration_logits']))
                self._params['emission_pdf'] = sampled['emission_pdf']

            # Initial likelihood
            curr_ll = self._compute_log_likelihood(X_valid).sum()
            self.conv.push_pull(curr_ll, 0, run_idx)

            # --- EM iterations ---
            for it in range(1, max_iter + 1):
                pdf = self._params.get('emission_pdf', None)
                if pdf is None or not hasattr(pdf, 'log_prob'):
                    raise RuntimeError("Emission PDF not initialized or incompatible with log_prob().")

                all_log_probs = pdf.log_prob(seq_tensor.unsqueeze(1))  # (T_total, K) or (B, T_total, K)
                if all_log_probs.ndim == 3 and all_log_probs.shape[0] == 1:
                    all_log_probs = all_log_probs.squeeze(0)

                # Rebuild log_probs per sequence
                X_valid.log_probs = [
                    all_log_probs[seq_offsets[i]:seq_offsets[i + 1]] for i in range(len(seq_offsets) - 1)
                ]
                for i, l in enumerate(X_valid.lengths):
                    if l == 0:
                        X_valid.log_probs[i] = torch.empty((0, self.n_states), dtype=DTYPE, device=device)

                # --- M-step ---
                new_params = self._estimate_model_params(X_valid, valid_theta)
                if not new_params:
                    raise RuntimeError("M-step returned no parameters.")

                # Update in-place for stability
                if 'init_logits' in new_params:
                    self._init_logits.copy_(_squeeze_logits(new_params['init_logits']))
                if 'transition_logits' in new_params:
                    self._transition_logits.copy_(_squeeze_logits(new_params['transition_logits']))
                if 'duration_logits' in new_params:
                    self._duration_logits.copy_(_squeeze_logits(new_params['duration_logits']))
                if 'emission_pdf' in new_params:
                    self._params['emission_pdf'] = new_params['emission_pdf']

                # --- Likelihood & convergence ---
                curr_ll = self._compute_log_likelihood(X_valid).sum()
                converged = self.conv.push_pull(curr_ll, it, run_idx)

                if converged and not ignore_conv:
                    if verbose:
                        print(f"[Run {run_idx + 1}] Converged at iteration {it}.")
                    break

            # --- Best run tracking ---
            run_score = float(curr_ll.item())
            if run_score > best_score:
                best_score = run_score
                best_state = {
                    'init_logits': self._init_logits.clone(),
                    'transition_logits': self._transition_logits.clone(),
                    'duration_logits': self._duration_logits.clone(),
                    'emission_pdf': self._params['emission_pdf'],
                }

        # --- Restore best parameters ---
        if best_state:
            self._init_logits.copy_(best_state['init_logits'])
            self._transition_logits.copy_(best_state['transition_logits'])
            self._duration_logits.copy_(best_state['duration_logits'])
            self._params['emission_pdf'] = best_state['emission_pdf']

        if plot_conv and hasattr(self, 'conv'):
            self.conv.plot_convergence()

        return self, best_score

    @torch.no_grad()
    def predict(
        self,
        X: torch.Tensor,
        lengths: Optional[list[int]] = None,
        algorithm: Literal["map", "viterbi"] = "viterbi",
        context: Optional[torch.Tensor] = None,
        batch_size: int = 256,
    ) -> list[torch.Tensor]:
        """
        Batched MAP or Viterbi decoding for HSMM sequences with explicit duration modeling.
        Supports contextual modulation, variable-length sequences, and GPU execution.

        Args:
            X: Observations [B, T, D] or concatenated [sum(T_i), D]
            lengths: Optional list of valid sequence lengths
            algorithm: 'viterbi' (segmental MAP) or 'map' (sum-product-style)
            context: Optional context tensor [B, C] or [T, C]
            batch_size: Emission evaluation batch size

        Returns:
            list[torch.Tensor]: decoded state sequences per sample (on CPU)
        """
        # ---- Setup ----
        X_valid = self.to_observations(X, lengths)
        B = len(X_valid.lengths)
        K = self.n_states
        max_T = max(X_valid.lengths) if X_valid.lengths else 0
        device = self.device
        dtype = DTYPE

        if max_T == 0:
            return [torch.empty(0, dtype=torch.long) for _ in range(B)]

        # ---- Context normalization ----
        if context is not None:
            if context.ndim == 1:
                context = context.unsqueeze(0)
            elif context.ndim > 2:
                context = context.mean(dim=0, keepdim=True)
            context = context.to(device=device, dtype=dtype)

        # ---- Emission log-probabilities (batched evaluation) ----
        seq_tensor = torch.cat(X_valid.sequence, dim=0) if X_valid.sequence else torch.empty((0, 0))
        seq_tensor = seq_tensor.to(device=device, dtype=dtype)
        seq_offsets = [0] + list(torch.cumsum(torch.tensor(X_valid.lengths, device=device), dim=0).tolist())

        pdf = getattr(self, "_params", {}).get("emission_pdf", None)
        if pdf is None or not hasattr(pdf, "log_prob"):
            raise RuntimeError("Emission PDF or module required for log_probs.")

        all_log_probs = []
        if seq_tensor.numel() == 0:
            all_log_probs = torch.empty((0, K), device=device, dtype=dtype)
        else:
            # batch the emission eval to avoid OOM
            for start in range(0, seq_tensor.shape[0], batch_size):
                end = min(start + batch_size, seq_tensor.shape[0])
                chunk = seq_tensor[start:end].unsqueeze(1)  # (N, 1, D)
                chunk_context = None
                if context is not None:
                    # If context is provided per-time/per-sample, slice accordingly; else pass None
                    if context.shape[0] == seq_tensor.shape[0]:
                        chunk_context = context[start:end]
                    elif context.shape[0] == 1:
                        chunk_context = context  # broadcast singleton
                # sample_emission_pdf expects theta keyword in your code paths; keep same call
                pdf_chunk = self.sample_emission_pdf(theta=chunk_context)
                lp = pdf_chunk.log_prob(chunk)
                if lp.ndim > 2:
                    # sum event dims if any (e.g. multivariate)
                    lp = lp.flatten(start_dim=2).sum(dim=2)
                all_log_probs.append(lp.to(device=device, dtype=dtype))
            all_log_probs = torch.cat(all_log_probs, dim=0)

        # split back per-sequence
        X_valid.log_probs = [
            all_log_probs[seq_offsets[i]:seq_offsets[i+1]] if seq_offsets[i] != seq_offsets[i+1]
            else torch.empty((0, K), device=device, dtype=dtype)
            for i in range(B)
        ]

        # ---- Model logits shaped per-batch ----
        # Align / broadcast initial/transition/duration logits to per-batch shapes when needed
        transition_logits = self._align_logits(getattr(self, "transition_logits"), B)  # (B, K, K)
        duration_logits = self._align_logits(getattr(self, "duration_logits"), B)      # (B, K, Dmax)
        init_logits = getattr(self, "init_logits", torch.zeros(K, device=device, dtype=dtype))
        # make init per-batch for simpler code below
        init_logits_b = self._align_logits(init_logits, B)  # (B, K) if init was 1D -> expanded

        Dmax = duration_logits.shape[-1]

        # ---- Precompute cumulative emission sums per sequence (with leading zero) ----
        cumsum_emit_list = []
        for lp in X_valid.log_probs:
            if lp.numel() == 0:
                cumsum_emit_list.append(torch.zeros((1, K), device=device, dtype=dtype))
            else:
                cumsum_emit_list.append(torch.cat(
                    [torch.zeros((1, K), device=device, dtype=dtype), torch.cumsum(lp, dim=0)],
                    dim=0
                ))

        # ---- Containers for DP results ----
        V = torch.full((B, max_T, K), -torch.inf, device=device, dtype=dtype)
        prev_state = torch.full((B, max_T, K), -1, dtype=torch.int64, device=device)
        prev_dur = torch.zeros((B, max_T, K), dtype=torch.int64, device=device)

        # ---- Per-sequence decode (keeps shapes simple and robust) ----
        paths: list[torch.Tensor] = []
        algo = algorithm.lower()
        for b in range(B):
            L = X_valid.lengths[b]
            if L == 0:
                paths.append(torch.empty(0, dtype=torch.long))
                continue

            emit_cumsum = cumsum_emit_list[b]      # [L+1, K]
            trans_b = transition_logits[b]          # [K, K]
            dur_b = duration_logits[b]              # [K, Dmax]
            init_b = init_logits_b[b]               # [K]
            lp_b = X_valid.log_probs[b]             # [L, K]

            Vb = V[b]
            prev_b = prev_state[b]
            dur_b_store = prev_dur[b]

            for t in range(L):
                max_d = min(Dmax, t + 1)
                # scores_by_d: [max_d, K] where axis 0 = duration index (d=1..max_d)
                scores_by_d = torch.full((max_d, K), -torch.inf, device=device, dtype=dtype)
                prev_state_by_d = torch.full((max_d, K), -1, dtype=torch.int64, device=device)

                for di, d in enumerate(range(1, max_d + 1)):
                    start = t - d + 1
                    # emission sum for the segment [start..t] inclusive
                    emit_sum = emit_cumsum[t + 1] - emit_cumsum[start]  # [K]

                    if start - 1 >= 0:
                        prev_scores = Vb[start - 1]                   # [K]
                    else:
                        prev_scores = init_b                          # [K]

                    # prev_scores.unsqueeze(1) + trans_b -> [K_prev, K_next]
                    prev_plus_trans = prev_scores.unsqueeze(1) + trans_b  # [K_prev, K_next]

                    if algo == "viterbi":
                        # max over previous states
                        trans_term, argmax_j = torch.max(prev_plus_trans, dim=0)  # both [K_next]
                    else:
                        # map-like sum-product (log-sum) over previous states
                        trans_term = torch.logsumexp(prev_plus_trans, dim=0)     # [K_next]
                        _, argmax_j = torch.max(prev_plus_trans, dim=0)

                    dur_term = dur_b[:, d - 1]  # [K_next]
                    scores_by_d[di] = trans_term + dur_term + emit_sum
                    prev_state_by_d[di] = argmax_j

                # select best duration (viterbi) or combine (map)
                if algo == "viterbi":
                    best_vals, best_d_idx = torch.max(scores_by_d, dim=0)    # [K]
                else:
                    best_vals = torch.logsumexp(scores_by_d, dim=0)          # [K]
                    _, best_d_idx = torch.max(scores_by_d, dim=0)            # for backtrace heuristic

                Vb[t] = best_vals
                prev_b[t] = prev_state_by_d[best_d_idx, torch.arange(K, device=device)]
                dur_b_store[t] = best_d_idx + 1

            # ---- Backtrace to produce sequence of labels ----
            t = L - 1
            cur_state = int(torch.argmax(Vb[t]).item())
            segments: list[torch.Tensor] = []
            while t >= 0:
                d = int(dur_b_store[t, cur_state].item())
                if d <= 0:
                    break
                start = max(0, t - d + 1)
                segments.insert(0, torch.full((t - start + 1,), cur_state, dtype=torch.long, device=device))
                prev = int(prev_b[t, cur_state].item())
                t = start - 1
                cur_state = prev if prev >= 0 else cur_state

            if segments:
                seq_path = torch.cat(segments, dim=0)
            else:
                seq_path = torch.full((L,), 0, dtype=torch.long, device=device)

            # pad if needed (shouldn't usually happen)
            if seq_path.shape[0] < L:
                pad_len = L - seq_path.shape[0]
                pad_val = seq_path[-1].item() if seq_path.numel() else 0
                seq_path = torch.cat([seq_path, torch.full((pad_len,), pad_val, dtype=torch.long, device=device)])

            paths.append(seq_path.detach().cpu())

        return paths

    def score(
        self,
        X: torch.Tensor,
        lengths: Optional[List[int]] = None,
        by_sample: bool = True,
        theta: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Duration-aware log-likelihood (forward algorithm) for HSMM.
        Fully vectorized across batch; memory-aware cumulative sums.
        """
        obs = self.to_observations(X, lengths, theta)
        B = len(obs.lengths)
        max_T = max(obs.lengths) if B > 0 else 0
        K, Dmax = self.n_states, self.max_duration
        device, dtype = self.device, DTYPE

        if B == 0 or max_T == 0:
            out = torch.zeros(B if by_sample else 1, dtype=dtype, device=device)
            return out.cpu()

        # --- Emission log-probabilities ---
        pdf = self._params.get("emission_pdf")
        if pdf is None or not hasattr(pdf, "log_prob"):
            raise RuntimeError("Emission PDF must implement log_prob().")

        all_seq = torch.cat(obs.sequence, dim=0).to(device=device, dtype=dtype)
        all_log_probs = pdf.log_prob(all_seq.unsqueeze(1))  # (total_T, K)

        # reshape back into padded batch [B, T, K]
        seq_offsets = torch.cat([torch.tensor([0], device=device),
                                 torch.cumsum(torch.tensor(obs.lengths, device=device), dim=0)])
        log_B = torch.full((B, max_T, K), -torch.inf, device=device, dtype=dtype)
        for i in range(B):
            s, e = seq_offsets[i].item(), seq_offsets[i+1].item()
            L = obs.lengths[i]
            log_B[i, :L] = all_log_probs[s:e]

        # --- Log transition/duration/init ---
        log_transition = F.log_softmax(self._transition_logits, dim=-1).to(device, dtype)
        log_duration = F.log_softmax(self._duration_logits, dim=-1).to(device, dtype)
        init_logits = self.init_logits.to(device, dtype)
        if init_logits.ndim == 1:
            init_logits = init_logits.expand(B, -1)

        # --- Forward recursion ---
        V = torch.full((B, max_T, K), -torch.inf, device=device, dtype=dtype)
        V[:, 0, :] = log_B[:, 0, :] + init_logits
        cumsum_B = torch.cat([torch.zeros((B, 1, K), device=device, dtype=dtype),
                              log_B.cumsum(dim=1)], dim=1)

        for t in range(1, max_T):
            max_d = min(Dmax, t + 1)
            start_idx = t - torch.arange(1, max_d + 1, device=device)

            prev_V, emit_sums = [], []
            dur_scores = log_duration[:, :, :max_d].transpose(1, 2)  # (B, max_d, K)

            for d_idx, s in enumerate(start_idx):
                if s < 0:
                    prev_V.append(init_logits)
                    emit_sums.append(cumsum_B[:, t+1, :])
                else:
                    prev_V.append(V[:, s, :])
                    emit_sums.append(cumsum_B[:, t+1, :] - cumsum_B[:, s, :])

            prev_V = torch.stack(prev_V, dim=1)      # (B, max_d, K)
            emit_sums = torch.stack(emit_sums, dim=1)
            log_trans_sum = torch.logsumexp(prev_V.unsqueeze(3) + log_transition.unsqueeze(0), dim=2)
            V[:, t, :] = torch.logsumexp(log_trans_sum + dur_scores + emit_sums, dim=1)

        # --- Mask and gather per-sequence likelihoods ---
        end_idx = torch.tensor(obs.lengths, device=device) - 1
        seq_ll = torch.logsumexp(V[torch.arange(B), end_idx], dim=-1)

        return seq_ll.cpu() if by_sample else seq_ll.sum().cpu()

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

        # --- Compute per-sequence log-likelihood safely ---
        try:
            log_likelihood = self.score(X, lengths=lengths, by_sample=by_sample).to(device=self.device, dtype=DTYPE)
        except Exception as e:
            raise RuntimeError(f"Failed to compute log-likelihood: {e}")

        # --- Count total observations (time steps, not features) ---
        if lengths is not None:
            n_obs = max(int(sum(lengths)), 1)
        elif X.ndim == 3:  # (B, T, F)
            n_obs = max(int(X.shape[0] * X.shape[1]), 1)
        else:  # (T, F) or flat
            n_obs = max(int(X.shape[0]), 1)

        # --- Degrees of freedom check ---
        dof = getattr(self, "dof", None)
        if dof is None:
            raise AttributeError(
                "Model degrees of freedom ('dof') not defined. "
                "Please set 'self.dof' during initialization."
            )

        # --- Compute IC ---
        ic_value = constraints.compute_information_criteria(
            n_obs=n_obs,
            log_likelihood=log_likelihood,
            dof=dof,
            criterion=criterion
        )

        # --- Ensure tensor, handle NaN/Inf ---
        if not isinstance(ic_value, torch.Tensor):
            ic_value = torch.tensor(ic_value, dtype=DTYPE, device=self.device)
        ic_value = ic_value.nan_to_num(nan=float('inf'), posinf=float('inf'), neginf=float('inf'))

        # --- Ensure per-sample shape if requested ---
        if by_sample and ic_value.ndim == 0:
            ic_value = ic_value.unsqueeze(0)

        return ic_value.detach().cpu()

