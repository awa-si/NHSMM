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
        alpha: float = None,
        min_covar: Optional[float] = None,
        seed: Optional[int] = None,
        transition_type: Any = None,
        context_dim: Optional[int] = None,
        device: Optional[torch.device] = None,
        emission_type: str = "gaussian",
        modulate_var: bool = False,
    ):
        super().__init__()

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transition_type = transition_type or constraints.Transitions.SEMI
        self._context: Optional[torch.Tensor] = None
        self.emission_type = emission_type.lower()
        self.encoder: Optional[nn.Module] = None
        self._seed_gen = SeedGenerator(seed)
        self.min_covar = min_covar or 1e-3
        self._params: Dict[str, Any] = {}
        self.max_duration = max_duration
        self.modulate_var = modulate_var
        self.context_dim = context_dim
        self.n_features = n_features
        self.n_states = n_states
        self.alpha = alpha

        self._init_modules()
        self._init_buffers()
        self._params['emission_pdf'] = self.sample_emission_pdf(None)

    def _init_modules(self):
        device, dtype = self.device, DTYPE

        self.emission_module = Emission(
            n_states=self.n_states,
            n_features=self.n_features,
            min_covar=self.min_covar,
            context_dim=self.context_dim,
            device=device,
            emission_type=self.emission_type,
            modulate_var=self.modulate_var,
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

    def _init_buffers(self):
        device = next(self.buffers(), torch.zeros(1, dtype=DTYPE)).device

        def sample_probs(shape):
            probs = constraints.sample_probs(self.alpha, shape)  # no seed keyword
            logits = torch.log(probs.clamp_min(1e-12)).to(device=device, dtype=DTYPE)
            return logits, probs.shape

        def sample_transitions(n_states):
            probs = constraints.sample_transition(self.alpha, n_states, self.transition_type)  # no seed
            logits = torch.log(probs.clamp_min(1e-12)).to(device=device, dtype=DTYPE)
            return logits, probs.shape

        init_logits, init_shape = sample_probs((self.n_states,))
        transition_logits, transition_shape = sample_transitions(self.n_states)
        duration_logits, duration_shape = sample_probs((self.n_states, self.max_duration))

        self.register_buffer("_init_logits", init_logits)
        self.register_buffer("_transition_logits", transition_logits)
        self.register_buffer("_duration_logits", duration_logits)

        self._init_shape = init_shape
        self._transition_shape = transition_shape
        self._duration_shape = duration_shape

        # Super-states (optional)
        n_super_states = getattr(self, "n_super_states", 0)
        if n_super_states > 1:
            super_init_logits, super_init_shape = sample_probs(n_super_states)
            super_transition_logits, super_transition_shape = sample_transitions(n_super_states)
            self.register_buffer("_super_init_logits", super_init_logits)
            self.register_buffer("_super_transition_logits", super_transition_logits)
            self._super_init_shape = super_init_shape
            self._super_transition_shape = super_transition_shape

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
        
        norm_val = logits.logsumexp(dim=0)
        zero = torch.tensor(0., dtype=DTYPE, device=logits.device)
        if not torch.allclose(norm_val, zero, atol=1e-6):
            raise ValueError(f"init_logits logits must normalize (logsumexp==0); got {norm_val.item():.3e}")

        self._init_logits.copy_(logits.to(dtype=DTYPE))

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
            raise ValueError(f"transition_logits must have shape ({self.n_states},{self.n_states})")
        
        # Relax row normalization check: allow tiny deviations
        row_norm = logits.logsumexp(dim=1)
        if not torch.allclose(row_norm, torch.zeros_like(row_norm), atol=1e-4):
            # Instead of raising, normalize rows automatically
            logits = logits - row_norm.unsqueeze(1)
        
        # Relax transition constraints: only warn if violated
        if not constraints.is_valid_transition(logits.exp(), self.transition_type):
            print("[transition_logits] Warning: logits do not fully satisfy transition constraints, applying anyway.")
            # Optionally: project small negative probabilities to zero
            probs = logits.exp().clamp_min(1e-12)
            logits = torch.log(probs)
        
        self._transition_logits.copy_(logits.to(dtype=DTYPE))

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
        
        row_norm = logits.logsumexp(dim=1)
        zeros = torch.zeros_like(row_norm)
        if not torch.allclose(row_norm, zeros, atol=1e-6):
            raise ValueError(f"Rows of duration_logits logits must normalize (logsumexp==0); got {row_norm.tolist()}")

        self._duration_logits.copy_(logits.to(dtype=DTYPE))

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
    def initialize_emissions(self, X: torch.Tensor, method: str = "moment") -> None:
        """
        Abstract method to initialize emission parameters from raw data.
        """
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

    def _align_logits(self, logits: torch.Tensor, B: int) -> torch.Tensor:
        """
        Expand or broadcast logits to match batch size B safely.
        Supports 1D (init), 2D (transition/duration), 3D (batched).
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
                raise ValueError(f"Cannot align logits with batch size {B}, shape {logits.shape}")

        elif logits.ndim == 2:  # [K,K] or [K,Dmax]
            return logits.unsqueeze(0).expand(B, -1, -1)

        elif logits.ndim == 1:  # [K] (initial state logits)
            return logits.unsqueeze(0).expand(B, -1)

        else:
            raise ValueError(f"Unsupported logits shape {logits.shape}")

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
            encoder=encoder,
            batch_first=batch_first,
            pool=pool,
            device=self.device,
            n_heads=n_heads,
            max_seq_len=max_seq_len
        )
        return self.encoder

    def encode_observations(self, X: torch.Tensor, pool: Optional[str] = None, store: bool = True) -> Optional[torch.Tensor]:
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

    def to_observations(
        self,
        X: torch.Tensor,
        lengths: Optional[List[int]] = None,
        theta: Optional[torch.Tensor] = None
    ) -> utils.Observations:
        """
        Convert raw input X (and optional context theta) into a structured Observations object.
        Returns log-probabilities as a tensor, never a Distribution.
        """
        device = next(self.parameters(), torch.tensor(0.0, device='cpu')).device
        X = torch.as_tensor(X, dtype=DTYPE, device=device) if not torch.is_tensor(X) else X.to(dtype=DTYPE, device=device)

        # Validate and clamp input
        X_valid = self.check_constraints(X)

        # Use global context if theta is None
        if theta is None and getattr(self, "_context", None) is not None:
            theta = self._context

        if theta is not None:
            theta = theta.to(dtype=DTYPE, device=device)
            # Broadcast batch context to sequence dimension if needed
            if X_valid.ndim == 3 and theta.ndim == 2:
                theta = theta[:, None, :].expand(-1, X_valid.shape[1], -1)
            elif X_valid.ndim == 2 and theta.ndim == 3 and theta.shape[1] == 1:
                theta = theta.squeeze(1)

        # Handle sequence lengths
        total_len = X_valid.shape[0] if X_valid.ndim == 2 else X_valid.shape[1]
        if lengths is None:
            seq_lengths = [total_len]
        else:
            if sum(lengths) != total_len:
                raise ValueError(f"Sum of lengths ({sum(lengths)}) does not match total samples ({total_len}).")
            seq_lengths = lengths

        # Flatten batch for splitting
        X_flat = X_valid.reshape(-1, X_valid.shape[-1]) if X_valid.ndim == 3 else X_valid
        sequences = list(torch.split(X_flat, seq_lengths))

        # Prepare theta per sequence
        theta_list = []
        for i, seq_len in enumerate(seq_lengths):
            if theta is None:
                theta_list.append(None)
            else:
                if theta.ndim == 3:
                    start_idx = sum(seq_lengths[:i])
                    theta_list.append(theta.reshape(-1, theta.shape[-1])[start_idx:start_idx+seq_len])
                else:
                    theta_list.append(theta[i] if theta.shape[0] > 1 else theta)

        # Compute log-probabilities
        log_probs_list = []
        for seq, seq_theta in zip(sequences, theta_list):
            logp = self._contextual_emission_pdf(seq, seq_theta)
            if logp.ndim == 1:
                logp = logp.unsqueeze(-1)
            log_probs_list.append(logp.to(dtype=DTYPE, device=device))

        return utils.Observations(
            sequence=sequences,
            log_probs=log_probs_list,
            lengths=seq_lengths
        )

    def _contextual_emission_pdf(
        self,
        X: torch.Tensor,
        theta: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute context-modulated emission log-probabilities as a tensor [B, T, K].
        Works with singleton emission PDFs or Categorical distributions returned by _estimate_model_params.
        Never returns a Distribution; always returns log-prob tensor for batching.
        """
        # Ensure batch dimension
        if X.ndim == 2:  # [T, F] → [1, T, F]
            X = X.unsqueeze(0)
        elif X.ndim != 3:
            raise ValueError(f"Unsupported input shape {X.shape}")

        B, T, _ = X.shape

        # Combine context
        theta_batch = self.combine_context(theta)
        if theta_batch is not None:
            if theta_batch.ndim == 2:  # [B, H] → [B, T, H]
                theta_batch = theta_batch.unsqueeze(1).expand(-1, T, -1)

        # Compute log-probabilities
        logp_list = []
        for b in range(B):
            logp_seq = self.emission_module.log_prob(
                X[b],
                context=None if theta_batch is None else theta_batch[b]
            )
            if logp_seq.ndim == 1:
                logp_seq = logp_seq.unsqueeze(-1)  # [T, K]
            logp_list.append(logp_seq)

        logp_tensor = torch.stack(logp_list, dim=0)  # [B, T, K]

        # Clamp & normalize
        logp_tensor = torch.nan_to_num(logp_tensor, nan=-1e8, posinf=-1e8, neginf=-1e8)
        logp_tensor = torch.clamp(logp_tensor, -1e6, 1e6)
        logp_tensor = torch.nn.functional.log_softmax(logp_tensor, dim=-1)

        return logp_tensor  # [B, T, K]

    def _contextual_duration_pdf(
        self,
        theta: Optional[ContextualVariables] = None
    ) -> torch.Tensor:
        """
        Compute context-modulated duration log-probabilities.

        Returns:
            log_duration: [B, K, max_duration]
        """
        base_duration = self._duration_logits.unsqueeze(0)  # [1, K, max_duration]
        theta_batch = self.combine_context(theta)
        B = theta_batch.shape[0] if theta_batch is not None else 1

        log_duration = self.duration_module._apply_context(
            base_duration.expand(B, *base_duration.shape[1:]),
            theta_batch
        )
        log_duration = torch.nan_to_num(log_duration, nan=-1e8, posinf=-1e8, neginf=-1e8)
        return log_duration - torch.logsumexp(log_duration, dim=2, keepdim=True)

    def _contextual_transition_matrix(
        self,
        theta: Optional[ContextualVariables] = None
    ) -> torch.Tensor:
        """
        Compute context-modulated transition log-probabilities.

        Returns:
            log_transition: [B, K, K]
        """
        base_transition = self._transition_logits.unsqueeze(0)  # [1, K, K]
        theta_batch = self.combine_context(theta)
        B = theta_batch.shape[0] if theta_batch is not None else 1

        log_transition = self.transition_module._apply_context(
            base_transition.expand(B, *base_transition.shape[1:]),
            theta_batch
        )

        # Apply transition constraints if defined
        if hasattr(self, "transition_type"):
            mask = constraints.mask_invalid_transitions(self.n_states, self.transition_type).to(log_transition.device)
            log_transition = log_transition.masked_fill(~mask.unsqueeze(0), -1e8)

        log_transition = torch.nan_to_num(log_transition, nan=-1e8, posinf=-1e8, neginf=-1e8)
        return log_transition - torch.logsumexp(log_transition, dim=2, keepdim=True)

    def to_observations(
        self,
        X: torch.Tensor,
        lengths: Optional[List[int]] = None,
        theta: Optional[torch.Tensor] = None
    ) -> utils.Observations:
        """
        Convert raw input X (and optional context theta) into a structured Observations object.
        Supports batched or single-sequence inputs, and computes context-modulated log-probabilities.
        Handles emission PDFs that are either tensors or torch.distributions.Distribution.

        Args:
            X: Observations tensor of shape [T, F] or [B, T, F].
            lengths: Optional list of sequence lengths.
            theta: Optional context tensor of shape [B, H] or [B, T, H].

        Returns:
            utils.Observations(sequence, log_probs, lengths)
        """
        device = next(self.parameters(), torch.tensor(0.0, device='cpu')).device
        X = torch.as_tensor(X, dtype=DTYPE, device=device) if not torch.is_tensor(X) else X.to(dtype=DTYPE, device=device)

        # --- Validate input ---
        X_valid = self.check_constraints(X)

        # --- Align context ---
        if theta is None and hasattr(self, "_context") and self._context is not None:
            theta = self._context

        if theta is not None:
            theta = theta.to(dtype=DTYPE, device=device)
            # Broadcast batch context to sequence dimension if needed
            if X_valid.ndim == 3 and theta.ndim == 2:
                theta = theta[:, None, :].expand(-1, X_valid.shape[1], -1)
            elif X_valid.ndim == 2 and theta.ndim == 3 and theta.shape[1] == 1:
                theta = theta.expand(X_valid.shape[0], -1)

        # --- Sequence length handling ---
        total_len = X_valid.shape[0] if X_valid.ndim == 2 else X_valid.shape[1]
        if lengths is None:
            seq_lengths = [total_len]
        else:
            if not isinstance(lengths, (list, tuple)) or not all(isinstance(l, int) for l in lengths):
                raise TypeError("`lengths` must be a list of integers.")
            if sum(lengths) != total_len:
                raise ValueError(f"Sum of lengths ({sum(lengths)}) does not match total samples ({total_len}).")
            seq_lengths = lengths

        # --- Flatten batch for splitting sequences ---
        X_flat = X_valid.reshape(-1, X_valid.shape[-1]) if X_valid.ndim == 3 else X_valid
        sequences = list(torch.split(X_flat, seq_lengths))

        # --- Prepare theta per sequence ---
        theta_list = []
        for i, seq_len in enumerate(seq_lengths):
            if theta is None:
                theta_list.append(None)
            else:
                if theta.ndim == 3:  # [B, T, H]
                    start_idx = sum(seq_lengths[:i])
                    theta_list.append(theta.reshape(-1, theta.shape[-1])[start_idx:start_idx+seq_len])
                else:  # [B, H] or single [H]
                    context_row = theta[i] if theta.shape[0] > 1 else theta
                    theta_list.append(context_row.expand(seq_len, -1))

        # --- Compute log-probabilities ---
        log_probs_list = []
        for seq, seq_theta in zip(sequences, theta_list):
            pdf_or_tensor = self._contextual_emission_pdf(seq, seq_theta)

            if isinstance(pdf_or_tensor, torch.distributions.Distribution):
                # Compute log-probabilities for the sequence
                log_probs = pdf_or_tensor.log_prob(seq.unsqueeze(-2))
                if log_probs.ndim > 2:
                    log_probs = log_probs.sum(dim=tuple(range(2, log_probs.ndim)))
            elif torch.is_tensor(pdf_or_tensor):
                log_probs = pdf_or_tensor
                if log_probs.ndim == 1:
                    log_probs = log_probs.unsqueeze(-1)
            else:
                raise TypeError(f"_contextual_emission_pdf must return a tensor or Distribution, got {type(pdf_or_tensor)}")

            # Ensure shape consistency [T, K]
            if log_probs.ndim == 0:
                log_probs = log_probs.unsqueeze(0)

            log_probs_list.append(log_probs.to(dtype=DTYPE, device=device))

        return utils.Observations(
            sequence=sequences,
            log_probs=log_probs_list,
            lengths=seq_lengths
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

        # -------------------------------
        # Initial state distribution π [K]
        # -------------------------------
        init_terms = [g[0] for g in gamma_list if g is not None and g.numel() > 0]
        if init_terms:
            pi_counts = torch.stack(init_terms, dim=0).sum(dim=0).clamp_min(EPS)
            new_init = constraints.log_normalize(torch.log(pi_counts), dim=0)
        else:
            new_init = self.init_logits.detach().clone().to(device=device, dtype=DTYPE)

        # -------------------------------
        # Transition matrix A [K, K]
        # -------------------------------
        xi_valid = [x for x in xi_list if x is not None and x.numel() > 0]
        if xi_valid:
            xi_cat = torch.cat(xi_valid, dim=0).to(device=device, dtype=DTYPE)
            trans_counts = xi_cat.sum(dim=0).clamp_min(EPS)
            new_transition = constraints.log_normalize(torch.log(trans_counts), dim=1)
        else:
            new_transition = self.transition_logits.detach().clone().to(device=device, dtype=DTYPE)

        # -------------------------------
        # Duration distributions P(d | z) [K, Dmax]
        # -------------------------------
        eta_valid = [e for e in eta_list if e is not None and e.numel() > 0]
        if eta_valid:
            eta_cat = torch.cat(eta_valid, dim=0).to(device=device, dtype=DTYPE)
            dur_counts = eta_cat.sum(dim=0).clamp_min(EPS)
            new_duration = constraints.log_normalize(torch.log(dur_counts), dim=1)
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
            all_gamma = torch.cat(
                [g for g in gamma_list if g is not None and g.numel() > 0], dim=0
            ).to(device=device, dtype=DTYPE)

            # --- Align contextual variable θ ---
            theta_full = None
            if theta is not None:
                theta = theta.to(device=device, dtype=DTYPE)
                if theta.ndim == 1:
                    theta_full = theta.unsqueeze(0).expand(all_X.shape[0], -1)
                elif theta.ndim == 2:
                    if theta.shape[0] != all_X.shape[0]:
                        repeats = all_X.shape[0] // theta.shape[0]
                        if all_X.shape[0] % theta.shape[0] != 0:
                            raise ValueError(
                                f"Theta shape {tuple(theta.shape)} cannot align with sequence length {all_X.shape[0]}"
                            )
                        theta_full = theta.repeat_interleave(repeats, dim=0)
                    else:
                        theta_full = theta
                elif theta.ndim == 3:
                    theta_full = theta.reshape(-1, theta.shape[-1])
                    if theta_full.shape[0] != all_X.shape[0]:
                        raise ValueError(
                            f"Context θ has shape {tuple(theta.shape)} but expected total length {all_X.shape[0]}"
                        )
                else:
                    raise ValueError(f"Unsupported θ shape: {tuple(theta.shape)}")

            # --- Compute emission PDF ---
            if hasattr(self, "_estimate_emission_pdf"):
                pdf_or_tensor = self._estimate_emission_pdf(all_X, all_gamma, theta_full)
            elif hasattr(self, "_contextual_emission_pdf"):
                pdf_or_tensor = self._contextual_emission_pdf(all_X, theta_full)
            else:
                raise RuntimeError(
                    "No emission estimator available — model missing _estimate_emission_pdf or _contextual_emission_pdf."
                )

            # Wrap tensor logits as Categorical if needed
            if isinstance(pdf_or_tensor, torch.distributions.Distribution):
                new_pdf = pdf_or_tensor
            elif torch.is_tensor(pdf_or_tensor):
                t = pdf_or_tensor.to(device=device, dtype=DTYPE)
                if t.ndim == 2 and t.shape[1] == K:
                    new_pdf = Categorical(logits=t)
                else:
                    raise TypeError(
                        "Estimator returned a tensor but its shape is not (N, K). "
                        "Return a Distribution instead for non-categorical emissions."
                    )
            else:
                raise TypeError(f"Emission estimator returned unsupported type {type(pdf_or_tensor)}")

        # -------------------------------
        # Return singleton-shaped logits
        # -------------------------------
        return {
            "init_logits": new_init.to(device=device, dtype=DTYPE),          # [K]
            "transition_logits": new_transition.to(device=device, dtype=DTYPE),  # [K, K]
            "duration_logits": new_duration.to(device=device, dtype=DTYPE),      # [K, Dmax]
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


    # HSMM precessing
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
            seq_batch, lengths = X.to_batch(return_mask=False)[1].to(device=device, dtype=dtype), X.lengths

        B = len(lengths)
        alpha_list: List[torch.Tensor] = []

        for b, seq_len in enumerate(lengths):
            if seq_len == 0:
                alpha_list.append(torch.full((0, K, Dmax), neg_inf, device=device, dtype=dtype))
                continue

            # --- Sequence emissions ---
            log_probs = seq_batch[b][:seq_len]  # [T, K]

            # If context θ is provided, modulate log_probs
            if theta is not None:
                theta_seq = theta if theta.shape[0] == seq_len else theta.repeat_interleave(seq_len // theta.shape[0], dim=0)
                log_probs = self._contextual_emission_pdf(log_probs, theta_seq) if hasattr(self, "_contextual_emission_pdf") else log_probs

            # --- Cumulative sum over durations for emission contributions ---
            cumsum_emit = torch.vstack((torch.zeros((1, K), device=device, dtype=dtype), torch.cumsum(log_probs, dim=0)))  # [T+1, K]

            # --- Initialize log-alpha ---
            log_alpha = torch.full((seq_len, K, Dmax), neg_inf, device=device, dtype=dtype)

            max_d0 = min(Dmax, seq_len)
            durations0 = torch.arange(1, max_d0 + 1, device=device)
            emit_sums0 = (cumsum_emit[durations0] - cumsum_emit[0]).T  # [K, max_d0]
            log_alpha[0, :, :max_d0] = init_logits.unsqueeze(1) + duration_logits[:, :max_d0] + emit_sums0

            # --- Recursion ---
            for t in range(1, seq_len):
                max_dt = min(Dmax, t + 1)
                durations = torch.arange(1, max_dt + 1, device=device)          # [max_dt]
                starts = t - durations + 1                                       # [max_dt]

                # Emission sums
                emit_sums_t = (cumsum_emit[t + 1].unsqueeze(0) - cumsum_emit[starts]).T  # [K, max_dt]

                # Previous alpha contributions
                idx = torch.clamp(starts - 1, min=0)
                prev_alpha_first = log_alpha[idx, :, 0]                       # [max_dt, K]

                # Handle durations starting at t=0
                mask = (starts == 0).unsqueeze(1)                             # [max_dt, 1]
                prev_alpha_first = torch.where(mask, init_logits.unsqueeze(0), prev_alpha_first)

                # Combine with transition matrix
                prev_alpha_exp = prev_alpha_first.unsqueeze(2) + transition_logits.unsqueeze(0)  # [max_dt, K, K]
                log_alpha_t = torch.logsumexp(prev_alpha_exp, dim=1).T        # [K, max_dt]

                # Add duration logits and emission sums
                log_alpha[t, :, :max_dt] = log_alpha_t + duration_logits[:, :max_dt] + emit_sums_t

            alpha_list.append(log_alpha)

        return alpha_list

    def _backward(self, X: utils.Observations, theta: Optional[torch.Tensor] = None) -> list[torch.Tensor]:
        """
        Vectorized backward pass for HSMM sequences (batched or singleton),
        context-aware via `theta`.
        Returns a list of log-beta tensors: shape (T_i, K, Dmax) per sequence.
        """
        K, Dmax = self.n_states, self.max_duration
        neg_inf = -torch.inf

        # Prepare model parameters
        transition_logits = self.transition_logits.to(self.device, dtype=DTYPE)  # [K, K]
        duration_logits = self.duration_logits.to(self.device, dtype=DTYPE)      # [K, Dmax]

        # Compute log_probs with optional context
        if theta is not None and hasattr(self, "_contextual_emission_pdf"):
            log_probs_list = [
                self._contextual_emission_pdf(seq, theta.to(self.device, dtype=DTYPE))
                if seq.numel() > 0 else torch.zeros((0, K), dtype=DTYPE, device=self.device)
                for seq in X.sequence
            ]
        else:
            log_probs_list = [seq.to(self.device, dtype=DTYPE) for seq in X.log_probs]

        beta_list = []

        for seq_idx, (seq_probs, seq_len) in enumerate(zip(log_probs_list, X.lengths)):
            if seq_len == 0:
                beta_list.append(torch.full((0, K, Dmax), neg_inf, device=self.device, dtype=DTYPE))
                continue

            seq_probs = seq_probs[:seq_len]  # [T, K]
            log_beta = torch.full((seq_len, K, Dmax), neg_inf, device=self.device, dtype=DTYPE)

            # Cumulative sum for efficient duration sums
            cumsum_emit = torch.vstack((torch.zeros((1, K), dtype=DTYPE, device=self.device),
                                        torch.cumsum(seq_probs, dim=0)))  # [T+1, K]

            # Last time step: beta[T-1,:,0] = 0
            log_beta[-1, :, 0] = 0.0

            # Backward recursion
            for t in reversed(range(seq_len - 1)):
                max_dt = min(Dmax, seq_len - t)
                durations = torch.arange(1, max_dt + 1, device=self.device)
                ends = t + durations  # [max_dt]

                # Compute emission sums for durations
                emit_start = cumsum_emit[t].unsqueeze(0).expand(max_dt, -1)       # [max_dt, K]
                emit_end = cumsum_emit[ends, :]                                   # [max_dt, K]
                emit_sums = emit_end - emit_start                                  # [max_dt, K]

                # Beta contribution from next states at duration 1
                beta_next = log_beta[ends - 1, :, 0]                               # [max_dt, K]

                # Duration scores
                dur_scores = duration_logits[:, :max_dt].T                         # [max_dt, K]

                combined = emit_sums + beta_next + dur_scores                      # [max_dt, K]
                log_beta[t, :, 0] = torch.logsumexp(combined.T, dim=1)             # [K]

                # Handle beta[:,1:Dmax] shifts
                if max_dt > 1:
                    shift_len = max_dt - 1
                    log_beta[t, :, 1:shift_len + 1] = log_beta[t + 1, :, :shift_len] + seq_probs[t + 1].unsqueeze(-1)

            beta_list.append(log_beta)

        return beta_list

    def _compute_posteriors(self, X: utils.Observations, theta: Optional[torch.Tensor] = None
                            ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Vectorized computation of HSMM posteriors: gamma, xi, eta.
        Supports context-modulated emissions (via theta) and per-sequence lengths.
        Returns lists of tensors per sequence.
        """
        K, Dmax = self.n_states, self.max_duration
        gamma_vec: List[torch.Tensor] = []
        xi_vec: List[torch.Tensor] = []
        eta_vec: List[torch.Tensor] = []

        init_logits = self.init_logits.to(device=self.device, dtype=DTYPE)
        transition_logits = self.transition_logits.to(device=self.device, dtype=DTYPE)

        alpha_list = self._forward(X)
        beta_list = self._backward(X, theta=theta)

        for seq_idx, (seq_probs, seq_len, alpha, beta) in enumerate(zip(X.log_probs, X.lengths, alpha_list, beta_list)):
            if seq_len == 0:
                gamma_vec.append(torch.zeros((0, K), dtype=DTYPE, device=self.device))
                eta_vec.append(torch.zeros((0, K, Dmax), dtype=DTYPE, device=self.device))
                xi_vec.append(torch.zeros((0, K, K), dtype=DTYPE, device=self.device))
                continue

            alpha = alpha.to(self.device, dtype=DTYPE)
            beta = beta.to(self.device, dtype=DTYPE)
            seq_probs = seq_probs.to(self.device, dtype=DTYPE)

            # --- gamma ---
            log_gamma = torch.logsumexp(alpha + beta, dim=2)  # [T, K]
            log_gamma -= torch.logsumexp(log_gamma, dim=1, keepdim=True)
            gamma = torch.clamp(log_gamma.exp(), min=EPS)
            gamma /= gamma.sum(dim=1, keepdim=True).clamp_min(EPS)
            gamma_vec.append(gamma)

            # --- eta ---
            log_eta = alpha + beta  # [T, K, Dmax]
            log_eta_flat = log_eta.view(seq_len, -1)
            log_eta_flat -= torch.logsumexp(log_eta_flat, dim=1, keepdim=True)
            eta = torch.clamp(log_eta_flat.exp().view(seq_len, K, Dmax), min=EPS)
            eta /= eta.view(seq_len, -1).sum(dim=1, keepdim=True).view(seq_len, 1, 1).clamp_min(EPS)
            eta_vec.append(eta)

            # --- xi ---
            if seq_len > 1:
                t_idx = torch.arange(1, seq_len, device=self.device)            # t=1..T-1
                d_idx = torch.arange(1, Dmax + 1, device=self.device).view(1, -1)  # durations [1..Dmax]
                start_idx = t_idx.view(-1, 1) - d_idx + 1                        # [T-1, Dmax]
                mask = start_idx >= 0
                start_idx_clamped = start_idx.clamp(min=0)

                # prev_alpha_first[t, d, k] = alpha[start_idx, k, 0] if valid else init_logits
                prev_alpha_first = alpha[start_idx_clamped, :, 0]                # [T-1, Dmax, K]
                prev_alpha_first = torch.where(mask.unsqueeze(2), prev_alpha_first, init_logits.unsqueeze(0).unsqueeze(1))

                alpha_sum = torch.logsumexp(prev_alpha_first, dim=1)             # [T-1, K]
                beta_next = torch.logsumexp(beta[1:, :, :Dmax], dim=2)           # [T-1, K]

                log_xi = alpha_sum.unsqueeze(2) + transition_logits.unsqueeze(0) + beta_next.unsqueeze(1)  # [T-1, K, K]
                log_xi_flat = log_xi.view(seq_len - 1, -1)
                xi_seq = torch.softmax(log_xi_flat, dim=1).view_as(log_xi)
                xi_seq = xi_seq.clamp_min(EPS)
                xi_seq /= xi_seq.sum(dim=(1, 2), keepdim=True).clamp_min(EPS)
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
        Fully vectorized HSMM Viterbi decoder (batch x state x duration).
        Supports context (theta) and duration weighting.
        Returns a list of state sequences per batch.
        """
        K, Dmax = self.n_states, self.max_duration
        B = len(X.log_probs)
        lengths = X.lengths
        max_len = max(lengths) if lengths else 0

        if max_len == 0:
            return [torch.empty(0, dtype=torch.int64, device=self.device) for _ in range(B)]

        # --- Model parameters ---
        init_logits = self.init_logits.to(self.device, dtype=DTYPE)
        transition_logits = self.transition_logits.to(self.device, dtype=DTYPE)
        duration_logits = self.duration_logits.to(self.device, dtype=DTYPE)
        neg_inf = -torch.inf

        dur_indices = torch.arange(1, Dmax + 1, device=self.device)
        if duration_weight > 0.0:
            dur_mean = (torch.softmax(duration_logits, dim=1) * dur_indices).sum(dim=1)
            dur_penalty = -((dur_indices.unsqueeze(0) - dur_mean.unsqueeze(1)) ** 2) / (2 * Dmax)
            dur_lp = (1 - duration_weight) * duration_logits + duration_weight * dur_penalty
        else:
            dur_lp = duration_logits  # [K, Dmax]

        # --- Emission cumulative sums ---
        if theta is not None and hasattr(self, "_contextual_emission_pdf"):
            # Flatten all sequences for context-aware emission
            all_seq = torch.cat([seq for seq in X.sequence], dim=0)
            emission = self._contextual_emission_pdf(all_seq, theta)  # [total_T, K]
            cumsum_emit_list = []
            offset = 0
            for L in lengths:
                seq_emit = emission[offset:offset + L, :]
                offset += L
                cumsum_emit_list.append(torch.cat([torch.zeros((1, K), device=self.device, dtype=DTYPE),
                                                   torch.cumsum(seq_emit, dim=0)], dim=0))
        else:
            cumsum_emit_list = [torch.cat([torch.zeros((1, K), device=self.device, dtype=DTYPE),
                                           torch.cumsum(seq.to(self.device, dtype=DTYPE), dim=0)], dim=0)
                                for seq in X.log_probs]

        # --- Initialize DP tensors ---
        V = torch.full((B, max_len, K), neg_inf, device=self.device, dtype=DTYPE)
        best_prev = torch.full((B, max_len, K), -1, dtype=torch.int64, device=self.device)
        best_dur = torch.zeros((B, max_len, K), dtype=torch.int64, device=self.device)

        # --- Vectorized DP over batch and duration ---
        dur_range = torch.arange(1, Dmax + 1, device=self.device)  # [Dmax]
        for t in range(max_len):
            active_mask = torch.tensor([t < L for L in lengths], device=self.device)
            if not active_mask.any():
                continue
            active_idx = active_mask.nonzero(as_tuple=True)[0]
            B_active = len(active_idx)

            max_d = min(Dmax, t + 1)
            durations = dur_range[:max_d]  # [max_d]

            # --- Prepare start indices for all active sequences ---
            # start_idx[b, d] = t - durations[d] + 1
            start_idx = t - durations.view(-1, 1) + torch.zeros(B_active, dtype=torch.long, device=self.device).view(1, -1)  # [max_d, B_active]
            start_idx = start_idx.clamp(min=0)

            # --- Emission sums ---
            emit_sums = torch.stack([
                (cumsum_emit_list[b][t + 1] - cumsum_emit_list[b][start_idx[:, i]]).T
                for i, b in enumerate(active_idx)
            ])  # [B_active, K, max_d]

            dur_scores = dur_lp[:, :max_d].unsqueeze(0).expand(B_active, -1, -1)  # [B_active, K, max_d]

            if t == 0:
                scores = init_logits.unsqueeze(0).unsqueeze(2) + dur_scores + emit_sums  # [B_active, K, max_d]
                prev_idx = torch.full_like(scores, -1, dtype=torch.int64)
            else:
                # Previous DP values
                prev_V = torch.stack([V[b, start_idx[:, i], :] for i, b in enumerate(active_idx)])  # [B_active, max_d, K]
                mask_start0 = (durations.view(-1, 1) == t + 1)
                prev_V[mask_start0.expand(-1, K)] = init_logits

                # Transition: expand and add
                scores_plus_trans = prev_V.unsqueeze(2) + transition_logits.unsqueeze(0)  # [B_active, max_d, K_prev, K]
                scores_max, argmax_prev = torch.max(scores_plus_trans, dim=1)  # [B_active, K_prev, K]
                scores = (scores_max + dur_scores + emit_sums)  # [B_active, K, max_d]
                prev_idx = argmax_prev  # [B_active, K, max_d]

            # --- Pick best duration ---
            best_score, best_d_idx = torch.max(scores, dim=2)  # [B_active, K]
            for i, b in enumerate(active_idx):
                V[b, t] = best_score[i]
                best_dur[b, t] = durations[best_d_idx[i]]
                best_prev[b, t] = prev_idx[i, torch.arange(K, device=self.device), best_d_idx[i]]

        # --- Backtrace ---
        paths: list[torch.Tensor] = []
        for b, L in enumerate(lengths):
            if L == 0:
                paths.append(torch.empty(0, dtype=torch.int64, device=self.device))
                continue

            t = L - 1
            cur_state = torch.argmax(V[b, t]).item()
            segments = []

            while t >= 0:
                d = best_dur[b, t, cur_state].item()
                if d <= 0:
                    break
                start = max(0, t - d + 1)
                segments.append((start, t, cur_state))
                prev_state = best_prev[b, t, cur_state].item()
                t = start - 1
                cur_state = prev_state if prev_state >= 0 else cur_state

            segments.reverse()
            seq_path = torch.cat([
                torch.full((end - start + 1,), st, dtype=torch.int64, device=self.device)
                for start, end, st in segments
            ])
            if seq_path.shape[0] < L:
                seq_path = torch.cat([seq_path, torch.full((L - seq_path.shape[0],), seq_path[-1],
                                                           dtype=torch.int64, device=self.device)])
            paths.append(seq_path)

        return paths

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

        # --- Optional emission initialization ---
        if sample_D_from_X:
            self._params['emission_pdf'] = self.sample_emission_pdf(X)

        # --- Encode observations if needed ---
        if theta is None and getattr(self, "encoder", None):
            theta = self.encode_observations(X)

        X_valid = self.to_observations(X, lengths, theta)

        # --- Align theta strictly with sequences ---
        if theta is not None:
            total_len = sum(X_valid.lengths)
            if theta.shape[0] != total_len:
                if theta.shape[0] == len(X_valid.sequence):
                    theta = torch.cat([theta[i].expand(l, -1) for i, l in enumerate(X_valid.lengths)], dim=0)
                elif total_len % theta.shape[0] == 0:
                    theta = theta.repeat(total_len // theta.shape[0], 1)
                else:
                    raise ValueError(f"Theta shape {theta.shape} cannot align with total length {total_len}")
            valid_theta = theta
        else:
            valid_theta = None

        # --- Convergence tracker ---
        self.conv = ConvergenceHandler(
            tol=tol, max_iter=max_iter, n_init=n_init, post_conv_iter=post_conv_iter, verbose=verbose
        )

        best_score = -float("inf")
        best_state = None

        # --- Flatten sequences and offsets ---
        nonempty_idx = [i for i, l in enumerate(X_valid.lengths) if l > 0]
        seq_tensor = torch.cat([X_valid.sequence[i] for i in nonempty_idx], dim=0).to(dtype=DTYPE, device=self.device)
        seq_offsets = torch.cumsum(torch.tensor([0] + [X_valid.lengths[i] for i in nonempty_idx], device=self.device), dim=0)

        def squeeze_if_singleton(t):
            return t.squeeze(0) if t.ndim == 3 and t.shape[0] == 1 else t

        for run_idx in range(n_init):
            if verbose:
                print(f"\n=== Run {run_idx + 1}/{n_init} ===")

            # --- Resample parameters after first run ---
            if run_idx > 0:
                sampled = self.sample_model_params(X, valid_theta)
                self._init_logits.copy_(squeeze_if_singleton(sampled['init_logits']))
                self._transition_logits.copy_(squeeze_if_singleton(sampled['transition_logits']))
                self._duration_logits.copy_(squeeze_if_singleton(sampled['duration_logits']))
                self._params['emission_pdf'] = sampled['emission_pdf']

            # --- Initial log-likelihood ---
            curr_ll = self._compute_log_likelihood(X_valid).sum()
            self.conv.push_pull(curr_ll, 0, run_idx)

            for it in range(1, max_iter + 1):
                # --- E-step: vectorized log_probs ---
                pdf = self._params['emission_pdf']
                if pdf is None or not hasattr(pdf, 'log_prob'):
                    raise RuntimeError("Emission PDF not initialized or incompatible with vectorized log_prob.")

                all_log_probs = pdf.log_prob(seq_tensor.unsqueeze(1))  # (T_total, K) or (B, T_total, K)
                if all_log_probs.ndim == 3 and all_log_probs.shape[0] == 1:
                    all_log_probs = all_log_probs.squeeze(0)

                # Split back into sequences
                X_valid.log_probs = [
                    all_log_probs[seq_offsets[i]:seq_offsets[i+1]] for i in range(len(seq_offsets)-1)
                ]

                # Fill empty sequences safely
                X_valid.log_probs = [
                    torch.empty((0, self.n_states), dtype=DTYPE, device=self.device)
                    if l == 0 else lp
                    for lp, l in zip(X_valid.log_probs, X_valid.lengths)
                ]

                # --- M-step: estimate model params ---
                new_params = self._estimate_model_params(X_valid, valid_theta)
                for key, attr in [('init_logits', '_init_logits'),
                                  ('transition_logits', '_transition_logits'),
                                  ('duration_logits', '_duration_logits'),
                                  ('emission_pdf', None)]:
                    if key in new_params:
                        val = new_params[key]
                        if key != 'emission_pdf':
                            if val.ndim == 3 and val.shape[0] == 1:
                                val = val.squeeze(0)
                            getattr(self, attr).copy_(val)
                        else:
                            self._params[key] = val

                # --- Evaluate log-likelihood & check convergence ---
                curr_ll = self._compute_log_likelihood(X_valid).sum()
                converged = self.conv.push_pull(curr_ll, it, run_idx)
                if converged and not ignore_conv and verbose:
                    print(f"[Run {run_idx + 1}] Converged at iteration {it}.")
                    break

            # --- Track best run ---
            run_score = float(curr_ll.item())
            if run_score > best_score:
                best_score = run_score
                best_state = {
                    'init_logits': self._init_logits.clone(),
                    'transition_logits': self._transition_logits.clone(),
                    'duration_logits': self._duration_logits.clone(),
                    'emission_pdf': self._params['emission_pdf']
                }

        # --- Restore best parameters ---
        if best_state is not None:
            self._init_logits.copy_(best_state['init_logits'])
            self._transition_logits.copy_(best_state['transition_logits'])
            self._duration_logits.copy_(best_state['duration_logits'])
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
        """
        Batched MAP or Viterbi decoding for HSMM sequences with optional context.
        Supports variable sequence lengths and GPU execution.
        """
        X_valid = self.to_observations(X, lengths)
        B = len(X_valid.lengths)
        K = self.n_states
        max_T = max(X_valid.lengths) if X_valid.lengths else 0
        device = self.device

        if max_T == 0:
            return [torch.empty(0, dtype=torch.long) for _ in range(B)]

        # --- Context ---
        if context is not None:
            if context.ndim == 1:
                context = context.unsqueeze(0)
            elif context.ndim > 2:
                context = context.mean(dim=0, keepdim=True)
            context = context.to(device=device, dtype=DTYPE)

        # --- Emission log-probs (batched) ---
        seq_tensor = torch.cat(X_valid.sequence, dim=0).to(device=device, dtype=DTYPE)
        seq_offsets = [0] + list(torch.cumsum(torch.tensor(X_valid.lengths, device=device), dim=0).tolist())
        log_probs_chunks = []

        pdf = getattr(self, "_params", {}).get("emission_pdf", None)
        if pdf is None or not hasattr(pdf, "log_prob"):
            raise RuntimeError("Emission PDF or module required for log_probs.")

        for start in range(0, seq_tensor.shape[0], batch_size):
            end = min(start + batch_size, seq_tensor.shape[0])
            chunk = seq_tensor[start:end].unsqueeze(1)
            # Pass per-sample context to emission
            chunk_context = context[start:end] if context is not None else None
            pdf = self.sample_emission_pdf(theta=chunk_context)
            lp = pdf.log_prob(chunk)
            if lp.ndim > 2:
                lp = lp.sum(dim=list(range(2, lp.ndim)))
            log_probs_chunks.append(lp)
        all_log_probs = torch.cat(log_probs_chunks, dim=0)
        X_valid.log_probs = [all_log_probs[seq_offsets[i]:seq_offsets[i+1]] for i in range(B)]

        # --- Transition & Duration logits ---
        transition_logits = self._align_logits(getattr(self, "transition_logits"), B)  # (B, K, K)
        duration_logits = self._align_logits(getattr(self, "duration_logits"), B)      # (B, K, Dmax)
        Dmax = duration_logits.shape[2]

        # --- Prepare batch tensor ---
        log_B = torch.full((B, max_T, K), -torch.inf, device=device, dtype=DTYPE)
        for i, lp in enumerate(X_valid.log_probs):
            log_B[i, :lp.shape[0]] = lp
        mask = torch.arange(max_T, device=device).unsqueeze(0) < torch.tensor(X_valid.lengths, device=device).unsqueeze(1)

        # --- Precompute duration terms ---
        dur_idx = torch.arange(max_T, device=device).unsqueeze(0).expand(B, -1)
        dur_idx = torch.clamp(dur_idx, max=Dmax-1)
        dur_term = torch.gather(duration_logits, 2, dur_idx.unsqueeze(1).expand(B, K, -1))  # (B, K, T)
        dur_term = dur_term.permute(0, 2, 1)  # (B, T, K)

        # --- Decoding ---
        algorithm = algorithm.lower()
        if algorithm == "map":
            alpha = torch.full((B, max_T, K), -torch.inf, device=device, dtype=DTYPE)
            alpha[:, 0] = log_B[:, 0]
            for t in range(1, max_T):
                prev = alpha[:, t-1].unsqueeze(2) + transition_logits  # (B, K, K)
                prev = torch.logsumexp(prev, dim=1) + dur_term[:, t]    # (B, K)
                alpha[:, t] = torch.where(mask[:, t].unsqueeze(-1), prev + log_B[:, t], alpha[:, t-1])
            decoded = torch.argmax(alpha, dim=-1)

        elif algorithm == "viterbi":
            delta = torch.full((B, max_T, K), -torch.inf, device=device, dtype=DTYPE)
            psi = torch.zeros((B, max_T, K), dtype=torch.long, device=device)
            delta[:, 0] = log_B[:, 0]

            for t in range(1, max_T):
                scores = delta[:, t-1].unsqueeze(2) + transition_logits + dur_term[:, t].unsqueeze(1)  # (B, K, K)
                psi[:, t] = torch.argmax(scores, dim=1)  # (B, K)
                delta[:, t] = torch.max(scores, dim=1).values + log_B[:, t]
                delta[:, t] = torch.where(mask[:, t].unsqueeze(-1), delta[:, t], delta[:, t-1])

            decoded = torch.zeros((B, max_T), dtype=torch.long, device=device)
            decoded[:, -1] = torch.argmax(delta[:, -1], dim=-1)
            for t in range(max_T-2, -1, -1):
                decoded[:, t] = torch.gather(psi[:, t+1], 1, decoded[:, t+1].unsqueeze(1)).squeeze(1)
                decoded[:, t] = torch.where(mask[:, t], decoded[:, t], decoded[:, t+1])
        else:
            raise ValueError(f"Unknown decoding algorithm '{algorithm}'")

        return [decoded[i, :L].detach().cpu() for i, L in enumerate(X_valid.lengths)]

    def score(
        self,
        X: torch.Tensor,
        lengths: Optional[List[int]] = None,
        by_sample: bool = True,
        theta: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Vectorized, duration-aware log-likelihood for HSMM with optional context.
        Batch-safe and memory-efficient version.
        """
        obs = self.to_observations(X, lengths, theta)
        B = len(obs.lengths)
        max_T = max(obs.lengths) if B > 0 else 0
        K, Dmax = self.n_states, self.max_duration

        if B == 0 or max_T == 0:
            return torch.zeros(B if by_sample else 1, dtype=DTYPE, device=self.device)

        # --- Emission log-probs ---
        pdf = self._params.get("emission_pdf")
        if pdf is None or not hasattr(pdf, "log_prob"):
            raise RuntimeError("Emission PDF must be initialized with log_prob method.")

        all_seq = torch.cat(obs.sequence, dim=0).to(dtype=DTYPE, device=self.device)
        all_log_probs = pdf.log_prob(all_seq.unsqueeze(1))  # (total_T, K)
        seq_offsets = torch.cat([torch.tensor([0], device=self.device),
                                 torch.cumsum(torch.tensor(obs.lengths, device=self.device), dim=0)])
        log_B = torch.full((B, max_T, K), -torch.inf, device=self.device, dtype=DTYPE)
        for i in range(B):
            start, end = seq_offsets[i].item(), seq_offsets[i+1].item()
            L = obs.lengths[i]
            log_B[i, :L] = all_log_probs[start:end]

        # --- Transition & duration logits ---
        log_transition = F.log_softmax(self._transition_logits, dim=-1).to(self.device, dtype=DTYPE)
        log_duration = F.log_softmax(self._duration_logits, dim=-1).to(self.device, dtype=DTYPE)
        init_logits = self.init_logits.to(self.device, dtype=DTYPE)

        # --- Forward DP ---
        V = torch.full((B, max_T, K), -torch.inf, device=self.device, dtype=DTYPE)
        V[:, 0] = log_B[:, 0] + init_logits

        cumsum_B = torch.cat([torch.zeros((B, 1, K), device=self.device, dtype=DTYPE), log_B.cumsum(dim=1)], dim=1)

        for t in range(1, max_T):
            max_d = min(Dmax, t + 1)
            durations = torch.arange(1, max_d + 1, device=self.device)  # [1..max_d]
            start_idx = t - durations + 1  # (max_d,)

            # Safe prev_V: for negative indices, use init_logits
            prev_V = []
            emit_sums = []
            dur_scores = log_duration[:, :max_d].T.unsqueeze(0).expand(B, -1, -1)  # (B, max_d, K)

            for d_idx, s in enumerate(start_idx):
                if s < 0:
                    prev_V.append(init_logits.unsqueeze(0))  # (B, K)
                    emit_sums.append(cumsum_B[:, t+1, :] - 0.0)
                else:
                    prev_V.append(V[:, s, :])
                    emit_sums.append(cumsum_B[:, t+1, :] - cumsum_B[:, s, :])

            prev_V = torch.stack(prev_V, dim=1)      # (B, max_d, K)
            emit_sums = torch.stack(emit_sums, dim=1)  # (B, max_d, K)

            # Efficient logsumexp over previous states without forming (B, max_d, K_prev, K_curr)
            max_prev = prev_V.unsqueeze(3) + log_transition.unsqueeze(0)  # (B, max_d, K_prev, K_curr)
            max_prev = torch.logsumexp(max_prev, dim=2)                    # (B, max_d, K_curr)

            # Combine duration, emission
            V[:, t, :] = torch.logsumexp(max_prev + dur_scores + emit_sums, dim=1)

        # --- Mask invalid positions ---
        mask = torch.arange(max_T, device=self.device).unsqueeze(0) < torch.tensor(obs.lengths, device=self.device).unsqueeze(1)
        V = torch.where(mask.unsqueeze(-1), V, torch.full_like(V, -torch.inf))

        # --- Sequence log-likelihood ---
        end_idx = torch.tensor(obs.lengths, device=self.device) - 1
        seq_ll = torch.logsumexp(V[torch.arange(B), end_idx], dim=-1)

        return seq_ll.detach().cpu() if by_sample else seq_ll.sum().detach().cpu()

    def _compute_log_likelihood(self, X: utils.Observations, verbose: bool = False) -> torch.Tensor:
        """
        Universal HSMM log-likelihood computation.
        Auto-detects whether `_forward()` returns a list of tensors or a batched tensor.
        Stores per-sequence log-likelihoods in X.log_likelihoods.

        Returns:
            torch.Tensor of shape (B,) with per-sequence log-likelihoods.
        """
        alpha_out = self._forward(X)
        if alpha_out is None:
            raise RuntimeError("Forward pass returned None. Model may be uninitialized.")

        dtype, device = DTYPE, self.device
        neg_inf = torch.tensor(-float("inf"), device=device, dtype=dtype)

        # --- Case 1: Batched tensor output [B, T, K, Dmax] ---
        if isinstance(alpha_out, torch.Tensor):
            if alpha_out.ndim != 4:
                raise ValueError(f"Expected batched alpha tensor [B, T, K, Dmax], got {tuple(alpha_out.shape)}.")

            B, T, K, Dmax = alpha_out.shape
            seq_lengths = torch.as_tensor(X.lengths, device=device, dtype=torch.long)
            if seq_lengths.numel() != B:
                raise ValueError("Mismatch between number of sequences and alpha batch size.")

            if seq_lengths.max().item() == 0:
                ll = torch.full((B,), neg_inf, device=device, dtype=dtype)
            else:
                alpha_batch = alpha_out.nan_to_num(nan=neg_inf, posinf=neg_inf, neginf=neg_inf)
                batch_idx = torch.arange(B, device=device)
                last_idx = (seq_lengths - 1).clamp(min=0)
                last_alpha = alpha_batch[batch_idx, last_idx]  # [B, K, Dmax]
                ll = torch.logsumexp(last_alpha.view(B, -1), dim=-1)
                ll = torch.where(seq_lengths > 0, ll, neg_inf)

            X.log_likelihoods = ll.detach().clone()

            if verbose:
                print(f"[compute_ll/batched] alpha_batch={alpha_out.shape}, seq_lengths={seq_lengths.tolist()}")
                print(f"[compute_ll/batched] log-likelihoods: {ll}")

            return ll

        # --- Case 2: List of tensors (variable-length per sequence) ---
        elif isinstance(alpha_out, (list, tuple)):
            if not alpha_out:
                raise RuntimeError("Forward pass returned empty list of alpha tensors.")

            B = len(alpha_out)
            seq_lengths = torch.tensor([a.shape[0] for a in alpha_out], device=device, dtype=torch.long)
            max_T = seq_lengths.max()

            if max_T == 0:
                ll = torch.full((B,), neg_inf, device=device, dtype=dtype)
            else:
                alpha_clamped = [a.nan_to_num(nan=neg_inf, posinf=neg_inf, neginf=neg_inf) for a in alpha_out]
                alpha_padded = torch.nn.utils.rnn.pad_sequence(alpha_clamped, batch_first=True, padding_value=neg_inf)
                batch_idx = torch.arange(B, device=device)
                last_idx = (seq_lengths - 1).clamp(min=0)
                last_alpha = alpha_padded[batch_idx, last_idx]
                ll = torch.logsumexp(last_alpha.view(B, -1), dim=-1)
                ll = torch.where(seq_lengths > 0, ll, neg_inf)

            X.log_likelihoods = ll.detach().clone()

            if verbose:
                print(f"[compute_ll/list] alpha_padded={alpha_padded.shape}, seq_lengths={seq_lengths.tolist()}")
                print(f"[compute_ll/list] log-likelihoods: {ll}")

            return ll

        else:
            raise TypeError(f"Unexpected alpha_out type: {type(alpha_out).__name__}")

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

