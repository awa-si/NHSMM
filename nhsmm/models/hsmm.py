# nhsmm/models/hsmm.py
from __future__ import annotations
from typing import Optional, List, Tuple, Any, Literal, Dict
from abc import ABC, abstractmethod
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, Categorical

from nhsmm.context import ContextEncoder
from nhsmm.defaults import (
    Emission, Duration, Transition, DTYPE, EPS, HSMMError
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
        init_emission: bool = True
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
            warnings.warn("Emission PDF initialized without data; may be degenerate.")
            self._params['emission_pdf'] = self.sample_emission_pdf(None)
        else:
            self._params['emission_pdf'] = None

    def _init_buffers(self):
        device = next(self.buffers(), torch.zeros(1, dtype=DTYPE)).device

        def sample_probs(shape):
            probs = constraints.sample_probs(self.alpha, shape, seed=self._seed_gen())
            probs = probs.clamp_min(self.min_covar)
            logits = torch.log(probs).to(device=device, dtype=DTYPE)
            return logits, probs.shape

        def sample_transitions(n_states):
            probs = constraints.sample_transition(
                self.alpha, n_states, self.transition_type, seed=self._seed_gen()
            )
            probs = probs.clamp_min(self.min_covar)
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

    def _reset_buffers(self):
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
        device = self.device

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

    def _align_logits(self, logits: torch.Tensor, B: int, name: str = "logits") -> torch.Tensor:
        """
        Align logits tensor to batch size B.

        Supports:
          - init_logits: [K], [1,K], [B,K]
          - transition_logits: [K,K], [1,K,K], [B,K,K]
          - duration_logits: [K,Dmax], [1,K,Dmax], [B,K,Dmax]

        Returns logits expanded to shape [B,...] suitable for batch operations.
        """
        if logits is None:
            raise ValueError(f"{name} cannot be None")
        
        logits = logits.to(self.device, dtype=DTYPE)
        ndim = logits.ndim

        # 1D: [K] -> [B,K]
        if ndim == 1:
            return logits.unsqueeze(0).expand(B, -1)

        # 2D: [1,K], [B,K], [K,Dmax] -> [B,K,...]
        elif ndim == 2:
            if logits.shape[0] == 1:
                return logits.expand(B, -1, -1) if logits.shape[1] > 1 else logits.expand(B, -1)
            elif logits.shape[0] == B:
                return logits
            else:
                # [K,Dmax] -> [B,K,Dmax]
                return logits.unsqueeze(0).expand(B, -1, -1)

        # 3D: [1,K,K], [B,K,K], [B,K,Dmax] -> keep B
        elif ndim == 3:
            if logits.shape[0] == 1:
                return logits.expand(B, -1, -1)
            elif logits.shape[0] == B:
                return logits
            else:
                raise ValueError(f"{name}: cannot broadcast 3D logits of shape {logits.shape} to batch {B}")

        else:
            raise ValueError(f"{name}: unsupported logits shape {logits.shape}")

    def _prepare_delta(self, delta: Optional[torch.Tensor], shape: Tuple[int,int,int], scale: float = 1.0, broadcast: bool = True) -> torch.Tensor:
        B_target, n_states, F_dim = shape
        device = self.device

        if delta is None:
            return torch.zeros((B_target, n_states, F_dim), dtype=DTYPE, device=device)

        delta = delta.to(device, DTYPE)

        # Flatten if needed
        if delta.ndim == 1:
            delta = delta.unsqueeze(0)
        elif delta.ndim > 2:
            delta = delta.reshape(delta.shape[0], -1)

        B, total_features = delta.shape
        expected_features = n_states * F_dim

        # Pad or truncate features
        if total_features < expected_features:
            delta = F.pad(delta, (0, expected_features - total_features))
        elif total_features > expected_features:
            delta = delta[..., :expected_features]

        delta = delta.reshape(B, n_states, F_dim)
        delta = scale * torch.tanh(torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0))

        if broadcast and B != B_target:
            if B == 1:
                delta = delta.expand(B_target, -1, -1)
            elif B < B_target:
                repeats = (B_target + B - 1) // B
                delta = delta.repeat(repeats, 1, 1)[:B_target]
            else:
                raise RuntimeError(f"Cannot broadcast delta with batch {B} to target {B_target}")

        return delta

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
        max_seq_len: int = 1024,
        n_heads: int = 4,
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

    def encode_observations(
        self,
        X: torch.Tensor,
        pool: Optional[str] = None,
        store: bool = True,
        detach_return: bool = True,
    ) -> Optional[torch.Tensor]:
        """
        Encode observations X into context vectors using the attached encoder.

        Supports inputs of shape:
            - (F,)       : single feature vector
            - (T,F)      : single sequence of length T
            - (B,F)      : batch of single-step sequences
            - (B,T,F)    : batch of sequences

        Handles device/dtype, optional pooling, detachment, and stores context.

        Args:
            X: Input tensor
            pool: Optional pooling mode override for the encoder
            store: If True, saves result to self._context
            detach_return: If True, detaches returned tensor to prevent gradient flow

        Returns:
            Context tensor of shape [B,H], or None if no encoder is attached
        """
        if self.encoder is None:
            if store:
                self._context = None
            return None

        # Move input to encoder device/dtype
        device = next(self.encoder.parameters()).device
        X = X.to(device=device, dtype=DTYPE)

        if X.numel() == 0:
            raise ValueError(f"Input X is empty: shape {X.shape}")

        # Normalize to (B,T,F)
        if X.ndim == 1:          # (F,) -> (1,1,F)
            X = X.unsqueeze(0).unsqueeze(0)
        elif X.ndim == 2:        # (T,F) or (B,F)
            if X.shape[1] == self.n_features:  # (T,F) -> (1,T,F)
                X = X.unsqueeze(0)
            else:  # (B,F) -> (B,1,F)
                X = X.unsqueeze(1)
        elif X.ndim != 3:
            raise ValueError(f"Expected input shape (F,), (T,F), or (B,T,F), got {X.shape}")

        # Handle optional pooling override
        original_pool = getattr(self.encoder, "pool", None)
        if pool is not None and hasattr(self.encoder, "pool"):
            self.encoder.pool = pool

        try:
            # Forward pass through encoder
            _ = self.encoder(X, return_context=True)
            vec = self.encoder.get_context()
            if vec is None:
                raise RuntimeError("Encoder returned None context")
        except Exception as e:
            logger.error(f"[encode_observations] Encoder failed for shape {X.shape}: {e}", exc_info=True)
            raise RuntimeError(f"Encoder forward() failed for input shape {X.shape}: {e}")
        finally:
            # Restore original pooling
            if hasattr(self.encoder, "pool"):
                self.encoder.pool = original_pool

        # Ensure device/dtype consistency
        if vec.device != device or vec.dtype != DTYPE:
            vec = vec.to(device=device, dtype=DTYPE)

        # Optionally detach to prevent gradient flow
        if detach_return:
            vec = vec.detach()

        # Store batch-aware context
        if store:
            self._context = vec

        return vec

    def check_constraints(self, value: torch.Tensor, clamp: bool = False) -> torch.Tensor:
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

    def _contextual_emission_pdf(self, X: torch.Tensor, theta: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute context-modulated emission log-probabilities [B, T, K].
        Always returns numeric log-probs, never a Distribution.
        Supports inputs: [F], [T,F], [B,F], [B,T,F].

        Tiny random jitter is added to prevent collapsed states during EM.
        """

        # --- Normalize input to [B,T,F] ---
        if X.ndim == 1:           # [F] -> [1,1,F]
            X = X.unsqueeze(0).unsqueeze(0)
        elif X.ndim == 2:
            if X.shape[1] == self.n_features:  # [T,F] -> [1,T,F]
                X = X.unsqueeze(0)
            else:                              # [B,F] -> [B,1,F]
                X = X.unsqueeze(1)
        elif X.ndim != 3:
            raise ValueError(f"Unsupported input shape {X.shape}")

        B, T, F = X.shape

        # --- Align context ---
        theta_batch = self.combine_context(theta)
        if theta_batch is not None:
            if theta_batch.ndim == 2:            # [B,H] -> [B,T,H]
                theta_batch = theta_batch.unsqueeze(1).expand(-1, T, -1)
            elif theta_batch.ndim == 3:          # [B,T,H] already ok
                if theta_batch.shape[0] != B or theta_batch.shape[1] != T:
                    theta_batch = theta_batch.expand(B, T, -1)
            else:
                raise ValueError(f"Unsupported theta shape {theta_batch.shape}")

        # --- Compute log-probs vectorized ---
        logp_tensor = self.emission_module.log_prob(X, context=theta_batch)  # [B,T,K] or [B,T]
        if logp_tensor.ndim == 2:                 # [B,T] -> [B,T,1]
            logp_tensor = logp_tensor.unsqueeze(-1)

        # --- Sanitize numerics to avoid EM collapse ---
        dtype_info = torch.finfo(logp_tensor.dtype)
        min_val = dtype_info.min / 2
        logp_tensor = torch.nan_to_num(logp_tensor, nan=min_val, neginf=min_val, posinf=dtype_info.max / 2)

        # --- Optional tiny jitter to prevent collapsed states ---
        if B > 1 and T > 1:
            logp_tensor = logp_tensor + 1e-6 * torch.rand_like(logp_tensor)

        return logp_tensor

    def _contextual_duration_pdf(self, theta: Optional[ContextualVariables] = None) -> torch.Tensor:
        """
        Compute context-modulated duration log-probabilities [B, K, max_duration].

        Tiny random jitter is added to avoid collapsed duration states during EM.
        """
        base_logits = self._duration_logits.unsqueeze(0)  # [1, K, Dmax]
        theta_batch = self.combine_context(theta)
        B = theta_batch.shape[0] if theta_batch is not None else 1

        # Expand base logits to batch
        base_logits = base_logits.expand(B, -1, -1)

        # Apply context modulation
        log_duration = self.duration_module._apply_context(base_logits, theta_batch)
        if not torch.is_tensor(log_duration):
            raise TypeError(f"Expected tensor from _apply_context, got {type(log_duration)}")

        # Sanitize numerics
        dtype_info = torch.finfo(log_duration.dtype)
        log_duration = torch.nan_to_num(log_duration, nan=dtype_info.min / 2, posinf=dtype_info.max / 2)

        # Optional tiny jitter to prevent EM collapse
        log_duration = log_duration + 1e-8 * torch.rand_like(log_duration)

        # Normalize along duration axis
        log_duration = log_duration - torch.logsumexp(log_duration, dim=-1, keepdim=True)

        return log_duration

    def _contextual_transition_matrix(self, theta: Optional[ContextualVariables] = None) -> torch.Tensor:
        """
        Compute context-modulated transition log-probabilities [B, K, K].

        Tiny random jitter is added to prevent collapsed transitions during EM.
        Structural transition constraints are enforced.
        """
        base_logits = self._transition_logits.unsqueeze(0)  # [1, K, K]
        theta_batch = self.combine_context(theta)
        B = theta_batch.shape[0] if theta_batch is not None else 1

        # Expand base logits to batch
        base_logits = base_logits.expand(B, -1, -1)

        # Contextual modulation
        log_transition = self.transition_module._apply_context(base_logits, theta_batch)
        if not torch.is_tensor(log_transition):
            raise TypeError(f"Expected tensor from _apply_context, got {type(log_transition)}")

        if log_transition.ndim == 2:
            log_transition = log_transition.unsqueeze(0)  # [1, K, K]

        # Enforce structural constraints
        if hasattr(self, "transition_type"):
            mask = constraints.mask_invalid_transitions(self.n_states, self.transition_type).to(log_transition.device)
            log_transition = log_transition.masked_fill(~mask.unsqueeze(0), float("-inf"))

        # Sanitize numerics
        dtype_info = torch.finfo(log_transition.dtype)
        log_transition = torch.nan_to_num(log_transition, nan=dtype_info.min / 2, posinf=dtype_info.max / 2)

        # Optional tiny jitter to avoid EM state collapse
        log_transition = log_transition + 1e-8 * torch.rand_like(log_transition)

        # Normalize rows
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
        Supports batched sequences, handles context, and computes context-modulated log-probabilities.
        """
        device = next(self.parameters(), torch.tensor(0.0)).device
        X = torch.as_tensor(X, dtype=DTYPE, device=device) if not torch.is_tensor(X) else X.to(dtype=DTYPE, device=device)

        # --- Normalize input to [B, T, F] ---
        if X.ndim == 2:           # [T, F] -> [1, T, F]
            X = X.unsqueeze(0)
        elif X.ndim != 3:
            raise ValueError(f"Unsupported input shape {X.shape}")
        B, T, F = X.shape

        # --- Validate input ---
        X_valid = self.check_constraints(X)

        # --- Align context ---
        if theta is None and getattr(self, "_context", None) is not None:
            theta = self._context

        if theta is not None:
            theta = theta.to(dtype=DTYPE, device=device)
            if theta.ndim == 2:             # [B, H] -> [B, T, H]
                theta = theta.unsqueeze(1).expand(-1, T, -1)
            elif theta.ndim == 3:           # [B, T, H]
                if theta.shape[1] != T:
                    theta = theta.expand(-1, T, -1)
            else:
                raise ValueError(f"Unsupported theta shape {theta.shape}")

        # --- Sequence lengths ---
        if lengths is None:
            lengths = [T] * B
        elif sum(lengths) != T:
            raise ValueError(f"Sum of lengths ({sum(lengths)}) does not match total samples ({T}).")
        
        # --- Split sequences ---
        sequences = [X_valid[i, :l] for i, l in enumerate(lengths)]
        context_list = [theta[i, :l] if theta is not None else None for i, l in enumerate(lengths)]

        # --- Compute log-probabilities ---
        log_probs_list = []
        for seq, seq_theta in zip(sequences, context_list):
            pdf_or_tensor = self._contextual_emission_pdf(seq, seq_theta)

            if isinstance(pdf_or_tensor, torch.distributions.Distribution):
                # compute log-prob for the sequence
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
            lengths=lengths,
            context=context_list,
            log_probs=log_probs_list
        )


    # hsmm.HSMM precessing
    def _forward(self, X: utils.Observations, theta: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """
        Batched, vectorized forward algorithm for HSMM with optional context θ.
        Returns a list of log-alpha tensors of shape (T_i, K, Dmax) per sequence.

        Adds global EPS jitter to avoid collapsed states in EM.
        """
        K, Dmax = self.n_states, self.max_duration
        neg_inf = -torch.finfo(DTYPE).min / 2
        device, dtype = self.device, DTYPE

        B = len(X.sequence)
        lengths = torch.tensor(X.lengths, device=device)
        T_max = lengths.max().item() if lengths.numel() > 0 else 0

        # --- Prepare log_probs tensor [B, T_max, K] with padding ---
        log_probs_padded = torch.full((B, T_max, K), neg_inf, device=device, dtype=dtype)
        for b, L in enumerate(lengths):
            if L == 0:
                continue
            lp = X.log_probs[b]
            if isinstance(lp, torch.distributions.Distribution):
                obs = X.sequence[b].to(device=device, dtype=dtype)
                lp = lp.log_prob(obs.unsqueeze(-2))
                if lp.ndim > 2:
                    lp = lp.sum(dim=tuple(range(2, lp.ndim)))
            lp = torch.nan_to_num(lp, nan=neg_inf, neginf=neg_inf, posinf=1e8)
            log_probs_padded[b, :L] = lp.to(device=device, dtype=dtype)

        # --- Align theta to [B, T_max, H] ---
        aligned_theta = None
        if theta is not None:
            theta = theta.to(device=device, dtype=dtype)
            if theta.ndim == 1:  # [H] global
                aligned_theta = theta.unsqueeze(0).unsqueeze(1).expand(B, T_max, -1)
            elif theta.ndim == 2:
                if theta.shape[0] == B:
                    aligned_theta = theta.unsqueeze(1).expand(B, T_max, -1)
                elif theta.shape[0] == log_probs_padded[:, :, 0].numel():
                    expanded = torch.zeros(B, T_max, theta.shape[-1], device=device, dtype=dtype)
                    offset = 0
                    for b, L in enumerate(lengths):
                        if L > 0:
                            expanded[b, :L] = theta[offset:offset + L]
                            offset += L
                    aligned_theta = expanded
                else:
                    raise ValueError(f"Cannot align theta shape {theta.shape} with batch {B}, T_max {T_max}")
            elif theta.ndim == 3:
                aligned_theta = theta[:, :T_max, :]
            else:
                raise ValueError(f"Unsupported theta ndim {theta.ndim}")
            # Apply context-modulated emission if method exists
            log_probs_padded = self._contextual_emission_pdf(log_probs_padded, aligned_theta)

        # --- Precompute cumulative sums over durations ---
        cumsum_emit = torch.zeros(B, T_max + 1, K, device=device, dtype=dtype)
        cumsum_emit[:, 1:, :] = torch.cumsum(log_probs_padded, dim=1)

        # --- Initialize log_alpha [B, T_max, K, Dmax] ---
        log_alpha = torch.full((B, T_max, K, Dmax), neg_inf, device=device, dtype=dtype)

        # --- Initial step ---
        durations0 = torch.arange(1, Dmax + 1, device=device)
        for b in range(B):
            L = lengths[b].item()
            if L == 0:
                continue
            d0 = min(Dmax, L)
            emit_sums0 = (cumsum_emit[b, durations0[:d0]] - cumsum_emit[b, 0]).T
            log_alpha[b, 0, :, :d0] = self.init_logits.unsqueeze(1) + self.duration_logits[:, :d0] + emit_sums0
            log_alpha[b, 0, :, :d0] += EPS  # global jitter

        # --- Recursion ---
        for t in range(1, T_max):
            for b in range(B):
                L = lengths[b].item()
                if t >= L:
                    continue
                dt = min(Dmax, t + 1)
                durations = torch.arange(1, dt + 1, device=device)
                starts = t - durations + 1
                emit_sums = (cumsum_emit[b, t + 1].unsqueeze(0) - cumsum_emit[b, starts]).T
                idx = torch.clamp(starts - 1, min=0)
                prev_alpha_first = log_alpha[b, idx, :, 0]
                mask = (starts == 0).unsqueeze(1)
                prev_alpha_first = torch.where(mask, self.init_logits.unsqueeze(0), prev_alpha_first)
                prev_alpha_exp = prev_alpha_first.unsqueeze(2) + self.transition_logits.unsqueeze(0)
                prev_alpha_exp = torch.nan_to_num(prev_alpha_exp, nan=neg_inf, neginf=neg_inf, posinf=1e8)
                log_alpha[b, t, :, :dt] = torch.logsumexp(prev_alpha_exp, dim=1).T + self.duration_logits[:, :dt] + emit_sums
                log_alpha[b, t, :, :dt] += EPS  # global jitter

        # --- Split back per sequence ---
        return [log_alpha[b, :lengths[b]] for b in range(B)]

    def _backward(self, X: utils.Observations, theta: Optional[torch.Tensor] = None) -> list[torch.Tensor]:
        """
        Vectorized backward algorithm for HSMM with optional context θ.
        Returns a list of log-beta tensors of shape (T_i, K, Dmax) per sequence.
        """
        device = self.device
        K, Dmax = self.n_states, self.max_duration
        neg_inf = -torch.finfo(DTYPE).min / 2

        # Model parameters
        transition_logits = self.transition_logits.to(device=device, dtype=DTYPE)
        duration_logits = self.duration_logits.to(device=device, dtype=DTYPE)

        # --- Batch info ---
        B = len(X.sequence)
        lengths = torch.tensor(X.lengths, device=device)
        T_max = lengths.max().item() if lengths.numel() > 0 else 0

        # --- Prepare log_probs tensor [B, T_max, K] ---
        log_probs_padded = torch.full((B, T_max, K), neg_inf, device=device, dtype=DTYPE)
        for b, L in enumerate(lengths):
            if L > 0:
                lp = X.log_probs[b]
                if isinstance(lp, torch.distributions.Distribution):
                    obs = X.sequence[b].to(device=device, dtype=DTYPE)
                    lp = lp.log_prob(obs.unsqueeze(-2))
                    if lp.ndim > 2:
                        lp = lp.sum(dim=tuple(range(2, lp.ndim)))
                log_probs_padded[b, :L] = lp.to(dtype=DTYPE, device=device)

        # --- Align theta to [B, T_max, H] ---
        if theta is not None:
            theta = theta.to(device=device, dtype=DTYPE)
            if theta.ndim == 1:  # global
                aligned_theta = theta.unsqueeze(0).unsqueeze(1).expand(B, T_max, -1)
            elif theta.ndim == 2:
                if theta.shape[0] == B:
                    aligned_theta = theta.unsqueeze(1).expand(B, T_max, -1)
                elif theta.shape[0] == T_max:
                    aligned_theta = theta.unsqueeze(0).expand(B, -1, -1)
                elif theta.shape[0] == 1:
                    aligned_theta = theta.unsqueeze(1).expand(B, T_max, -1)
                else:
                    raise ValueError(f"Cannot align theta shape {theta.shape} with batch {B}, T_max {T_max}")
            elif theta.ndim == 3 and theta.shape[0] == B and theta.shape[1] == T_max:
                aligned_theta = theta.clone()
            else:
                raise ValueError(f"Unsupported theta shape {theta.shape}")
            log_probs_padded = self._contextual_emission_pdf(log_probs_padded, aligned_theta)

        # --- Cumulative sums over durations ---
        cumsum_emit = torch.zeros(B, T_max + 1, K, device=device, dtype=DTYPE)
        cumsum_emit[:, 1:, :] = torch.cumsum(log_probs_padded, dim=1)

        # --- Initialize log_beta [B, T_max, K, Dmax] ---
        log_beta = torch.full((B, T_max, K, Dmax), neg_inf, device=device, dtype=DTYPE)
        # Last time step: beta[T-1,:,0] = 0
        for b in range(B):
            if lengths[b] > 0:
                log_beta[b, lengths[b]-1, :, 0] = 0.0

        # --- Backward recursion ---
        for t in reversed(range(T_max - 1)):
            dt = torch.arange(1, Dmax + 1, device=device)
            ends = t + dt  # [Dmax]
            mask_valid = ends < T_max

            # Only compute for sequences where t < length
            for b in range(B):
                L = lengths[b].item()
                if t >= L:
                    continue
                valid_dt = dt[mask_valid]
                ends_b = ends[mask_valid]

                # Emission sums
                emit_sums = cumsum_emit[b, ends_b] - cumsum_emit[b, t].unsqueeze(0)

                # Duration + beta of next
                beta_next = log_beta[b, ends_b - 1, :, 0]  # [valid_dt, K]
                dur_scores = duration_logits[:, :valid_dt.numel()].T  # [valid_dt, K]

                combined = emit_sums + beta_next + dur_scores + EPS
                log_beta[b, t, :, 0] = torch.logsumexp(combined.T, dim=1)

                if valid_dt.numel() > 1:
                    shift_len = valid_dt.numel() - 1
                    log_beta[b, t, :, 1:shift_len + 1] = log_beta[b, t + 1, :, :shift_len] + log_probs_padded[b, t + 1, :].unsqueeze(-1) + EPS

        # --- Split into list per sequence ---
        beta_list = [log_beta[b, :lengths[b]] for b in range(B)]
        return beta_list

    def _compute_posteriors(
        self,
        X: utils.Observations,
        theta: Optional[torch.Tensor] = None
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """
        Compute HSMM posteriors: gamma, xi, eta (batched, vectorized).

        Returns lists of tensors per sequence:
            gamma: [T_i, K]
            xi:    [T_i-1, K, K]
            eta:   [T_i, K, Dmax]

        Supports context-modulated emissions via theta.
        """
        device = self.device
        K, Dmax = self.n_states, self.max_duration
        EPS_ = EPS

        # --- Forward / Backward ---
        alpha_list = self._forward(X, theta=theta)
        beta_list = self._backward(X, theta=theta)

        init_logits = self.init_logits.to(device=device, dtype=DTYPE)
        transition_logits = self.transition_logits.to(device=device, dtype=DTYPE)

        gamma_list, xi_list, eta_list = [], [], []

        # Sequence lengths
        seq_lengths = X.lengths
        B = len(seq_lengths)
        max_T = max(seq_lengths) if seq_lengths else 0

        # --- Pad alpha/beta for vectorized computation ---
        alpha_batch = torch.full((B, max_T, K, Dmax), -torch.inf, device=device, dtype=DTYPE)
        beta_batch = torch.full_like(alpha_batch, -torch.inf)
        for b, (alpha, beta, L) in enumerate(zip(alpha_list, beta_list, seq_lengths)):
            if L > 0:
                alpha_batch[b, :L] = alpha
                beta_batch[b, :L] = beta

        # --- Compute log-eta (full duration-aware posterior) ---
        log_eta = alpha_batch + beta_batch  # [B, T, K, Dmax]
        # Flatten last two dims and normalize per timestep
        log_eta_flat = log_eta.view(B, max_T, K * Dmax)
        log_eta_norm = log_eta_flat - torch.logsumexp(log_eta_flat, dim=2, keepdim=True)
        eta_batch = torch.clamp(log_eta_norm.view(B, max_T, K, Dmax).exp(), min=EPS_)
        # normalize per timestep
        eta_batch = eta_batch / eta_batch.sum(dim=(2,3), keepdim=True).clamp_min(EPS_)
        eta_list = [eta_batch[b, :L] for b, L in enumerate(seq_lengths)]

        # --- Compute gamma (state marginal) ---
        gamma_batch = eta_batch.sum(dim=3)  # sum over durations -> [B, T, K]
        gamma_batch = torch.clamp(gamma_batch, min=EPS_)
        gamma_batch = gamma_batch / gamma_batch.sum(dim=2, keepdim=True).clamp_min(EPS_)
        gamma_list = [gamma_batch[b, :L] for b, L in enumerate(seq_lengths)]

        # --- Compute xi (state transitions) ---
        xi_list = []
        for b, L in enumerate(seq_lengths):
            if L <= 1:
                xi_list.append(torch.zeros((0, K, K), dtype=DTYPE, device=device))
                continue

            # collapse durations for alpha/beta
            alpha_seq = torch.logsumexp(alpha_batch[b, :L, :, :], dim=2)  # [T, K]
            beta_seq_next = torch.logsumexp(beta_batch[b, 1:L, :, :], dim=2)  # [T-1, K]

            # compute log xi
            log_xi = alpha_seq[:-1].unsqueeze(2) + transition_logits.unsqueeze(0) + beta_seq_next.unsqueeze(1)
            xi_seq = torch.clamp(torch.softmax(log_xi, dim=2), min=EPS_)
            xi_seq = xi_seq / xi_seq.sum(dim=(1,2), keepdim=True).clamp_min(EPS_)
            xi_list.append(xi_seq)

        return gamma_list, xi_list, eta_list

    def _map(self, X: utils.Observations) -> list[torch.Tensor]:
        """Fully vectorized MAP decoding of HSMM sequences from posterior state marginals."""
        gamma_list, _, _ = self._compute_posteriors(X)
        if gamma_list is None:
            raise RuntimeError("Posterior probabilities could not be computed — model parameters uninitialized.")

        device = self.device
        B = len(gamma_list)
        if B == 0:
            return []

        seq_lengths = torch.tensor([g.shape[0] for g in gamma_list], device=device)
        max_T = seq_lengths.max().item()
        K = self.n_states

        if max_T == 0:
            return [torch.empty(0, dtype=torch.long, device=device) for _ in range(B)]

        # Create padded tensor
        gamma_padded = torch.full((B, max_T, K), -float("inf"), device=device, dtype=DTYPE)
        for i, g in enumerate(gamma_list):
            gamma_padded[i, : g.shape[0]] = g

        # MAP decoding
        map_padded = gamma_padded.argmax(dim=-1)  # [B, T]

        # Fully vectorized split: flatten and mask
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, max_T)  # [B, T]
        time_mask = torch.arange(max_T, device=device).unsqueeze(0).expand(B, max_T) < seq_lengths.unsqueeze(1)  # [B, T]

        flat_map = map_padded[batch_idx, time_mask]  # flattened all valid timesteps
        splits = torch.cumsum(seq_lengths, dim=0)[:-1]
        map_sequences_list = flat_map.split(seq_lengths.tolist())

        # Ensure correct dtype
        map_sequences_list = [seq.to(dtype=torch.long, device=device) for seq in map_sequences_list]

        return map_sequences_list

    def _viterbi(
        self,
        X: utils.Observations,
        theta: Optional[torch.Tensor] = None,
        duration_weight: float = 0.0
    ) -> list[torch.Tensor]:
        """
        Batched, fully vectorized Viterbi for HSMM with optional context θ.
        Returns most likely state sequence per batch.
        """
        device = self.device
        K, Dmax = self.n_states, self.max_duration
        B = X.n_sequences
        lengths = X.lengths
        max_len = max(lengths) if lengths else 0
        if max_len == 0:
            return [torch.empty(0, dtype=torch.int64, device=device) for _ in range(B)]

        # Model params
        init_logits = self.init_logits.to(device=device, dtype=DTYPE)
        transition_logits = self.transition_logits.to(device=device, dtype=DTYPE)
        duration_logits = self.duration_logits.to(device=device, dtype=DTYPE)
        neg_inf = -torch.inf

        # Optional duration weighting
        dur_indices = torch.arange(1, Dmax + 1, device=device, dtype=DTYPE)
        if duration_weight > 0.0:
            dur_mean = (torch.softmax(duration_logits, dim=1) * dur_indices).sum(dim=1)
            dur_penalty = -((dur_indices.unsqueeze(0) - dur_mean.unsqueeze(1)) ** 2) / (2 * Dmax)
            dur_lp = (1 - duration_weight) * duration_logits + duration_weight * dur_penalty
        else:
            dur_lp = duration_logits

        # --- Batched emissions ---
        seq_batch, mask, log_batch, ctx_batch = X.to_batch(return_mask=True, include_log_probs=True, include_context=True)
        if theta is not None and hasattr(self, "_contextual_emission_pdf"):
            emission = self._contextual_emission_pdf(
                seq_batch if seq_batch.ndim == 3 else seq_batch.unsqueeze(-1),
                theta.to(device=device, dtype=DTYPE)
            )
            emission_log = emission.to(device=device, dtype=DTYPE)
        else:
            emission_log = log_batch.to(device=device, dtype=DTYPE)

        # Cumulative sum for emissions
        zero_pad = torch.zeros((B, 1, K), device=device, dtype=DTYPE)
        cumsum_emit = torch.cat([zero_pad, torch.cumsum(emission_log, dim=1)], dim=1)  # [B, T+1, K]

        # DP tensors
        V = torch.full((B, max_len, K), neg_inf, device=device, dtype=DTYPE)
        best_prev = torch.full((B, max_len, K), -1, dtype=torch.int64, device=device)
        best_dur = torch.zeros((B, max_len, K), dtype=torch.int64, device=device)
        dur_range = torch.arange(1, Dmax + 1, device=device, dtype=torch.long)

        # --- Fully vectorized DP ---
        for t in range(max_len):
            active_mask = torch.tensor([t < L for L in lengths], device=device)
            if not active_mask.any():
                continue
            active_idx = active_mask.nonzero(as_tuple=True)[0]
            B_active = active_idx.numel()
            max_d = min(Dmax, t + 1)
            durations = dur_range[:max_d]

            # Compute start indices
            starts = t - durations.unsqueeze(0) + torch.zeros((B_active, 1), device=device, dtype=torch.long)
            starts = starts.clamp(min=0)

            # Emission sums: [B_active, K, max_d]
            cumsum_active = cumsum_emit[active_idx]
            cumsum_starts = cumsum_active.gather(1, starts.unsqueeze(-1).expand(-1, -1, K))
            cumsum_end = cumsum_active[:, t + 1, :].unsqueeze(1).expand(-1, max_d, -1)
            emit_sums = (cumsum_end - cumsum_starts).permute(0, 2, 1)

            dur_scores = dur_lp[:, :max_d].unsqueeze(0).expand(B_active, -1, -1)

            if t == 0:
                scores = init_logits.unsqueeze(0).unsqueeze(2) + dur_scores + emit_sums
                prev_idx = torch.full((B_active, K, max_d), -1, dtype=torch.int64, device=device)
            else:
                prev_V = V[active_idx].gather(1, starts.unsqueeze(-1).expand(-1, -1, K))
                mask_start0 = (starts == 0)
                if mask_start0.any():
                    init_exp = init_logits.unsqueeze(0).unsqueeze(1).expand(B_active, max_d, K)
                    prev_V = torch.where(mask_start0.unsqueeze(-1), init_exp, prev_V)

                prev_V_exp = prev_V.unsqueeze(3)
                trans_exp = transition_logits.unsqueeze(0).unsqueeze(0)
                scores_plus_trans = prev_V_exp + trans_exp
                scores_max, argmax_prev = torch.max(scores_plus_trans, dim=2)
                scores = scores_max.permute(0, 2, 1) + dur_scores + emit_sums
                prev_idx = argmax_prev.permute(0, 2, 1).to(torch.int64)

            best_score, best_d_idx = torch.max(scores, dim=2)
            B_idx = active_idx.unsqueeze(1).expand(-1, K)
            K_idx = torch.arange(K, device=device).unsqueeze(0).expand(B_active, -1)
            V[B_idx, t, K_idx] = best_score
            best_dur[B_idx, t, K_idx] = durations[best_d_idx]
            best_prev[B_idx, t, K_idx] = prev_idx[torch.arange(B_active)[:, None], K_idx, best_d_idx]

        # --- Backtrace ---
        paths = []
        for b, L in enumerate(lengths):
            if L == 0:
                paths.append(torch.empty(0, dtype=torch.int64, device=device))
                continue
            t = L - 1
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

            # pad/truncate
            if seq_path.numel() < L:
                pad_len = L - seq_path.numel()
                fill_val = int(seq_path[-1].item()) if seq_path.numel() > 0 else 0
                seq_path = torch.cat([seq_path, torch.full((pad_len,), fill_val, dtype=torch.int64, device=device)])
            elif seq_path.numel() > L:
                seq_path = seq_path[:L]

            paths.append(seq_path)

        return paths

    @torch.no_grad()
    def _compute_log_likelihood(self, X: utils.Observations, theta: Optional[torch.Tensor] = None, verbose: bool = False) -> torch.Tensor:
        """
        GPU-vectorized HSMM log-likelihood computation with collapsed durations and optional context θ.
        Supports global [1,H], per-sequence [B,H], per-timestep [T_total,H], and per-batch-timestep [B,T,H] contexts.
        Returns per-sequence log-likelihoods and stores them in X.log_likelihoods.
        """
        device = self.device
        B = len(X.sequence)
        neg_inf = float("-inf")
        
        if B == 0:
            X.log_likelihoods = torch.full((0,), neg_inf, device=device, dtype=DTYPE)
            return X.log_likelihoods

        # --- Align context to batch & sequence ---
        T_max = max(X.lengths) if X.lengths else 0
        valid_theta = None
        if theta is not None:
            theta = theta.to(dtype=DTYPE, device=device)
            if theta.ndim == 1:  # global [H]
                valid_theta = theta.unsqueeze(0).unsqueeze(1).expand(B, T_max, -1)
            elif theta.ndim == 2:
                if theta.shape[0] == T_max:  # per timestep [T,H]
                    valid_theta = theta.unsqueeze(0).expand(B, -1, -1)
                elif theta.shape[0] == B:    # per sequence [B,H]
                    valid_theta = theta.unsqueeze(1).expand(B, T_max, -1)
                elif theta.shape[0] == 1:    # global singleton
                    valid_theta = theta.unsqueeze(1).expand(B, T_max, -1)
                else:
                    raise ValueError(f"Cannot align theta shape {theta.shape} with batch {B} and T_max {T_max}")
            elif theta.ndim == 3 and theta.shape[0] == B and theta.shape[1] == T_max:
                valid_theta = theta.clone()
            else:
                raise ValueError(f"Unsupported theta shape {theta.shape} for batch {B}, T_max {T_max}")

        # --- Compute alpha with optional context ---
        alpha_list = self._forward(X, theta=valid_theta)
        if alpha_list is None:
            raise RuntimeError("Forward pass returned None. Model may be uninitialized.")

        # --- Sequence lengths and offsets ---
        seq_lengths = torch.tensor([a.shape[0] for a in alpha_list], device=device, dtype=torch.long)
        cum_lengths = torch.zeros(B + 1, device=device, dtype=torch.long)
        cum_lengths[1:] = torch.cumsum(seq_lengths, dim=0)

        # --- Flatten all sequences: [total_T, K, Dmax] ---
        alpha_flat = torch.cat(alpha_list, dim=0)
        total_T, K, Dmax = alpha_flat.shape

        # --- Optional mask for collapsed durations ---
        if hasattr(X, "masks") and X.masks is not None:
            mask_flat = torch.cat(X.masks, dim=0).to(dtype=DTYPE, device=device)  # 1 for valid states/durations
            alpha_flat = alpha_flat + (mask_flat + EPS).log()

        # --- Optional context-modulated emissions (if alpha internally used log_probs) ---
        # Already handled in _forward; if not, user can supply precomputed X.log_probs

        # --- logsumexp over states and durations ---
        ll_flat = torch.logsumexp(alpha_flat, dim=(1,2))  # [total_T]

        # --- Gather last timestep log-likelihood per sequence ---
        last_idx = cum_lengths[1:] - 1
        ll = ll_flat.index_select(0, last_idx)

        # Store in Observations
        X.log_likelihoods = ll.detach().clone()

        if verbose:
            print(f"[compute_ll] total_timesteps={total_T}, sequences={B}, "
                  f"log-likelihoods min={ll.min():.4f}, max={ll.max():.4f}")

        return ll

    def get_model_params(
        self,
        X: Optional["Observations"] = None,
        theta: Optional[torch.Tensor] = None,
        mode: str = "estimate",
        inplace: bool = True,
    ) -> dict[str, Any]:
        """
        Unified parameter estimator and sampler for HSMM.

        Args:
            X: Observations object or None (for sampling from prior).
            theta: Optional context tensor [H], [T_total,H], [B,H], [B,T_max,H].
            mode: "estimate" (M-step) or "sample" (randomized).
            inplace: If True, update model attributes.

        Returns:
            Dict with 'init_logits', 'transition_logits', 'duration_logits', 'emission_pdf'.
        """
        device = self.device
        K, Dmax = self.n_states, self.max_duration

        # --- Align theta to flattened sequence length ---
        aligned_theta = None
        if theta is not None:
            if X is not None and isinstance(X, utils.Observations):
                total_len = sum(X.lengths)
                if theta.ndim == 1:  # [H]
                    aligned_theta = theta.unsqueeze(0).expand(total_len, -1)
                elif theta.ndim == 2:  # [T_total,H] or [B,H]
                    if theta.shape[0] == total_len:
                        aligned_theta = theta
                    elif theta.shape[0] == len(X.sequence):
                        aligned_theta = torch.cat([theta[i].expand(L, -1) for i, L in enumerate(X.lengths)], dim=0)
                    elif total_len % theta.shape[0] == 0:
                        aligned_theta = theta.repeat(total_len // theta.shape[0], 1)
                    else:
                        raise ValueError(f"Cannot align theta shape {tuple(theta.shape)} with length {total_len}")
                elif theta.ndim == 3:  # [B,T_max,H]
                    aligned_theta = torch.cat([theta[i, :L, :] for i, L in enumerate(X.lengths)], dim=0)
                else:
                    raise ValueError(f"Unsupported theta ndim {theta.ndim}")
                aligned_theta = aligned_theta.to(device, DTYPE)
            else:
                aligned_theta = theta.to(device, DTYPE)

        if mode == "sample":
            α = getattr(self, "alpha", 1.0)

            # Sample init, transition, duration logits
            init_logits = torch.log(constraints.sample_probs(α, (K,)).clamp_min(EPS)).to(dtype=DTYPE, device=device)
            transition_logits = torch.log(constraints.sample_transition(α, K, constraints.Transitions.SEMI).clamp_min(EPS)).to(dtype=DTYPE, device=device)
            duration_logits = torch.log(constraints.sample_probs(α, (K, Dmax)).clamp_min(EPS)).to(dtype=DTYPE, device=device)

            # --- Emission ---
            if X is not None:
                # Extract raw tensor if Observations
                if isinstance(X, utils.Observations):
                    all_X = torch.cat([seq for seq in X.sequence if seq.numel() > 0], dim=0).to(device, DTYPE)
                else:
                    all_X = X.to(device, DTYPE)

                emission_pdf = self.sample_emission_pdf(all_X, theta=aligned_theta)
                if torch.is_tensor(emission_pdf):
                    if emission_pdf.ndim == 2 and emission_pdf.shape[1] == K:
                        emission_pdf = Categorical(logits=emission_pdf)
                    else:
                        emission_pdf = torch.distributions.Independent(torch.distributions.Delta(emission_pdf), 1)
            else:
                emission_pdf = self._params.get("emission_pdf", None)

        elif mode == "estimate":
            if X is None or not isinstance(X, utils.Observations):
                raise RuntimeError("Must provide Observations X for M-step estimation.")

            gamma_list, xi_list, eta_list = self._compute_posteriors(X, theta=theta)

            def safe_sum(lst, dim=0):
                total = None
                for x in lst:
                    if x is not None and x.numel() > 0:
                        summed = x.sum(dim=dim)
                        total = summed if total is None else total + summed
                return total

            # --- Initial logits ---
            init_counts = safe_sum([g[0] for g in gamma_list if g is not None])
            new_init = (
                constraints.log_normalize(torch.clamp_min(torch.log(init_counts), EPS), dim=0)
                if init_counts is not None else self._init_logits.detach().clone()
            )

            # --- Transition logits ---
            trans_counts = safe_sum(xi_list)
            new_transition = (
                constraints.log_normalize(torch.clamp_min(torch.log(trans_counts), EPS), dim=1)
                if trans_counts is not None else self._transition_logits.detach().clone()
            )

            # --- Duration logits ---
            dur_counts = safe_sum(eta_list)
            new_duration = (
                constraints.log_normalize(torch.clamp_min(torch.log(dur_counts), EPS), dim=1)
                if dur_counts is not None else self._duration_logits.detach().clone()
            )

            # --- Emission ---
            all_X = torch.cat([seq for seq in X.sequence if seq.numel() > 0], dim=0).to(device, DTYPE)
            all_gamma = torch.cat([g for g in gamma_list if g is not None and g.numel() > 0], dim=0).to(device, DTYPE)
            pdf_or_tensor = self._estimate_emission_pdf(all_X, all_gamma, aligned_theta) if hasattr(self, "_estimate_emission_pdf") else self._params.get("emission_pdf")
            if torch.is_tensor(pdf_or_tensor):
                if pdf_or_tensor.ndim == 2 and pdf_or_tensor.shape[1] == K:
                    emission_pdf = Categorical(logits=pdf_or_tensor)
                else:
                    emission_pdf = torch.distributions.Independent(torch.distributions.Delta(pdf_or_tensor), 1)
            else:
                emission_pdf = pdf_or_tensor

        else:
            raise ValueError(f"Unsupported mode {mode}")

        if inplace:
            self._init_logits.data.copy_(init_logits if mode == "sample" else new_init)
            self._transition_logits.data.copy_(transition_logits if mode == "sample" else new_transition)
            self._duration_logits.data.copy_(duration_logits if mode == "sample" else new_duration)
            self._params["emission_pdf"] = emission_pdf

        return {
            "init_logits": init_logits if mode == "sample" else new_init,
            "transition_logits": transition_logits if mode == "sample" else new_transition,
            "duration_logits": duration_logits if mode == "sample" else new_duration,
            "emission_pdf": emission_pdf,
        }

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
        Supports global [1,H], per-sequence [B,H], per-timestep [T_total,H], and per-batch-timestep [B,T,H] contexts.
        """
        device = self.device

        # --- Optional emission initialization ---
        if sample_D_from_X:
            self._params['emission_pdf'] = self.sample_emission_pdf(X)

        # --- Encode observations if needed ---
        if theta is None and getattr(self, "encoder", None):
            theta = self.encode_observations(X)

        X_valid = self.to_observations(X, lengths, theta)

        # --- Batch info ---
        B = len(X_valid.sequence)
        T_max = max(X_valid.lengths) if X_valid.lengths else 0

        # --- Theta alignment ---
        if theta is not None:
            theta = theta.to(dtype=DTYPE, device=device)
            H = theta.shape[-1]
            valid_theta = torch.zeros((B, T_max, H), dtype=DTYPE, device=device)

            if theta.ndim == 1:  # global [H]
                valid_theta[:] = theta.unsqueeze(0)
            elif theta.ndim == 2:
                total_len = sum(X_valid.lengths)
                if theta.shape[0] == total_len:  # per-timestep [T_total, H]
                    offset = 0
                    for i, L in enumerate(X_valid.lengths):
                        if L > 0:
                            valid_theta[i, :L] = theta[offset:offset + L]
                            offset += L
                elif theta.shape[0] == B:  # per-sequence [B,H]
                    valid_theta[:] = theta.unsqueeze(1)
                elif theta.shape[0] == 1:  # global [1,H]
                    valid_theta[:] = theta.unsqueeze(1)
                else:
                    raise ValueError(f"Cannot align theta shape {tuple(theta.shape)} with batch {B} and max sequence length {T_max}")
            elif theta.ndim == 3:  # per-batch-timestep [B,T_max,H]
                for i, L in enumerate(X_valid.lengths):
                    if L > 0:
                        valid_theta[i, :L] = theta[i, :L]
            else:
                raise ValueError(f"Unsupported theta ndim {theta.ndim}")
        else:
            valid_theta = None

        # --- Convergence handler ---
        self.conv = ConvergenceHandler(
            tol=tol, max_iter=max_iter, n_init=n_init, post_conv_iter=post_conv_iter, verbose=verbose
        )

        best_score, best_state = -float("inf"), None

        # --- Flatten non-empty sequences ---
        nonempty_idx = [i for i, l in enumerate(X_valid.lengths) if l > 0]
        if nonempty_idx:
            seq_tensor = torch.cat([X_valid.sequence[i] for i in nonempty_idx], dim=0).to(dtype=DTYPE, device=device)
            seq_offsets = torch.cumsum(torch.tensor([0] + [X_valid.lengths[i] for i in nonempty_idx], device=device), dim=0)
        else:
            seq_tensor = torch.empty((0, X_valid.sequence[0].shape[-1] if X_valid.sequence else 0), device=device, dtype=DTYPE)
            seq_offsets = torch.tensor([0], device=device)

        def _squeeze_logits(t: torch.Tensor) -> torch.Tensor:
            return t.squeeze(0) if t.ndim == 3 and t.shape[0] == 1 else t

        # --- EM runs ---
        for run_idx in range(n_init):
            if verbose:
                print(f"\n=== Run {run_idx + 1}/{n_init} ===")

            # Controlled randomization
            g = torch.Generator(device=device).manual_seed(run_idx + 12345)

            if run_idx > 0:
                sampled = self.get_model_params(X_valid, valid_theta, mode="sample", inplace=True)
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

                # Compute log_probs for all sequences
                all_log_probs = pdf.log_prob(seq_tensor.unsqueeze(1))
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
                new_params = self.get_model_params(X_valid, valid_theta, mode="estimate", inplace=True)
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

        # --- Restore best run ---
        if best_state:
            self._init_logits.copy_(best_state['init_logits'])
            self._transition_logits.copy_(best_state['transition_logits'])
            self._duration_logits.copy_(best_state['duration_logits'])
            self._params['emission_pdf'] = best_state['emission_pdf']

        # --- Plot convergence ---
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
        Batched HSMM decode (Viterbi or approximate MAP) with durations.
        Supports variable-length sequences and context modulation.
        Returns list of CPU tensors with state sequences.
        """
        device = self.device
        Dmax, K = self.max_duration, self.n_states

        # --- Convert to Observations ---
        X_valid = self.to_observations(X, lengths)
        B = len(X_valid.lengths)
        max_T = max(X_valid.lengths) if X_valid.lengths else 0
        if max_T == 0:
            return [torch.empty(0, dtype=torch.long) for _ in range(B)]

        # --- Align context ---
        if context is not None:
            context = context.to(dtype=DTYPE, device=device)
            if context.ndim == 1:
                context = context.unsqueeze(0)
            elif context.ndim == 2 and context.shape[0] == 1:
                context = context
            elif context.ndim == 2 and context.shape[0] == sum(X_valid.lengths):
                # per timestep
                expanded = torch.zeros(B, max_T, context.shape[1], device=device, dtype=DTYPE)
                offset = 0
                for i, L in enumerate(X_valid.lengths):
                    if L > 0:
                        expanded[i, :L] = context[offset:offset + L]
                        offset += L
                context = expanded
            elif context.ndim == 3 and context.shape[0] == B and context.shape[1] == max_T:
                context = context.clone()
            else:
                raise ValueError(f"Unsupported context shape {context.shape}")
        valid_context = context if context is not None else None

        # --- Flatten observations ---
        seq_tensor = torch.cat(X_valid.sequence, dim=0).to(device=device, dtype=DTYPE) if X_valid.sequence else torch.empty((0, 0), device=device, dtype=DTYPE)
        offsets = [0] + list(torch.cumsum(torch.tensor(X_valid.lengths, device=device), dim=0).tolist())

        # --- Evaluate emissions in batch ---
        pdf = self._params.get("emission_pdf", None)
        if pdf is None or not hasattr(pdf, "log_prob"):
            raise RuntimeError("Emission PDF with log_prob() required for prediction.")

        all_log_probs = []
        if seq_tensor.numel() > 0:
            for start in range(0, seq_tensor.shape[0], batch_size):
                end = min(start + batch_size, seq_tensor.shape[0])
                chunk = seq_tensor[start:end].unsqueeze(1)
                chunk_context = None
                if valid_context is not None:
                    if valid_context.shape[0] == seq_tensor.shape[0]:
                        chunk_context = valid_context[start:end]
                    elif valid_context.shape[0] == 1:
                        chunk_context = valid_context
                lp = pdf.log_prob(chunk, theta=chunk_context) if chunk_context is not None else pdf.log_prob(chunk)
                if lp.ndim > 2:
                    lp = lp.flatten(start_dim=2).sum(dim=2)
                all_log_probs.append(lp)
            all_log_probs = torch.cat(all_log_probs, dim=0)
        else:
            all_log_probs = torch.empty((0, K), device=device, dtype=DTYPE)

        # --- Split per sequence ---
        X_valid.log_probs = [
            all_log_probs[offsets[i]:offsets[i+1]] if offsets[i] != offsets[i+1]
            else torch.empty((0, K), device=device, dtype=DTYPE)
            for i in range(B)
        ]

        # --- Model parameters ---
        init_logits_b = self._align_logits(getattr(self, "init_logits"), B)
        transition_logits = self._align_logits(getattr(self, "transition_logits"), B)
        duration_logits = self._align_logits(getattr(self, "duration_logits"), B)

        # --- Precompute cumulative emission sums ---
        cumsum_emit_list = []
        for lp in X_valid.log_probs:
            if lp.numel() == 0:
                cumsum_emit_list.append(torch.zeros((1, K), device=device, dtype=DTYPE))
            else:
                cumsum_emit_list.append(torch.cat([torch.zeros((1, K), device=device, dtype=DTYPE),
                                                   torch.cumsum(lp, dim=0)], dim=0))

        # --- DP tensors ---
        V = torch.full((B, max_T, K), -torch.inf, device=device, dtype=DTYPE)
        prev_state = torch.full((B, max_T, K), -1, dtype=torch.int64, device=device)
        prev_dur = torch.zeros((B, max_T, K), dtype=torch.int64, device=device)

        # --- Decode per sequence ---
        paths = []
        algo = algorithm.lower()
        for b in range(B):
            L = X_valid.lengths[b]
            if L == 0:
                paths.append(torch.empty(0, dtype=torch.long))
                continue

            emit_cumsum = cumsum_emit_list[b]
            trans_b = transition_logits[b]
            dur_b = duration_logits[b]
            init_b = init_logits_b[b]

            Vb = V[b]
            prev_b = prev_state[b]
            dur_store = prev_dur[b]

            for t in range(L):
                max_d = min(Dmax, t + 1)
                d_range = torch.arange(1, max_d + 1, device=device)

                starts = t - d_range + 1
                starts_clamped = starts.clamp(min=0)
                emit_sum = emit_cumsum[t + 1].unsqueeze(0) - emit_cumsum[starts_clamped]

                prev_scores = torch.stack([Vb[s - 1] if s > 0 else init_b for s in starts], dim=0)
                prev_plus_trans = prev_scores.unsqueeze(2) + trans_b.unsqueeze(0)

                if algo == "viterbi":
                    trans_term, argmax_j = torch.max(prev_plus_trans, dim=1)
                else:
                    trans_term = torch.logsumexp(prev_plus_trans, dim=1)
                    _, argmax_j = torch.max(prev_plus_trans, dim=1)

                dur_term = dur_b[:, d_range - 1].T
                scores_by_d = trans_term + dur_term + emit_sum

                if algo == "viterbi":
                    best_vals, best_d_idx = torch.max(scores_by_d, dim=0)
                else:
                    best_vals = torch.logsumexp(scores_by_d, dim=0)
                    _, best_d_idx = torch.max(scores_by_d, dim=0)

                Vb[t] = best_vals
                prev_b[t] = argmax_j[best_d_idx, torch.arange(K, device=device)]
                dur_store[t] = best_d_idx + 1

            # --- Backtrace ---
            t = L - 1
            cur_state = int(torch.argmax(Vb[t]).item())
            segments = []
            while t >= 0:
                d = int(dur_store[t, cur_state].item())
                if d <= 0:
                    break
                start = max(0, t - d + 1)
                segments.insert(0, torch.full((t - start + 1,), cur_state, dtype=torch.long, device=device))
                prev = int(prev_b[t, cur_state].item())
                t = start - 1
                cur_state = prev if prev >= 0 else cur_state

            seq_path = torch.cat(segments) if segments else torch.full((L,), 0, dtype=torch.long, device=device)
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
        device = self.device

        if B == 0 or max_T == 0:
            out = torch.zeros(B if by_sample else 1, dtype=DTYPE, device=device)
            return out.cpu()

        # --- Emission log-probabilities ---
        pdf = self._params.get("emission_pdf")
        if pdf is None or not hasattr(pdf, "log_prob"):
            raise RuntimeError("Emission PDF must implement log_prob().")

        all_seq = torch.cat(obs.sequence, dim=0).to(device=device, dtype=DTYPE)
        all_log_probs = pdf.log_prob(all_seq.unsqueeze(1))  # (total_T, K)

        # reshape back into padded batch [B, T, K]
        seq_offsets = torch.cat([torch.tensor([0], device=device),
                                 torch.cumsum(torch.tensor(obs.lengths, device=device), dim=0)])
        log_B = torch.full((B, max_T, K), -torch.inf, device=device, dtype=DTYPE)
        for i in range(B):
            s, e = seq_offsets[i].item(), seq_offsets[i+1].item()
            L = obs.lengths[i]
            log_B[i, :L] = all_log_probs[s:e]

        # --- Log transition/duration/init ---
        log_transition = F.log_softmax(self._transition_logits, dim=-1).to(device, DTYPE)
        log_duration = F.log_softmax(self._duration_logits, dim=-1).to(device, DTYPE)
        init_logits = self.init_logits.to(device, DTYPE)
        if init_logits.ndim == 1:
            init_logits = init_logits.expand(B, -1)

        # --- Forward recursion ---
        V = torch.full((B, max_T, K), -torch.inf, device=device, dtype=DTYPE)
        V[:, 0, :] = log_B[:, 0, :] + init_logits
        cumsum_B = torch.cat([torch.zeros((B, 1, K), device=device, dtype=DTYPE),
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

