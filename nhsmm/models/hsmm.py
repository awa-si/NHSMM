# nhsmm/models/hsmm.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Any, Literal, Dict
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, Categorical

from nhsmm.defaults import Emission, Duration, Transition, DTYPE, EPS, HSMMError, logger
from nhsmm import utils, constraints, SeedGenerator, ConvergenceMonitor
from nhsmm.context import ContextEncoder


class HSMM(nn.Module, ABC):
    """
    Hidden Semi-Markov Model (HSMM) base class.
    """

    def __init__(
        self,
        n_states: int,
        n_features: int,
        max_duration: int,
        seed: Optional[int] = None,
        transition_type: Any = None,
        alpha: Optional[float] = 1.0,
        emission_type: str = "gaussian",
        min_covar: Optional[float] = None,
        context_dim: Optional[int] = None,
        device: Optional[torch.device] = None,
        modulate_var: bool = False,
        init_emission: bool = True
    ):
        super().__init__()

        # --- Device & types ---
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transition_type = transition_type or constraints.Transitions.ERGODIC
        self.alpha = alpha if alpha is not None else 1.0
        self.emission_type = emission_type.lower()
        self.min_covar = min_covar or 1e-6
        self.max_duration = max_duration
        self.modulate_var = modulate_var
        self.context_dim = context_dim
        self.n_features = n_features
        self.n_states = n_states

        self._context: Optional[torch.Tensor] = None
        self.encoder: Optional[nn.Module] = None

        self._seed_gen = SeedGenerator(seed)
        self._params: Dict[str, Any] = {}

        # --- Modules & Buffers ---
        self._init_buffers()
        self._init_modules()

        # --- Optional emission PDF ---
        self._params['emission_pdf'] = None
        if init_emission:
            self._params['emission_pdf'] = self.sample_emission_pdf(None)

    def _reset_buffers(self):
        """Remove all buffers and shape attributes."""
        buffer_names = [
            "_init_logits", "_transition_logits", "_duration_logits",
            "_super_init_logits", "_super_transition_logits",
            "_init_prior_snapshot"
        ]
        for name in buffer_names:
            if hasattr(self, name):
                try:
                    self._buffers.pop(name, None)
                except AttributeError:
                    pass
                delattr(self, name)

        shape_names = [
            "_init_shape", "_transition_shape", "_duration_shape",
            "_super_init_shape", "_super_transition_shape"
        ]
        for name in shape_names:
            if hasattr(self, name):
                delattr(self, name)

    def _init_buffers(self, verbose: bool = True, max_resample: int = 5):
        """Initialize all HSMM buffers with collapse-safe sampling."""
        self._reset_buffers()  # prevent duplicate registration
        device = self.device

        def sample_probs_safe(shape):
            for attempt in range(max_resample):
                probs = constraints.sample_probs(self.alpha, shape, seed=self._seed_gen())
                probs = probs.clamp_min(self.min_covar)
                collapsed = (probs.max(dim=-1).values > 1.0 - 1e-3).nonzero(as_tuple=True)[0]
                if len(collapsed) == 0:
                    break
                if verbose:
                    logger.debug(f"Collapsed probs detected, resampling attempt {attempt+1}/{max_resample}")
            return torch.log(probs).to(device=device, dtype=DTYPE), probs.shape

        def sample_transitions_safe(n_states):
            for attempt in range(max_resample):
                probs = constraints.sample_transition(self.alpha, n_states, self.transition_type, seed=self._seed_gen())
                probs = probs.clamp_min(self.min_covar)
                probs /= probs.sum(dim=-1, keepdim=True).clamp_min(EPS)
                if constraints.is_valid_transition(probs, self.transition_type):
                    break
                if verbose:
                    logger.debug(f"Collapsed transition rows detected, resampling attempt {attempt+1}/{max_resample}")
            return torch.log(probs).to(device=device, dtype=DTYPE), probs.shape

        # Base HSMM buffers
        init_logits, init_shape = sample_probs_safe((self.n_states,))
        transition_logits, transition_shape = sample_transitions_safe(self.n_states)
        duration_logits, duration_shape = sample_probs_safe((self.n_states, self.max_duration))

        self.register_buffer("_init_logits", init_logits)
        self.register_buffer("_duration_logits", duration_logits)
        self.register_buffer("_transition_logits", transition_logits)

        self._init_shape = init_shape
        self._duration_shape = duration_shape
        self._transition_shape = transition_shape

        # Optional super-states
        n_super_states = getattr(self, "n_super_states", 0)
        if n_super_states > 1:
            super_init_logits, super_init_shape = sample_probs_safe((n_super_states,))
            super_duration_logits, super_duration_shape = sample_probs_safe((n_super_states,))
            super_transition_logits, super_transition_shape = sample_transitions_safe(n_super_states)
            self.register_buffer("_super_init_logits", super_init_logits)
            self.register_buffer("super_duration_logits", super_duration_logits)
            self.register_buffer("_super_transition_logits", super_transition_logits)
            self._super_init_shape = super_init_shape
            self.super_duration_shape = super_duration_shape
            self._super_transition_shape = super_transition_shape

        # Snapshot for monitoring
        summary = [init_logits.mean(), transition_logits.mean(), duration_logits.mean()]
        if n_super_states > 1:
            summary += [super_init_logits.mean(), super_transition_logits.mean()]
        self.register_buffer("_init_prior_snapshot", torch.stack(summary).detach())

        if verbose:
            logger.debug(f"Initialized buffers: init={init_shape}, transition={transition_shape}, duration={duration_shape}")
            if n_super_states > 1:
                logger.debug(f"Super-state buffers: init={super_init_shape}, transition={super_transition_shape}")

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
        )

        self.duration_module = Duration(
            n_states=self.n_states,
            max_duration=self.max_duration,
            context_dim=self.context_dim,
        )

        self.transition_module = Transition(
            n_states=self.n_states,
            context_dim=self.context_dim,
        )

    @property
    def seed(self) -> Optional[int]:
        return self._seed_gen.seed

    @property
    def init_logits(self) -> torch.Tensor:
        return self._init_logits

    @init_logits.setter
    def init_logits(self, logits: torch.Tensor, atol: float=1e-4):
        logits = logits.to(device=self._init_logits.device, dtype=DTYPE)

        if logits.shape != (self.n_states,):
            raise ValueError(f"init_logits must have shape ({self.n_states},) but got {tuple(logits.shape)}")

        logits = torch.clamp(logits, min=torch.log(torch.tensor(EPS, dtype=DTYPE, device=logits.device)))
        logits = logits - logits.logsumexp(dim=0)
        self._init_logits.copy_(logits)

        logger.debug(f"init_logits updated (mean={logits.mean():.4f}, max={logits.max():.4f})")

    @property
    def init_probs(self) -> torch.Tensor:
        return self._init_logits.softmax(0)

    @property
    def duration_logits(self) -> torch.Tensor:
        return self._duration_logits

    @duration_logits.setter
    def duration_logits(self, logits: torch.Tensor, atol: float=1e-4):
        logits = logits.to(device=self._duration_logits.device, dtype=DTYPE)

        if logits.shape != (self.n_states, self.max_duration):
            raise ValueError(f"duration_logits must have shape ({self.n_states},{self.max_duration})")

        logits = torch.clamp(logits, min=torch.log(torch.tensor(EPS, dtype=DTYPE, device=logits.device)))
        logits = logits - logits.logsumexp(dim=1, keepdim=True)
        self._duration_logits.copy_(logits)

        logger.debug(f"duration_logits updated (mean row logit={logits.mean():.4f}, max row logit={logits.max():.4f})")

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

        logits = logits - logits.logsumexp(dim=1, keepdim=True)
        logits = torch.clamp(logits, min=torch.log(torch.tensor(EPS, dtype=DTYPE, device=logits.device)))

        probs = logits.exp()
        if not constraints.is_valid_transition(probs, self.transition_type, atol=atol):
            logger.warning("[transition_logits] logits do not fully satisfy transition constraints, applying anyway.")
            probs = probs.clamp_min(EPS)
            logits = torch.log(probs)

        self._transition_logits.copy_(logits)
        logger.debug(f"transition_logits updated (mean={logits.mean():.4f}, max={logits.max():.4f})")

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
    def initialize_emissions(self, X: torch.Tensor, method: str = "moment") -> None:
        raise NotImplementedError("Subclasses must implement initialize_emissions().")

    @abstractmethod
    def _estimate_emission_pdf(self, X: torch.Tensor, posterior: torch.Tensor, theta: Optional[utils.ContextualVariables] = None) -> Distribution:
        raise NotImplementedError(
            "Subclasses must implement _estimate_emission_pdf. "
            "It should return a Distribution supporting `.log_prob` and handle optional context theta."
        )

    def attach_encoder(
        self,
        encoder: nn.Module,
        n_heads: int = 4,
        pool: Literal["mean", "last", "max", "attn", "mha"] = "mean",
    ) -> ContextEncoder:
        """Attach a ContextEncoder wrapper to the model."""
        self.encoder = ContextEncoder(
            pool=pool,
            n_heads=n_heads,
            encoder=encoder,
            device=self.device
        )
        return self.encoder

    def _validate(self, value: torch.Tensor, clamp: bool = False) -> torch.Tensor:
        """
        Validate and optionally clamp an observation tensor to the emission PDF support.
        Works for single-sequence inputs only (no batching).

        Args:
            value: Observation tensor of shape [T, F] or [F] (single time step)
            clamp: If True, values outside support are clamped to the nearest valid bound

        Returns:
            value tensor, aligned with PDF dtype/device and within support
        """
        pdf = self.pdf
        if pdf is None:
            raise RuntimeError(
                "Emission PDF not initialized. Call `sample_emission_pdf()` or `encode_observations()` first."
            )

        # Align device and dtype
        if hasattr(pdf, "mean") and isinstance(pdf.mean, torch.Tensor):
            value = value.to(device=pdf.mean.device, dtype=pdf.mean.dtype)

        # Flatten single time step to [1, F] if necessary
        event_shape = pdf.event_shape or ()
        if value.ndim == len(event_shape):
            value = value.unsqueeze(0)  # single time step

        # Validate support
        if hasattr(pdf.support, "check"):
            support_mask = pdf.support.check(value)
            if not torch.all(support_mask):
                if clamp:
                    # Prefer built-in clamp if available
                    if hasattr(pdf.support, "clamp"):
                        value = torch.where(support_mask, value, pdf.support.clamp(value))
                    else:
                        min_val = getattr(pdf.support, "lower_bound", -float("inf"))
                        max_val = getattr(pdf.support, "upper_bound", float("inf"))
                        value = value.clamp(min=min_val, max=max_val)
                else:
                    bad_vals = value[~support_mask].flatten().unique()
                    raise ValueError(f"Values outside PDF support detected: {bad_vals.tolist()}")

        # Validate event shape
        if event_shape and tuple(value.shape[-len(event_shape):]) != tuple(event_shape):
            raise ValueError(
                f"PDF event shape mismatch: expected {tuple(event_shape)}, got {tuple(value.shape[-len(event_shape):])}"
            )

        return value

    def encode_observations(
        self,
        X: torch.Tensor,
        pool: Optional[str] = None,
        detach_return: bool = True,
        store: bool = True,
    ) -> Optional[torch.Tensor]:
        """
        Encode observations X into a single context vector using the attached encoder.
        Batch handling is removed: all inputs are treated as a single sequence.

        Args:
            X: Input tensor of shape (F,) or (T,F)
            pool: Optional pooling mode override for the encoder
            store: If True, saves result to self._context
            detach_return: If True, detaches returned tensor to prevent gradient flow

        Returns:
            Context tensor of shape [1,H], or None if no encoder is attached
        """
        if self.encoder is None:
            if store:
                self._context = None
            return None

        device = next(self.encoder.parameters()).device
        X = X.to(device=device, dtype=DTYPE)

        if X.numel() == 0:
            raise ValueError(f"Input X is empty: shape {X.shape}")

        # Normalize to (1,T,F)
        if X.ndim == 1:       # (F,) -> (1,1,F)
            X = X.unsqueeze(0).unsqueeze(0)
        elif X.ndim == 2:     # (T,F) -> (1,T,F)
            X = X.unsqueeze(0)
        elif X.ndim != 3:
            raise ValueError(f"Expected input shape (F,) or (T,F), got {X.shape}")

        # Optional pooling override
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
            if hasattr(self.encoder, "pool"):
                self.encoder.pool = original_pool

        # Ensure device/dtype consistency
        if vec.device != device or vec.dtype != DTYPE:
            vec = vec.to(device=device, dtype=DTYPE)

        # Detach if requested
        if detach_return:
            vec = vec.detach()

        # Store single-sequence context
        if store:
            self._context = vec

        return vec

    def to_observations(
        self,
        X: torch.Tensor,
        theta: Optional[torch.Tensor] = None
    ) -> utils.Observations:
        """
        Converts raw input X (and optional context theta) into a structured Observations object.
        All inputs are treated as a single sequence; batch splitting is removed.

        Returns:
            utils.Observations(sequence, log_probs, lengths)
        """
        device = next(self.parameters(), torch.tensor(0.0)).device
        X_valid = self._validate(X).to(device=device, dtype=DTYPE)

        # ---- Align context ----
        if theta is None and hasattr(self, "_context") and self._context is not None:
            theta = self._context

        if theta is not None:
            theta = theta.to(device=device, dtype=DTYPE)
            # Broadcast context if needed
            if theta.ndim == 1:
                theta = theta.unsqueeze(0)
            if X_valid.ndim == 2 and theta.shape[0] == 1:
                theta = theta.expand(X_valid.shape[0], -1)
            elif X_valid.ndim == 2 and theta.shape[0] != X_valid.shape[0]:
                raise ValueError(f"Context length ({theta.shape[0]}) does not match X length ({X_valid.shape[0]})")
            elif X_valid.ndim == 3:
                # collapse time dimension to flatten
                X_valid = X_valid.reshape(-1, X_valid.shape[-1])
                if theta.shape[0] == 1:
                    theta = theta.expand(X_valid.shape[0], -1)
                elif theta.shape[0] != X_valid.shape[0]:
                    raise ValueError(f"Context length ({theta.shape[0]}) does not match flattened X length ({X_valid.shape[0]})")

        # ---- Compute log-probabilities ----
        logp_or_dist = self._contextual_emission_pdf(X_valid, theta)

        if isinstance(logp_or_dist, torch.distributions.Distribution):
            log_probs = logp_or_dist.log_prob(X_valid.unsqueeze(-2))
            if log_probs.ndim > 2:
                log_probs = log_probs.sum(dim=list(range(2, log_probs.ndim)))
        elif torch.is_tensor(logp_or_dist):
            log_probs = logp_or_dist
            if log_probs.ndim == 1:
                log_probs = log_probs.unsqueeze(-1)
        else:
            raise TypeError(f"Unsupported return type from _contextual_emission_pdf: {type(logp_or_dist)}")

        log_probs = log_probs.to(dtype=DTYPE, device=device)

        return utils.Observations(
            sequence=[X_valid],
            log_probs=[log_probs],
            lengths=[X_valid.shape[0]]
        )

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
        Compute context-modulated duration log-probabilities [K, Dmax] without batching.

        Tiny random jitter is added to avoid collapsed duration states during EM.
        """
        base_logits = self._duration_logits  # [K, Dmax]

        # Apply context if provided
        if theta is not None:
            log_duration = self.duration_module._apply_context(base_logits, theta)
            if not torch.is_tensor(log_duration):
                raise TypeError(f"Expected tensor from _apply_context, got {type(log_duration)}")
        else:
            log_duration = base_logits.clone()

        # Clean NaNs/Infs and add tiny jitter
        dtype_info = torch.finfo(log_duration.dtype)
        log_duration = torch.nan_to_num(log_duration, nan=dtype_info.min / 2, posinf=dtype_info.max / 2)
        log_duration = log_duration + 1e-8 * torch.rand_like(log_duration)

        # Normalize along duration dimension
        log_duration = log_duration - torch.logsumexp(log_duration, dim=-1, keepdim=True)

        return log_duration  # [K, Dmax]

    def _contextual_transition_matrix(self, theta: Optional[ContextualVariables] = None) -> torch.Tensor:
        """
        Compute context-modulated transition log-probabilities [K, K] (non-batched).

        Tiny random jitter is added to prevent collapsed transitions during EM.
        Structural transition constraints are enforced.
        """
        log_transition = self._transition_logits.clone()  # [K, K]

        if theta is not None:
            log_transition = self.transition_module._apply_context(log_transition, theta)
            if not torch.is_tensor(log_transition):
                raise TypeError(f"Expected tensor from _apply_context, got {type(log_transition)}")

        # Apply structural mask
        if hasattr(self, "transition_type"):
            mask = constraints.mask_invalid_transitions(self.n_states, self.transition_type).to(log_transition.device)
            log_transition = log_transition.masked_fill(~mask, float("-inf"))

        # Clean and normalize
        dtype_info = torch.finfo(log_transition.dtype)
        log_transition = torch.nan_to_num(log_transition, nan=dtype_info.min / 2, posinf=dtype_info.max / 2)
        log_transition = log_transition + 1e-8 * torch.rand_like(log_transition)
        log_transition = log_transition - torch.logsumexp(log_transition, dim=-1, keepdim=True)

        return log_transition  # [K, K]

    def get_model_params(
        self,
        X: Optional["Observations"] = None,
        theta: Optional[torch.Tensor] = None,
        mode: str = "estimate",
        inplace: bool = True,
    ) -> dict[str, Any]:
        """Return or sample HSMM parameters, fully collapse-safe including super-states."""
        device, K, Dmax = self.device, self.n_states, self.max_duration
        n_super = getattr(self, "n_super_states", 0)

        # --- Align theta ---
        aligned_theta = None
        if theta is not None:
            total_len = sum(X.lengths) if isinstance(X, utils.Observations) else None
            if theta.ndim == 1:
                aligned_theta = theta.unsqueeze(0).expand(total_len or 1, -1)
            elif theta.ndim == 2 and total_len:
                if theta.shape[0] == len(X.sequence):
                    aligned_theta = torch.cat([theta[i].expand(L, -1) for i, L in enumerate(X.lengths)], dim=0)
                elif theta.shape[0] == total_len:
                    aligned_theta = theta
                elif total_len % theta.shape[0] == 0:
                    aligned_theta = theta.repeat(total_len // theta.shape[0], 1)
            elif theta.ndim == 3 and total_len:
                aligned_theta = torch.cat([theta[i, :L, :] for i, L in enumerate(X.lengths)], dim=0)
            else:
                raise ValueError(f"Cannot align theta shape {tuple(theta.shape)}")
            aligned_theta = aligned_theta.to(device=device, dtype=DTYPE)

        # --- Collapse-safe samplers ---
        def sample_probs_safe(shape):
            for _ in range(5):
                p = constraints.sample_probs(getattr(self, "alpha", 1.0), shape)
                p = p.clamp_min(self.min_covar)
                if not (p.max(dim=-1).values < 1e-6).any():
                    break
            return torch.log(p).to(device=device, dtype=DTYPE), p.shape

        def sample_transition_safe(n):
            for _ in range(5):
                p = constraints.sample_transition(getattr(self, "alpha", 1.0), n, self.transition_type)
                p = p.clamp_min(self.min_covar)
                p = p / p.sum(dim=-1, keepdim=True).clamp_min(EPS)
                if constraints.is_valid_transition(p, self.transition_type):
                    break
            return torch.log(p).to(device=device, dtype=DTYPE), p.shape

        # --- Sample mode ---
        if mode == "sample":
            init_logits, _ = sample_probs_safe((K,))
            transition_logits, _ = sample_transition_safe(K)
            duration_logits, _ = sample_probs_safe((K, Dmax))

            if n_super > 1:
                super_init_logits, _ = sample_probs_safe((n_super,))
                super_transition_logits, _ = sample_transition_safe(n_super)

            if X is not None:
                all_X = torch.cat([s for s in X.sequence if s.numel() > 0], dim=0).to(device=device, dtype=DTYPE) \
                    if isinstance(X, utils.Observations) else X.to(device=device, dtype=DTYPE)
                emission_pdf = self.sample_emission_pdf(all_X, theta=aligned_theta)
                if torch.is_tensor(emission_pdf):
                    emission_pdf = Categorical(logits=emission_pdf) if emission_pdf.ndim == 2 and emission_pdf.shape[1] == K \
                        else torch.distributions.Independent(torch.distributions.Delta(emission_pdf), 1)
            else:
                emission_pdf = self._params.get("emission_pdf")

        # --- Estimate mode ---
        elif mode == "estimate":
            if not isinstance(X, utils.Observations):
                raise RuntimeError("Must provide Observations X for M-step estimation.")

            gamma_list, xi_list, eta_list = self._compute_posteriors(X, theta=theta)

            def safe_sum(lst, dim=0):
                out = [x.sum(dim=dim) for x in lst if x is not None and x.numel() > 0]
                return torch.stack(out).sum(0) if out else None

            init_counts = safe_sum([g[0] for g in gamma_list])
            trans_counts = safe_sum(xi_list)
            dur_counts = safe_sum(eta_list)

            # Resample collapsed states
            α = getattr(self, "alpha", 1.0)
            if init_counts is None or (init_counts < EPS).any():
                logger.debug("Resampling collapsed initial states")
                init_counts = (init_counts + constraints.sample_probs(α, (K,))) if init_counts is not None else constraints.sample_probs(α, (K,))
            if trans_counts is None or (trans_counts.sum(-1) < EPS).any():
                logger.debug("Resampling collapsed transition rows")
                trans_counts = (trans_counts + constraints.sample_transition(α, K, self.transition_type)) if trans_counts is not None else constraints.sample_transition(α, K, self.transition_type)
            if dur_counts is None or (dur_counts.sum(-1) < EPS).any():
                logger.debug("Resampling collapsed duration rows")
                dur_counts = (dur_counts + constraints.sample_probs(α, (K, Dmax))) if dur_counts is not None else constraints.sample_probs(α, (K, Dmax))

            new_init = constraints.log_normalize(torch.log(init_counts.add(EPS)), dim=0)
            new_transition = constraints.log_normalize(torch.log(trans_counts.add(EPS)), dim=1)
            new_duration = constraints.log_normalize(torch.log(dur_counts.add(EPS)), dim=1)

            # Super-state resampling
            if n_super > 1:
                super_init_counts = constraints.sample_probs(α, (n_super,))
                super_transition_counts = constraints.sample_transition(α, n_super, self.transition_type)
                super_init_logits = torch.log(super_init_counts).to(device=device, dtype=DTYPE)
                super_transition_logits = torch.log(super_transition_counts).to(device=device, dtype=DTYPE)

            all_X = torch.cat([s for s in X.sequence if s.numel() > 0], dim=0).to(device=device, dtype=DTYPE)
            all_gamma = torch.cat([g for g in gamma_list if g is not None and g.numel() > 0], dim=0).to(device=device, dtype=DTYPE)
            pdf_or_tensor = self._estimate_emission_pdf(all_X, all_gamma, aligned_theta) if hasattr(self, "_estimate_emission_pdf") else self._params.get("emission_pdf")
            if torch.is_tensor(pdf_or_tensor):
                emission_pdf = Categorical(logits=pdf_or_tensor) if pdf_or_tensor.ndim == 2 and pdf_or_tensor.shape[1] == K \
                    else torch.distributions.Independent(torch.distributions.Delta(pdf_or_tensor), 1)
            else:
                emission_pdf = pdf_or_tensor

        else:
            raise ValueError(f"Unsupported mode '{mode}'.")

        # --- Apply updates ---
        if inplace:
            if mode == "sample":
                self._init_logits.copy_(init_logits)
                self._transition_logits.copy_(transition_logits)
                self._duration_logits.copy_(duration_logits)
                if n_super > 1:
                    self._super_init_logits.copy_(super_init_logits)
                    self._super_transition_logits.copy_(super_transition_logits)
            else:
                self._init_logits.copy_(new_init)
                self._duration_logits.copy_(new_duration)
                self._transition_logits.copy_(new_transition)
                if n_super > 1:
                    self._super_init_logits.copy_(super_init_logits)
                    self._super_duration_logits.copy_(super_duration_logits)
                    self._super_transition_logits.copy_(super_transition_logits)
            self._params["emission_pdf"] = emission_pdf

        return {
            "init_logits": init_logits if mode == "sample" else new_init,
            "transition_logits": transition_logits if mode == "sample" else new_transition,
            "duration_logits": duration_logits if mode == "sample" else new_duration,
            "emission_pdf": emission_pdf,
            **(
                {
                    "super_init_logits": super_init_logits,
                    "super_duration_logits": super_duration_logits,
                    "super_transition_logits": super_transition_logits
                } if n_super > 1 else {}
            )
        }


    # hsmm.py HSMM precessing
    def _forward(self, X: utils.Observations, theta: Optional[ContextualVariables] = None) -> list[torch.Tensor]:
        K, Dmax, device, neg_inf = self.n_states, self.max_duration, self.device, -torch.inf

        min_prob: float = 1e-6
        resample: bool = True

        init_logits = self.init_logits.to(device=device, dtype=DTYPE)
        transition_logits = self.transition_logits.to(device=device, dtype=DTYPE)
        duration_logits = self.duration_logits.to(device=device, dtype=DTYPE)

        if theta is not None:
            transition_logits = self._contextual_transition_matrix(theta)[0]
            duration_logits = self._contextual_duration_pdf(theta)[0]

        # --- Clamp helper ---
        def clamp(logits, name):
            probs = logits.exp()
            if resample:
                collapsed = (probs.max(dim=-1).values > 1 - min_prob).nonzero(as_tuple=True)[0]
                if len(collapsed):
                    logger.error(f"[_forward] Collapsed {name} rows at indices {collapsed.tolist()}, applying EPS floor")
                    probs = probs.clamp(min=min_prob)
                    probs /= probs.sum(dim=-1, keepdim=True)
            return torch.log(probs.clamp_min(EPS))

        transition_logits = clamp(transition_logits, "transition")
        duration_logits = clamp(duration_logits, "duration")

        alpha_list = []
        for seq_probs, seq_len in zip(X.log_probs, X.lengths):
            if seq_len == 0:
                alpha_list.append(torch.full((0, K, Dmax), neg_inf, device=device, dtype=DTYPE))
                continue

            seq_probs = seq_probs.to(device=device, dtype=DTYPE)
            log_alpha = torch.full((seq_len, K, Dmax), neg_inf, device=device, dtype=DTYPE)
            cumsum_emit = torch.vstack((torch.zeros((1, K), dtype=DTYPE, device=device),
                                        torch.cumsum(seq_probs, dim=0)))

            max_d0 = min(Dmax, seq_len)
            emit_sums0 = (cumsum_emit[1:max_d0+1] - cumsum_emit[0]).T
            log_alpha[0, :, :max_d0] = init_logits.unsqueeze(1) + duration_logits[:, :max_d0] + emit_sums0

            for t in range(1, seq_len):
                max_dt = min(Dmax, t + 1)
                durations = torch.arange(1, max_dt + 1, device=device)
                starts = t - durations + 1
                end_cumsum = cumsum_emit[t+1].unsqueeze(0).expand(max_dt, K)
                start_cumsum = cumsum_emit[starts].T
                emit_sums_t = (end_cumsum.T - start_cumsum)
                idx = starts - 1
                prev_alpha_first = torch.where((starts == 0).unsqueeze(1),
                                               init_logits.unsqueeze(0),
                                               log_alpha[idx, :, 0])
                prev_alpha_sum = torch.logsumexp(prev_alpha_first.unsqueeze(2) + transition_logits.unsqueeze(0), dim=1).T
                log_alpha[t, :, :max_dt] = prev_alpha_sum + duration_logits[:, :max_dt] + emit_sums_t

            alpha_list.append(log_alpha)

        return alpha_list

    def _backward(self, X: utils.Observations, theta: Optional[ContextualVariables] = None) -> list[torch.Tensor]:
        K, Dmax, device, neg_inf = self.n_states, self.max_duration, self.device, -torch.inf

        transition_logits = self.transition_logits.to(device=device, dtype=DTYPE)
        duration_logits = self.duration_logits.to(device=device, dtype=DTYPE)
        if theta is not None:
            transition_logits = self._contextual_transition_matrix(theta)[0]
            duration_logits = self._contextual_duration_pdf(theta)[0]

        # Clamp small rows
        transition_logits = torch.log(transition_logits.exp().clamp_min(EPS))
        duration_logits = torch.log(duration_logits.exp().clamp_min(EPS))

        beta_list = []
        for seq_probs, seq_len in zip(X.log_probs, X.lengths):
            if seq_len == 0:
                beta_list.append(torch.full((0, K, Dmax), neg_inf, device=device, dtype=DTYPE))
                continue

            seq_probs = seq_probs.to(device=device, dtype=DTYPE)
            log_beta = torch.full((seq_len, K, Dmax), neg_inf, device=device, dtype=DTYPE)
            log_beta[-1] = 0.0

            cumsum_emit = torch.vstack((torch.zeros((1, K), dtype=DTYPE, device=device),
                                        torch.cumsum(seq_probs, dim=0)))

            for t in reversed(range(seq_len - 1)):
                max_dt = min(Dmax, seq_len - t)
                durations = torch.arange(1, max_dt + 1, device=device)
                ends = t + durations
                emit_sums = (cumsum_emit[ends] - cumsum_emit[t].unsqueeze(0)).T
                beta_next = log_beta[ends - 1, :, 0].T
                dur_scores = duration_logits[:, :max_dt]
                log_beta[t, :, 0] = torch.logsumexp(emit_sums + dur_scores + beta_next, dim=1)
                if max_dt > 1:
                    shift_len = min(max_dt - 1, seq_len - t - 1)
                    log_beta[t, :, 1:shift_len+1] = log_beta[t + 1, :, :shift_len] + seq_probs[t + 1].unsqueeze(-1)

            beta_list.append(log_beta)

        return beta_list

    def _compute_posteriors(self, X: utils.Observations, theta: Optional[ContextualVariables] = None):
        K, Dmax, device = self.n_states, self.max_duration, self.device

        alpha_list = self._forward(X, theta)
        beta_list = self._backward(X, theta)
        gamma_vec, xi_vec, eta_vec = [], [], []

        transition_logits = (self._contextual_transition_matrix(theta)[0] if theta is not None
                             else self.transition_logits.to(device=device, dtype=DTYPE))
        init_logits = self.init_logits.to(device=device, dtype=DTYPE)

        def clamp_tensor(tensor, name, dim=None):
            tensor = tensor.clamp_min(EPS)
            if dim is not None and (tensor.sum(dim=dim) < EPS).any():
                logger.error(f"[_compute_posteriors] Small {name} values detected, clamping")
            return tensor

        for seq_probs, seq_len, alpha, beta in zip(X.log_probs, X.lengths, alpha_list, beta_list):
            if seq_len == 0:
                gamma_vec.append(torch.zeros((0, K), dtype=DTYPE, device=device))
                eta_vec.append(torch.zeros((0, K, Dmax), dtype=DTYPE, device=device))
                xi_vec.append(torch.zeros((0, K, K), dtype=DTYPE, device=device))
                continue

            alpha, beta = alpha.to(device=device, dtype=DTYPE), beta.to(device=device, dtype=DTYPE)
            log_gamma = torch.logsumexp(alpha + beta, dim=2)
            gamma_vec.append(clamp_tensor(torch.softmax(log_gamma, dim=1), "gamma"))
            log_eta = alpha + beta
            eta_vec.append(clamp_tensor(torch.softmax(log_eta.reshape(seq_len, -1), dim=1).reshape(seq_len, K, Dmax), "eta"))

            if seq_len > 1:
                prev_alpha_sum = torch.logsumexp(alpha[:-1, :, :], dim=2)
                beta_next = torch.logsumexp(beta[1:, :, :], dim=2)
                log_xi = prev_alpha_sum.unsqueeze(2) + transition_logits.unsqueeze(0) + beta_next.unsqueeze(1)
                xi_vec.append(clamp_tensor(torch.softmax(log_xi.view(seq_len - 1, -1), dim=1).view(seq_len - 1, K, K), "xi", dim=2))
            else:
                xi_vec.append(torch.zeros((0, K, K), dtype=DTYPE, device=device))

        return gamma_vec, xi_vec, eta_vec

    def _map(self, X: utils.Observations) -> list[torch.Tensor]:
        """Vectorized MAP decoding of HSMM sequences from posterior state marginals."""
        gamma_list, _, _ = self._compute_posteriors(X)
        if gamma_list is None:
            raise RuntimeError("Posterior probabilities could not be computed — model parameters uninitialized.")

        seq_lengths = [g.shape[0] if g is not None else 0 for g in gamma_list]
        B = len(seq_lengths)
        if B == 0:
            return []

        max_T = max(seq_lengths)
        K = self.n_states
        neg_inf = torch.tensor(-float("inf"), device=self.device, dtype=DTYPE)

        if max_T == 0:
            return [torch.empty(0, dtype=torch.long, device=self.device) for _ in range(B)]

        # Prepare padded tensor
        gamma_padded = torch.full((B, max_T, K), neg_inf, device=self.device, dtype=DTYPE)
        for i, g in enumerate(gamma_list):
            if g is not None and g.numel() > 0:
                gamma_padded[i, :g.shape[0]] = torch.nan_to_num(g, nan=neg_inf, posinf=neg_inf, neginf=neg_inf)

        # MAP decoding: argmax along states
        map_padded = gamma_padded.argmax(dim=-1)

        # Split back into list per original sequence lengths
        map_sequences = [
            map_padded[i, :L].to(dtype=torch.long, device=self.device) if L > 0 else torch.empty(0, dtype=torch.long, device=self.device)
            for i, L in enumerate(seq_lengths)
        ]

        return map_sequences

    def _viterbi(
        self,
        X: utils.Observations,
        theta: Optional[torch.Tensor] = None,
        duration_weight: float = 0.0
    ) -> torch.Tensor:
        """
        Viterbi algorithm for a single HSMM sequence with optional context θ.
        Returns the most likely state sequence for that sequence.
        """
        device = self.device
        K, Dmax = self.n_states, self.max_duration
        L = X.lengths[0] if X.lengths else X.sequence[0].shape[0]
        if L == 0:
            return torch.empty(0, dtype=torch.int64, device=device)

        # --- Model parameters ---
        init_logits = self.init_logits.to(device=device, dtype=DTYPE)
        transition_logits = self.transition_logits.to(device=device, dtype=DTYPE)
        duration_logits = self.duration_logits.to(device=device, dtype=DTYPE)
        neg_inf = -torch.inf

        # Duration weighting
        dur_indices = torch.arange(1, Dmax + 1, device=device, dtype=DTYPE)
        if duration_weight > 0.0:
            dur_mean = (torch.softmax(duration_logits, dim=1) * dur_indices).sum(dim=1)
            dur_penalty = -((dur_indices - dur_mean.unsqueeze(1)) ** 2) / (2 * Dmax)
            dur_lp = (1 - duration_weight) * duration_logits + duration_weight * dur_penalty
        else:
            dur_lp = duration_logits

        # --- Emission log-probs ---
        seq = X.sequence[0].to(device=device, dtype=DTYPE)
        if theta is not None and hasattr(self, "_contextual_emission_pdf"):
            emission = self._contextual_emission_pdf(seq.unsqueeze(0), theta.to(device=device, dtype=DTYPE))
            emit_log = emission.squeeze(0)
        else:
            emit_log = X.log_probs[0].to(device=device, dtype=DTYPE)

        # --- DP arrays ---
        V = torch.full((L, K), neg_inf, device=device, dtype=DTYPE)
        best_prev = torch.full((L, K), -1, dtype=torch.int64, device=device)
        best_dur = torch.zeros((L, K), dtype=torch.int64, device=device)

        for t in range(L):
            max_d = min(Dmax, t + 1)
            durations = torch.arange(1, max_d + 1, device=device)
            scores_t = []

            for d_idx, d in enumerate(durations):
                start = t - d + 1
                emit_sum = emit_log[start:t + 1].sum(dim=0)  # sum over duration
                dur_score = dur_lp[:, d - 1]  # duration log-prob

                if start == 0:
                    scores = init_logits + dur_score + emit_sum
                    prev_idx = torch.full((K,), -1, dtype=torch.int64, device=device)
                else:
                    prev_V = V[start - 1] + transition_logits
                    scores, prev_idx = torch.max(prev_V, dim=1)
                    scores = scores + dur_score + emit_sum

                scores_t.append((scores, prev_idx))

            # Select best duration for each state
            scores_stack = torch.stack([s[0] for s in scores_t], dim=1)  # [K, max_d]
            best_score, best_d_idx = torch.max(scores_stack, dim=1)
            V[t] = best_score
            best_dur[t] = durations[best_d_idx]
            best_prev[t] = torch.stack([scores_t[i][1][k] for k, i in enumerate(best_d_idx)], dim=0)

        # --- Backtrace ---
        t = L - 1
        cur_state = int(torch.argmax(V[t]).item())
        segments = []

        while t >= 0:
            d = int(best_dur[t, cur_state].item())
            start = max(0, t - d + 1)
            segments.append((start, t, cur_state))
            prev_state = int(best_prev[t, cur_state].item())
            t = start - 1
            cur_state = prev_state if prev_state >= 0 else cur_state

        segments.reverse()
        seq_path = torch.cat([
            torch.full((end - start + 1,), st, dtype=torch.int64, device=device)
            for (start, end, st) in segments
        ])

        # Pad/truncate to match original length
        if seq_path.numel() < L:
            pad_len = L - seq_path.numel()
            fill_val = int(seq_path[-1].item()) if seq_path.numel() > 0 else 0
            seq_path = torch.cat([seq_path, torch.full((pad_len,), fill_val, dtype=torch.int64, device=device)])
        elif seq_path.numel() > L:
            seq_path = seq_path[:L]

        return seq_path

    def fit(
        self,
        X: torch.Tensor,
        n_init: int = 1,
        tol: float = 1e-4,
        max_iter: int = 15,
        post_conv_iter: int = 1,
        ignore_conv: bool = False,
        sample_D_from_X: bool = False,
        theta: Optional[torch.Tensor] = None,
        plot_conv: bool = False,
        verbose: bool = True,
    ):
        """EM fitting for a single-sequence HSMM with context-aware emissions."""
        X = X.to(self.device, dtype=DTYPE) if torch.is_tensor(X) else X

        if sample_D_from_X:
            self._params['emission_pdf'] = self.sample_emission_pdf(X, theta)

        if theta is None and getattr(self, "encoder", None):
            theta = self.encode_observations(X)

        # Convert to single-sequence Observations
        X_valid = self.to_observations(X, theta=theta)
        seq_tensor = X_valid.sequence[0].to(dtype=DTYPE, device=self.device)

        # Align context to sequence length
        valid_theta = None
        if theta is not None:
            if theta.ndim != 2 or theta.shape[0] != seq_tensor.shape[0]:
                raise ValueError(f"Theta must be 2D [T, D] and match sequence length ({seq_tensor.shape[0]}).")
            valid_theta = theta.to(self.device, dtype=DTYPE)

        self.conv = ConvergenceMonitor(tol=tol, max_iter=max_iter, n_init=n_init, post_conv_iter=post_conv_iter, verbose=verbose)
        best_score, best_state = -float("inf"), None

        for run_idx in range(n_init):
            if verbose:
                print(f"\n=== Run {run_idx + 1}/{n_init} ===")

            if run_idx > 0:
                self.get_model_params(X_valid, theta=valid_theta, mode="sample", inplace=True)

            # Initial log-likelihood
            base_ll = self._compute_log_likelihood(X_valid).sum()
            self.conv.push_pull(base_ll, 0, run_idx)

            for it in range(1, max_iter + 1):
                pdf = self._params.get('emission_pdf', None)
                if pdf is None or not hasattr(pdf, 'log_prob'):
                    raise RuntimeError("Emission PDF not initialized or incompatible with log_prob.")

                # Update emission log-probs
                X_valid.log_probs = [pdf.log_prob(seq_tensor.unsqueeze(1))]

                # Update model parameters via EM
                self.get_model_params(X_valid, theta=valid_theta, mode="estimate", inplace=True)

                curr_ll = self._compute_log_likelihood(X_valid).sum()
                converged = self.conv.push_pull(curr_ll, it, run_idx)
                if converged and not ignore_conv and verbose:
                    print(f"[Run {run_idx + 1}] Converged at iteration {it}.")
                    break

            # Save best state
            run_score = float(curr_ll.item())
            if run_score > best_score:
                best_score, best_state = run_score, {
                    'init_logits': self._init_logits.clone(),
                    'transition_logits': self._transition_logits.clone(),
                    'duration_logits': self._duration_logits.clone(),
                    'emission_pdf': copy.deepcopy(self._params['emission_pdf'])
                }

        # Restore best parameters
        if best_state:
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
        algorithm: Literal["map", "viterbi"] = "viterbi",
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode a single sequence for HSMM using MAP or Viterbi with safe duration handling.
        Batch logic removed; only one contiguous sequence is supported.
        """

        X_valid = self.to_observations(X)
        seq_tensor = X_valid.sequence[0].to(device=self.device, dtype=DTYPE)
        T = seq_tensor.shape[0]
        K = self.n_states

        # Align context to sequence length
        valid_context = None
        if context is not None:
            context = context.to(device=self.device, dtype=DTYPE)
            if context.ndim != 2 or context.shape[0] != T:
                raise ValueError(f"Context must be 2D [T, D] and match sequence length ({T})")
            valid_context = context

        # --- Emission log-probs ---
        pdf = getattr(self, "_params", {}).get("emission_pdf", None)
        if pdf is not None and hasattr(pdf, "log_prob"):
            log_probs = pdf.log_prob(seq_tensor.unsqueeze(1))
            if log_probs.ndim > 2:
                log_probs = log_probs.sum(dim=list(range(2, log_probs.ndim)))
        else:
            output = self.emission_module(valid_context)
            if isinstance(output, tuple):
                mu, var = output
                var = var.clamp(min=1e-6)
                dist = torch.distributions.Normal(mu, var.sqrt())
                log_probs = dist.log_prob(seq_tensor.unsqueeze(1)).sum(dim=-1)
            else:
                logits = output
                if seq_tensor.dtype != torch.long:
                    raise TypeError("seq_tensor must contain integer indices for categorical emission.")
                log_probs = F.log_softmax(logits, dim=-1)[torch.arange(T, device=self.device), seq_tensor.long()]

        # --- Transition logits ---
        transition_logits = torch.log_softmax(self.transition_module(valid_context), dim=-1) if hasattr(self, "transition_module") else self._transition_logits.to(self.device, DTYPE)

        # --- Duration logits ---
        if hasattr(self, "duration_module"):
            log_duration = self.duration_module.log_probs(context=valid_context) if hasattr(self.duration_module, "log_probs") else F.log_softmax(self.duration_module.logits, dim=-1)
        else:
            log_duration = getattr(self._params, "log_duration", None)
            if log_duration is not None:
                log_duration = log_duration.to(self.device, DTYPE)

        # --- Helper for safe duration indexing ---
        def dur_term(t: int) -> torch.Tensor:
            if log_duration is None:
                return torch.zeros(K, device=self.device, dtype=DTYPE)
            dur_idx = min(t, log_duration.shape[-1] - 1)
            if log_duration.ndim == 2:  # [K, Dmax]
                return log_duration[:, dur_idx]
            return log_duration[dur_idx]

        # --- Decoding ---
        algorithm = algorithm.lower()
        if algorithm == "map":
            alpha = torch.full((T, K), -torch.inf, device=self.device, dtype=DTYPE)
            alpha[0] = log_probs[0]
            for t in range(1, T):
                prev = alpha[t - 1].unsqueeze(1) + transition_logits
                prev += dur_term(t).unsqueeze(0)
                alpha[t] = torch.logsumexp(prev, dim=0) + log_probs[t]
            decoded = torch.argmax(alpha, dim=-1)

        elif algorithm == "viterbi":
            delta = torch.full((T, K), -torch.inf, device=self.device, dtype=DTYPE)
            psi = torch.zeros((T, K), dtype=torch.long, device=self.device)
            delta[0] = log_probs[0]

            for t in range(1, T):
                scores = delta[t - 1].unsqueeze(1) + transition_logits
                scores += dur_term(t).unsqueeze(0)
                psi[t] = torch.argmax(scores, dim=0)
                delta[t] = torch.max(scores, dim=0).values + log_probs[t]

            decoded = torch.zeros(T, dtype=torch.long, device=self.device)
            decoded[-1] = torch.argmax(delta[-1])
            for t in reversed(range(T - 1)):
                decoded[t] = psi[t + 1, decoded[t + 1]]

        else:
            raise ValueError(f"Unknown decoding algorithm '{algorithm}'.")

        return decoded.detach().cpu()

    @torch.no_grad()
    def _compute_log_likelihood(self, X: utils.Observations, theta: Optional[torch.Tensor] = None, verbose: bool = False) -> torch.Tensor:
        """
        Returns per-sequence log-likelihoods and stores them in X.log_likelihoods.
        """
        neg_inf = float("-inf")
        device = self.device
        B = len(X.sequence)
        
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

    def score(self, X: torch.Tensor, theta: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Duration-aware log-likelihood for a single sequence HSMM with optional context.
        Batch logic removed.
        """
        obs = self.to_observations(X, theta)
        seq = obs.sequence[0].to(dtype=DTYPE, device=self.device)
        T = seq.shape[0]
        K, Dmax = self.n_states, self.max_duration

        # -------- Emission log-probabilities --------
        pdf = self._params.get("emission_pdf")
        if pdf is None or not hasattr(pdf, "log_prob"):
            raise RuntimeError("Emission PDF must be initialized with log_prob method.")
        log_B = pdf.log_prob(seq.unsqueeze(1))  # (T, K)
        if log_B.ndim > 2:
            log_B = log_B.sum(dim=list(range(2, log_B.ndim)))

        # -------- Transition and duration --------
        log_transition = None
        if hasattr(self, "transition_module"):
            log_transition = self.transition_module.log_probs(context=theta) if hasattr(self.transition_module, "log_probs") else F.log_softmax(self.transition_module.logits, dim=-1)
        else:
            log_transition = F.log_softmax(self._transition_logits, dim=-1) if hasattr(self, "_transition_logits") else torch.full((K, K), 1.0 / K, device=self.device).log()

        log_duration = None
        if hasattr(self, "duration_module"):
            log_duration = self.duration_module.log_probs(context=theta) if hasattr(self.duration_module, "log_probs") else F.log_softmax(self.duration_module.logits, dim=-1)
        else:
            log_duration = F.log_softmax(self._duration_logits, dim=-1) if hasattr(self, "_duration_logits") else torch.zeros(K, Dmax, device=self.device, dtype=DTYPE)

        # -------- Forward algorithm (log-space) --------
        V = torch.full((T, K), -torch.inf, device=self.device, dtype=DTYPE)
        V[0] = log_B[0] + self.init_logits.to(device=self.device)

        for t in range(1, T):
            max_d = min(t + 1, Dmax)
            dur_idx = torch.arange(max_d, device=self.device)
            scores = []
            for d in dur_idx:
                prev = V[t - d]
                emit_sum = log_B[t - d + 1 : t + 1].sum(dim=0)
                dur_score = log_duration[:, d] if log_duration.ndim == 2 else log_duration[d]
                trans_score = prev.unsqueeze(1) + log_transition  # (K, K)
                trans_score = torch.logsumexp(trans_score, dim=0)
                scores.append(trans_score + emit_sum + dur_score)
            V[t] = torch.logsumexp(torch.stack(scores, dim=0), dim=0)

        # Sequence log-likelihood
        seq_ll = torch.logsumexp(V[-1], dim=-1)
        return seq_ll.detach().cpu()

    def info(
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

