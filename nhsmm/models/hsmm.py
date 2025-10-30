# nhsmm/models/hsmm.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Any, Literal, Dict
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, Categorical, MultivariateNormal, Independent, Laplace, StudentT

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
        min_covar: Optional[float] = 1e-6,
        context_dim: Optional[int] = None,
        device: Optional[torch.device] = None,
        modulate_var: bool = False,
        init_emission: bool = True
    ):
        super().__init__()

        self._seed_gen = SeedGenerator(seed)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transition_type = transition_type or constraints.Transitions.ERGODIC
        self.emission_type = emission_type
        self.max_duration = max_duration
        self.modulate_var = modulate_var
        self.context_dim = context_dim
        self.n_features = n_features
        self.min_covar = min_covar
        self.n_states = n_states
        self.alpha = alpha

        # self._context: Optional[torch.Tensor] = None
        # self.encoder: Optional[nn.Module] = None

        self._params: Dict[str, Any] = {'emission_pdf': None}

        # --- Modules & Buffers ---
        self._init_modules()
        self._init_buffers()

        # --- Optional emission PDF ---
        if init_emission:
            mvn = self.emission_module.initialize()
            self._params['emission_pdf'] = mvn
            self._emission_means.copy_(mvn.mean)
            self._emission_covs.copy_(mvn.covariance_matrix)

    def _reset_buffers(self):
        """Remove all buffers and shape attributes."""

        _buffers = [
            "_emission_means", "_emission_covs",
            "_init_logits", "_transition_logits", "_duration_logits",
            "_super_init_logits", "_super_transition_logits",
            "_init_prior_snapshot",
        ]
        for name in self._buffers:
            if hasattr(self, name):
                delattr(self, name)

        shapes = [
            "_init_shape",
            "_duration_shape",
            "_transition_shape",
            "_super_init_shape",
            "_super_duration_shape",
            "_super_transition_shape"
        ]
        for name in shapes:
            if hasattr(self, name):
                delattr(self, name)

    def _init_buffers(self, verbose: bool = True, max_resample: int = 5):
        """Initialize all HSMM buffers with collapse-safe sampling (single-sample)."""

        device = self.device
        self._reset_buffers()

        # Context buffer (optional)
        if not hasattr(self, "_context"):
            self.register_buffer("_context", None)

        # Emission buffers
        if not hasattr(self, "_emission_means"):
            self.register_buffer(
                "_emission_means",
                torch.zeros(self.n_states, self.n_features, dtype=DTYPE, device=device)
            )
        if not hasattr(self, "_emission_covs"):
            eye = torch.eye(self.n_features, dtype=DTYPE, device=device)
            self.register_buffer("_emission_covs", eye.unsqueeze(0).expand(self.n_states, -1, -1).clone())

        # ---------------- Helper functions ----------------
        def sample_probs_safe(shape):
            """Sample probability distributions safely with resampling."""
            for attempt in range(max_resample):
                probs = constraints.sample_probs(self.alpha, shape, seed=self._seed_gen())
                probs = probs.clamp_min(self.min_covar)
                collapsed = (probs.max(dim=-1).values > 1.0 - 1e-3).nonzero(as_tuple=True)[0]
                if len(collapsed) == 0:
                    break
                if verbose:
                    logger.debug(f"Collapsed probs detected, resampling attempt {attempt+1}/{max_resample}")
            return torch.log(probs.clamp_min(EPS)).to(device=device, dtype=DTYPE), probs.shape

        def sample_transitions_safe(n_states):
            """Sample transition matrix safely with resampling."""
            for attempt in range(max_resample):
                probs = constraints.sample_transition(self.alpha, n_states, self.transition_type, seed=self._seed_gen())
                probs = probs.clamp_min(self.min_covar)
                probs /= probs.sum(dim=-1, keepdim=True).clamp_min(EPS)
                if constraints.is_valid_transition(probs, self.transition_type):
                    break
                if verbose:
                    logger.debug(f"Collapsed transition rows detected, resampling attempt {attempt+1}/{max_resample}")
            return torch.log(probs.clamp_min(EPS)).to(device=device, dtype=DTYPE), probs.shape

        # ---------------- Sample base buffers ----------------
        init_logits, init_shape = sample_probs_safe((self.n_states,))
        transition_logits, transition_shape = sample_transitions_safe(self.n_states)
        duration_logits, duration_shape = sample_probs_safe((self.n_states, self.max_duration))

        # Register buffers if not existing
        if not hasattr(self, "_init_logits"):
            self.register_buffer("_init_logits", init_logits)
        if not hasattr(self, "_duration_logits"):
            self.register_buffer("_duration_logits", duration_logits)
        if not hasattr(self, "_transition_logits"):
            self.register_buffer("_transition_logits", transition_logits)

        self._init_shape = init_shape
        self._duration_shape = duration_shape
        self._transition_shape = transition_shape

        # ---------------- Optional super-states ----------------
        n_super_states = getattr(self, "n_super_states", 0)
        if n_super_states > 1:
            super_init_logits, super_init_shape = sample_probs_safe((n_super_states,))
            super_duration_logits, super_duration_shape = sample_probs_safe((n_super_states,))
            super_transition_logits, super_transition_shape = sample_transitions_safe(n_super_states)

            self.register_buffer("_super_init_logits", super_init_logits)
            self.register_buffer("_super_duration_logits", super_duration_logits)
            self.register_buffer("_super_transition_logits", super_transition_logits)

            self._super_init_shape = super_init_shape
            self._super_duration_shape = super_duration_shape
            self._super_transition_shape = super_transition_shape

        # ---------------- Snapshot for monitoring ----------------
        summary = [init_logits.mean(), transition_logits.mean(), duration_logits.mean()]
        if n_super_states > 1:
            summary += [super_init_logits.mean(), super_transition_logits.mean()]
        self.register_buffer("_init_prior_snapshot", torch.stack(summary).detach())

        if verbose:
            logger.debug(f"Initialized buffers: init={init_shape}, transition={transition_shape}, duration={duration_shape}")
            if n_super_states > 1:
                logger.debug(f"Super-state buffers: init={super_init_shape}, transition={super_transition_shape}, duration={super_duration_shape}")

    def _init_modules(self, seed: Optional[int] = None):
        """Initialize HSMM modules: emission, duration, and transition."""
        if seed is not None:
            torch.manual_seed(seed)

        device = self.device
        hidden_dim = getattr(self, "hidden_dim", None)

        self.emission_module = Emission(
            n_states=self.n_states,
            n_features=self.n_features,
            emission_type=self.emission_type,
            context_dim=self.context_dim,
            modulate_var=self.modulate_var,
            min_covar=self.min_covar,
            dof=getattr(self, "dof", 5.0),
            seed=getattr(self, "seed", 0),
        ).to(device)

        self.duration_module = Duration(
            n_states=self.n_states,
            max_duration=self.max_duration,
            context_dim=self.context_dim,
            hidden_dim=hidden_dim,
            temperature=getattr(self, "duration_temperature", 1.0),
            scale=getattr(self, "duration_scale", 1.0),
            debug=getattr(self, "debug", False),
        ).to(device)

        self.transition_module = Transition(
            n_states=self.n_states,
            context_dim=self.context_dim,
            hidden_dim=hidden_dim,
            temperature=getattr(self, "transition_temperature", 1.0),
            scale=getattr(self, "transition_scale", 1.0),
            debug=getattr(self, "debug", False),
        ).to(device)

        if getattr(self, "debug", False):
            logger.debug(
                f"HSMM modules initialized: emission ({self.emission_type}), duration, transition on device {device}"
            )

    @property
    def seed(self) -> Optional[int]:
        return self._seed_gen.seed

    @property
    def init_logits(self) -> torch.Tensor:
        return self._init_logits

    @init_logits.setter
    def init_logits(self, logits: torch.Tensor, atol: float = 1e-4):
        """
        Set initial state logits safely.

        Args:
            logits: tensor of shape [n_states], raw logits (any scale)
            atol: absolute tolerance for numerical clamping
        """
        # Ensure device/dtype match
        logits = logits.to(device=self._init_logits.device, dtype=DTYPE)

        # Validate shape
        if logits.shape != (self.n_states,):
            raise ValueError(f"init_logits must have shape ({self.n_states},) but got {tuple(logits.shape)}")

        # Clamp to avoid -inf or NaN
        min_log = torch.log(torch.tensor(EPS, dtype=DTYPE, device=logits.device))
        logits = torch.clamp(logits, min=min_log)

        # Normalize in log-space
        logits = logits - torch.logsumexp(logits, dim=0)

        # Avoid tiny differences causing instability
        if torch.any(torch.isnan(logits)) or torch.any(torch.isinf(logits)):
            raise RuntimeError("init_logits contains NaN or Inf after normalization.")

        # Copy into internal buffer
        self._init_logits.copy_(logits)

        # Optional debug info
        mean_val, max_val = logits.mean().item(), logits.max().item()
        logger.debug(f"init_logits updated: mean={mean_val:.4f}, max={max_val:.4f}")

    @property
    def duration_logits(self) -> torch.Tensor:
        return self._duration_logits

    @duration_logits.setter
    def duration_logits(self, logits: torch.Tensor, atol: float = 1e-4):
        """
        Set duration logits per state safely.

        Args:
            logits: tensor of shape [n_states, max_duration], raw logits
            atol: minimum clamp value to avoid -inf
        """
        # Ensure device/dtype consistency
        logits = logits.to(device=self._duration_logits.device, dtype=DTYPE)

        # Validate shape
        if logits.shape != (self.n_states, self.max_duration):
            raise ValueError(f"duration_logits must have shape ({self.n_states},{self.max_duration}) "
                             f"but got {tuple(logits.shape)}")

        # Clamp to avoid -inf
        min_log = torch.log(torch.tensor(EPS, dtype=DTYPE, device=logits.device))
        logits = torch.clamp(logits, min=min_log)

        # Normalize in log-space along duration axis
        logits = logits - torch.logsumexp(logits, dim=1, keepdim=True)

        # Safety check for NaN/Inf
        if torch.any(torch.isnan(logits)) or torch.any(torch.isinf(logits)):
            raise RuntimeError("duration_logits contains NaN or Inf after normalization.")

        # Copy into internal parameter
        self._duration_logits.copy_(logits)

        # Optional debug info
        mean_val, max_val = logits.mean().item(), logits.max().item()
        logger.debug(f"duration_logits updated (mean row logit={mean_val:.4f}, max row logit={max_val:.4f})")

    @property
    def transition_logits(self) -> torch.Tensor:
        return self._transition_logits

    @transition_logits.setter
    def transition_logits(self, logits: torch.Tensor, atol: float = 1e-4):
        """
        Set transition logits safely for HSMM.

        Args:
            logits: [n_states, n_states] raw transition logits
            atol: tolerance for validity checks
        """
        # Ensure device/dtype consistency
        logits = logits.to(device=self._transition_logits.device, dtype=DTYPE)

        # Validate shape
        if logits.shape != (self.n_states, self.n_states):
            raise ValueError(f"transition_logits must have shape ({self.n_states},{self.n_states}), "
                             f"got {tuple(logits.shape)}")

        # Log-space normalization along rows
        logits = logits - torch.logsumexp(logits, dim=1, keepdim=True)

        # Clamp to prevent -inf
        min_log = torch.log(torch.tensor(EPS, dtype=DTYPE, device=logits.device))
        logits = torch.clamp(logits, min=min_log)

        # Validate against transition constraints
        probs = logits.exp()
        if not constraints.is_valid_transition(probs, self.transition_type, atol=atol):
            logger.warning("[transition_logits] Logits violate transition constraints; applying clamp.")
            probs = probs.clamp_min(EPS)
            logits = torch.log(probs)
            # Re-normalize just in case
            logits = logits - torch.logsumexp(logits, dim=1, keepdim=True)

        # Copy into internal parameter
        self._transition_logits.copy_(logits)

        # Debug logging
        mean_val, max_val = logits.mean().item(), logits.max().item()
        logger.debug(f"transition_logits updated (mean={mean_val:.4f}, max={max_val:.4f})")

    @property
    def pdf(self) -> Any:
        return self._params.get('emission_pdf')

    @property
    @abstractmethod
    def dof(self) -> int:
        raise NotImplementedError

    @torch.no_grad()
    def compute_emission_pdf(
        self,
        X: Optional[torch.Tensor] = None,
        theta: Optional[torch.Tensor] = None,
        posterior: Optional[torch.Tensor] = None,
        emission_type: str = "gaussian",
        theta_scale: float = 0.1,
        mode: str = "estimate",
    ) -> Distribution:

        device = self.device
        K, F = self.n_states, self.n_features

        if mode == "sample":
            self.emission_module.to(device)
            pdf = self.emission_module.initialize(
                X=X, posterior=posterior, theta=theta, theta_scale=theta_scale
            )
            self._emission_means.copy_(self.emission_module._emission_means)
            self._emission_covs.copy_(self.emission_module._emission_covs)
            self._params["emission_pdf"] = pdf
            return pdf

        elif mode == "estimate":
            if X is None or posterior is None:
                raise ValueError("X and posterior must be provided for estimate mode.")

            X = X.to(dtype=DTYPE, device=device)
            posterior = posterior.to(dtype=DTYPE, device=device)

            # --- Weighted means ---
            weights_sum = posterior.sum(dim=0, keepdim=True).clamp_min(EPS)  # [1,K]
            means = (posterior.T @ X) / weights_sum.T                        # [K,F]

            # --- Context modulation ---
            if theta is not None:
                theta_tensor = theta.mean(dim=0, keepdim=True) if theta.ndim == 2 else theta
                means += theta_scale * theta_tensor.expand(K, -1)

            # --- Replace NaNs/Infs with fallback ---
            invalid_mask = ~torch.isfinite(means)
            if invalid_mask.any():
                fallback_mean = X.mean(dim=0, keepdim=True)
                means[invalid_mask] = fallback_mean.expand_as(means)[invalid_mask]
            means += 1e-6 * torch.randn_like(means)

            # --- Covariance / scale estimation ---
            eye_F = torch.eye(F, dtype=DTYPE, device=device)
            covs = torch.zeros(K, F, F, dtype=DTYPE, device=device)
            for k in range(K):
                w = posterior[:, k:k+1].clamp_min(EPS)
                diff = X - means[k:k+1]
                denom = w.sum()
                if denom < EPS:
                    covs[k] = eye_F * self.min_covar
                    continue
                C = (diff * w).T @ diff / denom
                # handle numerical issues
                if not torch.isfinite(C).all() or (C.trace() <= 0):
                    covs[k] = eye_F * self.min_covar
                else:
                    covs[k] = 0.5 * (C + C.T) + self.min_covar * eye_F

            self._emission_means.copy_(means)
            self._emission_covs.copy_(covs)

            # --- Construct distribution ---
            if emission_type == "gaussian":
                pdf = MultivariateNormal(loc=self._emission_means, covariance_matrix=self._emission_covs)

            elif emission_type == "laplace":
                scale = torch.sqrt(torch.diagonal(self._emission_covs, dim1=-2, dim2=-1)).clamp_min(self.min_covar)
                pdf = Independent(Laplace(self._emission_means, scale), 1)

            elif emission_type == "studentt":
                scale = torch.sqrt(torch.diagonal(self._emission_covs, dim1=-2, dim2=-1)).clamp_min(self.min_covar)
                pdf = Independent(StudentT(df=self.dof, loc=self._emission_means, scale=scale), 1)

            else:
                raise ValueError(f"Unsupported emission_type '{emission_type}'")

            self._params["emission_pdf"] = pdf
            return pdf

        else:
            raise ValueError(f"Unsupported mode '{mode}'")

    def get_model_params(
        self,
        X: Optional["Observations"] = None,
        theta: Optional[torch.Tensor] = None,
        mode: str = "estimate",
        inplace: bool = True,
        theta_scale: float = 0.1,
    ) -> dict[str, Any]:
        """
        Return or sample HSMM parameters for a single sequence, collapse-safe.

        Args:
            X: Observations object (required for 'estimate' mode)
            theta: Optional context vector [T, D] or [D]
            mode: 'sample' or 'estimate'
            inplace: If True, updates self._init/_transition/_duration_logits and emission_pdf
            theta_scale: scaling factor for sampled emission initialization

        Returns:
            dict[str, Any]: Logits and emission PDF
                Keys: 'init_logits', 'transition_logits', 'duration_logits', 'emission_pdf'
                If n_super_states > 1, adds 'super_init_logits' and 'super_transition_logits'
        """
        device, K, Dmax = self.device, self.n_states, self.max_duration
        n_super = getattr(self, "n_super_states", 0)

        # --- Align theta for single sequence ---
        def align_theta_single(theta, seq_len):
            if theta is None:
                return None
            if theta.ndim == 1:
                return theta.unsqueeze(0).expand(seq_len, -1).to(device=device, dtype=DTYPE)
            elif theta.ndim == 2:
                if theta.shape[0] == seq_len:
                    return theta.to(device=device, dtype=DTYPE)
                elif theta.shape[0] == 1:
                    return theta.expand(seq_len, -1).to(device=device, dtype=DTYPE)
            raise ValueError(f"Cannot align theta of shape {tuple(theta.shape)} with sequence length {seq_len}")

        seq_len = sum(X.lengths) if X is not None else 1
        aligned_theta = align_theta_single(theta, seq_len)

        # --- Helper: collapse-safe sampling ---
        def sample_probs_safe(shape):
            for _ in range(5):
                p = constraints.sample_probs(getattr(self, "alpha", 1.0), shape).clamp_min(self.min_covar)
                if not (p.max(dim=-1).values < 1e-6).any():
                    break
            return torch.log(p).to(device=device, dtype=DTYPE)

        def sample_transition_safe(n):
            for _ in range(5):
                p = constraints.sample_transition(getattr(self, "alpha", 1.0), n, self.transition_type)
                p = p.clamp_min(self.min_covar)
                p = p / p.sum(dim=-1, keepdim=True).clamp_min(EPS)
                if constraints.is_valid_transition(p, self.transition_type):
                    break
            return torch.log(p).to(device=device, dtype=DTYPE)

        # --- Concatenate sequence for emission ---
        all_X = None
        if X is not None:
            all_X = torch.cat([s for s in X.sequence if s.numel() > 0], dim=0).to(device=device, dtype=DTYPE)

        # --- Sample mode ---
        if mode == "sample":
            init_logits = sample_probs_safe((K,))
            transition_logits = sample_transition_safe(K)
            duration_logits = sample_probs_safe((K, Dmax))

            if n_super > 1:
                super_init_logits = sample_probs_safe((n_super,))
                super_transition_logits = sample_transition_safe(n_super)

            emission_pdf = self.compute_emission_pdf(
                X=all_X, posterior=None, theta=aligned_theta, theta_scale=theta_scale, mode="sample"
            )

        # --- Estimate mode ---
        elif mode == "estimate":
            if X is None or not isinstance(X, utils.Observations):
                raise RuntimeError("Must provide Observations X for M-step estimation.")

            gamma_list, xi_list, eta_list = self._compute_posteriors(X, theta=aligned_theta)

            def safe_sum(lst, dim=0):
                out = [x.sum(dim=dim) for x in lst if x is not None and x.numel() > 0]
                return torch.stack(out).sum(0) if out else None

            init_counts = safe_sum([g[0] for g in gamma_list])
            trans_counts = safe_sum(xi_list)
            dur_counts = safe_sum(eta_list)

            α = getattr(self, "alpha", 1.0)
            if init_counts is None or (init_counts < EPS).any():
                init_counts = init_counts + constraints.sample_probs(α, (K,)) if init_counts is not None else constraints.sample_probs(α, (K,))
            if trans_counts is None or (trans_counts.sum(-1) < EPS).any():
                trans_counts = trans_counts + constraints.sample_transition(α, K, self.transition_type) if trans_counts is not None else constraints.sample_transition(α, K, self.transition_type)
            if dur_counts is None or (dur_counts.sum(-1) < EPS).any():
                dur_counts = dur_counts + constraints.sample_probs(α, (K, Dmax)) if dur_counts is not None else constraints.sample_probs(α, (K, Dmax))

            init_logits = constraints.log_normalize(torch.log(init_counts + EPS), dim=0)
            transition_logits = constraints.log_normalize(torch.log(trans_counts + EPS), dim=1)
            duration_logits = constraints.log_normalize(torch.log(dur_counts + EPS), dim=1)

            if n_super > 1:
                super_init_logits = sample_probs_safe((n_super,))
                super_transition_logits = sample_transition_safe(n_super)

            all_gamma = torch.cat([g for g in gamma_list if g is not None and g.numel() > 0], dim=0)
            emission_pdf = self.compute_emission_pdf(
                X=all_X, posterior=all_gamma, theta=aligned_theta, theta_scale=theta_scale, mode="estimate"
            )

        else:
            raise ValueError(f"Unsupported mode '{mode}'.")

        # --- Apply inplace updates ---
        if inplace:
            self._init_logits.copy_(init_logits)
            self._transition_logits.copy_(transition_logits)
            self._duration_logits.copy_(duration_logits)
            if n_super > 1:
                self._super_init_logits.copy_(super_init_logits)
                self._super_transition_logits.copy_(super_transition_logits)
            self._params["emission_pdf"] = emission_pdf

        # --- Return dict ---
        out = {
            "init_logits": init_logits,
            "transition_logits": transition_logits,
            "duration_logits": duration_logits,
            "emission_pdf": emission_pdf,
        }
        if n_super > 1:
            out.update({
                "super_init_logits": super_init_logits,
                "super_transition_logits": super_transition_logits
            })

        return out

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
            if store: self._context = None
            return None

        device = self.device
        X = X.to(device=device, dtype=DTYPE)

        if X.numel() == 0:
            raise ValueError(f"Input X is empty: shape {X.shape}")

        # Normalize to (1,T,F)
        if X.ndim == 1:
            X = X.unsqueeze(0).unsqueeze(0)
        elif X.ndim == 2:
            X = X.unsqueeze(0)
        elif X.ndim != 3:
            raise ValueError(f"Expected input shape (F,) or (T,F), got {X.shape}")

        # Optional pooling override
        original_pool = getattr(self.encoder, "pool", None)
        if pool is not None and hasattr(self.encoder, "pool"):
            self.encoder.pool = pool

        try:
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

        # Ensure device/dtype and optionally detach
        vec = vec.to(device=device, dtype=DTYPE)
        if detach_return:
            vec = vec.detach()

        # Store single-sequence context
        if store:
            self._context = vec

        return vec

    def to_observations(self, X: torch.Tensor, theta: Optional[torch.Tensor] = None) -> utils.Observations:
        """
        Converts raw input X (and optional context theta) into a structured Observations object.
        All inputs are treated as a single sequence; batch splitting is removed.

        Returns:
            utils.Observations(sequence, log_probs, lengths)
        """
        device = self.device
        X_valid = self._validate(X).to(device=device, dtype=DTYPE)

        # ---- Align context ----
        theta = theta if theta is not None else getattr(self, "_context", None)
        if theta is not None:
            theta = theta.to(device=device, dtype=DTYPE)
            if theta.ndim == 1:
                theta = theta.unsqueeze(0)
            if theta.shape[0] != X_valid.shape[0]:
                if theta.shape[0] == 1:
                    theta = theta.expand(X_valid.shape[0], -1)
                else:
                    raise ValueError(f"Context length ({theta.shape[0]}) does not match X length ({X_valid.shape[0]})")

        # ---- Compute log-probabilities ----
        logp_or_dist = self._contextual_emission_pdf(X_valid, theta)

        if isinstance(logp_or_dist, torch.distributions.Distribution):
            # Unsqueeze -2 for state dimension; sum across extra dims if present
            log_probs = logp_or_dist.log_prob(X_valid.unsqueeze(-2))
            if log_probs.ndim > 2:
                log_probs = log_probs.flatten(start_dim=2).sum(dim=2)
        elif torch.is_tensor(logp_or_dist):
            log_probs = logp_or_dist.unsqueeze(-1) if logp_or_dist.ndim == 1 else logp_or_dist
        else:
            raise TypeError(f"Unsupported return type from _contextual_emission_pdf: {type(logp_or_dist)}")

        log_probs = log_probs.to(dtype=DTYPE, device=device)

        return utils.Observations(
            sequence=[X_valid],
            log_probs=[log_probs],
            lengths=[X_valid.shape[0]]
        )


    # Context
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


    # hsmm.py HSMM precessing
    def _forward(self, X: utils.Observations, theta: Optional[ContextualVariables] = None) -> list[torch.Tensor]:

        K, Dmax, device = self.n_states, self.max_duration, self.device
        neg_inf = torch.finfo(DTYPE).min / 2.0

        # --- Module-based logits ---
        init_logits = self.init_logits.to(device=device, dtype=DTYPE)

        if hasattr(self, "duration_module"):
            dur_logits = self.duration_module.log_matrix(context=theta)  # [K, Dmax]
        else:
            dur_logits = self.duration_logits.to(device=device, dtype=DTYPE)

        if hasattr(self, "transition_module"):
            trans_logits = self.transition_module.log_matrix(context=theta)  # [K, K]
        else:
            trans_logits = self.transition_logits.to(device=device, dtype=DTYPE)

        # Safe normalization to avoid degenerate rows
        def _safe_log_rows(logits: torch.Tensor, min_prob: float = 1e-6):
            probs = logits.exp().clamp_min(min_prob)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(EPS)
            return torch.log(probs)

        trans_logits = _safe_log_rows(trans_logits)
        dur_logits = _safe_log_rows(dur_logits)

        alpha_list = []

        for seq_logp, seq_len in zip(X.log_probs, X.lengths):
            if seq_len == 0:
                alpha_list.append(torch.full((0, K, Dmax), neg_inf, device=device, dtype=DTYPE))
                continue

            seq_logp = seq_logp.to(device=device, dtype=DTYPE)
            log_alpha = torch.full((seq_len, K, Dmax), neg_inf, device=device, dtype=DTYPE)

            # Precompute cumulative emission sums
            cumsum_emit = torch.vstack((torch.zeros((1, K), dtype=DTYPE, device=device),
                                        torch.cumsum(seq_logp, dim=0)))

            # --- t = 0 initialization ---
            max_d0 = min(Dmax, seq_len)
            emit_sums0 = (cumsum_emit[1:max_d0 + 1] - cumsum_emit[0]).T  # [K, max_d0]
            log_alpha[0, :, :max_d0] = init_logits.unsqueeze(1) + dur_logits[:, :max_d0] + emit_sums0

            # --- recursion t ≥ 1 ---
            for t in range(1, seq_len):
                max_dt = min(Dmax, t + 1)
                starts = t - torch.arange(max_dt, device=device)  # segment start indices
                emit_sums = (cumsum_emit[t + 1] - cumsum_emit[starts]).T  # [K, max_dt]

                prev_alpha = torch.full((max_dt, K), neg_inf, device=device, dtype=DTYPE)
                idx_prev = starts - 1
                valid_mask = idx_prev >= 0
                if valid_mask.any():
                    prev_vals = torch.logsumexp(log_alpha[idx_prev[valid_mask].long(), :, :], dim=2)
                    prev_alpha[valid_mask] = prev_vals

                start0_mask = starts == 0
                if start0_mask.any():
                    prev_alpha[start0_mask] = init_logits

                # Transition + duration + emission
                log_alpha[t, :, :max_dt] = (
                    torch.logsumexp(prev_alpha.unsqueeze(2) + trans_logits.unsqueeze(0), dim=1).T
                    + dur_logits[:, :max_dt]
                    + emit_sums
                )

            alpha_list.append(log_alpha)

        return alpha_list

    def _backward(self, X: utils.Observations, theta: Optional[ContextualVariables] = None) -> list[torch.Tensor]:

        K, Dmax, device = self.n_states, self.max_duration, self.device
        neg_inf = torch.finfo(DTYPE).min / 2.0

        # --- Module-based logits ---
        if hasattr(self, "duration_module"):
            dur_logits = self.duration_module.log_matrix(context=theta)
        else:
            dur_logits = self.duration_logits.to(device=device, dtype=DTYPE)

        if hasattr(self, "transition_module"):
            trans_logits = self.transition_module.log_matrix(context=theta)
        else:
            trans_logits = self.transition_logits.to(device=device, dtype=DTYPE)

        # Safe normalization
        def _safe_log_rows(logits: torch.Tensor):
            probs = logits.exp().clamp_min(EPS)
            probs /= probs.sum(dim=-1, keepdim=True).clamp_min(EPS)
            return torch.log(probs.clamp_min(EPS))

        trans_logits = _safe_log_rows(trans_logits)
        dur_logits = _safe_log_rows(dur_logits)

        beta_list = []

        for seq_logp, seq_len in zip(X.log_probs, X.lengths):
            if seq_len == 0:
                beta_list.append(torch.full((0, K, Dmax), neg_inf, device=device, dtype=DTYPE))
                continue

            seq_logp = seq_logp.to(device=device, dtype=DTYPE)
            log_beta = torch.full((seq_len, K, Dmax), neg_inf, device=device, dtype=DTYPE)
            log_beta[-1] = 0.0  # terminal step

            # Precompute cumulative emission sums
            cumsum_emit = torch.vstack((torch.zeros((1, K), device=device, dtype=DTYPE),
                                        torch.cumsum(seq_logp, dim=0)))

            # --- backward recursion ---
            for t in reversed(range(seq_len - 1)):
                max_dt = min(Dmax, seq_len - t)
                durations = torch.arange(1, max_dt + 1, device=device)
                ends = t + durations

                emit_sums = (cumsum_emit[ends] - cumsum_emit[t].unsqueeze(0)).T  # [K, max_dt]
                dur_scores = dur_logits[:, :max_dt]  # [K, max_dt]

                # β for next segments using transitions
                beta_next = torch.logsumexp(
                    trans_logits.unsqueeze(0) + log_beta[ends - 1, :, 0].unsqueeze(1), dim=2
                ).T  # [K, max_dt]

                # Combine emission, duration, and next β
                scores = emit_sums + dur_scores + beta_next
                log_beta[t, :, 0] = torch.logsumexp(scores, dim=1)

                # Within-segment propagation
                if max_dt > 1:
                    shift_len = min(max_dt - 1, seq_len - t - 1)
                    log_beta[t, :, 1:shift_len + 1] = log_beta[t + 1, :, :shift_len] + seq_logp[t + 1].unsqueeze(-1)

            beta_list.append(log_beta)

        return beta_list

    def _compute_posteriors(self, X: utils.Observations, theta: Optional[ContextualVariables] = None) -> Tuple:

        K, Dmax, device = self.n_states, self.max_duration, self.device

        alpha_list = self._forward(X, theta)
        beta_list = self._backward(X, theta)

        gamma_vec, xi_vec, eta_vec = [], [], []

        # --- Base logits (module-aware) ---
        if hasattr(self, "transition_module"):
            transition_logits = self.transition_module.log_matrix(context=theta)
        else:
            transition_logits = self.transition_logits.to(device=device, dtype=DTYPE)

        if hasattr(self, "init_logits"):
            init_logits = torch.log_softmax(self.init_logits.to(device=device, dtype=DTYPE), dim=0)
        else:
            init_logits = torch.zeros(K, device=device, dtype=DTYPE)

        # Ensure transition matrix is normalized in log-space
        transition_logits = torch.log_softmax(transition_logits, dim=-1)

        for seq_probs, seq_len, alpha, beta in zip(X.log_probs, X.lengths, alpha_list, beta_list):
            if seq_len == 0:
                gamma_vec.append(torch.zeros((0, K), dtype=DTYPE, device=device))
                eta_vec.append(torch.zeros((0, K, Dmax), dtype=DTYPE, device=device))
                xi_vec.append(torch.zeros((0, K, K), dtype=DTYPE, device=device))
                continue

            alpha, beta = alpha.to(device=device, dtype=DTYPE), beta.to(device=device, dtype=DTYPE)

            # --- γ: posterior over states ---
            log_gamma = torch.logsumexp(alpha + beta, dim=2)  # [T, K]
            gamma = torch.softmax(log_gamma, dim=1).clamp_min(EPS)
            gamma_vec.append(gamma)

            # --- η: posterior over state-duration pairs ---
            log_eta = alpha + beta  # [T, K, Dmax]
            eta = torch.softmax(log_eta.reshape(seq_len, -1), dim=1).reshape(seq_len, K, Dmax).clamp_min(EPS)
            eta_vec.append(eta)

            # --- ξ: posterior over transitions ---
            if seq_len > 1:
                prev_alpha = torch.logsumexp(alpha[:-1], dim=2)  # [T-1, K]
                next_beta = torch.logsumexp(beta[1:], dim=2)     # [T-1, K]

                log_xi = prev_alpha.unsqueeze(2) + transition_logits.unsqueeze(0) + next_beta.unsqueeze(1)  # [T-1,K,K]
                xi = torch.softmax(log_xi.reshape(seq_len - 1, -1), dim=1).reshape(seq_len - 1, K, K).clamp_min(EPS)
                xi_vec.append(xi)
            else:
                xi_vec.append(torch.zeros((0, K, K), dtype=DTYPE, device=device))

        return gamma_vec, xi_vec, eta_vec

    def _map(self, X: utils.Observations) -> list[torch.Tensor]:
        """MAP decoding of a single HSMM sequence from posterior state marginals."""
        gamma_list, _, _ = self._compute_posteriors(X)
        if not gamma_list or gamma_list[0] is None or gamma_list[0].numel() == 0:
            return [torch.empty(0, dtype=torch.long, device=self.device)]

        gamma = gamma_list[0]  # single sequence
        gamma = torch.nan_to_num(gamma, nan=-float("inf"), posinf=-float("inf"), neginf=-float("inf"))

        # MAP decoding: argmax along states
        map_seq = gamma.argmax(dim=-1).to(dtype=torch.long, device=self.device)
        return [map_seq]

    def _viterbi(self, X: utils.Observations, theta: Optional[torch.Tensor] = None, duration_weight: float = 0.0) -> torch.Tensor:

        device = self.device
        seq = X.sequence[0].to(device, DTYPE)
        L, K, Dmax = seq.shape[0], self.n_states, self.max_duration
        if L == 0:
            return torch.empty(0, dtype=torch.int64, device=device)

        # --- Base parameters ---
        init_logits = self.init_logits.to(device, DTYPE)
        trans_logits = self.transition_logits.to(device, DTYPE)
        dur_logits = self.duration_module.log_matrix(context=theta)  # [K, Dmax]
        neg_inf = -torch.inf

        # --- Optional duration weighting ---
        if duration_weight > 0:
            dur_indices = torch.arange(1, Dmax + 1, device=device, dtype=DTYPE)
            dur_mean = (dur_logits.exp() * dur_indices).sum(dim=1, keepdim=True)
            dur_penalty = -((dur_indices - dur_mean) ** 2) / (2 * (Dmax/3)**2)
            dur_logits = (1 - duration_weight) * dur_logits + duration_weight * dur_penalty

        # --- Emissions ---
        if theta is not None and hasattr(self, "_contextual_emission_pdf"):
            emit_log = self._contextual_emission_pdf(seq.unsqueeze(0), theta).squeeze(0)  # [L, K]
        else:
            emit_log = X.log_probs[0].to(device, DTYPE)

        # --- Precompute cumulative sums for durations ---
        cumsum_emit = torch.vstack((torch.zeros((1, K), device=device, dtype=DTYPE), torch.cumsum(emit_log, dim=0)))

        # --- DP tables ---
        V = torch.full((L, K), neg_inf, device=device, dtype=DTYPE)
        back_ptr = torch.full((L, K), -1, dtype=torch.int64, device=device)
        best_dur = torch.zeros((L, K), dtype=torch.int64, device=device)

        for t in range(L):
            max_d = min(Dmax, t + 1)
            durations = torch.arange(1, max_d + 1, device=device)

            # Vectorized emission sum per duration
            starts = t - durations + 1  # [max_d]
            emit_sums = (cumsum_emit[t + 1] - cumsum_emit[starts]).T  # [K, max_d]

            # Vectorized Viterbi scoring
            scores = []
            prev_idx = []
            for i, d in enumerate(durations):
                start = starts[i]
                dur_score = dur_logits[:, d - 1]  # [K]
                if start == 0:
                    s = init_logits + dur_score + emit_sums[:, i]
                    idx = torch.full((K,), -1, dtype=torch.int64, device=device)
                else:
                    s_prev = V[start - 1].unsqueeze(1) + trans_logits  # [K, K]
                    s_max, s_arg = torch.max(s_prev, dim=0)
                    s = s_max + dur_score + emit_sums[:, i]
                    idx = s_arg
                scores.append(s)
                prev_idx.append(idx)

            scores_stack = torch.stack(scores, dim=1)  # [K, max_d]
            best_score, best_d_idx = torch.max(scores_stack, dim=1)
            V[t] = best_score
            best_dur[t] = durations[best_d_idx]
            prev_idx_stack = torch.stack(prev_idx, dim=1)
            best_idx = best_d_idx.unsqueeze(1)
            back_ptr[t] = prev_idx_stack.gather(1, best_idx).squeeze(1)

        # --- Backtrace ---
        t = L - 1
        cur_state = int(torch.argmax(V[t]).item())
        segments = []

        while t >= 0:
            d = int(best_dur[t, cur_state].item())
            start = max(0, t - d + 1)
            segments.append((start, t, cur_state))
            prev_state = int(back_ptr[t, cur_state].item())
            t = start - 1
            cur_state = prev_state if prev_state >= 0 else cur_state

        segments.reverse()
        seq_path = torch.cat([torch.full((end - start + 1,), st, dtype=torch.int64, device=device)
                              for start, end, st in segments])

        if seq_path.numel() < L:
            seq_path = torch.cat([seq_path, torch.full((L - seq_path.numel(),), seq_path[-1], dtype=torch.int64, device=device)])
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

        X = X.to(self.device, dtype=DTYPE) if torch.is_tensor(X) else X

        if theta is None and getattr(self, "encoder", None):
            theta = self.encode_observations(X)
        X_valid = self.to_observations(X, theta=theta)

        aligned_theta = None
        if theta is not None:
            total_len = sum(X_valid.lengths)
            if theta.ndim == 1:
                aligned_theta = theta.unsqueeze(0).expand(total_len, -1)
            elif theta.ndim == 2:
                if theta.shape[0] == len(X_valid.sequence):
                    aligned_theta = torch.cat(
                        [theta[i].expand(L, -1) for i, L in enumerate(X_valid.lengths)], dim=0
                    )
                elif theta.shape[0] == total_len:
                    aligned_theta = theta
                else:
                    raise ValueError(f"Cannot align theta of shape {tuple(theta.shape)}")
            aligned_theta = aligned_theta.to(self.device, dtype=DTYPE)

        # --- Convergence monitor ---
        self.conv = ConvergenceMonitor(
            tol=tol, max_iter=max_iter, n_init=n_init,
            post_conv_iter=post_conv_iter, verbose=verbose
        )

        best_score, best_state = -float("inf"), None

        for run_idx in range(n_init):
            if verbose:
                print(f"\n=== Run {run_idx + 1}/{n_init} ===")

            # --- Initialize / resample parameters ---
            if run_idx > 0 or sample_D_from_X:
                self.get_model_params(X_valid, theta=aligned_theta, mode="sample", inplace=True)

            # --- Initial log-likelihood ---
            base_ll = self._compute_log_likelihood(X_valid).sum()
            self.conv.push_pull(base_ll, 0, run_idx)

            for it in range(1, max_iter + 1):
                # --- E-step: update per-sequence log-probs ---
                emission_pdf = self._params.get("emission_pdf")
                if emission_pdf is None or not hasattr(emission_pdf, "log_prob"):
                    raise RuntimeError("Emission PDF not initialized or missing log_prob().")

                # Compute per-sequence log-probs using emission module
                X_valid.log_probs = [
                    emission_pdf.log_prob(seq.unsqueeze(1)) for seq in X_valid.sequence
                ]

                # Compute posteriors
                gamma_list, xi_list, eta_list = self._compute_posteriors(X_valid, theta=aligned_theta)
                all_gamma = torch.cat(
                    [g for g in gamma_list if g.numel() > 0], dim=0
                ).to(dtype=DTYPE, device=self.device)

                # --- M-step: update model parameters ---
                params = self.get_model_params(
                    X_valid, theta=aligned_theta, mode="estimate", inplace=True
                )
                emission_pdf = params["emission_pdf"]

                # --- Log-likelihood and convergence check ---
                curr_ll = self._compute_log_likelihood(X_valid).sum()
                converged = self.conv.push_pull(curr_ll, it, run_idx)
                if converged and not ignore_conv:
                    if verbose:
                        print(f"[Run {run_idx + 1}] Converged at iteration {it}.")
                    break

            # --- Track best-scoring model ---
            run_score = float(curr_ll.item())
            if run_score > best_score:
                best_score = run_score
                best_state = {
                    "init_logits": self._init_logits.clone(),
                    "transition_logits": self._transition_logits.clone(),
                    "duration_logits": self._duration_logits.clone(),
                    "emission_pdf": copy.deepcopy(emission_pdf),
                }

        # --- Restore best parameters ---
        if best_state:
            self._init_logits.copy_(best_state["init_logits"])
            self._transition_logits.copy_(best_state["transition_logits"])
            self._duration_logits.copy_(best_state["duration_logits"])
            self._params["emission_pdf"] = best_state["emission_pdf"]

        # --- Optional convergence plotting ---
        if plot_conv and hasattr(self, "conv"):
            self.conv.plot_convergence()

        return self

    def predict(self, X: torch.Tensor,
        algorithm: Literal["map", "viterbi"] = "viterbi",
        context: Optional[torch.Tensor] = None,
        temp_transition: float = 1.0,
        temp_duration: float = 1.0
    ) -> torch.Tensor:

        X_valid = self.to_observations(X)
        seq_tensor = X_valid.sequence[0].to(device=self.device, dtype=DTYPE)
        T, K = seq_tensor.shape[0], self.n_states

        valid_context = None
        if context is not None:
            context = context.to(device=self.device, dtype=DTYPE)
            if context.ndim != 2 or context.shape[0] != T:
                raise ValueError(f"Context must be [T, D] and match sequence length ({T})")
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
                var = var.clamp(min=self.min_covar)
                dist = torch.distributions.Normal(mu, var.sqrt())
                log_probs = dist.log_prob(seq_tensor.unsqueeze(1)).sum(dim=-1)
            else:
                logits = output
                if seq_tensor.dtype != torch.long:
                    raise TypeError("seq_tensor must contain integer indices for categorical emission.")
                log_probs = F.log_softmax(logits, dim=-1)[torch.arange(T, device=self.device), seq_tensor.long()]

        # --- Transition log-probs ---
        if hasattr(self, "transition_module"):
            transition_logits = self.transition_module.log_matrix(context=valid_context) / max(temp_transition, 1e-6)
        else:
            transition_logits = F.log_softmax(self._transition_logits.to(self.device, DTYPE) / max(temp_transition, 1e-6), dim=-1)

        # --- Duration log-probs ---
        if hasattr(self, "duration_module"):
            log_duration = self.duration_module.log_matrix(context=valid_context) / max(temp_duration, 1e-6)
        else:
            log_duration = None

        # --- Safe duration helper ---
        def dur_term(t: int) -> torch.Tensor:
            if log_duration is None:
                return torch.zeros(K, device=self.device, dtype=DTYPE)
            dur_idx = min(t, log_duration.shape[-1] - 1)
            return log_duration[:, dur_idx] if log_duration.ndim == 2 else log_duration[dur_idx]

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
    def _compute_log_likelihood(
        self, 
        X: utils.Observations, 
        theta: Optional[torch.Tensor] = None, 
        verbose: bool = False
    ) -> torch.Tensor:
        """
        Compute per-sequence log-likelihoods log P(X | model, theta).
        Stores them in X.log_likelihoods and returns a tensor [B].
        """
        device = self.device
        neg_inf = torch.finfo(DTYPE).min / 2.0
        B = len(X.sequence)

        if B == 0:
            X.log_likelihoods = torch.full((0,), neg_inf, device=device, dtype=DTYPE)
            return X.log_likelihoods

        # --- Align context shape ---
        T_max = max(X.lengths)
        valid_theta = None
        if theta is not None:
            theta = theta.to(device=device, dtype=DTYPE)
            if theta.ndim == 1:               # global [H]
                valid_theta = [theta.unsqueeze(0).expand(L, -1) for L in X.lengths]
            elif theta.ndim == 2:             # [B,H] or [T,H] or [total_T,H]
                if theta.shape[0] == B:       # per sequence
                    valid_theta = [theta[b].unsqueeze(0).expand(L, -1) for b, L in enumerate(X.lengths)]
                elif theta.shape[0] == T_max: # per timestep
                    valid_theta = [theta[:L] for L in X.lengths]
                elif theta.shape[0] == sum(X.lengths):  # flattened
                    split = torch.split(theta, X.lengths)
                    valid_theta = [s for s in split]
                else:
                    raise ValueError(f"Cannot align theta shape {tuple(theta.shape)}")
            elif theta.ndim == 3 and theta.shape[0] == B:
                valid_theta = [theta[b, :L] for b, L in enumerate(X.lengths)]
            else:
                raise ValueError(f"Unsupported theta shape {theta.shape} for B={B}, T_max={T_max}")

        # --- Forward pass ---
        alpha_list = self._forward(X, theta=valid_theta)
        if not alpha_list:
            raise RuntimeError("Forward returned None or empty alpha_list.")

        # --- Compute per-sequence log-likelihoods ---
        ll_list = []
        for alpha, L in zip(alpha_list, X.lengths):
            if L == 0:
                ll_list.append(torch.tensor(neg_inf, device=device, dtype=DTYPE))
                continue

            # Optional masking
            if hasattr(X, "masks") and X.masks is not None:
                mask = X.masks.pop(0).to(device=device, dtype=DTYPE)
                alpha = alpha + torch.log(mask + EPS)

            ll_seq = torch.logsumexp(alpha[L-1], dim=(0,1))
            ll_list.append(ll_seq)

        X.log_likelihoods = torch.stack(ll_list)

        if verbose:
            ll = X.log_likelihoods
            print(f"[compute_ll] seqs={B}, min={ll.min():.4f}, max={ll.max():.4f}, mean={ll.mean():.4f}")

        return X.log_likelihoods

    def score(
        self,
        X: torch.Tensor,
        theta: Optional[torch.Tensor] = None,
        temp_transition: float = 1.0,
        temp_duration: float = 1.0
    ) -> torch.Tensor:

        obs = self.to_observations(X, theta)
        seq = obs.sequence[0].to(dtype=DTYPE, device=self.device)
        T = seq.shape[0]
        K, Dmax = self.n_states, self.max_duration
        neg_inf = torch.finfo(DTYPE).min / 2.0

        if T == 0:
            return torch.tensor(neg_inf, device="cpu")

        # --- Emission log-probabilities ---
        pdf = self._params.get("emission_pdf")
        if pdf is None or not hasattr(pdf, "log_prob"):
            raise RuntimeError("Emission PDF must be initialized with log_prob method.")

        log_B = pdf.log_prob(seq.unsqueeze(1))  # [T, K]
        if log_B.ndim > 2:
            log_B = log_B.sum(dim=list(range(2, log_B.ndim)))

        # --- Transition log-probs with temperature ---
        if hasattr(self, "transition_module"):
            log_transition = (
                self.transition_module.log_matrix(context=theta)
                if hasattr(self.transition_module, "log_matrix")
                else F.log_softmax(self.transition_module.logits / max(temp_transition, 1e-6), dim=-1)
            )
        else:
            log_transition = (
                F.log_softmax(self._transition_logits / max(temp_transition, 1e-6), dim=-1)
                if hasattr(self, "_transition_logits")
                else torch.full((K, K), 1.0 / K, device=self.device).log()
            )

        # --- Duration log-probs with temperature ---
        if hasattr(self, "duration_module"):
            log_duration = (
                self.duration_module.log_matrix(context=theta)
                if hasattr(self.duration_module, "log_matrix")
                else F.log_softmax(self.duration_module.logits / max(temp_duration, 1e-6), dim=-1)
            )
        else:
            log_duration = (
                F.log_softmax(self._duration_logits / max(temp_duration, 1e-6), dim=-1)
                if hasattr(self, "_duration_logits")
                else torch.zeros(K, Dmax, device=self.device, dtype=DTYPE)
            )

        # --- Forward recursion ---
        V = torch.full((T, K), neg_inf, device=self.device, dtype=DTYPE)
        V[0] = log_B[0] + torch.log_softmax(self.init_logits.to(device=self.device, dtype=DTYPE), dim=0)

        for t in range(1, T):
            max_d = min(Dmax, t + 1)
            dur_idx = torch.arange(max_d, device=self.device)
            scores = []

            for d in dur_idx:
                start = t - d + 1
                emit_sum = log_B[start : t + 1].sum(dim=0)
                dur_score = log_duration[:, d] if log_duration.ndim == 2 else log_duration[d]

                if start == 0:
                    trans_score = torch.zeros(K, device=self.device, dtype=DTYPE)
                else:
                    trans_score = torch.logsumexp(V[start - 1].unsqueeze(1) + log_transition, dim=0)

                scores.append(trans_score + dur_score + emit_sum)

            V[t] = torch.logsumexp(torch.stack(scores, dim=0), dim=0)

        # --- Total log-likelihood ---
        return torch.logsumexp(V[-1], dim=-1).detach().cpu()

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

