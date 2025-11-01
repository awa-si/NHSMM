# nhsmm/models/hsmm.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Any, Literal, Dict
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, Categorical, Independent

from nhsmm.context import ContextEncoder
from nhsmm.constants import DTYPE, EPS, HSMMError, logger
from nhsmm.modules import Initial, Emission, Duration, Transition
from nhsmm import utils, constraints, SeedGenerator, ConvergenceMonitor


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
        modulate_var: bool = False,
        transition_type: Any = None,
        alpha: Optional[float] = 1.0,
        emission_type: str = "gaussian",
        min_covar: Optional[float] = 1e-6,
        context_dim: Optional[int] = None,
        device: Optional[torch.device] = None,
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

        self._params: Dict[str, Any] = {'emission_pdf': None}
        self._context: Optional[torch.Tensor] = None
        self.encoder: Optional[nn.Module] = None

        self._init_modules()

    def _init_modules(self, seed: Optional[int] = None):
        """Initialize HSMM modules: emission, duration, and transition."""
        if seed is not None:
            torch.manual_seed(seed)

        device = self.device
        hidden_dim = getattr(self, "hidden_dim", None)

        self.initial_module = Initial(
            n_states=self.n_states,
            context_dim=self.context_dim,
            hidden_dim=hidden_dim,
            init_mode="uniform",
        ).to(device)

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
        self._params['emission_pdf'] = self.emission_module.initialize()

        self.duration_module = Duration(
            n_states=self.n_states,
            max_duration=self.max_duration,
            context_dim=self.context_dim,
            hidden_dim=hidden_dim,
            temperature=getattr(self, "duration_temp", 1.0),
            scale=getattr(self, "duration_scale", 1.0),
            debug=getattr(self, "debug", False),
        ).to(device)

        self.transition_module = Transition(
            n_states=self.n_states,
            context_dim=self.context_dim,
            hidden_dim=hidden_dim,
            temperature=getattr(self, "transition_temp", 1.0),
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
        emission_type: Optional[str] = None,
        theta_scale: float = 0.1,
        mode: str = "estimate",
        context: Optional[torch.Tensor] = None,
    ) -> Distribution:
        """
        Compute or sample the emission PDF via the emission module.

        Modes:
            - "sample": sample parameters using emission_module.initialize()
            - "estimate": compute weighted mean/covariance (continuous)
                          or empirical probabilities (discrete)
        """
        emission_type = (emission_type or self.emission_module.emission_type).lower()

        # ---------------- Sample from module ----------------
        if mode == "sample":
            pdf = self.emission_module.initialize(
                X=X, posterior=posterior, theta=theta, emission_type=emission_type, theta_scale=theta_scale, context=context
            )
            self._params["emission_pdf"] = pdf
            return pdf

        # ---------------- Estimate from data ----------------
        elif mode == "estimate":
            if X is None or (posterior is None and emission_type in {"gaussian", "laplace", "studentt"}):
                raise ValueError("X (and posterior for continuous) must be provided for estimate mode.")

            X = X.to(dtype=DTYPE, device=self.device)
            posterior = posterior.to(dtype=DTYPE, device=self.device) if posterior is not None else None

            # ---------------- Continuous ----------------
            if emission_type in {"gaussian", "laplace", "studentt"}:
                pdf = self.emission_module.initialize(
                    X=X, posterior=posterior, theta=theta, emission_type=emission_type, theta_scale=theta_scale, context=context
                )

            # ---------------- Discrete ----------------
            else:
                K, F = self.n_states, self.n_features
                if emission_type == "categorical":
                    counts = torch.stack([torch.bincount(X[:, f].long(), minlength=F) for f in range(F)], dim=1).T.float()
                    params = counts / counts.sum(-1, keepdim=True)
                else:  # bernoulli / poisson
                    params = X.float().mean(0, keepdim=True).expand(K, -1)

                # Theta/context modulation
                if theta is not None:
                    theta_tensor = theta.mean(dim=0, keepdim=True) if theta.ndim == 2 else theta
                    params = params + theta_scale * theta_tensor.expand(K, -1)
                params = self.emission_module._modulate_per_state(params, context)

                # Store in module buffer
                self.emission_module._emission_params.copy_(params)

                # Build distribution
                if emission_type == "categorical":
                    pdf = Categorical(probs=params)
                elif emission_type == "bernoulli":
                    pdf = Independent(Bernoulli(probs=params), 1)
                else:  # poisson
                    pdf = Independent(Poisson(params), 1)

            self._params["emission_pdf"] = pdf
            return pdf
        else:
            raise ValueError(f"Unsupported mode '{mode}'")

    def get_model_params(
        self,
        X: Optional["Observations"] = None,
        theta: Optional[torch.Tensor] = None,
        theta_scale: float = 0.1,
        mode: str = "estimate",
    ) -> dict[str, Any]:
        """
        Retrieve or sample HSMM parameters for a single sequence.
        Uses module methods directly and applies optional context modulation.
        """
        device, K, Dmax = self.device, self.n_states, self.max_duration

        # ---------------- Align theta ----------------
        def _align_theta(theta: Optional[torch.Tensor], seq_len: int) -> Optional[torch.Tensor]:
            if theta is None:
                return None
            if theta.ndim == 1:
                return theta.unsqueeze(0).expand(seq_len, -1).to(device=device, dtype=DTYPE)
            if theta.ndim == 2:
                if theta.shape[0] == seq_len:
                    return theta.to(device=device, dtype=DTYPE)
                if theta.shape[0] == 1:
                    return theta.expand(seq_len, -1).to(device=device, dtype=DTYPE)
            raise ValueError(f"Cannot align theta of shape {tuple(theta.shape)} with sequence length {seq_len}")

        seq_len = sum(getattr(X, "lengths", [1])) if X is not None else 1
        aligned_theta = _align_theta(theta, seq_len)

        # ---------------- Concatenate sequences ----------------
        if X is not None:
            all_X = torch.cat([s for s in getattr(X, "sequence", [X]) if s.numel() > 0], dim=0).to(device=device, dtype=DTYPE)
            if all_X.numel() == 0:
                all_X = torch.zeros(1, self.n_features, device=device, dtype=DTYPE)
        else:
            all_X = torch.zeros(1, self.n_features, device=device, dtype=DTYPE)

        # ---------------- Sampling helpers ----------------
        α = getattr(self, "alpha", 1.0)

        def sample_probs(shape: tuple[int, ...]) -> torch.Tensor:
            p = constraints.sample_probs(α, shape).clamp_min(self.min_covar)
            return torch.log(p).to(device=device, dtype=DTYPE)

        def sample_transition(n: int) -> torch.Tensor:
            p = constraints.sample_transition(α, n, self.transition_type)
            p = p / p.sum(dim=-1, keepdim=True).clamp_min(EPS)
            return torch.log(p).to(device=device, dtype=DTYPE)

        # ---------------- Sampling Mode ----------------
        if mode == "sample":
            init_logits = sample_probs((K,))
            transition_logits = sample_transition(K)
            duration_logits = sample_probs((K, Dmax))
            emission_pdf = self.compute_emission_pdf(
                X=all_X, posterior=None, theta=aligned_theta, theta_scale=theta_scale,
                mode="sample", context=aligned_theta
            )

        # ---------------- Estimation Mode ----------------
        elif mode == "estimate":
            if X is None or not isinstance(X, utils.Observations):
                raise RuntimeError("Must provide Observations X for estimate mode.")

            gamma_list, xi_list, eta_list = self._compute_posteriors(X, theta=aligned_theta)

            def safe_sum(lst: list[torch.Tensor], dim: int = 0) -> Optional[torch.Tensor]:
                out = [x.sum(dim=dim) for x in lst if x is not None and x.numel() > 0]
                return torch.stack(out).sum(0) if out else None

            init_counts = safe_sum([g[0] for g in gamma_list])
            trans_counts = safe_sum(xi_list)
            dur_counts = safe_sum(eta_list)

            # Fallback to sampling if counts are invalid
            if init_counts is None or (init_counts < EPS).any():
                init_counts = sample_probs((K,))
            if trans_counts is None or (trans_counts.sum(-1) < EPS).any():
                trans_counts = sample_transition(K)
            if dur_counts is None or (dur_counts.sum(-1) < EPS).any():
                dur_counts = sample_probs((K, Dmax))

            init_logits = constraints.log_normalize(torch.log(init_counts + EPS), dim=0).to(device=device, dtype=DTYPE)
            transition_logits = constraints.log_normalize(torch.log(trans_counts + EPS), dim=1).to(device=device, dtype=DTYPE)
            duration_logits = constraints.log_normalize(torch.log(dur_counts + EPS), dim=1).to(device=device, dtype=DTYPE)

            all_gamma = torch.cat([g for g in gamma_list if g is not None and g.numel() > 0], dim=0)
            emission_pdf = self.compute_emission_pdf(
                X=all_X, posterior=all_gamma, theta=aligned_theta, theta_scale=theta_scale,
                mode="estimate", context=aligned_theta
            )

        else:
            raise ValueError(f"Unsupported mode '{mode}'.")

        # ---------------- Return dict ----------------
        return {
            "init_logits": init_logits,
            "transition_logits": transition_logits,
            "duration_logits": duration_logits,
            "emission_pdf": emission_pdf,
        }

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
        Encode a single-sequence observation X into a context vector using the attached encoder.

        Args:
            X: Input tensor of shape (F,) or (T,F)
            pool: Optional pooling mode override for the encoder
            detach_return: If True, detach returned tensor to prevent gradient flow
            store: If True, save result to self._context

        Returns:
            Context tensor of shape [1,H], or None if no encoder is attached
        """
        if self.encoder is None:
            if store:
                self._context = None
            return None

        device = self.device
        X = X.to(device=device, dtype=DTYPE)

        if X.numel() == 0:
            raise ValueError(f"Input X is empty: shape {X.shape}")

        # Normalize to (1,T,F)
        if X.ndim == 1:
            X = X[None, None, :]
        elif X.ndim == 2:
            X = X[None, :, :]
        elif X.ndim != 3:
            raise ValueError(f"Expected input shape (F,) or (T,F), got {X.shape}")

        # Temporarily override pooling if requested
        original_pool = getattr(self.encoder, "pool", None)
        if pool is not None and hasattr(self.encoder, "pool"):
            self.encoder.pool = pool

        try:
            _ = self.encoder(X, return_context=True)
            vec = self.encoder.get_context()
            if vec is None:
                raise RuntimeError("Encoder returned None context")
        except Exception as e:
            raise RuntimeError(f"[encode_observations] Encoder forward failed for input shape {X.shape}: {e}") from e
        finally:
            if hasattr(self.encoder, "pool"):
                self.encoder.pool = original_pool

        vec = vec.to(device=device, dtype=DTYPE)
        if detach_return:
            vec = vec.detach()

        if store:
            self._context = vec

        return vec

    def to_observations(
        self,
        X: torch.Tensor,
        theta: Optional[torch.Tensor] = None
    ) -> utils.Observations:
        """
        Convert raw input X (and optional context theta) into a structured Observations object.
        Single-sequence only.

        Returns:
            utils.Observations(sequence, log_probs, lengths, context)
        """
        device = self.device
        X_valid = self._validate(X).to(device=device, dtype=DTYPE)
        
        # Ensure 2D [T, F]
        if X_valid.ndim == 1:
            X_valid = X_valid.unsqueeze(0)
        T, F = X_valid.shape
        X_valid = X_valid.view(T, F)

        # ---- Align context ----
        context = theta if theta is not None else getattr(self, "_context", None)
        if context is not None:
            context = context.to(device=device, dtype=DTYPE)
            if context.ndim == 1:
                context = context.unsqueeze(0)  # [1, D]
            if context.shape[0] != T:
                if context.shape[0] == 1:
                    context = context.expand(T, -1)
                else:
                    raise ValueError(f"Context length ({context.shape[0]}) does not match X length ({T})")

        # ---- Compute emission log-probs ----
        logp_or_dist = self._contextual_emission_pdf(X_valid, context)

        if isinstance(logp_or_dist, torch.distributions.Distribution):
            # Continuous or independent discrete distribution
            log_probs = logp_or_dist.log_prob(X_valid.unsqueeze(-2))  # [T, K, ...]
            # Sum over extra feature/event dims to get [T, K]
            if log_probs.ndim > 2:
                log_probs = log_probs.sum(dim=tuple(range(2, log_probs.ndim)))
        elif torch.is_tensor(logp_or_dist):
            log_probs = logp_or_dist
            # Ensure [T, K] shape
            if log_probs.ndim == 1:
                log_probs = log_probs.view(T, 1)
            elif log_probs.ndim == 2 and log_probs.shape[0] == 1:
                log_probs = log_probs.expand(T, -1)
            elif log_probs.ndim > 2:
                log_probs = log_probs.sum(dim=tuple(range(2, log_probs.ndim)))
        else:
            raise TypeError(f"Unsupported return type from _contextual_emission_pdf: {type(logp_or_dist)}")

        log_probs = log_probs.to(dtype=DTYPE, device=device)

        # ---- Build Observations ----
        return utils.Observations(
            sequence=[X_valid],
            log_probs=[log_probs],
            lengths=[T],
            context=[context] if context is not None else [None]
        )


    # hsmm.py HSMM precessing
    def _forward(self, X: utils.Observations, theta: Optional[ContextualVariables] = None) -> list[torch.Tensor]:
        """
        Vectorized forward pass for single-sequence HSMM with context-modulated modules.
        Returns a list of alpha tensors [T, K, Dmax] for each sequence.
        """
        K, Dmax, device = self.n_states, self.max_duration, self.device
        neg_inf = torch.finfo(DTYPE).min / 2.0

        # --- Module-based logits ---
        init_logits = self.initial_module.log_matrix(context=theta).to(device=device, dtype=DTYPE)
        dur_logits = self.duration_module.log_matrix(context=theta).to(device=device, dtype=DTYPE)
        trans_logits = self.transition_module.log_matrix(context=theta).to(device=device, dtype=DTYPE)

        # --- Safe normalization ---
        def _safe_log_rows(logits: torch.Tensor, dim=-1, min_prob=1e-6):
            probs = logits.exp().clamp_min(min_prob)
            probs = probs / probs.sum(dim=dim, keepdim=True).clamp_min(EPS)
            return torch.log(probs)

        init_logits = torch.log(F.softmax(init_logits, dim=0).clamp_min(1e-6))
        dur_logits = _safe_log_rows(dur_logits, dim=-1)
        trans_logits = _safe_log_rows(trans_logits, dim=-1)

        alpha_list = []

        for seq_logp, seq_len in zip(X.log_probs, X.lengths):
            if seq_len == 0:
                alpha_list.append(torch.full((0, K, Dmax), neg_inf, device=device, dtype=DTYPE))
                continue

            seq_logp = seq_logp.to(device=device, dtype=DTYPE)  # [T, K]
            log_alpha = torch.full((seq_len, K, Dmax), neg_inf, device=device, dtype=DTYPE)

            # --- Precompute cumulative emission sums ---
            cumsum_emit = torch.cat([torch.zeros(1, K, device=device, dtype=DTYPE),
                                     torch.cumsum(seq_logp, dim=0)], dim=0)  # [T+1, K]

            # --- t=0 initialization ---
            max_d0 = min(Dmax, seq_len)
            emit_sums0 = (cumsum_emit[1:max_d0+1] - cumsum_emit[0]).T  # [K, max_d0]
            log_alpha[0, :, :max_d0] = init_logits.unsqueeze(1) + dur_logits[:, :max_d0] + emit_sums0

            # --- Recursion for t >= 1 ---
            for t in range(1, seq_len):
                max_dt = min(Dmax, t+1)
                starts = t - torch.arange(max_dt, device=device)  # [max_dt]
                emit_sums = (cumsum_emit[t+1] - cumsum_emit[starts]).T  # [K, max_dt]

                # Vectorized previous alpha selection
                prev_alpha = torch.full((max_dt, K), neg_inf, device=device, dtype=DTYPE)
                idx_prev = starts - 1
                mask_valid = idx_prev >= 0
                if mask_valid.any():
                    prev_vals = torch.logsumexp(log_alpha[idx_prev[mask_valid].long(), :, :], dim=2)
                    prev_alpha[mask_valid] = prev_vals
                prev_alpha[starts == 0] = init_logits  # handle segments starting at t=0

                # Transition + duration + emission
                summed = torch.logsumexp(prev_alpha.unsqueeze(2) + trans_logits.unsqueeze(0), dim=1)  # [max_dt, K]
                log_alpha[t, :, :max_dt] = summed.T + dur_logits[:, :max_dt] + emit_sums

            alpha_list.append(log_alpha)

        return alpha_list

    def _backward(self, X: utils.Observations, theta: Optional[ContextualVariables] = None) -> list[torch.Tensor]:
        """
        Vectorized backward pass for single-sequence HSMM with context-modulated modules.
        Returns a list of beta tensors [T, K, Dmax] for each sequence.
        """
        K, Dmax, device = self.n_states, self.max_duration, self.device
        neg_inf = torch.finfo(DTYPE).min / 2.0

        # --- Module-based logits ---
        dur_logits = self.duration_module.log_matrix(context=theta).to(device=device, dtype=DTYPE)
        trans_logits = self.transition_module.log_matrix(context=theta).to(device=device, dtype=DTYPE)

        # Safe row-normalization
        def _safe_log_rows(logits: torch.Tensor):
            probs = logits.exp().clamp_min(EPS)
            probs /= probs.sum(dim=-1, keepdim=True).clamp_min(EPS)
            return torch.log(probs.clamp_min(EPS))

        dur_logits = _safe_log_rows(dur_logits)
        trans_logits = _safe_log_rows(trans_logits)

        beta_list = []

        for seq_logp, seq_len in zip(X.log_probs, X.lengths):
            if seq_len == 0:
                beta_list.append(torch.full((0, K, Dmax), neg_inf, device=device, dtype=DTYPE))
                continue

            seq_logp = seq_logp.to(device=device, dtype=DTYPE)  # [T, K]
            log_beta = torch.full((seq_len, K, Dmax), neg_inf, device=device, dtype=DTYPE)
            log_beta[-1, :, 0] = 0.0  # terminal step

            # --- Precompute cumulative sums of emission log-probs ---
            cumsum_emit = torch.cat([torch.zeros(1, K, device=device, dtype=DTYPE),
                                     torch.cumsum(seq_logp, dim=0)], dim=0)  # [T+1, K]

            # --- Backward recursion ---
            for t in reversed(range(seq_len - 1)):
                max_dt = min(Dmax, seq_len - t)
                durations = torch.arange(1, max_dt + 1, device=device)
                ends = t + durations  # segment end indices

                emit_sums = (cumsum_emit[ends] - cumsum_emit[t].unsqueeze(0)).T  # [K, max_dt]
                dur_scores = dur_logits[:, :max_dt]  # [K, max_dt]

                # Vectorized next-segment β computation
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

    def _compute_posteriors(
        self, X: utils.Observations, theta: Optional[ContextualVariables] = None
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:

        K, Dmax, device = self.n_states, self.max_duration, self.device
        B = len(X.sequence)
        EPS_ = EPS

        alpha_list = self._forward(X, theta)
        beta_list = self._backward(X, theta)

        # --- Module-aware logits ---
        init_logits = self.initial_module.log_matrix(context=theta).to(device=device, dtype=DTYPE)
        transition_logits = self.transition_module.log_matrix(context=theta).to(device=device, dtype=DTYPE)
        transition_logits = torch.log_softmax(transition_logits, dim=-1)

        gamma_vec, xi_vec, eta_vec = [], [], []

        # --- Prepare masks for variable-length sequences ---
        max_len = max(X.lengths)
        mask = torch.zeros(B, max_len, dtype=DTYPE, device=device)
        for b, L in enumerate(X.lengths):
            mask[b, :L] = 1.0

        # Stack alphas/betas
        alpha_pad = torch.stack([F.pad(a[:, :, 0], (0, Dmax-1, 0, max_len - a.shape[0]), value=torch.finfo(DTYPE).min/2.0)
                                 for a in alpha_list], dim=0)  # [B, T_max, K]
        beta_pad = torch.stack([F.pad(b[:, :, 0], (0, Dmax-1, 0, max_len - b.shape[0]), value=torch.finfo(DTYPE).min/2.0)
                                for b in beta_list], dim=0)    # [B, T_max, K]

        # --- η: posterior over state-duration pairs ---
        eta_pad = torch.stack([F.pad(a + b, (0, 0, 0, max_len - a.shape[0]), value=torch.finfo(DTYPE).min/2.0)
                               for a, b in zip(alpha_list, beta_list)], dim=0)  # [B, T_max, K, Dmax]
        eta_pad_flat = eta_pad.view(B, max_len, -1)
        eta_soft = F.softmax(eta_pad_flat, dim=-1) * mask.unsqueeze(-1)
        eta_soft = eta_soft.view(B, max_len, K, Dmax)

        # --- γ: posterior over states ---
        gamma = eta_soft.sum(dim=-1)
        gamma = gamma / gamma.sum(dim=-1, keepdim=True)  # normalize over states

        # --- ξ: posterior over transitions ---
        xi = torch.zeros(B, max_len-1, K, K, device=device, dtype=DTYPE)
        for b, L in enumerate(X.lengths):
            if L <= 1:
                continue
            a_prev = torch.logsumexp(alpha_list[b][:-1], dim=2)  # [T-1, K]
            b_next = torch.logsumexp(beta_list[b][1:], dim=2)     # [T-1, K]
            log_xi_b = a_prev.unsqueeze(2) + transition_logits.unsqueeze(0) + b_next.unsqueeze(1)  # [T-1, K, K]
            xi[b, :L-1] = F.softmax(log_xi_b.reshape(L-1, -1), dim=1).reshape(L-1, K, K).clamp_min(EPS_)

        # Split back per sequence
        gamma_vec = [gamma[b, :X.lengths[b]] for b in range(B)]
        eta_vec = [eta_soft[b, :X.lengths[b]] for b in range(B)]
        xi_vec = [xi[b, :X.lengths[b]-1] if X.lengths[b] > 1 else torch.zeros((0, K, K), device=device, dtype=DTYPE)
                  for b in range(B)]

        return gamma_vec, xi_vec, eta_vec


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
        neg_inf = torch.finfo(DTYPE).min / 2
        seq = X.sequence[0].to(device, DTYPE)
        L, K, Dmax = seq.shape[0], self.n_states, self.max_duration
        if L == 0:
            return torch.empty(0, dtype=torch.int64, device=device)

        # --- Module-aware base parameters ---
        init_logits = self.initial_module.log_matrix(context=theta).to(device, DTYPE)
        dur_logits = self.duration_module.log_matrix(context=theta).to(device, DTYPE)
        trans_logits = self.transition_module.log_matrix(context=theta).to(device, DTYPE)


        # --- Optional duration weighting ---
        if duration_weight > 0:
            dur_indices = torch.arange(1, Dmax + 1, device=device, dtype=DTYPE)
            dur_mean = (dur_logits.exp() * dur_indices).sum(dim=1, keepdim=True)
            dur_penalty = -((dur_indices - dur_mean) ** 2) / (2 * (Dmax / 3)**2)
            dur_logits = (1 - duration_weight) * dur_logits + duration_weight * dur_penalty

        # --- Emissions ---
        if hasattr(self, "_contextual_emission_pdf") and theta is not None:
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

            starts = t - durations + 1  # [max_d]
            emit_sums = (cumsum_emit[t + 1] - cumsum_emit[starts]).T  # [K, max_d]
            dur_scores = dur_logits[:, :max_d]  # [K, max_d]

            if t == 0:
                # Only initial state
                scores = init_logits.unsqueeze(1) + dur_scores + emit_sums  # [K, max_d]
                best_score, best_idx = scores.max(dim=1)
                best_dur[t] = durations[best_idx]
                V[t] = best_score
                back_ptr[t] = -1
            else:
                # Previous scores broadcasting
                prev_scores = V[starts - 1]  # [max_d, K]
                prev_scores = prev_scores.unsqueeze(2) + trans_logits.unsqueeze(0)  # [max_d, K, K]

                # Max over previous states for each duration
                prev_max, prev_arg = prev_scores.max(dim=1)  # [max_d, K], [max_d, K]

                # Add duration + emission scores
                scores = prev_max.T + dur_scores + emit_sums  # [K, max_d]

                # Find best duration per state
                best_score, best_d_idx = scores.max(dim=1)
                V[t] = best_score
                best_dur[t] = durations[best_d_idx]

                # Back pointers: track previous state for best duration
                idx_stack = prev_arg[best_d_idx, torch.arange(K, device=device)]  # [K]
                back_ptr[t] = idx_stack

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

        return seq_path[:L]

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

            # --- Initialize / sample parameters ---
            if run_idx > 0 or sample_D_from_X:
                params = self.get_model_params(X_valid, theta=aligned_theta, mode="sample")
            else:
                params = self.get_model_params(X_valid, theta=aligned_theta, mode="estimate")

            # Extract local copies (buffer-free)
            init_logits = params["init_logits"]
            transition_logits = params["transition_logits"]
            duration_logits = params["duration_logits"]
            emission_pdf = params["emission_pdf"]
            if hasattr(params, "super_init_logits"):
                super_init_logits = params.get("super_init_logits")
                super_transition_logits = params.get("super_transition_logits")

            # --- Initial log-likelihood ---
            X_valid.log_probs = []
            for seq in X_valid.sequence:
                seq_exp = seq.to(self.device, dtype=DTYPE).unsqueeze(1).expand(-1, self.n_states, -1)
                X_valid.log_probs.append(emission_pdf.log_prob(seq_exp))
            base_ll = self._compute_log_likelihood(X_valid).sum()
            self.conv.push_pull(base_ll, 0, run_idx)

            for it in range(1, max_iter + 1):
                # --- E-step ---
                X_valid.log_probs = []
                for seq in X_valid.sequence:
                    seq_exp = seq.to(self.device, dtype=DTYPE).unsqueeze(1).expand(-1, self.n_states, -1)
                    X_valid.log_probs.append(emission_pdf.log_prob(seq_exp))

                gamma_list, xi_list, eta_list = self._compute_posteriors(X_valid, theta=aligned_theta)
                all_gamma = torch.cat([g for g in gamma_list if g.numel() > 0], dim=0).to(dtype=DTYPE, device=self.device)

                # --- M-step ---
                params = self.get_model_params(X_valid, theta=aligned_theta, mode="estimate")
                init_logits = params["init_logits"]
                transition_logits = params["transition_logits"]
                duration_logits = params["duration_logits"]
                emission_pdf = params["emission_pdf"]
                if hasattr(params, "super_init_logits"):
                    super_init_logits = params.get("super_init_logits")
                    super_transition_logits = params.get("super_transition_logits")

                # --- Convergence check ---
                X_valid.log_probs = []
                for seq in X_valid.sequence:
                    seq_exp = seq.to(self.device, dtype=DTYPE).unsqueeze(1).expand(-1, self.n_states, -1)
                    X_valid.log_probs.append(emission_pdf.log_prob(seq_exp))

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
                    "init_logits": init_logits.clone(),
                    "transition_logits": transition_logits.clone(),
                    "duration_logits": duration_logits.clone(),
                    "emission_pdf": copy.deepcopy(emission_pdf),
                }
                # if n_super > 1:
                    # best_state.update({
                        # "super_init_logits": super_init_logits.clone(),
                        # "super_transition_logits": super_transition_logits.clone()
                    # })

        # --- Restore best parameters to modules ---
        if best_state:
            self.get_model_params(X_valid)  # optional: re-sync modules if needed
            self._params["emission_pdf"] = best_state["emission_pdf"]  # module-only reference

        # --- Optional convergence plotting ---
        if plot_conv and hasattr(self, "conv"):
            self.conv.plot_convergence()

        return self

    def predict(
        self,
        X: torch.Tensor,
        algorithm: Literal["map", "viterbi"] = "viterbi",
        context: Optional[torch.Tensor] = None,
        temp_transition: float = 1.0,
        temp_duration: float = 1.0,
    ) -> torch.Tensor:
        X_valid = self.to_observations(X)
        seq_tensor = X_valid.sequence[0].to(self.device, DTYPE)
        T, K = seq_tensor.shape[0], self.n_states

        # --- Context handling ---
        valid_context = None
        if context is not None:
            context = context.to(self.device, DTYPE)
            if context.ndim != 2 or context.shape[0] != T:
                raise ValueError(f"Context must be [T, D] and match sequence length ({T})")
            valid_context = context

        # --- Emission log-probs ---
        pdf = getattr(self, "pdf", None)
        if pdf is not None and hasattr(pdf, "log_prob"):
            # Ensure proper input shape for multivariate distributions
            x_input = seq_tensor.unsqueeze(1) if seq_tensor.ndim == 2 else seq_tensor
            log_probs = pdf.log_prob(x_input)
            if log_probs.ndim > 2:
                log_probs = log_probs.sum(dim=-1)  # sum over feature/event dimensions
        else:
            output = self.emission_module(valid_context)
            if isinstance(output, tuple):
                # Gaussian / Normal
                mu, var = output
                var = var.clamp(min=self.min_covar)
                dist = torch.distributions.Normal(mu, var.sqrt())
                log_probs = dist.log_prob(seq_tensor.unsqueeze(1)).sum(dim=-1)
            else:
                # Categorical / Discrete
                logits = output
                if seq_tensor.dtype != torch.long:
                    raise TypeError("seq_tensor must contain integer indices for categorical emission.")
                log_probs_all = F.log_softmax(logits, dim=-1)
                log_probs = log_probs_all[torch.arange(T, device=self.device), seq_tensor.long()]

        # Ensure shape [T, K] for downstream
        if log_probs.ndim == 1:
            log_probs = log_probs.unsqueeze(-1)

        # --- Transition & duration ---
        transition_logits = self.transition_module.log_matrix(context=valid_context) / max(temp_transition, self.min_covar)
        log_duration = self.duration_module.log_matrix(context=valid_context) / max(temp_duration, self.min_covar)

        # --- Duration helper ---
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
                prev = alpha[t - 1].unsqueeze(1) + transition_logits + dur_term(t).unsqueeze(0)
                alpha[t] = torch.logsumexp(prev, dim=0) + log_probs[t]
            decoded = torch.argmax(alpha, dim=-1)

        elif algorithm == "viterbi":
            delta = torch.full((T, K), -torch.inf, device=self.device, dtype=DTYPE)
            psi = torch.zeros((T, K), dtype=torch.long, device=self.device)
            delta[0] = log_probs[0]
            for t in range(1, T):
                scores = delta[t - 1].unsqueeze(1) + transition_logits + dur_term(t).unsqueeze(0)
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
        verbose: bool = False,
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
                elif theta.shape[0] == sum(X.lengths):  # flattened
                    split = torch.split(theta, X.lengths)
                    valid_theta = [s for s in split]
                else:
                    # fallback for timestep-wise [T,H] (only if single sequence)
                    valid_theta = [theta[:L] for L in X.lengths]
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
        for i, (alpha, L) in enumerate(zip(alpha_list, X.lengths)):
            if L == 0:
                ll_list.append(torch.tensor(neg_inf, device=device, dtype=DTYPE))
                continue

            # Optional masking (non-destructive)
            if hasattr(X, "masks") and X.masks is not None:
                mask = X.masks[i].to(device=device, dtype=DTYPE)
                alpha = alpha + torch.log(mask + EPS)

            # Ensure alpha shape: [T, K, Dmax] or [T,K]
            assert alpha.shape[0] >= L, f"alpha shape {alpha.shape} inconsistent with L={L}"
            if alpha.ndim == 3:
                ll_seq = torch.logsumexp(alpha[L-1], dim=(0,1))
            else:
                ll_seq = torch.logsumexp(alpha[L-1], dim=0)

            ll_list.append(ll_seq)

        X.log_likelihoods = torch.stack(ll_list)

        if verbose:
            ll = X.log_likelihoods
            msg = f"[compute_ll] seqs={B}, min={ll.min():.4f}, max={ll.max():.4f}, mean={ll.mean():.4f}"
            if logger is not None:
                logger.info(msg)
            else:
                print(msg)

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

        # --- Emission log-probs ---
        pdf = getattr(self, "_params", {}).get("emission_pdf", None)
        if pdf is None or not hasattr(pdf, "log_prob"):
            raise RuntimeError("Emission PDF must be initialized with log_prob method.")

        x_input = seq.unsqueeze(1) if seq.ndim == 2 else seq
        log_B = pdf.log_prob(x_input)
        if log_B.ndim > 2:
            log_B = log_B.flatten(start_dim=1).sum(dim=1)
        if log_B.ndim == 1:
            log_B = log_B.unsqueeze(-1)

        # --- Module-aware base parameters ---
        init_logits = self.initial_module.log_matrix(context=theta).to(device=self.device, dtype=DTYPE)
        dur_logits = self.duration_module.log_matrix(context=theta).to(device=self.device, dtype=DTYPE)
        trans_logits = self.transition_module.log_matrix(context=theta).to(device=self.device, dtype=DTYPE)

        # Apply temperature scaling
        trans_logits = trans_logits / max(temp_transition, self.min_covar)
        dur_logits = dur_logits / max(temp_duration, self.min_covar)

        # --- Forward recursion ---
        V = torch.full((T, K), neg_inf, device=self.device, dtype=DTYPE)
        V[0] = log_B[0] + torch.log_softmax(init_logits, dim=0)

        for t in range(1, T):
            max_d = min(Dmax, t + 1)
            dur_idx = torch.arange(max_d, device=self.device)
            scores = []

            for d in dur_idx:
                start = t - d + 1
                emit_sum = log_B[start:t+1].sum(dim=0)

                # Universal dur_score handling: works for [K, Dmax] or [Dmax]
                if dur_logits.ndim == 2:
                    dur_score = dur_logits[:, d]
                else:
                    dur_score = dur_logits[d].expand(K)

                if start == 0:
                    trans_score = torch.zeros(K, device=self.device, dtype=DTYPE)
                else:
                    trans_score = torch.logsumexp(V[start-1].unsqueeze(1) + trans_logits, dim=0)

                scores.append(trans_score + dur_score + emit_sum)

            V[t] = torch.logsumexp(torch.stack(scores, dim=0), dim=0)

        ll = torch.logsumexp(V[-1], dim=-1).detach().cpu()
        return ll

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

