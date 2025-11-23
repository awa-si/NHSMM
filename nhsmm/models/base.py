# nhsmm/models/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Any, Literal, Dict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nhsmm.context import ContextEncoder
from nhsmm.constants import DEBUG, DTYPE, EPS, HSMMError, logger, MAX_LOGITS
from nhsmm.tools import utils, constraints, SeedGenerator, ConvergenceTracker
from nhsmm.distributions import Categorical, Initial, Emission, Duration, Transition

torch.distributions.Categorical = Categorical


class HSMM(nn.Module, ABC):
    """
    Hidden Semi-Markov Model (HSMM) base class.

    Supports variable-length sequences with optional context or neural modulation 
    of initial, duration, transition, and emission parameters. Provides GPU-compatible, 
    batched computations for forward, backward, and Viterbi decoding.

    Core Components:
        - `initial_module`: Log-probabilities of initial states.
        - `duration_module`: Log-duration probabilities per state.
        - `transition_module`: State-to-state transition log-probabilities.
        - `emission_module`: Observation likelihoods (parametric or neural).
        - Optional `ContextEncoder` for context-conditioned parameters.

    Key Methods:
        - `_forward(X, theta)`: Log-probabilities α[t,k,d] for ending in state `k` at time `t` with duration `d`.
        - `_backward(X, theta)`: Log-probabilities β[t,k,d] for observations from time `t` onward.
        - `_compute_state_posteriors(X, theta)`: Returns γ (state marginals), ξ (transitions), η (state-duration posteriors).
        - `_viterbi(X, theta, duration_weight)`: Most likely state sequence, optionally weighting durations.

    Notes:
        - Handles discrete (Categorical) and continuous (Normal/MultivariateNormal) emissions.
        - Fully batched and GPU-ready.
        - Extensible for custom neural or context-modulated emission modules.

    Attributes:
        n_states: Number of hidden states (K)
        max_duration: Maximum allowed state duration (Dmax)
        device: Torch device (CPU/GPU)
    """

    def __init__(
        self,
        n_states: int,
        n_features: int,
        max_duration: int,
        temperature: float = 1.0,
        seed: Optional[int] = None,
        modulate_var: bool = False,
        alpha: Optional[float] = 1.0,
        emission_type: str = "gaussian",
        transition_type: Any = constraints.Transitions.ERGODIC,
        min_covar: Optional[float] = 1e-6,
        hidden_dim: Optional[int] = None,
        context_dim: Optional[int] = None,
        device: Optional[torch.device] = None,

        # HSMM encoder params
        encoder: Optional[nn.Module] = None,
        pool: Literal["mean", "last", "max", "attn", "mha"] = "mean",
        embed_dim: Optional[int] = None,
        precompute: bool = True,
        dropout: float = 0.0,
        debug: bool = False,
        n_heads: int = 4,
    ):
        super().__init__()

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed or SeedGenerator(seed).seed
        self.emission_type = emission_type
        self.max_duration = max_duration
        self.temperature = temperature
        self.n_features = n_features
        self.precompute = precompute
        self.n_states = n_states
        self.alpha = alpha
        self.debug = debug

        self._params: Dict[str, Any] = {}

        # ---------------- Seed ----------------
        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

        # ---------------- Encoder setup ----------------
        self.encoder: Optional[ContextEncoder] = None
        self._context: Optional[torch.Tensor] = None
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim or self.context_dim

        if encoder is not None:
            # Wrap encoder in ContextEncoder if not already
            if isinstance(encoder, ContextEncoder):
                self.encoder = encoder
            else:
                self.encoder = ContextEncoder(
                    encoder=encoder,
                    pool=pool,
                    n_heads=n_heads,
                    dropout=dropout,
                    debug=debug,
                )

            # Infer context/output dimension
            if hasattr(self.encoder, "out_dim") and getattr(self.encoder, "out_dim") is not None:
                self.context_dim = self.encoder.out_dim
            else:
                # Dummy forward to safely infer
                self.encoder.eval()
                with torch.no_grad():
                    dummy_seq_len = 16  # short dummy length
                    dummy_in = torch.zeros(
                        1, dummy_seq_len, n_features,
                        device=self.device, dtype=DTYPE
                    )
                    dummy_out, ctx, _ = self.encoder(dummy_in, return_context=True, return_sequence=True)
                    self.context_dim = ctx.shape[-1] if ctx is not None else dummy_out.shape[-1]
                # Restore training mode
                if getattr(self.encoder, "training", True):
                    self.encoder.train()

            # Set hidden_dim consistently
            self.hidden_dim = self.context_dim if hidden_dim is None else hidden_dim

        # ---------------- Initialize modules ----------------
        self._init_modules(
            transition_type=transition_type,
            emission_type=emission_type,
            modulate_var=modulate_var,
            max_duration=max_duration,
            temperature=temperature,
            min_covar=min_covar,
            device=self.device,
        )

        self.to(self.device)

    def _init_modules(
        self,
        device: torch.device | str,
        dof: float = 5.0,
        n_states: int = 4,
        scale: float = 1.0,
        n_features: int = 1,
        max_duration: int = 30,
        temperature: float = 1.0,
        modulate_var: bool = False,
        adaptive_scale: bool = True,
        spatial_adapter: bool = False,
        temporal_adapter: bool = False,
        min_covar: Optional[float] = 1e-6,
        init_mode_transition: str = "diag_bias",
        init_mode_duration: str = "uniform",
        init_mode_initial: str = "uniform",
        init_mode_emission: str = "data",
        transition_type: str = "ergodic",
        emission_type: str = "gaussian",
        cache_limit: int = 32,
        debug: bool = False,
    ):
        """Initialize all HSMM modules with consistent device, context, and precomputation."""

        device, debug = self.device, self.debug

        # ---------------- Initial ----------------
        self.initial_module = Initial(
            n_states=self.n_states,
            init_mode=init_mode_initial,
            context_dim=self.context_dim,
            hidden_dim=self.hidden_dim,
            temperature=temperature,
            cache_limit=cache_limit,
            debug=debug
        ).to(device)

        # ---------------- Emission ----------------
        self.emission_module = Emission(
            n_states=self.n_states,
            n_features=self.n_features,
            temporal_adapter=temporal_adapter,
            spatial_adapter=spatial_adapter,
            adaptive_scale=adaptive_scale,
            emission_type=emission_type,
            modulate_var=modulate_var,
            context_dim=self.context_dim,
            hidden_dim=self.hidden_dim,
            temperature=temperature,
            min_covar=min_covar,
            seed=self.seed,
            debug=debug,
            dof=dof,
        ).to(device)

        # ---------------- Duration ----------------
        self.duration_module = Duration(
            n_states=self.n_states,
            init_mode=init_mode_duration,
            max_duration=max_duration,
            context_dim=self.context_dim,
            hidden_dim=self.hidden_dim,
            temperature=temperature,
            cache_limit=cache_limit,
            debug=debug,
        ).to(device)

        # ---------------- Transition ----------------
        self.transition_module = Transition(
            n_states=self.n_states,
            init_mode=init_mode_transition,
            transition_type=transition_type,
            context_dim=self.context_dim,
            hidden_dim=self.hidden_dim,
            temperature=temperature,
            cache_limit=cache_limit,
            gate_factor=0.5,
            debug=debug,
        ).to(device)

        # ---------------- Optional Precompute ----------------
        # if self.precompute and self.context_dim is not None:
            # with torch.no_grad():
                # dummy_ctx = torch.zeros(1, self.context_dim, device=device)
                # self.initial_module(dummy_ctx)
                # self.duration_module(dummy_ctx)
                # self.transition_module(dummy_ctx)
                # self.emission_module(dummy_ctx)

        # ---------------- Initialize PDFs ----------------
        try:
            self._params.update({
                "initial_pdf": self.initial_module.initialize(mode=init_mode_initial),
                "duration_pdf": self.duration_module.initialize(mode=init_mode_duration),
                "transition_pdf": self.transition_module.initialize(mode=init_mode_transition),
                "emission_pdf": self.emission_module.initialize(mode=init_mode_emission),
            })
        except Exception as e:
            raise RuntimeError(f"Failed to initialize HSMM PDFs: {e}") from e

        # ---------------- Debug Logging ----------------
        if debug:
            logger.debug(
                f"HSMM modules initialized on {device}: "
                f"n_states={self.n_states}, n_features={self.n_features}, "
                f"context_dim={self.context_dim}, emission={self.emission_type}, "
                f"max_duration={self.max_duration}, precompute={self.precompute}"
            )

    def _align_theta(self, theta, seq_len: int):
        if theta is None:
            return None

        device = self.device

        # ---- CASE: list of tensors (concat along feature dim) ----
        if isinstance(theta, list):
            if not theta:
                return None
            if not all(torch.is_tensor(t) for t in theta):
                raise TypeError("All elements in theta list must be torch.Tensor")

            aligned = []
            for t in theta:

                # Fix: 1D -> treat as feature vector [F], not [T]
                if t.ndim == 1:
                    t = t.unsqueeze(0)  # [1, F]

                if t.ndim != 2:
                    raise TypeError(f"List tensor must be 1D or 2D, got {t.ndim}")

                T, F = t.shape
                pad = torch.zeros(seq_len, F, device=device, dtype=DTYPE)
                pad[:min(T, seq_len)] = t[:min(T, seq_len)]
                aligned.append(pad)

            # concat features
            return torch.cat(aligned, dim=-1).unsqueeze(0)  # [1, seq_len, F_total]

        # ---- CASE: tensor inputs ----
        ndim = theta.ndim

        if ndim == 1:
            return theta.unsqueeze(0).expand(seq_len, -1).contiguous()

        if ndim == 2:
            T, F = theta.shape
            pad = torch.zeros(seq_len, F, device=device, dtype=DTYPE)
            pad[:min(T, seq_len)] = theta[:min(T, seq_len)]
            return pad

        if ndim == 3:
            B, T, F = theta.shape
            pad = torch.zeros(B, seq_len, F, device=device, dtype=DTYPE)
            pad[:, :min(T, seq_len)] = theta[:, :min(T, seq_len)]
            return pad

        raise TypeError(f"Unsupported theta dimension {ndim}, expected 1, 2, 3, or list of tensors")

    @torch.no_grad()
    def _encode_observations(
        self,
        sequence: torch.Tensor,
        pool: Optional[str] = None,
        detach: bool = True,
        store: bool = True,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Encode a single sequence into:
          - canonical context: [1, 1, H] (for Initial/Duration/Transition)
          - per-timestep context: [T, H] (for Emission)

        Args:
            sequence: [T] or [T,F] tensor
            pool: optional pooling override for encoder
            detach: whether to detach context from computation graph
            store: whether to store contexts in self

        Returns:
            context_aligned: [T,H] (per-timestep)
            ctx_canonical: [1,1,H] (pooled canonical)
        """
        if self.encoder is None or sequence.numel() == 0:
            if store:
                self._context = None
                self._context_aligned = None
            return None, None

        original_pool = getattr(self.encoder, "pool", None)
        if pool is not None and hasattr(self.encoder, "pool"):
            self.encoder.pool = pool

        try:
            # Ensure [T,F]
            seq_tensor = sequence.unsqueeze(-1) if sequence.ndim == 1 else sequence
            T, F = seq_tensor.shape

            # Mask for valid timesteps, batch dim for encoder
            mask = torch.ones(1, T, dtype=torch.bool)

            # Forward pass
            _ = self.encoder(seq_tensor.unsqueeze(0), return_context=True, mask=mask)
            ctx_canonical = self.encoder.get_context(detach=detach)  # [1,1,H]

            # Per-timestep context for emission
            context_aligned = ctx_canonical.squeeze(0).expand(T, -1)  # [T,H]

        finally:
            if hasattr(self.encoder, "pool"):
                self.encoder.pool = original_pool

        if store:
            self._context = ctx_canonical           # canonical for Initial/Duration/Transition
            self._context_aligned = context_aligned  # per-timestep for Emission

        return context_aligned, ctx_canonical

    def _prepare_observations(
        self,
        X: torch.Tensor,
        theta: Optional[torch.Tensor] = None,
    ) -> utils.Observations:
        """
        Convert a raw sequence + optional canonical context into an Observations object.
        Handles cached per-timestep context for Emission and canonical context for 
        Initial/Duration/Transition. Supports theta=None.

        Args:
            X: [T,F] or [T] tensor
            theta: optional canonical context [1,1,H] or None

        Returns:
            utils.Observations with:
                sequence: list of [T,F]
                lengths: list of int
                log_probs: list of [T,K]
                context: list of [T,H]
                mask: list of [T,1]
        """
        # Ensure 2D
        seq = X if X.ndim > 1 else X.unsqueeze(-1)
        T, F = seq.shape
        mask = torch.ones(T, 1, dtype=torch.bool, device=seq.device)

        # --- Determine contexts ---
        if theta is None:
            ctx_canonical = getattr(self, "_context_canonical", None)
            context_aligned = getattr(self, "_context", None)

            if ctx_canonical is None or context_aligned is None:
                context_aligned, ctx_canonical = self._encode_observations(seq, store=True)
                self._context = context_aligned
                self._context_canonical = ctx_canonical
        else:
            # Provided canonical context; expand to per-timestep
            if theta.ndim == 3:  # [1,1,H]
                ctx_canonical = theta
                context_aligned = theta.squeeze(0).expand(T, -1)
            elif theta.ndim == 2:  # [1,H]
                ctx_canonical = theta.unsqueeze(0)
                context_aligned = theta.expand(T, -1)
            else:
                raise ValueError(f"Unsupported theta shape {theta.shape}")

        # --- Compute emission log-probs ---
        K = self.n_states
        dist_type = self.emission_module.dist_type
        log_probs = torch.empty(T, K, dtype=DTYPE, device=seq.device)

        if dist_type in (torch.distributions.Categorical, torch.distributions.Bernoulli):
            dist = self.emission_module.forward(context=context_aligned, return_dist=True)
            logits = dist.logits
            seq_cat = seq[:, 0].long()
            one_hot = F.one_hot(seq_cat, num_classes=logits.shape[-1]).float()
            log_probs = torch.einsum("tf,kf->tk", one_hot, logits)
        elif callable(dist_type) or dist_type == torch.distributions.Independent:
            means = self.emission_module._emission_means
            stds = self.emission_module._emission_covs.diagonal(dim1=-2, dim2=-1).sqrt()
            seq_exp = seq.unsqueeze(1).expand(T, K, F)
            means_exp = means.unsqueeze(0)
            stds_exp = stds.unsqueeze(0)
            log_norm = -0.5 * torch.log(2 * math.pi * stds_exp**2)
            log_exp = -0.5 * ((seq_exp - means_exp) ** 2 / (stds_exp**2))
            log_probs = (log_norm + log_exp).sum(-1)
        else:
            raise NotImplementedError(f"Unsupported emission type: {dist_type}")

        # --- Wrap in lists for Observations ---
        return utils.Observations(
            sequence=[seq],
            lengths=[T],
            log_probs=[log_probs],
            context=[context_aligned],
            mask=[mask],
        )

    def _forward(
        self,
        X: utils.Observations,
        theta: Optional[list[Optional[torch.Tensor]] | torch.Tensor] = None
    ) -> list[torch.Tensor]:
        """
        Vectorized Forward algorithm for multiple sequences using context-modulated
        Initial, Duration, and Transition distributions.

        Returns:
            list[Tensor]: One tensor per sequence of shape [T, K, Dmax].
        """
        device, K, Dmax = self.device, self.n_states, self.max_duration
        neg_inf = torch.finfo(DTYPE).min / 2.0
        alpha_list = []

        for i, (log_emissions, seq_len) in enumerate(zip(X.log_probs, X.lengths)):
            if seq_len == 0:
                alpha_list.append(torch.full((0, K, Dmax), neg_inf, device=device, dtype=DTYPE))
                continue

            T = seq_len

            # --- Select per-sequence context ---
            ctx = None
            if theta is not None:
                ctx = theta[i] if isinstance(theta, list) else theta

            # --- Context-modulated logits ---
            initial_logits = self.initial_module.log_matrix(context=ctx)        # [K]
            duration_logits = self.duration_module.log_matrix(context=ctx)      # [K,Dmax]
            transition_logits = self.transition_module.log_matrix(context=ctx)  # [K,K]

            # --- Precompute cumulative sums for segment emissions ---
            cumsum_emit = torch.zeros(T + 1, K, device=device, dtype=DTYPE)
            cumsum_emit[1:] = torch.cumsum(log_emissions, dim=0)

            # --- Initialize α tensor ---
            alpha_tensor = torch.full((T, K, Dmax), neg_inf, device=device, dtype=DTYPE)

            for t in range(T):
                max_d = min(Dmax, t + 1)
                durations = torch.arange(1, max_d + 1, device=device)
                starts = t - durations + 1  # inclusive start indices

                # Segment emission sums: [K, max_d]
                emit_sums = torch.stack([cumsum_emit[t + 1] - cumsum_emit[s] for s in starts], dim=1)

                # --- DEBUG ---
                print("DEBUG _forward (log_matrix) shapes at t=", t)
                print("  initial_logits.shape:", tuple(initial_logits.shape))
                print("  duration_logits.shape:", tuple(duration_logits.shape))
                print("  transition_logits.shape:", tuple(transition_logits.shape))
                print("  emit_sums.shape:", tuple(emit_sums.shape))
                print("  alpha_tensor slice shape (target):", tuple(alpha_tensor[t, :, :max_d].shape))
                print("  K, Dmax, max_d:", K, self.max_duration, max_d)

                if t == 0:
                    # initial step
                    alpha_tensor[t, :, :max_d] = initial_logits.unsqueeze(1) + duration_logits[:, :max_d] + emit_sums
                    continue

                # --- Previous α contributions ---
                valid_mask = starts > 0
                prev_alpha = torch.full((max_d, K), neg_inf, device=device, dtype=DTYPE)

                if valid_mask.any():
                    valid_idx = valid_mask.nonzero(as_tuple=True)[0]
                    prev_vals = torch.logsumexp(alpha_tensor[starts[valid_idx]-1, :, :], dim=2)
                    prev_alpha[valid_idx] = prev_vals

                if (~valid_mask).any():
                    idx = (~valid_mask).nonzero(as_tuple=True)[0]
                    prev_alpha[idx] = (initial_logits.unsqueeze(0).expand(len(idx), -1) + duration_logits[:, :max_d].T[idx])

                # Combine previous α with transition probabilities
                alpha_with_trans = torch.logsumexp(prev_alpha.unsqueeze(2) + transition_logits.unsqueeze(0), dim=1)  # [max_d, K]

                # Add duration scores and segment emissions
                # Use explicit broadcasting to avoid shape mismatch
                # Ensure all tensors have shape [K, max_d]
                duration_slice = duration_logits[:, :max_d]          # [K, max_d]
                alpha_trans_slice = alpha_with_trans.T              # [K, max_d] after transpose
                emit_slice = emit_sums                               # [K, max_d]

                alpha_tensor[t, :, :max_d] = alpha_trans_slice + duration_slice + emit_slice

            alpha_list.append(alpha_tensor)

        return alpha_list

    def _backward(
        self,
        X: utils.Observations,
        theta: Optional[ContextualVariables] = None
    ) -> list[torch.Tensor]:
        """
        Vectorized Backward algorithm for HSMM using context-modulated log matrices.

        Each β[t, k, d] represents the log-probability of generating observations 
        from time t onward, assuming state k started at time t-d+1 with duration d.

        Returns:
            list[Tensor]: One tensor per sequence of shape [T, K, Dmax].
        """
        K, Dmax, device = self.n_states, self.max_duration, self.device
        neg_inf = torch.finfo(DTYPE).min / 2.0

        # --- Context-modulated parameters ---
        initial_logits = self.initial_module.log_matrix(context=theta)       # [K]
        dur_logits_full = self.duration_module.log_matrix(context=theta)     # [K, Dmax]
        transition_logits = self.transition_module.log_matrix(context=theta) # [K, K]

        beta_list = []
        for seq_logp, seq_len in zip(X.log_probs, X.lengths):
            if seq_len == 0:
                beta_list.append(torch.full((0, K, Dmax), neg_inf, device=device, dtype=DTYPE))
                continue

            seq_logp = seq_logp
            T = seq_len

            log_beta = torch.full((T, K, Dmax), neg_inf, device=device, dtype=DTYPE)
            log_beta[-1, :, 0] = 0.0  # terminal condition

            # Precompute cumulative emission sums for segment likelihoods
            cumsum_emit = torch.zeros(T + 1, K, device=device, dtype=DTYPE)
            cumsum_emit[1:] = torch.cumsum(seq_logp, dim=0)

            for t in reversed(range(T - 1)):
                max_d = min(Dmax, T - t)
                ends = t + torch.arange(1, max_d + 1, device=device)  # segment ends

                # Segment emission sums [K, max_d]
                emit_sums = (cumsum_emit[ends] - cumsum_emit[t].unsqueeze(0))  # [max_d, K]
                emit_sums = emit_sums.T  # [K, max_d]

                dur_scores = dur_logits_full[:, :max_d]  # [K, max_d]

                # Transition contribution from next states
                next_beta = log_beta[ends - 1, :, 0]  # [max_d, K]
                beta_next = torch.logsumexp(transition_logits[None, :, :] + next_beta[:, :, None], dim=1)  # [max_d, K]
                beta_next = beta_next.T  # [K, max_d]

                # Combine segment, duration, and transition terms
                log_beta[t, :, 0] = torch.logsumexp(emit_sums + dur_scores + beta_next, dim=1)

                # Handle overlapping durations (carry forward shorter segments)
                if max_d > 1:
                    shift_len = min(max_d - 1, T - t - 1)
                    if shift_len > 0:
                        log_beta[t, :, 1:shift_len + 1] = log_beta[t + 1, :, :shift_len] + seq_logp[t + 1].unsqueeze(-1)

            beta_list.append(log_beta)

        return beta_list

    def _compute_state_posteriors(
        self,
        X: utils.Observations,
        theta: Optional[ContextualVariables] = None
    ) -> Tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """
        Compute HSMM posteriors with context-modulated modules per sequence.

        Returns:
            gamma_list: [T, K] state marginals per sequence
            xi_list: [T-1, K, K] state-to-state transitions per sequence
            eta_list: [T, K, Dmax] state-duration joint per sequence
        """
        K, Dmax = self.n_states, self.max_duration
        B = len(X.sequence)

        gamma_list = []
        xi_list = []
        eta_list = []

        for b in range(B):
            L = X.lengths[b]
            logp = X.log_probs[b].to(dtype=DTYPE)  # [L, K]

            # ---------------- Context handling ----------------
            if theta is None:
                ctx_canonical = ctx_aligned = None
            else:
                # Select sequence context
                ctx_b = theta[b] if isinstance(theta, list) else theta

                # canonical context: [1,1,H]
                if ctx_b.ndim == 3:         # [1,T,H] or [B,T,H]
                    ctx_canonical = ctx_b[:, :1, :].clone()  # [1,1,H]
                elif ctx_b.ndim == 2:       # [T,H]
                    ctx_canonical = ctx_b.mean(dim=0, keepdim=True).unsqueeze(0)  # [1,1,H]
                elif ctx_b.ndim == 1:       # [H]
                    ctx_canonical = ctx_b.unsqueeze(0).unsqueeze(0)  # [1,1,H]

                # per-timestep context: [T,H]
                if ctx_b.ndim == 3:         # [1,T,H]
                    ctx_aligned = ctx_b.squeeze(0)
                elif ctx_b.ndim == 2:       # [T,H]
                    ctx_aligned = ctx_b
                elif ctx_b.ndim == 1:       # [H]
                    ctx_aligned = ctx_b.expand(L, -1)

            # ---------------- Compute log matrices ----------------
            initial_logits = self.initial_module.log_matrix(context=ctx_canonical)
            transition_logits = self.transition_module.log_matrix(context=ctx_canonical)
            duration_logits = self.duration_module.log_matrix(context=ctx_canonical)

            # remove singleton batch dim if present
            if initial_logits.ndim == 2 and initial_logits.shape[0] == 1:
                initial_logits = initial_logits.squeeze(0)
            if transition_logits.ndim == 3 and transition_logits.shape[0] == 1:
                transition_logits = transition_logits.squeeze(0)
            if duration_logits.ndim == 3 and duration_logits.shape[0] == 1:
                duration_logits = duration_logits.squeeze(0)

            # ---------------- Forward / backward ----------------
            obs = utils.Observations(sequence=[logp], log_probs=[logp], lengths=[L])
            alpha, beta = self._forward(obs, theta=ctx_aligned)[0], self._backward(obs, theta=ctx_aligned)[0]

            # ---------------- State-duration posterior ----------------
            eta_log = alpha + beta
            eta_log_flat = eta_log.view(L, -1)
            eta_log_flat = eta_log_flat - torch.logsumexp(eta_log_flat, dim=-1, keepdim=True)
            eta_soft = eta_log_flat.view(L, K, Dmax).exp()  # [L, K, Dmax]

            # ---------------- State marginal ----------------
            gamma = eta_soft.sum(dim=-1)
            gamma = gamma / gamma.sum(dim=-1, keepdim=True).clamp_min(EPS)

            # ---------------- State-to-state transitions ----------------
            if L <= 1:
                xi = torch.zeros((0, K, K), dtype=DTYPE)
            else:
                alpha_prev = torch.logsumexp(alpha[:-1], dim=2)  # [L-1, K]
                beta_next = torch.logsumexp(beta[1:], dim=2)     # [L-1, K]

                trans = transition_logits
                if trans.ndim == 3:
                    trans = trans.squeeze(0)

                log_xi = alpha_prev[:, :, None] + trans[None, :, :] + beta_next[:, None, :]
                log_xi = log_xi - torch.logsumexp(log_xi.view(L-1, -1), dim=-1).view(L-1, 1, 1)
                xi = log_xi.exp()

            gamma_list.append(gamma)
            eta_list.append(eta_soft)
            xi_list.append(xi)

        return gamma_list, xi_list, eta_list

    def _model_params(
        self,
        X: Optional[utils.Observations] = None,
        theta: Optional[torch.Tensor] = None,
        theta_scale: float = 0.1,
        mode: str = "estimate",
        max_iter: int = 50,
        iter_idx: int = 0,
    ) -> dict[str, Any]:
        """
        Compute HSMM model parameters with EM-style estimate, sample-mode collapse, and neural updates.
        Uses cached contexts where available.

        Returns a dict of PDFs: emission, initial, duration, transition.
        """
        eps_collapse = 1e-3
        seq_len = sum(getattr(X, "lengths", [1]))
        aligned_theta = self._align_theta(theta, seq_len) if theta is not None else None

        # ---------------- Flatten sequences ----------------
        if X is not None:
            all_X = torch.cat([s for s in getattr(X, "sequence", [X]) if s.numel() > 0], dim=0)
            if all_X.numel() == 0:
                all_X = torch.zeros(1, self.n_features, dtype=DTYPE)
        else:
            all_X = torch.zeros(1, self.n_features, dtype=DTYPE)

        # Adaptive α-decay
        α_min, α_max = 0.05, self.alpha
        α = float(max(α_min, α_max * (1.0 - iter_idx / max_iter)))

        # ---------------- Helpers ----------------
        def collapse_logits(logits: torch.Tensor, dim: int) -> torch.Tensor:
            probs = logits.exp()
            probs = probs / probs.sum(dim=dim, keepdim=True).clamp_min(EPS)
            dir_alpha = probs * α + EPS
            shape = dir_alpha.shape
            perm = list(range(dir_alpha.ndim))
            perm[dim], perm[-1] = perm[-1], perm[dim]
            flat = dir_alpha.permute(perm).reshape(-1, shape[dim])
            samples = torch.distributions.Dirichlet(flat).rsample()
            samples = samples.reshape(*[shape[i] for i in perm]).permute(*perm)
            samples = samples / samples.sum(dim=dim, keepdim=True)
            samples = torch.where(samples > eps_collapse, samples, torch.full_like(samples, 1.0 / shape[dim]))
            return torch.log(samples + EPS)

        def safe_sum(lst: list[Optional[torch.Tensor]]) -> Optional[torch.Tensor]:
            tensors = [x for x in lst if x is not None and x.numel() > 0]
            return torch.cat(tensors, dim=0).sum(dim=0) if tensors else None

        # ---------------- Ensure contexts ----------------
        if aligned_theta is None:
            if hasattr(self, "_context") and self._context is not None:
                aligned_theta = self._context
            else:
                # Compute canonical context from all_X
                _, aligned_theta = self._encode_observations(all_X, store=True)

        # ---------------- Mode: sample ----------------
        if mode == "sample":
            with torch.no_grad():
                # Initial
                init_pdf = self.initial_module.forward(context=aligned_theta, return_dist=True)
                self.initial_module.update(new_logits=collapse_logits(init_pdf.logits, dim=0).exp(), from_probs=True)

                # Transition
                trans_pdf = self.transition_module.forward(context=aligned_theta, return_dist=True)
                self.transition_module.update(new_logits=collapse_logits(trans_pdf.logits, dim=1).exp(), from_probs=True)

                # Duration
                dur_pdf = self.duration_module.forward(context=aligned_theta, return_dist=True)
                self.duration_module.update(new_logits=collapse_logits(dur_pdf.logits, dim=1).exp(), from_probs=True)

                # Emission
                try:
                    emission_pdf = self.emission_module.forward(context=self._context_aligned, return_dist=True)
                except Exception as e:
                    logger.warning(f"Emission module failed forward pass: {e}, initializing.")
                    emission_pdf = self.emission_module.initialize(
                        X=all_X, context=self._context_aligned, theta=aligned_theta, theta_scale=theta_scale
                    )

        # ---------------- Mode: estimate ----------------
        elif mode == "estimate":
            if X is None or not isinstance(X, utils.Observations):
                raise RuntimeError("Observations X required for estimate mode.")

            gamma_list, xi_list, eta_list = self._compute_state_posteriors(X, theta=aligned_theta)

            # Posterior counts or fallback to buffer
            trans_counts_tmp = safe_sum(xi_list)
            trans_counts = trans_counts_tmp if trans_counts_tmp is not None else self.transition_module._mod_logits_buffer.exp()

            dur_counts_tmp = safe_sum(eta_list)
            dur_counts = dur_counts_tmp if dur_counts_tmp is not None else self.duration_module._mod_logits_buffer.exp()

            init_counts_tmp = safe_sum([g[0] for g in gamma_list])
            init_counts = init_counts_tmp if init_counts_tmp is not None else self.initial_module._mod_logits_buffer.exp()

            # α-blended log-normalization
            blend_init = constraints.log_normalize(
                torch.log(init_counts + EPS) * α + (1 - α) * self.initial_module._mod_logits_buffer, dim=0
            )
            blend_trans = constraints.log_normalize(
                torch.log(trans_counts + EPS) * α + (1 - α) * self.transition_module._mod_logits_buffer, dim=1
            )
            blend_dur = constraints.log_normalize(
                torch.log(dur_counts + EPS) * α + (1 - α) * self.duration_module._mod_logits_buffer, dim=1
            )

            # Update modules with EM + optional neural posterior
            for module, logits in [
                (self.initial_module, blend_init),
                (self.transition_module, blend_trans),
                (self.duration_module, blend_dur),
            ]:
                posterior = logits.exp() if any(p.requires_grad for p in module.parameters()) else None
                module.update(new_logits=logits.exp(), posterior=posterior, from_probs=True)

            # Emission
            all_gamma = torch.cat([g for g in gamma_list if g is not None and g.numel() > 0], dim=0)
            try:
                emission_pdf = self.emission_module.forward(context=self._context_aligned, return_dist=True)
            except Exception as e:
                logger.warning(f"Emission module failed forward pass: {e}, initializing.")
                emission_pdf = self.emission_module.initialize(
                    X=all_X, posterior=all_gamma, context=self._context_aligned, theta=aligned_theta, theta_scale=theta_scale
                )
        else:
            raise ValueError(f"Unsupported mode '{mode}'.")

        # ---------------- Return distributions ----------------
        initial_pdf = self.initial_module.forward(context=aligned_theta, return_dist=True)
        duration_pdf = self.duration_module.forward(context=aligned_theta, return_dist=True)
        transition_pdf = self.transition_module.forward(context=aligned_theta, return_dist=True)

        return {
            "emission_pdf": emission_pdf,
            "initial_pdf": initial_pdf,
            "duration_pdf": duration_pdf,
            "transition_pdf": transition_pdf,
        }


    # HSMM EM
    @torch.no_grad()
    def _compute_emit_log(
        self,
        X: utils.Observations,
        theta: Optional[torch.Tensor] = None,
        verbose: bool = False,
        device_output: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Vectorized per-sequence log-likelihood computation for an HSMM with optional context.
        Handles variable-length sequences, empty sequences, and zero-length context.

        Args:
            X: Observations object containing sequences and lengths.
            theta: Optional context tensor for sequences.
            verbose: If True, logs min/max/mean of log-likelihoods.
            device_output: Optional device for output tensor.

        Returns:
            Tensor of shape [B], one log-likelihood per sequence.
        """
        device_output = device_output or torch.device("cpu")
        device, B = self.device, len(X.sequence)
        neg_inf = torch.finfo(DTYPE).min / 2.0

        # Handle empty batch
        if B == 0:
            X.log_likelihoods = torch.full((0,), neg_inf, device=device_output, dtype=DTYPE)
            return X.log_likelihoods

        # Align theta for batch if provided
        aligned_theta = self._align_theta(theta, sum(X.lengths)) if theta is not None else None

        # Forward pass (vectorized HSMM)
        alpha_list = self._forward(X, theta=aligned_theta)

        # Determine max sequence length and alpha shape
        max_len = max(X.lengths)
        n_states, n_durations = alpha_list[0].shape[1:3] if alpha_list and alpha_list[0].ndim == 3 else (self.n_states, 1)

        # Preallocate padded alpha tensor
        alpha_padded = torch.full(
            (B, max_len, n_states, n_durations), neg_inf, dtype=DTYPE, device=device
        )

        # Pad each sequence
        for b, (alpha, L) in enumerate(zip(alpha_list, X.lengths)):
            if L > 0:
                alpha_padded[b, :L] = alpha[:L]

        # Compute per-sequence log-likelihood using logsumexp over states and durations
        lengths_tensor = torch.tensor(X.lengths, device=device)
        valid_mask = lengths_tensor > 0
        ll = torch.full((B,), neg_inf, dtype=DTYPE, device=device)
        if valid_mask.any():
            final_alpha = alpha_padded[valid_mask, lengths_tensor[valid_mask] - 1]  # [valid_B, n_states, n_durations]
            ll[valid_mask] = torch.logsumexp(final_alpha.flatten(start_dim=1), dim=1)

        # Move to output device
        ll = ll.to(device_output)
        X.log_likelihoods = ll

        if verbose:
            logger.info(
                f"[compute_emit_log] seqs={B}, min={ll.min():.4f}, max={ll.max():.4f}, mean={ll.mean():.4f}"
            )

        return X.log_likelihoods

    def fit(
        self,
        X: torch.Tensor,
        n_init: int = 1,
        tol: float = 1e-4,
        max_iter: int = 20,
        patience: int = 1,
        ignore_conv: bool = False,
        theta: Optional[torch.Tensor] = None,
        update_rate_max: float = 0.8,
        update_rate_min: float = 0.1,
        adapt_factor: float = 10.0,
        plot_conv: bool = False,
        verbose: bool = True,
    ):
        neural_update = getattr(self, "encoder", None) is not None and any(
            p.requires_grad for p in self.emission_module.parameters()
        )

        # Encode context if no theta provided
        if theta is None and getattr(self, "encoder", None):
            context_aligned, ctx_canonical = self._encode_observations(X)
            self._context = context_aligned      # per-timestep context for Emission
            self._context_canonical = ctx_canonical  # canonical context for Initial/Duration/Transition
            theta = ctx_canonical

        # Prepare observations using canonical context
        X_valid = self._prepare_observations(X, theta=theta)
        aligned_theta = self._align_theta(theta, sum(X_valid.lengths)) if theta is not None else None

        B = len(X_valid.sequence)
        max_len = max(X_valid.lengths) if B > 0 else 0
        F_dim = X_valid.sequence[0].shape[-1] if B > 0 else 0

        # ---------------- Prepare padded tensors ----------------
        seq_tensor = torch.zeros((B, max_len, F_dim), dtype=DTYPE)
        mask = torch.zeros(B, max_len, dtype=DTYPE)
        for b, seq in enumerate(X_valid.sequence):
            L = seq.shape[0]
            seq_tensor[b, :L] = seq
            mask[b, :L] = 1.0
        mask_exp = mask.unsqueeze(-1)

        # ---------------- Convergence tracker ----------------
        self._convergence = ConvergenceTracker(
            tol=tol,
            rel_tol=tol,
            n_init=n_init,
            max_iter=max_iter,
            patience=patience,
            verbose=verbose,
        )
        best_score = -float("inf")

        for run_idx in range(n_init):
            if verbose:
                print(f"\n=== Run {run_idx + 1}/{n_init} ===")

            mode = "sample" if run_idx > 0 else "estimate"
            params = self._model_params(X_valid, theta=aligned_theta, mode=mode)

            init_pdf = params["initial_pdf"]
            emission_pdf = params["emission_pdf"]
            duration_pdf = params["duration_pdf"]
            transition_pdf = params["transition_pdf"]

            # ---------------- Initial likelihood ----------------
            X_valid.log_probs = emission_pdf.log_prob(seq_tensor.unsqueeze(2)) * mask_exp
            prev_ll = self._compute_emit_log(X_valid).sum().item()
            self._convergence.update(prev_ll, 0, run_idx)

            for it in range(1, max_iter + 1):
                # -------- Compute posteriors --------
                gamma_list, xi_list, eta_list = self._compute_state_posteriors(X_valid, theta=aligned_theta)
                gamma_tensor = (
                    torch.nn.utils.rnn.pad_sequence(gamma_list, batch_first=True)
                    if gamma_list else torch.zeros(B, max_len, self.n_states, dtype=DTYPE)
                )
                xi_tensor = (
                    torch.nn.utils.rnn.pad_sequence([x for x in xi_list if x is not None], batch_first=True)
                    if xi_list else None
                )
                eta_tensor = (
                    torch.nn.utils.rnn.pad_sequence([e for e in eta_list if e is not None], batch_first=True)
                    if eta_list else None
                )

                # -------- Update categorical counts (EM) --------
                init_counts = gamma_tensor.sum((0, 1))
                dur_counts = eta_tensor.sum((0, 1)) if eta_tensor is not None else duration_pdf.logits.exp()
                trans_counts = xi_tensor.sum((0, 1)) if xi_tensor is not None else transition_pdf.logits.exp()

                init_counts /= init_counts.sum().clamp_min(EPS)
                dur_counts /= dur_counts.sum(dim=1, keepdim=True).clamp_min(EPS)
                trans_counts /= trans_counts.sum(dim=1, keepdim=True).clamp_min(EPS)

                # -------- EM alpha decay --------
                α_min, α_max = 0.05, getattr(self, "alpha", 1.0)
                α = max(α_min, α_max * (1.0 - it / max_iter))

                init_pdf = self.initial_module.dist_type(probs=α * init_counts + (1 - α) * init_pdf.probs)
                duration_pdf = self.duration_module.dist_type(probs=α * dur_counts + (1 - α) * duration_pdf.probs)
                transition_pdf = self.transition_module.dist_type(probs=α * trans_counts + (1 - α) * transition_pdf.probs)

                # -------- Flatten sequences for emission update --------
                flat_mask = mask.bool().reshape(-1)
                flat_X = seq_tensor.reshape(-1, F_dim)[flat_mask]
                flat_gamma = gamma_tensor.reshape(-1, self.n_states)[flat_mask]
                flat_theta = self._context.reshape(-1, self._context.shape[-1])[flat_mask] if self._context is not None else None

                # -------- Adaptive update rate --------
                delta_ll = max(
                    (emission_pdf.log_prob(seq_tensor.unsqueeze(2)) * mask_exp / F_dim).sum().item() - prev_ll, 0.0
                )
                adaptive_rate = min(update_rate_max, max(update_rate_min, adapt_factor * delta_ll))
                update_rate_final = α * adaptive_rate + (1 - α) * update_rate_min

                # -------- Update emission module (EM + neural) --------
                if neural_update:
                    self.emission_module.zero_grad()
                    emission_loss = - (flat_gamma * self.emission_module.log_prob(flat_X)).sum() / flat_gamma.sum()
                    emission_loss.backward()
                    with torch.no_grad():
                        for p in self.emission_module.parameters():
                            if p.grad is not None:
                                p.data.add_(update_rate_final * p.grad)
                    emission_pdf = self.emission_module.forward(context=flat_theta, return_dist=True)
                else:
                    emission_pdf = self.emission_module.update(
                        X=flat_X,
                        posterior=flat_gamma,
                        theta=flat_theta,
                        update_rate=update_rate_final
                    )

                self._params["emission_pdf"] = emission_pdf

                # -------- Likelihood & convergence --------
                X_valid.log_probs = emission_pdf.log_prob(seq_tensor.unsqueeze(2)) * mask_exp / F_dim
                curr_ll = self._compute_emit_log(X_valid).sum().item()
                delta = curr_ll - prev_ll

                if verbose:
                    print(f"[Iter {it:02d}] LL={curr_ll:.4f} Δ={delta:.3e} (update_rate={update_rate_final:.3f}, α={α:.3f})")

                if self._convergence.update(curr_ll, it, run_idx) and not ignore_conv:
                    if verbose:
                        print(f"[Run {run_idx + 1}] Converged at iteration {it}.")
                    break

                prev_ll = curr_ll

            # -------- Store best parameters --------
            if curr_ll > best_score:
                best_score = curr_ll
                self._params.update({
                    "initial_pdf": self.initial_module.dist_type(probs=init_pdf.probs.clone()),
                    "transition_pdf": self.transition_module.dist_type(probs=transition_pdf.probs.clone()),
                    "duration_pdf": self.duration_module.dist_type(probs=duration_pdf.probs.clone()),
                    "emission_pdf": emission_pdf
                })

        if plot_conv:
            self._convergence.plot()

        return self

    def _map(self, X: utils.Observations) -> list[torch.Tensor]:
        """
        MAP decoding of HSMM sequences using posterior state marginals.
        Returns a list of tensors, each of shape [T].
        """
        gamma_list, _, _ = self._compute_state_posteriors(X)
        results = []

        for gamma in gamma_list:
            if gamma is None or gamma.numel() == 0:
                results.append(torch.empty(0, dtype=torch.long, device=self.device))
                continue

            # Ensure tensor is on the correct device
            gamma = gamma.to(self.device)
            gamma = torch.nan_to_num(gamma, nan=-float("inf"), posinf=-float("inf"), neginf=-float("inf"))

            map_seq = gamma.argmax(dim=-1)
            results.append(map_seq)

        return results

    def _viterbi(
        self,
        X: utils.Observations,
        theta: Optional[torch.Tensor] = None,
        duration_weight: float = 0.0
    ) -> list[torch.Tensor]:
        """
        Vectorized Viterbi decoding for HSMM sequences with batch emission computation.
        Supports variable-length sequences, optional context (theta), and duration weighting.
        Uses emission_module.dist_type for clean distribution handling.
        """
        device, K, Dmax = self.device, self.n_states, self.max_duration
        neg_inf = torch.finfo(DTYPE).min / 2.0
        B = len(X.sequence)

        # --- Context-modulated logits ---
        initial_logits = self.initial_module.log_matrix(context=theta)
        duration_logits = self.duration_module.log_matrix(context=theta)
        transition_logits = self.transition_module.log_matrix(context=theta)

        if initial_logits.ndim == 1:
            initial_logits = initial_logits.unsqueeze(0).expand(B, -1)
        if duration_logits.ndim == 2:
            duration_logits = duration_logits.unsqueeze(0).expand(B, -1, -1)
        if transition_logits.ndim == 2:
            transition_logits = transition_logits.unsqueeze(0).expand(B, -1, -1)

        # --- Duration weighting ---
        if duration_weight > 0:
            dur_idx = torch.arange(1, Dmax + 1, device=device, dtype=DTYPE).view(1, 1, -1)
            dur_mean = (duration_logits.exp() * dur_idx).sum(dim=-1, keepdim=True)
            dur_penalty = -((dur_idx - dur_mean) ** 2) / (2 * (Dmax / 3) ** 2)
            duration_logits = (1 - duration_weight) * duration_logits + duration_weight * dur_penalty

        duration_logits = duration_logits.clamp(-MAX_LOGITS, MAX_LOGITS)
        transition_logits = transition_logits.clamp(-MAX_LOGITS, MAX_LOGITS)

        # --- Pad sequences ---
        lengths = torch.tensor([seq.shape[0] for seq in X.sequence], device=device)
        max_len = int(lengths.max())
        n_features = X.sequence[0].shape[1] if X.sequence[0].ndim > 1 else 1
        seq_tensor = torch.zeros(B, max_len, n_features, device=device, dtype=DTYPE)
        mask = torch.zeros(B, max_len, dtype=torch.bool, device=device)
        for b, seq in enumerate(X.sequence):
            L = seq.shape[0]
            seq_tensor[b, :L] = seq
            mask[b, :L] = True

        # --- Vectorized emission log-probs using dist_type ---
        dist_type = self.emission_module.dist_type
        if dist_type in (torch.distributions.Categorical, torch.distributions.Bernoulli):
            # Discrete emissions
            dist = self.emission_module.forward(context=theta, return_dist=True)
            logits = dist.logits
            seq_cat = seq_tensor[..., 0].long()  # assume first feature encodes categorical
            log_probs_all = F.log_softmax(logits, dim=-1) if logits.ndim == 2 else torch.stack([F.log_softmax(l, dim=-1) for l in logits])
            log_probs = log_probs_all[None, :, :].expand(B, -1, -1)
            log_probs = torch.gather(log_probs, 2, seq_cat.unsqueeze(1).expand(-1, K, -1)).transpose(1,2)

        elif callable(dist_type) or dist_type == torch.distributions.Independent:
            # Continuous emissions (Gaussian, Laplace, etc.)
            means = self.emission_module._emission_means  # [K,D]
            stds = self.emission_module._emission_covs.diagonal(dim1=-2, dim2=-1).sqrt()  # [K,D]
            seq_exp = seq_tensor.unsqueeze(2).expand(-1, max_len, K, -1)  # [B,T,K,D]
            log_norm = -0.5 * torch.log(2 * math.pi * stds**2).unsqueeze(0).unsqueeze(0)
            log_exp = -0.5 * ((seq_exp - means.unsqueeze(0).unsqueeze(0))**2 / stds.unsqueeze(0).unsqueeze(0)**2)
            log_probs = (log_norm + log_exp).sum(-1)  # [B,T,K]

        else:
            # fallback per-sequence precomputed log_probs
            log_probs = torch.stack([X.log_probs[b].to(device) for b in range(B)], dim=0)

        log_probs = log_probs.clamp(min=-MAX_LOGITS)

        # --- Viterbi per sequence ---
        predicted_sequences: list[torch.Tensor] = []
        durations_full = torch.arange(1, Dmax + 1, device=device)

        for b in range(B):
            L = lengths[b].item()
            if L == 0:
                predicted_sequences.append(torch.empty(0, dtype=torch.int64, device=device))
                continue

            emit_log = log_probs[b, :L]  # [T,K]
            cumsum_emit = torch.vstack((torch.zeros((1, K), device=device, dtype=DTYPE),
                                        torch.cumsum(emit_log, dim=0)))  # [L+1,K]

            V = torch.full((L, K), neg_inf, device=device, dtype=DTYPE)
            back_ptr = torch.full((L, K), -1, device=device, dtype=torch.int64)
            best_durations = torch.zeros((L, K), dtype=torch.int64, device=device)

            for t in range(L):
                max_d = min(Dmax, t + 1)
                durations = durations_full[:max_d]
                starts = t - durations + 1
                emit_sums = (cumsum_emit[t + 1].unsqueeze(0) - cumsum_emit[starts]).T  # [K,max_d]

                dur_scores = duration_logits[b, :, :max_d]

                if t == 0:
                    scores = initial_logits[b].unsqueeze(1) + dur_scores + emit_sums
                    best_score, best_idx = scores.max(dim=1)
                    V[t] = best_score
                    best_durations[t] = durations[best_idx]
                    back_ptr[t] = -1
                else:
                    prev_scores_base = V[torch.clamp(starts - 1, min=0)]
                    mask_start0 = (starts == 0).unsqueeze(1).expand(-1, K)
                    prev_scores_base = prev_scores_base.masked_fill(mask_start0, 0.0)

                    prev_scores = prev_scores_base.unsqueeze(2) + transition_logits[b].unsqueeze(0)
                    prev_max, prev_arg = prev_scores.max(dim=1)

                    scores = prev_max.T + dur_scores + emit_sums
                    best_score, best_d_idx = scores.max(dim=1)
                    V[t] = best_score
                    best_durations[t] = durations[best_d_idx]

                    state_idx = torch.arange(K, device=device)
                    prev_arg_selected = prev_arg[best_d_idx, state_idx]
                    back_ptr[t] = torch.where(durations[best_d_idx] == 1,
                                              torch.full_like(prev_arg_selected, -1),
                                              prev_arg_selected)

            # --- Backtrace ---
            t = L - 1
            cur_state = int(torch.argmax(V[t]).item())
            segments = []

            while t >= 0:
                d = int(best_durations[t, cur_state].item())
                start = max(0, t - d + 1)
                segments.append((start, t, cur_state))
                prev_state = int(back_ptr[t, cur_state].item())
                t = start - 1
                cur_state = prev_state if prev_state >= 0 else cur_state

            segments.reverse()
            seq_path = torch.cat([torch.full((end - start + 1,), st, dtype=torch.int64, device=device)
                                  for start, end, st in segments])
            predicted_sequences.append(seq_path[:L])

        return predicted_sequences

    def predict(
        self,
        X: torch.Tensor | list[torch.Tensor],
        algorithm: Literal["map", "viterbi"] = "viterbi",
        context: Optional[torch.Tensor | list[torch.Tensor]] = None,
        transition_temp: float = 1.0,
        duration_temp: float = 1.0,
        device_output: Optional[torch.device] = None,
    ) -> list[torch.Tensor]:
        """
        Predict hidden states using Viterbi or MAP decoding with optional contextual modulation.

        Args:
            X: Input sequence(s) [T,F], [B,T,F], or list of [T_i,F]
            algorithm: "viterbi" or "map"
            context: Optional context tensor/list
            transition_temp: Temperature scaling for transition probabilities
            duration_temp: Temperature scaling for duration probabilities
            device_output: Device for output tensors
        Returns:
            List of tensors with predicted states per sequence
        """
        device_output = device_output or torch.device("cpu")

        # --- Prepare sequences and context ---
        obs = self._prepare_observations(X, theta=context)
        B = len(obs.sequence)

        # --- Precompute uniform logits if context is None ---
        if all(c is None for c in obs.context):
            initial_pdf = self._params.get("initial_pdf")
            if initial_pdf is not None and hasattr(initial_pdf, "logits") and initial_pdf.logits is not None:
                initial_logits = initial_pdf.logits
            else:
                initial_logits = self.initial_module.log_matrix(context=None)

            transition_pdf = self._params.get("transition_pdf")
            if transition_pdf is not None and hasattr(transition_pdf, "logits") and transition_pdf.logits is not None:
                transition_logits = transition_pdf.logits
            else:
                transition_logits = self.transition_module.log_matrix(context=None) / max(transition_temp, 1e-6)
                transition_logits = transition_logits.clamp(-MAX_LOGITS, MAX_LOGITS)

            duration_pdf = self._params.get("duration_pdf")
            if duration_pdf is not None and hasattr(duration_pdf, "logits") and duration_pdf.logits is not None:
                duration_logits = duration_pdf.logits
            else:
                duration_logits = self.duration_module.log_matrix(context=None) / max(duration_temp, 1e-6)
                duration_logits = duration_logits.clamp(-MAX_LOGITS, MAX_LOGITS)

        preds: list[torch.Tensor] = []

        for b in range(B):
            seq = obs.sequence[b].to(self.device, DTYPE)
            log_probs = obs.log_probs[b].to(self.device, DTYPE)
            theta_b = obs.context[b]
            T = seq.shape[0]

            if T == 0:
                preds.append(torch.empty(0, dtype=torch.int64, device=device_output))
                continue

            # --- Contextual logits ---
            if theta_b is not None:
                initial_logits_b = self.initial_module.log_matrix(context=theta_b)
                transition_logits_b = self.transition_module.log_matrix(context=theta_b) / max(transition_temp, 1e-6)
                transition_logits_b = transition_logits_b.clamp(-MAX_LOGITS, MAX_LOGITS)
                duration_logits_b = self.duration_module.log_matrix(context=theta_b) / max(duration_temp, 1e-6)
                duration_logits_b = duration_logits_b.clamp(-MAX_LOGITS, MAX_LOGITS)
            else:
                initial_logits_b = initial_logits
                transition_logits_b = transition_logits
                duration_logits_b = duration_logits

            # --- Observations wrapper (already log_probs from _prepare_observations) ---
            obs_b = utils.Observations([seq], log_probs=[log_probs])

            # --- Decode ---
            if algorithm.lower() == "viterbi":
                decoded_list = self._viterbi(obs_b, theta=theta_b)
                preds.append(decoded_list[0].detach().to(device_output))
            elif algorithm.lower() == "map":
                decoded = self._map_decode(
                    log_probs,
                    initial_logits_b,
                    transition_logits_b,
                    duration_logits_b
                )
                preds.append(decoded.detach().to(device_output))
            else:
                raise ValueError(f"Unknown decoding algorithm '{algorithm}'.")

        return preds

    @torch.no_grad()
    def score(
        self,
        X: torch.Tensor | list[torch.Tensor],
        theta: Optional[torch.Tensor | list[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Compute log-likelihoods for sequences under the HSMM model with optional contextual modulation.
        Supports variable-length, time-varying context.
        Returns [B] log-likelihoods.
        """
        # --- Normalize X to list ---
        if torch.is_tensor(X):
            X_list = [X]
        elif isinstance(X, (list, tuple)):
            X_list = list(X)
        else:
            raise TypeError(f"Unsupported input type {type(X)}")
        
        B = len(X_list)
        if B == 0:
            return torch.empty(0, device=self.device, dtype=DTYPE)

        # --- Normalize theta to list of tensors or None per sequence ---
        if theta is None:
            theta_list: list[Optional[torch.Tensor]] = [None] * B
        elif torch.is_tensor(theta):
            # Split flat theta by sequence lengths
            lengths = [seq.shape[0] for seq in X_list]
            cum_lengths = torch.cat([torch.zeros(1, dtype=torch.long, device=theta.device),
                                     torch.tensor(lengths, device=theta.device).cumsum(0)])
            theta_list = [theta[cum_lengths[b]:cum_lengths[b + 1]].to(self.device, DTYPE) for b in range(B)]
        elif isinstance(theta, list):
            theta_list = [t.to(self.device, DTYPE) if isinstance(t, torch.Tensor) else None for t in theta]
            if len(theta_list) != B:
                raise ValueError(f"Length of theta list ({len(theta_list)}) does not match number of sequences ({B})")
        else:
            raise TypeError(f"Unsupported theta type {type(theta)}")

        # --- Compute emissions ---
        all_seq = torch.cat([seq.to(self.device, dtype=DTYPE) for seq in X_list], dim=0)
        all_theta = torch.cat([t for t in theta_list if t is not None], dim=0) if any(theta_list) else None
        log_B = self.emission_module.log_prob(all_seq, context=all_theta)  # [sum T_b, K]

        # Split emission log-probs per sequence
        seq_lengths = [seq.shape[0] for seq in X_list]
        log_probs_split, start = [], 0
        for L in seq_lengths:
            log_probs_split.append(log_B[start:start + L])
            start += L

        obs = utils.Observations(X_list, log_probs=log_probs_split)

        # --- Global context for discrete modules ---
        global_ctx = None if all(t is None for t in theta_list) else theta_list[0][0:1] if theta_list[0] is not None else None

        temperature = max(self.temperature, 1e-6)
        initial_logits = self.initial_module.log_matrix(context=global_ctx)
        duration_logits = (self.duration_module.log_matrix(context=global_ctx) / temperature).clamp(-MAX_LOGITS, MAX_LOGITS)
        transition_logits = (self.transition_module.log_matrix(context=global_ctx) / temperature).clamp(-MAX_LOGITS, MAX_LOGITS)

        # --- Forward algorithm for each sequence ---
        alpha_list = self._forward(obs, theta=theta_list)
        log_likelihoods = []
        for alpha, L in zip(alpha_list, seq_lengths):
            if L == 0:
                log_likelihoods.append(torch.tensor(0.0, device=self.device, dtype=DTYPE))
            else:
                # sum over hidden states and durations
                log_likelihoods.append(torch.logsumexp(alpha[L - 1], dim=(-2, -1)))

        return torch.stack(log_likelihoods, dim=0)

    @torch.no_grad()
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

        device = self.device

        # --- Compute per-sequence log-likelihood ---
        try:
            ll = self.score(X)  # Returns [B] tensor
            ll = ll.to(dtype=DTYPE, device=device)
        except Exception as e:
            raise RuntimeError(f"Failed to compute log-likelihood: {e}")

        # --- Count total observations ---
        if lengths is not None:
            n_obs = max(sum(lengths), 1)
        elif X.ndim == 3:  # (B, T, F)
            n_obs = max(X.shape[0] * X.shape[1], 1)
        elif X.ndim == 2:  # single sequence (T, F)
            n_obs = max(X.shape[0], 1)
        else:
            raise ValueError(f"Unsupported input shape {X.shape}")

        # --- Degrees of freedom ---
        dof = getattr(self, "dof", None)
        if dof is None:
            raise AttributeError(
                "Model degrees of freedom ('dof') not defined. "
                "Please set 'self.dof' during initialization."
            )

        # --- Compute information criterion ---
        ic_value = constraints.compute_information_criteria(
            n_obs=n_obs,
            log_likelihood=ll,
            dof=dof,
            criterion=criterion
        )

        # --- Convert to tensor and sanitize ---
        if not isinstance(ic_value, torch.Tensor):
            ic_value = torch.tensor(ic_value, dtype=DTYPE, device=device)
        ic_value = ic_value.nan_to_num(nan=float('inf'), posinf=float('inf'), neginf=float('inf'))

        # --- Ensure per-sample shape ---
        if by_sample and ic_value.ndim == 0:
            ic_value = ic_value.unsqueeze(0)

        return ic_value.detach().cpu()

    def decode(
        self,
        X: torch.Tensor | np.ndarray,
        algorithm: Literal["viterbi", "map"] = "viterbi",
        first_only: bool = True
    ) -> np.ndarray | list[np.ndarray]:
        """
        Decode hidden states from input sequence(s) using Viterbi or MAP.

        Args:
            X: Input sequence(s)
            algorithm: "viterbi" or "map"
            first_only: If True, return only first sequence (for single-sequence use)
                        If False, return list of arrays for all sequences
        Returns:
            NumPy array(s) of predicted states
        """
        if not torch.is_tensor(X):
            X = torch.as_tensor(X, dtype=DTYPE)

        preds = self.predict(X, algorithm=algorithm, context=None)
        preds_np = [p.cpu().numpy() for p in preds]
        return preds_np[0] if first_only else preds_np


